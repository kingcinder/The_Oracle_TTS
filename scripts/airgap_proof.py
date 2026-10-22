#!/usr/bin/env python3
"""Air-gapped offline-install proof.

One command that answers: *does the offline bundle install and render on a
machine that has never had a network?* It drives the Windows QEMU guest that
`docs/WINDOWS_VM.md` describes:

1. checks the bundle locally (manifest, Windows wheels, every pinned model);
2. boots the guest with a network, resolves its data volume by label, and
   stages only the bundle files whose size differs from what is already staged;
3. installs `scripts/airgap_proof_windows.ps1` and arms it as a one-shot logon
   task, after wiping the previous install, model cache and proof evidence;
4. reboots the guest with `ORACLE_VM_NO_NET=1` (no NIC at all), where the proof
   installs from the bundle, checks the seeded cache, renders through the
   managed launcher, writes `airgap-result.json` and powers the machine off;
5. boots the guest with a network again only to read that JSON, then asserts
   `PASS`, no failed assertions, zero adapters and the bundle's commit.

Exit status is 0 only when the guest reports PASS. The guest script owns the
individual assertions; this harness owns staging, arming, completion detection
and the final verdict.

The VM tooling lives outside the repository (see `docs/WINDOWS_VM.md`); point
``--vm-script`` at its launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GUEST_SCRIPT = REPO_ROOT / "scripts" / "airgap_proof_windows.ps1"

DEFAULT_SSH_PORT = 2222
VOLUME_LABEL = "ORACLEDATA"


class ProofError(RuntimeError):
    """A step could not be completed; the message says which and why."""


@dataclass(frozen=True)
class Layout:
    """Guest paths on the data volume (the C: install is never touched).

    The data volume's letter is not stable between boots, so the root comes
    from the label at run time and every path here is passed to the guest
    explicitly rather than being guessed by the guest script.
    """

    root: str
    stage_dir: str = "bundle"
    install_dir: str = "OracleOffline"
    proof_dir: str = "airgap"
    hub_dir: str = "hfhub"
    task: str = "OracleAirgap"

    @property
    def stage(self) -> str:
        return self.root + "\\" + self.stage_dir

    @property
    def install(self) -> str:
        return self.root + "\\" + self.install_dir

    @property
    def proof(self) -> str:
        return self.root + "\\" + self.proof_dir

    @property
    def hub(self) -> str:
        return self.root + "\\" + self.hub_dir

    @property
    def guest_script(self) -> str:
        return self.proof + "\\airgap-proof.ps1"

    @property
    def guest_arm(self) -> str:
        return self.proof + "\\airgap-arm.ps1"

    @property
    def result_path(self) -> str:
        return self.proof + "\\airgap-result.json"


def _remote_path(path: str) -> str:
    """Guest path in the forward-slash form ssh/scp arguments use."""
    return path.replace("\\", "/")


def _ps_string(value: str) -> str:
    """Quote a value for PowerShell single-quoted syntax."""
    return "'" + value.replace("'", "''") + "'"


class Runner:
    """The slice of the host we need: run, ssh, copy, stream files."""

    def __init__(self, *, ssh_key: Path, ssh_port: int = DEFAULT_SSH_PORT, host: str = "127.0.0.1") -> None:
        self.ssh_key = Path(ssh_key).expanduser()
        self.ssh_port = ssh_port
        self.host = host
        self._ssh = [
            "ssh",
            "-i",
            str(self.ssh_key),
            "-p",
            str(ssh_port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "LogLevel=ERROR",
            f"oracle@{host}",
        ]
        self._scp = [
            "scp",
            "-i",
            str(self.ssh_key),
            "-P",
            str(ssh_port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
        ]

    def command(self, argv: list[str], *, timeout: float = 600, env: dict[str, str] | None = None) -> str:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **env} if env else None,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            raise ProofError(f"{argv[0]} exited {completed.returncode}: {detail[-1] if detail else 'no output'}")
        return completed.stdout

    def ssh(self, remote_command: str, *, timeout: float = 600) -> str:
        return self.command([*self._ssh, remote_command], timeout=timeout)

    def scp(self, local: Path, remote: str, *, timeout: float = 600) -> None:
        self.command([*self._scp, str(local), f"oracle@{self.host}:{_remote_path(remote)}"], timeout=timeout)

    def stream_tar(self, source: Path, relative_paths: list[str], remote_dir: str, *, timeout: float = 7200) -> None:
        """Copy bundle files over with one ssh connection (guest has tar.exe)."""
        if not relative_paths:
            return
        tar = subprocess.Popen(
            ["tar", "-C", str(source), "-cf", "-", *relative_paths],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert tar.stdout is not None
        extract = subprocess.Popen(
            [*self._ssh, f"tar -xf - -C {_remote_path(remote_dir)}"],
            stdin=tar.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tar.stdout.close()
        try:
            _, extract_error = extract.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            extract.kill()
            tar.kill()
            raise ProofError(f"staging timed out after {timeout:.0f}s") from None
        tar_error = tar.stderr.read() if tar.stderr else b""
        tar.wait()
        if tar.returncode != 0:
            raise ProofError(f"tar failed: {tar_error.decode('utf-8', 'replace').strip()[-300:]}")
        if extract.returncode != 0:
            raise ProofError(f"guest tar failed: {extract_error.decode('utf-8', 'replace').strip()[-300:]}")


# --- 1. the bundle -----------------------------------------------------------


def bundle_manifest(bundle: Path) -> dict:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise ProofError(f"no bundle manifest at {manifest_path}")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProofError(f"unreadable bundle manifest: {exc}") from None


def bundle_problems(bundle: Path, manifest: dict) -> list[str]:
    """Everything the guest will need, checked before the VM is touched."""
    problems: list[str] = []
    for name in ("install.bat", "repo.tar.gz"):
        if not (bundle / name).is_file():
            problems.append(f"bundle has no {name}")
    if not list((bundle / "wheels" / "windows").glob("*.whl")):
        problems.append("bundle carries no windows wheels")
    for repo_id, sha in (manifest.get("model_pins") or {}).items():
        snapshot = bundle / "hf_cache" / ("models--" + repo_id.replace("/", "--")) / "snapshots" / sha
        if not snapshot.is_dir():
            problems.append(f"bundle is missing {repo_id} @ {str(sha)[:12]}")
    return problems


# --- 2. staging --------------------------------------------------------------


def remote_sizes(runner: Runner, stage: str) -> dict[str, int]:
    """Relative path -> size for every file already staged on the guest."""
    script = (
        f"$root = {_ps_string(stage)}; "
        "if (Test-Path -LiteralPath $root) { "
        "Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue | "
        "ForEach-Object { $_.FullName.Substring($root.Length + 1) + '|' + $_.Length } }"
    )
    sizes: dict[str, int] = {}
    for line in runner.ssh(script, timeout=300).splitlines():
        name, _, size = line.strip().rpartition("|")
        if name and size.isdigit():
            sizes[name.replace("\\", "/")] = int(size)
    return sizes


def files_to_stage(bundle: Path, remote: dict[str, int]) -> list[str]:
    """Bundle files whose staged copy is missing or a different size."""
    changed: list[str] = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(bundle).as_posix()
        if remote.get(relative) != path.stat().st_size:
            changed.append(relative)
    return changed


def stage_bundle(runner: Runner, bundle: Path, layout: Layout, *, log=print) -> list[str]:
    changed = files_to_stage(bundle, remote_sizes(runner, layout.stage))
    if not changed:
        log("bundle: guest copy is current (0 files to stage)")
        return changed
    log(f"bundle: staging {len(changed)} file(s) into {layout.stage}")
    runner.stream_tar(bundle, changed, layout.stage)
    return changed


def resolve_layout(runner: Runner, *, volume_label: str = VOLUME_LABEL, **layout: str) -> Layout:
    """Find the data volume's letter on the guest and build the paths from it."""
    letter = runner.ssh(
        f"$v = Get-Volume -FileSystemLabel {_ps_string(volume_label)} -ErrorAction SilentlyContinue | "
        "Select-Object -First 1; if ($v.DriveLetter) { $v.DriveLetter }",
        timeout=120,
    ).strip()
    if len(letter) != 1 or not letter.isalpha():
        raise ProofError(f"guest has no volume labelled {volume_label}")
    return Layout(root=letter.upper() + ":", **layout)


# --- 3. arming ---------------------------------------------------------------


def guest_arguments(layout: Layout) -> str:
    """The proof's parameters, quoted for the scheduled task's -Argument."""
    return (
        f'-File "{layout.guest_script}" '
        f"-Bundle {_ps_string(layout.stage)} "
        f"-InstallRoot {_ps_string(layout.install)} "
        f"-ProofDir {_ps_string(layout.proof)} "
        f"-HubCache {_ps_string(layout.hub)}"
    )


def arm_script(layout: Layout) -> str:
    """Reset guest state and register the one-shot proof task."""
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"foreach ($path in {_ps_string(layout.install)}, {_ps_string(layout.hub)}) {{",
            "    Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue",
            "}",
            "Remove-Item -LiteralPath \"$env:APPDATA\\Python\" -Recurse -Force -ErrorAction SilentlyContinue",
            f"Get-ChildItem -LiteralPath {_ps_string(layout.proof)} -Force -ErrorAction SilentlyContinue |",
            f"    Where-Object {{ $_.Name -ne 'airgap-proof.ps1' }} |",
            "    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue",
            f"Unregister-ScheduledTask -TaskName {_ps_string(layout.task)} -Confirm:$false "
            "-ErrorAction SilentlyContinue",
            f"$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument {_ps_string(guest_arguments(layout))}",
            "$trigger = New-ScheduledTaskTrigger -AtLogOn -User \"$env:COMPUTERNAME\\oracle\"",
            "$principal = New-ScheduledTaskPrincipal -UserId \"$env:COMPUTERNAME\\oracle\" "
            "-LogonType Interactive -RunLevel Highest",
            "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
            "-ExecutionTimeLimit (New-TimeSpan -Hours 4) -StartWhenAvailable",
            f"Register-ScheduledTask -TaskName {_ps_string(layout.task)} -Action $action -Trigger $trigger "
            "-Principal $principal -Settings $settings -Force | Out-Null",
            "Write-Output 'armed'",
        ]
    )


def arm_guest(runner: Runner, layout: Layout, *, proof_script: Path = GUEST_SCRIPT) -> None:
    runner.ssh(f"New-Item -ItemType Directory -Force -Path {_ps_string(layout.proof)} | Out-Null", timeout=120)
    with tempfile.TemporaryDirectory(prefix="airgap-arm-") as tmp:
        upload = Path(tmp) / "airgap-proof.ps1"
        upload.write_bytes(proof_script.read_bytes())
        runner.scp(upload, layout.guest_script)
        arm = Path(tmp) / "airgap-arm.ps1"
        arm.write_text(arm_script(layout), encoding="utf-8", newline="\r\n")
        runner.scp(arm, layout.guest_arm)
    output = runner.ssh(
        f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File {_remote_path(layout.guest_arm)}",
        timeout=300,
    )
    if "armed" not in output:
        raise ProofError(f"guest did not confirm the proof task was armed: {output.strip()[-300:]}")


# --- 4. boot phases ----------------------------------------------------------


def stop_guest(runner: Runner, vm_script: Path) -> None:
    runner.command([str(vm_script), "stop"], timeout=180)


def boot(runner: Runner, vm_script: Path, *, air_gapped: bool) -> None:
    env = {"ORACLE_VM_NO_NET": "1"} if air_gapped else {}
    runner.command([str(vm_script), "start", "run"], timeout=180, env=env)


def wait_for_ssh(runner: Runner, *, timeout: float, poll: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if "READY" in runner.ssh("echo READY", timeout=30):
                return
        except ProofError:
            pass
        time.sleep(poll)
    raise ProofError(f"guest did not answer over ssh within {timeout:.0f}s")


def wait_for_poweroff(runner: Runner, vm_script: Path, *, timeout: float, poll: float = 15, log=print) -> None:
    """The air-gapped guest powers itself off; nothing can be read over the wire."""
    started = time.monotonic()
    deadline = started + timeout
    last_note = started
    while time.monotonic() < deadline:
        try:
            status = runner.command([str(vm_script), "monitor", "info status"], timeout=60)
        except ProofError:
            status = ""
        if "shut down" in status.lower() or "qemu: stopped" in status.lower():
            log(f"guest powered off after {time.monotonic() - started:.0f}s")
            return
        if time.monotonic() - last_note > 300:
            last_note = time.monotonic()
            log(f"waiting for the offline proof to finish ({(last_note - started) / 60:.0f} min)")
        time.sleep(poll)
    raise ProofError(
        f"the guest did not finish within {timeout:.0f}s; check `{vm_script.name} screenshot` "
        "and E:\\airgap\\airgap.log before retrying"
    )


# --- 5. the verdict ----------------------------------------------------------


def read_result(runner: Runner, layout: Layout) -> dict:
    output = runner.ssh(
        f"if (Test-Path -LiteralPath {_ps_string(layout.result_path)}) "
        f"{{ Get-Content -LiteralPath {_ps_string(layout.result_path)} -Raw }}",
        timeout=120,
    )
    if not output.strip():
        raise ProofError(f"the guest wrote no {layout.result_path}")
    try:
        return json.loads(output)
    except ValueError as exc:
        raise ProofError(f"unreadable air-gap result: {exc}") from None


def result_problems(result: dict, expected_commit: str) -> list[str]:
    problems: list[str] = []
    if result.get("result") != "PASS":
        problems.append(f"guest verdict is {result.get('result')!r}")
    problems.extend(f"guest assertion failed: {item}" for item in (result.get("failures") or []))
    if str(result.get("adapters")) != "0":
        problems.append(f"guest had {result.get('adapters')} network adapter(s)")
    if expected_commit and result.get("repo_commit") != expected_commit:
        problems.append(
            f"guest installed {str(result.get('repo_commit'))[:12]} but the bundle is {expected_commit[:12]}"
        )
    return problems


def evidence(result: dict) -> str:
    return "\n".join(
        [
            "air-gap proof",
            f"  result     {result.get('result')}",
            f"  commit     {str(result.get('repo_commit'))[:12]}",
            f"  install    {result.get('install_seconds')}s",
            f"  render     {result.get('render_seconds')}s",
            f"  audio      {result.get('audio')}",
            f"  adapters   {result.get('adapters')}",
        ]
    )


# --- 6. the run --------------------------------------------------------------


def run(
    *,
    bundle: Path,
    vm_script: Path,
    ssh_key: Path,
    ssh_port: int,
    timeout: float,
    volume_label: str = VOLUME_LABEL,
    runner: Runner | None = None,
    log=print,
) -> int:
    bundle = Path(bundle)
    manifest = bundle_manifest(bundle)
    problems = bundle_problems(bundle, manifest)
    if problems:
        for problem in problems:
            log(f"bundle: {problem}")
        raise ProofError("the bundle is incomplete; nothing was staged")

    runner = runner or Runner(ssh_key=ssh_key, ssh_port=ssh_port)

    log("guest: booting with a network to stage the bundle")
    stop_guest(runner, vm_script)
    boot(runner, vm_script, air_gapped=False)
    wait_for_ssh(runner, timeout=600)
    layout = resolve_layout(runner, volume_label=volume_label)
    log(f"guest: data volume {volume_label} is {layout.root}")

    changed = stage_bundle(runner, bundle, layout, log=log)
    log(f"guest: arming the one-shot proof task ({len(changed)} file(s) staged)")
    arm_guest(runner, layout)

    log("guest: rebooting with no network adapter")
    stop_guest(runner, vm_script)
    boot(runner, vm_script, air_gapped=True)
    wait_for_poweroff(runner, vm_script, timeout=timeout, log=log)

    log("guest: rebooting to collect the result")
    boot(runner, vm_script, air_gapped=False)
    wait_for_ssh(runner, timeout=600)
    result = read_result(runner, layout)
    log(evidence(result))

    problems = result_problems(result, str(manifest.get("repo_commit") or ""))
    if problems:
        for problem in problems:
            log(f"FAIL: {problem}")
        return 1
    log("PASS: the bundle installed and rendered with no network adapter")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True, help="Offline bundle directory to prove.")
    parser.add_argument(
        "--vm-script",
        type=Path,
        default=Path.home() / ".oracle-win11-vm" / "run-vm.sh",
        help="Guest launcher from docs/WINDOWS_VM.md (default: %(default)s).",
    )
    parser.add_argument(
        "--ssh-key",
        type=Path,
        default=Path.home() / ".ssh" / "oracle-win11-vm",
        help="Guest SSH key (default: %(default)s).",
    )
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT)
    parser.add_argument("--volume-label", default=VOLUME_LABEL, help="Label of the guest data volume.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600,
        help="Seconds the offline phase may take before the run is called a hang (default: 3600).",
    )
    args = parser.parse_args(argv)

    try:
        return run(
            bundle=args.bundle,
            vm_script=args.vm_script,
            ssh_key=args.ssh_key,
            ssh_port=args.ssh_port,
            timeout=args.timeout,
            volume_label=args.volume_label,
        )
    except ProofError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
