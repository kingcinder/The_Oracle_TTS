"""Tests for the repeatable air-gap proof harness.

Covers:
  * scripts/airgap_proof.py: bundle preflight, size-based staging, the armed
    guest task (paths passed explicitly, previous state wiped), the verdict
    read back from the guest, and the full boot/stage/reboot sequence.
  * scripts/airgap_proof_windows.ps1: it takes the paths the arm task passes,
    keeps its own evidence, and powers the guest off when done.

Nothing here starts QEMU, ssh or a guest: the harness is driven through a fake
Runner so the sequence and the assertions can be tested on any host.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
GUEST_SCRIPT = SCRIPTS_DIR / "airgap_proof_windows.ps1"


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def proof():
    return _load_script_module("oracle_airgap_proof", "airgap_proof.py")


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    (root / "wheels" / "windows").mkdir(parents=True)
    (root / "wheels" / "windows" / "the_oracle-1.4.0-py3-none-any.whl").write_bytes(b"wheel")
    (root / "install.bat").write_text("@echo off\n", encoding="utf-8")
    (root / "repo.tar.gz").write_bytes(b"tar")
    pinned = root / "hf_cache" / "models--ResembleAI--chatterbox" / "snapshots" / ("a" * 40)
    pinned.mkdir(parents=True)
    (pinned / "config.json").write_text("{}\n", encoding="utf-8")
    manifest = {
        "repo_commit": "b" * 40,
        "platforms": ["windows"],
        "model_pins": {"ResembleAI/chatterbox": "a" * 40},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


# --- preflight ---------------------------------------------------------------


def test_a_complete_bundle_has_no_problems(proof, bundle: Path) -> None:
    assert proof.bundle_problems(bundle, proof.bundle_manifest(bundle)) == []


def test_a_missing_manifest_is_refused(proof, tmp_path: Path) -> None:
    with pytest.raises(proof.ProofError, match="manifest"):
        proof.bundle_manifest(tmp_path / "bundle")


def test_every_missing_piece_is_named(proof, bundle: Path) -> None:
    (bundle / "wheels" / "windows" / "the_oracle-1.4.0-py3-none-any.whl").unlink()
    (bundle / "install.bat").unlink()
    sha = "a" * 40
    (bundle / "hf_cache" / "models--ResembleAI--chatterbox" / "snapshots" / sha).rename(
        bundle / "hf_cache" / "models--ResembleAI--chatterbox" / "snapshots" / ("c" * 40)
    )

    problems = proof.bundle_problems(bundle, proof.bundle_manifest(bundle))

    assert any("install.bat" in item for item in problems)
    assert any("windows wheels" in item for item in problems)
    assert any("ResembleAI/chatterbox" in item for item in problems)


# --- staging -----------------------------------------------------------------


def test_only_missing_or_resized_files_are_staged(proof, bundle: Path) -> None:
    staged = {
        "install.bat": (bundle / "install.bat").stat().st_size,
        "manifest.json": 1,  # wrong size
        "wheels/windows/the_oracle-1.4.0-py3-none-any.whl": 5,
    }

    changed = proof.files_to_stage(bundle, staged)

    assert "install.bat" not in changed
    assert "manifest.json" in changed
    assert "repo.tar.gz" in changed
    assert all("\\" not in item for item in changed)


def test_remote_sizes_parses_the_guests_listing(proof) -> None:
    class FakeRunner:
        def ssh(self, remote_command: str, *, timeout: float = 600) -> str:
            assert "Get-ChildItem" in remote_command
            return "wheels\\windows\\a.whl|1234\nmodels\\x.bin|99\nnoise\n"

    sizes = proof.remote_sizes(FakeRunner(), r"E:\bundle")

    assert sizes == {"wheels/windows/a.whl": 1234, "models/x.bin": 99}


def test_a_current_guest_copy_skips_staging(proof, bundle: Path) -> None:
    staged = {path.relative_to(bundle).as_posix(): path.stat().st_size for path in bundle.rglob("*") if path.is_file()}

    class FakeRunner:
        streamed: list = []

        def ssh(self, remote_command: str, *, timeout: float = 600) -> str:
            return "".join(f"{name.replace('/', chr(92))}|{size}\n" for name, size in staged.items())

        def stream_tar(self, source, relative_paths, remote_dir, *, timeout: float = 7200) -> None:
            self.streamed.append(relative_paths)

    runner = FakeRunner()
    messages: list[str] = []
    layout = proof.Layout(root="E:")

    assert proof.stage_bundle(runner, bundle, layout, log=messages.append) == []
    assert runner.streamed == []
    assert "current" in messages[0]


# --- the armed guest task ----------------------------------------------------


def test_the_armed_task_names_every_path_explicitly(proof) -> None:
    layout = proof.Layout(root="E:")

    script = proof.arm_script(layout)

    assert f"-File \"{layout.guest_script}\"" in script
    assert "'E:\\bundle'" in script
    assert "'E:\\OracleOffline'" in script
    assert "'E:\\airgap'" in script
    assert "'E:\\hfhub'" in script


def test_arming_wipes_the_previous_install_and_cache_but_keeps_the_proof(proof) -> None:
    script = proof.arm_script(proof.Layout(root="E:"))

    assert "'E:\\OracleOffline', 'E:\\hfhub'" in script
    assert "Where-Object { $_.Name -ne 'airgap-proof.ps1' }" in script
    assert "Remove-Item -LiteralPath \"$env:APPDATA\\Python\"" in script
    assert "Unregister-ScheduledTask -TaskName 'OracleAirgap'" in script


def test_the_task_is_one_shot_and_runs_at_logon(proof) -> None:
    script = proof.arm_script(proof.Layout(root="E:"))

    assert "New-ScheduledTaskTrigger -AtLogOn" in script
    assert "Register-ScheduledTask -TaskName 'OracleAirgap'" in script
    assert "Write-Output 'armed'" in script


def test_the_data_volume_letter_is_read_from_the_guest(proof) -> None:
    class FakeRunner:
        def ssh(self, remote_command: str, *, timeout: float = 600) -> str:
            assert "Get-Volume -FileSystemLabel 'ORACLEDATA'" in remote_command
            return "F\r\n"

    layout = proof.resolve_layout(FakeRunner())

    assert layout.root == "F:"
    assert layout.guest_script == "F:\\airgap\\airgap-proof.ps1"


def test_a_guest_without_the_data_volume_is_refused(proof) -> None:
    class FakeRunner:
        def ssh(self, remote_command: str, *, timeout: float = 600) -> str:
            return ""

    with pytest.raises(proof.ProofError, match="ORACLEDATA"):
        proof.resolve_layout(FakeRunner())


def test_arming_uploads_the_committed_proof_script(proof) -> None:
    layout = proof.Layout(root="E:")
    copied: list[tuple[str, str, bytes]] = []

    class FakeRunner:
        def ssh(self, remote_command: str, *, timeout: float = 600) -> str:
            return "armed"

        def scp(self, local: Path, remote: str, *, timeout: float = 600) -> None:
            copied.append((Path(local).name, remote, Path(local).read_bytes()))

    proof.arm_guest(FakeRunner(), layout)

    assert [name for name, _, _ in copied] == ["airgap-proof.ps1", "airgap-arm.ps1"]
    assert [remote for _, remote, _ in copied] == [layout.guest_script, layout.guest_arm]
    uploaded_proof = next(body for name, _, body in copied if name == "airgap-proof.ps1")
    assert uploaded_proof == GUEST_SCRIPT.read_bytes()


# --- the verdict -------------------------------------------------------------


def _passing_result() -> dict:
    return {
        "result": "PASS",
        "repo_commit": "b" * 40,
        "adapters": 0,
        "install_seconds": 1218.7,
        "render_seconds": 490.0,
        "audio": {"seconds": 4.14, "rate": 24000, "peak": 1.0},
        "failures": [],
    }


def test_a_passing_run_has_no_problems(proof) -> None:
    assert proof.result_problems(_passing_result(), "b" * 40) == []


def test_a_failed_verdict_and_its_assertions_are_reported(proof) -> None:
    result = _passing_result() | {
        "result": "FAIL",
        "failures": ["seeded snapshot ResembleAI/chatterbox", "render exited 0"],
    }

    problems = proof.result_problems(result, "b" * 40)

    assert any("verdict" in item for item in problems)
    assert [item for item in problems if "assertion failed" in item] == [
        "guest assertion failed: seeded snapshot ResembleAI/chatterbox",
        "guest assertion failed: render exited 0",
    ]


def test_a_guest_that_kept_an_adapter_is_not_a_pass(proof) -> None:
    problems = proof.result_problems(_passing_result() | {"adapters": 1}, "b" * 40)

    assert any("adapter" in item for item in problems)


def test_a_bundle_from_another_commit_is_reported(proof) -> None:
    problems = proof.result_problems(_passing_result(), "c" * 40)

    assert any("bundle is cccccccccccc" in item for item in problems)


def test_a_missing_result_is_an_error(proof) -> None:
    class FakeRunner:
        def ssh(self, remote_command: str, *, timeout: float = 600) -> str:
            return ""

    with pytest.raises(proof.ProofError, match="airgap-result.json"):
        proof.read_result(FakeRunner(), proof.Layout(root="E:"))


# --- the whole sequence ------------------------------------------------------


class FakeVm:
    """Records every host-side step the harness takes."""

    def __init__(self, proof, result: dict, *, letter: str = "E") -> None:
        self.proof = proof
        self.result = result
        self.letter = letter
        self.steps: list[str] = []
        self.boots: list[bool] = []
        self.staged: list[list[str]] = []
        self.copy: list[str] = []

    # host side
    def command(self, argv, *, timeout: float = 600, env: dict | None = None) -> str:
        if argv[-2:] == ["start", "run"]:
            air_gapped = (env or {}).get("ORACLE_VM_NO_NET") == "1"
            self.boots.append(air_gapped)
            self.steps.append("boot-air-gapped" if air_gapped else "boot-networked")
            return ""
        if argv[-1] == "stop":
            self.steps.append("stop")
            return ""
        if argv[-1] == "info status":
            return "VM status: shut down"
        raise AssertionError(f"unexpected command: {argv}")

    # guest side
    def ssh(self, remote_command: str, *, timeout: float = 600) -> str:
        if remote_command == "echo READY":
            return "READY\r\n"
        if "Get-Volume" in remote_command:
            return self.letter + "\r\n"
        if "Get-ChildItem -LiteralPath $root" in remote_command:
            return ""
        if "New-Item -ItemType Directory" in remote_command:
            return ""
        if "airgap-arm.ps1" in remote_command:
            self.steps.append("arm")
            return "armed\r\n"
        if "airgap-result.json" in remote_command:
            self.steps.append("read-result")
            return json.dumps(self.result)
        raise AssertionError(f"unexpected ssh: {remote_command}")

    def scp(self, local: Path, remote: str, *, timeout: float = 600) -> None:
        self.copy.append(remote)

    def stream_tar(self, source, relative_paths, remote_dir, *, timeout: float = 7200) -> None:
        self.staged.append(list(relative_paths))


def _run(proof, bundle: Path, runner, tmp_path: Path, log: list[str] | None = None):
    vm_script = tmp_path / "run-vm.sh"
    vm_script.write_text("#!/bin/sh\n", encoding="utf-8")
    log = log if log is not None else []
    return proof.run(
        bundle=bundle,
        vm_script=vm_script,
        ssh_key=tmp_path / "key",
        ssh_port=2222,
        timeout=60,
        runner=runner,
        log=log.append,
    )


def test_a_passing_run_boots_stages_reboots_and_returns_zero(proof, bundle: Path, tmp_path: Path) -> None:
    runner = FakeVm(proof, _passing_result())
    log: list[str] = []

    assert _run(proof, bundle, runner, tmp_path, log) == 0

    assert runner.boots == [False, True, False]
    assert runner.steps.count("stop") == 2
    assert runner.steps.index("boot-air-gapped") < runner.steps.index("read-result")
    assert runner.staged and sorted(runner.staged[0]) == sorted(
        path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()
    )
    assert runner.copy == ["E:\\airgap\\airgap-proof.ps1", "E:\\airgap\\airgap-arm.ps1"]
    assert any("PASS" in line for line in log)


def test_a_failed_guest_run_returns_one_and_names_the_assertion(proof, bundle: Path, tmp_path: Path) -> None:
    runner = FakeVm(proof, _passing_result() | {"result": "FAIL", "failures": ["seeded snapshot x"]})
    log: list[str] = []

    assert _run(proof, bundle, runner, tmp_path, log) == 1
    assert any("FAIL: guest assertion failed: seeded snapshot x" in line for line in log)


def test_an_incomplete_bundle_never_touches_the_guest(proof, bundle: Path, tmp_path: Path) -> None:
    (bundle / "repo.tar.gz").unlink()
    runner = FakeVm(proof, _passing_result())

    with pytest.raises(proof.ProofError, match="incomplete"):
        _run(proof, bundle, runner, tmp_path)

    assert runner.boots == [] and runner.copy == []


# --- the guest-side script ---------------------------------------------------


def test_the_guest_script_takes_the_paths_the_task_passes() -> None:
    text = GUEST_SCRIPT.read_text(encoding="utf-8")

    for parameter in ("$Bundle", "$InstallRoot", "$ProofDir", "$HubCache"):
        assert parameter in text
    assert "airgap-result.json" in text
    assert "shutdown.exe /s" in text
    assert "done.flag" in text


def test_the_guest_script_asserts_the_air_gap_and_the_render() -> None:
    text = GUEST_SCRIPT.read_text(encoding="utf-8")

    assert "Get-NetAdapter" in text
    assert "Resolve-DnsName" in text
    assert "install.bat" in text
    assert "refs\\main" in text
    assert "the-oracle.cmd" in text
    assert "*.flac" in text
