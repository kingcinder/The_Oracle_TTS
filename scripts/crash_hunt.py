"""Reproduction harness: run render/GUI loops, classify exit codes, collect crash records.

Evidence-first debugging tool for the crash-eradication campaign
(docs/superpowers/specs/2026-09-28-crash-eradication-performance-design.md).
Exit-code classification is grounded in the repo's documented crash shapes:
pipeline.py notes the Qt/native SIGSEGV surfaces as exit status 245
(256 - 11), shells report it as 139 (128 + 11), and OOM kills arrive as 137.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def classify_exit(code: int) -> str:
    """Map a child exit code to a crash bucket."""
    if code == 0:
        return "ok"
    if code in (245, 139):
        return "sigsegv"
    if code in (137, -9):
        return "oom_kill"
    return "error"


@dataclass
class CrashReport:
    kind: str
    command: list[str]
    returncode: int
    stderr_tail: str
    crash_record_path: str | None = None


def _render_command(backend: str = "pytorch") -> list[str]:
    """A real render command against a tracked sample input.

    Output goes to build/crash_hunt/out so runs never touch Output/
    (the doctor's read-only gate only tolerates build/doctor_*).
    """
    command = [
        sys.executable,
        "-m",
        "the_oracle.cli",
        "render",
        "--input",
        str(REPO_ROOT / "Input" / "What is, reality.txt"),
        "--outdir",
        str(REPO_ROOT / "build" / "crash_hunt" / "out"),
    ]
    if backend != "pytorch":
        command += ["--inference-backend", backend]
    return command


def _vulkan_available() -> bool:
    try:
        from the_oracle.tts_engines.vulkan_backend import find_audiocpp_binary

        return bool(find_audiocpp_binary())
    except Exception:
        return False


def _run_acceptance(outdir: Path) -> int:
    """The plan's hard gate: 5x pytorch, 3x vulkan (or explicit skip), 3x GUI,
    zero non-zero exits and zero new crash records."""
    from the_oracle import crash

    root = REPO_ROOT
    records_before = set(crash.list_records(root))
    summary: dict = {"new_crash_records": 0}

    for leg, runs, extra in (
        ("render_pytorch", 5, []),
        ("render_vulkan", 3, ["--backend", "vulkan"]),
    ):
        if leg == "render_vulkan" and not _vulkan_available():
            summary[leg] = {"runs": 0, "non_zero": 0, "vulkan_unavailable": "audio.cpp build not found"}
            continue
        non_zero = 0
        for _ in range(runs):
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--mode", "render", "--runs", "1", *extra],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if proc.returncode != 0:
                non_zero += 1
        summary[leg] = {"runs": runs, "non_zero": non_zero}

    gui_non_zero = 0
    for _ in range(3):
        kind = _run_gui_once(root / "Output" / "logs")
        if kind != "ok":
            gui_non_zero += 1
    summary["gui"] = {"runs": 3, "non_zero": gui_non_zero}

    summary["new_crash_records"] = len(harvest_crash_records(root, records_before))
    verdict, reasons = _acceptance_summary(summary)
    summary["verdict_pass"] = verdict
    summary["reasons"] = reasons
    (outdir / "acceptance.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if verdict else 1


def harvest_crash_records(root: Path, before: set) -> list[str]:
    """Paths (as strings) of crash records that appeared since the snapshot."""
    from the_oracle import crash

    return [str(p) for p in crash.list_records(root) if p not in before]


# --- Task 11: acceptance verdict ---


def _acceptance_summary(summary: dict) -> tuple[bool, list[str]]:
    """Judge an acceptance run: (pass, reasons).

    Failures are named per leg; an explicitly-skipped backend
    (``vulkan_unavailable``) passes but stays visible in reasons.
    """
    reasons: list[str] = []
    for leg in ("render_pytorch", "render_vulkan", "gui"):
        counts = summary.get(leg) or {}
        if counts.get("vulkan_unavailable"):
            reasons.append(f"{leg} vulkan_unavailable: {counts['vulkan_unavailable']}")
            continue
        non_zero = counts.get("non_zero", 0)
        if non_zero:
            reasons.append(f"{leg}: {non_zero} of {counts.get('runs', 0)} runs exited non-zero")
    new_records = summary.get("new_crash_records", 0)
    if new_records:
        reasons.append(f"{new_records} new crash record(s) written during acceptance")
    failures = [r for r in reasons if "vulkan_unavailable" not in r]
    return (not failures), reasons


# --- Task 4: offscreen GUI loop mode ---

GUI_TIMING_LOG = "gui_launch_timing.json"


def _gui_env() -> dict[str, str]:
    """Child env with the offscreen Qt platform (CI/smoke convention)."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def _gui_command() -> list[str]:
    return [sys.executable, "-m", "the_oracle.cli", "gui"]


def _gui_child(timing_file: Path, baseline_mtime: int | None, timeout_s: float) -> int:
    """Launch the GUI, wait for mainwindow_built, then terminate it.

    Returns the child's exit code (-1 if killed after a successful launch,
    which _run_gui_once treats as success when the timing log advanced).
    A dedicated session lets the kill reap native descendants too.
    """
    import time

    process = subprocess.Popen(
        _gui_command(),
        cwd=str(REPO_ROOT),
        env=_gui_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=(os.name == "posix"),
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _timing_fresh(timing_file, baseline_mtime):
            _kill_session(process)
            return process.wait()
        if process.poll() is not None:
            return process.returncode
        time.sleep(0.25)
    _kill_session(process)
    return process.wait()


def _timing_fresh(timing_file: Path, baseline_mtime: int | None) -> bool:
    if not timing_file.exists():
        return False
    return baseline_mtime is None or timing_file.stat().st_mtime_ns > baseline_mtime


def _kill_session(process: "subprocess.Popen[str]") -> None:
    import signal

    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (ProcessLookupError, OSError):
        pass


def _run_gui_once(timing_dir: Path, timeout_s: float = 120.0) -> str:
    """One GUI launch cycle: ok | sigsegv | oom_kill | error | launch_timeout."""
    timing_file = Path(timing_dir) / GUI_TIMING_LOG
    baseline = timing_file.stat().st_mtime_ns if timing_file.exists() else None
    rc = _gui_child(timing_file, baseline, timeout_s)
    if _timing_fresh(timing_file, baseline):
        # Fresh mainwindow_built: launch succeeded. rc is our own SIGTERM.
        return "ok"
    if rc != 0:
        return classify_exit(rc)
    return "launch_timeout"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce and classify Oracle crashes.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--outdir", default=str(REPO_ROOT / "build" / "crash_hunt"))
    parser.add_argument("--mode", choices=("render", "gui", "acceptance"), default="render")
    parser.add_argument("--backend", choices=("pytorch", "vulkan"), default="pytorch")
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.mode == "acceptance":
        return _run_acceptance(outdir)
    failures: list[dict] = []
    if args.mode == "gui":
        timing_dir = REPO_ROOT / "Output" / "logs"
        for _ in range(args.runs):
            kind = _run_gui_once(timing_dir)
            if kind != "ok":
                failures.append(asdict(CrashReport(kind, _gui_command(), -1, "")))
    else:
        for _ in range(args.runs):
            command = _render_command(backend=args.backend)
            proc = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=3600,
            )
            kind = classify_exit(proc.returncode)
            if kind != "ok":
                failures.append(
                    asdict(
                        CrashReport(
                            kind=kind,
                            command=command,
                            returncode=proc.returncode,
                            stderr_tail=proc.stderr[-4000:],
                        )
                    )
                )
    report = {"runs": args.runs, "failures": failures, "crash_records": []}
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
