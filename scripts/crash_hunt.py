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


def _render_command() -> list[str]:
    """A real render command against a tracked sample input.

    Output goes to build/crash_hunt/out so runs never touch Output/
    (the doctor's read-only gate only tolerates build/doctor_*).
    """
    return [
        sys.executable,
        "-m",
        "the_oracle.cli",
        "render",
        "--input",
        str(REPO_ROOT / "Input" / "What is, reality.txt"),
        "--outdir",
        str(REPO_ROOT / "build" / "crash_hunt" / "out"),
    ]


def harvest_crash_records(root: Path, before: set) -> list[str]:
    """Paths (as strings) of crash records that appeared since the snapshot."""
    from the_oracle import crash

    return [str(p) for p in crash.list_records(root) if p not in before]


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
    parser.add_argument("--mode", choices=("render", "gui"), default="render")
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    if args.mode == "gui":
        timing_dir = REPO_ROOT / "Output" / "logs"
        for _ in range(args.runs):
            kind = _run_gui_once(timing_dir)
            if kind != "ok":
                failures.append(asdict(CrashReport(kind, _gui_command(), -1, "")))
    else:
        for _ in range(args.runs):
            command = _render_command()
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
