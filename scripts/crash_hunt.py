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
    return [sys.executable, "-m", "the_oracle.cli", "render"]  # refined in Task 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce and classify Oracle crashes.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--outdir", default=str(REPO_ROOT / "build" / "crash_hunt"))
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
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
