#!/usr/bin/env python3
"""Run the doctor twice and fail if the two reports differ.

The doctor is the launch gate, so its verdict must not depend on how many times
it has been run. That has regressed twice, both times because a check graded
state the doctor itself had created on an earlier run: the real-engine readiness
PASS cited a smoke FLAC it never produced, and the default-voice-source count was
inflated by the reference clips the previous run had written. Neither was visible
in a single run — only in the difference between two.

So this runs the doctor twice, in one mode, and compares the two JSON reports.
Every field must match except the paths in ``MEASURED_FIELDS``, which are
durations of the machine rather than statements about the install. Any field that
differs is named with both values, so a new measurement shows up as a precise
failure rather than a mystery.

Run by ``.github/workflows/ci.yml`` from the test matrix, via each runner's
default shell, so it gates every push and pull request on Linux and Windows::

    python scripts/doctor_idempotence.py --skip-model-init --ci

Exit 0 means the two reports were identical; exit 1 means they differed. It does
*not* fail because the doctor reports the install is not ready: that verdict is
the doctor's own exit code, and the sibling CI steps already gate on it. What is
asserted here is only that running the gate twice says the same thing.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: Report fields that measure this machine instead of describing its state.
#: These are the only fields allowed to differ between two runs. Adding a field
#: here is a deliberate act: it says "this is a measurement of the moment, not a
#: fact about the install". The registry is checked for drift, so renaming a
#: measured field in ``doctor.py`` fails this script rather than silently
#: un-protecting it.
MEASURED_FIELDS = (
    # doctor.py: round(perf_counter() - started, 3) around from_pretrained().
    "chatterbox_init.seconds",
    # doctor.py: round(perf_counter() - started, 3) around the smoke render.
    "deterministic_smoke.runtime_seconds",
)

#: Stands in for a value the comparison deliberately ignores, so the printed
#: diff shows the structural shape without the numbers.
_MASKED = "<measured>"

_MISSING = object()


@dataclass(frozen=True)
class DoctorRun:
    """One doctor invocation: its exit code and captured streams."""

    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def report(self) -> dict[str, Any] | None:
        """The parsed JSON report, or ``None`` if the run produced none."""
        try:
            payload = json.loads(self.stdout)
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None


def run_doctor(repo_root: Path, flags: list[str]) -> DoctorRun:
    """Invoke ``doctor.py`` once, asking for the machine-readable report."""
    argv = [
        sys.executable,
        str(SCRIPTS_DIR / "doctor.py"),
        "--repo-root",
        str(repo_root),
        "--json",
        *flags,
    ]
    completed = subprocess.run(
        argv,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return DoctorRun(completed.returncode, completed.stdout, completed.stderr)


def _resolve(report: Any, dotted: str) -> tuple[bool, Any]:
    """Look up ``a.b.c`` in a nested report; reports whether it was found."""
    node = report
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _is_measurement(value: Any) -> bool:
    # bool is an int subclass: a measurement is a number, a flag is not.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare(left: Any, right: Any, path: str, found: list[str], measured: tuple[str, ...]) -> None:
    # An exemption applies to a *number* that differs. Registering a path whose
    # value is a container therefore cannot excuse everything under it: the walk
    # descends and compares the children as usual.
    if path in measured and _is_measurement(left) and _is_measurement(right):
        return
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            _compare(left.get(key, _MISSING), right.get(key, _MISSING), child, found, measured)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            found.append(f"{path}: {len(left)} entries vs {len(right)}")
            return
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            _compare(item_left, item_right, f"{path}[{index}]", found, measured)
        return
    if left != right:
        found.append(f"{path}: {_render(left)} != {_render(right)}")


def _render(value: Any) -> str:
    return "<missing>" if value is _MISSING else json.dumps(value)


def differences(left: Any, right: Any, *, measured: tuple[str, ...] = MEASURED_FIELDS) -> list[str]:
    """Every way two reports disagree, ignoring only the registered measurements.

    A registered path that is absent from both reports is itself reported: it
    means the field was renamed or dropped, so the exemption now protects
    nothing and the next real change there would be compared under a stale rule.
    """
    found: list[str] = []
    _compare(left, right, "", found, measured)
    for path in measured:
        if not _resolve(left, path)[0] and not _resolve(right, path)[0]:
            found.append(f"{path}: registered as a measurement but absent from the report")
    return found


def mask(report: Any) -> Any:
    """A copy of the report with the registered measurements blanked out.

    Used only to print a readable diff: the comparison itself is stricter, since
    it distinguishes the numbers from everything around them.
    """
    masked = copy.deepcopy(report)
    for path in MEASURED_FIELDS:  # the registry the comparison uses by default
        parent = masked
        parts = path.split(".")
        for part in parts[:-1]:
            if not isinstance(parent, dict) or part not in parent:
                parent = None
                break
            parent = parent[part]
        if isinstance(parent, dict) and _is_measurement(parent.get(parts[-1])):
            parent[parts[-1]] = _MASKED
    return masked


def _tail(text: str, limit: int = 4000) -> str:
    stripped = text.strip()
    if not stripped:
        return "<no output captured>"
    return stripped[-limit:]


def _print_diff(left: dict[str, Any], right: dict[str, Any]) -> None:
    before = json.dumps(mask(left), indent=2, sort_keys=True).splitlines()
    after = json.dumps(mask(right), indent=2, sort_keys=True).splitlines()
    for line in difflib.unified_diff(before, after, "first run", "second run", lineterm=""):
        print(line)


def main(argv: list[str] | None = None, *, runner=run_doctor) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--skip-model-init", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--model-timeout", type=float, default=None)
    parser.add_argument("--qt-timeout", type=float, default=None)
    args = parser.parse_args(argv)

    flags: list[str] = []
    if args.skip_model_init:
        flags.append("--skip-model-init")
    if args.ci:
        flags.append("--ci")
    if args.model_timeout is not None:
        flags += ["--model-timeout", str(args.model_timeout)]
    if args.qt_timeout is not None:
        flags += ["--qt-timeout", str(args.qt_timeout)]

    repo_root = args.repo_root.expanduser().resolve()
    mode = " ".join(flags) or "full (no flags)"
    print(f"Doctor idempotence: {SCRIPTS_DIR / 'doctor.py'} twice, {mode}, at {repo_root}", flush=True)

    runs = {"first": runner(repo_root, flags), "second": runner(repo_root, flags)}
    for label, run in runs.items():
        if run.report is None:
            print(f"\nFAIL: the {label} run produced no report (exit {run.returncode}).", flush=True)
            print(_tail(run.stderr) if run.stderr.strip() else _tail(run.stdout))
            return 1

    first, second = runs["first"], runs["second"]
    assert first.report is not None and second.report is not None  # narrowed above
    print(
        f"First run exit {first.returncode} (ready={first.report.get('overall_ready')}); "
        f"second run exit {second.returncode} (ready={second.report.get('overall_ready')})"
    )

    found = differences(first.report, second.report)
    if first.returncode != second.returncode:
        found.insert(0, f"exit code: {first.returncode} != {second.returncode}")
    if found:
        print("\nFAIL: the two doctor reports differ, so the gate's verdict depends on how")
        print("many times it has been run. Each line is a field that must not change:")
        for line in found:
            print(f"  - {line}")
        print("\nWhat differed (measurements masked):")
        _print_diff(first.report, second.report)
        print(
            "\nIf a field above is a duration of the machine rather than a statement about\n"
            "the install, add its dotted path to MEASURED_FIELDS in this script. Otherwise\n"
            "the check it belongs to is grading state that varies between runs."
        )
        return 1

    print(f"PASS: both reports identical ({len(MEASURED_FIELDS)} measurements excluded).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
