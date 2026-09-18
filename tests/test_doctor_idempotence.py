"""Tests for the doctor idempotence check that CI gates on.

The check itself runs the doctor twice for real, which costs ~20s, so what is
tested here is everything around that: the comparison rules (including the
exemption registry and its drift guard), the failure messages, and the CI
contract that keeps the check running on every push and pull request. The
end-to-end run is exercised by ``.github/workflows/ci.yml`` on both runners and
was verified live against a deliberately re-introduced instance of the historical
defect (a readiness check writing the smoke inputs the next run counts).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture()
def checker():
    spec = importlib.util.spec_from_file_location(
        "oracle_doctor_idempotence", SCRIPTS_DIR / "doctor_idempotence.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(**overrides):
    """A small stand-in for the doctor's report, shaped like the real one."""
    report = {
        "overall_ready": True,
        "voice_sources": {"ok": True, "fallback_clip_count": 0},
        "chatterbox_init": {"ok": True, "seconds": 7.5, "skipped": False},
        "deterministic_smoke": {"ok": True, "runtime_seconds": 0.9, "error": ""},
        "cuda_backend": {"devices": [{"index": 0, "name": "GPU"}]},
        "next_steps": ["first", "second"],
    }
    for dotted, value in overrides.items():
        node = report
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return report


# --- the comparison rules -----------------------------------------------------


def test_identical_reports_have_no_differences(checker) -> None:
    assert checker.differences(_report(), _report()) == []


def test_a_changed_verdict_is_named_with_both_values(checker) -> None:
    """The exact shape of the historical defect: a count inflated by run 1."""
    found = checker.differences(_report(), _report(**{"voice_sources.fallback_clip_count": 2}))

    assert found == ["voice_sources.fallback_clip_count: 0 != 2"]


def test_registered_measurements_are_the_only_allowed_difference(checker) -> None:
    left = _report()
    right = _report(**{"deterministic_smoke.runtime_seconds": 1.4, "chatterbox_init.seconds": 9.1})

    assert checker.differences(left, right) == []


def test_the_exemption_registry_holds_only_the_known_measurements(checker) -> None:
    """The registry is the one hole in the comparison, so widening it is pinned.

    Every entry excuses a difference, so adding one has to be a deliberate,
    reviewable change rather than a quiet way to silence a failing field.
    """
    assert checker.MEASURED_FIELDS == (
        "chatterbox_init.seconds",
        "deterministic_smoke.runtime_seconds",
    )


def test_a_registered_path_excuses_only_that_number(checker) -> None:
    """An exemption is narrow: it cannot be widened into a subtree."""
    changed = _report(**{"voice_sources.fallback_clip_count": 2})

    # Registered exactly, the difference is excused...
    assert checker.differences(
        _report(), changed, measured=("voice_sources.fallback_clip_count",)
    ) == []
    # ...but registering the container above it does not excuse its children,
    # which is what stops a careless entry from hiding a whole check's state.
    assert checker.differences(_report(), changed, measured=("voice_sources",)) == [
        "voice_sources.fallback_clip_count: 0 != 2"
    ]


def test_a_measurement_that_changes_shape_is_still_a_difference(checker) -> None:
    """The exemption covers a number on both sides, not the presence of a value.

    A skipped init reports ``null``; if one run skipped and the other measured,
    that is a real difference in what the gate did and must not be swallowed by
    the exemption.
    """
    found = checker.differences(_report(), _report(**{"chatterbox_init.seconds": None}))

    assert found == ["chatterbox_init.seconds: 7.5 != null"]


def test_an_unregistered_duration_is_reported(checker) -> None:
    """A new measurement must fail loudly, pointing at the registry."""
    left = _report()
    right = _report()
    left["qt"] = {"ok": True, "seconds": 0.4}
    right["qt"] = {"ok": True, "seconds": 0.6}

    assert checker.differences(left, right) == ["qt.seconds: 0.4 != 0.6"]


def test_a_renamed_measurement_fails_as_registry_drift(checker) -> None:
    """Both runs lacking the field is the rename case: only the guard catches it."""
    report = _report()
    del report["deterministic_smoke"]["runtime_seconds"]
    renamed = _report()
    del renamed["deterministic_smoke"]["runtime_seconds"]

    assert checker.differences(report, renamed) == [
        "deterministic_smoke.runtime_seconds: registered as a measurement but absent from the report"
    ]


def test_a_missing_measurement_on_one_side_is_a_difference(checker) -> None:
    report = _report()
    del report["deterministic_smoke"]["runtime_seconds"]

    found = checker.differences(_report(), report)

    assert "deterministic_smoke.runtime_seconds: 0.9 != <missing>" in found


def test_list_differences_are_reported_with_their_index(checker) -> None:
    left = _report()
    right = _report(**{"next_steps": ["first", "changed"]})

    assert checker.differences(left, right) == ["next_steps[1]: \"second\" != \"changed\""]


def test_list_length_differences_are_reported(checker) -> None:
    found = checker.differences(_report(), _report(**{"next_steps": ["first"]}))

    assert found == ["next_steps: 2 entries vs 1"]


def test_added_and_removed_fields_are_reported(checker) -> None:
    right = _report()
    right["brand_new"] = True

    assert checker.differences(_report(), right) == ["brand_new: <missing> != true"]


def test_mask_blanks_only_registered_measurements(checker) -> None:
    """Masking is for the printed diff; it must not touch anything else."""
    masked = checker.mask(_report())

    assert masked["deterministic_smoke"]["runtime_seconds"] == "<measured>"
    assert masked["chatterbox_init"]["seconds"] == "<measured>"
    assert masked["voice_sources"]["fallback_clip_count"] == 0
    assert masked["deterministic_smoke"]["ok"] is True


# --- driving the check --------------------------------------------------------


class _FakeRunner:
    """Replaces `run_doctor`: canned reports, and a record of how it was called."""

    def __init__(self, doctor_run, *reports, returncode: int = 0, stderr: str = "") -> None:
        self._doctor_run = doctor_run
        self._reports = list(reports)
        self._returncode = returncode
        self._stderr = stderr
        self.calls: list[tuple[Path, list[str]]] = []

    def __call__(self, repo_root: Path, flags: list[str]):
        self.calls.append((repo_root, list(flags)))
        report = self._reports.pop(0) if len(self._reports) > 1 else self._reports[0]
        stdout = report if isinstance(report, str) else json.dumps(report)
        return self._doctor_run(self._returncode, stdout, self._stderr)


def test_main_passes_when_only_measurements_differ(checker, capsys) -> None:
    runner = _FakeRunner(
        checker.DoctorRun,
        _report(),
        _report(**{"deterministic_smoke.runtime_seconds": 3.3}),
    )

    assert checker.main(["--skip-model-init", "--ci"], runner=runner) == 0
    assert "PASS: both reports identical" in capsys.readouterr().out


def test_main_fails_when_the_reports_differ(checker, capsys) -> None:
    runner = _FakeRunner(checker.DoctorRun, _report(), _report(**{"voice_sources.fallback_clip_count": 2}))

    assert checker.main([], runner=runner) == 1
    out = capsys.readouterr().out
    assert "voice_sources.fallback_clip_count: 0 != 2" in out
    # The failure has to tell the next person what to do about it.
    assert "MEASURED_FIELDS" in out


def test_main_runs_both_runs_in_the_same_mode(checker) -> None:
    """A comparison is only meaningful if both runs were asked the same thing."""
    runner = _FakeRunner(checker.DoctorRun, _report(), _report())

    checker.main(["--skip-model-init", "--ci", "--qt-timeout", "30"], runner=runner)

    assert len(runner.calls) == 2
    assert runner.calls[1][1] == runner.calls[0][1]
    assert runner.calls[0][1] == ["--skip-model-init", "--ci", "--qt-timeout", "30.0"]
    assert runner.calls[0][0] == runner.calls[1][0]


def test_main_fails_when_a_run_produces_no_report(checker, capsys) -> None:
    runner = _FakeRunner(
        checker.DoctorRun,
        _report(),
        "Traceback (most recent call last): boom",
        returncode=2,
        stderr="boom",
    )

    assert checker.main([], runner=runner) == 1
    out = capsys.readouterr().out
    assert "the second run produced no report (exit 2)" in out
    assert "boom" in out


def test_main_reports_the_exit_codes_it_compared(checker, capsys) -> None:
    runner = _FakeRunner(checker.DoctorRun, _report(), _report(), returncode=1)

    checker.main([], runner=runner)

    assert "First run exit 1" in capsys.readouterr().out


# --- the CI contract ----------------------------------------------------------


def test_workflow_runs_the_idempotence_check_on_every_runner() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]

    step = next(
        step for step in steps if "doctor_idempotence.py" in step.get("run", "")
    )
    # No `if:` and no `shell:` override, so each runner uses its own default
    # shell (bash on Linux, pwsh on Windows) and the check runs on both.
    assert "if" not in step
    assert "shell" not in step
    assert step["run"].strip() == "python scripts/doctor_idempotence.py --skip-model-init --ci"

    # Same mode as the doctor steps the gate already runs, so the check measures
    # the gate as it is actually invoked.
    doctor_steps = [
        s
        for s in steps
        if "manage_install.py doctor" in s.get("run", "") or "doctor_oracle_tts" in s.get("run", "")
    ]
    assert doctor_steps
    for doctor_step in doctor_steps:
        assert "--skip-model-init" in doctor_step["run"]
        assert "--ci" in doctor_step["run"]
