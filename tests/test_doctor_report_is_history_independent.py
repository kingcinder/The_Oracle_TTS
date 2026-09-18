"""Every doctor PASS line must be earned by this run, not inherited from an earlier one.

The doctor gates a launch, so its verdicts have to describe the machine rather
than the doctor's own history. Two regressions came from exactly that: the
real-engine readiness PASS cited a smoke FLAC the check had never produced, and
the default-voice-source count was inflated by reference clips a previous run had
written. Both were invisible in a single run, and both were fixed -- but nothing
asserted the property for the *other* PASS lines, so a third check could
reintroduce it and only be noticed if its write happened to change a report the
idempotence check compares.

So this runs the real doctor in the mode CI gates on, then runs it again with
``build/`` -- everything the doctor generates -- parked, and requires *every
verdict field* to be identical. Removing all of it is what makes the check
catch the historical case as well: those reference clips lived in
``build/real_engine_smoke/inputs``, not in the smoke's own project.

The comparison is on verdicts rather than on the whole report, because the rest
of the report is measurement by design. ``voice_sources.fallback_clip_count``
answers "how many reference clips are on disk", and answering that differently
when the files are removed is the field doing its job; a *PASS* moving is the
defect. That also means no exemptions are registered here at all: the verdict
payload contains no durations, so none are excluded and nothing can be excused.
The comparison itself is the one the CI idempotence gate uses, so the two cannot
drift on how reports are compared.

What this does not cover, stated plainly: a check that grades an artifact it did
not produce and never wrote -- the git-ignored native ``audio.cpp`` build, a
user-placed reference clip -- is a capability difference, not a history
difference, and is reported as such by ``scripts/fresh_clone_acceptance.py``
against a clean checkout. This test is about the doctor's own footprints.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: Everything a doctor run generates. ``tests/test_doctor_read_only.py`` measured
#: that a run writes only inside ``build/doctor_deterministic_smoke/``; parking
#: all of ``build/`` covers that and the residue earlier runs left elsewhere.
BUILD_DIR = REPO_ROOT / "build"
PARKED_BUILD = REPO_ROOT / "build.__history_test_parked"

#: Same mode the CI matrix invokes the gate with, so this measures the gate as
#: it is actually used rather than a mode only a test chooses.
DOCTOR_FLAGS = ("--skip-model-init", "--ci")

#: The PASS lines a report must contain for this test to mean anything. A field
#: renamed away would otherwise shrink the comparison silently.
EXPECTED_VERDICTS = frozenset(
    {
        "chatterbox_import.ok",
        "deterministic_smoke.ok",
        "perth.ok",
        "python.ok",
        "qt.ok",
        "real_engine_smoke.ok",
        "real_engine_smoke.ready",
        "vulkan_backend.ok",
        "voice_sources.ok",
    }
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def idempotence():
    return _load("oracle_doctor_idempotence_for_history", "doctor_idempotence.py")


def _doctor_report() -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "doctor.py"),
            "--repo-root",
            str(REPO_ROOT),
            "--json",
            *DOCTOR_FLAGS,
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        report = json.loads(completed.stdout)
    except ValueError as error:  # pragma: no cover - a doctor failure is its own test
        raise AssertionError(
            f"the doctor produced no report (exit {completed.returncode}): {error}"
        ) from error
    assert isinstance(report, dict)
    return report


def verdict_payload(report: dict) -> dict:
    """The report reduced to its verdict fields, as ``{section: {key: bool}}``.

    Gathered by *shape* rather than from a list of names: a check added later is
    covered without anyone remembering to add it here. Non-dict sections keep
    their own verdict if they have one, so ``overall_ready`` is included too.
    """
    payload: dict[str, Any] = {}
    for section, value in report.items():
        if isinstance(value, bool) and section == "overall_ready":
            payload[section] = value
        elif isinstance(value, dict):
            verdicts = {
                key: item
                for key, item in value.items()
                if isinstance(item, bool) and (key in {"ok", "ready"} or key.endswith("_ok"))
            }
            if verdicts:
                payload[section] = verdicts
    return payload


def verdicts_in(report: dict) -> set[str]:
    """The dotted names of every verdict field, for the vacuity guard."""
    return {
        f"{section}.{key}" if section != "overall_ready" else section
        for section, value in verdict_payload(report).items()
        for key in (value if isinstance(value, dict) else [section])
    }


@contextmanager
def generated_state_absent():
    """Run the body with everything the doctor generates gone, then put it back.

    A rename rather than a copy, so this is instant even though the tree holds
    gigabytes, and the body's own doctor run is discarded afterwards so the tree
    ends exactly as it began.
    """
    had_build = BUILD_DIR.exists()
    if had_build:
        shutil.rmtree(PARKED_BUILD, ignore_errors=True)
        shutil.move(str(BUILD_DIR), str(PARKED_BUILD))
    try:
        yield
    finally:
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
        if had_build:
            shutil.move(str(PARKED_BUILD), str(BUILD_DIR))


def test_no_verdict_depends_on_state_this_machine_left_behind(idempotence) -> None:
    already_run = verdict_payload(_doctor_report())

    with generated_state_absent():
        assert not BUILD_DIR.exists(), (
            "the doctor's generated state is still present, so this run would prove nothing"
        )
        fresh_machine = verdict_payload(_doctor_report())

    # measured=() because this payload is verdicts only: there are no durations
    # to exempt, and registering one would excuse a field that is not here.
    found = idempotence.differences(already_run, fresh_machine, measured=())

    assert found == [], (
        "these PASS lines changed when the state this machine left behind was removed, "
        "so they are graded from history rather than from a capability verified in "
        "the run:\n  " + "\n  ".join(found)
    )


def test_the_report_actually_contains_the_pass_lines_being_guarded() -> None:
    """Guards the guard: comparing two empty verdict sets would pass for free."""
    report = _doctor_report()

    present = verdicts_in(report)

    assert EXPECTED_VERDICTS <= present, f"missing from the report: {EXPECTED_VERDICTS - present}"
    assert "overall_ready" in report
    assert verdict_payload(report), "the verdict payload is empty"


def test_the_comparison_would_notice_a_verdict_that_moved(idempotence) -> None:
    """`differences() == []` only means something if it can be non-empty."""
    left = {"python": {"ok": True}, "deterministic_smoke": {"ok": True, "runtime_seconds": 1.0}}
    right = {"python": {"ok": True}, "deterministic_smoke": {"ok": False, "runtime_seconds": 2.0}}

    found = idempotence.differences(left, right)

    assert any("deterministic_smoke.ok" in line for line in found)
    # ... while the registered measurement stays exempt, so the empty result
    # above is the exemption working and not the comparison being inert.
    assert not any("runtime_seconds" in line for line in found)
