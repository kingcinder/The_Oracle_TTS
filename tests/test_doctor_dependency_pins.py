"""Tests for the doctor's dependency-pin check (scripts/doctor.py).

The check exists because a full-suite green claim is only valid against the
declared dependency set: an out-of-band pip install once silently replaced a
pinned package and broke tests that were green the day before (JUNO_FIXES.log,
2026-09-20 docs entry). These tests pin both directions: the real venv against
the real pyproject (no drift allowed), and synthetic drift detection.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTOR_PATH = REPO_ROOT / "scripts" / "doctor.py"


def _load_doctor():
    spec = importlib.util.spec_from_file_location("oracle_doctor_pins", DOCTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_real_venv_matches_real_pyproject_pins() -> None:
    """The gate's live verdict on this tree: zero drift."""
    doctor = _load_doctor()
    status = doctor._dependency_pin_status(REPO_ROOT)
    assert status["ok"] is True, status["error"]
    assert status["mismatches"] == []
    assert status["missing"] == []
    assert status["checked_count"] >= 15  # main deps + ml + chatterbox + dev extras


def test_pin_drift_is_detected(monkeypatch, tmp_path: Path) -> None:
    """An installed version violating the pin is reported per requirement."""
    doctor = _load_doctor()
    real_version = doctor.importlib.metadata.version

    def fake_version(name: str) -> str:
        return "9.9.9" if name == "soundfile" else real_version(name)

    monkeypatch.setattr(doctor.importlib.metadata, "version", fake_version)
    status = doctor._dependency_pin_status(REPO_ROOT)
    assert status["ok"] is False
    assert any(entry["requirement"].startswith("soundfile") for entry in status["mismatches"])
    assert "soundfile" in status["error"]


def test_missing_package_is_reported(monkeypatch) -> None:
    doctor = _load_doctor()
    real_version = doctor.importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "mutagen":
            raise doctor.importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(doctor.importlib.metadata, "version", fake_version)
    status = doctor._dependency_pin_status(REPO_ROOT)
    assert status["ok"] is False
    assert any(entry["requirement"].startswith("mutagen") for entry in status["missing"])
    assert "mutagen" in status["error"]


def test_every_pinned_group_is_checked_including_dev() -> None:
    """The check covers main dependencies and every optional extra."""
    import tomllib

    doctor = _load_doctor()
    status = doctor._dependency_pin_status(REPO_ROOT)
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    project = data["project"]
    declared = len(project["dependencies"]) + sum(
        len(group) for group in project.get("optional-dependencies", {}).values()
    )
    assert status["checked_count"] == declared
    assert "dev" in project.get("optional-dependencies", {})


def test_multi_occurrence_drift_names_every_pin_in_the_error() -> None:
    """The 71d8840 completeness precedent, applied to the doctor's per-record
    reporting: TWO drifted pins AND a missing package in one report must ALL be
    named in the composed error — the single string both the human report line
    and the next-steps step render. A first-match-only composition (one mismatch
    wins, the rest silently dropped) would hide all but the earliest drift while
    the check still reported a failure. Pinned with exact membership per token,
    vacuity-guarded by the clean run above proving the scan reads real pins."""
    doctor = _load_doctor()
    real_version = doctor.importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "soundfile":
            return "9.9.9"  # drifts the soundfile pin
        if name == "mutagen":
            return "0.0.1"  # drifts the mutagen pin too
        if name == "numpy":
            raise doctor.importlib.metadata.PackageNotFoundError(name)  # missing pin
        return real_version(name)

    monkeypatch = __import__("pytest").MonkeyPatch()
    try:
        monkeypatch.setattr(doctor.importlib.metadata, "version", fake_version)
        status = doctor._dependency_pin_status(REPO_ROOT)
    finally:
        monkeypatch.undo()
    assert status["ok"] is False
    assert status["missing"], "vacuity: the probe must hit the real pin set"
    error = status["error"]
    for token in ("soundfile", "mutagen", "numpy"):
        assert token in error, (
            f"the composed error must name EVERY drifted/missing pin, not just the "
            f"first: {token!r} is absent from {error!r}"
        )
    # And the structured lists agree with the prose — per-occurrence, not deduped.
    assert {entry["requirement"].split("==")[0] for entry in status["mismatches"]} >= {"soundfile", "mutagen"}
    assert [entry["requirement"] for entry in status["missing"]] == ["numpy>=1.26"]


def test_missing_pyproject_fails_cleanly(tmp_path: Path) -> None:
    doctor = _load_doctor()
    status = doctor._dependency_pin_status(tmp_path)
    assert status["ok"] is False
    assert "pyproject" in status["error"]


def test_range_pins_are_satisfied_not_just_exact_pins() -> None:
    """numpy>=1.26 style ranges must evaluate against the real installed version."""
    import packaging.requirements

    requirement = packaging.requirements.Requirement("numpy>=1.26")
    import importlib.metadata

    installed = importlib.metadata.version("numpy")
    assert requirement.specifier.contains(installed, prereleases=True)


def test_check_is_wired_into_the_report_and_gate() -> None:
    """The status function is registered as a required check in run()."""
    source = DOCTOR_PATH.read_text(encoding="utf-8")
    assert '"dependency_pins": _dependency_pin_status(repo_root)' in source
    assert 'report["dependency_pins"]["ok"]' in source
    # The human report and next-steps both name it.
    assert 'Dependency pins:' in source
    assert 'Dependency drift detected:' in source


def _load_script(name: str):
    """Import a scripts/ module by path (they are entry points, not a package)."""
    spec = importlib.util.spec_from_file_location(
        f"oracle_{name.removesuffix('.py')}", REPO_ROOT / "scripts" / name
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_python_version_guard_matches_doctor_target() -> None:
    """One support range, agreed on by pyproject, the doctor, the installer and CI.

    This replaced an assertion that the *runner* was Python 3.12
    (``sys.version_info[:2] == (3, 12)``). GitHub's matrix includes py3.11 on
    both operating systems, so that assertion was red on a fully supported
    interpreter — and it said nothing about the guard its name promises.

    What must hold anywhere is agreement, not a version: pyproject's
    ``requires-python``, the doctor's ``SUPPORTED_PYTHON_MIN``/``MAX``, the
    installer's copies of those constants, and every ``python-version`` row the
    test job runs. The interpreter running this suite must be inside the range
    too, or every other check in this file is measuring an unsupported one.
    """
    import platform
    import re
    import tomllib

    import packaging.specifiers

    doctor = _load_doctor()
    installer = _load_script("manage_install.py")

    def declared_range(module) -> str:
        low, high = module.SUPPORTED_PYTHON_MIN, module.SUPPORTED_PYTHON_MAX
        return f">={low[0]}.{low[1]},<{high[0]}.{high[1]}"

    def compact(text: str) -> str:
        return re.sub(r"\s+", "", text)

    requires = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["requires-python"]
    assert (
        compact(requires)
        == compact(declared_range(doctor))
        == compact(declared_range(installer))
    ), (
        "the supported Python range drifted: pyproject says "
        f"{requires!r}, the doctor says {declared_range(doctor)!r}, the "
        f"installer says {declared_range(installer)!r}"
    )

    supported = packaging.specifiers.SpecifierSet(requires)
    assert supported.contains(platform.python_version()), (
        f"this suite is running on Python {platform.python_version()}, which "
        f"the project does not support ({requires})"
    )

    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    assert matrix, "the test matrix declares no python-version rows"
    for version in matrix:
        assert supported.contains(str(version)), (
            f"the CI matrix runs Python {version}, which the project does not "
            f"support ({requires})"
        )
