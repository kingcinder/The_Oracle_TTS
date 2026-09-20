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


def test_python_version_guard_matches_doctor_target() -> None:
    """Sanity: the suite itself runs on the interpreter the doctor validates."""
    assert sys.version_info[:2] == (3, 12)
