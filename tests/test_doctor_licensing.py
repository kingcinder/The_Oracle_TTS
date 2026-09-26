"""The doctor's licensing check (scripts/doctor.py::_licensing_status).

Per docs/LICENSING_DESIGN.md §5: unlicensed is a valid state (``ok=true``),
each failure carries its offline-safe remedy inline, the check is read-only
and idempotent, and it is deliberately excluded from ``overall_ready`` (an
expired license is a billing state, not a broken install).

**Mutation contract:** flipping the no_token branch to ``ok=False`` fails
test_unlicensed_install_is_a_valid_state; dropping the expired-state
detection (reporting expired as ok) fails test_expired_license_fails_its_own_check.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTOR_PATH = REPO_ROOT / "scripts" / "doctor.py"

SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")

PAYLOAD = {
    "v": 1,
    "key_id": "k1",
    "lic_id": "55555555-5555-4555-8555-555555555555",
    "edition": "studio",
    "licensee": "Cody",
    "machine_hash": "",
    "iat": 1700000000,
    "exp": None,
}


def _load_doctor():
    spec = importlib.util.spec_from_file_location("oracle_doctor_licensing", DOCTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _store_token(root: Path, payload: dict) -> None:
    from the_oracle.licensing import store, tokens

    token = tokens.mint(payload, SEED)
    result = store.save_token(root, token, verify_before_save=lambda t: tokens.verify_token(t))
    assert result.state == "saved", result


def _write_token_direct(root: Path, payload: dict) -> None:
    """Write a token bypassing the save gate — the shape of an install whose
    stored token AGED past its exp (the save gate correctly refuses such a
    token at activation time; it cannot prevent one from expiring later)."""
    from the_oracle.licensing import store, tokens

    token = tokens.mint(payload, SEED)
    store.store_path(root).write_text(json.dumps({"token": token}), encoding="utf-8")


def test_real_repo_has_no_license_and_reports_valid(tmp_path: Path) -> None:
    doctor = _load_doctor()
    status = doctor._licensing_status(tmp_path)
    assert status["ok"] is True
    assert status["state"] == "no_token"
    assert status["edition"] == "community"


def test_unlicensed_install_is_a_valid_state(tmp_path: Path) -> None:
    """**Mutation target:** no_token must be ok=True — the doctor never fails
    an install for lacking a license."""
    doctor = _load_doctor()
    status = doctor._licensing_status(tmp_path)
    assert status["ok"] is True, status


def test_valid_license_reports_edition_and_licensee(tmp_path: Path) -> None:
    doctor = _load_doctor()
    _store_token(tmp_path, PAYLOAD)
    status = doctor._licensing_status(tmp_path)
    assert status["ok"] is True
    assert status["state"] == "valid"
    assert status["edition"] == "studio"
    assert status["licensee"] == "Cody"
    assert status["key_id"] == "k1"


def test_expired_license_fails_its_own_check(tmp_path: Path) -> None:
    doctor = _load_doctor()
    _write_token_direct(tmp_path, dict(PAYLOAD, exp=int(time.time()) - 10))
    status = doctor._licensing_status(tmp_path)
    assert status["ok"] is False
    assert status["state"] == "expired"
    assert status["detail"]


def test_corrupted_store_reports_store_error_with_remedy(tmp_path: Path) -> None:
    from the_oracle.licensing import store

    doctor = _load_doctor()
    store.store_path(tmp_path).write_text("{broken", encoding="utf-8")
    status = doctor._licensing_status(tmp_path)
    assert status["ok"] is False
    assert status["state"] == "store_error"
    assert "activate" in status["detail"]


def test_check_is_read_only_and_idempotent(tmp_path: Path) -> None:
    doctor = _load_doctor()
    _store_token(tmp_path, PAYLOAD)

    def snapshot() -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    before = snapshot()
    first = doctor._licensing_status(tmp_path)
    second = doctor._licensing_status(tmp_path)
    assert first == second
    assert snapshot() == before


def test_all_remedy_strings_are_offline_safe(tmp_path: Path) -> None:
    """No failure detail may instruct a network action (design doc §5/§6)."""
    doctor = _load_doctor()
    from the_oracle.licensing import store

    # Scenario 1: no token at all.
    status = doctor._licensing_status(tmp_path)
    detail = (status.get("detail") or "").lower()
    assert "go online" not in detail
    assert "connect to" not in detail
    assert "download" not in detail

    # Scenario 2: an expired license (aged past exp after activation).
    _write_token_direct(tmp_path, dict(PAYLOAD, exp=int(time.time()) - 10))
    status = doctor._licensing_status(tmp_path)
    detail = (status.get("detail") or "").lower()
    assert "go online" not in detail
    assert "connect to" not in detail
    assert "download" not in detail
    store.clear_token(tmp_path)


def test_doctor_wiring_pins() -> None:
    """The check is wired into build_report, next_steps, and the human report.

    Source-scan pins (the repo's established convention for wiring that a
    full build_report call cannot reach in a unit test without running the
    heavy checks). Removing the wiring flips this test."""
    source = DOCTOR_PATH.read_text(encoding="utf-8")
    assert '"licensing": _licensing_status(repo_root)' in source
    assert 'License issue' in source  # next_steps entry
    assert "License: not activated" in source  # human-report line


def test_licensing_is_excluded_from_overall_ready() -> None:
    """Deliberate policy: licensing must NOT appear in required_checks."""
    source = DOCTOR_PATH.read_text(encoding="utf-8")
    required_block = source.split("required_checks = [", 1)[1].split("]", 1)[0]
    assert "licensing" not in required_block
