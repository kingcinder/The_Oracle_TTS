"""The doctor's crash_reports check (scripts/doctor.py::_crash_reports_status).

Per docs/CRASH_TELEMETRY_DESIGN §7: opted-out is a valid state, cap and
native-dump states are explained inline, writability failures fail the check
with a local remedy, and the check is read-only and idempotent.

**Mutation contract (M-DOC-CRASH):** flipping the opted-out branch to
``ok=False`` fails test_opted_out_is_a_valid_state; treating a not-yet-
existing directory as a write failure fails
test_never_crashed_install_is_not_a_failure.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTOR_PATH = REPO_ROOT / "scripts" / "doctor.py"


def _load_doctor():
    spec = importlib.util.spec_from_file_location("oracle_doctor_crash", DOCTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _status(root: Path) -> dict:
    return _load_doctor()._crash_reports_status(root)


def test_opted_out_is_a_valid_state(tmp_path: Path) -> None:
    """**M-DOC-CRASH target:** a privacy preference is never a failure."""
    status = _status(tmp_path)
    assert status["ok"] is True
    assert status["consent"] is False
    assert "privacy-opt-in" in status["detail"]


def test_opted_in_never_crashed_install_is_not_a_failure(tmp_path: Path) -> None:
    from the_oracle import crash

    crash.write_consent(tmp_path, True)
    status = _status(tmp_path)
    assert status["ok"] is True
    assert status["consent"] is True
    assert status["record_count"] == 0
    assert status["newest"] is None


def test_records_are_counted_and_newest_exception_named(tmp_path: Path) -> None:
    from the_oracle import crash

    crash.write_consent(tmp_path, True)
    crash.write_record(tmp_path, {"exception": {"type": "ValueError", "message": "x"}, "log_tail": []})
    crash.write_record(tmp_path, {"exception": {"type": "OSError", "message": "y"}, "log_tail": []})
    status = _status(tmp_path)
    assert status["record_count"] == 2
    assert status["newest_exception"] == "OSError"
    assert status["total_bytes"] > 0
    assert status["at_cap"] is False


def test_cap_state_is_reported(tmp_path: Path) -> None:
    from the_oracle import crash

    crash.write_consent(tmp_path, True)
    for i in range(crash.MAX_RECORDS):
        crash.write_record(tmp_path, {"exception": {"type": "T", "message": str(i)}, "log_tail": []})
    status = _status(tmp_path)
    assert status["at_cap"] is True
    assert "at cap" in status["detail"]


def test_native_dump_is_surfaced(tmp_path: Path) -> None:
    from the_oracle import crash

    crash.write_consent(tmp_path, True)
    dump = crash.crash_dir(tmp_path) / "native-crash.txt"
    dump.parent.mkdir(parents=True, exist_ok=True)  # an enable-less install may never have crashed
    dump.write_text("faulthandler dump\n", encoding="utf-8")
    status = _status(tmp_path)
    assert status["native_dump_present"] is True


def test_unwritable_directory_fails_with_local_remedy(tmp_path: Path) -> None:
    from the_oracle import crash

    crash.write_consent(tmp_path, True)
    directory = crash.crash_dir(tmp_path)
    directory.mkdir()
    directory.chmod(0o500)
    try:
        status = _status(tmp_path)
        assert status["ok"] is False
        assert "not writable" in status["detail"]
        assert "go online" not in status["detail"].lower()
    finally:
        directory.chmod(0o700)


def test_check_is_read_only_and_idempotent(tmp_path: Path) -> None:
    from the_oracle import crash

    crash.write_consent(tmp_path, True)
    crash.write_record(tmp_path, {"exception": {"type": "T", "message": "m"}, "log_tail": []})

    def snapshot() -> dict[str, bytes]:
        return {str(p): p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()}

    before = snapshot()
    first = _status(tmp_path)
    second = _status(tmp_path)
    assert first == second
    assert snapshot() == before


def test_doctor_wiring_pins() -> None:
    """Wired into build_report, next_steps, and the human report (source-scan
    pins, the repo's convention for wiring a heavy build_report can't reach)."""
    source = DOCTOR_PATH.read_text(encoding="utf-8")
    assert '"crash_reports": _crash_reports_status(repo_root)' in source
    assert "native-crash dump exists" in source  # next_steps surfacing
    assert "Crash reports: disabled" in source  # human-report line
