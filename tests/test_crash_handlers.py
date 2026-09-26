"""The handlers: chaining, fail-safety, fire-time consent, idempotence.

**Mutation contracts:** making _capture re-raise (or consent-off capture
write anyway) fails the dedicated tests here; removing the previous-hook
chaining fails the suppression test; making enable_faulthandler_catch arm
without consent fails the native-crash test.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import crash  # noqa: E402
from the_oracle.crash import bundle, consent, handlers  # noqa: E402

LOG_FILE = "fake.log"


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch):
    """Install handlers against a sandbox root with a tiny fake log."""
    log_path = tmp_path / LOG_FILE
    log_path.write_text("INFO | line one\nINFO | line two\n", encoding="utf-8")
    handlers.install(root=tmp_path, log_file=log_path)
    yield tmp_path
    # Restore the real hooks no matter what the test did.
    sys.excepthook = sys.__excepthook__
    threading.excepthook = threading.__excepthook__
    handlers._STATE.clear()


def test_install_and_idempotence(wired: Path) -> None:
    # pytest's threadexception plugin owns threading.excepthook during test
    # bodies, so identity is pinned on sys.excepthook (pytest leaves it alone)
    # and on remember-once semantics for the threading hook.
    assert sys.excepthook is handlers._sys_excepthook
    remembered_threading = handlers._STATE.get("previous_threading_hook")
    handlers.install(root=wired)  # second install: no stacking, no re-remember
    assert handlers._STATE.get("previous_threading_hook") is remembered_threading
    assert sys.excepthook is handlers._sys_excepthook


def test_consent_off_captures_nothing(wired: Path) -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    assert crash.list_records(wired) == []


def test_consent_on_captures_and_sanitizes(wired: Path) -> None:
    consent.write_consent(wired, True)
    try:
        raise ValueError(f"bad path {wired}/Input/secret.txt")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    records = crash.list_records(wired)
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["exception"]["type"] == "ValueError"
    assert str(wired) not in json.dumps(record)  # sandbox path never leaks
    assert "<input>" in json.dumps(record)
    assert record["log_tail"][-1] == "INFO | line two"


def test_previous_hook_is_chained_not_suppressed(wired: Path) -> None:
    # Simulate a fresh install: the fixture's install already consumed the
    # remember-once slot, so clear the remembered sys hook first.
    handlers._STATE.pop("previous_sys_hook", None)
    calls: list[tuple] = []

    def previous_hook(exc_type, exc_value, exc_tb) -> None:
        calls.append((exc_type, exc_value, exc_tb))

    sys.excepthook = previous_hook
    handlers.install(root=wired)
    assert sys.excepthook is not previous_hook
    assert handlers._STATE.get("previous_sys_hook") is previous_hook
    try:
        raise KeyError("chained")
    except KeyError:
        sys.excepthook(*sys.exc_info())
    assert len(calls) == 1  # the previous hook still ran


def test_threading_hook_captures_with_thread_name(wired: Path) -> None:
    """The threading hook is invoked DIRECTLY with a synthesized ExceptHookArgs:
    pytest's threadexception plugin owns threading.excepthook inside the suite,
    so spawning a real thread here would test pytest's wrapper, not ours."""
    consent.write_consent(wired, True)
    try:
        raise RuntimeError("thread boom")
    except RuntimeError:
        args = threading.ExceptHookArgs(
            (RuntimeError, RuntimeError("thread boom"), sys.exc_info()[2],
             threading.Thread(target=lambda: None, name="oracle-worker")),
        )
    handlers._threading_excepthook(args)
    records = crash.list_records(wired)
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["thread"] == "oracle-worker"
    assert record["exception"]["message"] == "thread boom"


def test_failing_handler_writes_nothing_and_stays_silent(wired: Path, monkeypatch) -> None:
    consent.write_consent(wired, True)
    monkeypatch.setattr(handlers, "build_record", lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    try:
        raise RuntimeError("trigger")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())  # must not raise
    assert crash.list_records(wired) == []


def test_mid_session_revocation_suppresses_the_next_event(wired: Path) -> None:
    consent.write_consent(wired, True)
    try:
        raise RuntimeError("captured")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    assert len(crash.list_records(wired)) == 1
    consent.write_consent(wired, False)  # revoke without reinstall
    try:
        raise RuntimeError("not captured")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    assert len(crash.list_records(wired)) == 1  # unchanged


def test_faulthandler_arms_only_with_consent(wired: Path) -> None:
    assert handlers.enable_faulthandler_catch(wired) is False  # consent off
    consent.write_consent(wired, True)
    assert handlers.enable_faulthandler_catch(wired) is True
    handlers.disable_faulthandler_catch()


def test_record_schema_shape(wired: Path) -> None:
    consent.write_consent(wired, True)
    try:
        raise OSError("schema")
    except OSError:
        sys.excepthook(*sys.exc_info())
    record = json.loads(crash.list_records(wired)[0].read_text(encoding="utf-8"))
    for key in ("record_version", "timestamp", "app_version", "platform", "exception", "traceback", "log_tail", "edition"):
        assert key in record, key
    assert record["platform"]["os"] == sys.platform
    assert record["exception"]["message"] == "schema"


def test_cap_enforced_by_bundled_writes(wired: Path) -> None:
    consent.write_consent(wired, True)
    for i in range(bundle.MAX_RECORDS + 3):
        bundle.write_record(wired, {"exception": {"type": "T", "message": str(i)}, "log_tail": []})
    assert len(bundle.list_records(wired)) == bundle.MAX_RECORDS
