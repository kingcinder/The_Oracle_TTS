"""Shared pytest fixtures for The Oracle's test suite.

Two jobs live here. :func:`_restore_oracle_env` is an autouse fixture that makes
``ORACLE_*`` environment variables a per-test sandbox: without it, a test that (or
whose code under test) writes ``ORACLE_AUDIOCPP_MODEL`` / ``ORACLE_AUDIOCPP_CLI``
straight into ``os.environ`` silently changes the state every later test sees —
the classic cross-file pollution that made ``test_vulkan_backend.py`` fail only
when suites ran together.

The other is the skip audit at the bottom of this file: every skip is reported,
and a job that provisions its own requirements can demand there be none (see
``tests/skip_audit.py`` for why a silent skip is a coverage hole rather than a
pass).
"""

from __future__ import annotations

import os

import pytest

from tests import skip_audit

_ORACLE_ENV_PREFIX = "ORACLE_"

#: Every skip reported this session, in the order pytest reports them.
_SKIPS: list[skip_audit.SkipRecord] = []


@pytest.fixture(autouse=True)
def _restore_oracle_env():
    """Snapshot and restore every ``ORACLE_*`` environment variable per test.

    ``monkeypatch`` alone cannot guarantee a clean session env: it only
    restores the keys a test *explicitly* patched, whereas the code under test
    (``vulkan_setup.run_vulkan_setup``, the GUI's Vulkan handlers) writes
    ``ORACLE_AUDIOCPP_MODEL`` / ``ORACLE_AUDIOCPP_CLI`` directly into
    ``os.environ`` — and tests exercise that code. This fixture therefore
    snapshots the whole ``ORACLE_*`` set before every test and restores it in
    teardown: values changed are put back, keys deleted are re-added, and keys
    created during the test are removed. Tests can never leak session state
    (e.g. a downloaded model path) into a later test or suite.
    """
    saved = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_ORACLE_ENV_PREFIX)
    }
    yield
    # Walk the union of saved and current keys: a key created by the test is
    # popped, a changed key is put back, and a pre-existing key that the test
    # *deleted* (absent from the current env, so a walk of os.environ alone
    # would never see it) is re-added.
    for key in set(saved) | {key for key in os.environ if key.startswith(_ORACLE_ENV_PREFIX)}:
        if key in saved:
            if os.environ.get(key) != saved[key]:
                os.environ[key] = saved[key]
        else:
            os.environ.pop(key, None)


# --- skip audit -----------------------------------------------------------------


def pytest_sessionstart(session) -> None:
    # A fresh session (the suite can run in-process, e.g. under a wrapper that
    # calls pytest.main twice) must not inherit the previous run's skips.
    _SKIPS.clear()


def _record_skip(nodeid: str, longrepr: object) -> None:
    _SKIPS.append(skip_audit.SkipRecord(nodeid=nodeid, reason=skip_audit.skip_reason(longrepr)))


def pytest_runtest_logreport(report) -> None:
    if getattr(report, "skipped", False):
        _record_skip(report.nodeid, getattr(report, "longrepr", None))


def pytest_collectreport(report) -> None:
    # A module skipped at collection time never reports per test, so without this
    # a whole file could opt out of the suite and the audit would not mention it.
    if getattr(report, "skipped", False):
        _record_skip(report.nodeid, getattr(report, "longrepr", None))


def pytest_terminal_summary(terminalreporter) -> None:
    for line in skip_audit.format_audit(_SKIPS):
        terminalreporter.write_line(line)
    if skip_audit.fail_on_skip_enabled() and _SKIPS:
        terminalreporter.write_line("")
        for line in skip_audit.format_failure(_SKIPS):
            terminalreporter.write_line(line)


def pytest_sessionfinish(session, exitstatus) -> None:
    # Reported above; here the verdict is enforced. A run that demanded no skips
    # must not exit 0 having skipped tests, or the demand means nothing.
    if skip_audit.fail_on_skip_enabled() and _SKIPS:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
