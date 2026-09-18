"""Tests for the skip audit: the reporting, and the strict verdict it enforces.

The hooks are driven directly with fake reports rather than by running pytest in a
subprocess, so what is pinned is the behaviour that matters — a skip is recorded,
always reported, and turns the run red when the job demanded that nothing be
skipped — on any machine, in milliseconds.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests import conftest, skip_audit


@pytest.fixture(autouse=True)
def _clean_audit():
    """Each test starts and ends with no recorded skips."""
    conftest.pytest_sessionstart(SimpleNamespace())
    yield
    conftest.pytest_sessionstart(SimpleNamespace())


def _skip_report(nodeid: str, reason: str = "no GPU here"):
    return SimpleNamespace(nodeid=nodeid, skipped=True, longrepr=("tests/fake.py", 12, reason))


def _pass_report(nodeid: str):
    return SimpleNamespace(nodeid=nodeid, skipped=False, longrepr=None)


class _TerminalReporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        self.lines.append(line)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


# --- the logic ------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_fail_on_skip_reads_truthy_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv(skip_audit.FAIL_ON_SKIP_ENV, value)

    assert skip_audit.fail_on_skip_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_fail_on_skip_is_off_for_anything_else(monkeypatch, value: str) -> None:
    monkeypatch.setenv(skip_audit.FAIL_ON_SKIP_ENV, value)

    assert skip_audit.fail_on_skip_enabled() is False


def test_fail_on_skip_is_off_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(skip_audit.FAIL_ON_SKIP_ENV, raising=False)

    assert skip_audit.fail_on_skip_enabled() is False


def test_skip_reason_reads_pytests_three_tuple() -> None:
    assert skip_audit.skip_reason(("tests/x.py", 3, " needs a GPU ")) == "needs a GPU"


def test_skip_reason_falls_back_to_the_longrepr_text() -> None:
    assert skip_audit.skip_reason("Skipped: not on Windows") == "Skipped: not on Windows"


def test_skip_reason_never_leaves_an_empty_explanation() -> None:
    assert skip_audit.skip_reason(("tests/x.py", 3, "   ")) == skip_audit.NO_REASON
    assert skip_audit.skip_reason(None) == skip_audit.NO_REASON


def test_audit_reports_the_test_and_its_reason() -> None:
    lines = skip_audit.format_audit([skip_audit.SkipRecord("tests/x.py::test_a", "no GPU here")])

    assert lines[0] == "1 test(s) did not run here:"
    assert "tests/x.py::test_a" in lines[1]
    assert "no GPU here" in lines[2]


def test_audit_reports_nothing_when_nothing_was_skipped() -> None:
    assert skip_audit.format_audit([]) == []


def test_audit_lists_a_module_skip_once() -> None:
    records = [skip_audit.SkipRecord("tests/x.py", "module needs Windows")] * 3

    lines = skip_audit.format_audit(records)

    assert lines[0] == "1 test(s) did not run here:"
    assert sum("tests/x.py" in line for line in lines) == 1


def test_failure_names_the_skipped_tests_and_both_ways_out() -> None:
    lines = skip_audit.format_failure([skip_audit.SkipRecord("tests/x.py::test_a", "no GPU here")])
    text = "\n".join(lines)

    assert skip_audit.FAIL_ON_SKIP_ENV in text
    assert "tests/x.py::test_a" in text
    assert "no GPU here" in text
    # The message has to tell the reader what to do, not just that it failed.
    assert "provision" in text
    assert f"drop {skip_audit.FAIL_ON_SKIP_ENV}" in text


# --- the conftest wiring --------------------------------------------------------


def test_a_skipped_test_is_recorded_and_reported() -> None:
    conftest.pytest_runtest_logreport(_skip_report("tests/x.py::test_a", "missing binary"))
    reporter = _TerminalReporter()

    conftest.pytest_terminal_summary(reporter)

    assert "tests/x.py::test_a" in reporter.text
    assert "missing binary" in reporter.text


def test_a_passing_test_is_not_recorded() -> None:
    conftest.pytest_runtest_logreport(_pass_report("tests/x.py::test_a"))
    reporter = _TerminalReporter()

    conftest.pytest_terminal_summary(reporter)

    assert reporter.text == ""


def test_a_module_skipped_at_collection_is_recorded() -> None:
    conftest.pytest_collectreport(_skip_report("tests/test_windows_only.py", "POSIX shell needed"))
    reporter = _TerminalReporter()

    conftest.pytest_terminal_summary(reporter)

    assert "tests/test_windows_only.py" in reporter.text
    assert "POSIX shell needed" in reporter.text


def test_a_skip_does_not_fail_the_session_by_default(monkeypatch) -> None:
    monkeypatch.delenv(skip_audit.FAIL_ON_SKIP_ENV, raising=False)
    conftest.pytest_runtest_logreport(_skip_report("tests/x.py::test_a"))
    session = SimpleNamespace(exitstatus=0)

    conftest.pytest_sessionfinish(session, 0)

    assert session.exitstatus == 0


def test_a_skip_fails_the_session_when_none_are_allowed(monkeypatch, capsys) -> None:
    monkeypatch.setenv(skip_audit.FAIL_ON_SKIP_ENV, "1")
    conftest.pytest_runtest_logreport(_skip_report("tests/x.py::test_a", "audiocpp_cli is not built"))
    session = SimpleNamespace(exitstatus=0)
    reporter = _TerminalReporter()

    conftest.pytest_terminal_summary(reporter)
    conftest.pytest_sessionfinish(session, 0)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert "audiocpp_cli is not built" in reporter.text


def test_a_strict_run_with_no_skips_still_passes(monkeypatch) -> None:
    monkeypatch.setenv(skip_audit.FAIL_ON_SKIP_ENV, "1")
    conftest.pytest_runtest_logreport(_pass_report("tests/x.py::test_a"))
    session = SimpleNamespace(exitstatus=0)

    conftest.pytest_sessionfinish(session, 0)

    assert session.exitstatus == 0


def test_the_same_test_skipping_twice_is_reported_once(monkeypatch) -> None:
    monkeypatch.delenv(skip_audit.FAIL_ON_SKIP_ENV, raising=False)
    conftest.pytest_runtest_logreport(_skip_report("tests/x.py::test_a", "first"))
    conftest.pytest_runtest_logreport(_skip_report("tests/x.py::test_a", "second"))
    reporter = _TerminalReporter()

    conftest.pytest_terminal_summary(reporter)

    # One test, counted once, and the first explanation is the one shown.
    assert "1 test(s) did not run here:" in reporter.text
    assert reporter.text.count("tests/x.py::test_a") == 1
    assert "first" in reporter.text
    assert "second" not in reporter.text
