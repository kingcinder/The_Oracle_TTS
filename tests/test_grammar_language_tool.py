"""Tests for the bounded LanguageTool load: a slow first-use download must fail
fast and fall back to the local fixes instead of blocking a render for the
duration of a hundreds-of-MB download."""

from __future__ import annotations

import threading

import pytest

from the_oracle.text_repair import grammar
from the_oracle.text_repair.grammar import GrammarCorrector


@pytest.fixture(autouse=True)
def _isolate_language_tool_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh abandoned-flag and timeout per test so the process-level fallback
    never leaks between tests (and the default 25s timeout is never hit)."""
    monkeypatch.setattr(grammar, "_LANGUAGE_TOOL_ABANDONED", threading.Event())
    monkeypatch.setattr(grammar, "LANGUAGE_TOOL_LOAD_TIMEOUT_SECONDS", 25.0)


def _fake_tool() -> object:
    return object()


def test_fast_language_tool_load_returns_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A LanguageTool that constructs quickly is used (no behavior change for
    machines with a cached or fast-to-fetch tool)."""
    import language_tool_python

    fake = _fake_tool()
    monkeypatch.setattr(language_tool_python, "LanguageTool", lambda _lang: fake)

    corrector = GrammarCorrector()

    assert corrector._tool is fake
    assert not grammar._LANGUAGE_TOOL_ABANDONED.is_set()


def test_slow_language_tool_load_falls_back_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A LanguageTool whose construction blocks (first-use download) must not
    hold up the render: after the bounded timeout the corrector falls back to
    the local fixes and the process remembers not to retry."""
    import language_tool_python

    never = threading.Event()
    monkeypatch.setattr(
        language_tool_python,
        "LanguageTool",
        lambda _lang: never.wait(3600) or _fake_tool(),
    )
    monkeypatch.setattr(grammar, "LANGUAGE_TOOL_LOAD_TIMEOUT_SECONDS", 0.05)

    corrector = GrammarCorrector()

    assert corrector._tool is None
    assert grammar._LANGUAGE_TOOL_ABANDONED.is_set()


def test_abandoned_process_short_circuits_further_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once a load timed out this process, later correctors don't spawn another
    racing download thread -- they fall back immediately."""
    import language_tool_python

    calls: list[str] = []
    never = threading.Event()

    def blocking_load(_lang):
        calls.append("LanguageTool")
        never.wait(3600)
        return _fake_tool()

    monkeypatch.setattr(language_tool_python, "LanguageTool", blocking_load)
    monkeypatch.setattr(grammar, "LANGUAGE_TOOL_LOAD_TIMEOUT_SECONDS", 0.05)

    first = GrammarCorrector()
    assert first._tool is None
    assert calls == ["LanguageTool"]

    second = GrammarCorrector()
    assert second._tool is None
    assert calls == ["LanguageTool"], "no second download attempt this process"


def test_local_fallback_fixes_common_mistakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without LanguageTool the COMMON_FIXES fallback still repairs the most
    common typos (the designed degraded path). The abandoned flag is pre-set so
    this test never triggers a real LanguageTool load or download."""
    grammar._LANGUAGE_TOOL_ABANDONED.set()
    corrector = GrammarCorrector()

    corrected = corrector.correct("  i  dont  want  that ")

    assert corrected == "I don't want that"


def test_timeout_constant_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timeout is tunable via ORACLE_LANGUAGE_TOOL_TIMEOUT (for CI) and
    falls back to the 25s default on garbage input."""
    import importlib

    monkeypatch.setenv("ORACLE_LANGUAGE_TOOL_TIMEOUT", "7")
    importlib.reload(grammar)
    assert grammar.LANGUAGE_TOOL_LOAD_TIMEOUT_SECONDS == 7.0

    monkeypatch.setenv("ORACLE_LANGUAGE_TOOL_TIMEOUT", "not-a-number")
    importlib.reload(grammar)
    assert grammar.LANGUAGE_TOOL_LOAD_TIMEOUT_SECONDS == 25.0
