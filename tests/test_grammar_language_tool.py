"""Tests for the bounded LanguageTool load: a slow or missing first-use download
must fail fast and fall back to the local fixes instead of blocking a render
for the duration of a hundreds-of-MB download."""

from __future__ import annotations

import os
import threading
import time

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


def _assume_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the download probe so the bounded-load paths are exercised
    deterministically regardless of the machine's real cache state."""
    monkeypatch.setattr(grammar, "_language_tool_download_ready", lambda: True)


def test_fast_language_tool_load_returns_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A LanguageTool that constructs quickly is used (no behavior change for
    machines with a cached or fast-to-fetch tool)."""
    import language_tool_python

    _assume_cached(monkeypatch)
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

    _assume_cached(monkeypatch)
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

    _assume_cached(monkeypatch)
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


def test_not_cached_skips_download_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine without the (hundreds-of-MB) LanguageTool download must not
    stall the render waiting for it: it falls back immediately and hands the
    download off to a detached helper process for a later run."""
    import tempfile
    import language_tool_python

    fake_cache = tempfile.mkdtemp(prefix="oracle_lt_cache_")
    calls: list[str] = []
    monkeypatch.setattr(
        language_tool_python,
        "LanguageTool",
        lambda _lang: calls.append("LanguageTool") or _fake_tool(),
    )
    monkeypatch.setattr(grammar, "_language_tool_download_ready", lambda: False)
    monkeypatch.setattr(
        "language_tool_python.download_lt.get_language_tool_download_path",
        lambda: fake_cache,
    )

    spawned: list[list[str]] = []
    monkeypatch.setattr(
        grammar.subprocess,
        "Popen",
        lambda cmd, **kwargs: spawned.append(cmd) or object(),
    )

    start = time.monotonic()
    corrector = GrammarCorrector()
    elapsed = time.monotonic() - start

    assert corrector._tool is None
    assert grammar._LANGUAGE_TOOL_ABANDONED.is_set()
    assert elapsed < 1.0, "render must not wait on the missing download"
    assert calls == [], "download must run in the helper process, not inline"
    assert spawned and "download_lt" in " ".join(spawned[0])


def test_warm_skips_when_download_already_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rapid renders must not stack duplicate warm downloads: once a helper
    claims the lock, later warm-ups return without spawning another one."""
    import os
    import tempfile

    fake_cache = tempfile.mkdtemp(prefix="oracle_lt_cache_")
    monkeypatch.setattr(
        grammar,
        "_language_tool_download_ready",
        lambda: False,
    )

    spawned: list[list[str]] = []
    monkeypatch.setattr(
        grammar.subprocess,
        "Popen",
        lambda cmd, **kwargs: spawned.append(cmd) or object(),
    )
    monkeypatch.setattr(
        "language_tool_python.download_lt.get_language_tool_download_path",
        lambda: fake_cache,
    )

    grammar._warm_language_tool_download()
    grammar._warm_language_tool_download()

    assert len(spawned) == 1, "second warm must see the lock and skip"
    # The helper is responsible for removing the lock when it exits.
    assert os.path.exists(os.path.join(fake_cache, ".oracle_lt_warm.lock"))


def test_download_ready_accepts_matching_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe is True when the exact version the library needs is cached."""
    import language_tool_python.download_lt as lt_download

    def fake_find(folder: str) -> list[str]:
        return [os.path.join(folder, "LanguageTool-6.7-SNAPSHOT")]

    monkeypatch.setattr(lt_download, "find_existing_language_tool_downloads", fake_find)
    monkeypatch.setattr(lt_download, "get_language_tool_download_path", lambda: "/fake/cache")

    assert grammar._language_tool_download_ready() is True


def test_download_ready_rejects_version_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale/mismatched cache directory (e.g. a leftover snapshot from a
    different library version) must be treated as not-ready -- that is exactly
    the case that otherwise re-downloads hundreds of MB on every render."""
    import language_tool_python.download_lt as lt_download

    def fake_find(folder: str) -> list[str]:
        return [os.path.join(folder, "LanguageTool-6.9-SNAPSHOT")]

    monkeypatch.setattr(lt_download, "find_existing_language_tool_downloads", fake_find)
    monkeypatch.setattr(lt_download, "get_language_tool_download_path", lambda: "/fake/cache")

    assert grammar._language_tool_download_ready() is False


def test_download_ready_probe_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the probe itself errors it returns True (assume cached) so the bounded
    load attempt -- the real safety net -- is never skipped by our own code."""
    import language_tool_python.download_lt as lt_download

    def broken_find(folder: str) -> list[str]:
        raise RuntimeError("cache unreadable")

    monkeypatch.setattr(lt_download, "find_existing_language_tool_downloads", broken_find)
    monkeypatch.setattr(lt_download, "get_language_tool_download_path", lambda: "/fake/cache")

    assert grammar._language_tool_download_ready() is True
