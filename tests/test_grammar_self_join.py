"""Regression: LanguageTool's destructor must never join its own consumer thread.

The intermittent ``RuntimeError: cannot join current thread`` at GUI-suite
teardown was a GC-triggered destructor self-join, captured verbatim with an
instrumented ``threading.Thread.join`` across a full-suite run (2026-09-20):

    threading.py run() epilogue (``del self._target``)
      -> language_tool_python/server.py ``__del__`` -> ``close()``
      -> ``_terminate_server()`` -> ``_consumer_thread.join(timeout=5)``

language_tool_python's stdout consumer thread targets a lambda closing over
the LanguageTool object, so when the thread finishes, dropping its target
can free the LAST reference to the tool — running ``__del__`` on the
consumer thread itself, which then joins that very thread. Intermittent
because it fires only when the lambda is the last surviving reference at
thread exit.

The fix (``grammar._guarded_language_tool``) defuses exactly that: the
wrapped ``__del__`` clears the ``_consumer_thread`` handle when it runs on
the consumer thread, so the destructor skips ONLY the self-join (the stop
event and server kill still run) before delegating to the library's
destructor.

No test here constructs a real LanguageTool (that starts a Java server):
destructor semantics are exercised on ``cls.__new__(cls)`` instances with
the attributes ``__del__`` touches seeded by hand — the same shape GC sees.
"""

from __future__ import annotations

import threading

import pytest

from the_oracle.text_repair import grammar


def _guarded(raw: type) -> type:
    return grammar._guarded_language_tool(raw)


class _StubServer:
    """The slice of Popen that close()/_terminate_server actually touch.

    ``poll() -> None`` keeps ``_server_is_alive()`` True so the destructor
    walks the kill branch; ``pid`` points at a practically nonexistent PID
    so psutil hits its own caught NoSuchProcess path; the streams are None
    so the finally block skips closing them; terminate/communicate record
    that the server kill still ran after the join was skipped.
    """

    pid = 2**31
    stdin = None
    stdout = None
    stderr = None

    def __init__(self) -> None:
        self.terminated = False

    def poll(self):  # noqa: ANN202 - alive while None, like Popen
        return None

    def terminate(self) -> None:
        self.terminated = True

    def communicate(self, timeout=None):  # noqa: ANN001, ANN202
        return (None, b"")


def _bare(cls, *, consumer: threading.Thread | None) -> object:
    """A cls instance shaped exactly as __del__/close()/_terminate_server
    see it.

    ``object.__new__`` skips __init__ (and therefore the Java server). The
    seeded attributes mirror the library's runtime state: a consumer thread,
    a stop event, a live-looking server (so ``_server_is_alive()`` is True
    and the destructor walks the real kill branch), and the spellings state
    close() reads.
    """
    tool = cls.__new__(cls)
    tool._consumer_thread = consumer
    tool._stop_consume_event = threading.Event()
    tool._server = _StubServer()
    tool._new_spellings_persist = True  # close() skips the spellings branch
    tool._new_spellings = []
    return tool


# --- the regression core: raw raises, guarded does not ------------------------


def test_raw_library_destructor_self_joins() -> None:
    """Pin the BUG on the raw library class: __del__ on the consumer thread
    raises ``cannot join current thread``. This is the exact captured
    failure; it documents why the guard exists and fails loudly if the
    library ever changes shape (new attribute names would break _bare)."""
    language_tool_python = pytest.importorskip("language_tool_python")
    raw = language_tool_python.LanguageTool

    outcome: list[str] = []

    def _drop_on_own_stack() -> None:
        tool = _bare(raw, consumer=threading.current_thread())
        try:
            tool.__del__()  # exactly what GC invokes
        except BaseException as exc:  # noqa: BLE001 - the assertion IS the type
            outcome.append(f"{type(exc).__name__}: {exc}")
            # Neutralize the booby trap: this object's eventual GC __del__
            # would otherwise re-raise as an unraisable warning later in the
            # session (an inert tool raises nothing).
            tool._server = None
            tool._consumer_thread = None
        else:
            outcome.append("no-raise")

    thread = threading.Thread(target=_drop_on_own_stack, daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert outcome and outcome[0].startswith("RuntimeError: cannot join current thread"), outcome


def test_guarded_destructor_does_not_self_join() -> None:
    """The guarded class defuses the captured failure: no raise, the
    consumer handle is cleared (the join is skipped), and the stop event is
    still set — the shutdown semantics below the join are untouched."""
    guarded_cls = _guarded(_raw_library_class())

    outcome: list[str] = []

    def _drop_on_own_stack() -> None:
        tool = _bare(guarded_cls, consumer=threading.current_thread())
        try:
            tool.__del__()
        except BaseException as exc:  # noqa: BLE001
            outcome.append(f"{type(exc).__name__}: {exc}")
        else:
            outcome.append("no-raise")

    thread = threading.Thread(target=_drop_on_own_stack, daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert outcome == ["no-raise"], outcome

    # Normal-path destructor (foreign thread, live-looking server): the
    # consumer handle survives, the stop event is set, and the server kill
    # still runs — only the self-join is defused.
    tool = _bare(guarded_cls, consumer=None)
    consumer = threading.Thread(target=lambda: None, daemon=True)
    consumer.start()
    consumer.join()
    tool._consumer_thread = consumer
    tool.__del__()  # from the MAIN thread
    assert tool._consumer_thread is consumer
    assert tool._stop_consume_event.is_set()
    assert tool._server is None, "kill branch must still run on the normal path"


def _raw_library_class() -> type:
    language_tool_python = pytest.importorskip("language_tool_python")
    return language_tool_python.LanguageTool


def test_guarded_close_from_another_thread_untouched() -> None:
    """An explicit close() from a foreign thread keeps library semantics:
    the server is only terminated when alive, and a dead-server close is a
    no-op that must not raise."""
    guarded_cls = _guarded(_raw_library_class())

    calls: list[str] = []

    class _Probe(guarded_cls):
        def _server_is_alive(self) -> bool:  # noqa: D102 - probe
            return False

        def _terminate_server(self) -> None:  # noqa: D102 - probe
            calls.append("terminate")

    probe = object.__new__(_Probe)
    probe._consumer_thread = None
    probe._new_spellings_persist = True
    probe._new_spellings = []
    _Probe.close(probe)
    assert calls == [], "close() must not terminate a dead server (library semantics)"


def test_defusal_is_identity_keyed_not_name_keyed() -> None:
    """The guard skips the join only when __del__ runs on the CONSUMER
    thread object itself — a different thread, even named identically,
    still joins normally."""
    guarded_cls = _guarded(_raw_library_class())

    tool = _bare(guarded_cls, consumer=None)
    consumer = threading.Thread(target=lambda: None, name="lt-consumer")
    consumer.start()
    consumer.join()
    tool._consumer_thread = consumer

    outcome: list[str] = []

    def _drop_from_impostor() -> None:
        # A DIFFERENT thread that merely looks like the consumer: the
        # identity check must not match, so the join runs — and joining an
        # already-finished thread from elsewhere is fine (no raise).
        try:
            tool.__del__()
        except BaseException as exc:  # noqa: BLE001
            outcome.append(f"{type(exc).__name__}: {exc}")
        else:
            outcome.append("no-raise")

    impostor = threading.Thread(target=_drop_from_impostor, name="lt-consumer", daemon=True)
    impostor.start()
    impostor.join(timeout=10)
    assert outcome == ["no-raise"], outcome
    assert tool._consumer_thread is consumer, "impostor close must keep the handle"
    assert tool._server is None, "impostor close still walks the kill branch"


# --- the production wiring ----------------------------------------------------


def test_guarded_class_subclasses_library_language_tool() -> None:
    """The guarded variant must subclass the live library class (the MRO
    keeps every library method real), without constructing a server."""
    language_tool_python = pytest.importorskip("language_tool_python")
    assert issubclass(
        grammar._guarded_language_tool(language_tool_python.LanguageTool),
        language_tool_python.LanguageTool,
    )


def test_guard_wraps_live_library_class_and_caches_per_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard resolves *base* at each use (so monkeypatched test classes
    are honored) and caches per base class, not globally."""
    pytest.importorskip("language_tool_python")

    class _FakeA:
        pass

    class _FakeB:
        pass

    guarded_a = grammar._guarded_language_tool(_FakeA)
    assert issubclass(guarded_a, _FakeA)
    assert grammar._guarded_language_tool(_FakeA) is guarded_a, "same base -> same wrapper"
    assert grammar._guarded_language_tool(_FakeB) is not guarded_a, "different base -> different wrapper"

    # A non-class double (plain callable) passes through unchanged.
    def _callable_double(_lang):  # noqa: ANN001
        return object()

    assert grammar._guarded_language_tool(_callable_double) is _callable_double


def test_production_builder_uses_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vacuity guard: the loader must route the LIVE library class through
    ``_guarded_language_tool`` — otherwise a refactor could silently
    un-land the fix."""
    pytest.importorskip("language_tool_python")
    import language_tool_python

    monkeypatch.setattr(grammar, "_LANGUAGE_TOOL_ABANDONED", threading.Event())
    monkeypatch.setattr(grammar, "_language_tool_download_ready", lambda: True)
    monkeypatch.delenv("LTP_JAR_DIR_PATH", raising=False)

    seen_bases: list[object] = []

    def _spy_guard(base):
        seen_bases.append(base)

        class _Stub:
            def __init__(self, language: str) -> None:
                assert language == "en-US"

        return _Stub

    monkeypatch.setattr(grammar, "_guarded_language_tool", _spy_guard)

    corrector = grammar.GrammarCorrector(use_language_tool=True)

    assert seen_bases == [language_tool_python.LanguageTool], (
        "the loader must pass the live library class through the guard"
    )
    assert corrector._tool is not None, "the guarded stub's tool must be kept"
