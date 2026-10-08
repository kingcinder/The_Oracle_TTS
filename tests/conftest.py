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

import ast
import inspect
import io
import os
import tokenize
from pathlib import Path

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


# --- standalone Qt app-fixture guard ----------------------------------------

#: Plain-text markers for constructing a QWidget (only QWidgets abort without
#: a QApplication) or one of the project's known window builders. QObject
#: types (QAction, QButtonGroup, QTableWidgetItem, ...) never abort, so they
#: deliberately do not appear here. QApplication( itself is the app and is
#: fine.
_QT_WIDGET_BUILD_MARKERS: tuple[str, ...] = (
    "QWidget(",
    "QMainWindow(",
    "QDialog(",
    "QLabel(",
    "QPushButton(",
    "QToolButton(",
    "QComboBox(",
    "QCheckBox(",
    "QRadioButton(",
    "QSlider(",
    "QSpinBox(",
    "QDoubleSpinBox(",
    "QLineEdit(",
    "QTextEdit(",
    "QPlainTextEdit(",
    "QTextBrowser(",
    "QTableWidget(",
    "QTreeWidget(",
    "QListWidget(",
    "QTabWidget(",
    "QGroupBox(",
    "QMenuBar(",
    "QMenu(",
    "QToolBar(",
    "QStatusBar(",
    "QSplitter(",
    "QScrollArea(",
    "QDockWidget(",
    "QFrame(",
    "QProgressBar(",
    "QStackedWidget(",
    "QHeaderView(",
    # Project builders: every tour wizard's class name ends in "Wizard", and
    # the shared test helper builds the real MainWindow.
    "Wizard(",
    "MainWindow(",
    "_build_window(",
)

#: Token categories whose text is string/comment content, not code: a test
#: may legitimately QUOTE a widget constructor when asserting on source text.
_STRINGY_TOKENS = {
    getattr(tokenize, name)
    for name in ("STRING", "COMMENT", "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
    if hasattr(tokenize, name)
}

#: Fixtures that guarantee a QApplication exists before the test body runs.
_QT_APP_FIXTURES = frozenset({"qt_app"})

#: Transitive scan bounds: helpers-of-helpers up to this depth, and at most
#: this many called names followed per function (sorted for determinism).
_MAX_HELPER_DEPTH = 2
_MAX_HELPER_CALLS = 15

_TESTS_DIR = Path(__file__).resolve().parent


def _called_function_names(tree: ast.AST) -> set[str]:
    """Names of functions/methods called anywhere in ``tree``.

    Bare calls (``helper()``) and attribute calls (``mod.helper()`` /
    ``obj.method()``) both contribute a candidate name; resolution later
    decides which of those actually point at a tests/-tree helper.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


def _tests_tree_function(obj: object) -> bool:
    """True when obj is a plain function defined under tests/ — a test
    helper or fixture. Production code is deliberately excluded: widgets
    built inside src/ are the code under test, not a hidden test build."""
    if not inspect.isfunction(obj):
        return False
    try:
        path = inspect.getsourcefile(obj)
    except TypeError:
        return False
    if not path:
        return False
    resolved = Path(path).resolve()
    return resolved.parent == _TESTS_DIR or _TESTS_DIR in resolved.parents


def widget_builds_reachable_from(func, extra_names=()) -> tuple[str, str] | None:
    """``(marker, origin-qualname)`` when ``func`` itself — or a tests/-tree
    helper it calls, or a fixture it requests via ``extra_names`` —
    constructs a Qt widget; else None.

    Breadth/depth bounded (depth 2, capped branch) so a pathological module
    cannot make collection quadratic. Never raises: an uninferable helper is
    skipped, so the worst case is a miss, never a false trip.
    """
    namespace = getattr(func, "__globals__", None) or {}
    frontier: list[tuple[str, object]] = [(getattr(func, "__qualname__", "<test>"), func)]
    for name in extra_names:
        if name in _QT_APP_FIXTURES:
            continue
        obj = namespace.get(name)
        if obj is not None:
            frontier.append((name, obj))
    seen: set[object] = set()
    depth = 0
    while frontier and depth <= _MAX_HELPER_DEPTH:
        following: list[tuple[str, object]] = []
        for name, fn in frontier:
            key = (getattr(fn, "__module__", None), getattr(fn, "__qualname__", None))
            if key in seen:
                continue
            seen.add(key)
            if not inspect.isfunction(fn):
                continue
            try:
                source = inspect.getsource(fn)
            except (OSError, TypeError):
                continue
            marker = widget_builds_in_source(source)
            if marker is not None:
                return marker, name
            if depth == _MAX_HELPER_DEPTH:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            fn_globals = getattr(fn, "__globals__", None) or namespace
            for callee in sorted(_called_function_names(tree))[:_MAX_HELPER_CALLS]:
                if callee in _QT_APP_FIXTURES:
                    continue
                obj = fn_globals.get(callee)
                if obj is not None and _tests_tree_function(obj):
                    following.append((callee, obj))
        frontier = following
        depth += 1
    return None


def widget_builds_in_source(source: str) -> str | None:
    """Return the first widget-builder marker present in ``source``, else None.

    String literals and comments are stripped via tokenize first: tests that
    scan GUI code and assert on quoted constructors must not trip the guard.
    A fast raw-substring pre-filter keeps the cost negligible for the (vast
    majority of) tests that mention no Qt names at all.
    """
    if not any(marker in source for marker in _QT_WIDGET_BUILD_MARKERS):
        return None
    try:
        code = "".join(
            tok.string
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type not in _STRINGY_TOKENS
        )
    except (tokenize.TokenError, IndentationError):
        code = source
    for marker in _QT_WIDGET_BUILD_MARKERS:
        if marker in code:
            return marker
    return None


@pytest.fixture(autouse=True)
def _qt_standalone_app_guard(request):
    """Fail loudly when a test builds Qt widgets but requests no app fixture.

    PySide6 aborts the whole process (SIGABRT) on the first widget
    constructed without a QApplication, so such a test never "fails" — it
    passes whenever an earlier file in the same session happened to create
    the app, and dumps core when run standalone. File order becomes a silent
    dependency (real case: ``test_help_stage_is_first...`` aborted rc=134
    standalone, passed in grouped runs). The guard follows the test's own
    calls and fixture requests into tests/-tree helpers before it runs and,
    unless it requests ``qt_app``, fails it with an actionable message
    naming the builder — deterministic in grouped AND standalone runs.
    """
    if not isinstance(request.node, pytest.Function):
        yield
        return
    if _QT_APP_FIXTURES & set(request.fixturenames):
        yield
        return
    # The node itself is not a Python function; ``obj`` is the real (possibly
    # decorated) callable whose source defines the test body.
    target = getattr(request.node, "obj", None)
    if target is None:
        yield
        return
    hit = widget_builds_reachable_from(
        target, [name for name in request.fixturenames if name not in _QT_APP_FIXTURES]
    )
    if hit is not None:
        marker, origin = hit
        via = (
            "the test body"
            if origin == getattr(target, "__qualname__", None)
            else f"helper/fixture `{origin}`"
        )
        pytest.fail(
            f"{request.node.name} constructs Qt widgets via {via} ({marker} ...) "
            "but requests no QApplication fixture: standalone this aborts the "
            "process (SIGABRT) and it only passes when another test file "
            "created the app first. Add the `qt_app` fixture to this test.",
            pytrace=False,
        )
    yield


# --- skip audit -----------------------------------------------------------------


def pytest_sessionstart(session) -> None:
    # A fresh session (the suite can run in-process, e.g. under a wrapper that
    # calls pytest.main twice) must not inherit the previous run's skips.
    _SKIPS.clear()


# --- window lifetime ----------------------------------------------------------


def destroy_leftover_main_windows(app=None) -> None:
    """Stop every live MainWindow's pending startup timers and destroy it.

    PySide6 holds connected callables — the lambdas ``_build_menu`` wires to
    ``triggered`` and the bound methods cached on ``SignalInstance``
    attributes — in C-level connection structures CPython's garbage collector
    cannot trace. A MainWindow a test builds and simply drops is therefore
    pinned alive forever: its closure cells and bound methods stay reachable
    only through those invisible roots. Enough of them accumulate and any
    later ``processEvents()`` replays every deferred ``_on_gui_shown`` at
    once — a full application-wide stylesheet restyle per window — which
    hangs the suite.

    Explicit ``deleteLater()`` is the cut: destroying the C++ window
    destroys its actions, signals and connections, releasing every callable
    they hold so the Python wrapper's reference count can finally reach
    zero. The pending startup timers are stopped first so the teardown never
    fires ``_on_gui_shown`` (whose crash-flow / wizard side effects must not
    run implicitly from a test ending).
    """
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    app = app or QApplication.instance()
    if app is None:
        return
    from the_oracle.app_gui import MainWindow
    from the_oracle.inference_wizard import InferenceSetupWizard

    leftovers = [
        w
        for w in app.topLevelWidgets()
        if isinstance(w, (MainWindow, InferenceSetupWizard))
    ]
    if not leftovers:
        return
    for window in leftovers:
        for attr in (
            "_gui_shown_timer",
            "_crash_flow_timer",
            "_wizard_launch_timer",
        ):
            timer = getattr(window, attr, None)
            if timer is not None:
                timer.stop()
        window.close()
        window.deleteLater()
    # A bare processEvents() does not deliver DeferredDelete events posted
    # while outside an event loop — the windows would survive it and stay
    # pinned. The explicit typed flush is what actually destroys them.
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


@pytest.fixture(autouse=True)
def _destroy_leftover_main_windows():
    """Run :func:`destroy_leftover_main_windows` after every test.

    Tests construct MainWindows freely and usually just drop them; without
    this, each one survives for the rest of the session (see the helper's
    docstring) and the first test that processes events replays all of their
    deferred startup work at once.
    """
    yield
    destroy_leftover_main_windows()


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
