"""Window-lifetime regression: no MainWindow may outlive its test.

PySide6 holds connected callables (lambdas in ``triggered.connect(...)``,
bound methods cached on ``SignalInstance`` attributes) in C-level connection
structures that CPython's garbage collector cannot trace. A MainWindow built
by a test and simply dropped therefore stays pinned alive: its cells and
bound methods are reachable only through those C-level roots, so every
construction leaked one full window for the rest of the process. When a later
test ran ``processEvents()``, every pinned window's deferred ``_on_gui_shown``
fired at once — each running a full ``QApplication.setStyleSheet`` over the
accumulating widget set — hanging the suite (faulthandler showed
``gui_themes.apply_theme`` ← ``app_gui._on_gui_shown`` ← the triggering test)
and arming wizard popups on stale windows.

The fix has two halves, both pinned here:
1. ``tests.conftest.destroy_leftover_main_windows`` runs after every test —
   it stops the window's pending startup timers and ``deleteLater()``s the
   window, destroying its C++ side so every connection (and the callables it
   holds) is released.
2. The deferred startup callbacks are armed from single-shot child timers of
   the window instead of the context-less ``QTimer.singleShot`` helper, so a
   window destroyed before its timer fired cancels the pending call instead
   of being pinned by it (and no wizard can launch from a dead window).
"""

import gc
import os
import weakref

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from tests.conftest import destroy_leftover_main_windows
from tests.test_app_gui_profiles import _build_window


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_teardown_destroys_the_window_so_it_cannot_pin(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    ref = weakref.ref(window)
    destroy_leftover_main_windows(qt_app)
    del window
    gc.collect()
    assert ref() is None, (
        "the teardown helper left the MainWindow pinned alive "
        "(PySide connections still hold callables that reference it)"
    )


def test_wizard_launch_timer_does_not_outlive_the_window(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from the_oracle.inference_wizard import InferenceSetupWizard

    window, _paths = _build_window(monkeypatch, tmp_path)
    # Run the queued _on_gui_shown: it arms the crash flow and the 250 ms
    # wizard launch. The window stays hidden, as in the tests that used to
    # leak.
    qt_app.processEvents()
    ref = weakref.ref(window)
    destroy_leftover_main_windows(qt_app)
    del window
    gc.collect()
    assert ref() is None, (
        "the pending wizard-launch timer pinned the MainWindow after "
        "teardown"
    )
    # Let the 250 ms deadline pass: with the window gone the timer must be
    # gone with it — no wizard may spawn from a destroyed window.
    QTest.qWait(400)
    wizards = [
        w
        for w in qt_app.topLevelWidgets()
        if isinstance(w, InferenceSetupWizard)
    ]
    assert not wizards, "wizard launched after its owning window was destroyed"
