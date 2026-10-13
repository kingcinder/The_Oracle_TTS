"""Lifetime contract for Ctrl+hover help (the 1.4.0 certifier fault).

The 1.4.0 CI certifier faulted inside ``CtrlHoverHelp.eventFilter`` ->
``event.type()`` on the *second* window build, after an earlier window had
been torn down and deleted. These tests pin the lifecycle mitigation: each
window owns its helper, and no product filter is installed on QApplication.
The original native fault remains unconfirmed on Linux.

These tests pin the mitigation, not the crash (a segfault cannot be asserted —
it takes the test process with it):

* no event filter is ever installed on the QApplication,
* the filter rides the window and each tooltipped widget in its subtree,
* tearing a window down leaves nothing installed and nothing registered,
* each window owns a distinct filter instance,
* ten consecutive real MainWindow build/destroy cycles leave zero app-level
  filters and zero leftover per-widget filters.

``scripts/gui_window_churn.py`` is the out-of-band version of the last one
(50 passes, faulthandler armed) for the certifier artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QWidget

from the_oracle.gui_tooltips import CtrlHoverHelp, install_ctrl_hover_help


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def app_install_spy(qt_app, monkeypatch):
    """Record every Python-level event-filter install on the QApplication."""
    installed: list[QObject] = []
    original = QApplication.installEventFilter

    def spy(self, obj):  # noqa: ANN001 - Qt signature
        installed.append(obj)
        return original(self, obj)

    monkeypatch.setattr(QApplication, "installEventFilter", spy)
    return installed


def _release_singleton(qt_app) -> None:
    """Clear a detached test helper between cases."""
    helper = install_ctrl_hover_help(qt_app)
    if helper is not None:
        helper.release()


def test_install_never_filters_at_application_scope(qt_app, app_install_spy):
    """The app-level install is the bug: it must not happen for any target."""
    assert install_ctrl_hover_help(qt_app) is not None
    assert app_install_spy == []

    window = QWidget()
    try:
        helper = install_ctrl_hover_help(window)
        assert isinstance(helper, CtrlHoverHelp)
        assert app_install_spy == []
        assert helper.window is window
    finally:
        _release_singleton(qt_app)
        window.close()
        window.deleteLater()


def test_window_target_binds_the_filter_and_registrations_survive(qt_app):
    window = QWidget()
    button = QPushButton("Analyze", window)
    try:
        helper = install_ctrl_hover_help(window)
        helper.register(button, "Analyzes the input script.")
        assert helper.window is window
        assert helper.parent() is window, "Qt must own the filter for exactly this window's lifetime"
        assert helper.description_for(button) == "Analyzes the input script."
    finally:
        _release_singleton(qt_app)
        window.close()
        window.deleteLater()


def test_release_detaches_everything_it_installed(qt_app):
    window = QWidget()
    button = QPushButton("Render", window)
    button.setToolTip("native text")
    field = QLineEdit(window)
    field.setToolTip("another native text")
    removed: list[QObject] = []
    original = QWidget.removeEventFilter

    def spy(self, obj):  # noqa: ANN001 - Qt signature
        removed.append(obj)
        return original(self, obj)

    try:
        helper = install_ctrl_hover_help(window)
        # Tooltip owners are filtered individually; empty-tooltip widgets are not.
        assert button in helper._swept
        assert field in helper._swept
        assert window not in helper._swept

        import unittest.mock as _mock

        with _mock.patch.object(QWidget, "removeEventFilter", spy):
            helper.release()
        assert removed, "release() must detach every filter it installed"
        assert helper.window is None
        assert helper._swept == set()
        assert helper._descriptions == {}
        assert helper._action_descriptions == {}
        assert not helper._timer.isActive()
    finally:
        _release_singleton(qt_app)
        window.close()
        window.deleteLater()


def test_destroyed_window_detaches_the_filter(qt_app):
    window = QWidget()
    helper = install_ctrl_hover_help(window)
    helper.register(QPushButton("x", window), "x")
    try:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QApplication.processEvents()
        assert helper.window is None
        assert helper._descriptions == {}
    finally:
        _release_singleton(qt_app)


def test_second_window_gets_its_own_window_parented_filter(qt_app):
    """The certifier's exact shape: window A torn down, window B built."""
    first, second = QWidget(), QWidget()
    try:
        first_helper = install_ctrl_hover_help(first)
        assert first_helper is not None
        assert first_helper.parent() is first
        first_helper.register(QPushButton("a", first), "from the first window")

        second_helper = install_ctrl_hover_help(second)
        assert second_helper is not None
        assert second_helper is not first_helper
        assert second_helper.parent() is second
        assert second_helper.window is second
        assert not any(text == "from the first window" for text in second_helper._descriptions.values())
    finally:
        _release_singleton(qt_app)
        for window in (first, second):
            window.close()
            window.deleteLater()


def test_tooltip_owner_appearing_later_is_swept(qt_app):
    """Dialogs and lazy controls are built long after the window: polished
    children must still get the swallow filter."""
    window = QWidget()
    try:
        helper = install_ctrl_hover_help(window)
        late = QPushButton("Download", window)
        late.setToolTip("Downloads the Vulkan model.")
        assert late not in helper._swept
        # Qt polishes a child when it is first shown/built; the window filter
        # sees that as a QChildEvent.
        from PySide6.QtCore import QChildEvent

        helper.eventFilter(window, QChildEvent(QEvent.Type.ChildPolished, late))
        assert late in helper._swept
    finally:
        _release_singleton(qt_app)
        window.close()
        window.deleteLater()


def test_real_main_window_churn_leaves_no_filters_behind(qt_app, app_install_spy, monkeypatch, tmp_path):
    """Ten real build/destroy cycles: no app-level filter, nothing leaked.

    The window is the real MainWindow with heavyweight collaborators faked
    (``tests/test_app_gui_profiles.py::_build_window``), which is the same
    construction the certifier drives.
    """
    from tests.test_app_gui_profiles import _build_window

    for index in range(10):
        window, _paths = _build_window(monkeypatch, tmp_path / f"pass{index}")
        try:
            helper = window._ctrl_help
            assert helper is not None
            assert helper.window is window
            # The strict gate covers the real window's tooltip owners.
            owners = [
                widget
                for widget in window.findChildren(QWidget)
                if widget.toolTip()
            ]
            assert owners, "the real window is expected to carry tooltips"
            missing = [w for w in owners if w not in helper._swept]
            assert missing == [], "every tooltip owner must be swallowed"
        finally:
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            QApplication.processEvents()
        assert app_install_spy == [], "MainWindow must never filter the QApplication"
    _release_singleton(qt_app)
