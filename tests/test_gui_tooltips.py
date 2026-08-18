"""Tests for Ctrl+hover help tooltips (the_oracle.gui_tooltips.py).

The GUI is involved, so these follow the repo convention of slow-marking
Qt-backed tests; the poller itself is exercised through injected test seams
(no real cursor or QToolTip calls).
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QAction, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolTip,
)

from the_oracle.gui_tooltips import CtrlHoverHelp, install_ctrl_hover_help

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_install_is_singleton_per_app(qt_app):
    first = install_ctrl_hover_help(qt_app)
    second = install_ctrl_hover_help(qt_app)
    assert first is second
    assert isinstance(first, CtrlHoverHelp)


def test_install_returns_none_without_app(monkeypatch):
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    assert install_ctrl_hover_help(None) is None


def test_key_events_control_timer_and_hide(qt_app, monkeypatch):
    help = install_ctrl_hover_help(qt_app)
    hidden: list[bool] = []
    monkeypatch.setattr(help, "_hide", lambda: hidden.append(True))

    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Control, Qt.KeyboardModifier.ControlModifier)
    help.eventFilter(qt_app, press)
    assert help._ctrl_down is True
    assert help._timer.isActive()

    release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Control, Qt.KeyboardModifier.NoModifier)
    help.eventFilter(qt_app, release)
    assert help._ctrl_down is False
    assert not help._timer.isActive()
    assert hidden == [True]


def test_poll_shows_description_for_registered_widget(qt_app, monkeypatch):
    help = install_ctrl_hover_help(qt_app)
    widget = QPushButton("Analyze")
    help.register(widget, "Analyzes the input script.")

    shown: list[tuple] = []
    monkeypatch.setattr(help, "_modifiers_have_ctrl", lambda: True)
    monkeypatch.setattr(help, "_cursor_pos", lambda: QPoint(40, 40))
    monkeypatch.setattr(help, "_widget_at", lambda pos: widget)
    monkeypatch.setattr(help, "_show", lambda pos, text, w: shown.append((pos, text, w)))

    help._ctrl_down = True
    help._poll()

    assert len(shown) == 1
    pos, text, target = shown[0]
    assert text == "Analyzes the input script."
    assert target is widget
    # _show receives the raw cursor position; the real implementation adds
    # the tooltip offset when calling QToolTip.showText.
    assert pos == QPoint(40, 40)


def test_poll_does_not_reshow_for_same_widget(qt_app, monkeypatch):
    help = install_ctrl_hover_help(qt_app)
    widget = QPushButton("Render")
    help.register(widget, "Renders FLAC.")

    shown: list[tuple] = []
    monkeypatch.setattr(help, "_modifiers_have_ctrl", lambda: True)
    monkeypatch.setattr(help, "_cursor_pos", lambda: QPoint(10, 10))
    monkeypatch.setattr(help, "_widget_at", lambda pos: widget)
    monkeypatch.setattr(help, "_show", lambda pos, text, w: shown.append((pos, text, w)))

    help._ctrl_down = True
    help._last_widget = widget
    help._poll()

    assert shown == []


def test_poll_hides_when_pointer_leaves(qt_app, monkeypatch):
    help = install_ctrl_hover_help(qt_app)
    hidden: list[bool] = []
    shown: list[tuple] = []
    monkeypatch.setattr(help, "_modifiers_have_ctrl", lambda: True)
    monkeypatch.setattr(help, "_cursor_pos", lambda: QPoint(0, 0))
    monkeypatch.setattr(help, "_widget_at", lambda pos: None)
    monkeypatch.setattr(help, "_hide", lambda: hidden.append(True))
    monkeypatch.setattr(help, "_show", lambda *args: shown.append(args))

    help._ctrl_down = True
    help._last_widget = QLabel("x")
    help._poll()

    assert hidden == [True]
    assert shown == []
    assert help._last_widget is None


def test_poll_stops_when_ctrl_released_via_modifiers(qt_app, monkeypatch):
    help = install_ctrl_hover_help(qt_app)
    hidden: list[bool] = []
    monkeypatch.setattr(help, "_modifiers_have_ctrl", lambda: False)
    monkeypatch.setattr(help, "_hide", lambda: hidden.append(True))

    help._ctrl_down = True
    help._timer.start()
    help._poll()

    assert help._ctrl_down is False
    assert not help._timer.isActive()
    assert hidden == [True]


def test_description_falls_back_to_tooltip_and_parent_walk(qt_app):
    help = install_ctrl_hover_help(qt_app)
    parent = QGroupBox("Shared Render Settings")
    child = QPushButton("child", parent)

    # Unregistered child with no tooltip anywhere -> no help.
    assert help._description_for(child, QPoint(0, 0)) == ""

    # Native tooltip on the child itself is used as a fallback.
    child.setToolTip("child tooltip")
    assert help._description_for(child, QPoint(0, 0)) == "child tooltip"

    # Registered parent wins over the child's native tooltip via the walk-up.
    child.setToolTip("")
    help.register(parent, "Shared render settings live here.")
    assert help._description_for(child, QPoint(0, 0)) == "Shared render settings live here."


def test_menu_action_description(qt_app):
    help = install_ctrl_hover_help(qt_app)

    class StubMenu(QMenu):
        def __init__(self) -> None:
            super().__init__()
            self.hovered_action = None

        def actionAt(self, _pos):
            return self.hovered_action

        def mapFromGlobal(self, _pos):
            return QPoint(0, 0)

    menu = StubMenu()
    action = QAction("Open Project", menu)
    help.register_action(action, "Loads a saved project manifest.")

    menu.hovered_action = action
    assert help._description_for(menu, QPoint(0, 0)) == "Loads a saved project manifest."

    # No action under the pointer -> no help.
    menu.hovered_action = None
    assert help._description_for(menu, QPoint(0, 0)) == ""


def test_register_form_label_for_layout_field(qt_app):
    """A form row whose field is a layout (e.g. Inference Backend) still gets
    its label registered via labelForField(layout)."""
    help = install_ctrl_hover_help(qt_app)
    box = QGroupBox("Settings")
    form = QFormLayout(box)
    row = QHBoxLayout()
    combo = QPushButton("PyTorch (CPU)")
    button = QPushButton("Test")
    row.addWidget(combo)
    row.addWidget(button)
    form.addRow("Inference Backend", row)
    help.register(combo, "Which engine synthesizes audio.")

    label = form.labelForField(row)
    assert label is not None
    help.register(label, help.description_for(combo))

    assert help._description_for(label, QPoint(0, 0)) == "Which engine synthesizes audio."


def test_window_activate_with_ctrl_starts_timer(qt_app, monkeypatch):
    """Ctrl held before the app got focus (KeyPress never seen) is picked up
    when the window is activated."""
    help = install_ctrl_hover_help(qt_app)
    monkeypatch.setattr(help, "_modifiers_have_ctrl", lambda: True)

    activate = QEvent(QEvent.Type.WindowActivate)
    help.eventFilter(qt_app, activate)

    assert help._ctrl_down is True
    assert help._timer.isActive()

    # And a later deactivation hides/clears as usual.
    monkeypatch.setattr(help, "_hide", lambda: None)
    deactivate = QEvent(QEvent.Type.WindowDeactivate)
    help.eventFilter(qt_app, deactivate)
    assert help._ctrl_down is False
    assert not help._timer.isActive()


def test_register_form_labels_maps_label_to_field_text(qt_app):
    """Hovering a control's *name* (form row label) shows the same help as
    hovering the control itself."""
    help = install_ctrl_hover_help(qt_app)
    box = QGroupBox("Settings")
    form = QFormLayout(box)
    field = QPushButton("Value")
    form.addRow("Model Variant", field)
    help.register(field, "Which model variant to use.")

    help.register_form_labels(form, [field])

    label = form.labelForField(field)
    assert label is not None
    # Hovering the label resolves to its registered description.
    assert help._description_for(label, QPoint(0, 0)) == "Which model variant to use."


def test_poll_keeps_tooltip_visible_over_own_tooltip(qt_app, monkeypatch):
    """Regression: the pointer over our own tooltip popup must not hide it.

    The mask check exists because ToolTip = 0xa|Window, so a bare top-level
    window (Window flag only) must NOT be mistaken for the tooltip.
    """
    help = install_ctrl_hover_help(qt_app)
    tip_label = QLabel("the tooltip")
    tip_label.setWindowFlag(Qt.WindowType.ToolTip, True)
    hidden: list[bool] = []
    shown: list[tuple] = []
    monkeypatch.setattr(help, "_modifiers_have_ctrl", lambda: True)
    monkeypatch.setattr(help, "_cursor_pos", lambda: QPoint(0, 0))
    monkeypatch.setattr(help, "_widget_at", lambda pos: tip_label)
    monkeypatch.setattr(help, "_hide", lambda: hidden.append(True))
    monkeypatch.setattr(help, "_show", lambda *args: shown.append(args))

    help._ctrl_down = True
    help._last_widget = QLabel("original target")
    help._poll()

    assert shown == []
    assert hidden == []  # tooltip stays visible


def test_native_tooltip_swallowed_while_ctrl_down(qt_app):
    help = install_ctrl_hover_help(qt_app)
    tip_event = QEvent(QEvent.Type.ToolTip)

    help._ctrl_down = True
    assert help.eventFilter(qt_app, tip_event) is True

    help._ctrl_down = False
    assert help.eventFilter(qt_app, tip_event) is False


def test_main_window_registers_ctrl_help(qt_app, monkeypatch, tmp_path):
    """The real window installs Ctrl+hover help and registers its controls."""
    from tests.test_app_gui_profiles import _build_window

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        help = window._ctrl_help
        assert help is not None
        assert help._descriptions.get(window.analyze_button)
        assert help._descriptions.get(window.render_button)
        assert help._descriptions.get(window.inference_backend_combo)
        assert help._descriptions.get(window.export_srt_check)
        assert help._descriptions.get(window.speaker_a.cfg_weight)
        assert help._descriptions.get(window.speaker_b.pause_spin)
        assert help._descriptions.get(window.new_action)
        assert help._descriptions.get(window.download_vulkan_model_action)
        # Labels (the control's "name") share the field's description.
        assert help._descriptions.get(window.output_name_label)
        assert help._descriptions.get(window.status_label)
        assert help._descriptions.get(window._path_row_labels[0])
        assert help._descriptions.get(window._path_row_buttons[0])
        variant_label = window._project_settings_form.labelForField(window.variant_combo)
        assert variant_label is not None
        assert (
            help._descriptions.get(variant_label)
            == help._descriptions.get(window.variant_combo)
        )
        # Layout-field row (Inference Backend) label is registered explicitly.
        backend_label = window._project_settings_form.labelForField(window._inference_backend_row)
        assert backend_label is not None
        assert (
            help._descriptions.get(backend_label)
            == help._descriptions.get(window.inference_backend_combo)
        )
    finally:
        window.close()
