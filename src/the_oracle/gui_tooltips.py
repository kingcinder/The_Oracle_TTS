"""Ctrl+hover help: hold the left Control key and hover any control to see a
short description of what it does and how to use it.

The help text comes from a per-widget registry installed by the GUI. Widgets
without a curated entry fall back to their regular Qt tooltip, so anything
that already explains itself needs no extra registration.

Implementation notes
--------------------
- A single :class:`CtrlHoverHelp` is installed on the QApplication. While the
  Control key is held (tracked via an event filter, with a keyboard-modifier
  check as a safety net for releases that happened while another window had
  focus), a lightweight poller watches the widget under the cursor and drives
  ``QToolTip`` directly, so the description appears immediately instead of
  after Qt's native tooltip delay.
- While Ctrl-help is active, the native ``ToolTip`` event is swallowed so Qt's
  delayed tooltip cannot fight the description we are showing.
- Menu items are ``QAction``\\ s rather than widgets; hovering a menu resolves
  the action under the pointer via ``actionAt``.
- Registering a widget keeps a strong reference to it (widgets already live
  for the whole window lifetime, so this adds nothing to their lifecycle).
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QFormLayout, QMenu, QMenuBar, QToolTip, QWidget

_POLL_MS = 90
_TOOLTIP_OFFSET = QPoint(14, 18)

_INSTANCE: CtrlHoverHelp | None = None  # type: ignore[valid-type]


class CtrlHoverHelp(QObject):
    """App-wide Ctrl+hover description tooltips.

    Install once per QApplication via :func:`install_ctrl_hover_help`.
    Widgets (and menu QActions) are registered with a description; while the
    Control key is held, the description is shown as a tooltip at the cursor
    whenever the pointer is over a registered control (or a child of one).
    """

    def __init__(self, app: QApplication | None = None) -> None:
        super().__init__(app or QApplication.instance())
        self._descriptions: dict[QWidget, str] = {}
        self._action_descriptions: dict[object, str] = {}
        self._ctrl_down = False
        self._last_widget: QWidget | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._app = self.parent()
        assert self._app is not None
        self._app.installEventFilter(self)

    # -- registration ------------------------------------------------------

    def register(self, widget: QWidget, description: str) -> None:
        """Associate a widget with the description shown on Ctrl+hover."""
        self._descriptions[widget] = description

    def register_action(self, action: object, description: str) -> None:
        """Associate a menu QAction with the description shown on Ctrl+hover."""
        self._action_descriptions[action] = description

    def register_many(self, pairs: list[tuple[QWidget, str]]) -> None:
        """Register several (widget, description) pairs at once."""
        for widget, description in pairs:
            self.register(widget, description)

    def description_for(self, widget: QWidget) -> str:
        """Return the registered description for a widget ('' if none)."""
        return self._descriptions.get(widget, "")

    def register_form_labels(self, form: QFormLayout, fields: list[QWidget]) -> None:
        """Register each form row label with the same help as its field.

        This is what makes hovering a control's *name* (e.g. "Model Variant"
        or "CFG Weight") show the same description as hovering the control
        itself. Fields without a registered description are skipped.
        """
        for field in fields:
            text = self._descriptions.get(field)
            if not text:
                continue
            label = form.labelForField(field)
            if label is not None:
                self._descriptions[label] = text

    # -- test seams --------------------------------------------------------

    def _widget_at(self, pos: QPoint) -> QWidget | None:
        return QApplication.widgetAt(pos)

    def _cursor_pos(self) -> QPoint:
        return QCursor.pos()

    def _modifiers_have_ctrl(self) -> bool:
        return bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)

    def _show(self, pos: QPoint, text: str, widget: QWidget) -> None:
        QToolTip.showText(pos + _TOOLTIP_OFFSET, text, widget)

    def _hide(self) -> None:
        QToolTip.hideText()

    # -- event filter ------------------------------------------------------

    def eventFilter(self, obj: object, event: QEvent) -> bool:
        etype = event.type()
        if etype == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Control:
            self._ctrl_down = True
            self._timer.start()
        elif etype == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_Control:
            self._ctrl_down = False
            self._timer.stop()
            self._hide()
            self._last_widget = None
        elif etype in (QEvent.Type.WindowDeactivate, QEvent.Type.ApplicationDeactivate):
            self._ctrl_down = False
            self._timer.stop()
            self._hide()
            self._last_widget = None
        elif etype in (QEvent.Type.WindowActivate, QEvent.Type.FocusIn):
            # Ctrl may have been pressed while another window had focus; the
            # KeyPress never reached us, so pick it up from the modifier state.
            if self._modifiers_have_ctrl():
                self._ctrl_down = True
                self._timer.start()
        elif etype == QEvent.Type.ToolTip and self._ctrl_down:
            # While Ctrl-help is active, swallow Qt's delayed native tooltip
            # so it cannot fight the description we are showing.
            return True
        return False

    # -- poller ------------------------------------------------------------

    def _poll(self) -> None:
        ctrl_now = self._modifiers_have_ctrl()
        if not (self._ctrl_down or ctrl_now):
            self._timer.stop()
            return
        if not ctrl_now:
            # KeyRelease was lost (e.g. Ctrl released while another window had
            # focus); treat it as released.
            self._ctrl_down = False
            self._timer.stop()
            self._hide()
            self._last_widget = None
            return
        pos = self._cursor_pos()
        widget = self._widget_at(pos)
        # Pointer is over our own tooltip popup; keep it visible. Mask the
        # window-type bits first: ToolTip is 0xa|Window, so a bare top-level
        # window (Window flag set) would otherwise look like a tooltip.
        if widget is not None and (
            widget.windowFlags() & Qt.WindowType.WindowType_Mask
        ) == Qt.WindowType.ToolTip:
            return
        if widget is None:
            if self._last_widget is not None:
                self._hide()
                self._last_widget = None
            return
        text = self._description_for(widget, pos)
        if not text:
            if self._last_widget is not None:
                self._hide()
                self._last_widget = None
            return
        if widget is self._last_widget:
            return  # already showing for this widget
        self._show(pos, text, widget)
        self._last_widget = widget

    # -- description resolution --------------------------------------------

    def _description_for(self, widget: QWidget, pos: QPoint) -> str:
        """Return the description for the widget under the cursor, else ''.

        Walks up the parent chain so hovering a nested child (e.g. the line
        edit inside a spin box) resolves to the nearest registered control.
        Menu items are resolved via ``actionAt`` first.
        """
        if isinstance(widget, (QMenu, QMenuBar)):
            action = widget.actionAt(widget.mapFromGlobal(pos))
            if action is not None:
                text = self._action_descriptions.get(action)
                if text:
                    return text
                tip = action.toolTip()
                if tip:
                    return tip
        current: QWidget | None = widget
        while current is not None:
            text = self._descriptions.get(current)
            if text:
                return text
            tip = current.toolTip()
            if tip:
                return tip
            current = current.parentWidget()
        return ""


def install_ctrl_hover_help(app: QApplication | None = None) -> CtrlHoverHelp | None:
    """Install (once per QApplication) the Ctrl+hover help poller.

    Returns the singleton instance for the given app, or ``None`` when no
    QApplication exists yet (callers should guard on that).
    """
    global _INSTANCE
    app = app or QApplication.instance()
    if app is None:
        return None
    if _INSTANCE is None or _INSTANCE.parent() is not app:
        _INSTANCE = CtrlHoverHelp(app)
    return _INSTANCE
