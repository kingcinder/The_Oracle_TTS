"""Ctrl+hover help: hold the left Control key and hover any control to see a
short description of what it does and how to use it.

The help text comes from a per-widget registry installed by the GUI. Widgets
without a curated entry fall back to their regular Qt tooltip, so anything
that already explains itself needs no extra registration.

Implementation notes
--------------------
- A :class:`CtrlHoverHelp` is owned by its *main window*. No event filter is
  installed on the QApplication. On Windows it uses timer polling without
  native event filters because Qt delivered an invalid event wrapper during
  window layout on the 2026-10-16 CI runner. On other platforms the filter
  rides the window and tooltip owners. An application-wide Python filter receives
  every event of every object in
  the process, including objects that are mid-destruction; that is the frame
  the 1.4.0 certifier faulted in on the second window build (2026-10-13).
  The filter rides the window plus each widget that can raise a tooltip on
  platforms that support the event-filter path; Windows stores and suppresses
  those native tips while the same timer poller supplies Ctrl+hover help.
  :meth:`CtrlHoverHelp.release` restores native tips and detaches filters before
  the window goes away.
- The ToolTip swallow is per tooltip *owner*, not per window: Qt delivers
  ``QEvent::ToolTip`` to the widget that owns the tooltip and never to its
  ancestors (verified against Qt 6 offscreen — a window-level filter sees
  nothing when a child is hovered), so gating from the window alone would
  silently re-enable native tooltips on every control. Empty-tooltip widgets
  cannot raise one, so the sweep only has to cover the widgets that carry
  text, and it re-runs when a child is polished (dialogs and lazily built
  controls appear long after the window was built).
- While the Control key is held (tracked by the main window's key handlers,
  with a keyboard-modifier check as a safety net for releases that happened while
  another window had focus), a lightweight poller watches the widget under
  the cursor and drives ``QToolTip`` directly, so the description appears
  immediately instead of after Qt's native tooltip delay.
- Registration is lifetime-tracked: a registered widget's ``destroyed``
  signal drops its entry, so the registry — and the parent walk that reads
  it — can never reach a widget Qt has already freed.
- While Ctrl-help is active, the native ``ToolTip`` event is always
  swallowed: Qt's delayed tooltip can never fire on plain hover, so the
  Ctrl+hover description is the only tooltip in the app. Tour step
  explanations are exempt because the wizards call ``QToolTip.showText``
  directly (not hover-driven).
- Menu items are ``QAction``\\ s rather than widgets; hovering a menu resolves
  the action under the pointer via ``actionAt``.
- Registering a widget keeps a strong reference to it (widgets already live
  for the whole window lifetime, so this adds nothing to their lifecycle).
- The filter's first check skips everything while ``gui_themes``'s
  ``bulk_restyle`` scope is open: a stylesheet switch delivers ~20k+
  StyleChange/PaletteChange/FontChange events synchronously, and paying a
  Python round-trip for each one taxed the restyle without changing a single
  outcome (no branch in this filter acts on any event Qt generates inside
  ``setStyleSheet``, and no user input can be processed until that call
  returns to the event loop).
"""

from __future__ import annotations

from collections.abc import Callable

from shiboken6 import isValid as _shiboken_is_valid
from PySide6.QtCore import QChildEvent, QElapsedTimer, QEvent, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QFormLayout, QMenu, QMenuBar, QToolTip, QWidget

from the_oracle.platform_support import is_windows

from the_oracle.gui_themes import in_bulk_restyle

_POLL_MS = 90
_TOOLTIP_OFFSET = QPoint(14, 18)
#: Window-level events that mean the widget tree may have grown (a collapsed
#: section built on its first expand, a lazily created row). The sweep they
#: trigger is debounced: these fire constantly during layout and repaint.
_RESWEEP_EVENTS = frozenset(
    {
        QEvent.Type.LayoutRequest,
        QEvent.Type.Show,
        QEvent.Type.Resize,
        QEvent.Type.Polish,
        QEvent.Type.Paint,
    }
)
_RESWEEP_MIN_INTERVAL_MS = 250

_INSTANCE: CtrlHoverHelp | None = None  # type: ignore[valid-type]


class CtrlHoverHelp(QObject):
    """Ctrl+hover description tooltips, scoped to the window that binds them.

    Bind once per main window via :func:`install_ctrl_hover_help`. Window
    helpers are distinct; detached helpers are reused for app-level unit tests.
    Widgets (and menu QActions) are registered with a description; while the Control key is held, the
    description is shown as a tooltip at the cursor whenever the pointer is
    over a registered control (or a child of one).
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
        self._sweep_timer = QTimer(self)
        self._sweep_timer.setInterval(1000)
        self._sweep_timer.timeout.connect(self._maybe_resweep)
        self._app = QApplication.instance()
        assert self._app is not None
        # The window the filter rides. Nothing is installed at construction;
        # the original native fault landed in the former app-wide filter, but
        # its root cause is unconfirmed (module docstring).
        self._window: QWidget | None = None
        #: Widgets that own a tooltip text we are swallowing ToolTip on.
        self._swept: set[QWidget] = set()
        self._original_tooltips: dict[QWidget, str] = {}
        self._sweep_clock = QElapsedTimer()

    # -- installation ------------------------------------------------------

    @property
    def window(self) -> QWidget | None:
        """The window the filter is installed on (``None`` when detached)."""
        return self._window

    def attach(self, window: QWidget) -> None:
        """Install the filter on ``window``, detaching from any previous one."""
        if self._window is window:
            return
        self.release()
        if self.parent() is not window:
            self.setParent(window)
        self._window = window
        if not is_windows():
            window.installEventFilter(self)
        else:
            # Native Windows QEvent wrappers can be invalidated during Qt
            # layout churn before PySide calls eventFilter. Use the existing
            # cursor poller and a bounded tooltip sweep there, with no native
            # event-filter callbacks in the path.
            self._sweep_timer.start()
        # PySide emits `destroyed` with no argument (the C++ signature's
        # default), so the window is bound explicitly rather than taken from
        # the signal.
        window.destroyed.connect(lambda *_args, _w=window: self._detach(_w))
        self._sweep(window)

    def _sweep(self, root: QWidget) -> None:
        """Swallow ToolTip on every tooltipped widget in ``root``'s subtree.

        The strict gate is "plain hover never produces a native tooltip", and
        Qt sends the ToolTip event to the owning widget only — so the filter
        has to ride each widget that carries text. Widgets without a tooltip
        cannot raise one; they still resolve through the parent walk in
        ``_description_for``.
        """
        if not isinstance(root, QWidget):
            return
        for widget in (root, *root.findChildren(QWidget)):
            if widget is self._window:
                continue
            if widget in self._swept:
                if is_windows() and widget.toolTip():
                    tip = widget.toolTip()
                    self._original_tooltips.setdefault(widget, tip)
                    self._descriptions.setdefault(widget, tip)
                    widget.setToolTip("")
                continue
            if not widget.toolTip():
                continue
            if is_windows():
                tip = widget.toolTip()
                self._original_tooltips.setdefault(widget, tip)
                self._descriptions.setdefault(widget, tip)
                widget.setToolTip("")
            else:
                widget.installEventFilter(self)
            self._swept.add(widget)
            widget.destroyed.connect(lambda *_args, _w=widget: self._swept.discard(_w))

    def _maybe_resweep(self) -> None:
        """Sweep the bound window again, at most a few times per second."""
        if self._window is None:
            return
        if self._sweep_clock.isValid() and self._sweep_clock.elapsed() < _RESWEEP_MIN_INTERVAL_MS:
            return
        self._sweep_clock.restart()
        self._sweep(self._window)

    def _detach(self, window: QWidget) -> None:
        """``window.destroyed``: detach, unless a newer window already bound."""
        if self._window is not window:
            return
        self.release()

    def release(self) -> None:
        """Detach from the window and drop everything that pointed at it.

        Called on window destruction, on re-binding, and at the end of
        ``MainWindow.closeEvent`` — so a teardown window can never leave a
        filter (or a registry entry) aimed at freed C++ objects.
        """
        global _INSTANCE
        window = self._window
        self._window = None
        for widget in (window, *self._swept):
            if widget is None:
                continue
            if not is_windows():
                try:
                    widget.removeEventFilter(self)
                except RuntimeError:
                    # The C++ widget is already gone; the filter went with it.
                    pass
        for widget, text in self._original_tooltips.items():
            try:
                if _shiboken_is_valid(widget):
                    widget.setToolTip(text)
            except RuntimeError:
                pass
        self._swept.clear()
        self._original_tooltips.clear()
        self._sweep_clock.invalidate()
        self._sweep_timer.stop()
        self._timer.stop()
        self._hide()
        self._last_widget = None
        self._ctrl_down = False
        self._descriptions.clear()
        self._action_descriptions.clear()
        if _INSTANCE is self:
            _INSTANCE = None

    def control_pressed(self) -> None:
        """Start Ctrl-hover polling from the owning window's key handler."""
        self._ctrl_down = True
        self._timer.start()

    def control_released(self) -> None:
        """Stop Ctrl-hover polling from the owning window's key handler."""
        self._ctrl_down = False
        self._timer.stop()
        self._hide()
        self._last_widget = None

    # -- registration ------------------------------------------------------

    def register(self, widget: QWidget, description: str) -> None:
        """Associate a widget with the description shown on Ctrl+hover.

        The entry is dropped when the widget is destroyed: the registry holds
        strong references, and a stale one would leave ``_description_for``
        walking (or reading) a freed C++ widget.
        """
        if widget not in self._descriptions and isinstance(widget, QObject):
            widget.destroyed.connect(lambda *_args, _w=widget: self._forget(_w))
        self._descriptions[widget] = description

    def register_action(self, action: object, description: str) -> None:
        """Associate a menu QAction with the description shown on Ctrl+hover."""
        self._action_descriptions[action] = description

    def _forget(self, widget: QWidget) -> None:
        """``widget.destroyed``: drop the registry entry for a freed widget."""
        self._descriptions.pop(widget, None)

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
        if in_bulk_restyle():
            # Bulk stylesheet restyle in progress (gui_themes.apply_theme):
            # tens of thousands of style/palette/font events stream past in
            # one synchronous call. None of them match a branch below, and
            # real input cannot arrive until the call returns to the event
            # loop — skip the per-event Python round-trip entirely.
            return False
        # During native Qt teardown, Windows/PySide can dispatch a Python
        # wrapper whose QEvent was already deleted. Calling event.type() on
        # that stale wrapper can access-violate before Python can catch it.
        if not _shiboken_is_valid(event) or (
            isinstance(obj, QObject) and not _shiboken_is_valid(obj)
        ):
            return False
        etype = event.type()
        if etype in (QEvent.Type.WindowDeactivate, QEvent.Type.ApplicationDeactivate):
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
        elif etype == QEvent.Type.ToolTip:
            # Strict gate: native tooltips never appear on plain hover —
            # widgets, menus, sliders, headers. The only tooltip render path
            # is the poller's _show() below. Tour explanations bypass this
            # because the wizards call QToolTip.showText directly.
            return True
        elif etype in (QEvent.Type.ChildAdded, QEvent.Type.ChildPolished):
            # A control (or a whole dialog) appeared after the window was
            # built; sweep it so its tooltips are swallowed too. Dialogs are
            # children of the window object, so their arrival lands here.
            if isinstance(event, QChildEvent):
                self._sweep(event.child())
        elif etype in _RESWEEP_EVENTS:
            # Layout/visibility churn on the window: new controls may exist.
            self._maybe_resweep()
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


def install_ctrl_hover_help(
    target: QApplication | QWidget | None = None,
) -> CtrlHoverHelp | None:
    """Bind a window-owned Ctrl+hover poller.

    Passing a QWidget creates or returns that window's helper. Passing a
    QApplication (or nothing) returns a detached helper for unit-level callers;
    product windows should always pass themselves so Qt owns the helper's
    lifetime. No event filter is installed on the QApplication.

    Returns the window's helper, a detached singleton for app-level test
    callers, or ``None`` when no QApplication exists yet.
    """
    global _INSTANCE
    target = target if target is not None else QApplication.instance()
    if target is None:
        return None
    app = target if isinstance(target, QApplication) else QApplication.instance()
    if app is None:
        return None
    if isinstance(target, QWidget):
        if _INSTANCE is not None and _INSTANCE.window is target:
            return _INSTANCE
        helper = CtrlHoverHelp(target)
        helper.attach(target)
        _INSTANCE = helper
        return helper
    if _INSTANCE is None or _INSTANCE.parent() is not app:
        _INSTANCE = CtrlHoverHelp(app)
    return _INSTANCE


def register_ctrl_help_descriptions(
    main_window, *, cuda_reason_fn: Callable[[], str]
) -> None:
    """Register Ctrl+hover help text for the main window's controls.

    The (widget, description) table for the installed :class:`CtrlHoverHelp`:
    buttons, fields, spin boxes, backend knobs, speaker controls, and menu
    actions get a curated description, and unregistered widgets fall back to
    their regular Qt tooltip if one is set. The widgets and actions ride
    ``main_window`` (the gui_ingest_tools direction rule: nothing here
    imports app_gui).

    ``cuda_reason_fn`` is injected (extraction slice 11, 2026-10-10): the
    caller passes its own ``cuda_reason`` reference, so the PyTorch-device
    sentence is composed from the CALLER's module globals. app_gui's delegate
    passes the bare app_gui name -- it resolves at call time, keeping the
    suite's app_gui-level ``cuda_reason`` rebinds live for this moved text
    (the ``_vulkan_prerequisite_missing`` ``find_binary=`` precedent).

    Section-owned descriptions follow their sections' laziness (lazy
    collapsed sections, 2026-10-10): a restored-collapsed section has no
    widgets to describe until it builds, so each block registers now for a
    built section or on the section's first ``content_built``.
    """
    ctrl_help = main_window._ctrl_help
    if ctrl_help is None:
        return
    _register_action_buttons(main_window, ctrl_help)
    _register_action_help(main_window, ctrl_help)
    _register_menubar_help(main_window, ctrl_help)
    _register_when_built(main_window, "paths", lambda: _register_paths_help(main_window, ctrl_help))
    _register_when_built(main_window, "table", lambda: _register_table_help(main_window, ctrl_help))
    _register_when_built(main_window, "status", lambda: _register_status_help(main_window, ctrl_help))
    _register_when_built(
        main_window, "shared", lambda: _register_shared_help(main_window, ctrl_help, cuda_reason_fn)
    )
    for key, group in (("speaker_a", main_window.speaker_a), ("speaker_b", main_window.speaker_b)):
        _register_when_built(main_window, key, lambda g=group: _register_speaker_group_help(g, ctrl_help))


def _register_when_built(main_window, key: str, register) -> None:
    """Run ``register`` now if the section's content exists, otherwise on its
    first build: a deferred section's widgets only come into existence when
    it expands (or when code touches one), and their help must follow."""
    section = main_window._section_registry[key][0]
    if section.is_content_built():
        register()
    else:
        section.content_built.connect(register)


def _register_paths_help(main_window, ctrl_help) -> None:
    ctrl_help.register_many([
        (
            main_window.input_path,
            "Input script or transcript file to convert into dialogue audio. "
            "Click Browse or type a path; Analyze then reads, repairs, and "
            "attributes the lines.",
        ),
        (
            main_window.outdir_path,
            "Output folder for the rendered FLAC (plus stems and optional "
            "SRT). Defaults to the repo's Output/ folder; typing a new "
            "path creates it.",
        ),
        (
            main_window.output_name,
            "Filename for the final render, without the extension. Leave "
            "blank to auto-derive from the input file name.",
        ),
    ])
    # The row labels and Browse buttons share their field's description,
    # so hovering the *name* of a control works too (the user asked for
    # "names, selections, buttons, toggles, or other areas of interest").
    ctrl_help.register(main_window._path_row_labels[0], ctrl_help.description_for(main_window.input_path))
    ctrl_help.register(main_window._path_row_buttons[0], "Open a file picker to choose the input script or transcript.")
    ctrl_help.register(main_window._path_row_labels[1], ctrl_help.description_for(main_window.outdir_path))
    ctrl_help.register(main_window._path_row_buttons[1], "Open a folder picker to choose where the rendered FLAC is written.")
    ctrl_help.register(main_window.output_name_label, ctrl_help.description_for(main_window.output_name))


def _register_action_buttons(main_window, ctrl_help) -> None:
    """The bare Analyze/Render action row (never inside a section)."""
    ctrl_help.register_many([
        (
            main_window.analyze_button,
            "Analyze the input: repair text, attribute lines to speakers "
            "A and B, detect emotions, and fill the review table. No "
            "audio is rendered yet.",
        ),
        (
            main_window.render_button,
            "Render the full dialogue to FLAC: synthesize every utterance "
            "with the selected inference backend, assemble the two "
            "speakers, and write the output file.",
        ),
    ])


def _register_table_help(main_window, ctrl_help) -> None:
    ctrl_help.register_many([
        (
            main_window.table,
            "Review table, one row per utterance: index, speaker, "
            "original/repaired text, emotion (editable), duration, "
            "status, preview, and stem on/off.",
        ),
    ])


def _register_status_help(main_window, ctrl_help) -> None:
    ctrl_help.register_many([
        (
            main_window.error_panel,
            "Status and error log. Render progress, Vulkan preflight "
            "reports, model downloads, and failures are appended here.",
        ),
    ])
    ctrl_help.register(main_window.status_label, ctrl_help.description_for(main_window.error_panel))


def _register_shared_help(main_window, ctrl_help, cuda_reason_fn) -> None:
    ctrl_help.register_many([
        (
            main_window.variant_combo,
            "Chatterbox model variant: standard (default), multilingual "
            "(per-speaker language), or turbo (faster; PyTorch-only, so "
            "it disables the Vulkan backend).",
        ),
        (
            main_window.correction_mode_combo,
            "How aggressively the text-repair pass fixes grammar, "
            "spelling, punctuation, and directives before synthesis.",
        ),
        (
            main_window.loudness_combo,
            "Loudness normalization applied to the final render: off, "
            "light, or medium.",
        ),
        (
            main_window.crossfade_spin,
            "Crossfade in milliseconds between consecutive utterances so "
            "speaker turns blend smoothly.",
        ),
        (
            main_window.export_srt_check,
            "Also write a .srt subtitle file next to the rendered FLAC, "
            "one cue per utterance.",
        ),
        (
            main_window.inference_backend_combo,
            "Inference backend: PyTorch runs the Chatterbox model through the "
            "installed PyTorch runtime, while Vulkan delegates to audio.cpp. "
            "The separate PyTorch Device picker chooses CPU/DRAM or a usable "
            "CUDA/NVIDIA card when PyTorch has CUDA support.",
        ),
        (
            main_window.pytorch_device_combo,
            "PyTorch execution device: CPU / system DRAM, or a suitable "
            "CUDA / NVIDIA GPU. Disabled CUDA entries are detected hardware "
            "that is missing a CUDA-enabled PyTorch runtime or does not meet "
            "the 4 GiB Chatterbox VRAM floor. " + cuda_reason_fn(),
        ),
        (
            main_window.test_vulkan_button,
            "Run a quick audio.cpp preflight (binary + model + device "
            "list) and report which GPU the Vulkan backend would use, "
            "before rendering.",
        ),
        (
            main_window.vulkan_prerequisite_warning,
            "Inline warning shown when the Vulkan backend is selected but "
            "audiocpp_cli or the Chatterbox model is not configured yet.",
        ),
        (
            main_window.audio_cpp_device_combo,
            "Vulkan device passed to audio.cpp (--device). Auto lets "
            "audio.cpp pick; the dropdown lists detected devices. "
            "Equivalent to ORACLE_AUDIOCPP_DEVICE.",
        ),
        (
            main_window.audio_cpp_device_label,
            "Read-only summary of the Vulkan devices audio.cpp detected; "
            "the dropdown above is populated from this list.",
        ),
        (
            main_window.audio_cpp_threads_spin,
            "Thread count passed to audio.cpp (--threads). Default lets "
            "audio.cpp decide. Equivalent to ORACLE_AUDIOCPP_THREADS.",
        ),
        (
            main_window.audio_cpp_timeout_spin,
            "Per-synthesis timeout in seconds for audio.cpp. Default (0) "
            "uses audio.cpp's 600 s. Equivalent to ORACLE_AUDIOCPP_TIMEOUT.",
        ),
        (
            main_window.audio_cpp_max_batch_spin,
            "Maximum cache-missing stems per audio.cpp --request-sequence "
            "subprocess. Default (0) uses the engine's 32-request cap. "
            "Equivalent to ORACLE_AUDIOCPP_MAX_BATCH.",
        ),
    ])
    # Form row labels (e.g. "Model Variant", "CFG Weight") get the same
    # help as their field.
    ctrl_help.register_form_labels(main_window._project_settings_form, [
        main_window.variant_combo,
        main_window.correction_mode_combo,
        main_window.loudness_combo,
        main_window.crossfade_spin,
        main_window.inference_backend_combo,
        main_window.audio_cpp_device_combo,
        main_window.audio_cpp_threads_spin,
        main_window.audio_cpp_timeout_spin,
        main_window.audio_cpp_max_batch_spin,
        main_window.pytorch_device_combo,
    ])
    # The Inference Backend row's field is a layout, so its label is not
    # covered by register_form_labels; register it explicitly.
    backend_label = main_window._project_settings_form.labelForField(main_window._inference_backend_row)
    if backend_label is not None:
        ctrl_help.register(backend_label, ctrl_help.description_for(main_window.inference_backend_combo))


def _register_speaker_group_help(group, ctrl_help) -> None:
    ctrl_help.register_many([
            (
                group.reference_picker,
                "Voice reference audio this speaker clones: a default "
                "voice, a recent custom clip, or a new file you choose.",
            ),
            (
                group.language_combo,
                "Spoken language for this speaker. Only selectable on the "
                "multilingual variant; otherwise fixed to English.",
            ),
            (
                group.blend_weight_spin,
                "Voice Dominance: preference weight (0-100%) for the "
                "two-voice hybrid. 100% keeps the base voice alone, 0% "
                "hands the voice to the second clip, 50% meets in the "
                "middle. Deterministic and content-hashed, so the same "
                "blend always renders identically on both backends.",
            ),
            (
                group.cfg_weight,
                "Identity Lock: classifier-free guidance weight (0.0-1.5). "
                "How strongly the conditioned guess wins over the "
                "unconditioned one when sampling each token. High = glued "
                "to the reference clip; low = freer and drifts more.",
            ),
            (
                group.exaggeration,
                "Emphasis Punch: Chatterbox's expression knob (0.0-1.5). "
                "Scales pitch and stress swing around the neutral read: "
                "0.0 flat, 0.5 model default, 1.5 theatrical.",
            ),
            (
                group.temperature,
                "Delivery Variety: sampling randomness (0.1-1.5). Low = the "
                "most predictable, steady pronunciation; high = livelier "
                "takes that can occasionally stumble. Held constant per "
                "speaker so the character does not drift between lines.",
            ),
            (
                group.emotion_intensity,
                "Emotion Depth: blend weight (0.0-2.0) for the emotion "
                "auto-detected on each line. 1.0 fully applies the preset, "
                "0.5 halfway blends with your sliders, 0.0 ignores "
                "detection. Only emphasis and pacing move; timbre stays "
                "locked.",
            ),
            (
                group.naturalness,
                "Human Drift: heuristic (0.0-1.0) applied once per speaker. "
                "Per 0.1: cfg_weight -0.12, temperature +0.18, "
                "repetition_penalty -0.25, min_p +0.03, breaths +12%. "
                "Reads as a relaxed human take; 0.0 leaves every other "
                "slider untouched.",
            ),
            (
                group.pause_spin,
                "Breath After This Speaker: silence (0-2000 ms) after each "
                "turn, scaled by the line's final punctuation: x1.0 "
                "period, x1.3 !, x1.25 ?, x1.6 ellipsis, x0.7 trailing "
                "line. Chunk seams breathe ~35% (min 40 ms).",
            ),
        ])
    ctrl_help.register_form_labels(group.form, [
        group.reference_picker,
        group.blend_picker,
        group.blend_weight_spin,
        group.blend_mode_combo,
        group.language_combo,
        group.cfg_weight,
        group.exaggeration,
        group.temperature,
        group.emotion_intensity,
        group.naturalness,
        group.pause_spin,
    ])


def _register_action_help(main_window, ctrl_help) -> None:
    ctrl_help.register_many([
        (main_window.new_action, "Start a new project: clears the review table and plan while keeping the current shared and speaker settings."),
        (main_window.open_action, "Load a saved project manifest, restoring the plan, review table, and settings."),
        (main_window.save_action, "Save the current project to its manifest file."),
        (main_window.save_as_action, "Save the current project to a new manifest file."),
        (main_window.reset_defaults_action, "Restore every setting to its built-in default."),
        (main_window.save_settings_action, "Export the current GUI settings (shared + speakers) to a profile file."),
        (main_window.load_settings_action, "Apply a previously saved GUI settings profile."),
        (main_window.save_template_action, "Save the current settings as a named template for the Load Template menu."),
        (main_window.confirmation_action, "Re-enable the delete-confirmation prompt for removing review-table rows."),
        (main_window.remember_backend_action, "Remember which inference backend was chosen (PyTorch CPU or Vulkan GPU) plus the audiocpp_cli and Chatterbox model paths, and restore them automatically at next launch so the GPU needs no setup again. Uncheck to always start on PyTorch (CPU)."),
        (main_window.download_vulkan_model_action, "Fetch the audio.cpp Chatterbox model in the background and report the ORACLE_AUDIOCPP_MODEL path to use."),
        (main_window.replay_full_wizard_action, "Replay the complete hardware discovery and dependency-ordered GUI tutorial."),
        (main_window.replay_discovery_wizard_action, "Replay only the hardware and inference availability discovery step."),
        (main_window.replay_main_wizard_action, "Replay only the main-page GUI feature tour, starting after hardware discovery."),
        (main_window.recording_wizard_action, "Replay the Recording Studio setup guide, including microphone, folder, naming, and speaking technique."),
        (main_window.output_filename_warning_action, "Show or hide the caution before a render uses its automatically derived generic output filename."),
    ])


def _register_menubar_help(main_window, ctrl_help) -> None:
    # The bar itself (not just its actions): covers the menu-bar padding
    # and Qt's internal overflow extension button, which has no text of
    # its own and resolves its description through the parent walk.
    ctrl_help.register(
        main_window.menuBar(),
        "Menu bar: project, settings, recording, theme, and help menus. "
        "Hold Ctrl and hover a menu to read what each entry does.",
    )
    menubar_actions = main_window.menuBar().actions()
    if len(menubar_actions) >= 1:
        ctrl_help.register_action(menubar_actions[0], "Project management: new, open, save, save-as.")
    if len(menubar_actions) >= 2:
        ctrl_help.register_action(menubar_actions[1], "GUI settings: profiles, templates, and Vulkan backend setup.")
    if len(menubar_actions) >= 3:
        ctrl_help.register_action(menubar_actions[2], main_window.recording_studio_action.toolTip())
