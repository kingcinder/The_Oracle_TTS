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

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QFormLayout, QMenu, QMenuBar, QToolTip, QWidget

from the_oracle.gui_themes import in_bulk_restyle

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
        if in_bulk_restyle():
            # Bulk stylesheet restyle in progress (gui_themes.apply_theme):
            # tens of thousands of style/palette/font events stream past in
            # one synchronous call. None of them match a branch below, and
            # real input cannot arrive until the call returns to the event
            # loop — skip the per-event Python round-trip entirely.
            return False
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
        elif etype == QEvent.Type.ToolTip:
            # Strict gate: native tooltips never appear on plain hover —
            # widgets, menus, sliders, headers. The only tooltip render path
            # is the poller's _show() below. Tour explanations bypass this
            # because the wizards call QToolTip.showText directly.
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
    """
    ctrl_help = main_window._ctrl_help
    if ctrl_help is None:
        return
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
        (
            main_window.table,
            "Review table, one row per utterance: index, speaker, "
            "original/repaired text, emotion (editable), duration, "
            "status, preview, and stem on/off.",
        ),
        (
            main_window.error_panel,
            "Status and error log. Render progress, Vulkan preflight "
            "reports, model downloads, and failures are appended here.",
        ),
    ])
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
    for group in main_window._all_speaker_groups().values():
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
    # The row labels and Browse buttons share their field's description,
    # so hovering the *name* of a control works too (the user asked for
    # "names, selections, buttons, toggles, or other areas of interest").
    ctrl_help.register(main_window._path_row_labels[0], ctrl_help.description_for(main_window.input_path))
    ctrl_help.register(main_window._path_row_buttons[0], "Open a file picker to choose the input script or transcript.")
    ctrl_help.register(main_window._path_row_labels[1], ctrl_help.description_for(main_window.outdir_path))
    ctrl_help.register(main_window._path_row_buttons[1], "Open a folder picker to choose where the rendered FLAC is written.")
    ctrl_help.register(main_window.output_name_label, ctrl_help.description_for(main_window.output_name))
    ctrl_help.register(main_window.status_label, ctrl_help.description_for(main_window.error_panel))
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
    for group in main_window._all_speaker_groups().values():
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
