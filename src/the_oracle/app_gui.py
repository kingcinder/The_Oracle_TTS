"""PySide6 desktop GUI for the Chatterbox-only The Oracle app."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter, time
import difflib
import json
import os
import sys
import subprocess  # patch surface: tests resolve app_gui.subprocess
import threading
from typing import Callable

from PySide6.QtCore import QThread, Qt, QUrl, Signal, QTimer
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from the_oracle.app_paths import (
    OraclePaths,
    ensure_repo_default_paths,
    normalize_output_filename,
    default_output_filename,
    resolve_output_filename,
)
from the_oracle.correction_modes import CORRECTION_MODE_OPTIONS, correction_mode_label, normalize_correction_mode
from the_oracle.emotion.goemotions import SUPPORTED_EMOTIONS
from the_oracle.gui_settings import (
    GUISettingsError,
    PayloadDefaults,
    WidgetSnapshot,
    cast_request_from_payload,
    clear_trusted_input_files,
    current_gui_settings_payload,
    default_gui_settings_payload,
    input_file_is_trusted,
    list_templates,
    load_app_settings,
    load_gui_settings,
    load_recent_reference_paths,
    load_template,
    remember_format_backup,
    remember_recent_reference_path,
    remember_trusted_input_file,
    save_app_settings,
    save_gui_settings,
    save_template,
    speaker_config_from_payload,
)
from the_oracle.gui_utils import (
    MAX_CAST_SPEAKERS,
    CastModel,
    kill_process_tree,
    next_speaker_key,
    normalize_cast_keys,
)
# The Recording Studio cluster (2026-09-28 slice) lives in gui_recording;
# the dialog is re-exported so MainWindow constructs it from app_gui globals
# and app_gui-level class patches stay live. RecordStudioWorker is
# deliberately NOT re-exported: the dialog builds its worker from
# gui_recording's globals, so an app_gui-level patch of it would be a silent
# no-op (the _CastRow precedent; pinned in tests/test_gui_recording_owner.py).
from the_oracle.gui_recording import RecordingStudioDialog
from the_oracle.device_support import CUDADeviceInfo, cuda_devices, cuda_reason
from the_oracle.gui_themes import DEFAULT_THEME, THEMES, apply_theme
from the_oracle.gui_tooltips import install_ctrl_hover_help
from the_oracle.inference_wizard import InferenceSetupWizard
from the_oracle.recording_wizard import RecordingStudioSetupWizard
from the_oracle.gui_widgets import PerceptualSlider
from the_oracle.gui_render import PreviewWorker, RenderProgressDialog, RenderWorker, _render_child_environment
from the_oracle.gui_chrome import LivePanel, build_live_section
from the_oracle.gui_sections import QHSectionGroup, collapsible_section
from the_oracle.models.project import RenderPlan, VoiceProfile, VoiceSettings, Utterance
from the_oracle.models.settings import RenderSettings, SpeakerSettings
from the_oracle.pipeline import OraclePipeline, RenderProgress
from the_oracle.project_manifest import build_saved_project, load_project_manifest, save_project_manifest
from the_oracle.voice_catalog import (
    VoiceChoice,
    blend_voice_choices,
    default_voice_choices,
    save_blend_voice,
)
from the_oracle.tts_engines.chatterbox_engine import SUPPORTED_VARIANTS, ChatterboxEngine
from the_oracle.gui_vulkan import (
    AudioCppUnavailableError,  # re-export: tests raise/expect it via app_gui
    ModelDownloadThread,
    VulkanDeviceProbeThread,
    VulkanPreflightThread,
    VulkanSetupThread,
    _device_row_text,
    _parse_oracle_model_path,
    _vulkan_preflight_report as _vulkan_preflight_report_impl,
    _vulkan_prerequisite_missing as _vulkan_prerequisite_missing_impl,
)
# find_audiocpp_binary is this module's PATCH SEAM: ~12 test sites monkeypatch
# it here by name, and the delegates below resolve it at call time. It is also
# a deliberate re-export (window-fixture tests patch app_gui.AudioCppVulkanEngine).
from the_oracle.tts_engines.vulkan_backend import AudioCppVulkanEngine, find_audiocpp_binary  # noqa: F401

# CPU is the only verified Chatterbox execution path in this project.
# Preview and render always use "cpu"; the constant is defined here so
# it can be updated in one place if a verified GPU path is added later.
_DEVICE_MODE: str = "cpu"

#: One bullet style for every speaker-ref hint line. A dialog list has no way to
#: make a warning look different from a suggestion, so unlike the CLI's terminal
#: output it does not try; the sentences themselves come from
#: ``the_oracle.speaker_ref_hints``, which owns the wording for both surfaces.
_SPEAKER_HINT_BULLET = "  \u2022 "


def _vulkan_prerequisite_missing() -> list[str]:
    """Delegate to the gui_vulkan policy body (2026-09-28 extraction).

    The body lives in gui_vulkan; the binary probe crosses as INJECTION: the
    bare ``find_audiocpp_binary`` name resolves from THIS module's globals at
    call time, so app_gui-level monkeypatches keep intercepting it exactly as
    they did before the move (the 2026-09-20 harm history is why the seam is
    shaped this way).
    """
    return _vulkan_prerequisite_missing_impl(find_binary=find_audiocpp_binary)


def _vulkan_preflight_report(device_index: int | None) -> str:
    """Delegate to the gui_vulkan policy body (2026-09-28 extraction);
    the binary probe is injected from this module's globals at call time."""
    return _vulkan_preflight_report_impl(device_index, find_binary=find_audiocpp_binary)
# The cast-management cluster (2026-10-05, slice 9) lives in gui_cast; the
# identical objects are re-exported here so MainWindow constructs them from
# app_gui globals and app_gui-level class patches stay live for this module's
# sites. SpeakerGroup, _speaker_settings_from_group and
# _apply_speaker_settings_to_group now have read sites on BOTH sides — the
# patch-surface net's PARTIAL_OWNED records that an app_gui-level patch of
# them covers only this module's sites. _CastRow is dialog-only and is NOT
# re-exported: an app_gui-level patch of it would be a silent no-op, so it
# is listed in moved_owners.
from the_oracle.gui_cast import (
    CastManagementDialog,
    SpeakerGroup,
    _apply_speaker_settings_to_group,
    _speaker_settings_from_group,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._startup_t0 = perf_counter()
        # App-level settings (remembered inference backend + resolved audio.cpp
        # paths) are loaded before the UI so the menu action can reflect them;
        # they are applied to the widgets in _on_gui_shown. Persistence of
        # changes is gated until then so the initial default apply never writes
        # over the remembered choice.
        self._app_settings = load_app_settings()
        self._app_settings_ready = False
        self._inference_wizard: InferenceSetupWizard | None = None
        self._recording_wizard: RecordingStudioSetupWizard | None = None
        # Recording guide launch, armed by open_recording_studio: a
        # single-shot child timer like the startup three, so the pending call
        # lives and dies with this window instead of riding the context-less
        # QTimer.singleShot helper.
        self._recording_wizard_launch_timer = QTimer(self)
        self._recording_wizard_launch_timer.setSingleShot(True)
        self._recording_wizard_launch_timer.timeout.connect(self._start_recording_wizard)
        # Section layout registry: key -> (section, splitter, index). Filled
        # during _build_ui; drives persistence of splitter sizes, section
        # size sliders, and collapse states.
        self._section_registry: dict[str, tuple[QHSectionGroup, QSplitter, int]] = {}
        self._current_theme = str(self._app_settings.get("theme") or DEFAULT_THEME)
        if self._current_theme not in THEMES:
            self._current_theme = DEFAULT_THEME
        self._startup_marks: list[tuple[str, float]] = []
        self._gui_shown_wall: float | None = None
        self.repo_root = Path(__file__).resolve().parents[2]
        self.paths: OraclePaths = ensure_repo_default_paths(self.repo_root)
        # User-selectable workspace defaults are separate from the repository's
        # bundled folders.  They are restored before the UI is built so a
        # first-run wizard choice controls both the initial fields and every
        # subsequent Browse dialog.
        self._default_input_dir = Path(
            self._app_settings.get("default_input_dir") or self.paths.input_dir
        ).expanduser()
        self._default_output_dir = Path(
            self._app_settings.get("default_output_dir") or self.paths.output_dir
        ).expanduser()
        self.output_filename_warning_enabled = bool(
            self._app_settings.get("output_filename_warning", True)
        )
        self.pipeline: OraclePipeline | None = None
        self._prewarmed_pipeline: OraclePipeline | None = None
        self._prewarmed_engine = None
        self._prewarm_state = "not_started"  # not_started, warming, ready, failed
        self._prewarm_thread: PrewarmThread | None = None
        self._prewarm_lock = threading.Lock()
        self._prewarm_timing: dict[str, float] | None = None
        self.plan: RenderPlan | None = None
        self.current_project_path: Path | None = None
        self.render_worker: RenderWorker | None = None
        self.preview_worker: PreviewWorker | None = None
        # Speaker cast: ordered voice keys (["A", "B"] by default) plus
        # optional character names. The cast-management dialog edits these;
        # the main window's panels always mirror them.
        self._cast: list[str] = ["A", "B"]
        self._speaker_names: dict[str, str] = {}
        # Open child windows, tracked per-instance so no worker or completion
        # event is ever orphaned when several are open at once.
        self._recording_studios: set[RecordingStudioDialog] = set()
        self._cast_dialogs: set[CastManagementDialog] = set()
        self._vulkan_probe_thread: VulkanDeviceProbeThread | None = None
        self._vulkan_devices_probed = False
        self._cuda_devices: list[CUDADeviceInfo] = cuda_devices()
        self._vulkan_preflight_thread: VulkanPreflightThread | None = None
        self._model_download_thread: ModelDownloadThread | None = None
        # Automatic CPU→GPU setup: when the Vulkan backend is selected but its
        # prerequisites are missing, the GUI builds audiocpp_cli / downloads the
        # model in the background (VulkanSetupThread) and queues any render or
        # preview that arrived in the meantime, so the switch completes by itself.
        self._vulkan_setup_thread: VulkanSetupThread | None = None
        self._vulkan_setup_attempted = False
        self._render_queued_after_setup = False
        self._preview_queued_after_setup = False
        # The row whose preview was queued behind the setup, so the completion
        # handler can re-fire that exact preview without the user re-clicking.
        self._preview_row_queued_after_setup: int | None = None
        self._preflight_queued_after_setup = False
        self.progress_dialog: RenderProgressDialog | None = None
        self.preview_dialog: RenderProgressDialog | None = None
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        # EndOfMedia must never stop/delete the player from inside its own
        # mediaStatusChanged emission (QtMultimedia use-after-free); defer the
        # stop out of the handler so the backend finishes delivering first.
        self.player.mediaStatusChanged.connect(self._on_preview_playback_status)
        self.setWindowTitle("The Oracle")
        self.resize(1320, 1060)
        self.setMinimumSize(1160, 760)
        self._mark_startup("mainwindow_init_begin")
        self._ctrl_help = install_ctrl_hover_help(QApplication.instance())
        self._build_ui()
        self._build_menu()
        self._register_ctrl_help_descriptions()
        self.delete_confirm_enabled = True
        self._apply_gui_settings_payload(self._default_gui_settings_payload())
        self._mark_startup("mainwindow_init_end")
        self._write_startup_timeline()
        # Arm the deferred startup callbacks from single-shot children of the
        # window, not the context-less QTimer.singleShot helper: that helper
        # holds a strong reference to the bound method, so a window destroyed
        # before its timer fired stayed pinned alive (every MainWindow built
        # by a test that never runs an event loop leaked, and a later
        # processEvents() replayed all of their _on_gui_shown calls at once —
        # a full stylesheet restyle per window, hanging the suite). A child
        # timer dies with the window and cancels the pending call.
        self._gui_shown_timer = QTimer(self)
        self._gui_shown_timer.setSingleShot(True)
        self._gui_shown_timer.timeout.connect(self._on_gui_shown)
        self._gui_shown_timer.start()

    def _mark_startup(self, label: str) -> None:
        self._startup_marks.append((label, perf_counter() - self._startup_t0))

    def _write_startup_timeline(self) -> None:
        try:
            log_dir = self.paths.output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            payload = {"events": self._startup_marks}
            (log_dir / "gui_startup_timing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_gui_shown(self) -> None:
        self._gui_shown_wall = time()
        # Restore the remembered inference backend (and audio.cpp paths) before
        # anything else runs, so a previous Vulkan session picks up right where
        # it left off. If the persisted paths are stale, the normal
        # selection-time prerequisite check auto-starts the setup again.
        self._apply_remembered_backend()
        # Crash surface (U3, CRASH §11 steps 3–4): install the Qt message
        # handler once (fatal/critical Qt messages join the crash pipeline),
        # then the next-session review / first-run consent branch. Both are
        # consent-driven and never block startup: consent-off installs get a
        # single ask, reviewed installs get the report review, everyone else
        # gets nothing.
        from the_oracle.gui_crash import install_qt_message_handler, maybe_run_startup_flow

        install_qt_message_handler()
        self._crash_flow_timer = QTimer(self)
        self._crash_flow_timer.setSingleShot(True)
        self._crash_flow_timer.timeout.connect(self._maybe_run_crash_startup_flow)
        self._crash_flow_timer.start()
        # Restore the saved workspace layout (theme, splitters, section
        # sliders/collapses, window size) before enabling persistence, so the
        # restore itself never writes over the stored values.
        self._apply_workspace_layout()
        apply_theme(QApplication.instance(), self._current_theme)
        self._sync_theme_actions()
        # Persistence of backend/knob/layout changes is only enabled after the
        # restores above, so a restore can never overwrite the stored values
        # with widget defaults.
        self._app_settings_ready = True
        self._start_prewarm()
        # A new installation gets a guided hardware discovery and GUI tour.
        # Use a short queued delay so the first frame, theme, and tooltips are
        # fully realized before the non-modal tutorial highlights a control.
        if not self._app_settings.get("inference_wizard_completed", False) and not self._app_settings.get("inference_wizard_dismissed", False):
            self._wizard_launch_timer = QTimer(self)
            self._wizard_launch_timer.setSingleShot(True)
            self._wizard_launch_timer.timeout.connect(self._start_inference_wizard)
            # Explicit 250 ms: QTimer's default interval is 0, which would
            # launch the tutorial on the next event-loop pass — before the
            # first frame is painted (the delay the comment above exists for).
            self._wizard_launch_timer.start(250)

    def _log_action_timing(self, label: str, wall: float | None = None, extra: dict | None = None) -> None:
        try:
            log_dir = self.paths.output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            payload_path = log_dir / "gui_action_timing.json"
            existing = json.loads(payload_path.read_text()) if payload_path.exists() else []
            entry = {"label": label, "wall": wall or time()}
            if extra:
                entry.update(extra)
            existing.append(entry)
            payload_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _build_ui(self) -> None:
        root = QWidget(self)
        hbox = QHBoxLayout(root)
        hbox.setContentsMargins(0, 0, 0, 0)

        left = QWidget()
        # Explicit floor: QSplitter honors an explicit minimumWidth over the
        # layout's minimumSizeHint (which sums every section's intrinsic
        # minimum and would otherwise pin the whole window). 880 fits the
        # three settings sections at their own explicit floors.
        left.setMinimumWidth(880)
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)

        self.live_panel = LivePanel()

        controls = QGridLayout()
        self.input_path = QLineEdit()
        self.outdir_path = QLineEdit()
        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("Auto-derived from the input file; a caution appears before using the generic name")
        self._output_name_edited = False
        self.input_path.textChanged.connect(self._handle_outdir_changed)
        self.outdir_path.textChanged.connect(self._handle_outdir_changed)
        self.output_name.textEdited.connect(self._mark_output_name_edited)
        self._path_row_labels: list[QLabel | None] = [None, None]
        self._path_row_buttons: list[QPushButton | None] = [None, None]
        self._add_path_row(controls, 0, "Input", self.input_path, self._pick_input)
        self._add_path_row(controls, 1, "Output Folder", self.outdir_path, self._pick_outdir)
        self.output_name_label = QLabel("Output Filename")
        controls.addWidget(self.output_name_label, 2, 0)
        controls.addWidget(self.output_name, 2, 1, 1, 2)
        # The controls grid joins the Script & Cast section further down
        # (phase-3 section chrome), not the bare left-column layout.
        # Default input on a fresh install (or when the remembered input file
        # no longer exists): the user's pain-point test file. A remembered
        # input from a previous session wins over this default (see
        # _apply_workspace_layout).
        self._default_input_path = self._default_input_dir / "What is, reality.txt"
        if not self._default_input_path.exists():
            bundled_default = self.paths.input_dir / "What is, reality.txt"
            self._default_input_path = bundled_default
        if self._default_input_path.exists():
            self.input_path.setText(str(self._default_input_path))

        # Section chrome: every settings area is a QHSectionGroup (collapse
        # toggle + Section Size slider) laid out inside a QSplitter, so each
        # section can be resized by dragging its slider or the splitter
        # handle. Positions are persisted in the app settings file.
        shared_settings = self._build_project_settings()
        self.speaker_a = SpeakerGroup("A", self.paths.voice_dir, on_save_blend=self._save_blend_as_voice)
        self.speaker_b = SpeakerGroup("B", self.paths.voice_dir, on_save_blend=self._save_blend_as_voice)
        # Extra character voices (C..X) for audiobook casts. They are created
        # lazily when a plan detects more than two speakers; the layout holds
        # them in a scrollable column so a 24-voice cast stays usable.
        self.extra_speaker_groups: dict[str, SpeakerGroup] = {}
        self.extra_speaker_scroll = QScrollArea()
        self.extra_speaker_scroll.setWidgetResizable(True)
        self.extra_speaker_scroll.setMinimumWidth(300)
        self.extra_speaker_container = QWidget()
        self.extra_speaker_layout = QVBoxLayout(self.extra_speaker_container)
        self.extra_speaker_layout.setContentsMargins(0, 0, 0, 0)
        self.extra_speaker_scroll.setWidget(self.extra_speaker_container)
        # Wrapped in section chrome: the column never shows inline (the cast
        # dialog owns characters beyond A/B), so the whole section stays
        # hidden exactly as the bare scroll area did.
        self.extra_voices_section = QHSectionGroup("Extra Voices", collapsible=True, resizable=True)
        extra_voices_layout = QVBoxLayout(self.extra_voices_section)
        extra_voices_layout.setContentsMargins(0, 0, 0, 0)
        extra_voices_layout.addWidget(self.extra_speaker_scroll)
        self.extra_voices_section.hide()

        self._sections_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._sections_splitter.setHandleWidth(6)
        self._sections_splitter.setChildrenCollapsible(False)
        for widget in (shared_settings, self.speaker_a, self.speaker_b, self.extra_voices_section):
            self._sections_splitter.addWidget(widget)
        self._register_section("shared", shared_settings, self._sections_splitter, 0)
        self._register_section("speaker_a", self.speaker_a, self._sections_splitter, 1)
        self._register_section("speaker_b", self.speaker_b, self._sections_splitter, 2)
        self._register_section("extra_voices", self.extra_voices_section, self._sections_splitter, 3)
        self._sections_splitter.setSizes([460, 360, 360, 160])
        # Cast summary bar: the full cast lives here (count + names); the
        # dialog owns add/remove/configure, the main window mirrors it.
        cast_bar = QHBoxLayout()
        cast_bar.addWidget(QLabel("Cast:"))
        self.cast_summary_label = QLabel()
        self.cast_summary_label.setToolTip(
            "The current speaker cast. Speakers beyond A/B are managed in the cast dialog."
        )
        cast_bar.addWidget(self.cast_summary_label, 1)
        self.manage_cast_button = QPushButton("Manage cast...")
        self.manage_cast_button.setToolTip(
            "Open the cast manager: add/remove speakers, name characters, and "
            "configure each voice. One speaker left returns to monologue mode."
        )
        self.manage_cast_button.clicked.connect(self.open_cast_manager)
        cast_bar.addWidget(self.manage_cast_button)
        self._refresh_cast_bar()
        # cast_bar and the sections splitter are assembled into the Script &
        # Cast section / paths splitter further down.

        actions = QHBoxLayout()
        actions.setSpacing(12)
        actions.addStretch(1)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.prepare_project)
        self._style_action_button(self.analyze_button)
        self.render_button = QPushButton("Render FLAC")
        self.render_button.clicked.connect(self.render_project)
        self._style_action_button(self.render_button, accent=True)
        actions.addWidget(self.analyze_button)
        actions.addWidget(self.render_button)
        # The Analyze/Render row stays bare (out of any section): collapsing
        # it would hide the primary actions. It is assembled into the rest
        # container below.

        self.table = QTableWidget(0, 9)
        # Floor for the review table: splitters can shrink it, but never to
        # an unreadable sliver (the settings sections above are tall).
        self.table.setMinimumHeight(140)
        self.table.setHorizontalHeaderLabels([
            "Index",
            "Speaker",
            "Original Text",
            "Repaired Text",
            "Emotion",
            "Duration",
            "Status",
            "Preview",
            "+/-",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)

        # The review table and the status/error panel share a vertical
        # splitter; the status panel is a section with its own size slider.
        self._lower_splitter = QSplitter(Qt.Orientation.Vertical)
        self._lower_splitter.setHandleWidth(6)
        self._lower_splitter.setChildrenCollapsible(False)
        review_section = QHSectionGroup("Review", collapsible=True, resizable=True)
        review_layout = QVBoxLayout(review_section)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.addWidget(self.table)
        self._lower_splitter.addWidget(review_section)
        self._register_section("table", review_section, self._lower_splitter, 0)
        status_section = QHSectionGroup("Status / Errors", collapsible=True, resizable=True)
        status_layout = QVBoxLayout(status_section)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.error_panel = QTextEdit()
        self.error_panel.setReadOnly(True)
        self.error_panel.setMinimumHeight(70)
        self.error_panel.setPlaceholderText("Status and model errors appear here.")
        status_layout.addWidget(self.error_panel)
        self._lower_splitter.addWidget(status_section)
        self._register_section("status", status_section, self._lower_splitter, 1)
        # The section header replaces the old plain label; kept as the
        # Ctrl+hover registration target for the status area.
        self.status_label = status_section
        self._lower_splitter.setSizes([540, 180])

        # Paths strip + everything below it: a vertical splitter so the
        # Script & Cast section gets its own size slider (sliders only work
        # inside a splitter), sharing height with the settings/table stack —
        # mirroring how Status/Errors shares _lower_splitter with the table.
        self.paths_section = QHSectionGroup("Script & Cast", collapsible=True, resizable=True)
        paths_layout = QVBoxLayout(self.paths_section)
        paths_layout.setContentsMargins(0, 0, 0, 0)
        paths_layout.addLayout(controls)
        paths_layout.addLayout(cast_bar)
        self._paths_splitter = QSplitter(Qt.Orientation.Vertical)
        self._paths_splitter.setHandleWidth(6)
        self._paths_splitter.setChildrenCollapsible(False)
        self._paths_splitter.addWidget(self.paths_section)
        rest = QWidget()
        rest_layout = QVBoxLayout(rest)
        rest_layout.setContentsMargins(0, 0, 0, 0)
        rest_layout.addWidget(self._sections_splitter)
        rest_layout.addLayout(actions)
        rest_layout.addWidget(self._lower_splitter, stretch=1)
        self._paths_splitter.addWidget(rest)
        layout.addWidget(self._paths_splitter)
        self._register_section("paths", self.paths_section, self._paths_splitter, 0)

        # Live progress column: its own section so it can be resized (slider
        # or handle) and collapsed like every other section. The section
        # chrome lives in gui_chrome alongside the panel it wraps.
        live_section = build_live_section(self.live_panel)
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setHandleWidth(6)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(left)
        self._main_splitter.addWidget(live_section)
        self._register_section("live", live_section, self._main_splitter, 1)
        self._main_splitter.setSizes([1040, 280])
        hbox.addWidget(self._main_splitter)
        for splitter in (
            self._main_splitter,
            self._sections_splitter,
            self._lower_splitter,
            self._paths_splitter,
        ):
            splitter.splitterMoved.connect(lambda _pos, _index: self._persist_workspace_layout())

        self.setCentralWidget(root)
        self.outdir_path.setText(str(self._default_output_dir))
        self._refresh_language_options()
        self._refresh_reference_pickers()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        new_action = QAction("New Project", self)
        new_action.triggered.connect(self.new_project)
        open_action = QAction("Open Project", self)
        open_action.triggered.connect(self.open_project)
        save_action = QAction("Save Project", self)
        save_action.triggered.connect(self.save_project)
        save_as_action = QAction("Save Project As", self)
        save_as_action.triggered.connect(self.save_project_as)
        for action in (new_action, open_action, save_action, save_as_action):
            file_menu.addAction(action)
        file_menu.addSeparator()
        save_profile_action = QAction("Save Profile…", self)
        save_profile_action.setToolTip(
            "Save every current GUI option and slider position (shared render "
            "settings and all speaker voice settings) to a reusable profile "
            "file. Same as Settings → Save Settings..."
        )
        save_profile_action.triggered.connect(self.save_settings_profile)
        file_menu.addAction(save_profile_action)
        load_profile_action = QAction("Load Profile…", self)
        load_profile_action.setToolTip(
            "Load a saved profile file, restoring all options and slider "
            "positions. Same as Settings → Load Settings..."
        )
        load_profile_action.triggered.connect(self.load_settings_profile)
        file_menu.addAction(load_profile_action)
        file_menu.addSeparator()
        batch_fix_action = QAction("Batch Fix Input Folder...", self)
        batch_fix_action.setToolTip(
            "Scan a folder of input scripts for formatting problems the engine "
            "would misread, show one combined preview of every proposed "
            "correction, and fix all accepted files in place (each original "
            "backed up)."
        )
        batch_fix_action.triggered.connect(self._batch_fix_input_folder)
        file_menu.addAction(batch_fix_action)

        settings_menu = self.menuBar().addMenu("Settings")
        reset_defaults_action = QAction("Reset to Defaults", self)
        reset_defaults_action.triggered.connect(self.reset_settings_to_defaults)
        save_settings_action = QAction("Save Settings...", self)
        save_settings_action.triggered.connect(self.save_settings_profile)
        load_settings_action = QAction("Load Settings...", self)
        load_settings_action.triggered.connect(self.load_settings_profile)
        save_template_action = QAction("Save Current as Template...", self)
        save_template_action.triggered.connect(self.save_template_profile)
        settings_menu.addAction(reset_defaults_action)
        settings_menu.addSeparator()
        settings_menu.addAction(save_settings_action)
        settings_menu.addAction(load_settings_action)
        settings_menu.addSeparator()
        settings_menu.addAction(save_template_action)
        settings_menu.addSeparator()
        self.inference_wizard_menu = settings_menu.addMenu("Replay Setup & Tutorial")
        self.replay_full_wizard_action = QAction("Replay Entire Setup & Tutorial", self)
        self.replay_full_wizard_action.triggered.connect(lambda: self._start_inference_wizard("full", force=True))
        self.replay_discovery_wizard_action = QAction("Replay Hardware Discovery Only", self)
        self.replay_discovery_wizard_action.triggered.connect(lambda: self._start_inference_wizard("discovery", force=True))
        self.replay_main_wizard_action = QAction("Replay Main GUI Tour Only", self)
        self.replay_main_wizard_action.triggered.connect(lambda: self._start_inference_wizard("main", force=True))
        for wizard_action in (self.replay_full_wizard_action, self.replay_discovery_wizard_action, self.replay_main_wizard_action):
            self.inference_wizard_menu.addAction(wizard_action)
        self.recording_wizard_action = QAction("Replay Recording Studio Setup & Guide", self)
        self.recording_wizard_action.triggered.connect(lambda: self._start_recording_wizard(force=True))
        settings_menu.addAction(self.recording_wizard_action)
        self.output_filename_warning_action = QAction("Warn before generic output filenames", self)
        self.output_filename_warning_action.setCheckable(True)
        self.output_filename_warning_action.setChecked(self.output_filename_warning_enabled)
        self.output_filename_warning_action.setToolTip(
            "Show a caution before a render uses the automatically derived input-based "
            "filename. Re-enable this setting here after disabling it in the warning."
        )
        self.output_filename_warning_action.toggled.connect(self._on_output_filename_warning_toggled)
        settings_menu.addAction(self.output_filename_warning_action)

        self.templates_menu = settings_menu.addMenu("Load Template")
        self.templates_menu.aboutToShow.connect(self._rebuild_templates_menu)
        self.confirmation_action = QAction("Re-enable delete confirmations", self)
        self.confirmation_action.triggered.connect(self._enable_delete_confirmation)
        clear_trust_action = QAction("Forget remembered auto-fix approvals", self)
        clear_trust_action.setToolTip(
            "Files approved via 'Remember this choice' in the fix preview will "
            "prompt for review again before any automatic correction."
        )
        clear_trust_action.triggered.connect(self._clear_trusted_input_files)
        restore_backup_action = QAction("Restore most recent input-file backup", self)
        restore_backup_action.setToolTip(
            "Undo the most recent input-formatting fix by restoring the file from "
            "the timestamped backup taken before the correction. Regret a fix? "
            "This puts the original back."
        )
        restore_backup_action.triggered.connect(self._restore_most_recent_format_backup)
        settings_menu.addSeparator()
        settings_menu.addAction(self.confirmation_action)
        settings_menu.addAction(clear_trust_action)
        settings_menu.addAction(restore_backup_action)
        settings_menu.addSeparator()
        self.remember_backend_action = QAction("Remember GPU/CPU choice", self)
        self.remember_backend_action.setCheckable(True)
        self.remember_backend_action.setChecked(bool(self._app_settings.get("remember_backend", True)))
        self.remember_backend_action.setToolTip(
            "Remember which inference backend was chosen (PyTorch CPU or Vulkan "
            "GPU) plus the audiocpp_cli and Chatterbox model paths, and restore "
            "them automatically at next launch so the GPU needs no setup again. "
            "Uncheck to always start on PyTorch (CPU)."
        )
        # toggled (not triggered): PySide6's triggered binding here takes no
        # checked argument, so a real menu click would not deliver the bool to
        # the handler. toggled(bool) always does, and nothing programmatically
        # toggles the action after the setChecked above (which precedes the
        # connect), so no spurious early emission is possible.
        self.remember_backend_action.toggled.connect(self._on_remember_backend_toggled)
        settings_menu.addAction(self.remember_backend_action)
        self.download_vulkan_model_action = QAction("Download Vulkan Model...", self)
        self.download_vulkan_model_action.setToolTip(
            "Run scripts/download_audio_cpp_model.sh in the background and report "
            "the resulting ORACLE_AUDIOCPP_MODEL path to use for the Vulkan backend."
        )
        self.download_vulkan_model_action.triggered.connect(self.download_vulkan_model)
        settings_menu.addAction(self.download_vulkan_model_action)

        # Top-of-window Recording Studio: a plain menu-bar action (not a menu)
        # that opens the voice-recording window in its own dialog.
        self.recording_studio_action = QAction("Recording Studio…", self)
        self.recording_studio_action.setToolTip(
            "Open the Custom Voice Recording Studio: record a new reference "
            "voice (a 'Seashell') from a microphone with a teleprompter, then "
            "pick it for any speaker."
        )
        self.recording_studio_action.triggered.connect(self.open_recording_studio)
        self.menuBar().addAction(self.recording_studio_action)

        # Help menu: privacy/crash reports, license activation, About (U3).
        help_menu = self.menuBar().addMenu("Help")
        self.privacy_action = QAction("Privacy && Crash Reports…", self)
        self.privacy_action.setToolTip(
            "Review crash reports from last session, change crash-report consent, "
            "and open the privacy policy. Reports never leave this machine unless "
            "you share one yourself."
        )
        self.privacy_action.triggered.connect(self._open_crash_privacy_dialog)
        help_menu.addAction(self.privacy_action)
        self.activate_license_action = QAction("Activate License…", self)
        self.activate_license_action.setToolTip(
            "Paste your license token (offline activation — no server is contacted)."
        )
        self.activate_license_action.triggered.connect(self._open_license_activation)
        help_menu.addAction(self.activate_license_action)
        self.about_action = QAction("About The Oracle…", self)
        self.about_action.setToolTip("Which edition this install runs, and the privacy posture in one glance.")
        self.about_action.triggered.connect(self._open_about_panel)
        help_menu.addAction(self.about_action)

        # Keep references for Ctrl+hover help registration (see
        # _register_ctrl_help_descriptions).
        self.new_action = new_action
        self.open_action = open_action
        self.save_action = save_action
        self.save_as_action = save_as_action
        self.reset_defaults_action = reset_defaults_action
        self.save_settings_action = save_settings_action
        self.load_settings_action = load_settings_action
        self.save_template_action = save_template_action
        self.save_profile_action = save_profile_action
        self.load_profile_action = load_profile_action

        # Theme menu: six built-in looks; the choice is persisted immediately.
        theme_menu = self.menuBar().addMenu("Theme")
        self._theme_actions: dict[str, QAction] = {}
        for key in THEMES:
            tokens = THEMES[key]
            action = QAction(tokens.name, self)
            action.setCheckable(True)
            action.setToolTip(f"{tokens.description} Contrast-checked for legibility.")
            action.triggered.connect(lambda _checked=False, current=key: self._apply_theme_selection(current))
            theme_menu.addAction(action)
            self._theme_actions[key] = action

        # Workspace persistence: the input file joins backend choice, theme,
        # splitters, section states, and the configured default folders in the
        # app settings file.  textEdited deliberately excludes programmatic
        # restore/default changes during startup.
        self.input_path.textEdited.connect(self._remember_input_file_edit)
        self.outdir_path.textEdited.connect(lambda: self._persist_workspace_layout())
        self.output_name.textEdited.connect(lambda: self._persist_workspace_layout())

    # --------------------
    # Themes
    # --------------------
    def _apply_theme_selection(self, key: str) -> None:
        """Switch to theme ``key``, persist it, and sync the checkmarks."""
        if key not in THEMES:
            return
        self._current_theme = apply_theme(QApplication.instance(), key)
        self._persist_workspace_layout()
        self._sync_theme_actions()
        tokens = THEMES[self._current_theme]
        self.error_panel.append(f"Theme: {tokens.name} - {tokens.description}")

    def _sync_theme_actions(self) -> None:
        for key, action in self._theme_actions.items():
            action.blockSignals(True)
            action.setChecked(key == self._current_theme)
            action.blockSignals(False)

    def _register_ctrl_help_descriptions(self) -> None:
        """Register Ctrl+hover help text for the main controls.

        Holding the left Control key while hovering a control pops up a short
        description of what it does and how to use it (see
        the_oracle.gui_tooltips). Buttons, fields, spin boxes, backend knobs,
        speaker controls, and menu actions are covered; unregistered widgets
        fall back to their regular Qt tooltip if one is set.
        """
        ctrl_help = self._ctrl_help
        if ctrl_help is None:
            return
        ctrl_help.register_many([
            (
                self.input_path,
                "Input script or transcript file to convert into dialogue audio. "
                "Click Browse or type a path; Analyze then reads, repairs, and "
                "attributes the lines.",
            ),
            (
                self.outdir_path,
                "Output folder for the rendered FLAC (plus stems and optional "
                "SRT). Defaults to the repo's Output/ folder; typing a new "
                "path creates it.",
            ),
            (
                self.output_name,
                "Filename for the final render, without the extension. Leave "
                "blank to auto-derive from the input file name.",
            ),
            (
                self.analyze_button,
                "Analyze the input: repair text, attribute lines to speakers "
                "A and B, detect emotions, and fill the review table. No "
                "audio is rendered yet.",
            ),
            (
                self.render_button,
                "Render the full dialogue to FLAC: synthesize every utterance "
                "with the selected inference backend, assemble the two "
                "speakers, and write the output file.",
            ),
            (
                self.table,
                "Review table, one row per utterance: index, speaker, "
                "original/repaired text, emotion (editable), duration, "
                "status, preview, and stem on/off.",
            ),
            (
                self.error_panel,
                "Status and error log. Render progress, Vulkan preflight "
                "reports, model downloads, and failures are appended here.",
            ),
        ])
        ctrl_help.register_many([
            (
                self.variant_combo,
                "Chatterbox model variant: standard (default), multilingual "
                "(per-speaker language), or turbo (faster; PyTorch-only, so "
                "it disables the Vulkan backend).",
            ),
            (
                self.correction_mode_combo,
                "How aggressively the text-repair pass fixes grammar, "
                "spelling, punctuation, and directives before synthesis.",
            ),
            (
                self.loudness_combo,
                "Loudness normalization applied to the final render: off, "
                "light, or medium.",
            ),
            (
                self.crossfade_spin,
                "Crossfade in milliseconds between consecutive utterances so "
                "speaker turns blend smoothly.",
            ),
            (
                self.export_srt_check,
                "Also write a .srt subtitle file next to the rendered FLAC, "
                "one cue per utterance.",
            ),
            (
                self.inference_backend_combo,
                "Inference backend: PyTorch runs the Chatterbox model through the "
                "installed PyTorch runtime, while Vulkan delegates to audio.cpp. "
                "The separate PyTorch Device picker chooses CPU/DRAM or a usable "
                "CUDA/NVIDIA card when PyTorch has CUDA support.",
            ),
            (
                self.pytorch_device_combo,
                "PyTorch execution device: CPU / system DRAM, or a suitable "
                "CUDA / NVIDIA GPU. Disabled CUDA entries are detected hardware "
                "that is missing a CUDA-enabled PyTorch runtime or does not meet "
                "the 4 GiB Chatterbox VRAM floor. " + cuda_reason(),
            ),
            (
                self.test_vulkan_button,
                "Run a quick audio.cpp preflight (binary + model + device "
                "list) and report which GPU the Vulkan backend would use, "
                "before rendering.",
            ),
            (
                self.vulkan_prerequisite_warning,
                "Inline warning shown when the Vulkan backend is selected but "
                "audiocpp_cli or the Chatterbox model is not configured yet.",
            ),
            (
                self.audio_cpp_device_combo,
                "Vulkan device passed to audio.cpp (--device). Auto lets "
                "audio.cpp pick; the dropdown lists detected devices. "
                "Equivalent to ORACLE_AUDIOCPP_DEVICE.",
            ),
            (
                self.audio_cpp_device_label,
                "Read-only summary of the Vulkan devices audio.cpp detected; "
                "the dropdown above is populated from this list.",
            ),
            (
                self.audio_cpp_threads_spin,
                "Thread count passed to audio.cpp (--threads). Default lets "
                "audio.cpp decide. Equivalent to ORACLE_AUDIOCPP_THREADS.",
            ),
            (
                self.audio_cpp_timeout_spin,
                "Per-synthesis timeout in seconds for audio.cpp. Default (0) "
                "uses audio.cpp's 600 s. Equivalent to ORACLE_AUDIOCPP_TIMEOUT.",
            ),
            (
                self.audio_cpp_max_batch_spin,
                "Maximum cache-missing stems per audio.cpp --request-sequence "
                "subprocess. Default (0) uses the engine's 32-request cap. "
                "Equivalent to ORACLE_AUDIOCPP_MAX_BATCH.",
            ),
        ])
        for group in self._all_speaker_groups().values():
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
            (self.new_action, "Start a new project: clears the review table and plan while keeping the current shared and speaker settings."),
            (self.open_action, "Load a saved project manifest, restoring the plan, review table, and settings."),
            (self.save_action, "Save the current project to its manifest file."),
            (self.save_as_action, "Save the current project to a new manifest file."),
            (self.reset_defaults_action, "Restore every setting to its built-in default."),
            (self.save_settings_action, "Export the current GUI settings (shared + speakers) to a profile file."),
            (self.load_settings_action, "Apply a previously saved GUI settings profile."),
            (self.save_template_action, "Save the current settings as a named template for the Load Template menu."),
            (self.confirmation_action, "Re-enable the delete-confirmation prompt for removing review-table rows."),
            (self.remember_backend_action, "Remember which inference backend was chosen (PyTorch CPU or Vulkan GPU) plus the audiocpp_cli and Chatterbox model paths, and restore them automatically at next launch so the GPU needs no setup again. Uncheck to always start on PyTorch (CPU)."),
            (self.download_vulkan_model_action, "Fetch the audio.cpp Chatterbox model in the background and report the ORACLE_AUDIOCPP_MODEL path to use."),
            (self.replay_full_wizard_action, "Replay the complete hardware discovery and dependency-ordered GUI tutorial."),
            (self.replay_discovery_wizard_action, "Replay only the hardware and inference availability discovery step."),
            (self.replay_main_wizard_action, "Replay only the main-page GUI feature tour, starting after hardware discovery."),
            (self.recording_wizard_action, "Replay the Recording Studio setup guide, including microphone, folder, naming, and speaking technique."),
            (self.output_filename_warning_action, "Show or hide the caution before a render uses its automatically derived generic output filename."),
        ])
        # The row labels and Browse buttons share their field's description,
        # so hovering the *name* of a control works too (the user asked for
        # "names, selections, buttons, toggles, or other areas of interest").
        ctrl_help.register(self._path_row_labels[0], ctrl_help.description_for(self.input_path))
        ctrl_help.register(self._path_row_buttons[0], "Open a file picker to choose the input script or transcript.")
        ctrl_help.register(self._path_row_labels[1], ctrl_help.description_for(self.outdir_path))
        ctrl_help.register(self._path_row_buttons[1], "Open a folder picker to choose where the rendered FLAC is written.")
        ctrl_help.register(self.output_name_label, ctrl_help.description_for(self.output_name))
        ctrl_help.register(self.status_label, ctrl_help.description_for(self.error_panel))
        # Form row labels (e.g. "Model Variant", "CFG Weight") get the same
        # help as their field.
        ctrl_help.register_form_labels(self._project_settings_form, [
            self.variant_combo,
            self.correction_mode_combo,
            self.loudness_combo,
            self.crossfade_spin,
            self.inference_backend_combo,
            self.audio_cpp_device_combo,
            self.audio_cpp_threads_spin,
            self.audio_cpp_timeout_spin,
            self.audio_cpp_max_batch_spin,
            self.pytorch_device_combo,
        ])
        # The Inference Backend row's field is a layout, so its label is not
        # covered by register_form_labels; register it explicitly.
        backend_label = self._project_settings_form.labelForField(self._inference_backend_row)
        if backend_label is not None:
            ctrl_help.register(backend_label, ctrl_help.description_for(self.inference_backend_combo))
        for group in self._all_speaker_groups().values():
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
            self.menuBar(),
            "Menu bar: project, settings, recording, theme, and help menus. "
            "Hold Ctrl and hover a menu to read what each entry does.",
        )
        menubar_actions = self.menuBar().actions()
        if len(menubar_actions) >= 1:
            ctrl_help.register_action(menubar_actions[0], "Project management: new, open, save, save-as.")
        if len(menubar_actions) >= 2:
            ctrl_help.register_action(menubar_actions[1], "GUI settings: profiles, templates, and Vulkan backend setup.")
        if len(menubar_actions) >= 3:
            ctrl_help.register_action(menubar_actions[2], self.recording_studio_action.toolTip())

    def _build_project_settings(self) -> QGroupBox:
        box = QHSectionGroup("Shared Render Settings", collapsible=True, resizable=True)
        form = QFormLayout(box)
        self._project_settings_form = form  # kept so Ctrl+hover help can register row labels
        self.variant_combo = QComboBox()
        self.variant_combo.addItems(list(SUPPORTED_VARIANTS))
        self.variant_combo.currentTextChanged.connect(self._refresh_language_options)
        self.correction_mode_combo = QComboBox()
        for label, value in CORRECTION_MODE_OPTIONS:
            self.correction_mode_combo.addItem(label, value)
        self.correction_mode_combo.setToolTip(
            "How much the text is cleaned before synthesis. Verbatim (no "
            "changes) passes the source text through exactly as written -- no "
            "spelling, grammar, or punctuation edits -- so narration is always "
            "faithful to the file. The TTS engine still applies its own "
            "required text normalization internally (identical on both "
            "backends)."
        )
        self._set_correction_mode(RenderSettings().correction_mode)
        self.loudness_combo = QComboBox()
        self.loudness_combo.addItems(["off", "light", "medium"])
        self.loudness_combo.setCurrentText(RenderSettings().loudness_preset)
        self.loudness_combo.setToolTip(
            "Post-render loudness normalization of the finished FLAC and its "
            "stems. Off writes the raw mix; Light applies a gentle gain "
            "match; Medium applies stronger compression plus gain toward a "
            "broadcast-ish level. Applied once to the assembled audio, after "
            "all synthesis and crossfading."
        )
        self.crossfade_spin = QSpinBox()
        self.crossfade_spin.setRange(0, 500)
        self.crossfade_spin.setValue(RenderSettings().crossfade_ms)
        self.crossfade_spin.setToolTip(
            "Equal-power crossfade (0-500 ms) applied where two synthesized "
            "stems are spliced together in the final render. Higher values "
            "soften the join between utterances but can smear word edges; 20 "
            "ms is the default. This is a splice join, not a pause - silence "
            "between turns is controlled per speaker by 'Breath After This "
            "Speaker'."
        )
        self.export_srt_check = QCheckBox("Export SRT subtitles")
        self.export_srt_check.setToolTip("Write a .srt subtitle file next to the rendered FLAC, one cue per utterance.")
        self.monologue_check = QCheckBox("Monologue (single narrator voice)")
        self.monologue_check.setToolTip(
            "Render the entire input in one narrator voice (Speaker A), ignoring "
            "per-line attribution. Use this to read a book aloud as a single "
            "narrator instead of a cast of characters."
        )
        self.inference_backend_combo = QComboBox()
        self.inference_backend_combo.addItem("PyTorch", "pytorch")
        self.inference_backend_combo.addItem("Vulkan (audio.cpp)", "vulkan")
        self.inference_backend_combo.setCurrentIndex(0)
        self.pytorch_device_combo = QComboBox()
        self.pytorch_device_combo.addItem("CPU / system DRAM", "cpu")
        for device in self._cuda_devices:
            self.pytorch_device_combo.addItem(device.label, f"cuda:{device.index}")
            item = self.pytorch_device_combo.model().item(self.pytorch_device_combo.count() - 1)
            if item is not None and (not device.torch_available or not device.suitable):
                item.setEnabled(False)
        has_usable_cuda = any(device.torch_available and device.suitable for device in self._cuda_devices)
        if not has_usable_cuda:
            self.pytorch_device_combo.addItem("CUDA unavailable (see tooltip)", "cuda-unavailable")
            item = self.pytorch_device_combo.model().item(self.pytorch_device_combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
        self.pytorch_device_combo.setToolTip(
            "PyTorch execution device. CPU uses system DRAM. CUDA uses a suitable "
            "NVIDIA GPU when the installed PyTorch runtime exposes it. "
            + cuda_reason()
        )
        self.inference_backend_combo.setToolTip(
            "Inference backend for render and preview. PyTorch is the in-process "
            "Chatterbox path and uses the separate PyTorch Device picker for CPU or "
            "CUDA. Vulkan (audio.cpp) is the alternate GPU path: selecting "
            "it automatically builds audiocpp_cli and downloads the Chatterbox model "
            "if missing, then sets the env vars for the session (see README "
            "'Vulkan Backend')."
        )
        self.test_vulkan_button = QPushButton("Test Vulkan Backend")
        self.test_vulkan_button.setToolTip(
            "Run a quick audio.cpp preflight (binary + model + --list-devices) and "
            "report which GPU the Vulkan backend would use, before rendering."
        )
        self.test_vulkan_button.clicked.connect(self.test_vulkan_backend)
        self.test_vulkan_button.setEnabled(False)
        self.vulkan_prerequisite_warning = QLabel("")
        self.vulkan_prerequisite_warning.setWordWrap(True)
        # Theme-aware warning color (the themed stylesheet styles this via
        # QLabel#warning, so no hardcoded amber can fight a dark theme).
        self.vulkan_prerequisite_warning.setObjectName("warning")
        self.vulkan_prerequisite_warning.hide()
        self.audio_cpp_device_combo = QComboBox()
        self.audio_cpp_device_combo.addItem("Auto (audio.cpp default)", None)
        self.audio_cpp_device_combo.setToolTip(
            "Vulkan device passed to audio.cpp as --device <N> on multi-GPU "
            "machines. Auto lets audio.cpp pick; the dropdown is populated from "
            "'audiocpp_cli --backend vulkan --list-devices' (or the doctor). "
            "The value is equivalent to ORACLE_AUDIOCPP_DEVICE but needs no "
            "environment variable."
        )
        self.audio_cpp_device_label = QLabel("")
        self.audio_cpp_device_label.setWordWrap(True)
        self.audio_cpp_device_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.audio_cpp_device_label.setToolTip(
            "The Vulkan devices audio.cpp detects (from audiocpp_cli --list-devices). "
            "The dropdown lists them by name; picking one is equivalent to "
            "ORACLE_AUDIOCPP_DEVICE."
        )
        self.audio_cpp_threads_spin = QSpinBox()
        self.audio_cpp_threads_spin.setRange(0, 128)
        self.audio_cpp_threads_spin.setValue(0)
        self.audio_cpp_threads_spin.setSpecialValueText("Default (audio.cpp's own)")
        self.audio_cpp_threads_spin.setToolTip(
            "Thread count passed to audio.cpp as --threads <N>. Default lets "
            "audio.cpp decide; the value is equivalent to ORACLE_AUDIOCPP_THREADS "
            "but needs no environment variable."
        )
        self.audio_cpp_timeout_spin = QSpinBox()
        self.audio_cpp_timeout_spin.setRange(0, 3600)
        self.audio_cpp_timeout_spin.setValue(0)
        self.audio_cpp_timeout_spin.setSpecialValueText("Default (600s)")
        self.audio_cpp_timeout_spin.setToolTip(
            "Per-synthesis timeout in seconds passed to audio.cpp. Default (0) uses "
            "audio.cpp's 600s; the value is equivalent to ORACLE_AUDIOCPP_TIMEOUT "
            "but needs no environment variable."
        )
        self.audio_cpp_max_batch_spin = QSpinBox()
        self.audio_cpp_max_batch_spin.setRange(0, 1024)
        self.audio_cpp_max_batch_spin.setValue(0)
        self.audio_cpp_max_batch_spin.setSpecialValueText("Default (32)")
        self.audio_cpp_max_batch_spin.setToolTip(
            "Maximum cache-missing stems per audio.cpp --request-sequence subprocess. "
            "Default (0) uses the engine's 32-request cap; the value is equivalent "
            "to ORACLE_AUDIOCPP_MAX_BATCH but needs no environment variable."
        )
        self.variant_combo.currentTextChanged.connect(self._refresh_inference_backend_options)
        self.inference_backend_combo.currentIndexChanged.connect(self._refresh_audio_cpp_knob_options)
        self.pytorch_device_combo.currentIndexChanged.connect(self._persist_remembered_settings)
        # Keep the remembered backend choice and Vulkan knobs in sync with the
        # widgets so the next launch restores exactly what the user last had
        # selected. The handlers are gated on _app_settings_ready, so the
        # initial default apply (and the launch-time restore itself) never
        # writes over the stored values.
        self.inference_backend_combo.currentIndexChanged.connect(self._persist_remembered_settings)
        self.audio_cpp_device_combo.currentIndexChanged.connect(self._persist_remembered_settings)
        self.audio_cpp_threads_spin.valueChanged.connect(self._persist_remembered_settings)
        self.audio_cpp_timeout_spin.valueChanged.connect(self._persist_remembered_settings)
        self.audio_cpp_max_batch_spin.valueChanged.connect(self._persist_remembered_settings)
        form.addRow("Model Variant", self.variant_combo)
        backend_row = QHBoxLayout()
        backend_row.addWidget(self.inference_backend_combo, 1)
        backend_row.addWidget(self.test_vulkan_button, 0)
        form.addRow("Inference Backend", backend_row)
        form.addRow("PyTorch Device", self.pytorch_device_combo)
        # The row's field is a layout, so labelForField() must be given the
        # layout (not the combo) to find the "Inference Backend" label; kept
        # for Ctrl+hover label registration.
        self._inference_backend_row = backend_row
        form.addRow("", self.vulkan_prerequisite_warning)
        form.addRow("Vulkan Device", self.audio_cpp_device_combo)
        form.addRow("", self.audio_cpp_device_label)
        form.addRow("Vulkan Threads", self.audio_cpp_threads_spin)
        form.addRow("Vulkan Timeout (s)", self.audio_cpp_timeout_spin)
        form.addRow("Vulkan Max Batch", self.audio_cpp_max_batch_spin)
        form.addRow("Correction Mode", self.correction_mode_combo)
        form.addRow("Loudness", self.loudness_combo)
        form.addRow("Crossfade (ms)", self.crossfade_spin)
        form.addRow("", self.export_srt_check)
        form.addRow("", self.monologue_check)
        return box

    def _set_correction_mode(self, value: str) -> None:
        normalized = normalize_correction_mode(value)
        idx = self.correction_mode_combo.findData(normalized)
        if idx < 0:
            idx = self.correction_mode_combo.findData(normalize_correction_mode("moderate"))
        if idx >= 0:
            self.correction_mode_combo.setCurrentIndex(idx)

    def _add_path_row(self, layout: QGridLayout, row: int, label: str, field: QLineEdit, callback) -> None:
        label_widget = QLabel(label)
        button = QPushButton("Browse")
        button.clicked.connect(callback)
        layout.addWidget(label_widget, row, 0)
        layout.addWidget(field, row, 1)
        layout.addWidget(button, row, 2)
        # Keep references so Ctrl+hover help can describe the row label and
        # its Browse button (the user asked for hovering "names" too).
        self._path_row_labels[row] = label_widget
        self._path_row_buttons[row] = button

    def _style_action_button(self, button: QPushButton, accent: bool = False) -> None:
        button.setMinimumHeight(48)
        button.setMinimumWidth(170 if not accent else 210)
        # Theme classes, not hardcoded colors: the active theme styles
        # QPushButton[accent="true"] and QPushButton.action itself.
        if accent:
            button.setProperty("accent", True)
        else:
            button.setProperty("buttonRole", "action")

    def _mark_output_name_edited(self) -> None:
        self._output_name_edited = True
        self._persist_workspace_layout()

    def _on_output_filename_warning_toggled(self, checked: bool) -> None:
        self.output_filename_warning_enabled = bool(checked)
        self._app_settings["output_filename_warning"] = bool(checked)
        if self._app_settings_ready:
            self._persist_workspace_layout()
        else:
            try:
                save_app_settings(self._app_settings)
            except Exception as exc:
                self.error_panel.append(f"Could not persist output filename warning preference: {exc}")

    def _remember_input_file_edit(self) -> None:
        value = self.input_path.text().strip()
        if value:
            self._app_settings["last_input_file"] = value
        if self._app_settings_ready:
            self._persist_workspace_layout()
        else:
            try:
                save_app_settings(self._app_settings)
            except Exception as exc:
                self.error_panel.append(f"Could not persist last input file: {exc}")

    def _pick_input(self) -> None:
        current_text = self.input_path.text().strip()

        if not current_text:
            start_dir = self._default_input_dir
        else:
            current_input = Path(current_text).expanduser()
            start_dir = current_input.parent if current_input.exists() else self._default_input_dir
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Input",
            str(start_dir),
            "Dialogue Scripts (*.txt *.md);;Subtitles (*.srt *.vtt);;All Files (*)",
        )
        if path:
            # A picked subtitle file is converted up front (sibling script,
            # subtitle untouched) and the *script* goes into the field, so
            # everything downstream sees canonical dialogue.
            path = self._convert_subtitle_input(path)
            if not path:
                return
            self.input_path.setText(path)
            # A picked file becomes the remembered default for future sessions
            # (settled before setText's textChanged side effects finish).
            self._app_settings["last_input_file"] = path.strip()
            if self._app_settings_ready:
                self._persist_workspace_layout()

    def _convert_subtitle_input(self, path: str) -> str | None:
        """Convert a picked ``.srt``/``.vtt`` file to its dialogue script.

        Non-subtitle paths pass through unchanged. On success the sibling
        script's path is returned and a status line explains the
        conversion; an existing script from a previous run is reused. The
        convert-not-overwrite policy is ``srt_ingest.
        ensure_subtitle_script`` -- the same owner the CLI render path
        uses -- so the two surfaces cannot convert differently; this
        wrapper only renders the outcome in the status panel.
        """
        from the_oracle.srt_ingest import ensure_subtitle_script

        file_path = Path(path)
        if file_path.suffix.lower() not in (".srt", ".vtt"):
            return path
        try:
            script_path, outcome = ensure_subtitle_script(path)
        except (OSError, ValueError) as exc:
            self.error_panel.append(f"Could not convert {file_path.name}: {exc}")
            return None
        if outcome is None:
            self.error_panel.append(
                f"{file_path.name} does not contain valid subtitle cues; "
                "loading it as-is."
            )
            return path
        if outcome == "reused":
            self.error_panel.append(
                f"Reusing previously converted script {Path(script_path).name}; "
                "the subtitle file itself was not modified."
            )
        else:
            self.error_panel.append(
                f"Converted {file_path.name} into {Path(script_path).name}; "
                "the subtitle file itself was not modified."
            )
        return script_path

    def _handle_outdir_changed(self) -> None:
        # Typing in the output-folder field must not touch the filesystem:
        # no directory is created here. The folder is created lazily (with
        # errors surfaced) when a render or analysis actually needs it.
        if not self.output_name.text().strip():
            default_name = default_output_filename(self.input_path.text() or "")
            if not self._output_name_edited:
                self.output_name.setText(default_name)

    def _ensure_outdir_exists(self, path: str | Path) -> Path:
        """Create the output folder lazily, right before it is needed.

        Failures are logged and raised so the caller (render/analysis) can
        report them, instead of silently leaving the write to fail later.
        """
        folder = Path(path or self.paths.output_dir).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            message = f"Could not create output folder {folder}: {exc}"
            self.error_panel.append(message)
            raise OSError(message) from exc
        return folder

    def _pick_outdir(self) -> None:
        current_outdir = Path(self.outdir_path.text()).expanduser()
        start_dir = current_outdir if current_outdir.exists() else self._default_output_dir
        path = QFileDialog.getExistingDirectory(self, "Choose Output Directory", str(start_dir))
        if path:
            self.outdir_path.setText(path)

    def _refresh_language_options(self) -> None:
        variant = self.variant_combo.currentText() if hasattr(self, "variant_combo") else "standard"
        languages = {"en": "English"} if variant != "multilingual" else ChatterboxEngine(variant).supported_languages()
        is_multilingual = variant == "multilingual"
        for group in self._all_speaker_groups().values():
            group.set_language_options(languages, is_multilingual)

    def _refresh_inference_backend_options(self) -> None:
        """Disable the Vulkan backend option when the turbo variant is selected.

        The turbo Chatterbox variant is PyTorch-only; the Vulkan (audio.cpp)
        backend rejects it with a clear error. Blocking it here keeps the GUI
        from offering an unusable combination.
        """
        if not hasattr(self, "inference_backend_combo"):
            return
        is_turbo = self.variant_combo.currentText() == "turbo"
        vulkan_index = self.inference_backend_combo.findData("vulkan")
        if vulkan_index < 0:
            return
        self.inference_backend_combo.model().item(vulkan_index).setEnabled(not is_turbo)
        if is_turbo and self.inference_backend_combo.currentData() == "vulkan":
            self.inference_backend_combo.setCurrentIndex(self.inference_backend_combo.findData("pytorch"))
        self._refresh_audio_cpp_knob_options()
        self._refresh_vulkan_preflight_button()

    def _refresh_audio_cpp_knob_options(self) -> None:
        """Enable the Vulkan device/threads/timeout knobs only when the Vulkan
        backend is selected; they are ignored (and disabled) on the PyTorch path.
        Selecting Vulkan also kicks off the background device probe that
        populates the device list under the picker, and surfaces a warning if
        the backend's prerequisites are not configured yet."""
        if not hasattr(self, "audio_cpp_device_combo"):
            return
        is_vulkan = self.inference_backend_combo.currentData() == "vulkan"
        self.audio_cpp_device_combo.setEnabled(is_vulkan)
        self.audio_cpp_threads_spin.setEnabled(is_vulkan)
        self.audio_cpp_timeout_spin.setEnabled(is_vulkan)
        self.audio_cpp_max_batch_spin.setEnabled(is_vulkan)
        self.audio_cpp_device_label.setEnabled(is_vulkan)
        # Keep the CPU choice available even when no suitable CUDA device is
        # present; only the unavailable CUDA rows are disabled. Vulkan uses a
        # separate backend picker and temporarily disables this control.
        self.pytorch_device_combo.setEnabled(not is_vulkan)
        if is_vulkan:
            self._start_vulkan_device_probe()
            self._refresh_vulkan_prerequisite_warning()
        else:
            self.vulkan_prerequisite_warning.hide()
        self._refresh_vulkan_preflight_button()

    def _refresh_vulkan_preflight_button(self) -> None:
        """Enable the Test Vulkan Backend button whenever the Vulkan option is
        usable (turbo is PyTorch-only) and no preflight is already running.
        The button works from either backend so setup can be validated before
        switching."""
        if not hasattr(self, "test_vulkan_button"):
            return
        vulkan_index = self.inference_backend_combo.findData("vulkan")
        selectable = (
            vulkan_index >= 0
            and self.inference_backend_combo.model().item(vulkan_index).isEnabled()
            and self._vulkan_preflight_thread is None
        )
        self.test_vulkan_button.setEnabled(selectable)

    def _refresh_vulkan_prerequisite_warning(self) -> None:
        """Auto-start the CPU→GPU switch when the Vulkan backend can't run yet.

        Instead of only warning, this kicks off the background setup (build
        audiocpp_cli / download the model) once per selection, showing live
        progress in the warning label. Once setup finishes the warning clears
        and any queued render/preview proceeds on its own. The attempt guard is
        cleared on failure so an explicit Render/Preview click (the retry
        trigger) or a fresh selection starts a new attempt rather than silently
        failing forever."""
        missing = _vulkan_prerequisite_missing()
        if not missing:
            self.vulkan_prerequisite_warning.hide()
            return
        if self._vulkan_setup_thread is None and not self._vulkan_setup_attempted:
            self.vulkan_prerequisite_warning.setText(
                "Vulkan backend selected: setting it up automatically (building "
                "audiocpp_cli and/or downloading the Chatterbox model). "
                "Progress appears here; rendering/preview will continue by itself "
                "once setup finishes."
            )
            self.vulkan_prerequisite_warning.show()
            self._start_vulkan_setup()
            return
        # A setup already ran (or is running); surface the current state rather
        # than a stale "missing" warning.
        if self._vulkan_setup_thread is not None:
            self.vulkan_prerequisite_warning.setText(
                "Vulkan backend setup is still running; rendering/preview will "
                "continue by itself once it finishes."
            )
            self.vulkan_prerequisite_warning.show()

    def _start_vulkan_setup(self) -> None:
        """Launch the background Vulkan backend setup (build + model download)."""
        if self._vulkan_setup_thread is not None:
            return
        self._vulkan_setup_attempted = True
        self.error_panel.append("Starting automatic Vulkan backend setup (build + model download)...")
        # Parented to the window; on close the subprocess is cancelled and the
        # thread waited on so a running QThread is never destroyed.
        thread = VulkanSetupThread(self.repo_root, self)
        thread.progress.connect(self._handle_vulkan_setup_progress)
        thread.completed.connect(self._handle_vulkan_setup_completed)
        thread.failed.connect(self._handle_vulkan_setup_failed)
        thread.finished.connect(self._cleanup_vulkan_setup_thread)
        self._vulkan_setup_thread = thread
        thread.start()

    def _handle_vulkan_setup_progress(self, line: str) -> None:
        """Stream one setup output line into the warning label."""
        if not line.strip():
            return
        self.vulkan_prerequisite_warning.setText(f"Vulkan backend setup: {line.strip()[-140:]}")
        self.vulkan_prerequisite_warning.show()

    def _handle_vulkan_setup_completed(self, result) -> None:
        """Setup finished: env vars are set for the session, so clear the
        warning, re-probe devices (the binary now exists), and continue any
        queued render/preview."""
        self.error_panel.append("Vulkan backend setup complete.")
        for msg in result.messages:
            self.error_panel.append(f"  {msg}")
        # Remember the resolved audio.cpp paths for future sessions so a later
        # launch can restore the GPU stack without re-running setup. getattr
        # tolerates result objects that predate the binary/model fields (tests).
        self._persist_remembered_settings(
            binary=getattr(result, "binary", None),
            model=getattr(result, "model", None),
        )
        # Prerequisites are now met; clear the attempt guard so a later change
        # that makes them stale again (e.g. the model file deleted) re-triggers
        # the automatic setup instead of silently skipping it.
        self._vulkan_setup_attempted = False
        self._refresh_vulkan_prerequisite_warning()
        if self._vulkan_devices_probed:
            # The binary now exists, so the earlier failed probe may be stale.
            self._vulkan_devices_probed = False
            self._start_vulkan_device_probe()
        if self._render_queued_after_setup:
            self._render_queued_after_setup = False
            # A full render subsumes a queued single-row preview (the two
            # cannot run concurrently), so drop the preview queue instead of
            # leaving it to fire later — no stale state lingers.
            self._preview_queued_after_setup = False
            self._preview_row_queued_after_setup = None
            self.render_project()
        elif self._preview_queued_after_setup:
            # Auto-fire the exact preview the user asked for while setup was
            # running — re-entering preview_utterance starts the worker and
            # shows the preview progress dialog, no second click needed.
            row = self._preview_row_queued_after_setup
            self._preview_queued_after_setup = False
            self._preview_row_queued_after_setup = None
            if row is not None and self.plan is not None and 0 <= row < len(self.plan.utterances):
                self.error_panel.append("Vulkan backend ready: starting the queued preview...")
                self.preview_utterance(row)
            else:
                # The plan/row changed while setup ran (e.g. re-analyzed);
                # never crash on a stale row — just ask for a fresh click.
                self.error_panel.append("Vulkan backend ready: select a row and preview again.")
        if self._preflight_queued_after_setup:
            # A Test Vulkan Backend click arrived while setup was running;
            # prerequisites are now met, so fire the queued preflight on its
            # own (the report appears in the status panel + dialog). Note: if
            # a render/preview was queued at the same time, both fire here —
            # the preflight dialog may appear mid-render, which is acceptable
            # (the user asked for both actions). The preflight also runs even
            # if the user switched back to PyTorch meanwhile; it validates the
            # Vulkan setup regardless of the current selection.
            self._preflight_queued_after_setup = False
            self.test_vulkan_backend()

    def _handle_vulkan_setup_failed(self, message: str) -> None:
        """Surface the failure visibly with the manual fallback commands."""
        self._render_queued_after_setup = False
        self._preview_queued_after_setup = False
        self._preview_row_queued_after_setup = None
        self._preflight_queued_after_setup = False
        # Allow a later explicit Render/Preview click to retry setup; a failed
        # attempt is user-visible (not silently looped), and the user's next
        # action is the retry trigger.
        self._vulkan_setup_attempted = False
        self.error_panel.append(f"Vulkan backend setup failed: {message}")
        self.vulkan_prerequisite_warning.setText(
            "Vulkan backend setup failed: " + message.splitlines()[0][:200]
            + " See the error panel and README 'Vulkan Backend (audio.cpp)'; "
            "run scripts/build_audio_cpp.sh and scripts/download_audio_cpp_model.sh "
            "manually, or switch the Inference Backend back to PyTorch."
        )
        self.vulkan_prerequisite_warning.show()

    def _cleanup_vulkan_setup_thread(self) -> None:
        if self._vulkan_setup_thread is not None:
            self._vulkan_setup_thread.deleteLater()
            self._vulkan_setup_thread = None
        self._refresh_vulkan_preflight_button()

    def _start_vulkan_device_probe(self) -> None:
        """Probe audio.cpp's Vulkan devices once, off the UI thread."""
        if self._vulkan_devices_probed or self._vulkan_probe_thread is not None:
            return
        self.audio_cpp_device_label.setText("Probing audio.cpp devices...")
        # Parented to the window so a close while probing can wait() on it
        # instead of destroying a running QThread (which Qt aborts on).
        thread = VulkanDeviceProbeThread(self)
        thread.devices.connect(self._handle_vulkan_devices)
        thread.failed.connect(self._handle_vulkan_probe_failed)
        thread.finished.connect(self._cleanup_vulkan_probe_thread)
        self._vulkan_probe_thread = thread
        thread.start()

    def _repopulate_audio_cpp_device_combo(self, devices: list) -> None:
        """Rebuild the Vulkan Device dropdown from audio.cpp's detected devices.

        Item 0 is always "Auto (audio.cpp default)". A currently selected
        value survives the rebuild: a detected device stays selected, and a
        stale value (e.g. from a saved manifest) is kept as an explicit
        "(not detected)" row rather than silently changed.
        """
        current = self._audio_cpp_device_value()
        combo = self.audio_cpp_device_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Auto (audio.cpp default)", None)
        for item in devices:
            combo.addItem(_device_row_text(int(item["index"]), item["name"]), int(item["index"]))
        if current is not None:
            index = combo.findData(current)
            if index < 0:
                combo.addItem(_device_row_text(current, "(not detected)"), current)
                index = combo.findData(current)
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _handle_vulkan_devices(self, devices: list) -> None:
        """Populate the Vulkan Device picker from audio.cpp's actual device
        list, so users pick a real GPU by name instead of a blind index range.
        A stale selection (e.g. from a saved manifest) is preserved as an
        explicit "(not detected)" row and surfaced in the label, never applied
        silently."""
        self._vulkan_devices_probed = True
        self._repopulate_audio_cpp_device_combo(devices)
        current = self._audio_cpp_device_value()
        detected = {int(item["index"]) for item in devices}
        note = ""
        if current is not None and current not in detected:
            note = (
                f"\nNote: device {current} is not in the detected list and is "
                f"kept as a custom entry; choose a detected device or Auto."
            )
        if not devices:
            self.audio_cpp_device_label.setText("No Vulkan devices detected by audio.cpp." + note)
            return
        labels = [_device_row_text(int(item["index"]), item["name"]) for item in devices]
        self.audio_cpp_device_label.setText("\n".join(labels) + note)
        detail = " | ".join(f"{item['index']}={item['name']}" for item in devices)
        self.audio_cpp_device_combo.setToolTip(
            f"Detected Vulkan devices:\n{detail}\n\n"
            "Pick the device to use (equivalent to ORACLE_AUDIOCPP_DEVICE); "
            "Auto lets audio.cpp choose."
        )

    def _handle_vulkan_probe_failed(self, message: str) -> None:
        self._vulkan_devices_probed = True
        self.audio_cpp_device_label.setText(f"audio.cpp devices unavailable: {message}")
        self.audio_cpp_device_combo.setToolTip(
            f"Could not list audio.cpp devices: {message}\n\n"
            "Auto lets audio.cpp choose; otherwise pick a device index "
            "(equivalent to ORACLE_AUDIOCPP_DEVICE)."
        )

    def _cleanup_vulkan_probe_thread(self) -> None:
        if self._vulkan_probe_thread is not None:
            self._vulkan_probe_thread.deleteLater()
            self._vulkan_probe_thread = None

    def test_vulkan_backend(self) -> None:
        """Run a quick audio.cpp preflight in a background thread and report
        which GPU the Vulkan backend would use.

        When the backend's prerequisites are missing (or a previous setup is
        still running), the automatic CPU→GPU setup runs first and the
        preflight is queued to execute on its own once setup finishes — the
        same no-manual-steps behavior as Render/Preview, so the button never
        demands a hand-run script."""
        if self._vulkan_preflight_thread is not None:
            self.error_panel.append("A Vulkan backend preflight is already running.")
            return
        missing = _vulkan_prerequisite_missing()
        if missing:
            if self._preflight_queued_after_setup:
                self.error_panel.append(
                    "Vulkan backend test already queued; it will run when the setup finishes."
                )
                return
            # Start (or reuse) the automatic setup and queue the preflight
            # behind it. A failed setup is surfaced by
            # _handle_vulkan_setup_failed (which also clears the queue), so
            # the test can never be left dangling.
            self._start_vulkan_setup()
            self._preflight_queued_after_setup = True
            self.error_panel.append(
                "Vulkan backend test queued: the backend is being set up automatically; "
                "the test will run by itself once setup finishes."
            )
            QMessageBox.information(
                self,
                "Vulkan Backend Setup",
                "The Vulkan backend is being set up automatically (building audiocpp_cli "
                "and/or downloading the Chatterbox model). The backend test will run by "
                "itself once setup finishes.",
            )
            return
        self.error_panel.append("Testing Vulkan backend (binary, model, devices)...")
        # Parented to the window so a close while preflighting can wait() on it
        # instead of destroying a running QThread (which Qt aborts on).
        thread = VulkanPreflightThread(
            self,
            device_index=self._audio_cpp_device_value(),
            preflight_report=_vulkan_preflight_report,
        )
        thread.completed.connect(self._handle_vulkan_preflight_completed)
        thread.failed.connect(self._handle_vulkan_preflight_failed)
        thread.finished.connect(self._cleanup_vulkan_preflight_thread)
        # Assign before refreshing so the button locks while the preflight runs.
        self._vulkan_preflight_thread = thread
        self._refresh_vulkan_preflight_button()
        thread.start()

    def _handle_vulkan_preflight_completed(self, report: str) -> None:
        self.error_panel.append(report)
        QMessageBox.information(self, "Vulkan Backend Test", report)

    def _handle_vulkan_preflight_failed(self, message: str) -> None:
        self.error_panel.append(f"Vulkan backend test failed: {message}")
        QMessageBox.warning(self, "Vulkan Backend Test", message)

    def _cleanup_vulkan_preflight_thread(self) -> None:
        if self._vulkan_preflight_thread is not None:
            self._vulkan_preflight_thread.deleteLater()
            self._vulkan_preflight_thread = None
            self._refresh_vulkan_preflight_button()

    def download_vulkan_model(self) -> None:
        """Fetch the Chatterbox model in the background.

        Runs scripts/download_audio_cpp_model.sh off the UI thread (model
        downloads are large and slow). On success the reported
        ORACLE_AUDIOCPP_MODEL path is set for this session so the prerequisite
        warning clears and renders are ready; the exact export line is shown
        for users who want to persist it in their shell.
        """
        if self._model_download_thread is not None:
            self.error_panel.append("A Vulkan model download is already in progress.")
            return
        script = self.repo_root / "scripts" / "download_audio_cpp_model.sh"
        if not script.exists():
            # Defensive: the script ships with this repo, so its absence means a
            # broken checkout rather than a missing audio.cpp clone.
            message = "This checkout is missing scripts/download_audio_cpp_model.sh."
            self.error_panel.append(message)
            QMessageBox.warning(self, "Vulkan Model Download Unavailable", message)
            return
        self.download_vulkan_model_action.setEnabled(False)
        self.error_panel.append("Downloading the Vulkan Chatterbox model in the background...")
        # Parented to the window; on close the subprocess is cancelled and the
        # thread waited on so a running QThread is never destroyed.
        thread = ModelDownloadThread(script, self)
        thread.completed.connect(self._handle_vulkan_model_downloaded)
        thread.failed.connect(self._handle_vulkan_model_download_failed)
        thread.finished.connect(self._cleanup_model_download_thread)
        self._model_download_thread = thread
        thread.start()

    def _handle_vulkan_model_downloaded(self, model_path: str) -> None:
        """The model is installed; make it this session's Vulkan model and show
        the exact export line so the user can persist it in their shell."""
        os.environ["ORACLE_AUDIOCPP_MODEL"] = model_path
        # Persist the resolved path so a later launch can restore it (when
        # 'Remember GPU/CPU choice' is enabled) without re-downloading.
        self._persist_remembered_settings(model=model_path)
        self.error_panel.append(f"Vulkan model downloaded: ORACLE_AUDIOCPP_MODEL={model_path}")
        # The inline warning only belongs on the Vulkan backend; the download
        # can be started from the Settings menu while PyTorch is still active.
        if self.inference_backend_combo.currentData() == "vulkan":
            self._refresh_vulkan_prerequisite_warning()
        else:
            self.vulkan_prerequisite_warning.hide()
        QMessageBox.information(
            self,
            "Vulkan Model Downloaded",
            f"The Chatterbox model is installed and ORACLE_AUDIOCPP_MODEL is set for this session:\n\n"
            f'    export ORACLE_AUDIOCPP_MODEL="{model_path}"\n\n'
            f"The Vulkan backend is now ready to render. With Settings → Remember "
            f"GPU/CPU choice enabled (the default), this path is restored "
            f"automatically at the next launch — no re-download. Alternatively, "
            f"add that export line to your shell profile.",
        )

    def _handle_vulkan_model_download_failed(self, message: str) -> None:
        self.error_panel.append(f"Vulkan model download failed: {message}")
        QMessageBox.warning(self, "Vulkan Model Download Failed", message)

    def _cleanup_model_download_thread(self) -> None:
        if self._model_download_thread is not None:
            self._model_download_thread.deleteLater()
            self._model_download_thread = None
            self.download_vulkan_model_action.setEnabled(True)

    # ---------------------
    # Help-menu surfaces (U3): crash privacy/review, license activation, About.

    @staticmethod
    def _consent_root() -> Path:
        """The repo-local root the crash/privacy consent store lives under
        (the same repo-root convention utils.logging.default_log_file uses)."""
        return Path(__file__).resolve().parents[2]

    def _open_path_with_viewer(self, path: str) -> None:
        """Open a file with the OS-assigned viewer (the share flow's first
        half — the user copies or sends the report themselves; no upload)."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _maybe_run_crash_startup_flow(self) -> None:
        """Startup branch (D8): next-session review when reports exist,
        otherwise the one-time first-run consent ask. Consent-off installs
        that already answered get nothing."""
        from the_oracle.gui_crash import maybe_run_startup_flow

        maybe_run_startup_flow(
            self,
            consent_root=self._consent_root(),
            dialog_cls=QDialog,
            message_box_cls=QMessageBox,
            open_path_fn=self._open_path_with_viewer,
        )

    def _open_crash_privacy_dialog(self) -> None:
        from the_oracle.crash import consent as crash_consent
        from the_oracle.gui_crash import run_first_run_consent, run_next_session_review

        root = self._consent_root()
        run_next_session_review(
            self, root, dialog_cls=QDialog, message_box_cls=QMessageBox, open_path_fn=self._open_path_with_viewer
        )
        # The enabling path for a previously-declined install lives here:
        # the first-run dialog's "not now" means never asked again, so Help
        # is the only route back (plan contract).
        if not crash_consent.read_consent(root):
            if run_first_run_consent(self, root, dialog_cls=QDialog):
                from the_oracle.crash import handlers as crash_handlers

                crash_handlers.enable_faulthandler_catch(root)
                self.error_panel.append("Local crash reporting enabled.")
        else:
            from the_oracle.crash import bundle as crash_bundle

            count = len(crash_bundle.list_records(root))
            self.error_panel.append(
                f"Local crash reporting is enabled ({count} report(s) on disk)."
            )

    def _open_license_activation(self) -> None:
        from the_oracle.gui_license import run_activation_dialog

        if run_activation_dialog(self, self._consent_root(), dialog_cls=QDialog, message_box_cls=QMessageBox):
            self.error_panel.append("License activated — see About for the new edition.")

    def _open_about_panel(self) -> None:
        from the_oracle.gui_license import run_about_panel

        run_about_panel(self, dialog_cls=QDialog)

    def closeEvent(self, event) -> None:
        """Close only after every GUI-owned QThread has stopped.

        Destroying a live QThread is a hard Qt abort, not a normal Python
        exception. This matters most for RenderWorker: the user can close the
        window while rendering, and the old close path only waited for Vulkan
        helpers. If any thread cannot stop within its bounded shutdown window,
        keep the window open rather than trading a close click for a crash.
        """
        def wait_for_thread(thread, label: str, timeout_ms: int) -> bool:
            wait = getattr(thread, "wait", None)
            if not callable(wait):
                # Test doubles and already-detached helpers may not expose the
                # QThread API; there is nothing useful to wait for in that case.
                return True
            try:
                finished = bool(wait(timeout_ms))
            except Exception as exc:
                self.error_panel.append(f"Could not stop {label}: {exc}")
                return False
            if not finished:
                self.error_panel.append(
                    f"{label} is still running. Finish or cancel it before closing The Oracle."
                )
                return False
            return True

        # Render/preview workers and startup prewarm are also QThreads. Waiting
        # here prevents Qt's "QThread: Destroyed while thread is still
        # running" abort when the user closes during one of these operations.
        for thread, label, timeout_ms in (
            (self.render_worker, "Render", 1000),
            (self.preview_worker, "Preview", 1000),
            (self._prewarm_thread, "Startup warmup", 1000),
        ):
            if thread is None:
                continue
            cancel = getattr(thread, "request_cancel", None)
            if callable(cancel):
                cancel()
            if not wait_for_thread(thread, label, timeout_ms):
                event.ignore()
                return

        for studio in list(self._recording_studios):
            if studio._worker is None:
                continue
            # The studio dialogs are parented to this window: closing the main
            # window mid-take would destroy a dialog (and its QThread) while
            # its capture thread still runs — a hard Qt abort. Join every
            # studio's worker here with the same bounded-wait discipline as
            # the render workers.
            studio._stop_recording()
            if not wait_for_thread(studio._worker, "Recording take", 2000):
                event.ignore()
                return
            studio._worker.deleteLater()
            studio._worker = None
        for studio in list(self._recording_studios):
            # Workers are stopped above; closing the dialog itself runs its
            # per-instance finished handler (picker refresh + take report).
            studio.close()

        probe = self._vulkan_probe_thread
        if probe is not None:
            if not wait_for_thread(probe, "Vulkan device probe", 2000):
                event.ignore()
                return
            self._vulkan_probe_thread = None
        preflight = self._vulkan_preflight_thread
        if preflight is not None:
            # Disconnect so a finishing preflight can't pop a modal report
            # dialog during app close; the subprocess probe is quick, so a
            # bounded wait is enough.
            try:
                preflight.completed.disconnect()
                preflight.failed.disconnect()
                preflight.finished.disconnect()
            except RuntimeError:
                pass
            if not wait_for_thread(preflight, "Vulkan preflight", 2000):
                event.ignore()
                return
            self._vulkan_preflight_thread = None
        download = self._model_download_thread
        if download is not None:
            # A model download can take minutes; cancel the subprocess and wait
            # so a running QThread is never destroyed. Signals are disconnected
            # first so the cancel's failure path can't pop a modal dialog
            # during app close.
            try:
                download.completed.disconnect()
                download.failed.disconnect()
                download.finished.disconnect()
            except RuntimeError:
                pass
            download.request_cancel()
            if not wait_for_thread(download, "Vulkan model download", 5000):
                event.ignore()
                return
            self._model_download_thread = None
        setup = self._vulkan_setup_thread
        if setup is not None:
            # A build/download can take minutes; cancel the subprocess group and
            # wait so a running QThread is never destroyed. Signals are
            # disconnected first so the cancel's failure path can't pop a modal
            # dialog during app close.
            try:
                setup.progress.disconnect()
                setup.completed.disconnect()
                setup.failed.disconnect()
                setup.finished.disconnect()
            except RuntimeError:
                pass
            setup.request_cancel()
            if not wait_for_thread(setup, "Vulkan backend setup", 5000):
                event.ignore()
                return
            self._vulkan_setup_thread = None
        # Cast dialogs are modal, so the main window can't be closing while one
        # is open - but close any stragglers defensively so their pending edits
        # apply (or discard) before the window is destroyed.
        for dialog in list(self._cast_dialogs):
            dialog.close()
        # Stop the persistent preview player before the window (and its child
        # player object) is destroyed, so the QtMultimedia backend isn't torn
        # down mid-playback during app exit.
        self._stop_preview_player()
        # Capture every current option and slider position at the final normal
        # close, including controls without a per-widget persistence signal.
        # This is the last checkpoint before the process exits.
        self._persist_workspace_layout()
        super().closeEvent(event)

    def _audio_cpp_device_value(self) -> int | None:
        data = self.audio_cpp_device_combo.currentData()
        return data if isinstance(data, int) else None

    def _audio_cpp_threads_value(self) -> int | None:
        value = self.audio_cpp_threads_spin.value()
        return None if value < 1 else value

    def _audio_cpp_timeout_value(self) -> int | None:
        value = self.audio_cpp_timeout_spin.value()
        return None if value < 1 else value

    def _audio_cpp_max_batch_value(self) -> int | None:
        value = self.audio_cpp_max_batch_spin.value()
        return None if value < 1 else value

    def _set_audio_cpp_device_value(self, value: int | None) -> None:
        combo = self.audio_cpp_device_combo
        if value is None:
            # "Auto (audio.cpp default)" is always item 0.
            combo.setCurrentIndex(0)
            return
        value = max(0, int(value))
        index = combo.findData(value)
        if index < 0:
            combo.addItem(_device_row_text(value, "(not detected)"), value)
            index = combo.findData(value)
        combo.setCurrentIndex(index)

    def _set_audio_cpp_threads_value(self, value: int | None) -> None:
        self.audio_cpp_threads_spin.setValue(0 if value is None else max(1, int(value)))

    def _set_audio_cpp_timeout_value(self, value: int | None) -> None:
        self.audio_cpp_timeout_spin.setValue(0 if value is None else max(1, int(value)))

    def _set_audio_cpp_max_batch_value(self, value: int | None) -> None:
        self.audio_cpp_max_batch_spin.setValue(0 if value is None else max(1, int(value)))

    # --------------------
    # Cross-session backend memory
    # --------------------
    def _remember_backend_enabled(self) -> bool:
        """Whether the 'Remember GPU/CPU choice' option is currently checked."""
        if hasattr(self, "remember_backend_action"):
            return self.remember_backend_action.isChecked()
        return bool(self._app_settings.get("remember_backend", True))

    def _persist_remembered_settings(self, _value=None, *, binary: str | None = None, model: str | None = None) -> None:
        """Write the current backend selection + Vulkan knobs + audio.cpp paths
        to the app-level settings file, so the next launch can restore them.

        Connected to the backend combo and Vulkan knob changes (gated on
        ``_app_settings_ready``) and called explicitly with the resolved paths
        when the automatic setup or a model download finishes. ``binary`` /
        ``model`` (when given) win over the current environment, so a fresh
        setup result is recorded even if the env vars were not pre-set.
        """
        if not self._app_settings_ready:
            return
        enabled = self._remember_backend_enabled()
        if not enabled:
            # Option off = manual control: don't keep writing the settings file
            # on every backend/knob change. The disabled flag itself was already
            # persisted by the toggle handler, so the next launch starts on the
            # default (PyTorch).
            return
        self._app_settings.update({
            "remember_backend": enabled,
            "inference_backend": self.inference_backend_combo.currentData() or "pytorch",
            "device_mode": self._pytorch_device_selection()[0],
            "cuda_device": self._pytorch_device_selection()[1],
            "audio_cpp_device": self._audio_cpp_device_value(),
            "audio_cpp_threads": self._audio_cpp_threads_value(),
            "audio_cpp_timeout": self._audio_cpp_timeout_value(),
            "audio_cpp_max_batch": self._audio_cpp_max_batch_value(),
            "audio_cpp_cli": binary
            or os.environ.get("ORACLE_AUDIOCPP_CLI")
            or self._app_settings.get("audio_cpp_cli", ""),
            "audio_cpp_model": model
            or os.environ.get("ORACLE_AUDIOCPP_MODEL")
            or self._app_settings.get("audio_cpp_model", ""),
        })
        try:
            save_app_settings(self._app_settings)
        except Exception as exc:
            self.error_panel.append(f"Could not persist GPU settings: {exc}")

    def _on_remember_backend_toggled(self, checked: bool) -> None:
        """React to the Settings menu toggle: record the new state immediately.

        Turning it on persists the current backend + paths right away (so the
        very next launch already restores them); turning it off persists just
        the disabled flag so future launches start on PyTorch (CPU).
        """
        self._app_settings["remember_backend"] = checked
        if checked:
            self._persist_remembered_settings()
        else:
            # Disabled = manual control: reset to the default backend so a
            # stale in-memory selection (e.g. vulkan from construction) is
            # not persisted as the "remembered" state.
            self._app_settings["inference_backend"] = "pytorch"
            try:
                save_app_settings(self._app_settings)
            except Exception as exc:
                self.error_panel.append(f"Could not persist GPU settings: {exc}")
        self.error_panel.append(
            "Remember GPU/CPU choice enabled: the backend selection and audio.cpp "
            "paths are restored automatically on the next launch."
            if checked
            else "Remember GPU/CPU choice disabled: the app will start on PyTorch (CPU)."
        )

    def _apply_remembered_backend(self) -> None:
        """Restore the remembered inference backend and audio.cpp paths at launch.

        Runs once, after the window is shown. When 'Remember GPU/CPU choice' is
        enabled it re-applies the persisted ``ORACLE_AUDIOCPP_CLI`` /
        ``ORACLE_AUDIOCPP_MODEL`` paths to the environment (only if the files
        still exist — a stale path is surfaced, never silently applied) and
        selects the remembered backend + Vulkan knobs. Selecting Vulkan then
        runs the normal prerequisite check: if the persisted paths are valid
        there is nothing left to do (zero setup across sessions); if they are
        stale the automatic setup re-runs on its own.
        """
        if not self._app_settings.get("remember_backend", True):
            return
        cli_path = self._app_settings.get("audio_cpp_cli", "")
        model_path = self._app_settings.get("audio_cpp_model", "")
        if cli_path:
            resolved_cli = Path(cli_path).expanduser()
            if resolved_cli.exists():
                os.environ["ORACLE_AUDIOCPP_CLI"] = str(resolved_cli)
            else:
                self.error_panel.append(f"Remembered audiocpp_cli path no longer exists: {cli_path}")
        if model_path:
            resolved_model = Path(model_path).expanduser()
            if resolved_model.exists():
                os.environ["ORACLE_AUDIOCPP_MODEL"] = str(resolved_model)
            else:
                self.error_panel.append(f"Remembered Chatterbox model path no longer exists: {model_path}")
        backend = self._app_settings.get("inference_backend", "pytorch")
        if backend == "vulkan":
            index = self.inference_backend_combo.findData("vulkan")
            if index >= 0:
                self.inference_backend_combo.setCurrentIndex(index)
            self._set_audio_cpp_device_value(self._app_settings.get("audio_cpp_device"))
            self._set_audio_cpp_threads_value(self._app_settings.get("audio_cpp_threads"))
            self._set_audio_cpp_timeout_value(self._app_settings.get("audio_cpp_timeout"))
            self._set_audio_cpp_max_batch_value(self._app_settings.get("audio_cpp_max_batch"))
            self.error_panel.append("Restored the remembered Vulkan backend from the last session.")
        elif self._app_settings.get("device_mode") == "cuda":
            target = self._app_settings.get("cuda_device")
            target_data = f"cuda:{int(target)}" if target is not None else None
            index = self.pytorch_device_combo.findData(target_data) if target_data else -1
            self.pytorch_device_combo.setCurrentIndex(index if index >= 0 else 0)
            self.error_panel.append("Restored the remembered CUDA PyTorch device from the last session.")

    def _refresh_reference_pickers(self) -> None:
        # Generous limit so a freshly recorded Seashell (in Seashells/ root)
        # shows up in the pickers right after the bundled generic voices
        # instead of being cut off by the catalog's default 10-entry cap.
        defaults = default_voice_choices(self.repo_root, limit=40)
        blends = blend_voice_choices(self.paths.profile_dir)
        recents = [path for path in load_recent_reference_paths() if Path(path).exists()]
        for group in self._all_speaker_groups().values():
            group.set_reference_choices(defaults, recents, group.reference_path.text(), blends=blends)
            group.set_blend_choices(defaults, recents, group.blend_target_path(), blends=blends)

    # --------------------
    # Inference setup and replayable GUI tutorial
    # --------------------
    def _apply_inference_wizard_preferences(self, payload: dict) -> None:
        """Apply the wizard's default-folder choices immediately and persist them.

        Folder preferences are intentionally independent of the repository's
        built-in ``Input``/``Output`` directories.  Unchecking a preference
        restores that built-in location while preserving the last-used file
        separately, so a later file choice still wins on the next launch.
        """
        input_dir = str(payload.get("default_input_dir") or "").strip()
        output_dir = str(payload.get("default_output_dir") or "").strip()
        previous_input_dir = self._default_input_dir
        previous_output_dir = self._default_output_dir
        if payload.get("remember_input_folder") and input_dir:
            self._default_input_dir = Path(input_dir).expanduser()
            self._default_input_dir.mkdir(parents=True, exist_ok=True)
            self._app_settings["default_input_dir"] = str(self._default_input_dir)
        else:
            self._default_input_dir = self.paths.input_dir
            self._app_settings.pop("default_input_dir", None)
        if payload.get("remember_output_folder") and output_dir:
            self._default_output_dir = Path(output_dir).expanduser()
            self._default_output_dir.mkdir(parents=True, exist_ok=True)
            self._app_settings["default_output_dir"] = str(self._default_output_dir)
        else:
            self._default_output_dir = self.paths.output_dir
            self._app_settings.pop("default_output_dir", None)
        current_output_text = self.outdir_path.text().strip()
        if not current_output_text or current_output_text in {
            str(self.paths.output_dir),
            str(previous_output_dir),
        }:
            self.outdir_path.setText(str(self._default_output_dir))
        current_input_text = self.input_path.text().strip()
        previous_default_script = previous_input_dir / "What is, reality.txt"
        if current_input_text in {"", str(previous_default_script), str(self.paths.input_dir / "What is, reality.txt")}:
            new_default_script = self._default_input_dir / "What is, reality.txt"
            if new_default_script.exists():
                self.input_path.setText(str(new_default_script))
        if self._app_settings_ready:
            self._persist_workspace_layout()
        else:
            # The wizard may be exercised immediately after construction (and
            # before the queued show callback flips the startup gate). Persist
            # the explicit folder choice without snapshotting half-built GUI
            # state.
            try:
                save_app_settings(self._app_settings)
            except Exception as exc:
                self.error_panel.append(f"Could not persist folder preferences: {exc}")

    def _apply_inference_wizard_selection(self, device_mode: str, cuda_device: int | None) -> None:
        """Apply the wizard's chosen inference path to the real GUI pickers."""
        if device_mode == "vulkan":
            index = self.inference_backend_combo.findData("vulkan")
            if index >= 0 and self.inference_backend_combo.model().item(index).isEnabled():
                self.inference_backend_combo.setCurrentIndex(index)
            return
        backend_index = self.inference_backend_combo.findData("pytorch")
        if backend_index >= 0:
            self.inference_backend_combo.setCurrentIndex(backend_index)
        target = f"cuda:{int(cuda_device)}" if device_mode == "cuda" and cuda_device is not None else "cpu"
        device_index = self.pytorch_device_combo.findData(target)
        if device_index >= 0:
            self.pytorch_device_combo.setCurrentIndex(device_index)
        else:
            self.pytorch_device_combo.setCurrentIndex(self.pytorch_device_combo.findData("cpu"))
        self._persist_remembered_settings()

    def _handle_inference_wizard_completed(self, accepted: bool, mode: str) -> None:
        """Remember whether first-run onboarding was completed or dismissed."""
        self._inference_wizard = None
        if mode == "full" and accepted:
            self._app_settings["inference_wizard_completed"] = True
            self._app_settings["inference_wizard_dismissed"] = False
        elif mode == "full" and not accepted:
            self._app_settings["inference_wizard_dismissed"] = True
        if mode == "full":
            try:
                save_app_settings(self._app_settings)
            except Exception as exc:
                self.error_panel.append(f"Could not save tutorial state: {exc}")
        self.error_panel.append(
            "Inference setup and tutorial completed. Replay it from Settings when needed."
            if accepted else
            "Inference setup tutorial skipped. Replay it from Settings at any time."
        )

    def _refresh_pytorch_device_options(self) -> None:
        """Re-probe and rebuild the PyTorch device picker for wizard replay.

        A GPU can be installed or enabled while The Oracle is still open. A
        replayed discovery stage must therefore use fresh hardware facts,
        rather than the snapshot captured during MainWindow construction.
        """
        if not hasattr(self, "pytorch_device_combo"):
            return
        current = self.pytorch_device_combo.currentData()
        combo = self.pytorch_device_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("CPU / system DRAM", "cpu")
        for device in self._cuda_devices:
            combo.addItem(device.label, f"cuda:{device.index}")
            item = combo.model().item(combo.count() - 1)
            if item is not None and (not device.torch_available or not device.suitable):
                item.setEnabled(False)
                item.setToolTip(device.reason)
        if not any(device.torch_available and device.suitable for device in self._cuda_devices):
            combo.addItem("CUDA unavailable (see tooltip)", "cuda-unavailable")
            item = combo.model().item(combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
                item.setToolTip("CUDA is not currently usable: " + cuda_reason())
        restored = combo.findData(current)
        combo.setCurrentIndex(restored if restored >= 0 else 0)
        combo.blockSignals(False)

    def _start_inference_wizard(self, mode: str = "full", *, force: bool = False) -> None:
        """Show the hardware-aware tutorial, keeping only one copy alive."""
        if self._inference_wizard is not None:
            self._inference_wizard.raise_()
            self._inference_wizard.activateWindow()
            return
        if mode == "full" and not force and (self._app_settings.get("inference_wizard_completed") or self._app_settings.get("inference_wizard_dismissed")):
            return
        self._cuda_devices = cuda_devices()
        self._refresh_pytorch_device_options()
        wizard = InferenceSetupWizard(
            self,
            devices=self._cuda_devices,
            mode=mode,
            on_selection=self._apply_inference_wizard_selection,
            on_preferences=self._apply_inference_wizard_preferences,
        )
        wizard.completed.connect(self._handle_inference_wizard_completed)
        wizard.finished.connect(wizard.deleteLater)
        self._inference_wizard = wizard
        wizard.start()

    # --------------------
    # Custom Voice Recording Studio
    # --------------------
    def open_recording_studio(self) -> None:
        """Open the Recording Studio (non-modal, its own window).

        Every dialog is tracked in ``_recording_studios`` with its own
        completion handler, so opening several keeps each one's worker and
        saved take instead of orphaning the earlier ones.
        """
        dialog = RecordingStudioDialog(
            repo_root=self.repo_root,
            voice_dir=self.paths.voice_dir,
            input_dir=self._default_input_dir,
            parent=self,
            on_assign=self._assign_recording_to_speaker,
        )
        dialog.apply_setup_preferences(self._app_settings.get("recording_settings") or {})
        self._recording_studios.add(dialog)
        dialog.set_assign_speakers(
            [(key, (self._speaker_names.get(key) or "").strip()) for key in self._cast]
        )
        # Refresh the voice pickers whenever a studio closes - whether or not
        # a recording was saved. Each dialog carries its own handler, so
        # several open studios never orphan each other's close events.
        dialog.finished.connect(lambda result, dlg=dialog: self._on_recording_studio_closed(dlg, result))
        dialog.finished.connect(dialog.deleteLater)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        if not self._app_settings.get("recording_wizard_completed", False) and not self._app_settings.get("recording_wizard_dismissed", False):
            # 150 ms so the studio's first frame paints before the guide
            # highlights a control. start() restarts: opening another studio
            # while a launch is pending moves the deadline instead of
            # doubling it.
            self._recording_wizard_launch_timer.start(150)

    def _apply_recording_wizard_preferences(self, payload: dict) -> None:
        """Apply and persist the setup choices made in the recording guide."""
        for studio in list(self._recording_studios):
            studio.apply_setup_preferences(payload)
        if payload.get("remember_input_default"):
            self._app_settings["recording_settings"] = dict(payload)
        elif isinstance(self._app_settings.get("recording_settings"), dict):
            self._app_settings["recording_settings"].pop("input_file", None)
        if payload.get("remember_output_default"):
            self._app_settings.setdefault("recording_settings", {}).update({"output_dir": payload.get("output_dir", "")})
        else:
            self._app_settings.setdefault("recording_settings", {}).pop("output_dir", None)
        self._app_settings["recording_settings"] = {
            **self._app_settings.get("recording_settings", {}),
            "microphone_index": payload.get("microphone_index"),
            "samplerate": payload.get("samplerate"),
            "generic_name_warning": payload.get("generic_name_warning", True),
            "remember_input_default": payload.get("remember_input_default", True),
            "remember_output_default": payload.get("remember_output_default", True),
        }
        self._persist_recording_settings()

    def _persist_recording_settings(self) -> None:
        if not self._app_settings_ready:
            return
        try:
            save_app_settings(self._app_settings)
        except Exception as exc:
            self.error_panel.append(f"Could not persist Recording Studio settings: {exc}")

    def _persist_recording_dialog_state(self, studio: RecordingStudioDialog) -> None:
        """Remember the last Recording Studio choices when its window closes.

        The wizard supplies the initial defaults, but normal use continues after
        the guide is complete. Capturing the live dialog state here means the
        next open returns to the last script, microphone, rate, folder, and
        naming preference without requiring the wizard to be replayed.
        """
        payload = {
            "microphone_index": studio.mic_combo.currentData(),
            "samplerate": studio.rate_combo.currentData(),
            "input_file": studio.script_combo.currentData() or "",
            "output_dir": studio.outdir_combo.currentText().strip(),
            "output_filename": studio.name_edit.text().strip(),
            "generic_name_warning": bool(studio.generic_name_warning_enabled),
            "remember_input_default": bool(studio.remember_input_default),
            "remember_output_default": bool(studio.remember_output_default),
        }
        if not payload["remember_input_default"]:
            payload.pop("input_file", None)
        if not payload["remember_output_default"]:
            payload.pop("output_dir", None)
        self._app_settings["recording_settings"] = {
            **(self._app_settings.get("recording_settings") or {}),
            **payload,
        }
        self._persist_recording_settings()

    def _start_recording_wizard(self, *, force: bool = False) -> None:
        """Show the first-open/replayable Recording Studio guide."""
        studios = list(self._recording_studios)
        studio = studios[-1] if studios else None
        if studio is None or self._recording_wizard is not None:
            return
        if not force and (self._app_settings.get("recording_wizard_completed") or self._app_settings.get("recording_wizard_dismissed")):
            return
        studio.apply_setup_preferences(self._app_settings.get("recording_settings") or {})
        wizard = RecordingStudioSetupWizard(studio, on_apply=self._apply_recording_wizard_preferences)
        wizard.completed.connect(self._handle_recording_wizard_completed)
        wizard.finished.connect(wizard.deleteLater)
        self._recording_wizard = wizard
        wizard.start()

    def _handle_recording_wizard_completed(self, accepted: bool, mode: str) -> None:
        self._recording_wizard = None
        if mode == "full":
            self._app_settings["recording_wizard_completed"] = bool(accepted)
            self._app_settings["recording_wizard_dismissed"] = not accepted
            self._persist_recording_settings()
        self.error_panel.append(
            "Recording Studio setup and speaking guide completed."
            if accepted else
            "Recording Studio guide skipped; replay it from Settings when ready."
        )

    def _assign_recording_to_speaker(self, speaker: str, path: Path) -> None:
        """Point a speaker's Custom Voice Reference Audio at a fresh Seashell.

        Sets the group's reference path and refreshes the pickers so the new
        voice is selected everywhere immediately (the dialog stays open for
        further takes; the main window's pickers update behind it).
        """
        try:
            group = self._all_speaker_groups()[speaker]
        except KeyError:
            self.error_panel.append(f"No Speaker {speaker} group exists to assign the voice to.")
            return
        group.reference_path.setText(str(path))
        self._refresh_reference_pickers()
        self.error_panel.append(
            f"Assigned {Path(path).name} as the voice for Speaker {speaker} - it will "
            "be used on the next render."
        )

    def _on_recording_studio_closed(self, dialog: RecordingStudioDialog, _result: int) -> None:
        # Per-instance completion: each studio reports its own saved take, so
        # a second dialog never steals or drops the first one's recording.
        self._recording_studios.discard(dialog)
        self._persist_recording_dialog_state(dialog)
        self._refresh_reference_pickers()
        if dialog._last_saved_path is not None:
            self.error_panel.append(
                f"Recorded new Seashell: {dialog._last_saved_path.name} - "
                "pick it under Custom Voice Reference Audio for any speaker."
            )

    def _save_blend_as_voice(self, group: SpeakerGroup) -> None:
        """Save the group's current blend configuration as a named picker voice.

        The derived clip is materialized deterministically (same inputs,
        weight, and mode always produce the same wav), then the voice appears
        under 'Saved Blends' in every speaker panel's pickers.
        """
        base = group.reference_path.text().strip()
        target = group.blend_target_path()
        if not base or not target:
            self.error_panel.append(
                "Pick a base voice (Voice Reference) and a Hybrid Second Voice before saving."
            )
            QMessageBox.information(
                self,
                "Save Blend As",
                "Pick a base voice and a Hybrid Second Voice first, then save the blend.",
            )
            return
        name, ok = QInputDialog.getText(
            self,
            "Save Blend As Voice",
            "Name for this blended voice (shown in the voice picker):",
        )
        if not ok or not name.strip():
            return
        try:
            choice = save_blend_voice(
                self.paths.profile_dir,
                name,
                base,
                target,
                weight=group.blend_weight_spin.value() / 100.0,
                mode=group.blend_mode_combo.currentData() or "mix",
            )
        except Exception as exc:
            self.error_panel.append(f"Could not save blend voice: {exc}")
            QMessageBox.critical(self, "Save Blend As", str(exc))
            return
        self._refresh_reference_pickers()
        self.error_panel.append(f"Saved blend voice '{name}' — select it under Saved Blends in any speaker panel.")

    # --------------------
    # Prewarm management
    # --------------------
    def _start_prewarm(self) -> None:
        with self._prewarm_lock:
            if self._prewarm_state in {"warming", "ready"}:
                return
            self._prewarm_state = "warming"
        # Keep the UI responsive: disable heavy actions while warmup runs, but do not block the event loop.
        self.analyze_button.setEnabled(False)
        self.render_button.setEnabled(False)
        self._prewarm_thread = PrewarmThread(device=self._pytorch_device_selection()[0])
        self._prewarm_thread.ready.connect(self._handle_prewarm_ready)
        self._prewarm_thread.failed.connect(self._handle_prewarm_failed)
        # Teardown only via finished: ready/failed fire from run()'s final
        # lines while the thread is still exiting, so deleteLater there races
        # the thread's own exit (QThread destroyed while running -> abort).
        self._prewarm_thread.finished.connect(self._cleanup_prewarm_thread)
        try:
            self._prewarm_thread.start()
        except Exception as exc:
            self._handle_prewarm_failed(str(exc), {"prewarm_failed": time()})

    def _handle_prewarm_ready(self, pipeline: OraclePipeline, engine: object, timing: dict[str, float]) -> None:
        with self._prewarm_lock:
            self._prewarm_state = "ready"
            self._prewarmed_pipeline = pipeline
            self._prewarmed_engine = engine
            self._prewarm_timing = timing
        # The worker detaches itself via finished -> _cleanup_prewarm_thread;
        # never deleteLater from here (thread may still be exiting).
        self._write_prewarm_timing(success=True)
        # Enable actions now that warmup finished
        self.analyze_button.setEnabled(True)
        self.render_button.setEnabled(True)

    def _handle_prewarm_failed(self, message: str, timing: dict[str, float]) -> None:
        with self._prewarm_lock:
            self._prewarm_state = "failed"
            self._prewarmed_pipeline = None
            self._prewarmed_engine = None
            self._prewarm_timing = timing | {"error": message}
        # Worker detaches itself via finished (see _cleanup_prewarm_thread).
        self.error_panel.append(f"Background prewarm failed: {message}")
        self._write_prewarm_timing(success=False)
        # Allow the user to proceed manually even if warmup failed.
        self.analyze_button.setEnabled(True)
        self.render_button.setEnabled(True)

    def _cleanup_prewarm_thread(self) -> None:
        # Run on the GUI thread only after run() has returned, so detaching
        # and deleting the QThread object here is safe.
        thread = self._prewarm_thread
        if thread is not None:
            self._prewarm_thread = None
            thread.deleteLater()

    def _write_prewarm_timing(self, success: bool) -> None:
        try:
            log_dir = self.paths.output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            timing = dict(self._prewarm_timing or {})
            if self._gui_shown_wall:
                timing["gui_shown"] = self._gui_shown_wall
            timing["prewarm_success"] = success
            (log_dir / "gui_prewarm_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        except Exception:
            pass

    # --------------------
    # Section layout persistence (splitters, section-size sliders, collapses)
    # --------------------
    def _register_section(self, key: str, section: QHSectionGroup, splitter, index: int) -> None:
        """Track one section and wire its chrome signals to persistence."""
        self._section_registry[key] = (section, splitter, index)
        section.attach_splitter(splitter, index)
        section.size_share_changed.connect(self._persist_workspace_layout)
        section.collapsed_changed.connect(self._persist_workspace_layout)

    def _persist_workspace_layout(self, *_args) -> None:
        """Snapshot splitters + section sliders/collapses into app settings."""
        if not self._app_settings_ready:
            return
        sections: dict[str, dict] = {}
        for key, (section, splitter, index) in self._section_registry.items():
            sizes = splitter.sizes()
            total = sum(sizes)
            share = 50
            if total > 0 and 0 <= index < len(sizes):
                share = int(round(100.0 * sizes[index] / total))
            sections[key] = {
                "size_share": section.size_share(),
                "collapsed": section.is_collapsed(),
            }
        self._app_settings.update({
            "theme": self._current_theme,
            # Keep the last successful/typed file when New Project temporarily
            # clears the field; a fresh launch should still return to the last
            # used input rather than losing the user's working context.
            "last_input_file": self.input_path.text().strip()
            or self._app_settings.get("last_input_file", ""),
            # Every current option and slider position (shared + speakers),
            # so the whole GUI returns exactly as left.
            "gui": self._current_gui_settings_payload(),
            "splitters": {
                "main": self._main_splitter.sizes(),
                "sections": self._sections_splitter.sizes(),
                "lower": self._lower_splitter.sizes(),
                "paths": self._paths_splitter.sizes(),
            },
            "sections": sections,
            "window_geometry": [self.width(), self.height()],
        })
        try:
            save_app_settings(self._app_settings)
        except Exception as exc:
            self.error_panel.append(f"Could not persist layout: {exc}")

    def _apply_workspace_layout(self) -> None:
        """Restore the saved workspace: all options and slider positions first,
        then splitters, section shares, collapses, and window geometry."""
        saved_gui = self._app_settings.get("gui")
        if isinstance(saved_gui, dict) and "project" in saved_gui and "speakers" in saved_gui:
            try:
                self._apply_gui_settings_payload(saved_gui)
            except Exception as exc:
                self.error_panel.append(f"Could not restore the saved options ({exc}); using defaults.")
        splitters = self._app_settings.get("splitters") or {}

        def restore_splitter(splitter, saved) -> bool:
            """Apply saved sizes to a splitter's VISIBLE panes (Qt only honors
            a setSizes request that matches the visible pane count)."""
            if not isinstance(saved, list) or sum(1 for v in saved if isinstance(v, (int, float))) == 0:
                return False
            visible = [
                index
                for index in range(splitter.count())
                if splitter.widget(index) is not None and not splitter.widget(index).isHidden()
            ]
            if len(visible) == 0 or len(saved) < len(visible):
                return False
            values = [max(20, int(saved[index])) for index in visible]
            if sum(values) <= 0:
                return False
            splitter.setSizes(values)
            return True

        restore_splitter(self._main_splitter, splitters.get("main"))
        restore_splitter(self._sections_splitter, splitters.get("sections"))
        restore_splitter(self._lower_splitter, splitters.get("lower"))
        restore_splitter(self._paths_splitter, splitters.get("paths"))
        sections = self._app_settings.get("sections") or {}
        for key, (section, _splitter, _index) in self._section_registry.items():
            data = sections.get(key)
            if isinstance(data, dict):
                share = data.get("size_share")
                if isinstance(share, (int, float)):
                    section.set_size_share(int(share))
                if bool(data.get("collapsed", False)):
                    section.set_collapsed(True)
        geometry = self._app_settings.get("window_geometry")
        if isinstance(geometry, list) and len(geometry) == 2:
            width, height = geometry
            if isinstance(width, int) and isinstance(height, int) and width >= 800 and height >= 600:
                self.resize(width, height)
        # Remembered input file wins over the launch default; fall back to the
        # default when it is missing or no longer exists.
        remembered_input = str(self._app_settings.get("last_input_file") or "")
        if remembered_input and Path(remembered_input).expanduser().exists():
            self.input_path.setText(remembered_input)
        elif not self.input_path.text().strip() and self._default_input_path.exists():
            self.input_path.setText(str(self._default_input_path))

    def cast_keys(self) -> list[str]:
        """Ordered speaker keys of the current cast (["A", "B"] by default)."""
        return list(self._cast)

    def speaker_names(self) -> dict[str, str]:
        """Optional character names per speaker key."""
        return dict(self._speaker_names)

    def _all_speaker_groups(self) -> dict[str, SpeakerGroup]:
        """Every speaker group in the current cast, in cast order."""
        groups = {"A": self.speaker_a, "B": self.speaker_b, **self.extra_speaker_groups}
        return {key: groups[key] for key in self._cast if key in groups}

    def apply_cast(
        self,
        keys: list[str],
        names: dict[str, str] | None,
        settings: dict[str, SpeakerSettings] | None,
    ) -> None:
        """Replace the speaker cast and rebuild the panels to match.

        Called by the cast-management dialog and by settings/profile load, so
        a cast of any size always leaves exactly the right panels behind (no
        stale speaker panels contaminating later renders).
        """
        cast = normalize_cast_keys(keys)
        self._cast = cast
        names = names if isinstance(names, dict) else {}
        self._speaker_names = {key: str(names.get(key, "") or "").strip() for key in cast}
        settings = settings if isinstance(settings, dict) else {}
        self._sync_extra_speaker_groups([key for key in cast if key not in ("A", "B")])
        # Speakers beyond A/B are configured in the cast dialog; the main
        # window keeps their widgets alive (the render path reads them) but
        # does not show them inline.
        self.extra_voices_section.hide()
        groups = {"A": self.speaker_a, "B": self.speaker_b, **self.extra_speaker_groups}
        for key in cast:
            group = groups.get(key)
            if group is None:
                continue
            panel_settings = settings.get(key)
            if panel_settings is not None:
                _apply_speaker_settings_to_group(group, panel_settings)
            self._update_speaker_group_title(key, group)
        # Monologue layout: with a single speaker only the narrator's panel
        # shows; a cast of one is monologue, more than one is dialogue.
        self.speaker_b.setVisible(len(cast) > 1)
        self.monologue_check.setChecked(len(cast) == 1)
        self._refresh_cast_bar()
        self._refresh_reference_pickers()
        self._refresh_language_options()

    def _update_speaker_group_title(self, key: str, group: SpeakerGroup) -> None:
        name = (self._speaker_names.get(key) or "").strip()
        group.setTitle(f"Speaker {key} — {name}" if name else f"Speaker {key}")

    def _refresh_cast_bar(self) -> None:
        parts = []
        for key in self._cast:
            name = (self._speaker_names.get(key) or "").strip()
            parts.append(f"{key} ({name})" if name else key)
        count = len(self._cast)
        noun = "speaker" if count == 1 else "speakers"
        text = f"{', '.join(parts)} — {count} {noun}"
        if count == 1 and self.monologue_check.isChecked():
            text += " (monologue)"
        self.cast_summary_label.setText(text)

    def open_cast_manager(self) -> None:
        """Open the cast-management dialog (modal).

        Tracked per-instance like the recording-studio dialogs so a second
        open can never orphan the first one's completion handling.
        """
        for dialog in list(self._cast_dialogs):
            dialog.raise_()
            dialog.activateWindow()
            return
        dialog = CastManagementDialog(self)
        self._cast_dialogs.add(dialog)
        dialog.finished.connect(lambda _result, dlg=dialog: self._cast_dialogs.discard(dlg))
        try:
            dialog.exec()
        finally:
            self._cast_dialogs.discard(dialog)

    def _sync_extra_speaker_groups(self, speakers: list[str]) -> None:
        """Create/refresh SpeakerGroup widgets for characters beyond A and B
        (up to the engine's voice capacity) so an audiobook cast gets one
        voice panel each. Panels beyond A/B are configured in the cast dialog;
        this only keeps the widgets in sync, it never shows them inline."""
        extras = sorted(speaker for speaker in speakers if speaker not in ("A", "B"))
        for key in extras:
            if key not in self.extra_speaker_groups:
                group = SpeakerGroup(key, self.paths.voice_dir, on_save_blend=self._save_blend_as_voice)
                self.extra_speaker_groups[key] = group
                self.extra_speaker_layout.addWidget(group)
                self._update_speaker_group_title(key, group)
        stale = [key for key in self.extra_speaker_groups if key not in extras]
        for key in stale:
            group = self.extra_speaker_groups.pop(key)
            self.extra_speaker_layout.removeWidget(group)
            group.deleteLater()
        self._refresh_reference_pickers()
        self._refresh_language_options()

    def _speaker_settings(self) -> dict[str, SpeakerSettings]:
        variant = self.variant_combo.currentText()
        crossfade_ms = self.crossfade_spin.value()
        return {
            key: _speaker_settings_from_group(group, variant, crossfade_ms)
            for key, group in self._all_speaker_groups().items()
        }

    @staticmethod
    def _speaker_config_from_payload(key: str, data: dict) -> SpeakerSettings:
        """Decode via the policy owner, then construct the engine dataclass."""
        return SpeakerSettings(**speaker_config_from_payload(data))

    def _pytorch_device_selection(self) -> tuple[str, int | None]:
        """Return the selected PyTorch device mode and CUDA index."""
        value = self.pytorch_device_combo.currentData()
        if isinstance(value, str) and value.startswith("cuda:"):
            try:
                return "cuda", int(value.split(":", 1)[1])
            except ValueError:
                pass
        return "cpu", None

    def _render_settings(self) -> RenderSettings:
        variant = self.variant_combo.currentText()
        mode_value = self.correction_mode_combo.currentData() or self.correction_mode_combo.currentText()
        inference_backend = self.inference_backend_combo.currentData() or "pytorch"
        device_mode, cuda_device = self._pytorch_device_selection()
        is_vulkan = inference_backend == "vulkan"
        if is_vulkan:
            device_mode, cuda_device = "cpu", None
        return RenderSettings(
            correction_mode=mode_value,
            model_variant=variant,
            language=self.speaker_a.language_combo.currentData() or "en",
            export_stems=True,
            loudness_preset=self.loudness_combo.currentText(),
            pause_between_turns_ms=self.speaker_a.pause_spin.value(),
            crossfade_ms=self.crossfade_spin.value(),
            device_mode=device_mode,
            cuda_device=cuda_device,
            inference_backend=inference_backend,
            audio_cpp_device=self._audio_cpp_device_value() if is_vulkan else None,
            audio_cpp_threads=self._audio_cpp_threads_value() if is_vulkan else None,
            audio_cpp_timeout=self._audio_cpp_timeout_value() if is_vulkan else None,
            audio_cpp_max_batch=self._audio_cpp_max_batch_value() if is_vulkan else None,
            monologue=self.monologue_check.isChecked(),
            metadata={
                "output_filename": normalize_output_filename(self.output_name.text()),
                "export_srt": "1" if self.export_srt_check.isChecked() else "",
            },
        )

    def _pipeline(self) -> OraclePipeline:
        if self.pipeline is not None:
            return self.pipeline
        with self._prewarm_lock:
            state = self._prewarm_state
        if state == "ready" and self._prewarmed_pipeline is not None:
            self.pipeline = self._prewarmed_pipeline
            return self.pipeline
        # Keep GUI analysis on the deterministic local fallbacks. The optional
        # transformer/PyTorch and LanguageTool stacks can load native code or
        # spawn helper threads inside the Qt process; on Ubuntu 24.04 that
        # combination has been observed to segfault when Analyze is clicked.
        # The CLI still uses OraclePipeline's feature-rich defaults.
        self.pipeline = OraclePipeline(
            use_transformers=False,
            use_language_tool=False,
            use_punctuation_model=False,
        )
        return self.pipeline

    def _prewarmed_engine_ready(self):
        with self._prewarm_lock:
            state = self._prewarm_state
        if state == "ready":
            # Do not pass live engine across threads; return sentinel to signal no reuse.
            return None
        return None

    def _default_gui_settings_payload(self) -> dict:
        return default_gui_settings_payload(str(self._default_output_dir), self._payload_defaults())

    def _payload_defaults(self) -> PayloadDefaults:
        """Engine defaults, supplied so gui_settings never imports pipeline."""
        default_render = RenderSettings()
        default_voice = VoiceSettings(variant=default_render.model_variant)
        return PayloadDefaults(
            model_variant=default_render.model_variant,
            correction_mode=default_render.correction_mode,
            loudness_preset=default_render.loudness_preset,
            crossfade_ms=default_render.crossfade_ms,
            inference_backend=default_render.inference_backend,
            device_mode=default_render.device_mode,
            cuda_device=default_render.cuda_device,
            audio_cpp_device=default_render.audio_cpp_device,
            audio_cpp_threads=default_render.audio_cpp_threads,
            audio_cpp_timeout=default_render.audio_cpp_timeout,
            audio_cpp_max_batch=default_render.audio_cpp_max_batch,
            default_voice_dict=default_voice.to_dict(),
        )

    def _load_project_into_ui(self, saved_project) -> None:
        self.current_project_path = None
        self.plan = saved_project.plan
        self.input_path.setText(saved_project.input_path)
        self.outdir_path.setText(saved_project.output_path)
        self.output_name.setText(str(saved_project.render_settings.metadata.get("output_filename", "")))
        self.export_srt_check.setChecked(bool(saved_project.render_settings.metadata.get("export_srt")))
        self.monologue_check.setChecked(saved_project.render_settings.monologue)
        self.variant_combo.setCurrentText(saved_project.render_settings.model_variant)
        self._refresh_language_options()
        self._set_correction_mode(saved_project.render_settings.correction_mode)
        self.loudness_combo.setCurrentText(saved_project.render_settings.loudness_preset)
        self.crossfade_spin.setValue(saved_project.render_settings.crossfade_ms)
        backend_index = self.inference_backend_combo.findData(saved_project.render_settings.inference_backend)
        if backend_index >= 0:
            self.inference_backend_combo.setCurrentIndex(backend_index)
        cuda_target = (
            f"cuda:{saved_project.render_settings.cuda_device}"
            if saved_project.render_settings.device_mode == "cuda"
            and saved_project.render_settings.cuda_device is not None
            else "cpu"
        )
        cuda_index = self.pytorch_device_combo.findData(cuda_target)
        self.pytorch_device_combo.setCurrentIndex(cuda_index if cuda_index >= 0 else 0)
        self._set_audio_cpp_device_value(saved_project.render_settings.audio_cpp_device)
        self._set_audio_cpp_threads_value(saved_project.render_settings.audio_cpp_threads)
        self._set_audio_cpp_timeout_value(saved_project.render_settings.audio_cpp_timeout)
        self._set_audio_cpp_max_batch_value(saved_project.render_settings.audio_cpp_max_batch)
        self._refresh_inference_backend_options()
        # Rebuild the full cast from the manifest: settings are applied into
        # the matching panels, and a smaller manifest deletes stale panels
        # rather than leaving orphaned voices behind.
        requested = normalize_cast_keys(saved_project.speaker_settings.keys()) or ["A", "B"]
        self.apply_cast(
            requested,
            {},
            {key: saved_project.speaker_settings[key] for key in requested if key in saved_project.speaker_settings},
        )
        # apply_cast defaults monologue from cast size; the manifest's flag wins.
        self.monologue_check.setChecked(saved_project.render_settings.monologue)
        self._populate_table(self.plan)

    def _current_saved_project(self):
        if not self.plan:
            self.prepare_project()
        if not self.plan:
            raise ValueError("No project is available to save.")
        self._sync_plan_from_table()
        return build_saved_project(self.plan, self._render_settings(), self._speaker_settings())

    def _widget_snapshot(self) -> WidgetSnapshot:
        """Every widget read the current-payload builder needs, in one place."""
        device_mode, cuda_device = self._pytorch_device_selection()
        return WidgetSnapshot(
            cast_keys=self.cast_keys(),
            speaker_names=dict(self._speaker_names),
            model_variant=self.variant_combo.currentText(),
            correction_mode=self.correction_mode_combo.currentData() or self.correction_mode_combo.currentText(),
            loudness_preset=self.loudness_combo.currentText(),
            crossfade_ms=self.crossfade_spin.value(),
            inference_backend=self.inference_backend_combo.currentData() or "pytorch",
            device_mode=device_mode,
            cuda_device=cuda_device,
            output_dir=self.outdir_path.text() or str(self.paths.output_dir),
            output_filename=self.output_name.text(),
            export_srt=self.export_srt_check.isChecked(),
            monologue=self.monologue_check.isChecked(),
            delete_confirm_enabled=self.delete_confirm_enabled,
            output_filename_warning_enabled=self.output_filename_warning_enabled,
            audio_cpp_values={
                "audio_cpp_device": self._audio_cpp_device_value(),
                "audio_cpp_threads": self._audio_cpp_threads_value(),
                "audio_cpp_timeout": self._audio_cpp_timeout_value(),
                "audio_cpp_max_batch": self._audio_cpp_max_batch_value(),
            },
            speaker_settings=self._speaker_settings(),
        )

    def _current_gui_settings_payload(self) -> dict:
        return current_gui_settings_payload(self._widget_snapshot())

    def _apply_gui_settings_payload(self, payload: dict) -> None:
        defaults = self._default_gui_settings_payload()
        project = {**defaults["project"], **payload["project"]}
        self.variant_combo.setCurrentText(project.get("model_variant", "standard"))
        self._refresh_language_options()
        self._set_correction_mode(project.get("correction_mode", "moderate"))
        self.loudness_combo.setCurrentText(project.get("loudness_preset", RenderSettings().loudness_preset))
        self.crossfade_spin.setValue(int(project.get("crossfade_ms", 20)))
        device_mode = str(project.get("device_mode", payload.get("device_mode", "cpu")))
        cuda_device = project.get("cuda_device")
        target_device = "cpu"
        if device_mode == "cuda" and cuda_device is not None:
            target_device = f"cuda:{int(cuda_device)}"
        device_index = self.pytorch_device_combo.findData(target_device)
        self.pytorch_device_combo.setCurrentIndex(device_index if device_index >= 0 else 0)
        self.outdir_path.setText(str(project.get("output_dir", self.paths.output_dir)))
        self._output_name_edited = bool(project.get("output_filename", ""))
        self.output_name.setText(normalize_output_filename(str(project.get("output_filename", ""))))
        self.export_srt_check.setChecked(bool(project.get("export_srt", False)))
        self.delete_confirm_enabled = bool(project.get("delete_confirm_enabled", True))
        self.output_filename_warning_enabled = bool(
            project.get("output_filename_warning", self.output_filename_warning_enabled)
        )
        if hasattr(self, "output_filename_warning_action"):
            self.output_filename_warning_action.blockSignals(True)
            self.output_filename_warning_action.setChecked(self.output_filename_warning_enabled)
            self.output_filename_warning_action.blockSignals(False)
        backend_index = self.inference_backend_combo.findData(project.get("inference_backend", "pytorch"))
        if backend_index >= 0:
            self.inference_backend_combo.setCurrentIndex(backend_index)
        self._set_audio_cpp_device_value(project.get("audio_cpp_device"))
        self._set_audio_cpp_threads_value(project.get("audio_cpp_threads"))
        self._set_audio_cpp_timeout_value(project.get("audio_cpp_timeout"))
        self._set_audio_cpp_max_batch_value(project.get("audio_cpp_max_batch"))
        # A hand-edited profile could pair turbo with vulkan; the turbo guard
        # falls back to pytorch instead of leaving the invalid combination
        # selected (RenderSettings would otherwise pass it through to a
        # confusing engine error at render time).
        self._refresh_inference_backend_options()
        # Rebuild the whole cast from the payload's ordered "cast" list
        # (falling back to the saved speaker keys, then A/B). Loading fewer
        # speakers than are currently shown deletes the stale panels outright
        # instead of leaving orphaned voices behind; blend (hybrid) fields
        # restore through _speaker_config_from_payload.
        speakers = payload.get("speakers", {})
        if not isinstance(speakers, dict):
            speakers = {}
        requested = cast_request_from_payload(payload)
        cast_settings: dict[str, SpeakerSettings] = {}
        names: dict[str, str] = {}
        for key in requested:
            data = speakers.get(key)
            if isinstance(data, dict) and data:
                default_config = defaults["speakers"].get(key, defaults["speakers"]["A"])
                merged = {**default_config, **data}
                cast_settings[key] = self._speaker_config_from_payload(key, merged)
                names[key] = str(merged.get("name", "") or "")
        self.apply_cast(requested, names, cast_settings)
        # apply_cast defaults monologue from cast size; an explicit saved flag
        # (e.g. monologue with two configured voices) wins.
        self.monologue_check.setChecked(bool(project.get("monologue", False)))
        self._refresh_reference_pickers()

    def reset_settings_to_defaults(self) -> None:
        self._apply_gui_settings_payload(self._default_gui_settings_payload())
        self.error_panel.append("Settings reset to defaults.")

    def save_settings_profile(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Settings",
            str(self.paths.profile_dir / "oracle_profile.json"),
            "Settings Files (*.json)",
        )
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        payload = self._current_gui_settings_payload()
        try:
            save_gui_settings(destination, payload)
            self.error_panel.append(f"Saved settings: {destination}")
        except Exception as exc:
            self.error_panel.append(f"Save settings failed: {exc}")
            QMessageBox.critical(self, "Save Settings Failed", str(exc))

    def load_settings_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Settings", str(self.paths.profile_dir), "Settings Files (*.json)")
        if not path:
            return
        try:
            self._apply_gui_settings_payload(load_gui_settings(path))
            for speaker in self._speaker_settings().values():
                if speaker.reference_path:
                    remember_recent_reference_path(speaker.reference_path)
            self._refresh_reference_pickers()
            self.error_panel.append(f"Loaded settings: {path}")
        except Exception as exc:
            self.error_panel.append(f"Load settings failed: {exc}")
            QMessageBox.critical(self, "Load Settings Failed", str(exc))

    def save_template_profile(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Template", "Template name")
        if not ok or not name.strip():
            return
        payload = self._current_gui_settings_payload()
        payload["name"] = name.strip()
        try:
            destination = save_template(name.strip(), payload)
            self.error_panel.append(f"Saved template: {destination}")
        except Exception as exc:
            self.error_panel.append(f"Save template failed: {exc}")
            QMessageBox.critical(self, "Save Template Failed", str(exc))

    def _rebuild_templates_menu(self) -> None:
        self.templates_menu.clear()
        names = list_templates()
        if not names:
            empty = QAction("No Templates Saved", self)
            empty.setEnabled(False)
            self.templates_menu.addAction(empty)
            return
        for name in names:
            action = QAction(name, self)
            action.triggered.connect(lambda _checked=False, current=name: self._load_template_by_name(current))
            self.templates_menu.addAction(action)

    def _load_template_by_name(self, name: str) -> None:
        try:
            self._apply_gui_settings_payload(load_template(name))
            for speaker in self._speaker_settings().values():
                if speaker.reference_path:
                    remember_recent_reference_path(speaker.reference_path)
            self._refresh_reference_pickers()
            self.error_panel.append(f"Loaded template: {name}")
        except GUISettingsError as exc:
            self.error_panel.append(f"Load template failed: {exc}")
            QMessageBox.critical(self, "Load Template Failed", str(exc))

    def new_project(self) -> None:
        """Clear the current document (input path, analysis table, plan) while
        intentionally preserving the voice configuration (speaker reference
        clips, voice settings, render settings, output folder).  This mirrors
        the expected workflow: load a new script into an already-configured
        session without having to re-enter reference paths every time."""
        self.current_project_path = None
        self.plan = None
        preserved_output_name = self.output_name.text()
        preserved_output_edited = self._output_name_edited
        self.input_path.clear()
        self.output_name.setText(preserved_output_name)
        self._output_name_edited = preserved_output_edited or bool(preserved_output_name)
        self.error_panel.clear()
        self.table.setRowCount(0)
        self._refresh_reference_pickers()

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Project Files (*.json)")
        if not path:
            return
        try:
            saved_project = load_project_manifest(path)
            self._load_project_into_ui(saved_project)
            self.current_project_path = Path(path)
            for speaker in saved_project.speaker_settings.values():
                if speaker.reference_path:
                    remember_recent_reference_path(speaker.reference_path)
            self._refresh_reference_pickers()
            self.error_panel.append(f"Loaded project: {path}")
        except Exception as exc:
            self.error_panel.append(f"Open failed: {exc}")
            QMessageBox.critical(self, "Open Project Failed", str(exc))

    def save_project(self) -> None:
        if self.current_project_path is None:
            self.save_project_as()
            return
        self._save_project_to_path(self.current_project_path)

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", "Project Files (*.json)")
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        self._save_project_to_path(destination)

    def _save_project_to_path(self, path: Path) -> None:
        try:
            saved_project = self._current_saved_project()
            save_project_manifest(path, saved_project)
            self.current_project_path = path
            self.error_panel.append(f"Saved project: {path}")
        except Exception as exc:
            self.error_panel.append(f"Save failed: {exc}")
            QMessageBox.critical(self, "Save Project Failed", str(exc))

    def _batch_fix_input_folder(self) -> None:
        """Delegate to the gui_ingest_tools flow body (extraction slice 10)."""
        from the_oracle.gui_ingest_tools import batch_fix_input_folder

        # INJECTION seam: the bare QMessageBox/QFileDialog names resolve from
        # THIS module's globals at call time, so app_gui-level rebinds keep
        # intercepting the moved body exactly as before the move.
        batch_fix_input_folder(self, message_box_cls=QMessageBox, file_dialog_cls=QFileDialog)

    def _color_rule_labels_in_view(self, view, rules: list[str]) -> None:
        """Delegate to the single owner in gui_ingest."""
        from the_oracle.gui_ingest import color_rule_labels_in_view

        color_rule_labels_in_view(view, rules)

    def _show_batch_fix_preview_dialog(
        self,
        folder: str,
        fixes: list,
        warnings_only: list,
    ) -> list:
        """Folder tree + per-file diff preview; the fixes to apply, or []."""
        from the_oracle.gui_ingest import run_batch_fix_preview

        return run_batch_fix_preview(self, folder, fixes, warnings_only, dialog_cls=QDialog)

    def _run_ingest_transformer_check(self) -> bool:
        """Delegate to the gui_ingest_tools flow body (extraction slice 10)."""
        from the_oracle.gui_ingest_tools import run_ingest_transformer_check

        # INJECTION seam: the bare QMessageBox name resolves from THIS
        # module's globals at call time, so app_gui-level rebinds keep
        # intercepting the moved body exactly as before the move.
        return run_ingest_transformer_check(self, message_box_cls=QMessageBox)

    def _speaker_ref_hint_lines(self, input_file: str) -> list[str]:
        """--speaker-ref suggestions for a script's cast, for the popup list.

        Computed from the *post-transform* text, so a cast only visible after a
        fix (or a subtitle conversion) is still suggested. The wording comes
        from ``speaker_ref_hints`` -- the same owner the CLI prints from -- so
        the advice cannot read differently depending on where it appears; only
        the bullet is this panel's.
        """
        from the_oracle.ingest_transformer import speaker_ref_report_for_file
        from the_oracle.speaker_ref_hints import sentence_lines

        try:
            refs, rejected = speaker_ref_report_for_file(Path(input_file), None)
        except Exception:
            return []
        return sentence_lines(refs, rejected, bullet=_SPEAKER_HINT_BULLET)

    def _input_file_is_trusted(self, input_file: str) -> bool:
        """True when the user pre-approved auto-fixes for this exact file."""
        return input_file_is_trusted(self._app_settings, input_file)

    def _remember_trusted_input_file(self, input_file: str) -> None:
        """Add the file to the trusted list and persist app settings."""
        remember_trusted_input_file(self._app_settings, input_file)
        if self._app_settings_ready:
            try:
                save_app_settings(self._app_settings)
            except OSError:
                pass

    def _clear_trusted_input_files(self) -> None:
        """Forget every pre-approved file (Settings menu action)."""
        clear_trusted_input_files(self._app_settings)
        if self._app_settings_ready:
            try:
                save_app_settings(self._app_settings)
                self.error_panel.append(
                    "Cleared the remembered auto-fix approvals; every input file "
                    "will be reviewed again before a fix."
                )
            except OSError:
                pass

    def _remember_format_backup(self, fixed_file: str, backup_path: str | None) -> None:
        """Record a fix's backup so it can be restored from Settings later."""
        remember_format_backup(self._app_settings, fixed_file, backup_path)
        if self._app_settings_ready:
            try:
                save_app_settings(self._app_settings)
            except OSError:
                pass

    def _restore_most_recent_format_backup(self) -> None:
        """Delegate to the gui_ingest_tools flow body (extraction slice 10)."""
        from the_oracle.gui_ingest_tools import restore_most_recent_format_backup

        # INJECTION seam: the bare QMessageBox name resolves from THIS
        # module's globals at call time, so app_gui-level rebinds keep
        # intercepting the moved body exactly as before the move.
        restore_most_recent_format_backup(self, message_box_cls=QMessageBox)

    def _show_fix_preview_dialog(
        self,
        original_text: str,
        fixed_text: str,
        fix_count: int,
        line_fixes: list | None = None,
        input_file: str | None = None,
    ) -> bool:
        """Show the exact rewrite for review side by side; True when accepted.

        Two aligned panes: the original script on the left, the corrected
        script on the right. Rewritten rows are paired line-for-line and the
        right-hand cell is labeled with the fix rule that produced it (e.g.
        ``[dash/pipe separator]``); rows scroll in sync so long scripts stay
        reviewable.
        """
        from the_oracle.gui_ingest import run_fix_preview

        result = run_fix_preview(
            self,
            original_text,
            fixed_text,
            fix_count,
            line_fixes,
            input_file,
            dialog_cls=QDialog,
            saved_geometry=self._app_settings.get("preview_dialog_geometry"),
        )
        # Remember the dialog's size and position for the next session
        # (persists through the app-settings file alongside the workspace).
        if self._app_settings_ready and result.geometry:
            self._app_settings["preview_dialog_geometry"] = result.geometry
            try:
                save_app_settings(self._app_settings)
            except Exception as exc:
                self.error_panel.append(f"Could not persist preview dialog geometry: {exc}")
        if result.accepted and result.remember and input_file:
            self._remember_trusted_input_file(input_file)
        return result.accepted

    def prepare_project(self) -> None:
        with self._prewarm_lock:
            if self._prewarm_state == "warming":
                self.error_panel.append("Background warmup still running; please wait a moment before analyzing.")
                return
        analyze_click_wall = time()
        self._log_action_timing("analyze_click", analyze_click_wall)
        # A subtitle path (typed, remembered, or loaded from a project) is
        # converted before the transformer check, mirroring the file
        # picker; the field is re-pointed at the script for this run.
        converted = self._convert_subtitle_input(self.input_path.text().strip())
        if converted is None:
            return
        if converted != self.input_path.text().strip():
            self.input_path.setText(converted)
        if not self._run_ingest_transformer_check():
            self.error_panel.append("Analysis cancelled: fix the input file formatting and try again.")
            return
        try:
            # Lazily create the output folder now that analysis actually needs
            # it; typing in the field alone never creates directories.
            output_dir = str(self._ensure_outdir_exists(self.outdir_path.text()))
            self.plan = self._pipeline().prepare_plan(
                self.input_path.text(),
                output_dir,
                self._speaker_settings(),
                self._render_settings(),
            )
            plan_ready_wall = time()
            self._log_action_timing("plan_ready", plan_ready_wall, {"elapsed": plan_ready_wall - analyze_click_wall})
            for speaker in self._speaker_settings().values():
                if speaker.reference_path:
                    remember_recent_reference_path(speaker.reference_path)
            # The text may mention more speakers than the current cast holds:
            # extend the cast so every plan speaker is configurable (and
            # rendered with its intended voice) instead of silently falling
            # back to defaults.
            detected = sorted({item.speaker for item in self.plan.utterances})
            extended = normalize_cast_keys([*self._cast, *detected])
            if extended != self._cast:
                self.apply_cast(extended, dict(self._speaker_names), self._speaker_settings())
            self._populate_table(self.plan)
            self.error_panel.append("Analysis complete.")
        except Exception as exc:
            self.error_panel.append(str(exc))
            QMessageBox.critical(self, "Analysis Failed", str(exc))

    def _populate_table(self, plan: RenderPlan) -> None:
        self.table.setRowCount(len(plan.utterances))
        for row, utterance in enumerate(plan.utterances):
            index_item = QTableWidgetItem(str(utterance.index))
            # The Index/Duration columns are computed, not inputs: default
            # items are editable, and a user typing into them would corrupt
            # the display until the next repopulate.
            index_item.setFlags(index_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, index_item)
            speaker_combo = QComboBox()
            speakers = sorted(set(self._cast) | {item.speaker for item in plan.utterances})
            speaker_combo.addItems(speakers or ["A", "B"])
            speaker_combo.setCurrentText(utterance.speaker)
            self.table.setCellWidget(row, 1, speaker_combo)
            original_item = QTableWidgetItem(utterance.original_text)
            # Source text is the untouched input record; only Repaired Text
            # (column 3) is meant to be edited.
            original_item.setFlags(original_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, original_item)
            repaired = QTableWidgetItem(utterance.repaired_text)
            repaired.setFlags(repaired.flags() | Qt.ItemIsEditable)
            self.table.setItem(row, 3, repaired)
            emotion = self._create_emotion_combo(utterance.emotion)
            self.table.setCellWidget(row, 4, emotion)
            duration = "" if utterance.duration_seconds is None else f"{utterance.duration_seconds:.2f}s"
            duration_item = QTableWidgetItem(duration)
            duration_item.setFlags(duration_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 5, duration_item)
            # Show status in a dedicated column
            status = QTableWidgetItem(utterance.status)
            status.setFlags(status.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 6, status)
            preview = QPushButton("Preview")
            preview.clicked.connect(lambda _checked=False, current=row: self.preview_utterance(current))
            self.table.setCellWidget(row, 7, preview)
            control = self._create_row_action(row)
            self.table.setCellWidget(row, 8, control)

    def _create_row_action(self, row: int) -> QComboBox:
        control = QComboBox()
        control.addItems(["+/-", "Extra", "Remove"])
        control.setMaximumWidth(80)
        control.currentIndexChanged.connect(lambda idx, r=row, c=control: self._handle_row_action(idx, r, c))
        return control

    def _create_emotion_combo(self, value: str) -> QComboBox:
        combo = QComboBox()
        for emotion in SUPPORTED_EMOTIONS:
            combo.addItem(emotion, emotion)
        if value and value not in SUPPORTED_EMOTIONS:
            combo.addItem(value, value)
        target = value if value else "neutral"
        idx = combo.findData(target)
        if idx < 0:
            idx = combo.findData("neutral")
        combo.setCurrentIndex(max(0, idx))
        return combo

    def _handle_row_action(self, idx: int, row: int, control: QComboBox) -> None:
        if idx == 0 or not self.plan:
            return
        # Sync the table into the plan BEFORE mutating it: the repopulate at
        # the end rebuilds every row from the plan, so unsaved edits in other
        # rows (repaired text, speaker, emotion) would be silently discarded.
        self._sync_plan_from_table()
        if idx == 1:
            self.plan.utterances.insert(row + 1, self._blank_utterance())
        elif idx == 2 and 0 <= row < len(self.plan.utterances):
            utterance = self.plan.utterances[row]
            if self._needs_delete_confirmation(utterance):
                if not self._confirm_delete():
                    control.blockSignals(True)
                    control.setCurrentIndex(0)
                    control.blockSignals(False)
                    return
            self.plan.utterances.pop(row)
        control.blockSignals(True)
        control.setCurrentIndex(0)
        control.blockSignals(False)
        self._reindex_utterances()
        self._populate_table(self.plan)

    def _blank_utterance(self) -> Utterance:
        return Utterance(
            index=0,
            original_text="",
            repaired_text="",
            speaker="A",
            emotion="neutral",
            duration_seconds=None,
        )

    def _reindex_utterances(self) -> None:
        if not self.plan:
            return
        for idx, utterance in enumerate(self.plan.utterances):
            utterance.index = idx

    def _needs_delete_confirmation(self, utterance: Utterance) -> bool:
        return self.delete_confirm_enabled and any(
            getattr(utterance, attr) for attr in ("original_text", "repaired_text", "emotion")
        )

    def _confirm_delete(self) -> bool:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Confirm Delete")
        dialog.setText("This row contains text. Look ready to delete?")
        checkbox = QCheckBox("Click here to hide this window into the program settings menu above.", dialog)
        dialog.setCheckBox(checkbox)
        dialog.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        result = dialog.exec()
        if checkbox.isChecked():
            self.delete_confirm_enabled = False
        return result == QMessageBox.Ok

    def _enable_delete_confirmation(self) -> None:
        self.delete_confirm_enabled = True
        self.error_panel.append("Delete confirmations re-enabled.")

    def _sync_plan_from_table(self) -> None:
        if not self.plan:
            return
        for row, utterance in enumerate(self.plan.utterances):
            speaker_widget = self.table.cellWidget(row, 1)
            if isinstance(speaker_widget, QComboBox):
                selected_speaker = speaker_widget.currentText()
                utterance.manual_speaker_override = utterance.manual_speaker_override or selected_speaker != utterance.speaker
                utterance.speaker = selected_speaker
            repaired_item = self.table.item(row, 3)
            if repaired_item:
                repaired_text = repaired_item.text().strip()
                utterance.manual_text_override = utterance.manual_text_override or repaired_text != utterance.repaired_text
                utterance.repaired_text = repaired_text
            emotion_widget = self.table.cellWidget(row, 4)
            if isinstance(emotion_widget, QComboBox):
                emotion_text = emotion_widget.currentData() or emotion_widget.currentText()
                utterance.manual_emotion_override = utterance.manual_emotion_override or emotion_text != utterance.emotion
                utterance.emotion = emotion_text
        speaker_settings = self._speaker_settings()
        self._sync_extra_speaker_groups(list(speaker_settings))
        merged_profiles = dict(self.plan.voice_profiles)
        for speaker, config in speaker_settings.items():
            if speaker in merged_profiles:
                merged_profiles[speaker] = replace(
                    merged_profiles[speaker], engine_params=config.voice_settings
                )
        self.plan.voice_profiles = merged_profiles
        self.plan.source_path = self.input_path.text()
        self.plan.output_dir = self.outdir_path.text()
        self.plan.metadata["model_variant"] = self.variant_combo.currentText()
        speaker_languages = {
            speaker: VoiceSettings.from_mapping(config.voice_settings).language
            for speaker, config in speaker_settings.items()
        }
        self.plan.metadata["language"] = speaker_languages["A"] if len(set(speaker_languages.values())) == 1 else "mixed"
        self.plan.update_hashes()

    def preview_utterance(self, row: int) -> None:
        if not self.plan:
            return
        if self.render_worker is not None:
            self.error_panel.append("Preview is unavailable while a render is in progress.")
            return
        if self.preview_worker is not None:
            self.error_panel.append("Preview is already in progress.")
            return
        try:
            self._sync_plan_from_table()
            utterance = self.plan.utterances[row]
        except Exception as exc:
            self.error_panel.append(f"Preview failed: {exc}")
            QMessageBox.critical(self, "Preview Failed", str(exc))
            return
        # Resolve the voice profile BEFORE the UI goes busy: a speaker with no
        # profile (stale row, edited cast) must report gracefully instead of
        # raising KeyError mid-startup and leaving the UI stuck busy.
        profile = self.plan.voice_profiles.get(utterance.speaker)
        if profile is None:
            message = (
                f"Preview failed: no voice profile for Speaker {utterance.speaker}. "
                "Re-run Analyze so every row's speaker has a voice assigned."
            )
            self.error_panel.append(message)
            QMessageBox.warning(self, "Preview Failed", message)
            return

        self.preview_dialog = RenderProgressDialog(self, title="Generating Preview")
        self.preview_dialog.show()
        # A new preview session starts its own self-heal tally (the tally
        # deliberately survives set_idle; this is its reset point).
        self.live_panel.reset_retry_tally()
        self._set_preview_busy(True)
        inference_backend = self.inference_backend_combo.currentData() or "pytorch"
        if inference_backend == "vulkan" and _vulkan_prerequisite_missing():
            # Auto-setup: the preview waits for the CPU→GPU switch to finish
            # instead of failing deep in the worker. The row is remembered so
            # the completion handler can re-fire this exact preview by itself
            # (no second click); explicit clicks always retry.
            self._start_vulkan_setup()
            self._preview_queued_after_setup = True
            self._preview_row_queued_after_setup = row
            self.preview_dialog.close()
            self.preview_dialog = None
            # The queued preview's dialog is gone: idle the persistent
            # sidebar half too — the same mirror rule as _finish_render.
            self.live_panel.set_idle()
            self._set_preview_busy(False)
            self.error_panel.append(
                "Preview queued: the Vulkan backend is being set up automatically; "
                "the preview will start by itself once setup completes."
            )
            QMessageBox.information(
                self,
                "Vulkan Backend Setup",
                "The Vulkan backend is being set up automatically (building audiocpp_cli "
                "and/or downloading the Chatterbox model). The preview will start by "
                "itself once the setup finishes.",
            )
            return
        # Vulkan-only knobs: only forwarded (like _render_settings) when the
        # Vulkan backend is actually selected; render_preview ignores them on
        # the PyTorch path regardless.
        is_vulkan = inference_backend == "vulkan"
        device_mode, cuda_device = self._pytorch_device_selection()
        self.preview_worker = PreviewWorker(
            utterance,
            profile,
            self.variant_combo.currentText(),
            device_mode,
            pipeline=self._pipeline(),
            cuda_device=cuda_device,
            inference_backend=inference_backend,
            audio_cpp_device=self._audio_cpp_device_value() if is_vulkan else None,
            audio_cpp_threads=self._audio_cpp_threads_value() if is_vulkan else None,
            audio_cpp_timeout=self._audio_cpp_timeout_value() if is_vulkan else None,
            audio_cpp_max_batch=self._audio_cpp_max_batch_value() if is_vulkan else None,
            # Preview synthesizes the model in an isolated child process for
            # the same reason render does: loading Chatterbox/PyTorch inside
            # the Qt Multimedia process segfaults on Ubuntu (SIGSEGV / 245).
            run_in_subprocess=True,
            python_executable=sys.executable,
            repo_root=self.repo_root,
        )
        self.preview_worker.progress.connect(self._update_preview_progress)
        self.preview_worker.completed.connect(lambda path: self._finish_preview(row, path))
        self.preview_worker.failed.connect(self._fail_preview)
        self.preview_worker.finished.connect(self._cleanup_preview_worker)
        try:
            self.preview_worker.start()
        except Exception as exc:
            # Worker startup must never leave the UI stuck busy with a
            # dangling progress dialog: tear both down and report.
            worker = self.preview_worker
            self.preview_worker = None
            if worker is not None:
                worker.deleteLater()
            self._set_preview_busy(False)
            if self.preview_dialog is not None:
                self.preview_dialog.close()
                self.preview_dialog = None
            # Startup failure is still a dismissal: idle the persistent
            # sidebar half — the same mirror rule as _finish_render.
            self.live_panel.set_idle()
            self.error_panel.append(f"Preview failed: {exc}")
            QMessageBox.critical(self, "Preview Failed", str(exc))

    def render_project(self) -> None:
        with self._prewarm_lock:
            if self._prewarm_state == "warming":
                self.error_panel.append("Background warmup still running; render will be available shortly.")
                QMessageBox.information(self, "Warmup In Progress", "Background warmup is still running. Please try Render again in a moment.")
                return
        if self.preview_worker is not None:
            self.error_panel.append("Wait for the active preview to finish before rendering.")
            return
        if not self.plan:
            message = "Analyze the project before rendering so render work stays off the UI thread."
            self.error_panel.append(message)
            QMessageBox.information(self, "Analyze First", message)
            return
        if self.render_worker is not None:
            self.error_panel.append("Render is already in progress.")
            return
        try:
            self._sync_plan_from_table()
            output_filename = resolve_output_filename(
                self.input_path.text(),
                self.outdir_path.text(),
                self._default_output_dir,
                self.output_name.text(),
            )
            if not output_filename:
                raise ValueError("Choose an output filename before rendering outside the default Output folder.")
            # Lazily create the output folder now that a render actually needs
            # it; typing in the field alone never creates directories.
            self.plan.output_dir = str(self._ensure_outdir_exists(self.outdir_path.text()))
        except Exception as exc:
            self.error_panel.append(f"Render failed: {exc}")
            QMessageBox.critical(self, "Render Failed", str(exc))
            return

        render_settings = self._render_settings()
        resolved_default_name = resolve_output_filename(
            self.input_path.text(), self.outdir_path.text(), self._default_output_dir, ""
        )
        auto_name = default_output_filename(self.input_path.text())
        current_output_name = normalize_output_filename(self.output_name.text())
        if (
            self.output_filename_warning_enabled
            and not self._output_name_edited
            and resolved_default_name
            and resolved_default_name == auto_name
            and current_output_name in {"", auto_name}
        ):
            warning = QMessageBox(self)
            warning.setIcon(QMessageBox.Icon.Warning)
            warning.setWindowTitle("Generic output filename")
            warning.setText("This render will use an automatically derived filename.")
            warning.setInformativeText(
                "The output will be named from the input file. Close this warning "
                "and enter a descriptive filename now if you want a clearer name."
            )
            disable = QCheckBox(
                "Click here to disable this warning; re-enable it in the Settings menu"
            )
            warning.setCheckBox(disable)
            warning.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            warning.setDefaultButton(QMessageBox.StandardButton.Cancel)
            result = warning.exec()
            if disable.isChecked():
                self.output_filename_warning_enabled = False
                self._app_settings["output_filename_warning"] = False
                self._persist_workspace_layout()
            if result == QMessageBox.StandardButton.Cancel:
                return

        if render_settings.inference_backend == "vulkan":
            missing = _vulkan_prerequisite_missing()
            if missing:
                # Explicit user action: always (re)start the auto-setup — the
                # guard on _vulkan_setup_thread prevents double-starting, and
                # _start_vulkan_setup resets the failure state so a stale
                # setup can't leave the render queued forever.
                self._start_vulkan_setup()
                self._render_queued_after_setup = True
                message = (
                    "The Vulkan backend is being set up automatically (building "
                    "audiocpp_cli and/or downloading the Chatterbox model). "
                    "The render will start by itself once setup completes."
                )
                self.error_panel.append(f"Render queued: {message}")
                QMessageBox.information(self, "Vulkan Backend Setup", message)
                return

        render_click_wall = time()
        self._log_action_timing("render_click", render_click_wall)
        self.progress_dialog = RenderProgressDialog(self, title="Rendering")
        self.progress_dialog.show()
        # A new render session starts its own self-heal tally (the tally
        # deliberately survives set_idle; this is its reset point).
        self.live_panel.reset_retry_tally()
        self._set_render_busy(True)
        render_settings.metadata["output_filename"] = output_filename
        self.render_worker = RenderWorker(
            self.plan,
            render_settings,
            # Native Chatterbox/Perth synthesis must never initialize in the
            # Qt Multimedia process. The child starts with no QApplication and
            # therefore cannot reproduce the QMediaPlayer + QThread segfault.
            run_in_subprocess=True,
            python_executable=sys.executable,
            repo_root=self.repo_root,
            render_click_wall=render_click_wall,
        )
        self.render_worker.progress.connect(self._update_render_progress)
        self.render_worker.completed.connect(self._finish_render)
        self.render_worker.failed.connect(self._fail_render)
        self.render_worker.finished.connect(self._cleanup_render_worker)
        self.render_worker.start()

    def _update_render_progress(self, progress: RenderProgress) -> None:
        self.live_panel.update_from_progress(progress)
        if self.progress_dialog is not None:
            self.progress_dialog.update_progress(progress)
        if progress.retry_note:
            # A gate-rejected synthesis self-healed through the engine's
            # one-shot fresh-seed retry. Log it so self-healing is visible —
            # a render that quietly fixed itself is a fact the user should
            # see, not one they only discover by reading engine logs.
            self.error_panel.append(progress.retry_note)

    def _finish_render(self, plan_payload: dict, output_path: str) -> None:
        self.plan = RenderPlan.from_dict(plan_payload)
        self._populate_table(self.plan)
        self.error_panel.append(f"Render complete: {output_path}")
        srt_path = self.plan.metadata.get("srt_path")
        if srt_path:
            self.error_panel.append(f"Subtitles written: {srt_path}")
        retry_note_count = self.plan.metadata.get("synthesis_retries")
        if retry_note_count:
            # Mirrors the live hiccup-retry notes logged in
            # _update_render_progress: the summary names the total so a render
            # that self-healed is visible even after the log scrolled.
            self.error_panel.append(
                f"Recovered from {retry_note_count} synthesis hiccup{'s' if retry_note_count != '1' else ''} "
                "(automatic one-shot retry at a fresh seed)"
            )
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None
        # The sidebar is the persistent half of the same progress mirror: it
        # must return to idle with the dialog, or the finished render's last
        # frame (100%, its final stage) stays on screen indefinitely.
        self.live_panel.set_idle()

    def _fail_render(self, plan_payload: object, message: str | None = None) -> None:
        # RenderWorker includes its defensive plan copy in the failure signal.
        # This keeps partial row statuses/durations even if Qt delivers the
        # finished/cleanup slot before this queued failure handler.
        if message is None:
            # Backward-compatible path for direct callers/tests that only have
            # an error string; normal worker failures always use the payload.
            message = str(plan_payload)
            if self.render_worker is not None:
                self.plan = RenderPlan.from_dict(self.render_worker.plan.to_dict())
        elif isinstance(plan_payload, dict):
            self.plan = RenderPlan.from_dict(plan_payload)
        self.error_panel.append(f"Render failed: {message}")
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None
        # A failed render would otherwise leave the sidebar frozen on its last
        # frame — the same persistent-mirror rule as _finish_render.
        self.live_panel.set_idle()

        # Refresh the table to show truthful row state after failure:
        # completed rows show their status/duration, failed rows show failed,
        # and rows never reached remain pending.
        if self.plan is not None:
            self._populate_table(self.plan)

        # Show failure summary with row-level information
        failed_rows = self.plan.metadata.get("failed_rows", "") if self.plan is not None else ""
        if failed_rows:
            QMessageBox.critical(self, "Render Failed", 
                               f"Render failed. Failed rows: {failed_rows}\n\nSee error panel for details.")
        else:
            QMessageBox.critical(self, "Render Failed", message)

    def _cleanup_render_worker(self) -> None:
        self._set_render_busy(False)
        if self.render_worker is not None:
            self.render_worker.deleteLater()
            self.render_worker = None

    def _update_preview_progress(self, progress: RenderProgress) -> None:
        self.live_panel.update_from_progress(progress)
        if self.preview_dialog is not None:
            self.preview_dialog.update_progress(progress)
        if progress.retry_note:
            self.error_panel.append(progress.retry_note)

    def _on_preview_playback_status(self, status) -> None:  # type: ignore[no-untyped-def]
        # NEVER call stop()/deleteLater() on the player from inside its own
        # mediaStatusChanged emission — that is a QtMultimedia use-after-free
        # (backend threads still mid-emission). Defer any stop out of the
        # handler with a zero-timer so the backend finishes delivering first.
        end_state = getattr(status, "EndOfMedia", None)
        if end_state is not None and status == end_state:
            # Context form: the pending stop dies with this window instead of
            # riding the context-less overload's undocumented receiver
            # semantics.
            QTimer.singleShot(0, self, self._stop_preview_player)

    def _stop_preview_player(self) -> None:
        # GUI-thread contexts only (EndOfMedia deferral, window close) — the
        # persistent player is stopped, never deleted, so it can be reused for
        # the next preview without backend teardown churn.
        try:
            self.player.stop()
        except Exception:
            pass

    def _finish_preview(self, row: int, preview_path: str) -> None:
        # Do NOT persist preview state to row-level render fields.
        # Preview is a probe operation, not a render. Row-level duration_seconds
        # and status represent full-render truth, not preview-local truth.
        # For chunked rows, preview duration would be first-chunk only (misleading),
        # and preview success is not the same as render success.
        # The GUI Duration/Status columns remain unchanged (showing pending or
        # previous render state) until a full render completes.

        self.player.setSource(QUrl.fromLocalFile(preview_path))
        self.player.play()
        self.error_panel.append(f"Preview ready: {preview_path}")
        if self.preview_dialog is not None:
            self.preview_dialog.close()
            self.preview_dialog = None
        # The sidebar is the persistent half of the same progress mirror (see
        # _finish_render): a finished preview must idle it too, or the last
        # frame stays on screen indefinitely.
        self.live_panel.set_idle()

    def _fail_preview(self, message: str) -> None:
        self.error_panel.append(f"Preview failed: {message}")
        if self.preview_dialog is not None:
            self.preview_dialog.close()
            self.preview_dialog = None
        # A failed preview would otherwise leave the sidebar frozen on its
        # last frame — the same persistent-mirror rule as _finish_render.
        self.live_panel.set_idle()
        QMessageBox.critical(self, "Preview Failed", message)

    def _cleanup_preview_worker(self) -> None:
        self._set_preview_busy(False)
        if self.preview_worker is not None:
            self.preview_worker.deleteLater()
            self.preview_worker = None

    def _set_busy(self, busy: bool) -> None:
        """Disable or re-enable all interactive widgets during a render or preview.

        Centralised here so adding a new interactive widget only requires one
        update rather than mirroring the change in both a render and a preview
        variant.
        """
        self.render_button.setEnabled(not busy)
        self.analyze_button.setEnabled(not busy)
        self.table.setEnabled(not busy)

    # Convenience aliases so call-sites read naturally.
    def _set_render_busy(self, busy: bool) -> None:
        self._set_busy(busy)

    def _set_preview_busy(self, busy: bool) -> None:
        self._set_busy(busy)


def launch_gui() -> None:
    launch_t0 = perf_counter()
    # Arm the native-crash net before any Qt work: launch_gui is a public
    # entry point reachable without cli.main's install/arm, and the
    # window-build phase itself is native-crash territory (the 08:06:51
    # segfault died 52s into a launch with no armed net — STATE.md Noticed,
    # 2026-09-28). Idempotent; fails closed to unarmed when consent is off.
    from the_oracle.crash import handlers as crash_handlers

    crash_handlers.arm_native_capture()
    app = QApplication.instance() or QApplication([])
    launch_marks: list[tuple[str, float]] = [("qt_app_created", perf_counter() - launch_t0)]
    window = MainWindow()
    launch_marks.append(("mainwindow_built", perf_counter() - launch_t0))
    window.show()
    try:
        log_dir = Path(__file__).resolve().parents[2] / "Output" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {"events": launch_marks}
        (log_dir / "gui_launch_timing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    app.exec()
class PrewarmThread(QThread):
    ready = Signal(object, object, dict)
    failed = Signal(str, dict)

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device

    def run(self) -> None:
        timeline: dict[str, float] = {}
        try:
            start_wall = time()
            timeline["prewarm_start"] = start_wall
            # Startup prewarm must never risk taking down the GUI process.
            # Keep automatic warmup to a lightweight timing/status pass and
            # leave heavyweight backend construction to explicit user actions.
            ready_wall = time()
            timeline["pipeline_ready"] = ready_wall
            timeline["repair_ready"] = ready_wall
            timeline["emotion_ready"] = ready_wall
            timeline["engine_ready"] = ready_wall
            timeline["prewarm_complete"] = ready_wall
            self.ready.emit(None, None, timeline)
        except Exception as exc:  # pragma: no cover - GUI-only path
            timeline["prewarm_failed"] = time()
            self.failed.emit(str(exc), timeline)
