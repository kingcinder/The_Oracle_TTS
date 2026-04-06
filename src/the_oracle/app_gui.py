"""PySide6 desktop GUI for the Chatterbox-only The Oracle app."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter, time
import json
import threading

from PySide6.QtCore import QThread, Qt, QUrl, Signal, QTimer
from PySide6.QtGui import QAction, QFont, QFontDatabase
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
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
    list_templates,
    load_gui_settings,
    load_recent_reference_paths,
    load_template,
    remember_recent_reference_path,
    save_gui_settings,
    save_template,
)
from the_oracle.models.project import RenderPlan, VoiceProfile, VoiceSettings, Utterance
from the_oracle.pipeline import OraclePipeline, RenderProgress, RenderSettings, SpeakerSettings
from the_oracle.project_manifest import build_saved_project, load_project_manifest, save_project_manifest
from the_oracle.voice_catalog import VoiceChoice, default_voice_choices
from the_oracle.tts_engines.chatterbox_engine import SUPPORTED_VARIANTS, ChatterboxEngine

# CPU is the only verified Chatterbox execution path in this project.
# Preview and render always use "cpu"; the constant is defined here so
# it can be updated in one place if a verified GPU path is added later.
_DEVICE_MODE: str = "cpu"

_DISPLAY_FONT_CANDIDATES = (
    "Aptos Display",
    "Segoe UI Variable Display",
    "SF Pro Display",
    "Inter",
    "Noto Sans",
    "DejaVu Sans",
)
_BODY_FONT_CANDIDATES = (
    "Aptos",
    "Segoe UI Variable Text",
    "SF Pro Text",
    "Inter",
    "Noto Sans",
    "DejaVu Sans",
)
_APP_STYLESHEET = """
QMainWindow, QDialog {
    background-color: #090a0d;
    color: #f3eee4;
}
QWidget#oracleRoot {
    background-color: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #111319,
        stop: 0.45 #161921,
        stop: 1 #090a0d
    );
}
QWidget#topBarCard, QWidget#contentCard {
    background-color: rgba(18, 21, 28, 0.94);
    border: 1px solid rgba(214, 179, 122, 0.16);
    border-radius: 24px;
}
QWidget#controlRail {
    background: transparent;
}
QWidget#workspacePanel {
    background: transparent;
}
QLabel {
    color: #f3eee4;
}
QLabel#appKicker {
    color: #d6b37a;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}
QLabel#appTitle {
    color: #fbf7ef;
    font-size: 26px;
    font-weight: 700;
}
QLabel#appSummary {
    color: #aab1bc;
    font-size: 13px;
}
QLabel#fieldLabel {
    color: #cbb696;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
QLabel#sectionLabel {
    color: #d9c3a0;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}
QGroupBox {
    margin-top: 14px;
    padding: 20px 16px 16px 16px;
    border-radius: 20px;
    border: 1px solid rgba(214, 179, 122, 0.18);
    background-color: rgba(24, 27, 34, 0.9);
    font-size: 13px;
    font-weight: 600;
    color: #f3eee4;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
    color: #d6b37a;
}
QMenuBar {
    background: transparent;
    color: #f3eee4;
    border: none;
    padding: 4px 10px;
}
QMenuBar::item {
    background: transparent;
    padding: 8px 12px;
    border-radius: 10px;
}
QMenuBar::item:selected {
    background-color: rgba(214, 179, 122, 0.16);
}
QMenu {
    background-color: #12141a;
    border: 1px solid rgba(214, 179, 122, 0.22);
    border-radius: 12px;
    padding: 8px;
}
QMenu::item {
    padding: 8px 16px;
    border-radius: 8px;
}
QMenu::item:selected {
    background-color: rgba(214, 179, 122, 0.16);
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget {
    background-color: rgba(10, 11, 15, 0.88);
    border: 1px solid rgba(214, 179, 122, 0.16);
    border-radius: 14px;
    color: #f6f1e7;
    padding: 10px 12px;
    selection-background-color: rgba(214, 179, 122, 0.28);
    selection-color: #fffdf7;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QTableWidget:focus {
    border: 1px solid rgba(214, 179, 122, 0.72);
}
QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    border: none;
    width: 22px;
}
QPushButton {
    min-height: 38px;
    padding: 10px 18px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background-color: rgba(37, 40, 49, 0.96);
    color: #f6f1e7;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: rgba(49, 54, 66, 0.98);
    border-color: rgba(214, 179, 122, 0.35);
}
QPushButton:pressed {
    background-color: rgba(27, 30, 37, 0.98);
}
QPushButton[accentButton="true"] {
    background-color: #d6b37a;
    color: #161312;
    border: 1px solid #f2d09a;
    font-weight: 700;
}
QPushButton[accentButton="true"]:hover {
    background-color: #e3c18b;
}
QPushButton[utilityButton="true"] {
    min-height: 34px;
    border-radius: 12px;
    background-color: rgba(20, 22, 29, 0.86);
}
QPushButton:disabled {
    background-color: rgba(42, 44, 52, 0.45);
    color: rgba(243, 238, 228, 0.45);
    border-color: rgba(255, 255, 255, 0.04);
}
QHeaderView::section {
    background-color: rgba(214, 179, 122, 0.12);
    color: #e8d5b4;
    border: none;
    border-bottom: 1px solid rgba(214, 179, 122, 0.18);
    padding: 12px 10px;
    font-size: 12px;
    font-weight: 700;
}
QTableWidget {
    gridline-color: transparent;
    alternate-background-color: rgba(255, 255, 255, 0.02);
}
QSplitter::handle {
    background: transparent;
}
QSplitter::handle:horizontal {
    width: 12px;
}
QSplitter::handle:vertical {
    height: 12px;
}
QTableCornerButton::section {
    background-color: rgba(214, 179, 122, 0.12);
    border: none;
}
QProgressBar {
    min-height: 18px;
    border-radius: 9px;
    border: 1px solid rgba(214, 179, 122, 0.18);
    background-color: rgba(10, 11, 15, 0.9);
    text-align: center;
    color: #f8f3ea;
}
QProgressBar::chunk {
    border-radius: 8px;
    background-color: #d6b37a;
}
QScrollBar:vertical {
    width: 12px;
    margin: 8px 0 8px 0;
    background: transparent;
}
QScrollBar::handle:vertical {
    min-height: 36px;
    border-radius: 6px;
    background-color: rgba(214, 179, 122, 0.34);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    height: 0;
    background: transparent;
}
"""


def _pick_font_family(candidates: tuple[str, ...]) -> str:
    families = set(QFontDatabase.families())
    for candidate in candidates:
        if candidate in families:
            return candidate
    return QFont().defaultFamily()


def _configure_application(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(_APP_STYLESHEET)
    app.setFont(QFont(_pick_font_family(_BODY_FONT_CANDIDATES), 10))


class RenderWorker(QThread):
    progress = Signal(object)
    completed = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self,
        plan: RenderPlan,
        settings: RenderSettings,
        *,
        pipeline: OraclePipeline | None = None,
        prewarmed_engine=None,
        render_click_wall: float | None = None,
    ) -> None:
        super().__init__()
        self.plan = RenderPlan.from_dict(plan.to_dict())
        self.settings = deepcopy(settings)
        self._pipeline = pipeline
        self._prewarmed_engine = prewarmed_engine
        self._render_click_wall = render_click_wall

    def run(self) -> None:
        try:
            pipeline = self._pipeline or OraclePipeline()
            output_path = pipeline.render(
                self.plan,
                self.settings,
                progress_callback=self.progress.emit,
                prewarmed_engine=self._prewarmed_engine,
                render_click_wall=self._render_click_wall or time(),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(self.plan.to_dict(), str(output_path))


class PreviewWorker(QThread):
    progress = Signal(object)
    completed = Signal(str)  # preview_path only
    failed = Signal(str)

    def __init__(self, utterance: Utterance, profile: VoiceProfile, model_variant: str, device_mode: str, *, pipeline: OraclePipeline | None = None) -> None:
        super().__init__()
        self.utterance = Utterance.from_dict(utterance.to_dict())
        self.profile = VoiceProfile.from_dict(profile.to_dict())
        self.model_variant = model_variant
        self.device_mode = device_mode
        self._pipeline = pipeline

    def run(self) -> None:
        try:
            preview_path = (self._pipeline or OraclePipeline()).render_preview(
                self.utterance,
                self.profile,
                self.model_variant,
                device_mode=self.device_mode,
                progress_callback=self.progress.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        # Emit only the preview path - preview does not mutate row-level render state
        self.completed.emit(str(preview_path))


class RenderProgressDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, *, title: str = "Rendering") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        self.stage_label = QLabel("Starting render...")
        self.segment_label = QLabel("Segments: 0/0")
        self.eta_label = QLabel("ETA: calculating...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.segment_label)
        layout.addWidget(self.eta_label)
        layout.addWidget(self.progress_bar)
        self.reset()

    def reset(self) -> None:
        self.progress_bar.setValue(0)
        self.stage_label.setText("Starting render...")
        self.segment_label.setText("Segments: 0/0")
        self.eta_label.setText("ETA: calculating...")

    def update_progress(self, progress: RenderProgress) -> None:
        percent = 0 if progress.total_steps <= 0 else int(round((progress.current_step / progress.total_steps) * 100))
        self.progress_bar.setValue(max(0, min(100, percent)))
        self.stage_label.setText(f"{progress.stage}: {progress.detail}")
        if progress.total_segments > 0:
            self.segment_label.setText(f"Segments: {progress.current_segment}/{progress.total_segments}")
        elif progress.total_steps > 0:
            self.segment_label.setText(f"Steps: {progress.current_step}/{progress.total_steps}")
        else:
            self.segment_label.setText("Segments: preparing...")
        if progress.eta_seconds is None:
            self.eta_label.setText(f"Elapsed: {self._format_seconds(progress.elapsed_seconds)} | ETA: calculating...")
        else:
            self.eta_label.setText(
                f"Elapsed: {self._format_seconds(progress.elapsed_seconds)} | ETA: {self._format_seconds(progress.eta_seconds)}"
            )

    @staticmethod
    def _format_seconds(value: float) -> str:
        seconds = max(0, int(round(value)))
        minutes, seconds = divmod(seconds, 60)
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"


class SpeakerGroup(QGroupBox):
    def __init__(self, speaker: str, custom_reference_dir: Path) -> None:
        super().__init__(f"Speaker {speaker}")
        self.custom_reference_dir = custom_reference_dir
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.reference_path = QLineEdit()
        self.reference_picker = QComboBox()
        self.reference_picker.currentIndexChanged.connect(self._handle_reference_selection)
        # Use activated so selecting the already-current custom option still fires.
        self.reference_picker.activated.connect(self._handle_reference_selection)
        self._available_reference_paths: set[str] = set()

        self.language_combo = QComboBox()
        self.cfg_weight = self._double_box(0.0, 1.5, 0.5, 0.05)
        self.exaggeration = self._double_box(0.0, 1.5, 0.5, 0.05)
        self.temperature = self._double_box(0.1, 1.5, 0.8, 0.05)
        self.emotion_intensity = self._double_box(0.0, 2.0, 1.0, 0.1)
        self.naturalness = self._double_box(0.0, 1.0, 0.0, 0.05)
        self.pause_spin = QSpinBox()
        self.pause_spin.setRange(0, 2000)
        self.pause_spin.setValue(180)

        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.addRow("Custom Voice Reference Audio", self.reference_picker)
        form.addRow("Language", self.language_combo)
        form.addRow("CFG Weight", self.cfg_weight)
        form.addRow("Exaggeration", self.exaggeration)
        form.addRow("Temperature", self.temperature)
        form.addRow("Emotion Intensity", self.emotion_intensity)
        form.addRow("Naturalness (Heuristic)", self.naturalness)
        form.addRow("Pause After Speaker Turn (ms)", self.pause_spin)

    def _double_box(self, minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(2)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _pick_audio(self) -> None:
        current_reference = Path(self.reference_path.text()).expanduser()
        start_dir = current_reference.parent if current_reference.exists() else self.custom_reference_dir
        path, _ = QFileDialog.getOpenFileName(self, "Choose Reference Audio", str(start_dir), "Audio Files (*.wav *.flac *.mp3)")
        if path:
            self.reference_path.setText(path)

    def set_language_options(self, languages: dict[str, str], enabled: bool) -> None:
        selected = self.language_combo.currentData() or "en"
        self.language_combo.clear()
        for code, name in languages.items():
            self.language_combo.addItem(f"{code} - {name}", code)
        index = self.language_combo.findData(selected if enabled else "en")
        if index < 0:
            index = self.language_combo.findData("en")
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        self.language_combo.setEnabled(enabled)

    def set_reference_choices(self, defaults: list[VoiceChoice], recents: list[str], selected_path: str = "") -> None:
        current_path = selected_path or self.reference_path.text()
        self.reference_picker.blockSignals(True)
        self.reference_picker.clear()
        self._available_reference_paths = set()
        if defaults:
            header_index = self.reference_picker.count()
            self.reference_picker.addItem("Default Voices")
            header_item = self.reference_picker.model().item(header_index)
            if header_item is not None:
                header_item.setEnabled(False)
            for voice in defaults[:10]:
                self.reference_picker.addItem(f"  {voice.label}", voice.path)
                self._available_reference_paths.add(voice.path)
        if recents:
            header_index = self.reference_picker.count()
            self.reference_picker.addItem("Recent Custom Clips")
            header_item = self.reference_picker.model().item(header_index)
            if header_item is not None:
                header_item.setEnabled(False)
            for path in recents[:10]:
                resolved = str(Path(path).expanduser())
                self.reference_picker.addItem(f"  {Path(resolved).name}", resolved)
                self._available_reference_paths.add(resolved)
        self.reference_picker.addItem("Custom Voice Reference Audio...", "__custom__")
        target_index = self.reference_picker.findData(current_path)
        if target_index < 0:
            target_index = self.reference_picker.findData("__custom__")
        self.reference_picker.setCurrentIndex(target_index)
        self.reference_picker.blockSignals(False)
        if current_path in self._available_reference_paths:
            self.reference_path.setText(current_path)

    def _handle_reference_selection(self, _index=None) -> None:
        data = self.reference_picker.currentData()
        if data == "__custom__":
            self._pick_audio()
            return
        if isinstance(data, str) and data:
            self.reference_path.setText(data)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._startup_t0 = perf_counter()
        self._startup_marks: list[tuple[str, float]] = []
        self._gui_shown_wall: float | None = None
        self.repo_root = Path(__file__).resolve().parents[2]
        self.paths: OraclePaths = ensure_repo_default_paths(self.repo_root)
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
        self.progress_dialog: RenderProgressDialog | None = None
        self.preview_dialog: RenderProgressDialog | None = None
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.setWindowTitle("The Oracle")
        self.resize(1320, 900)
        self._mark_startup("mainwindow_init_begin")
        self._build_ui()
        self._build_menu()
        self.delete_confirm_enabled = True
        self._apply_gui_settings_payload(self._default_gui_settings_payload())
        self._mark_startup("mainwindow_init_end")
        self._write_startup_timeline()
        QTimer.singleShot(0, self._on_gui_shown)

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
        self._start_prewarm()

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
        root.setObjectName("oracleRoot")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(20)

        self.input_path = QLineEdit()
        self.outdir_path = QLineEdit()
        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("Auto-derived from the input file when using the default Output folder")
        self.input_path.textChanged.connect(self._handle_outdir_changed)
        self.outdir_path.textChanged.connect(self._handle_outdir_changed)
        self.speaker_a = SpeakerGroup("A", self.paths.voice_dir)
        self.speaker_b = SpeakerGroup("B", self.paths.voice_dir)
        self.table = QTableWidget(0, 9)
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
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(2, 340)
        self.table.setColumnWidth(3, 340)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setWordWrap(True)
        self.table.setMinimumHeight(360)
        self.error_panel = QTextEdit()
        self.error_panel.setReadOnly(True)
        self.error_panel.setPlaceholderText("Status and model errors appear here.")
        self.error_panel.setMinimumHeight(170)

        layout.addWidget(self._build_top_bar())

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self._build_control_column())
        self.main_splitter.addWidget(self._build_workspace_column())
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([420, 900])
        layout.addWidget(self.main_splitter, stretch=1)

        self.setCentralWidget(root)
        self.outdir_path.setText(str(self.paths.output_dir))
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

        self.templates_menu = settings_menu.addMenu("Load Template")
        self.templates_menu.aboutToShow.connect(self._rebuild_templates_menu)
        self.confirmation_action = QAction("Re-enable delete confirmations", self)
        self.confirmation_action.triggered.connect(self._enable_delete_confirmation)
        settings_menu.addSeparator()
        settings_menu.addAction(self.confirmation_action)

    def _build_project_settings(self) -> QGroupBox:
        box = QGroupBox("Shared Render Settings")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.variant_combo = QComboBox()
        self.variant_combo.addItems(list(SUPPORTED_VARIANTS))
        self.variant_combo.currentTextChanged.connect(self._refresh_language_options)
        self.correction_mode_combo = QComboBox()
        for label, value in CORRECTION_MODE_OPTIONS:
            self.correction_mode_combo.addItem(label, value)
        self._set_correction_mode(RenderSettings().correction_mode)
        self.loudness_combo = QComboBox()
        self.loudness_combo.addItems(["off", "light", "medium"])
        self.loudness_combo.setCurrentText(RenderSettings().loudness_preset)
        self.crossfade_spin = QSpinBox()
        self.crossfade_spin.setRange(0, 500)
        self.crossfade_spin.setValue(RenderSettings().crossfade_ms)
        form.addRow("Model Variant", self.variant_combo)
        form.addRow("Correction Mode", self.correction_mode_combo)
        form.addRow("Loudness", self.loudness_combo)
        form.addRow("Crossfade (ms)", self.crossfade_spin)
        return box

    def _set_correction_mode(self, value: str) -> None:
        normalized = normalize_correction_mode(value)
        idx = self.correction_mode_combo.findData(normalized)
        if idx < 0:
            idx = self.correction_mode_combo.findData(normalize_correction_mode("moderate"))
        if idx >= 0:
            self.correction_mode_combo.setCurrentIndex(idx)

    def _build_top_bar(self) -> QWidget:
        card = QWidget()
        card.setObjectName("topBarCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        top_row = QHBoxLayout()
        top_row.setSpacing(18)

        copy_column = QVBoxLayout()
        copy_column.setSpacing(4)

        eyebrow = QLabel("Oracle Control Room")
        eyebrow.setObjectName("appKicker")

        title = QLabel("Direct two voices, inspect every line, and render with discipline.")
        title.setObjectName("appTitle")
        title.setWordWrap(True)
        title.setFont(QFont(_pick_font_family(_DISPLAY_FONT_CANDIDATES), 16, QFont.Bold))

        subtitle = QLabel(
            "A denser studio layout for actual production work: controlled inputs on the left, review and status on the right."
        )
        subtitle.setObjectName("appSummary")
        subtitle.setWordWrap(True)

        copy_column.addWidget(eyebrow)
        copy_column.addWidget(title)
        copy_column.addWidget(subtitle)
        copy_column.addStretch(1)
        top_row.addLayout(copy_column, stretch=1)

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
        top_row.addLayout(actions, stretch=0)

        controls = QGridLayout()
        controls.setHorizontalSpacing(12)
        controls.setVerticalSpacing(8)
        self._add_path_row(controls, 0, "Input", self.input_path, self._pick_input)
        self._add_path_row(controls, 1, "Output Folder", self.outdir_path, self._pick_outdir)
        controls.addWidget(self._build_field_label("Output Filename"), 2, 0)
        controls.addWidget(self.output_name, 2, 1, 1, 2)

        layout.addLayout(top_row)
        layout.addLayout(controls)
        return card

    def _build_control_column(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("controlRail")
        rail.setMinimumWidth(360)
        rail.setMaximumWidth(460)

        stack = QWidget()
        stack_layout = QVBoxLayout(stack)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        stack_layout.setSpacing(14)
        stack_layout.addWidget(self._build_section_label("Render Parameters"))
        stack_layout.addWidget(self._build_project_settings())
        stack_layout.addWidget(self.speaker_a)
        stack_layout.addWidget(self.speaker_b)
        stack_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(stack)

        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        return rail

    def _build_workspace_column(self) -> QWidget:
        workspace = QWidget()
        workspace.setObjectName("workspacePanel")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(self._wrap_card("Dialogue Review", self.table), stretch=1)
        layout.addWidget(self._wrap_card("Operations Log", self.error_panel, minimum_height=210))
        return workspace

    def _build_section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    def _build_field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _wrap_card(self, title: str, widget: QWidget, *, minimum_height: int | None = None) -> QWidget:
        card = QWidget()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(self._build_section_label(title))
        if minimum_height is not None:
            widget.setMinimumHeight(minimum_height)
        layout.addWidget(widget, stretch=1)
        return card

    def _add_path_row(self, layout: QGridLayout, row: int, label: str, field: QLineEdit, callback) -> None:
        button = QPushButton("Browse")
        button.clicked.connect(callback)
        field.setClearButtonEnabled(True)
        self._style_utility_button(button)
        layout.addWidget(self._build_field_label(label), row, 0)
        layout.addWidget(field, row, 1)
        layout.addWidget(button, row, 2)

    def _style_action_button(self, button: QPushButton, accent: bool = False) -> None:
        button.setMinimumHeight(48)
        button.setMinimumWidth(170 if not accent else 210)
        button.setProperty("accentButton", accent)
        button.style().unpolish(button)
        button.style().polish(button)

    def _style_utility_button(self, button: QPushButton) -> None:
        button.setProperty("utilityButton", True)
        button.style().unpolish(button)
        button.style().polish(button)

    def _pick_input(self) -> None:
        current_text = self.input_path.text().strip()
        if not current_text:
            start_dir = self.paths.input_dir
        else:
            current_input = Path(current_text).expanduser()
            start_dir = current_input.parent if current_input.exists() else self.paths.input_dir
        path, _ = QFileDialog.getOpenFileName(self, "Choose Input", str(start_dir), "Text Files (*.txt *.md)")
        if path:
            self.input_path.setText(path)

    def _handle_outdir_changed(self) -> None:
        folder = Path(self.outdir_path.text() or self.paths.output_dir).expanduser()
        default_folder = Path(self.paths.output_dir)
        if folder == default_folder:
            return
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        if not self.output_name.text().strip():
            default_name = default_output_filename(self.input_path.text() or "")
            self.output_name.setText(default_name)

    def _pick_outdir(self) -> None:
        current_outdir = Path(self.outdir_path.text()).expanduser()
        start_dir = current_outdir if current_outdir.exists() else self.paths.output_dir
        path = QFileDialog.getExistingDirectory(self, "Choose Output Directory", str(start_dir))
        if path:
            self.outdir_path.setText(path)

    def _refresh_language_options(self) -> None:
        variant = self.variant_combo.currentText() if hasattr(self, "variant_combo") else "standard"
        languages = {"en": "English"} if variant != "multilingual" else ChatterboxEngine(variant).supported_languages()
        is_multilingual = variant == "multilingual"
        self.speaker_a.set_language_options(languages, is_multilingual)
        self.speaker_b.set_language_options(languages, is_multilingual)

    def _refresh_reference_pickers(self) -> None:
        defaults = default_voice_choices(self.repo_root)
        recents = [path for path in load_recent_reference_paths() if Path(path).exists()]
        self.speaker_a.set_reference_choices(defaults, recents, self.speaker_a.reference_path.text())
        self.speaker_b.set_reference_choices(defaults, recents, self.speaker_b.reference_path.text())

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
        self._prewarm_thread = PrewarmThread(device=_DEVICE_MODE)
        self._prewarm_thread.ready.connect(self._handle_prewarm_ready)
        self._prewarm_thread.failed.connect(self._handle_prewarm_failed)
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
            thread = self._prewarm_thread
            self._prewarm_thread = None
        if thread is not None:
            thread.deleteLater()
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
            thread = self._prewarm_thread
            self._prewarm_thread = None
        if thread is not None:
            thread.deleteLater()
        self.error_panel.append(f"Background prewarm failed: {message}")
        self._write_prewarm_timing(success=False)
        # Allow the user to proceed manually even if warmup failed.
        self.analyze_button.setEnabled(True)
        self.render_button.setEnabled(True)

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

    def _speaker_settings(self) -> dict[str, SpeakerSettings]:
        variant = self.variant_combo.currentText()
        return {
            "A": SpeakerSettings(
                reference_path=self.speaker_a.reference_path.text(),
                voice_settings=VoiceSettings(
                    variant=variant,
                    language=self.speaker_a.language_combo.currentData() or "en",
                    cfg_weight=self.speaker_a.cfg_weight.value(),
                    exaggeration=self.speaker_a.exaggeration.value(),
                    temperature=self.speaker_a.temperature.value(),
                    emotion_intensity=self.speaker_a.emotion_intensity.value(),
                    naturalness=self.speaker_a.naturalness.value(),
                    pause_ms=self.speaker_a.pause_spin.value(),
                    crossfade_ms=self.crossfade_spin.value(),
                ),
            ),
            "B": SpeakerSettings(
                reference_path=self.speaker_b.reference_path.text(),
                voice_settings=VoiceSettings(
                    variant=variant,
                    language=self.speaker_b.language_combo.currentData() or "en",
                    cfg_weight=self.speaker_b.cfg_weight.value(),
                    exaggeration=self.speaker_b.exaggeration.value(),
                    temperature=self.speaker_b.temperature.value(),
                    emotion_intensity=self.speaker_b.emotion_intensity.value(),
                    naturalness=self.speaker_b.naturalness.value(),
                    pause_ms=self.speaker_b.pause_spin.value(),
                    crossfade_ms=self.crossfade_spin.value(),
                ),
            ),
        }

    def _render_settings(self) -> RenderSettings:
        variant = self.variant_combo.currentText()
        mode_value = self.correction_mode_combo.currentData() or self.correction_mode_combo.currentText()
        return RenderSettings(
            correction_mode=mode_value,
            model_variant=variant,
            language=self.speaker_a.language_combo.currentData() or "en",
            export_stems=True,
            loudness_preset=self.loudness_combo.currentText(),
            pause_between_turns_ms=self.speaker_a.pause_spin.value(),
            crossfade_ms=self.crossfade_spin.value(),
            device_mode=_DEVICE_MODE,
            metadata={"output_filename": normalize_output_filename(self.output_name.text())},
        )

    def _pipeline(self) -> OraclePipeline:
        if self.pipeline is not None:
            return self.pipeline
        with self._prewarm_lock:
            state = self._prewarm_state
        if state == "ready" and self._prewarmed_pipeline is not None:
            self.pipeline = self._prewarmed_pipeline
            return self.pipeline
        self.pipeline = OraclePipeline()
        return self.pipeline

    def _prewarmed_engine_ready(self):
        with self._prewarm_lock:
            state = self._prewarm_state
        if state == "ready":
            # Do not pass live engine across threads; return sentinel to signal no reuse.
            return None
        return None

    def _default_gui_settings_payload(self) -> dict:
        default_render = RenderSettings()
        default_voice = VoiceSettings(variant=default_render.model_variant)
        return {
            "version": 1,
            "name": "",
            "device_mode": default_render.device_mode,
            "project": {
                "model_variant": default_render.model_variant,
                "correction_mode": default_render.correction_mode,
                "loudness_preset": default_render.loudness_preset,
                "crossfade_ms": default_render.crossfade_ms,
                "output_dir": str(self.paths.output_dir),
                "output_filename": "",
            },
            "speakers": {
                speaker: {
                    "reference_path": "",
                    "voice_settings": default_voice.to_dict(),
                    "emotion_reference_paths": {},
                }
                for speaker in ("A", "B")
            },
        }

    def _apply_speaker_group(self, group: SpeakerGroup, settings: SpeakerSettings) -> None:
        voice = VoiceSettings.from_mapping(settings.voice_settings)
        group.reference_path.setText(settings.reference_path)
        language_index = group.language_combo.findData(voice.language)
        if language_index >= 0:
            group.language_combo.setCurrentIndex(language_index)
        group.cfg_weight.setValue(voice.cfg_weight)
        group.exaggeration.setValue(voice.exaggeration)
        group.temperature.setValue(voice.temperature)
        group.emotion_intensity.setValue(voice.emotion_intensity)
        group.naturalness.setValue(voice.naturalness)
        group.pause_spin.setValue(voice.pause_ms)
        self._refresh_reference_pickers()

    def _load_project_into_ui(self, saved_project) -> None:
        self.current_project_path = None
        self.plan = saved_project.plan
        self.input_path.setText(saved_project.input_path)
        self.outdir_path.setText(saved_project.output_path)
        self.output_name.setText(str(saved_project.render_settings.metadata.get("output_filename", "")))
        self.variant_combo.setCurrentText(saved_project.render_settings.model_variant)
        self._refresh_language_options()
        self._set_correction_mode(saved_project.render_settings.correction_mode)
        self.loudness_combo.setCurrentText(saved_project.render_settings.loudness_preset)
        self.crossfade_spin.setValue(saved_project.render_settings.crossfade_ms)
        self._apply_speaker_group(self.speaker_a, saved_project.speaker_settings["A"])
        self._apply_speaker_group(self.speaker_b, saved_project.speaker_settings["B"])
        self._populate_table(self.plan)

    def _current_saved_project(self):
        if not self.plan:
            self.prepare_project()
        if not self.plan:
            raise ValueError("No project is available to save.")
        self._sync_plan_from_table()
        return build_saved_project(self.plan, self._render_settings(), self._speaker_settings())

    def _current_gui_settings_payload(self) -> dict:
        return {
            "version": 1,
            "name": "",
            "device_mode": _DEVICE_MODE,
            "project": {
                "model_variant": self.variant_combo.currentText(),
                "correction_mode": normalize_correction_mode(self.correction_mode_combo.currentData() or self.correction_mode_combo.currentText()),
                "loudness_preset": self.loudness_combo.currentText(),
                "crossfade_ms": self.crossfade_spin.value(),
                "output_dir": self.outdir_path.text() or str(self.paths.output_dir),
                "output_filename": normalize_output_filename(self.output_name.text()),
                "delete_confirm_enabled": self.delete_confirm_enabled,
            },
            "speakers": {
                speaker: {
                    "reference_path": settings.reference_path,
                    "voice_settings": VoiceSettings.from_mapping(settings.voice_settings).to_dict(),
                    "emotion_reference_paths": dict(settings.emotion_reference_paths),
                }
                for speaker, settings in self._speaker_settings().items()
            },
        }

    def _apply_gui_settings_payload(self, payload: dict) -> None:
        defaults = self._default_gui_settings_payload()
        project = {**defaults["project"], **payload["project"]}
        self.variant_combo.setCurrentText(project.get("model_variant", "standard"))
        self._refresh_language_options()
        self._set_correction_mode(project.get("correction_mode", "moderate"))
        self.loudness_combo.setCurrentText(project.get("loudness_preset", RenderSettings().loudness_preset))
        self.crossfade_spin.setValue(int(project.get("crossfade_ms", 20)))
        self.outdir_path.setText(str(project.get("output_dir", self.paths.output_dir)))
        self.output_name.setText(normalize_output_filename(str(project.get("output_filename", ""))))
        self.delete_confirm_enabled = bool(project.get("delete_confirm_enabled", True))
        for speaker, group in (("A", self.speaker_a), ("B", self.speaker_b)):
            config = {**defaults["speakers"][speaker], **payload["speakers"][speaker]}
            voice = VoiceSettings.from_mapping(config.get("voice_settings"))
            group.reference_path.setText(config.get("reference_path", ""))
            language_index = group.language_combo.findData(voice.language)
            if language_index >= 0:
                group.language_combo.setCurrentIndex(language_index)
            group.cfg_weight.setValue(voice.cfg_weight)
            group.exaggeration.setValue(voice.exaggeration)
            group.temperature.setValue(voice.temperature)
            group.emotion_intensity.setValue(voice.emotion_intensity)
            group.naturalness.setValue(voice.naturalness)
            group.pause_spin.setValue(voice.pause_ms)
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
        self.input_path.clear()
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

    def prepare_project(self) -> None:
        with self._prewarm_lock:
            if self._prewarm_state == "warming":
                self.error_panel.append("Background warmup still running; please wait a moment before analyzing.")
                return
        analyze_click_wall = time()
        self._log_action_timing("analyze_click", analyze_click_wall)
        try:
            self.plan = self._pipeline().prepare_plan(
                self.input_path.text(),
                self.outdir_path.text(),
                self._speaker_settings(),
                self._render_settings(),
            )
            plan_ready_wall = time()
            self._log_action_timing("plan_ready", plan_ready_wall, {"elapsed": plan_ready_wall - analyze_click_wall})
            for speaker in self._speaker_settings().values():
                if speaker.reference_path:
                    remember_recent_reference_path(speaker.reference_path)
            self._refresh_reference_pickers()
            self._populate_table(self.plan)
            self.error_panel.append("Analysis complete.")
        except Exception as exc:
            self.error_panel.append(str(exc))
            QMessageBox.critical(self, "Analysis Failed", str(exc))

    def _populate_table(self, plan: RenderPlan) -> None:
        self.table.setRowCount(len(plan.utterances))
        for row, utterance in enumerate(plan.utterances):
            self.table.setItem(row, 0, QTableWidgetItem(str(utterance.index)))
            speaker_combo = QComboBox()
            speaker_combo.addItems(["A", "B"])
            speaker_combo.setCurrentText(utterance.speaker)
            self.table.setCellWidget(row, 1, speaker_combo)
            self.table.setItem(row, 2, QTableWidgetItem(utterance.original_text))
            repaired = QTableWidgetItem(utterance.repaired_text)
            repaired.setFlags(repaired.flags() | Qt.ItemIsEditable)
            self.table.setItem(row, 3, repaired)
            emotion = self._create_emotion_combo(utterance.emotion)
            self.table.setCellWidget(row, 4, emotion)
            duration = "" if utterance.duration_seconds is None else f"{utterance.duration_seconds:.2f}s"
            self.table.setItem(row, 5, QTableWidgetItem(duration))
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
        self.plan.voice_profiles = self.plan.voice_profiles | {
            "A": replace(self.plan.voice_profiles["A"], engine_params=speaker_settings["A"].voice_settings),
            "B": replace(self.plan.voice_profiles["B"], engine_params=speaker_settings["B"].voice_settings),
        }
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

        self.preview_dialog = RenderProgressDialog(self, title="Generating Preview")
        self.preview_dialog.show()
        self._set_preview_busy(True)
        self.preview_worker = PreviewWorker(
            utterance,
            self.plan.voice_profiles[utterance.speaker],
            self.variant_combo.currentText(),
            _DEVICE_MODE,  # CPU is the only verified execution path
        )
        self.preview_worker.progress.connect(self._update_preview_progress)
        self.preview_worker.completed.connect(lambda path: self._finish_preview(row, path))
        self.preview_worker.failed.connect(self._fail_preview)
        self.preview_worker.finished.connect(self._cleanup_preview_worker)
        self.preview_worker.start()

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
                self.paths.output_dir,
                self.output_name.text(),
            )
            if not output_filename:
                raise ValueError("Choose an output filename before rendering outside the default Output folder.")
            self.plan.output_dir = self.outdir_path.text() or str(self.paths.output_dir)
        except Exception as exc:
            self.error_panel.append(f"Render failed: {exc}")
            QMessageBox.critical(self, "Render Failed", str(exc))
            return

        render_click_wall = time()
        self._log_action_timing("render_click", render_click_wall)
        self.progress_dialog = RenderProgressDialog(self, title="Rendering")
        self.progress_dialog.show()
        self._set_render_busy(True)
        render_settings = self._render_settings()
        render_settings.metadata["output_filename"] = output_filename
        self.render_worker = RenderWorker(
            self.plan,
            render_settings,
            pipeline=self._pipeline(),
            prewarmed_engine=self._prewarmed_engine_ready(),
            render_click_wall=render_click_wall,
        )
        self.render_worker.progress.connect(self._update_render_progress)
        self.render_worker.completed.connect(self._finish_render)
        self.render_worker.failed.connect(self._fail_render)
        self.render_worker.finished.connect(self._cleanup_render_worker)
        self.render_worker.start()

    def _update_render_progress(self, progress: RenderProgress) -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.update_progress(progress)

    def _finish_render(self, plan_payload: dict, output_path: str) -> None:
        self.plan = RenderPlan.from_dict(plan_payload)
        self._populate_table(self.plan)
        self.error_panel.append(f"Render complete: {output_path}")
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None

    def _fail_render(self, message: str) -> None:
        self.error_panel.append(f"Render failed: {message}")
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        # Refresh the table to show truthful row state after failure
        # Rows that completed will show their status/duration
        # Rows that failed will show status="failed"
        # Rows never reached will show status="pending"
        self._populate_table(self.plan)
        
        # Show failure summary with row-level information
        failed_rows = self.plan.metadata.get("failed_rows", "")
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
        if self.preview_dialog is not None:
            self.preview_dialog.update_progress(progress)

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

    def _fail_preview(self, message: str) -> None:
        self.error_panel.append(f"Preview failed: {message}")
        if self.preview_dialog is not None:
            self.preview_dialog.close()
            self.preview_dialog = None
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
    app = QApplication.instance() or QApplication([])
    _configure_application(app)
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
