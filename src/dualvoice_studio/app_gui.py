"""PySide6 desktop GUI for the Chatterbox-only DualVoice Studio app."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
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
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dualvoice_studio.models.project import RenderPlan, VoiceSettings
from dualvoice_studio.pipeline import DualVoicePipeline, RenderSettings, SpeakerSettings
from dualvoice_studio.project_manifest import build_saved_project, load_project_manifest, save_project_manifest


class SpeakerGroup(QGroupBox):
    def __init__(self, speaker: str) -> None:
        super().__init__(f"Speaker {speaker}")
        self.reference_path = QLineEdit()
        browse = QPushButton("Browse")
        browse.clicked.connect(self._pick_audio)

        self.cfg_weight = self._double_box(0.0, 1.5, 0.5, 0.05)
        self.exaggeration = self._double_box(0.0, 1.5, 0.5, 0.05)
        self.temperature = self._double_box(0.1, 1.5, 0.8, 0.05)

        path_row = QHBoxLayout()
        path_row.addWidget(self.reference_path)
        path_row.addWidget(browse)

        form = QFormLayout(self)
        form.addRow("Reference Clip", self._wrap(path_row))
        form.addRow("CFG Weight", self.cfg_weight)
        form.addRow("Exaggeration", self.exaggeration)
        form.addRow("Temperature", self.temperature)

    def _double_box(self, minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(2)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _wrap(self, layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _pick_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose Reference Audio", "", "Audio Files (*.wav *.flac *.mp3)")
        if path:
            self.reference_path.setText(path)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = DualVoicePipeline()
        self.plan: RenderPlan | None = None
        self.current_project_path: Path | None = None
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.setWindowTitle("DualVoice Studio")
        self.resize(1320, 900)
        self._build_ui()
        self._build_menu()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)

        controls = QGridLayout()
        self.input_path = QLineEdit()
        self.outdir_path = QLineEdit()
        self._add_path_row(controls, 0, "Input", self.input_path, self._pick_input)
        self._add_path_row(controls, 1, "Output Folder", self.outdir_path, self._pick_outdir)
        layout.addLayout(controls)

        settings_row = QHBoxLayout()
        settings_row.addWidget(self._build_project_settings())
        self.speaker_a = SpeakerGroup("A")
        self.speaker_b = SpeakerGroup("B")
        settings_row.addWidget(self.speaker_a)
        settings_row.addWidget(self.speaker_b)
        layout.addLayout(settings_row)

        actions = QHBoxLayout()
        analyze = QPushButton("Analyze")
        analyze.clicked.connect(self.prepare_project)
        render = QPushButton("Render FLAC")
        render.clicked.connect(self.render_project)
        actions.addWidget(analyze)
        actions.addWidget(render)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Index", "Speaker", "Original Text", "Repaired Text", "Emotion", "Duration", "Preview"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, stretch=1)

        self.error_panel = QTextEdit()
        self.error_panel.setReadOnly(True)
        self.error_panel.setPlaceholderText("Status and model errors appear here.")
        layout.addWidget(QLabel("Status / Errors"))
        layout.addWidget(self.error_panel)

        self.setCentralWidget(root)
        self._refresh_language_options()

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

    def _build_project_settings(self) -> QGroupBox:
        box = QGroupBox("Project Settings")
        form = QFormLayout(box)
        self.variant_combo = QComboBox()
        self.variant_combo.addItems(self.pipeline.available_model_variants())
        self.variant_combo.currentTextChanged.connect(self._refresh_language_options)
        self.language_combo = QComboBox()
        self.correction_mode_combo = QComboBox()
        self.correction_mode_combo.addItems(["conservative", "aggressive"])
        self.loudness_combo = QComboBox()
        self.loudness_combo.addItems(["off", "light", "medium"])
        self.pause_spin = QSpinBox()
        self.pause_spin.setRange(0, 2000)
        self.pause_spin.setValue(180)
        self.crossfade_spin = QSpinBox()
        self.crossfade_spin.setRange(0, 500)
        self.crossfade_spin.setValue(20)
        form.addRow("Model Variant", self.variant_combo)
        form.addRow("Language", self.language_combo)
        form.addRow("Correction Mode", self.correction_mode_combo)
        form.addRow("Loudness", self.loudness_combo)
        form.addRow("Pause Between Turns (ms)", self.pause_spin)
        form.addRow("Crossfade (ms)", self.crossfade_spin)
        return box

    def _add_path_row(self, layout: QGridLayout, row: int, label: str, field: QLineEdit, callback) -> None:
        button = QPushButton("Browse")
        button.clicked.connect(callback)
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(field, row, 1)
        layout.addWidget(button, row, 2)

    def _pick_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose Input", "", "Text Files (*.txt *.md)")
        if path:
            self.input_path.setText(path)
            if not self.outdir_path.text():
                self.outdir_path.setText(str(Path(path).with_suffix("")))

    def _pick_outdir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose Output Directory")
        if path:
            self.outdir_path.setText(path)

    def _refresh_language_options(self) -> None:
        variant = self.variant_combo.currentText() if hasattr(self, "variant_combo") else "standard"
        languages = self.pipeline.supported_languages(variant)
        self.language_combo.clear()
        for code, name in languages.items():
            self.language_combo.addItem(f"{code} - {name}", code)
        is_multilingual = variant == "multilingual"
        self.language_combo.setEnabled(is_multilingual)
        if not is_multilingual:
            index = self.language_combo.findData("en")
            if index >= 0:
                self.language_combo.setCurrentIndex(index)

    def _current_language(self) -> str:
        return self.language_combo.currentData() or "en"

    def _speaker_settings(self) -> dict[str, SpeakerSettings]:
        variant = self.variant_combo.currentText()
        language = self._current_language() if variant == "multilingual" else "en"
        return {
            "A": SpeakerSettings(
                reference_path=self.speaker_a.reference_path.text(),
                voice_settings=VoiceSettings(
                    variant=variant,
                    language=language,
                    cfg_weight=self.speaker_a.cfg_weight.value(),
                    exaggeration=self.speaker_a.exaggeration.value(),
                    temperature=self.speaker_a.temperature.value(),
                    pause_ms=self.pause_spin.value(),
                    crossfade_ms=self.crossfade_spin.value(),
                ),
            ),
            "B": SpeakerSettings(
                reference_path=self.speaker_b.reference_path.text(),
                voice_settings=VoiceSettings(
                    variant=variant,
                    language=language,
                    cfg_weight=self.speaker_b.cfg_weight.value(),
                    exaggeration=self.speaker_b.exaggeration.value(),
                    temperature=self.speaker_b.temperature.value(),
                    pause_ms=self.pause_spin.value(),
                    crossfade_ms=self.crossfade_spin.value(),
                ),
            ),
        }

    def _render_settings(self) -> RenderSettings:
        variant = self.variant_combo.currentText()
        return RenderSettings(
            correction_mode=self.correction_mode_combo.currentText(),
            model_variant=variant,
            language=self._current_language() if variant == "multilingual" else "en",
            export_stems=True,
            loudness_preset=self.loudness_combo.currentText(),
            pause_between_turns_ms=self.pause_spin.value(),
            crossfade_ms=self.crossfade_spin.value(),
        )

    def _apply_speaker_group(self, group: SpeakerGroup, settings: SpeakerSettings) -> None:
        voice = VoiceSettings.from_mapping(settings.voice_settings)
        group.reference_path.setText(settings.reference_path)
        group.cfg_weight.setValue(voice.cfg_weight)
        group.exaggeration.setValue(voice.exaggeration)
        group.temperature.setValue(voice.temperature)

    def _load_project_into_ui(self, saved_project) -> None:
        self.current_project_path = None
        self.plan = saved_project.plan
        self.input_path.setText(saved_project.input_path)
        self.outdir_path.setText(saved_project.output_path)
        self.variant_combo.setCurrentText(saved_project.render_settings.model_variant)
        self._refresh_language_options()
        language_index = self.language_combo.findData(saved_project.render_settings.language)
        if language_index >= 0:
            self.language_combo.setCurrentIndex(language_index)
        self.correction_mode_combo.setCurrentText(saved_project.render_settings.correction_mode)
        self.loudness_combo.setCurrentText(saved_project.render_settings.loudness_preset)
        self.pause_spin.setValue(saved_project.render_settings.pause_between_turns_ms)
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

    def new_project(self) -> None:
        self.current_project_path = None
        self.plan = None
        self.input_path.clear()
        self.outdir_path.clear()
        self.speaker_a.reference_path.clear()
        self.speaker_b.reference_path.clear()
        self.error_panel.clear()
        self.table.setRowCount(0)

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Project Files (*.json)")
        if not path:
            return
        try:
            saved_project = load_project_manifest(path)
            self._load_project_into_ui(saved_project)
            self.current_project_path = Path(path)
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
        try:
            self.plan = self.pipeline.prepare_plan(
                self.input_path.text(),
                self.outdir_path.text(),
                self._speaker_settings(),
                self._render_settings(),
            )
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
            emotion = QTableWidgetItem(utterance.emotion)
            emotion.setFlags(emotion.flags() | Qt.ItemIsEditable)
            self.table.setItem(row, 4, emotion)
            duration = "" if utterance.duration_seconds is None else f"{utterance.duration_seconds:.2f}s"
            self.table.setItem(row, 5, QTableWidgetItem(duration))
            preview = QPushButton("Preview")
            preview.clicked.connect(lambda _checked=False, current=row: self.preview_utterance(current))
            self.table.setCellWidget(row, 6, preview)

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
            emotion_item = self.table.item(row, 4)
            if emotion_item:
                emotion_text = emotion_item.text().strip()
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
        self.plan.metadata["language"] = self._current_language() if self.variant_combo.currentText() == "multilingual" else "en"
        self.plan.update_hashes()

    def preview_utterance(self, row: int) -> None:
        if not self.plan:
            return
        try:
            self._sync_plan_from_table()
            utterance = self.plan.utterances[row]
            preview_path = self.pipeline.render_preview(utterance, self.plan.voice_profiles[utterance.speaker], self.variant_combo.currentText())
            self.player.setSource(QUrl.fromLocalFile(str(preview_path)))
            self.player.play()
        except Exception as exc:
            self.error_panel.append(f"Preview failed: {exc}")

    def render_project(self) -> None:
        if not self.plan:
            self.prepare_project()
            if not self.plan:
                return
        try:
            self._sync_plan_from_table()
            self.plan.output_dir = self.outdir_path.text()
            self.pipeline.render(self.plan, self._render_settings())
            self._populate_table(self.plan)
            self.error_panel.append("Render complete.")
        except Exception as exc:
            self.error_panel.append(f"Render failed: {exc}")
            QMessageBox.critical(self, "Render Failed", str(exc))


def launch_gui() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
