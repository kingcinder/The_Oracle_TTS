"""First-open Recording Studio setup and speaking guide.

This dialog is intentionally separate from the recorder itself.  It lets a
new user configure the microphone, source script, Seashell folder, and naming
policy before recording, then walks through the techniques that make a useful
voice-reference clip.  The wizard edits the live RecordingStudioDialog only
when the user continues/finishes, so abandoning it never leaves half-applied
settings behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QToolTip,
)

from the_oracle.audio import recorder


@dataclass(frozen=True, slots=True)
class RecordingGuideStage:
    key: str
    title: str
    explanation: str
    target_name: str | None = None


STAGES: tuple[RecordingGuideStage, ...] = (
    RecordingGuideStage(
        "microphone",
        "1. Choose a microphone and sample rate",
        "The microphone is the source of the Seashell reference clip. The wizard lists the input devices that PortAudio can actually open and puts the system default first. Choose the microphone you will keep using: changing microphones later changes the room tone, distance, and color that the voice model learns. The sample-rate list is probed against that exact device. If no input device is found, recording is unavailable until a microphone and capture backend are installed; you can replay this guide from Settings after connecting one.",
        "mic_combo",
    ),
    RecordingGuideStage(
        "script",
        "2. Select a consistent reading script",
        "The teleprompter script gives every take the same words and sounds, making it a better voice reference than improvisation. Input/ is the normal home for scripts and is shared with the main GUI. The bundled What is, reality.txt file is selected when present because it exercises a broad alphabetical and linguistic range. You may choose another TXT/MD file or type your own lines; the selected script is remembered for the next Recording Studio session.",
        "script_combo",
    ),
    RecordingGuideStage(
        "folder",
        "3. Choose the Seashell folder",
        "Seashells/ is the default recording destination and is reserved for voice-reference recordings, not finished dialogue renders. The folder can be changed to another directory; the selected folder is remembered when you tick the checkbox below. The name preview is kept inside this folder and the next free Seashell_No_N.wav is generated so normal takes never overwrite an earlier take.",
        "outdir_combo",
    ),
    RecordingGuideStage(
        "naming",
        "4. Confirm the naming policy",
        "A generated Seashell_No_N.wav name is a safe boilerplate name, but it does not tell you who, when, or under which conditions the take was recorded. The first time you record with a generic name, The Oracle shows a caution and lets you disable that caution. For a production library, close the warning, replace the name with something descriptive such as Cody_close_mic_warm.wav, and then record. The warning preference is independent of the recording itself and can be restored in Settings.",
        "name_edit",
    ),
    RecordingGuideStage(
        "technique",
        "5. Speak for a clean, expressive reference",
        "Position the microphone 15-25 cm (6-10 in) from your mouth, slightly to the side rather than directly in front, with a pop filter if available. Keep the distance and angle fixed. Record in a quiet, acoustically damped room; turn off fans and notifications, and leave a few seconds of room silence if you want to diagnose noise. Use a relaxed conversational loudness, clear consonants, and complete words. Vary feeling through intention, pitch, pace, and emphasis in small controlled amounts: give a sentence a genuine question, reassurance, urgency, sadness, or amusement without acting at the microphone. Breathe quietly between phrases, avoid plosives, clicks, lip smacks, clipping, whispering, and exaggerated announcer diction. Read the full range steadily; consistency and intelligibility are more valuable than theatrical extremes. Listen back and re-record a noisy, clipped, rushed, or over-performed take.",
        "prompt_area",
    ),
    RecordingGuideStage(
        "finish",
        "6. Review, audition, and assign",
        "After recording, The Oracle saves a mono PCM WAV, offers Listen again, and lets you assign the take to Speaker A or B. Use the audition to check that the beginning and ending are clean, speech is intelligible, and the voice is natural rather than overdone. A Seashell is a reference input for the selected speaker; it does not itself render a dialogue. You can record more takes at any time and reopen this guide from Settings → Replay Recording Studio Setup & Guide.",
        "play_button",
    ),
)


class RecordingStudioSetupWizard(QDialog):
    """Non-modal first-open setup wizard for :class:`RecordingStudioDialog`."""

    completed = Signal(bool, str)

    def __init__(
        self,
        studio: QWidget,
        *,
        mode: str = "full",
        on_apply: Callable[[dict], None] | None = None,
    ) -> None:
        super().__init__(studio)
        self.studio = studio
        self.mode = mode if mode in {"full", "guide"} else "full"
        self.on_apply = on_apply
        self._stage_index = 0
        self._finished_once = False
        self._highlighted: QWidget | None = None
        self._stages = list(STAGES) if self.mode == "full" else list(STAGES[4:])
        self.setWindowTitle("The Oracle - Recording Studio Setup & Guide")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setModal(False)
        self.setMinimumSize(600, 470)
        self.resize(730, 620)

        self.heading = QLabel()
        self.heading.setWordWrap(True)
        font = QFont()
        font.setPointSize(15)
        font.setBold(True)
        self.heading.setFont(font)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setObjectName("recordingWizardSummary")
        self.setup_box = QWidget()
        form = QFormLayout(self.setup_box)
        form.setContentsMargins(0, 0, 0, 0)
        self.mic_picker = QComboBox()
        self.rate_picker = QComboBox()
        self.script_picker = QComboBox()
        self.folder_edit = QLineEdit()
        self.folder_browse = QPushButton("Browse…")
        self.folder_browse.clicked.connect(self._browse_folder)
        folder_row = QWidget()
        folder_layout = QVBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_line = QFormLayout()
        folder_line.setContentsMargins(0, 0, 0, 0)
        folder_line.addRow(self.folder_edit, self.folder_browse)
        folder_layout.addLayout(folder_line)
        self.name_edit = QLineEdit()
        self.remember_input = QCheckBox("Remember this input script for future sessions")
        self.remember_output = QCheckBox("Remember this recording folder for future sessions")
        self.generic_warning = QCheckBox("Show the caution before generic Seashell names")
        form.addRow("Microphone", self.mic_picker)
        form.addRow("Sample rate", self.rate_picker)
        form.addRow("Input script", self.script_picker)
        form.addRow("Seashell folder", folder_row)
        form.addRow("Next file name", self.name_edit)
        form.addRow("", self.remember_input)
        form.addRow("", self.remember_output)
        form.addRow("", self.generic_warning)
        self.setup_box.setObjectName("recordingWizardSetup")

        self.explanation = QPlainTextEdit()
        self.explanation.setReadOnly(True)
        self.explanation.setMinimumHeight(250)
        self.explanation.setObjectName("recordingWizardExplanation")
        self.progress = QLabel()
        self.progress.setObjectName("recordingWizardProgress")
        self.note = QLabel("Continue applies the setup choices and advances to the next dependent feature. Skip for now closes without marking the guide complete.")
        self.note.setWordWrap(True)
        self.note.setObjectName("recordingWizardNote")
        self.buttons = QDialogButtonBox()
        self.continue_button = QPushButton("Continue")
        self.continue_button.setDefault(True)
        self.continue_button.clicked.connect(self._continue)
        self.skip_button = QPushButton("Skip for now")
        self.skip_button.clicked.connect(lambda: self._finish(False))
        self.buttons.addButton(self.continue_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.addButton(self.skip_button, QDialogButtonBox.ButtonRole.RejectRole)

        layout = QVBoxLayout(self)
        layout.addWidget(self.heading)
        layout.addWidget(self.summary)
        layout.addWidget(self.setup_box)
        layout.addWidget(self.explanation, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.note)
        layout.addWidget(self.buttons)
        self._populate_from_studio()
        self._render_stage()

    def _populate_from_studio(self) -> None:
        studio = self.studio
        self.mic_picker.clear()
        for index in range(studio.mic_combo.count()):
            self.mic_picker.addItem(studio.mic_combo.itemText(index), studio.mic_combo.itemData(index))
        self.mic_picker.setCurrentIndex(max(0, studio.mic_combo.currentIndex()))
        self.rate_picker.clear()
        for index in range(studio.rate_combo.count()):
            self.rate_picker.addItem(studio.rate_combo.itemText(index), studio.rate_combo.itemData(index))
        self.rate_picker.setCurrentIndex(max(0, studio.rate_combo.currentIndex()))
        self.script_picker.clear()
        for index in range(studio.script_combo.count()):
            self.script_picker.addItem(studio.script_combo.itemText(index), studio.script_combo.itemData(index))
        self.script_picker.setCurrentIndex(max(0, studio.script_combo.currentIndex()))
        self.folder_edit.setText(studio.outdir_combo.currentText())
        self.name_edit.setText(studio.name_edit.text())
        self.generic_warning.setChecked(bool(getattr(studio, "generic_name_warning_enabled", True)))
        self.remember_input.setChecked(bool(getattr(studio, "remember_input_default", True)))
        self.remember_output.setChecked(bool(getattr(studio, "remember_output_default", True)))
        self.mic_picker.currentIndexChanged.connect(self._wizard_mic_changed)

    def _wizard_mic_changed(self, index: int) -> None:
        data = self.mic_picker.itemData(index)
        self.rate_picker.clear()
        if data is None:
            self.rate_picker.addItem("(no supported input rate)", None)
            self.rate_picker.setEnabled(False)
            return
        try:
            rates = recorder.samplerates_for_device(int(data))
        except Exception:
            rates = []
        if not rates:
            self.rate_picker.addItem("(no supported input rate)", None)
            self.rate_picker.setEnabled(False)
            return
        for rate in rates:
            self.rate_picker.addItem(f"{rate} Hz", rate)
        self.rate_picker.setEnabled(True)

    def _browse_folder(self) -> None:
        start = self.folder_edit.text().strip() or str(self.studio.voice_dir)
        chosen = QFileDialog.getExistingDirectory(self, "Choose Seashell folder", start)
        if chosen:
            self.folder_edit.setText(chosen)
            self.name_edit.setText(recorder.next_seashell_name(chosen) + ".wav")

    def _summary_text(self) -> str:
        devices = getattr(self.studio, "_devices", [])
        if devices:
            return f"Capture devices found: {len(devices)}. The selected device is {self.mic_picker.currentText() or 'not selected'}."
        if recorder.have_capture_backend():
            return "No microphone input was found. You can finish the guide, but recording will remain disabled until a microphone is installed or connected."
        return "The sounddevice capture backend is not available. Install the recording dependency, then replay this guide from Settings."

    def _render_stage(self) -> None:
        stage = self._stages[self._stage_index]
        self.heading.setText(stage.title)
        self.explanation.setPlainText(stage.explanation)
        self.progress.setText(f"Recording Studio step {self._stage_index + 1} of {len(self._stages)}")
        self.summary.setText(self._summary_text())
        self.continue_button.setText("Finish" if self._stage_index == len(self._stages) - 1 else "Continue")
        self.setup_box.setVisible(self._stage_index < 4)
        self._set_highlight(stage.target_name)
        self._move_near_target(stage.target_name)

    def _target(self, name: str | None) -> QWidget | None:
        if not name:
            return None
        target = getattr(self.studio, name, None)
        return target if isinstance(target, QWidget) else None

    def _set_highlight(self, name: str | None) -> None:
        if self._highlighted is not None:
            self._highlighted.setProperty("wizardFocus", False)
            self._highlighted.style().unpolish(self._highlighted)
            self._highlighted.style().polish(self._highlighted)
        self._highlighted = self._target(name)
        if self._highlighted is None:
            return
        self._highlighted.setProperty("wizardFocus", True)
        self._highlighted.style().unpolish(self._highlighted)
        self._highlighted.style().polish(self._highlighted)
        tooltip = self._highlighted.toolTip()
        if tooltip:
            QToolTip.showText(self._highlighted.mapToGlobal(QPoint(8, self._highlighted.height() + 6)), tooltip, self._highlighted)

    def _move_near_target(self, name: str | None) -> None:
        target = self._target(name)
        screen = target.screen().availableGeometry() if target is not None and target.screen() else self.screen().availableGeometry()
        if target is None or not target.isVisible():
            center = self.studio.frameGeometry().center()
            self.move(center.x() - self.width() // 2, center.y() - self.height() // 2)
            return
        point = target.mapToGlobal(QPoint(0, 0))
        right = point.x() + target.width() + 18
        left = point.x() - self.width() - 18
        x = right if right + self.width() <= screen.right() else left
        x = min(max(screen.left() + 12, x), screen.right() - self.width() - 12)
        y = point.y() + max(0, (target.height() - self.height()) // 2)
        y = min(max(screen.top() + 12, y), screen.bottom() - self.height() - 12)
        self.move(x, y)

    def _payload(self) -> dict:
        mic = self.mic_picker.currentData()
        rate = self.rate_picker.currentData()
        script = self.script_picker.currentData() or ""
        return {
            "microphone_index": int(mic) if isinstance(mic, int) else None,
            "samplerate": int(rate) if isinstance(rate, int) else None,
            "input_file": str(script),
            "output_dir": self.folder_edit.text().strip(),
            "output_filename": self.name_edit.text().strip(),
            "remember_input_default": self.remember_input.isChecked(),
            "remember_output_default": self.remember_output.isChecked(),
            "generic_name_warning": self.generic_warning.isChecked(),
        }

    def _continue(self) -> None:
        if self._stage_index < 4 and self.on_apply is not None:
            self.on_apply(self._payload())
        if self._stage_index >= len(self._stages) - 1:
            self._finish(True)
            return
        self._stage_index += 1
        self._render_stage()

    def _finish(self, accepted: bool) -> None:
        if self._finished_once:
            return
        self._finished_once = True
        if accepted and self.on_apply is not None:
            self.on_apply(self._payload())
        if self._highlighted is not None:
            self._highlighted.setProperty("wizardFocus", False)
            self._highlighted.style().unpolish(self._highlighted)
            self._highlighted.style().polish(self._highlighted)
        self.completed.emit(accepted, self.mode)
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._finished_once:
            self._finish(False)
        event.accept()

    def start(self) -> None:
        self._render_stage()
        self.show()
        self.raise_()
        self.activateWindow()
