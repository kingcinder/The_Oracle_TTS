"""First-run inference discovery and replayable GUI tutorial.

The wizard is deliberately a small Qt dialog rather than a second application
shell.  It consumes the same hardware facts and inference pickers as the main
window, presents hardware limitations plainly, and walks from prerequisites to
features that depend on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QToolTip,
)

from the_oracle.device_support import CUDADeviceInfo, available_device_modes


@dataclass(frozen=True, slots=True)
class TutorialStage:
    key: str
    title: str
    explanation: str
    target_name: str | None = None
    discovery: bool = False


STAGES: tuple[TutorialStage, ...] = (
    TutorialStage(
        "discovery",
        "1. Discover your inference hardware",
        "The Oracle checks the devices visible to this installation before it asks you to choose. CPU / system DRAM is always available. A CUDA choice is offered only when PyTorch can see an NVIDIA card and that card meets the Chatterbox memory floor. Cards that are detected but unsuitable remain visible with the reason, so an old or 1 GiB NVIDIA card is not mistaken for an AI-capable option. Vulkan is a separate audio.cpp path and may require its runtime/model setup.",
        "pytorch_device_combo",
        True,
    ),
    TutorialStage(
        "backend",
        "2. Choose the inference path",
        "PyTorch is the normal Chatterbox path and uses the PyTorch Device picker for CPU or CUDA. CPU uses system DRAM and is the reliable fallback. CUDA keeps model inference on the selected NVIDIA GPU. Vulkan delegates to audio.cpp and has its own device picker; it is useful for supported GPUs such as AMD when the Vulkan runtime and audio.cpp model are installed. The backend choice affects Analyze's render settings, previews, and the final render together.",
        "inference_backend_combo",
    ),
    TutorialStage(
        "input",
        "3. Select the source and destination",
        "Input is the transcript or script that will be repaired, split into utterances, and assigned to speakers. The Output Folder receives the final FLAC, stems, and optional subtitles. Analyze reads the input without rendering; Render FLAC uses the analyzed plan and the inference path chosen above. The last selected input is remembered, while a fresh install starts with the bundled What is, reality.txt test script when it exists.",
        "input_path",
    ),
    TutorialStage(
        "model",
        "4. Set text and model behavior",
        "Model Variant chooses the Chatterbox model family. Correction Mode controls how much the transcript is repaired before synthesis; Verbatim preserves the source annotation text while the synthesis boundary cleanup prevents marked word junctions from being fused. Turbo is PyTorch-only. These choices must be made before voice and timing choices are interpreted, so this tutorial covers them first.",
        "variant_combo",
    ),
    TutorialStage(
        "voices",
        "5. Configure the voice character",
        "Each speaker's reference clip supplies the identity that Chatterbox imitates. Identity Lock keeps the delivery close to that reference; Emphasis Punch changes pitch and stress movement; Delivery Variety changes sampling surprise; Emotion Depth blends detected emotion into emphasis and pacing; Human Drift moves several sampling controls together toward a looser human read. These controls affect the sound of each line, not the silence between turns.",
        "speaker_a",
    ),
    TutorialStage(
        "timing",
        "6. Shape breaths and turn boundaries",
        "Breath After This Speaker inserts silence after a speaker's turn. The amount is scaled by punctuation: questions, exclamations, ellipses, and trailing commas receive different natural pauses. Crossfade is a separate splice treatment for adjacent audio; it smooths joins but does not create a conversational pause. Set voice identity first, then timing, because timing controls the assembled performance rather than the voice character.",
        "speaker_a",
    ),
    TutorialStage(
        "hybrid",
        "7. Optional hybrid voices",
        "Hybridize Voices is independent of the normal voice modifiers. Pick a second reference, choose Voice Dominance, and choose Mix, Alternate, or Layer. The base voice and second clip are combined deterministically into the conditioning reference; the same inputs produce the same result on CPU and Vulkan. A hybrid changes the voice identity input, so configure it after choosing the base voice and before judging the character sliders.",
        "speaker_a",
    ),
    TutorialStage(
        "review",
        "8. Analyze, preview, then render",
        "Analyze creates the review table so you can inspect repaired text, speaker attribution, emotion, and per-line status before any full render. Preview tests one utterance with the active voice and inference settings. Render FLAC synthesizes the complete plan, applies pacing and crossfades, and writes the output. If a GPU becomes unavailable later, switch to CPU here; the setup wizard can be reopened from Settings at any time.",
        "analyze_button",
    ),
)


class InferenceSetupWizard(QDialog):
    """Hardware discovery plus a dependency-ordered, replayable GUI tour.

    ``mode`` may be ``full``, ``discovery``, or ``main``.  The dialog is
    non-modal so it can highlight a real control and remain out of the user's
    way; Continue advances one stage at a time and the main window applies the
    selected inference choice as soon as the discovery stage is continued.
    """

    completed = Signal(bool, str)

    def __init__(
        self,
        main_window: QWidget,
        *,
        devices: list[CUDADeviceInfo] | None = None,
        mode: str = "full",
        on_selection: Callable[[str, int | None], None] | None = None,
    ) -> None:
        super().__init__(None)
        self.main_window = main_window
        self.devices = list(devices or [])
        self.mode = mode if mode in {"full", "discovery", "main"} else "full"
        self.on_selection = on_selection
        self._stage_index = 0
        self._stages = self._select_stages(self.mode)
        self._highlighted: QWidget | None = None
        self._finished_once = False
        self.setWindowTitle("The Oracle - Inference Setup & Tour")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setModal(False)
        self.setMinimumSize(560, 390)
        self.resize(680, 500)

        self.heading = QLabel()
        self.heading.setWordWrap(True)
        heading_font = QFont()
        heading_font.setPointSize(15)
        heading_font.setBold(True)
        self.heading.setFont(heading_font)
        self.hardware_summary = QLabel()
        self.hardware_summary.setWordWrap(True)
        self.hardware_summary.setObjectName("wizardHardwareSummary")
        self.selection_box = QWidget()
        selection_form = QFormLayout(self.selection_box)
        selection_form.setContentsMargins(0, 0, 0, 0)
        self.backend_picker = QComboBox()
        self.backend_picker.addItem("PyTorch (Chatterbox)", "pytorch")
        # Vulkan is an audio.cpp backend, not Torch's experimental Vulkan
        # backend. It must remain selectable even when torch.backends.vulkan
        # is absent: the main window's existing first-use setup can build the
        # audio.cpp runtime and model after the user chooses it.
        self.backend_picker.addItem("Vulkan (audio.cpp; setup if needed)", "vulkan")
        vulkan_mode = next((item for item in available_device_modes() if item.key == "vulkan"), None)
        if vulkan_mode is not None:
            item = self.backend_picker.model().item(self.backend_picker.count() - 1)
            if item is not None:
                item.setToolTip(
                    "This is The Oracle's audio.cpp Vulkan path. "
                    + vulkan_mode.reason
                    + " The main window will offer automatic setup if its binary or model is missing."
                )
        self.device_picker = QComboBox()
        self.device_picker.addItem("CPU / system DRAM", "cpu")
        current_device = getattr(getattr(main_window, "pytorch_device_combo", None), "currentData", lambda: "cpu")()
        for device in self.devices:
            self.device_picker.addItem(device.label, f"cuda:{device.index}")
            item = self.device_picker.model().item(self.device_picker.count() - 1)
            if item is not None and not (device.torch_available and device.suitable):
                item.setEnabled(False)
                item.setToolTip(device.reason)
        selected_device_index = self.device_picker.findData(current_device)
        self.device_picker.setCurrentIndex(selected_device_index if selected_device_index >= 0 else 0)
        current_backend = getattr(getattr(main_window, "inference_backend_combo", None), "currentData", lambda: "pytorch")()
        backend_index = self.backend_picker.findData(current_backend)
        self.backend_picker.setCurrentIndex(backend_index if backend_index >= 0 else 0)
        selection_form.addRow("Inference path", self.backend_picker)
        selection_form.addRow("PyTorch device", self.device_picker)
        self.selection_box.setObjectName("wizardSelection")
        self.explanation = QPlainTextEdit()
        self.explanation.setReadOnly(True)
        self.explanation.setMinimumHeight(220)
        self.explanation.setObjectName("wizardExplanation")
        self.progress = QLabel()
        self.progress.setObjectName("wizardProgress")
        self.help_note = QLabel(
            "The highlighted control is the feature being explained. Its normal tooltip is shown while this step is active; move the tutorial window if it obscures the control."
        )
        self.help_note.setWordWrap(True)
        self.help_note.setObjectName("wizardHelpNote")
        self.buttons = QDialogButtonBox()
        self.continue_button = QPushButton("Continue")
        self.continue_button.setDefault(True)
        self.continue_button.clicked.connect(self._continue)
        self.skip_button = QPushButton("Skip for now")
        self.skip_button.clicked.connect(self._skip)
        self.buttons.addButton(self.continue_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.addButton(self.skip_button, QDialogButtonBox.ButtonRole.RejectRole)

        layout = QVBoxLayout(self)
        layout.addWidget(self.heading)
        layout.addWidget(self.hardware_summary)
        layout.addWidget(self.selection_box)
        layout.addWidget(self.explanation, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.help_note)
        layout.addWidget(self.buttons)
        self._render_stage()

    @staticmethod
    def _select_stages(mode: str) -> list[TutorialStage]:
        if mode == "discovery":
            return [STAGES[0]]
        if mode == "main":
            return list(STAGES[1:])
        return list(STAGES)

    def _hardware_text(self) -> str:
        return _format_hardware_summary(self.devices)

    def _render_stage(self) -> None:
        stage = self._stages[self._stage_index]
        self.heading.setText(stage.title)
        self.explanation.setPlainText(stage.explanation)
        self.progress.setText(f"Tutorial step {self._stage_index + 1} of {len(self._stages)}")
        self.hardware_summary.setText(self._hardware_text() if stage.discovery else "")
        self.hardware_summary.setVisible(stage.discovery)
        self.selection_box.setVisible(stage.discovery)
        self.continue_button.setText("Finish" if self._stage_index == len(self._stages) - 1 else "Continue")
        self._set_highlight(stage.target_name)
        self._move_near_target(stage.target_name)

    def _target(self, name: str | None) -> QWidget | None:
        if not name:
            return None
        target = getattr(self.main_window, name, None)
        if isinstance(target, QWidget):
            return target
        if name == "speaker_a":
            return getattr(self.main_window, "speaker_a", None)
        return None

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
        if target is None or not target.isVisible():
            center = self.main_window.frameGeometry().center()
            self.move(center.x() - self.width() // 2, center.y() - self.height() // 2)
            return
        target_top_left = target.mapToGlobal(QPoint(0, 0))
        screen = target.screen().availableGeometry() if target.screen() else self.screen().availableGeometry()
        # Keep the dialog beside the target: the target's normal Qt tooltip is
        # shown below it, so placing the tutorial below would cover the very
        # explanation the user is meant to read.
        right_x = target_top_left.x() + target.width() + 18
        left_x = target_top_left.x() - self.width() - 18
        x = right_x if right_x + self.width() <= screen.right() else left_x
        x = min(max(screen.left() + 12, x), screen.right() - self.width() - 12)
        y = target_top_left.y() + max(0, (target.height() - self.height()) // 2)
        y = min(max(screen.top() + 12, y), screen.bottom() - self.height() - 12)
        self.move(x, y)

    def _apply_discovery_selection(self) -> None:
        if not self._stages or not self._stages[self._stage_index].discovery:
            return
        backend = self.backend_picker.currentData()
        if backend == "vulkan":
            if self.on_selection:
                self.on_selection("vulkan", None)
            return
        value = self.device_picker.currentData()
        if isinstance(value, str) and value.startswith("cuda:"):
            try:
                if self.on_selection:
                    self.on_selection("cuda", int(value.split(":", 1)[1]))
            except ValueError:
                pass
        elif value == "cpu" and self.on_selection:
            self.on_selection("cpu", None)

    def _continue(self) -> None:
        self._apply_discovery_selection()
        if self._stage_index >= len(self._stages) - 1:
            self._finish(True)
            return
        self._stage_index += 1
        self._render_stage()

    def _skip(self) -> None:
        self._finish(False)

    def _finish(self, accepted: bool) -> None:
        if self._finished_once:
            return
        self._finished_once = True
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
        QTimer.singleShot(0, lambda: self._move_near_target(self._stages[self._stage_index].target_name))


def _format_hardware_summary(devices: list[CUDADeviceInfo]) -> str:
    modes = {item.key: item for item in available_device_modes()}
    lines = ["Hardware discovery:", "  CPU / system DRAM - available (reliable fallback)."]
    if devices:
        for device in devices:
            status = "available" if device.torch_available and device.suitable else "not selectable"
            lines.append(f"  {device.label} - {status}. {device.reason}")
    else:
        lines.append("  NVIDIA CUDA - no CUDA device was detected by this installation.")
    vulkan = modes.get("vulkan")
    if vulkan is not None:
        state = "available to probe" if vulkan.available else "not verified here"
        lines.append(f"  Vulkan / audio.cpp - {state}. {vulkan.reason}")
    lines.append("A disabled GPU row is informational: use CPU until a suitable GPU and runtime are installed.")
    return "\n".join(lines)


def hardware_summary(devices: list[CUDADeviceInfo] | None = None) -> str:
    """Pure helper used by tests and diagnostics."""
    return _format_hardware_summary(list(devices or []))
