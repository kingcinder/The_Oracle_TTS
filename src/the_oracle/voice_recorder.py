"""Voice-recording workflow for creating repo-local custom reference clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt, QUrl
from PySide6.QtMultimedia import QAudioDevice, QAudioFormat, QAudioOutput, QAudioSource, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from the_oracle.app_paths import OraclePaths
from the_oracle.audio.export_flac import write_flac


VOICE_RECORDING_SAMPLE_RATE = 48_000
VOICE_RECORDING_CHANNELS = 1
_DEFAULT_CLIP_NAME = "voice_sample"


def sanitize_voice_clip_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or _DEFAULT_CLIP_NAME


def load_prompt_text(path: str | Path) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8", errors="replace")


def build_voice_recording_path(voice_dir: str | Path, clip_name: str) -> Path:
    return Path(voice_dir).expanduser().resolve() / f"{sanitize_voice_clip_name(clip_name)}.flac"


def build_recording_format(device: QAudioDevice) -> QAudioFormat:
    preferred = device.preferredFormat()
    candidates = [
        preferred.sampleFormat(),
        QAudioFormat.SampleFormat.Int16,
        QAudioFormat.SampleFormat.Float,
        QAudioFormat.SampleFormat.Int32,
        QAudioFormat.SampleFormat.UInt8,
    ]
    for sample_format in candidates:
        if sample_format == QAudioFormat.SampleFormat.Unknown:
            continue
        audio_format = QAudioFormat()
        audio_format.setSampleRate(VOICE_RECORDING_SAMPLE_RATE)
        audio_format.setChannelCount(VOICE_RECORDING_CHANNELS)
        audio_format.setSampleFormat(sample_format)
        if device.isFormatSupported(audio_format):
            return audio_format
    raise ValueError(f"{device.description()} does not support mono {VOICE_RECORDING_SAMPLE_RATE} Hz capture.")


def pcm_bytes_to_mono_float32(raw: bytes, audio_format: QAudioFormat) -> np.ndarray:
    sample_format = audio_format.sampleFormat()
    channel_count = max(1, audio_format.channelCount())
    bytes_per_sample = audio_format.bytesPerSample()
    if bytes_per_sample <= 0:
        raise ValueError("Unsupported audio sample size.")

    frame_width = bytes_per_sample * channel_count
    if frame_width <= 0:
        raise ValueError("Invalid audio frame width.")
    usable = len(raw) - (len(raw) % frame_width)
    raw = raw[:usable]
    if not raw:
        return np.asarray([], dtype=np.float32)

    if sample_format == QAudioFormat.SampleFormat.UInt8:
        array = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        array = (array - 128.0) / 128.0
    elif sample_format == QAudioFormat.SampleFormat.Int16:
        array = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_format == QAudioFormat.SampleFormat.Int32:
        array = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_format == QAudioFormat.SampleFormat.Float:
        array = np.frombuffer(raw, dtype=np.float32).astype(np.float32)
    else:
        raise ValueError(f"Unsupported sample format: {sample_format}")

    if channel_count > 1:
        array = array.reshape(-1, channel_count).mean(axis=1)
    return np.clip(array, -1.0, 1.0).astype(np.float32)


@dataclass(slots=True)
class QtAudioCaptureSession:
    device: QAudioDevice

    def __post_init__(self) -> None:
        self.audio_format = build_recording_format(self.device)
        self._buffer_bytes = QByteArray()
        self._buffer = QBuffer(self._buffer_bytes)
        self._audio_source: QAudioSource | None = None
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    def start(self) -> None:
        self._buffer_bytes.clear()
        self._buffer.open(QIODevice.OpenModeFlag.WriteOnly | QIODevice.OpenModeFlag.Truncate)
        self._audio_source = QAudioSource(self.device, self.audio_format)
        self._audio_source.start(self._buffer)
        self._paused = False

    def pause(self) -> None:
        if self._audio_source is not None:
            self._audio_source.suspend()
            self._paused = True

    def resume(self) -> None:
        if self._audio_source is not None:
            self._audio_source.resume()
            self._paused = False

    def stop(self, destination: Path, metadata: dict[str, str]) -> Path:
        if self._audio_source is None:
            raise RuntimeError("Recording has not started.")
        self._audio_source.stop()
        self._buffer.close()
        audio = pcm_bytes_to_mono_float32(bytes(self._buffer_bytes), self.audio_format)
        self._audio_source.deleteLater()
        self._audio_source = None
        self._paused = False
        if audio.size == 0:
            raise ValueError("No audio was captured from the selected microphone.")
        return write_flac(destination, audio, self.audio_format.sampleRate(), metadata)


class _SystemAudioDevices:
    def audio_inputs(self) -> list[QAudioDevice]:
        return list(QMediaDevices.audioInputs())

    def default_audio_input(self) -> QAudioDevice:
        return QMediaDevices.defaultAudioInput()


class VoiceRecorderDialog(QDialog):
    def __init__(
        self,
        paths: OraclePaths,
        *,
        parent: QWidget | None = None,
        capture_factory=None,
        player_factory=QMediaPlayer,
        audio_output_factory=QAudioOutput,
        audio_devices=None,
    ) -> None:
        super().__init__(parent)
        self.paths = paths
        self._capture_factory = capture_factory or (lambda device: QtAudioCaptureSession(device))
        self._audio_devices = audio_devices or _SystemAudioDevices()
        self._capture_session = None
        self._last_recording_path: Path | None = None

        self.player = player_factory(self)
        self.audio_output = audio_output_factory(self)
        self.player.setAudioOutput(self.audio_output)
        if hasattr(self.player, "playbackStateChanged"):
            self.player.playbackStateChanged.connect(lambda _state: self._sync_transport_buttons())

        self.setWindowTitle("Voice Recorder")
        self.resize(980, 760)
        self._build_ui()
        self.refresh_microphones()
        self._sync_transport_buttons()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Record Custom Voice References")
        title.setObjectName("appTitle")
        subtitle = QLabel(
            f"Capture mono {VOICE_RECORDING_SAMPLE_RATE} Hz reference audio directly into {self.paths.voice_dir.name}."
        )
        subtitle.setObjectName("appSummary")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls = QGridLayout()
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(8)

        self.prompt_path = QLineEdit()
        self.prompt_path.setReadOnly(True)
        self.prompt_browse = QPushButton("Choose Text")
        self.prompt_browse.clicked.connect(self._pick_prompt_file)
        self.prompt_browse.setProperty("utilityButton", True)

        self.clip_name = QLineEdit()
        self.clip_name.setPlaceholderText("voice_sample")

        self.mic_combo = QComboBox()
        self.refresh_mics_button = QPushButton("Refresh Mics")
        self.refresh_mics_button.clicked.connect(self.refresh_microphones)
        self.refresh_mics_button.setProperty("utilityButton", True)

        self.output_path_label = QLabel(str(build_voice_recording_path(self.paths.voice_dir, _DEFAULT_CLIP_NAME)))
        self.output_path_label.setObjectName("appSummary")

        controls.addWidget(self._field_label("Prompt File"), 0, 0)
        controls.addWidget(self.prompt_path, 0, 1)
        controls.addWidget(self.prompt_browse, 0, 2)
        controls.addWidget(self._field_label("Clip Name"), 1, 0)
        controls.addWidget(self.clip_name, 1, 1)
        controls.addWidget(self._field_label("Microphone"), 2, 0)
        controls.addWidget(self.mic_combo, 2, 1)
        controls.addWidget(self.refresh_mics_button, 2, 2)
        controls.addWidget(self._field_label("Saves To"), 3, 0)
        controls.addWidget(self.output_path_label, 3, 1, 1, 2)

        self.prompt_view = QTextEdit()
        self.prompt_view.setReadOnly(True)
        self.prompt_view.setPlaceholderText("Choose a text file to display the recording script here.")
        self.prompt_view.setMinimumHeight(360)

        transport_row = QHBoxLayout()
        transport_row.setSpacing(12)

        self.record_button = QPushButton("Record")
        self.stop_button = QPushButton("Stop")
        self.playback_button = QPushButton("Playback")
        self.pause_button = QPushButton("Pause")
        for button in (self.record_button, self.stop_button, self.playback_button, self.pause_button):
            button.setMinimumHeight(62)
            button.setMinimumWidth(150)
            button.setProperty("transportButton", True)
        self.record_button.setProperty("recordButton", True)
        self.playback_button.setProperty("accentButton", True)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_transport)
        self.playback_button.clicked.connect(self.playback_recording)
        self.pause_button.clicked.connect(self.pause_transport)

        transport_row.addWidget(self.record_button)
        transport_row.addWidget(self.stop_button)
        transport_row.addWidget(self.playback_button)
        transport_row.addWidget(self.pause_button)

        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("appSummary")

        layout.addLayout(controls)
        layout.addWidget(self.prompt_view, stretch=1)
        layout.addLayout(transport_row)
        layout.addWidget(self.status_label)

        self.clip_name.textChanged.connect(self._update_output_path_preview)

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def refresh_microphones(self) -> None:
        current_id = self._selected_device_id()
        self.mic_combo.clear()
        inputs = self._audio_devices.audio_inputs()
        default_input = self._audio_devices.default_audio_input() if inputs else None
        for device in inputs:
            self.mic_combo.addItem(device.description(), device)
        if not inputs:
            self.status_label.setText("No microphones detected.")
            self._sync_transport_buttons()
            return
        target_id = current_id or (bytes(default_input.id()) if default_input is not None else None)
        if target_id is not None:
            for index in range(self.mic_combo.count()):
                device = self.mic_combo.itemData(index)
                if device is not None and bytes(device.id()) == target_id:
                    self.mic_combo.setCurrentIndex(index)
                    break
        self.status_label.setText(f"{len(inputs)} microphone(s) available.")
        self._sync_transport_buttons()

    def _selected_device_id(self) -> bytes | None:
        device = self.mic_combo.currentData()
        if device is None:
            return None
        return bytes(device.id())

    def _selected_device(self):
        return self.mic_combo.currentData()

    def _pick_prompt_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Recording Script",
            str(self.paths.input_dir),
            "Text Files (*.txt *.md)",
        )
        if not path:
            return
        self.prompt_path.setText(path)
        self.prompt_view.setPlainText(load_prompt_text(path))
        if not self.clip_name.text().strip():
            self.clip_name.setText(Path(path).stem)
        self.status_label.setText(f"Loaded prompt: {Path(path).name}")
        self._update_output_path_preview()

    def _update_output_path_preview(self) -> None:
        self.output_path_label.setText(str(build_voice_recording_path(self.paths.voice_dir, self.clip_name.text() or _DEFAULT_CLIP_NAME)))

    def _recording_destination(self) -> Path:
        return build_voice_recording_path(self.paths.voice_dir, self.clip_name.text() or _DEFAULT_CLIP_NAME)

    def start_recording(self) -> None:
        if self._capture_session is not None:
            self.status_label.setText("Recording is already in progress.")
            return
        device = self._selected_device()
        if device is None:
            QMessageBox.critical(self, "Recorder", "No microphone is available to record from.")
            return
        clip_name = sanitize_voice_clip_name(self.clip_name.text())
        self.clip_name.setText(clip_name)
        try:
            self._capture_session = self._capture_factory(device)
            self._capture_session.start()
        except Exception as exc:
            self._capture_session = None
            QMessageBox.critical(self, "Recorder", str(exc))
            self.status_label.setText(f"Record failed: {exc}")
            return
        self.status_label.setText(f"Recording to {clip_name}.flac from {device.description()}.")
        self._sync_transport_buttons()

    def stop_transport(self) -> None:
        if self._capture_session is not None:
            destination = self._recording_destination()
            metadata = {
                "title": sanitize_voice_clip_name(destination.stem),
                "software": "The Oracle Voice Recorder",
                "prompt_file": self.prompt_path.text(),
                "sample_rate": str(VOICE_RECORDING_SAMPLE_RATE),
                "channels": "1",
            }
            try:
                saved_path = self._capture_session.stop(destination, metadata)
            except Exception as exc:
                QMessageBox.critical(self, "Recorder", str(exc))
                self.status_label.setText(f"Stop failed: {exc}")
            else:
                self._last_recording_path = saved_path
                self.status_label.setText(f"Saved recording: {saved_path.name}")
            finally:
                self._capture_session = None
                self._sync_transport_buttons()
            return

        if hasattr(self.player, "stop"):
            self.player.stop()
            self.status_label.setText("Playback stopped.")
        self._sync_transport_buttons()

    def playback_recording(self) -> None:
        if self._capture_session is not None:
            self.status_label.setText("Stop recording before playback.")
            return
        recording_path = self._last_recording_path or self._recording_destination()
        if not recording_path.exists():
            QMessageBox.critical(self, "Recorder", "No recorded FLAC file is available for playback yet.")
            return
        self.player.setSource(QUrl.fromLocalFile(str(recording_path)))
        self.player.play()
        self.status_label.setText(f"Playing back {recording_path.name}.")
        self._sync_transport_buttons()

    def pause_transport(self) -> None:
        if self._capture_session is not None:
            if self._capture_session.is_paused:
                self._capture_session.resume()
                self.status_label.setText("Recording resumed.")
            else:
                self._capture_session.pause()
                self.status_label.setText("Recording paused.")
            self._sync_transport_buttons()
            return

        if hasattr(self.player, "playbackState"):
            playing_state = getattr(QMediaPlayer.PlaybackState, "PlayingState", None)
            paused_state = getattr(QMediaPlayer.PlaybackState, "PausedState", None)
            if self.player.playbackState() == playing_state:
                self.player.pause()
                self.status_label.setText("Playback paused.")
            elif self.player.playbackState() == paused_state:
                self.player.play()
                self.status_label.setText("Playback resumed.")
        self._sync_transport_buttons()

    def _sync_transport_buttons(self) -> None:
        has_mic = self.mic_combo.count() > 0
        is_recording = self._capture_session is not None
        is_paused = bool(is_recording and self._capture_session.is_paused)
        playing_state = getattr(QMediaPlayer.PlaybackState, "PlayingState", None)
        paused_state = getattr(QMediaPlayer.PlaybackState, "PausedState", None)
        playback_state = self.player.playbackState() if hasattr(self.player, "playbackState") else None
        is_playing = playback_state == playing_state
        is_playback_paused = playback_state == paused_state

        self.record_button.setEnabled(has_mic and not is_recording and not is_playing and not is_playback_paused)
        self.stop_button.setEnabled(is_recording or is_playing or is_playback_paused)
        self.playback_button.setEnabled((self._last_recording_path or self._recording_destination()).exists() and not is_recording)
        self.pause_button.setEnabled(is_recording or is_playing or is_playback_paused)
        self.pause_button.setText("Resume" if is_paused else "Pause")

