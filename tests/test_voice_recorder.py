import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QByteArray
from PySide6.QtMultimedia import QAudioFormat, QMediaPlayer
from PySide6.QtWidgets import QApplication

from the_oracle.app_paths import ensure_repo_default_paths
from the_oracle.audio.export_flac import write_flac
from the_oracle.voice_recorder import (
    COMMON_MIC_SAMPLE_RATES,
    VOICE_RECORDING_CHANNELS,
    VoiceRecorderDialog,
    build_recording_format,
    format_sample_rate_label,
    build_voice_recording_path,
    pcm_bytes_to_mono_float32,
    sanitize_voice_clip_name,
    supported_recording_sample_rates,
)


class _Signal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, value) -> None:
        for callback in list(self._callbacks):
            callback(value)


class _FakeAudioOutput:
    def __init__(self, *_args, **_kwargs) -> None:
        pass


class _FakeMediaPlayer:
    def __init__(self, *_args, **_kwargs) -> None:
        self.audio_output = None
        self.source = None
        self._state = QMediaPlayer.PlaybackState.StoppedState
        self.playbackStateChanged = _Signal()

    def setAudioOutput(self, output) -> None:
        self.audio_output = output

    def playbackState(self):
        return self._state

    def setSource(self, source) -> None:
        self.source = source

    def play(self) -> None:
        self._state = QMediaPlayer.PlaybackState.PlayingState
        self.playbackStateChanged.emit(self._state)

    def pause(self) -> None:
        self._state = QMediaPlayer.PlaybackState.PausedState
        self.playbackStateChanged.emit(self._state)

    def stop(self) -> None:
        self._state = QMediaPlayer.PlaybackState.StoppedState
        self.playbackStateChanged.emit(self._state)


class _FakeAudioDevice:
    def __init__(self, name: str = "Studio Mic") -> None:
        self._name = name
        self._preferred = QAudioFormat()
        self._preferred.setSampleRate(44_100)
        self._preferred.setChannelCount(2)
        self._preferred.setSampleFormat(QAudioFormat.SampleFormat.Int16)

    def description(self) -> str:
        return self._name

    def id(self):
        return QByteArray(self._name.encode("utf-8"))

    def preferredFormat(self) -> QAudioFormat:
        return self._preferred

    def isFormatSupported(self, audio_format: QAudioFormat) -> bool:
        return (
            audio_format.sampleRate() in COMMON_MIC_SAMPLE_RATES
            and audio_format.channelCount() == VOICE_RECORDING_CHANNELS
            and audio_format.sampleFormat() == QAudioFormat.SampleFormat.Int16
        )


class _StereoOnlyFloatDevice(_FakeAudioDevice):
    def __init__(self, name: str = "USB PnP Audio Device") -> None:
        super().__init__(name=name)
        self._preferred.setSampleRate(48_000)
        self._preferred.setChannelCount(2)
        self._preferred.setSampleFormat(QAudioFormat.SampleFormat.Float)

    def isFormatSupported(self, audio_format: QAudioFormat) -> bool:
        return (
            audio_format.sampleRate() in COMMON_MIC_SAMPLE_RATES
            and audio_format.channelCount() == 2
            and audio_format.sampleFormat() == QAudioFormat.SampleFormat.Float
        )


class _FakeAudioDevices:
    def __init__(self) -> None:
        self.device = _FakeAudioDevice()

    def audio_inputs(self):
        return [self.device]

    def default_audio_input(self):
        return self.device


class _FakeCaptureSession:
    def __init__(self, _device, sample_rate: int) -> None:
        self.started = False
        self.is_paused = False
        self.sample_rate = sample_rate

    def start(self) -> None:
        self.started = True

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False

    def stop(self, destination: Path, metadata: dict[str, str]) -> Path:
        audio = np.linspace(-0.2, 0.2, num=max(200, self.sample_rate // 8), dtype=np.float32)
        return write_flac(destination, audio, self.sample_rate, metadata)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_sanitize_voice_clip_name_and_output_path(tmp_path: Path) -> None:
    voice_dir = tmp_path / "Seashells"
    voice_dir.mkdir()

    assert sanitize_voice_clip_name("  My Voice / Clip 01  ") == "My_Voice_Clip_01"
    assert build_voice_recording_path(voice_dir, "  My Voice / Clip 01  ").name == "My_Voice_Clip_01.flac"
    assert format_sample_rate_label(44_100) == "44.1 kHz"
    assert format_sample_rate_label(48_000) == "48 kHz"


def test_supported_recording_sample_rates_cover_common_frequency_catalog() -> None:
    supported = supported_recording_sample_rates(_FakeAudioDevice())
    assert supported == list(COMMON_MIC_SAMPLE_RATES)


def test_supported_recording_sample_rates_cover_stereo_only_float_devices() -> None:
    supported = supported_recording_sample_rates(_StereoOnlyFloatDevice())
    assert supported == list(COMMON_MIC_SAMPLE_RATES)


@pytest.mark.parametrize("sample_rate", COMMON_MIC_SAMPLE_RATES)
def test_build_recording_format_supports_each_common_rate(sample_rate: int) -> None:
    audio_format = build_recording_format(_FakeAudioDevice(), sample_rate)
    assert audio_format.sampleRate() == sample_rate
    assert audio_format.channelCount() == VOICE_RECORDING_CHANNELS
    assert audio_format.sampleFormat() == QAudioFormat.SampleFormat.Int16


def test_build_recording_format_falls_back_to_preferred_channels_when_mono_is_unsupported() -> None:
    audio_format = build_recording_format(_StereoOnlyFloatDevice(), 48_000)
    assert audio_format.sampleRate() == 48_000
    assert audio_format.channelCount() == 2
    assert audio_format.sampleFormat() == QAudioFormat.SampleFormat.Float


def test_pcm_bytes_to_mono_float32_downmixes_stereo_int16() -> None:
    audio_format = QAudioFormat()
    audio_format.setSampleRate(44_100)
    audio_format.setChannelCount(2)
    audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    samples = np.asarray([[32767, -32768], [16384, 16384]], dtype=np.int16)

    result = pcm_bytes_to_mono_float32(samples.tobytes(), audio_format)

    assert result.shape == (2,)
    assert result[0] == pytest.approx(-1.5258789e-05, abs=1e-6)
    assert result[1] == pytest.approx(0.5, abs=1e-4)


def test_voice_recorder_defaults_to_microphone_preferred_sample_rate(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = ensure_repo_default_paths(tmp_path / "repo")
    dialog = VoiceRecorderDialog(
        paths,
        capture_factory=_FakeCaptureSession,
        player_factory=_FakeMediaPlayer,
        audio_output_factory=_FakeAudioOutput,
        audio_devices=_FakeAudioDevices(),
    )

    try:
        assert dialog.sample_rate_combo.currentData() == 44_100
        assert dialog.sample_rate_combo.currentText() == "44.1 kHz"
    finally:
        dialog.close()


@pytest.mark.parametrize("sample_rate", COMMON_MIC_SAMPLE_RATES)
def test_voice_recorder_dialog_loads_prompt_and_records_flac(
    qt_app,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_rate: int,
) -> None:
    paths = ensure_repo_default_paths(tmp_path / "repo")
    prompt_path = paths.input_dir / "sample_prompt.txt"
    prompt_path.write_text("Read this script into the mic.\nSlow and steady.\n", encoding="utf-8")

    dialog = VoiceRecorderDialog(
        paths,
        capture_factory=_FakeCaptureSession,
        player_factory=_FakeMediaPlayer,
        audio_output_factory=_FakeAudioOutput,
        audio_devices=_FakeAudioDevices(),
    )

    try:
        monkeypatch.setattr(
            "the_oracle.voice_recorder.QFileDialog.getOpenFileName",
            lambda *_args, **_kwargs: (str(prompt_path), ""),
        )

        dialog._pick_prompt_file()

        assert dialog.prompt_path.text() == str(prompt_path)
        assert "Slow and steady." in dialog.prompt_view.toPlainText()
        assert dialog.clip_name.text() == "sample_prompt"
        assert paths.voice_dir.name in dialog.output_path_label.text()
        assert dialog.sample_rate_combo.count() == len(COMMON_MIC_SAMPLE_RATES)
        assert dialog.record_button.minimumHeight() >= 60
        assert dialog.stop_button.minimumHeight() >= 60
        assert dialog.playback_button.minimumHeight() >= 60
        assert dialog.pause_button.minimumHeight() >= 60
        dialog.sample_rate_combo.setCurrentIndex(dialog.sample_rate_combo.findData(sample_rate))
        assert dialog.sample_rate_combo.currentData() == sample_rate

        dialog.start_recording()
        assert dialog._capture_session is not None
        assert dialog.pause_button.isEnabled() is True

        dialog.pause_transport()
        assert dialog.pause_button.text() == "Resume"

        dialog.pause_transport()
        assert dialog.pause_button.text() == "Pause"

        dialog.stop_transport()
        saved = paths.voice_dir / "sample_prompt.flac"
        assert saved.exists()
        assert dialog._last_recording_path == saved
        assert "Saved recording" in dialog.status_label.text()
        _audio, actual_sample_rate = sf.read(saved)
        assert actual_sample_rate == sample_rate

        dialog.playback_recording()
        assert str(dialog.player.source.toString()).endswith("sample_prompt.flac")

        dialog.pause_transport()
        assert dialog.player.playbackState() == QMediaPlayer.PlaybackState.PausedState

        dialog.stop_transport()
        assert dialog.player.playbackState() == QMediaPlayer.PlaybackState.StoppedState
    finally:
        dialog.close()
