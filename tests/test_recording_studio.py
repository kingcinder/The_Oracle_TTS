import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from the_oracle.app_gui import RecordingStudioDialog

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    voice = repo / "Seashells"
    inputs = repo / "Input"
    voice.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    return repo, voice, inputs


def _dialog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import the_oracle.audio.recorder as recorder_mod

    monkeypatch.setattr(recorder_mod, "_sd", object())  # pretend backend present
    repo, voice, inputs = _make_dirs(tmp_path)
    (inputs / "Read Aloud transcript.txt").write_text("Hello there.\n", encoding="utf-8")
    (inputs / "scene one.md").write_text("# Scene\n", encoding="utf-8")
    (inputs / "ignored.wav").write_bytes(b"\x00")
    return RecordingStudioDialog(repo, voice, inputs), voice


def test_teleprompter_lists_only_text_scripts(tmp_path, monkeypatch, qt_app):
    dialog, _ = _dialog(tmp_path, monkeypatch)
    names = [dialog.script_combo.itemText(i) for i in range(dialog.script_combo.count())]
    assert "Read Aloud transcript.txt" in names
    assert "scene one.md" in names
    assert not any(name.endswith(".wav") for name in names)


def test_default_filename_auto_increments_and_never_overwrites(tmp_path, monkeypatch, qt_app):
    dialog, voice = _dialog(tmp_path, monkeypatch)
    assert dialog.name_edit.text() == "Seashell_No_1.wav"
    (voice / "Seashell_No_1.wav").write_bytes(b"x")
    (voice / "Seashell_No_2.wav").write_bytes(b"x")
    # Re-open computes the next free number from the folder contents.
    dialog.close()
    repo, voice2, inputs = _make_dirs(tmp_path)
    (voice2 / "Seashell_No_1.wav").write_bytes(b"x")
    (voice2 / "Seashell_No_3.wav").write_bytes(b"x")
    second = RecordingStudioDialog(tmp_path / "repo", voice2, inputs)
    assert second.name_edit.text() == "Seashell_No_4.wav"


def test_no_microphone_disables_record(tmp_path, monkeypatch, qt_app):
    import the_oracle.audio.recorder as recorder_mod

    repo, voice, inputs = _make_dirs(tmp_path)
    monkeypatch.setattr(recorder_mod, "_sd", object())
    monkeypatch.setattr(recorder_mod, "list_input_devices", lambda: [])
    dialog = RecordingStudioDialog(repo, voice, inputs)
    assert dialog.record_button.isEnabled() is False
    assert "no microphone" in dialog.mic_combo.itemText(0)


def test_rate_menu_appears_after_mic_selected(tmp_path, monkeypatch, qt_app):
    import the_oracle.audio.recorder as recorder_mod

    class _Device:
        index = 0
        name = "USB Mic"
        max_input_channels = 1
        default_samplerate = 48000

    monkeypatch.setattr(recorder_mod, "list_input_devices", lambda: [_Device()])
    monkeypatch.setattr(recorder_mod, "samplerates_for_device", lambda index: [48000, 44100])
    repo, voice, inputs = _make_dirs(tmp_path)
    dialog = RecordingStudioDialog(repo, voice, inputs)
    assert dialog.record_button.isEnabled() is True
    assert dialog.rate_combo.isEnabled() is True
    assert dialog.rate_combo.currentData() == 48000
    assert dialog.rate_combo.count() == 2


def _enabled_dialog(tmp_path, monkeypatch):
    import the_oracle.audio.recorder as recorder_mod

    class _Device:
        index = 0
        name = "USB Mic"
        max_input_channels = 1
        default_samplerate = 48000

    monkeypatch.setattr(recorder_mod, "_sd", object())
    monkeypatch.setattr(recorder_mod, "list_input_devices", lambda: [_Device()])
    monkeypatch.setattr(recorder_mod, "samplerates_for_device", lambda index: [48000, 44100])
    repo, voice, inputs = _make_dirs(tmp_path)
    return RecordingStudioDialog(repo, voice, inputs), voice


class _TakeSignal:
    def connect(self, *_args, **_kwargs) -> None:
        return None


class _FakeMediaPlayer:
    last = None

    def __init__(self, *_args, **_kwargs) -> None:
        self.source = None
        self.played = False
        self.stopped = 0
        self.mediaStatusChanged = _TakeSignal()
        _FakeMediaPlayer.last = self

    def setAudioOutput(self, output) -> None:
        self.audio_output = output

    def setSource(self, source) -> None:
        self.source = source

    def play(self) -> None:
        self.played = True

    def stop(self) -> None:
        self.stopped += 1

    def deleteLater(self) -> None:
        return None


class _FakeAudioOutput:
    def __init__(self, *_args, **_kwargs) -> None:
        pass


def test_successful_take_auto_auditions_and_enables_actions(tmp_path, monkeypatch, qt_app):
    import numpy as np

    import the_oracle.app_gui as app_gui

    monkeypatch.setattr(app_gui, "QMediaPlayer", _FakeMediaPlayer)
    monkeypatch.setattr(app_gui, "QAudioOutput", _FakeAudioOutput)
    dialog, voice = _enabled_dialog(tmp_path, monkeypatch)
    assert dialog.audition_check.isChecked() is True
    assert dialog.play_button.isEnabled() is False
    assert dialog.assign_a_button.isEnabled() is False

    audio = np.full(4800, 0.2, dtype=np.float32)
    dialog._on_captured(audio)  # simulate a finished take arriving from the worker

    saved = voice / "Seashell_No_1.wav"
    assert saved.exists()
    assert dialog._last_saved_path == saved
    assert dialog.play_button.isEnabled() is True
    assert dialog.assign_a_button.isEnabled() is True
    assert dialog.assign_b_button.isEnabled() is True
    player = _FakeMediaPlayer.last
    assert player is not None and player.played is True
    assert player.source.toLocalFile().endswith("Seashell_No_1.wav")


def test_auto_audition_can_be_disabled(tmp_path, monkeypatch, qt_app):
    import numpy as np

    import the_oracle.app_gui as app_gui

    monkeypatch.setattr(app_gui, "QMediaPlayer", _FakeMediaPlayer)
    monkeypatch.setattr(app_gui, "QAudioOutput", _FakeAudioOutput)
    _FakeMediaPlayer.last = None
    dialog, voice = _enabled_dialog(tmp_path, monkeypatch)
    dialog.audition_check.setChecked(False)
    dialog._on_captured(np.full(4800, 0.2, dtype=np.float32))
    assert (voice / "Seashell_No_1.wav").exists()
    assert _FakeMediaPlayer.last is None  # nothing auto-played


def test_assign_to_speaker_invokes_callback_with_path(tmp_path, monkeypatch, qt_app):
    assigned = []
    repo, voice, inputs = _make_dirs(tmp_path)
    saved = voice / "Seashell_No_1.wav"
    saved.write_bytes(b"\x00" * 44)
    dialog = RecordingStudioDialog(repo, voice, inputs, on_assign=lambda speaker, path: assigned.append((speaker, path)))
    dialog._last_saved_path = saved
    dialog._refresh_take_actions()
    assert dialog.assign_a_button.isEnabled() is True
    assert dialog.assign_b_button.isEnabled() is True
    dialog._use_for_speaker("A")
    assert assigned == [("A", saved)]


def test_main_window_assign_wires_the_group_reference(monkeypatch, tmp_path, qt_app):
    window, paths = _build_main_window(monkeypatch, tmp_path)
    saved = paths.voice_dir / "Seashell_No_9.wav"
    saved.write_bytes(b"\x00" * 44)
    refresh_calls = []
    window._refresh_reference_pickers = lambda: refresh_calls.append(1)
    window._assign_recording_to_speaker("B", saved)
    assert window.speaker_b.reference_path.text() == str(saved)
    assert refresh_calls


def _build_main_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Minimal MainWindow with the same fakes the profile GUI tests use."""
    import the_oracle.app_gui as app_gui
    from tests.test_app_gui_profiles import (
        _FakeAudioOutput,
        _FakeMediaPlayer,
        _FakePipeline,
        _FakeChatterboxEngine,
        _FakeVulkanProbeThread,
        _FakeVulkanPreflightThread,
        _FakeVulkanSetupThread,
        _FakeModelDownloadThread,
    )
    from the_oracle.app_paths import ensure_repo_default_paths

    paths = ensure_repo_default_paths(tmp_path / "repo")
    monkeypatch.setattr(app_gui, "QAudioOutput", _FakeAudioOutput)
    monkeypatch.setattr(app_gui, "QMediaPlayer", _FakeMediaPlayer)
    monkeypatch.setattr(app_gui, "OraclePipeline", _FakePipeline)
    monkeypatch.setattr(app_gui, "ChatterboxEngine", _FakeChatterboxEngine)
    monkeypatch.setattr(app_gui, "VulkanDeviceProbeThread", _FakeVulkanProbeThread)
    monkeypatch.setattr(app_gui, "VulkanPreflightThread", _FakeVulkanPreflightThread)
    monkeypatch.setattr(app_gui, "VulkanSetupThread", _FakeVulkanSetupThread)
    monkeypatch.setattr(app_gui, "ModelDownloadThread", _FakeModelDownloadThread)
    monkeypatch.setattr(app_gui, "ensure_repo_default_paths", lambda _repo_root: paths)
    monkeypatch.setattr(app_gui, "default_voice_choices", lambda _repo_root, limit=10: [])
    monkeypatch.setattr(app_gui, "blend_voice_choices", lambda _profiles: [])
    monkeypatch.setattr(app_gui, "load_recent_reference_paths", lambda limit=10: [])
    monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: tmp_path / "audiocpp_cli")
    model_file = tmp_path / "chatterbox-model"
    model_file.write_text("model", encoding="utf-8")
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model_file))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return app_gui.MainWindow(), paths


def test_main_window_recording_studio_action_and_close_refresh(monkeypatch, tmp_path, qt_app):
    window, paths = _build_main_window(monkeypatch, tmp_path)
    assert window.recording_studio_action.text().startswith("Recording Studio")
    refresh_calls = []
    window._refresh_reference_pickers = lambda: refresh_calls.append(1)

    window.open_recording_studio()
    dialog = window._recording_studio
    assert dialog is not None
    # Closing with no successful recording must still refresh the pickers.
    dialog.close()
    assert refresh_calls, "voice pickers must refresh when the studio closes without a recording"


def test_main_window_reports_saved_seashell_and_refreshes(monkeypatch, tmp_path, qt_app):
    window, paths = _build_main_window(monkeypatch, tmp_path)
    refresh_calls = []
    window._refresh_reference_pickers = lambda: refresh_calls.append(1)

    window.open_recording_studio()
    dialog = window._recording_studio
    recorded = paths.voice_dir / "Seashell_No_1.wav"
    recorded.write_bytes(b"\x00" * 44)  # only the path matters for this test
    dialog._last_saved_path = recorded
    dialog.close()
    assert refresh_calls
    assert "Recorded new Seashell: Seashell_No_1.wav" in window.error_panel.toPlainText()
