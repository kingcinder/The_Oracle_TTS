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


class _TakeSignal:
    def __init__(self) -> None:
        self._slots: list = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def emit(self, *args) -> None:
        for slot in list(self._slots):
            slot(*args)


class _SpyMediaPlayer:
    """Fake with an actually-emittable mediaStatusChanged signal."""

    last = None

    def __init__(self, *_args, **_kwargs) -> None:
        self.source = None
        self.played = 0
        self.stopped = 0
        self.deleted = False
        self.mediaStatusChanged = _TakeSignal()
        _SpyMediaPlayer.last = self

    def setAudioOutput(self, output) -> None:
        self.audio_output = output

    def setSource(self, source) -> None:
        self.source = source

    def play(self) -> None:
        self.played += 1

    def stop(self) -> None:
        self.stopped += 1

    def deleteLater(self) -> None:
        self.deleted = True


class _SpyAudioOutput:
    def __init__(self, *_args, **_kwargs) -> None:
        pass


def test_end_of_media_does_not_delete_or_stop_player_in_handler(
    tmp_path, monkeypatch, qt_app
):
    """Regression: deleting the QMediaPlayer from inside its own
    mediaStatusChanged handler is a QtMultimedia use-after-free (the kernel
    crash signature). The handler must defer the stop and never delete the
    persistent player.
    """
    import numpy as np

    import the_oracle.app_gui as app_gui

    monkeypatch.setattr(app_gui, "QMediaPlayer", _SpyMediaPlayer)
    monkeypatch.setattr(app_gui, "QAudioOutput", _SpyAudioOutput)
    _SpyMediaPlayer.last = None
    dialog, voice = _enabled_dialog(tmp_path, monkeypatch)
    dialog._on_captured(np.full(4800, 0.2, dtype=np.float32))  # saves + auditions
    player = _SpyMediaPlayer.last
    assert player is not None
    assert player.played == 1

    # EndOfMedia arrives: the old code called stop() + deleteLater() right here
    # (crash). Now it must defer and keep the persistent player alive.
    dialog._on_playback_status(player.mediaStatusChanged)
    assert player.deleted is False, "player must never be deleteLater'd by its own handler"
    # The zero-timer defers the stop; drive the event loop so it lands.
    import PySide6.QtCore as qtcore

    qtcore.QTimer.singleShot(0, lambda: None)
    dialog._on_playback_status(player.mediaStatusChanged)
    assert dialog._player is player, "persistent player must survive EndOfMedia"


def test_playback_reuses_one_player_across_takes(tmp_path, monkeypatch, qt_app):
    """Regression: a fresh QMediaPlayer per take (stop + deleteLater each
    time) races the FFmpeg backend threads. The dialog must reuse one player.
    """
    import numpy as np

    import the_oracle.app_gui as app_gui

    monkeypatch.setattr(app_gui, "QMediaPlayer", _SpyMediaPlayer)
    monkeypatch.setattr(app_gui, "QAudioOutput", _SpyAudioOutput)
    dialog, voice = _enabled_dialog(tmp_path, monkeypatch)
    dialog._on_captured(np.full(4800, 0.2, dtype=np.float32))
    first = _SpyMediaPlayer.last
    assert first is not None
    # Second take: same player object reused, old one never deleted.
    (voice / "Seashell_No_2.wav").write_bytes(b"\x00" * 44)
    dialog._last_saved_path = voice / "Seashell_No_2.wav"
    dialog._play_take()
    assert dialog._player is first
    assert first.deleted is False
    assert first.played == 2


class _SpyWorker:
    """Stand-in for RecordStudioWorker exposing the finished lifecycle."""

    def __init__(self) -> None:
        self.finished = _TakeSignal()
        self.delete_later_called = False
        self.running = True

    def isRunning(self) -> bool:
        return self.running

    def request_stop(self) -> None:
        self.running = False

    def wait(self, _ms: int) -> bool:
        self.running = False
        return True

    def deleteLater(self) -> None:
        self.delete_later_called = True


def test_worker_teardown_happens_via_finished_not_captured_slot(
    tmp_path, monkeypatch, qt_app
):
    """Regression: deleteLater on the QThread from the captured/failed slots
    races run()'s exit (QThread destroyed while running -> abort). Teardown
    must happen only from the finished handler.
    """
    import numpy as np

    dialog, voice = _enabled_dialog(tmp_path, monkeypatch)
    worker = _SpyWorker()
    dialog._worker = worker
    dialog._worker.finished.connect(dialog._on_worker_finished)

    # Simulate a take completing: captured arrives while the thread is still
    # winding down. The old code deleteLater'd here — the fix must not.
    dialog._on_captured(np.full(4800, 0.2, dtype=np.float32))
    assert worker.delete_later_called is False, "captured must not delete the worker"
    assert dialog._worker is worker, "worker stays owned until finished fires"

    # Thread fully exits -> finished fires -> safe detach + delete.
    worker.running = False
    worker.finished.emit()
    assert dialog._worker is None
    assert worker.delete_later_called is True


def test_close_waits_for_worker_before_destroying(tmp_path, monkeypatch, qt_app):
    """Regression: closing mid-take must join the capture thread, not destroy
    a live QThread (hard Qt abort / use-after-free).
    """
    dialog, voice = _enabled_dialog(tmp_path, monkeypatch)
    worker = _SpyWorker()
    dialog._worker = worker
    dialog._stop_recording()  # requests stop, as closeEvent does
    # closeEvent's bounded wait joins the thread and then detaches it.
    import PySide6.QtGui as qtgui

    evt = qtgui.QCloseEvent()
    dialog.closeEvent(evt)
    assert dialog._worker is None
    assert worker.delete_later_called is True


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


def test_main_window_preview_player_deferred_end_of_media(monkeypatch, tmp_path, qt_app):
    """Regression: the MainWindow preview player must reuse one persistent
    instance and defer its EndOfMedia stop out of the signal handler — the
    same use-after-free discipline as the Recording Studio audition player.
    """
    window, _ = _build_main_window(monkeypatch, tmp_path)
    player = window.player
    assert player is not None

    # A preview starts playback on the persistent player (no per-preview
    # create/delete churn).
    window._finish_preview(0, "/tmp/some_preview.wav")
    assert window.player is player
    assert player.played == 1
    assert player.stopped == 0

    # EndOfMedia arrives: the handler must defer the stop (zero-timer), never
    # stop/delete the player from inside the emission, and keep the same
    # persistent instance alive for the next preview.
    player.mediaStatusChanged.emit(player.mediaStatusChanged)
    assert window.player is player
    assert player.stopped == 0, "stop must be deferred, not run inside the handler"

    # Drive the deferred stop so the next preview starts from a stopped state.
    window._stop_preview_player()
    assert player.stopped == 1
    window._finish_preview(1, "/tmp/another_preview.wav")
    assert window.player is player, "one persistent player across previews"


def test_main_window_close_stops_preview_player(monkeypatch, tmp_path, qt_app):
    """Regression: closing the main window must stop the persistent preview
    player so the QtMultimedia backend isn't torn down mid-playback.
    """
    import PySide6.QtGui as qtgui

    window, _ = _build_main_window(monkeypatch, tmp_path)
    window._finish_preview(0, "/tmp/some_preview.wav")
    assert window.player.stopped == 0

    evt = qtgui.QCloseEvent()
    window.closeEvent(evt)
    assert window.player.stopped >= 1, "close must stop the preview player"


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
