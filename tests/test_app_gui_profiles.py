import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from the_oracle.app_paths import ensure_repo_default_paths
from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import RenderSettings


class _FakeAudioOutput:
    def __init__(self, *_args, **_kwargs) -> None:
        pass


class _FakeMediaPlayer:
    def __init__(self, *_args, **_kwargs) -> None:
        self.audio_output = None
        self.source = None

    def setAudioOutput(self, output) -> None:
        self.audio_output = output

    def setSource(self, source) -> None:
        self.source = source

    def play(self) -> None:
        return None


class _FakePipeline:
    def available_model_variants(self) -> list[str]:
        return ["standard", "multilingual", "turbo"]

    def supported_languages(self, model_variant: str = "standard") -> dict[str, str]:
        if model_variant == "multilingual":
            return {"en": "English", "es": "Spanish"}
        return {"en": "English"}


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _build_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import the_oracle.app_gui as app_gui

    paths = ensure_repo_default_paths(tmp_path / "repo")
    monkeypatch.setattr(app_gui, "QAudioOutput", _FakeAudioOutput)
    monkeypatch.setattr(app_gui, "QMediaPlayer", _FakeMediaPlayer)
    monkeypatch.setattr(app_gui, "OraclePipeline", _FakePipeline)
    monkeypatch.setattr(app_gui, "ensure_repo_default_paths", lambda _repo_root: paths)
    monkeypatch.setattr(app_gui, "default_voice_choices", lambda _repo_root: [])
    monkeypatch.setattr(app_gui, "load_recent_reference_paths", lambda limit=10: [])
    window = app_gui.MainWindow()
    return window, paths


def test_loading_profile_payload_replaces_current_settings(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    custom_output = tmp_path / "custom_output"
    custom_output.mkdir()

    try:
        window.variant_combo.setCurrentText("multilingual")
        window.correction_mode_combo.setCurrentText("aggressive")
        window.loudness_combo.setCurrentText("medium")
        window.crossfade_spin.setValue(75)
        window.outdir_path.setText(str(custom_output))
        window.output_name.setText("chapter_one")
        window.speaker_a.reference_path.setText("/tmp/speaker_a.wav")
        window.speaker_a.cfg_weight.setValue(0.9)
        window.speaker_b.reference_path.setText("/tmp/speaker_b.wav")
        window.speaker_b.pause_spin.setValue(420)
        payload = window._current_gui_settings_payload()

        window.variant_combo.setCurrentText("standard")
        window.correction_mode_combo.setCurrentText("conservative")
        window.loudness_combo.setCurrentText("off")
        window.crossfade_spin.setValue(20)
        window.outdir_path.setText(str(paths.output_dir))
        window.output_name.clear()
        window.speaker_a.reference_path.clear()
        window.speaker_a.cfg_weight.setValue(0.5)
        window.speaker_b.reference_path.clear()
        window.speaker_b.pause_spin.setValue(180)

        window._apply_gui_settings_payload(payload)

        assert window.variant_combo.currentText() == "multilingual"
        assert window.correction_mode_combo.currentText() == "aggressive"
        assert window.loudness_combo.currentText() == "medium"
        assert window.crossfade_spin.value() == 75
        assert window.outdir_path.text() == str(custom_output)
        assert window.output_name.text() == "chapter_one.flac"
        assert window.speaker_a.reference_path.text() == "/tmp/speaker_a.wav"
        assert window.speaker_a.cfg_weight.value() == pytest.approx(0.9)
        assert window.speaker_b.reference_path.text() == "/tmp/speaker_b.wav"
        assert window.speaker_b.pause_spin.value() == 420
    finally:
        window.close()


def test_new_project_keeps_current_profile_settings(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    custom_output = tmp_path / "custom_output"
    custom_output.mkdir()

    try:
        window.input_path.setText(str(tmp_path / "dialogue.txt"))
        window.outdir_path.setText(str(custom_output))
        window.output_name.setText("keep_me")
        window.loudness_combo.setCurrentText("medium")
        window.speaker_a.reference_path.setText("/tmp/speaker_a.wav")
        window.speaker_b.reference_path.setText("/tmp/speaker_b.wav")

        window.new_project()

        assert window.input_path.text() == ""
        assert window.table.rowCount() == 0
        assert window.outdir_path.text() == str(custom_output)
        assert window.output_name.text() == "keep_me"
        assert window.loudness_combo.currentText() == "medium"
        assert window.speaker_a.reference_path.text() == "/tmp/speaker_a.wav"
        assert window.speaker_b.reference_path.text() == "/tmp/speaker_b.wav"
        assert window.paths.output_dir == paths.output_dir
    finally:
        window.close()


def test_reset_to_defaults_restores_profile_baseline(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    defaults = VoiceSettings()
    render_defaults = RenderSettings()

    try:
        window.variant_combo.setCurrentText("multilingual")
        window.correction_mode_combo.setCurrentText("aggressive")
        window.loudness_combo.setCurrentText("medium")
        window.crossfade_spin.setValue(60)
        window.outdir_path.setText(str(tmp_path / "custom_output"))
        window.output_name.setText("custom_render")
        window.speaker_a.reference_path.setText("/tmp/speaker_a.wav")
        window.speaker_a.cfg_weight.setValue(1.0)
        window.speaker_b.reference_path.setText("/tmp/speaker_b.wav")
        window.speaker_b.pause_spin.setValue(360)

        window.reset_settings_to_defaults()

        assert window.variant_combo.currentText() == render_defaults.model_variant
        assert window.correction_mode_combo.currentText() == render_defaults.correction_mode
        assert window.loudness_combo.currentText() == render_defaults.loudness_preset
        assert window.crossfade_spin.value() == render_defaults.crossfade_ms
        assert window.outdir_path.text() == str(paths.output_dir)
        assert window.output_name.text() == ""
        assert window.speaker_a.reference_path.text() == ""
        assert window.speaker_b.reference_path.text() == ""
        assert window.speaker_a.cfg_weight.value() == pytest.approx(defaults.cfg_weight)
        assert window.speaker_b.pause_spin.value() == defaults.pause_ms
    finally:
        window.close()
