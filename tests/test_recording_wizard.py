from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from the_oracle.app_gui import RecordingStudioDialog
from the_oracle.recording_wizard import RecordingStudioSetupWizard, STAGES


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _dialog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import the_oracle.audio.recorder as recorder

    class Device:
        index = 2
        name = "Studio USB Mic"
        max_input_channels = 1
        default_samplerate = 48000

    monkeypatch.setattr(recorder, "_sd", object())
    monkeypatch.setattr(recorder, "list_input_devices", lambda: [Device()])
    monkeypatch.setattr(recorder, "samplerates_for_device", lambda _index: [48000, 44100])
    repo = tmp_path / "repo"
    voice = repo / "Seashells"
    inputs = repo / "Input"
    voice.mkdir(parents=True)
    inputs.mkdir()
    script = inputs / "What is, reality.txt"
    script.write_text("Alphabetical voice reference text.", encoding="utf-8")
    dialog = RecordingStudioDialog(repo, voice, inputs)
    return dialog, voice, script


def test_stages_cover_setup_before_speaking_guidance():
    assert [stage.key for stage in STAGES] == [
        "microphone", "script", "folder", "naming", "technique", "finish"
    ]
    assert STAGES.index(next(stage for stage in STAGES if stage.key == "technique")) > STAGES.index(next(stage for stage in STAGES if stage.key == "naming"))
    assert "15-25 cm" in next(stage.explanation for stage in STAGES if stage.key == "technique")
    assert "plosives" in next(stage.explanation for stage in STAGES if stage.key == "technique")


def test_first_open_wizard_exposes_mic_folder_script_and_naming_controls(tmp_path, monkeypatch, qt_app):
    dialog, voice, script = _dialog(tmp_path, monkeypatch)
    wizard = RecordingStudioSetupWizard(dialog)
    try:
        assert wizard.mic_picker.currentText() == "Studio USB Mic"
        assert wizard.rate_picker.currentData() == 48000
        assert wizard.script_picker.findData(str(script)) >= 0
        assert wizard.folder_edit.text() == str(voice)
        assert wizard.name_edit.text() == "Seashell_No_1.wav"
        assert wizard.generic_warning.isChecked() is True
        assert wizard.remember_input.isChecked() is True
        assert wizard.remember_output.isChecked() is True
    finally:
        wizard.close()
        dialog.close()


def test_wizard_continue_applies_live_preferences_and_reaches_speaking_step(tmp_path, monkeypatch, qt_app):
    dialog, voice, script = _dialog(tmp_path, monkeypatch)
    applied: list[dict] = []
    def apply(payload: dict) -> None:
        applied.append(payload)
        dialog.apply_setup_preferences(payload)
    wizard = RecordingStudioSetupWizard(dialog, on_apply=apply)
    try:
        wizard.script_picker.setCurrentIndex(wizard.script_picker.findData(str(script)))
        wizard.folder_edit.setText(str(voice / "takes"))
        wizard.name_edit.setText("Cody_warm.wav")
        wizard.remember_input.setChecked(True)
        wizard.remember_output.setChecked(True)
        wizard._continue()
        assert applied
        assert dialog.outdir_combo.currentText() == str(voice / "takes")
        assert dialog.name_edit.text() == "Cody_warm.wav"
        while wizard._stage_index < 4:
            wizard._continue()
        assert wizard._stages[wizard._stage_index].key == "technique"
        assert "microphone" in wizard.explanation.toPlainText().lower()
        assert wizard._highlighted is dialog.prompt_area
    finally:
        wizard.close()
        dialog.close()


def test_no_microphone_is_explicitly_explained(tmp_path, monkeypatch, qt_app):
    import the_oracle.audio.recorder as recorder

    monkeypatch.setattr(recorder, "_sd", object())
    monkeypatch.setattr(recorder, "list_input_devices", lambda: [])
    repo = tmp_path / "repo"
    voice = repo / "Seashells"
    inputs = repo / "Input"
    voice.mkdir(parents=True)
    inputs.mkdir()
    dialog = RecordingStudioDialog(repo, voice, inputs)
    wizard = RecordingStudioSetupWizard(dialog)
    try:
        assert "No microphone" in wizard.summary.text()
        assert "recording will remain disabled" in wizard.summary.text()
        assert wizard.mic_picker.currentData() is None
    finally:
        wizard.close()
        dialog.close()
