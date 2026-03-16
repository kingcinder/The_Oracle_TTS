import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QComboBox, QApplication

from the_oracle.models.project import RenderPlan, Utterance, VoiceProfile, VoiceSettings

from tests.test_app_gui_profiles import _build_window

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _plan_with_utterance(paths: Path, emotion: str) -> RenderPlan:
    profile_a = VoiceProfile(name="Speaker A", speaker="A", reference_audio=[], engine_params=VoiceSettings())
    profile_b = VoiceProfile(name="Speaker B", speaker="B", reference_audio=[], engine_params=VoiceSettings())
    utterances = [
        Utterance(
            index=0,
            original_text="Hello",
            repaired_text="Hello",
            speaker="A",
            emotion=emotion,
        )
    ]
    return RenderPlan(
        title="emotion test",
        source_path="",
        output_dir=str(paths),
        engine="chatterbox",
        correction_mode="moderate",
        metadata={"model_variant": "standard"},
        utterances=utterances,
        voice_profiles={"A": profile_a, "B": profile_b},
    )


def test_emotion_combo_round_trip(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        plan = _plan_with_utterance(paths.output_dir, "joy")
        window.plan = plan
        window._populate_table(plan)

        combo = window.table.cellWidget(0, 4)
        assert isinstance(combo, QComboBox)
        assert combo.currentData() == "joy"

        combo.setCurrentText("anger")
        window._sync_plan_from_table()
        assert plan.utterances[0].emotion == "anger"
    finally:
        window.close()


def test_emotion_combo_handles_unsupported_value(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        plan = _plan_with_utterance(paths.output_dir, "alien")
        window.plan = plan
        window._populate_table(plan)

        combo = window.table.cellWidget(0, 4)
        assert isinstance(combo, QComboBox)
        assert combo.currentText() == "alien"

        combo.setCurrentText("neutral")
        window._sync_plan_from_table()
        assert plan.utterances[0].emotion == "neutral"
    finally:
        window.close()
