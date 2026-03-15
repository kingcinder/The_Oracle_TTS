from pathlib import Path
from unittest.mock import patch

from dualvoice_studio.models.project import VoiceSettings
from dualvoice_studio.pipeline import DualVoicePipeline, RenderSettings, SpeakerSettings
from dualvoice_studio.smoke import _write_reference


class _JoyClassifier:
    def classify(self, text: str):
        return type("EmotionResult", (), {"label": "joy", "confidence": 1.0})()

    def controls_for_emotion(self, label: str) -> dict[str, float | int]:
        return {"cfg_weight": 0.3, "exaggeration": 0.9, "temperature": 0.9, "pause_ms": 140}


def test_emotion_intensity_scales_emotion_mapping(tmp_path: Path) -> None:
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text("Speaker A: We made it.\n", encoding="utf-8")
    ref = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    low = VoiceSettings(emotion_intensity=0.0, naturalness=0.0, pause_ms=180)
    high = VoiceSettings(emotion_intensity=1.5, naturalness=0.0, pause_ms=180)
    settings = RenderSettings(model_variant="standard", language="en")

    with patch("dualvoice_studio.pipeline.GoEmotionsClassifier", _JoyClassifier):
        pipeline = DualVoicePipeline()
        low_plan = pipeline.prepare_plan(
            dialogue,
            tmp_path / "out_low",
            {"A": SpeakerSettings(reference_path=str(ref), voice_settings=low), "B": SpeakerSettings(reference_path=str(ref), voice_settings=low)},
            settings,
        )
        high_plan = pipeline.prepare_plan(
            dialogue,
            tmp_path / "out_high",
            {"A": SpeakerSettings(reference_path=str(ref), voice_settings=high), "B": SpeakerSettings(reference_path=str(ref), voice_settings=high)},
            settings,
        )

    assert high_plan.utterances[0].engine_settings.exaggeration > low_plan.utterances[0].engine_settings.exaggeration
    assert high_plan.utterances[0].pause_after_ms < low_plan.utterances[0].pause_after_ms


def test_naturalness_heuristic_changes_sampling_controls(tmp_path: Path) -> None:
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text("Speaker A: We made it.\n", encoding="utf-8")
    ref = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    plain = VoiceSettings(emotion_intensity=1.0, naturalness=0.0)
    loose = VoiceSettings(emotion_intensity=1.0, naturalness=1.0)
    settings = RenderSettings(model_variant="standard", language="en")

    with patch("dualvoice_studio.pipeline.GoEmotionsClassifier", _JoyClassifier):
        pipeline = DualVoicePipeline()
        plain_plan = pipeline.prepare_plan(
            dialogue,
            tmp_path / "out_plain",
            {"A": SpeakerSettings(reference_path=str(ref), voice_settings=plain), "B": SpeakerSettings(reference_path=str(ref), voice_settings=plain)},
            settings,
        )
        loose_plan = pipeline.prepare_plan(
            dialogue,
            tmp_path / "out_loose",
            {"A": SpeakerSettings(reference_path=str(ref), voice_settings=loose), "B": SpeakerSettings(reference_path=str(ref), voice_settings=loose)},
            settings,
        )

    assert loose_plan.utterances[0].engine_settings.temperature > plain_plan.utterances[0].engine_settings.temperature
    assert loose_plan.utterances[0].engine_settings.repetition_penalty < plain_plan.utterances[0].engine_settings.repetition_penalty
