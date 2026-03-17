from pathlib import Path
from unittest.mock import patch

import pytest

from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import OraclePipeline, RenderSettings, SpeakerSettings
from the_oracle.smoke import _DeterministicChatterboxEngine, _SmokeEmotionClassifier, _write_reference
from the_oracle.text_repair.repairer import RepairResult


class _StubTextRepairPipeline:
    def repair(self, text: str, mode: str = "moderate") -> RepairResult:
        return RepairResult(text=text, corrections=[])


def _build_speakers(tmp_path: Path) -> dict[str, SpeakerSettings]:
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    voice = VoiceSettings(variant="standard", language="en", pause_ms=180, crossfade_ms=0)
    return {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=voice),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=voice),
    }


def test_target_wpm_adjusts_pause_mapping(tmp_path: Path) -> None:
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text("Speaker A: Slow pace test.\nSpeaker B: Another sentence here.\n", encoding="utf-8")
    speakers = _build_speakers(tmp_path)
    settings = RenderSettings(model_variant="standard", loudness_preset="off", target_wpm=90.0, pause_between_turns_ms=180)

    with (
        patch("the_oracle.pipeline.ChatterboxEngine", _DeterministicChatterboxEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.pipeline.TextRepairPipeline", _StubTextRepairPipeline),
    ):
        plan = OraclePipeline().prepare_plan(dialogue, tmp_path / "output", speakers, settings)

    pauses = [utterance.pause_after_ms for utterance in plan.utterances]
    assert all(pause >= settings.pause_between_turns_ms for pause in pauses)
    assert float(plan.metadata.get("target_wpm") or 0.0) == pytest.approx(90.0)


def test_measured_wpm_matches_word_count_and_duration(tmp_path: Path) -> None:
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text("Speaker A: Measure this rate.\n", encoding="utf-8")
    speakers = _build_speakers(tmp_path)
    voice = VoiceSettings(variant="standard", language="en", pause_ms=0, crossfade_ms=0)
    speakers["A"] = SpeakerSettings(reference_path=speakers["A"].reference_path, voice_settings=voice)
    settings = RenderSettings(
        model_variant="standard",
        loudness_preset="off",
        crossfade_ms=0,
        pause_between_turns_ms=0,
        target_wpm=None,
        device_mode="cpu",
    )

    with (
        patch("the_oracle.pipeline.ChatterboxEngine", _DeterministicChatterboxEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.pipeline._should_use_worker_pool", return_value=False),
        patch("the_oracle.pipeline.TextRepairPipeline", _StubTextRepairPipeline),
    ):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speakers, settings)
        pipeline.render(plan, settings)

    duration = float(plan.metadata.get("measured_duration_seconds", "0") or "0")
    measured_wpm = float(plan.metadata.get("measured_wpm", "0") or "0")
    words = int(plan.metadata.get("word_count", "0") or "0")

    assert duration > 0
    assert measured_wpm == pytest.approx(words / (duration / 60.0))
    assert all(utterance.duration_seconds is not None for utterance in plan.utterances)
