"""Tests for prosody/stage directives such as (tone: sad) and [pause=500]."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import OraclePipeline, RenderSettings, SpeakerSettings
from the_oracle.smoke import _write_reference
from the_oracle.text_repair.directives import apply_directives, parse_directives


class _FixedClassifier:
    def classify(self, text: str):
        return type("EmotionResult", (), {"label": "neutral", "confidence": 0.9})()

    def controls_for_emotion(self, label: str) -> dict[str, float | int]:
        return {"cfg_weight": 0.5, "exaggeration": 0.5, "temperature": 0.8, "pause_ms": 180}


def test_parse_tone_directive() -> None:
    text, overrides = parse_directives("(tone: sad) I missed you.")
    assert text == "I missed you."
    assert overrides["emotion"] == "sadness"


def test_parse_emotion_alias_maps_to_canonical_label() -> None:
    text, overrides = parse_directives("(tone: angry) I can't believe this.")
    assert text == "I can't believe this."
    assert overrides["emotion"] == "anger"


def test_parse_whisper_and_pause_directives() -> None:
    text, overrides = parse_directives("(whisper) [pause=500] Don't tell anyone.")
    assert text == "Don't tell anyone."
    assert overrides["pause_ms"] == 500
    assert overrides["whisper"] is True


def test_parse_non_numeric_value_for_numeric_tag_is_skipped() -> None:
    text, overrides = parse_directives("[pause=slow] [exaggeration=fast] Let it go.")
    assert text == "Let it go."
    assert "pause_ms" not in overrides
    assert "exaggeration" not in overrides


def test_parse_stage_direction_maps_to_emotion() -> None:
    text, overrides = parse_directives("(laughs) That was funny.")
    assert text == "That was funny."
    assert overrides["emotion"] == "joy"


def test_parse_stage_direction_does_not_clobber_explicit_tone() -> None:
    text, overrides = parse_directives("(tone: joy) (sighs) We made it.")
    assert text == "We made it."
    assert overrides["emotion"] == "joy"


def test_parse_plain_text_is_unchanged() -> None:
    text, overrides = parse_directives("A plain line of dialogue.")
    assert text == "A plain line of dialogue."
    assert overrides == {}


def test_apply_directives_merges_overrides() -> None:
    settings = VoiceSettings(pause_ms=180, exaggeration=0.5)
    merged = apply_directives(settings, {"pause_ms": 900, "exaggeration": 0.9})
    assert merged.pause_ms == 900
    assert merged.exaggeration == 0.9
    assert merged.temperature == 0.8  # untouched


@pytest.mark.slow  # full repair pipeline downloads LanguageTool (hundreds of MB)
def test_prepare_plan_strips_and_applies_directives(tmp_path: Path) -> None:
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text("Speaker A: (tone: sad) [pause=900] We made it.\n", encoding="utf-8")
    ref = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    settings = RenderSettings(model_variant="standard", language="en")
    speakers = {
        "A": SpeakerSettings(reference_path=str(ref), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(ref), voice_settings=VoiceSettings()),
    }
    with patch("the_oracle.pipeline.GoEmotionsClassifier", _FixedClassifier):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "out", speakers, settings)

    utterance = plan.utterances[0]
    assert "(tone: sad)" not in utterance.repaired_text
    assert "[pause=900]" not in utterance.text_for_tts()
    assert utterance.emotion == "sadness"
    assert utterance.engine_settings.pause_ms == 900
