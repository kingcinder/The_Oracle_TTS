"""CPU (PyTorch) vs Vulkan parity for the voice-craft changes.

Every pacing/emotion decision lives in plan preparation and assembly - stages the
two inference backends share - so a rendered utterance's engine settings and
pause profile must be byte-identical whether the user renders on PyTorch (CPU)
or on the audio.cpp Vulkan backend. These tests lock that property in.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.slow  # exercises the full repair pipeline

from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import OraclePipeline, RenderSettings, SpeakerSettings
from the_oracle.smoke import _write_reference


class _MixedClassifier:
    """Deterministic label + control map so both plans see identical input."""

    def classify(self, text: str):
        import re

        label = "joy" if re.search(r"[!]", text) else "sadness" if re.search(r"\?", text) else "neutral"
        return type("EmotionResult", (), {"label": label, "confidence": 0.9})()

    def controls_for_emotion(self, label: str) -> dict[str, float | int]:
        return {
            "neutral": {"cfg_weight": 0.5, "exaggeration": 0.5, "temperature": 0.8, "pause_ms": 180},
            "joy": {"cfg_weight": 0.42, "exaggeration": 0.68, "temperature": 0.82, "pause_ms": 150},
            "sadness": {"cfg_weight": 0.38, "exaggeration": 0.45, "temperature": 0.76, "pause_ms": 260},
        }[label]


_DIALOGUE = """Speaker A: We made it at last!
Speaker B: Are you sure about that?
Speaker A: Yes, I am certain.
Speaker B: Then we celebrate.
"""


def _plans(tmp_path: Path) -> tuple[dict, dict]:
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(_DIALOGUE, encoding="utf-8")
    ref = _write_reference(tmp_path / "speaker_ref.wav", 220.0)
    settings: dict[str, SpeakerSettings] = {
        "A": SpeakerSettings(reference_path=str(ref), voice_settings=VoiceSettings(pause_ms=180, emotion_intensity=1.0)),
        "B": SpeakerSettings(reference_path=str(ref), voice_settings=VoiceSettings(pause_ms=180, emotion_intensity=1.0)),
    }
    cpu = RenderSettings(model_variant="standard", language="en", inference_backend="pytorch")
    vulkan = RenderSettings(model_variant="standard", language="en", inference_backend="vulkan")
    with patch("the_oracle.pipeline.GoEmotionsClassifier", _MixedClassifier):
        pipeline = OraclePipeline()
        cpu_plan = pipeline.prepare_plan(dialogue, tmp_path / "out_cpu", settings, cpu)
        vulkan_plan = pipeline.prepare_plan(dialogue, tmp_path / "out_vulkan", settings, vulkan)
    return {u.index: u for u in cpu_plan.utterances}, {u.index: u for u in vulkan_plan.utterances}


def _speaker_assertion(utterance) -> tuple[str, float]:
    settings = utterance.engine_settings
    # Temperature and CFG must be speaker-locked (identical across every line of
    # that speaker) so the voice character is consistent on BOTH backends.
    return utterance.speaker, settings.temperature


def test_punctuation_pauses_and_engine_settings_identical_across_backends(tmp_path: Path) -> None:
    cpu, vulkan = _plans(tmp_path)
    assert set(cpu) == set(vulkan)
    for index in cpu:
        assert cpu[index].pause_after_ms == vulkan[index].pause_after_ms, f"utterance {index} pause differs"
        assert cpu[index].engine_settings.to_dict() == vulkan[index].engine_settings.to_dict(), (
            f"utterance {index} engine settings differ between backends"
        )
    # Punctuation-aware pacing is active on both backends (note: the repair
    # pass may normalize an exclamation to a period - e.g. "last!" becomes
    # "last." - so the text-repaired line ends exactly how it is spoken). The
    # question line (1.25x on top of its saddened pause) must breathe longer
    # than the neutral period-closed lines, identically on both backends.
    pauses_cpu = [cpu[i].pause_after_ms for i in sorted(cpu)]
    pauses_vulkan = [vulkan[i].pause_after_ms for i in sorted(vulkan)]
    assert pauses_cpu == pauses_vulkan
    assert pauses_cpu[1] > pauses_cpu[0]  # "Are you sure about that?" breathes longest
    assert pauses_cpu[0] == pauses_cpu[2] == pauses_cpu[3]  # period-closed neutrals keep 180


def test_timbre_lock_keeps_temperature_per_speaker_on_both_backends(tmp_path: Path) -> None:
    cpu, vulkan = _plans(tmp_path)
    for plans in (cpu, vulkan):
        by_speaker: dict[str, set[float]] = {}
        for utterance in plans.values():
            by_speaker.setdefault(utterance.speaker, set()).add(utterance.engine_settings.temperature)
        for speaker, temperatures in by_speaker.items():
            assert len(temperatures) == 1, f"speaker {speaker} temperature drifted per line: {temperatures}"
    # And the same per-line temperature values on both backends.
    assert _speaker_assertion(cpu[0])[0] == _speaker_assertion(vulkan[0])[0]
    for index in cpu:
        assert cpu[index].engine_settings.temperature == vulkan[index].engine_settings.temperature
