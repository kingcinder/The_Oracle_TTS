"""Regression and feature tests for audio assembly order, voice blending, and
monologue speaker profiles."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from the_oracle.audio.assemble import AudioSegment, assemble_dialogue
from the_oracle.audio.blend import blend_references
from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import OraclePipeline, RenderSettings, SpeakerSettings
from the_oracle.smoke import _SmokeEmotionClassifier, _write_reference
from the_oracle.text_repair.repairer import RepairResult


class _StubTextRepairPipeline:
    def repair(self, text: str, mode: str = "moderate") -> RepairResult:
        return RepairResult(text=text, corrections=[])


def _dominant_frequency(audio: np.ndarray, rate: int) -> int:
    windowed = audio * np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / rate)
    return int(round(freqs[np.argmax(spectrum)]))


def test_chunked_utterance_keeps_every_chunk_in_order(tmp_path: Path) -> None:
    """Two stems of the SAME utterance (chunked) must both be present, in order.

    Regression: the parallel stem loader keyed its dict on ``segment_index``
    (the source utterance index), so every chunk after the first silently
    overwrote the previous one and the render repeated one chunk's words while
    skipping the rest.
    """
    first = _write_reference(tmp_path / "chunk_a.wav", 440.0)
    second = _write_reference(tmp_path / "chunk_b.wav", 880.0)
    segments = [
        AudioSegment(
            path=str(first),
            sample_rate=24000,
            pause_after_ms=0,
            duration_seconds=0.25,
            segment_index=5,
        ),
        AudioSegment(
            path=str(second),
            sample_rate=24000,
            pause_after_ms=0,
            duration_seconds=0.25,
            segment_index=5,
        ),
    ]
    audio, rate = assemble_dialogue(segments, crossfade_ms=0, loudness_preset="off")
    midpoint = len(audio) // 2
    assert _dominant_frequency(audio[:midpoint], rate) == 440
    assert _dominant_frequency(audio[midpoint:], rate) == 880


def test_assemble_preserves_order_for_distinct_utterances(tmp_path: Path) -> None:
    first = _write_reference(tmp_path / "utt_0.wav", 300.0)
    second = _write_reference(tmp_path / "utt_1.wav", 600.0)
    segments = [
        AudioSegment(path=str(first), sample_rate=24000, pause_after_ms=0, duration_seconds=0.25, segment_index=0),
        AudioSegment(path=str(second), sample_rate=24000, pause_after_ms=0, duration_seconds=0.25, segment_index=1),
    ]
    audio, rate = assemble_dialogue(segments, crossfade_ms=0, loudness_preset="off")
    midpoint = len(audio) // 2
    assert _dominant_frequency(audio[:midpoint], rate) == 300
    assert _dominant_frequency(audio[midpoint:], rate) == 600


def test_blend_references_is_deterministic_and_honors_weight(tmp_path: Path) -> None:
    clip_a = _write_reference(tmp_path / "voice_a.wav", 220.0)
    clip_b = _write_reference(tmp_path / "voice_b.wav", 440.0)
    out_dir = tmp_path / "blends"

    pure_a = blend_references(clip_a, clip_b, weight_a=1.0, mode="mix", out_dir=out_dir)
    pure_b = blend_references(clip_a, clip_b, weight_a=0.0, mode="mix", out_dir=out_dir)
    assert pure_a.exists() and pure_b.exists()
    # Deterministic: same inputs -> same cached file.
    assert blend_references(clip_a, clip_b, weight_a=1.0, mode="mix", out_dir=out_dir) == pure_a

    import soundfile as sf

    audio_a, _ = sf.read(str(pure_a), dtype="float32")
    audio_b, _ = sf.read(str(pure_b), dtype="float32")
    assert np.corrcoef(audio_a, audio_b)[0, 1] < 0.5
    # Pure A correlates with A and not with B.
    source_a, _ = sf.read(str(clip_a), dtype="float32")
    source_b, _ = sf.read(str(clip_b), dtype="float32")
    n = min(len(audio_a), len(source_a))
    assert np.corrcoef(audio_a[:n], source_a[:n])[0, 1] > 0.95
    n = min(len(audio_b), len(source_b))
    assert np.corrcoef(audio_b[:n], source_b[:n])[0, 1] > 0.95


def test_blend_references_rejects_bad_mode_and_missing_file(tmp_path: Path) -> None:
    clip_a = _write_reference(tmp_path / "voice_a.wav", 220.0)
    clip_b = _write_reference(tmp_path / "voice_b.wav", 440.0)
    with pytest.raises(ValueError, match="blend mode"):
        blend_references(clip_a, clip_b, mode="chorus", out_dir=tmp_path / "blends")
    with pytest.raises(FileNotFoundError):
        blend_references(clip_a, tmp_path / "missing.wav", out_dir=tmp_path / "blends")


def test_monologue_plan_skips_unused_speaker_profiles(tmp_path: Path) -> None:
    """Monologue with only Speaker A configured must not carry a Speaker B
    profile with an empty reference (which used to balk the render with
    'Voice profile B has no reference audio configured')."""
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(
        "Speaker A: First line of the narration.\n"
        "Speaker A: Second line of the narration.\n",
        encoding="utf-8",
    )
    reference = _write_reference(tmp_path / "ref_a.wav", 220.0)
    speakers = {"A": SpeakerSettings(reference_path=str(reference), voice_settings=VoiceSettings())}
    settings = RenderSettings(monologue=True, loudness_preset="off")

    with (
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.pipeline.TextRepairPipeline", _StubTextRepairPipeline),
    ):
        plan = OraclePipeline().prepare_plan(dialogue, tmp_path / "output", speakers, settings)

    assert set(plan.voice_profiles) == {"A"}
    assert all(utterance.speaker == "A" for utterance in plan.utterances)


def test_dual_speaker_plan_keeps_both_profiles(tmp_path: Path) -> None:
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(
        "Speaker A: Hello there.\nSpeaker B: Hi yourself.\n",
        encoding="utf-8",
    )
    speakers = {
        "A": SpeakerSettings(reference_path=str(_write_reference(tmp_path / "a.wav", 220.0))),
        "B": SpeakerSettings(reference_path=str(_write_reference(tmp_path / "b.wav", 330.0))),
    }
    settings = RenderSettings(loudness_preset="off")

    with (
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.pipeline.TextRepairPipeline", _StubTextRepairPipeline),
    ):
        plan = OraclePipeline().prepare_plan(dialogue, tmp_path / "output", speakers, settings)

    assert set(plan.voice_profiles) == {"A", "B"}