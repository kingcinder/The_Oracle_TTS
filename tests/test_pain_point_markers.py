"""Tests for the word~word pain-point markers in input scripts.

A '~' glued between two words marks a junction where the TTS engine hung up or
mispronounced. The marker is an annotation for the review table: it must be
stripped from synthesis text in every correction mode (including Verbatim),
while unattached '~' characters are read verbatim.
"""

from __future__ import annotations

from the_oracle.models.project import Utterance, strip_pain_point_markers


def test_engine_boundaries_strip_markers_before_synthesis(monkeypatch, tmp_path) -> None:
    """The annotation must be removed even when callers bypass the pipeline."""
    import numpy as np
    from the_oracle.tts_engines import chatterbox_engine

    class FakeConditionals:
        @classmethod
        def load(cls, *_args, **_kwargs):
            return cls()

        def to(self, _device):
            return self

    class FakeModel:
        def __init__(self):
            self.conds = None

        def generate(self, **kwargs):
            seen["text"] = kwargs["text"]
            return np.zeros((1, 16), dtype=np.float32)

    import threading
    seen: dict[str, str] = {}
    engine = object.__new__(chatterbox_engine.ChatterboxEngine)
    engine.variant = "standard"
    engine.device = "cpu"
    engine.seed = None
    engine._condition_cls = FakeConditionals
    engine._loaded_conditioning = {}
    engine._model = FakeModel()
    engine._synthesize_lock = threading.Lock()
    conditioning_path = tmp_path / "conditioning.pt"
    conditioning_path.write_bytes(b"x")
    conditioning = chatterbox_engine.ChatterboxConditioning("id", conditioning_path, "hash", "A", "standard")

    engine.synthesize("syncronized~lockstep", conditioning, chatterbox_engine.VoiceSettings())

    assert seen["text"] == "syncronized lockstep"


def test_strip_removes_marker_between_words() -> None:
    assert strip_pain_point_markers("syncronized~lockstep") == "syncronized lockstep"
    assert strip_pain_point_markers("atmospheric~soup from") == "atmospheric soup from"
    assert strip_pain_point_markers("just what~speakers your") == "just what speakers your"


def test_strip_leaves_attached_tilde_alone() -> None:
    # A '~' at a word edge is genuine content (e.g. an approximation "~42"),
    # not a marker between two words.
    assert strip_pain_point_markers("about ~42 units") == "about ~42 units"
    assert strip_pain_point_markers("42~ units") == "42~ units"


def test_strip_is_idempotent_and_safe_on_plain_text() -> None:
    assert strip_pain_point_markers("plain narration line") == "plain narration line"
    once = strip_pain_point_markers("a~b and ~x")
    assert strip_pain_point_markers(once) == once


def test_utterance_text_for_tts_strips_markers_in_verbatim_mode() -> None:
    utterance = Utterance(
        index=0,
        original_text="syncronized~lockstep with one another",
        repaired_text="syncronized~lockstep with one another",  # Verbatim passthrough
    )
    assert utterance.text_for_tts() == "syncronized lockstep with one another"
    # The review table keeps the author's annotation.
    assert utterance.repaired_text == "syncronized~lockstep with one another"


def test_prepare_plan_removes_markers_before_repair_and_keeps_review_annotation(tmp_path) -> None:
    from the_oracle.pipeline import OraclePipeline, RenderSettings, SpeakerSettings
    from the_oracle.smoke import _write_reference

    source = tmp_path / "marked.txt"
    source.write_text("What if syncronized~lockstep works?\n", encoding="utf-8")
    reference = _write_reference(tmp_path / "ref.wav", 220.0)
    pipeline = OraclePipeline(use_transformers=False, use_language_tool=False, use_punctuation_model=False)
    plan = pipeline.prepare_plan(
        source,
        tmp_path / "out",
        {"A": SpeakerSettings(reference_path=str(reference))},
        RenderSettings(correction_mode="verbatim"),
    )

    utterance = plan.utterances[0]
    assert "~" in utterance.original_text
    assert "~" not in utterance.repaired_text
    assert utterance.text_for_tts() == utterance.repaired_text


def test_utterance_text_for_tts_falls_back_to_original_and_strips() -> None:
    utterance = Utterance(
        index=0,
        original_text="atmospheric~soup",
        repaired_text="",
    )
    assert utterance.text_for_tts() == "atmospheric soup"
