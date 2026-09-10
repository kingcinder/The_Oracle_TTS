"""Regression tests for the text ingest / voice attribution / emotion /
grammar review fixes.

Pure-logic tests only: heavy optional dependencies (transformers, torch,
language_tool_python, soundfile, huggingface_hub) are never required. The
real-engine smoke validation test stubs the two missing audio/model modules
just long enough to import the module under test, then cleans up so no
other test file observes the stubs.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from the_oracle.emotion.goemotions import EmotionResult, GoEmotionsClassifier
from the_oracle.emotion.infer import EmotionInferer, apply_emotion_settings
from the_oracle.models.project import Utterance, VoiceSettings
from the_oracle.speaker_attribution.assign import SpeakerAttributor
from the_oracle.speaker_attribution.heuristics import (
    MAX_SPEAKERS,
    AnchorAssignments,
    DualSpeakerAttributor,
)
from the_oracle.text_ingest import TextIngestor, ingest_text_file
from the_oracle.text_repair.directives import parse_directives
from the_oracle.text_repair.grammar import GrammarCorrector
from the_oracle.text_repair.normalize import normalize_text
from the_oracle.text_repair.punctuation import PunctuationRestorer
from the_oracle.voice_catalog import blend_catalog_path, blend_voice_choices


@contextlib.contextmanager
def _stubbed_heavy_imports():
    """Temporarily stub soundfile/huggingface_hub for a heavy module import.

    Restores sys.modules afterwards, including dropping modules that were
    imported while the stubs were active, so other test files never observe
    the stubbed dependencies.
    """
    soundfile_stub = MagicMock(name="soundfile")
    hub_stub = MagicMock(name="huggingface_hub")
    hub_stub.snapshot_download = MagicMock(name="snapshot_download")
    stubs = {"soundfile": soundfile_stub, "huggingface_hub": hub_stub}
    previous = {name: sys.modules[name] for name in stubs if name in sys.modules}
    before = set(sys.modules)
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name in stubs:
            if name in previous:
                sys.modules[name] = previous[name]
            else:
                sys.modules.pop(name, None)
        for name in set(sys.modules) - before:
            if name.split(".")[0] == "the_oracle":
                sys.modules.pop(name, None)


def _utterances(lines: list[tuple[str | None, str]]) -> list[Utterance]:
    return [
        Utterance(index=index, original_text=text, explicit_speaker=speaker)
        for index, (speaker, text) in enumerate(lines)
    ]


# ----------------------------------------------------------------------
# 1. CP1252 / Latin-1 fallback decoding
# ----------------------------------------------------------------------


def test_cp1252_encoded_text_does_not_crash_ingest(tmp_path: Path) -> None:
    # \x96 is an en-dash in CP1252 but invalid UTF-8: the old
    # read_text(encoding="utf-8") raised UnicodeDecodeError here.
    source = tmp_path / "legacy.txt"
    source.write_bytes("Caf\xe9 \x96 na\xefve dialogue here".encode("latin-1"))

    document = ingest_text_file(source)

    assert "Café – naïve dialogue here" in document.raw_text


def test_utf8_text_still_decodes_normally(tmp_path: Path) -> None:
    source = tmp_path / "plain.txt"
    source.write_text("Alice: Hello there.", encoding="utf-8")

    document = ingest_text_file(source)

    assert document.segments[0].explicit_speaker == "Alice"


# ----------------------------------------------------------------------
# 2. UTF-8 BOM handling
# ----------------------------------------------------------------------


def test_utf8_bom_does_not_break_first_line_speaker_detection(tmp_path: Path) -> None:
    source = tmp_path / "bom.txt"
    source.write_bytes(b"\xef\xbb\xbfAlice: Hello there.\nBob: Hi Alice.\n")

    document = ingest_text_file(source)

    assert not document.raw_text.startswith("\ufeff")
    assert document.segments[0].explicit_speaker == "Alice"
    assert document.segments[0].text == "Hello there."


# ----------------------------------------------------------------------
# 3. Out-of-range speaker-anchor indices
# ----------------------------------------------------------------------


def test_out_of_range_anchor_indices_are_dropped_not_fatal() -> None:
    attributor = DualSpeakerAttributor()
    decisions = attributor.assign(
        ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"],
        anchors=AnchorAssignments(speaker_a_indices=[0, 999], speaker_b_indices=[1, -5]),
    )

    assert len(decisions) == 3
    assert {decision.speaker for decision in decisions} <= {"A", "B"}


def test_fully_invalid_anchors_fall_back_to_label_heuristics() -> None:
    attributor = DualSpeakerAttributor()
    decisions = attributor.assign(
        ["Alice: first line here", "Bob: second line here", "Alice: third line here"],
        explicit_speakers=["Alice", "Bob", "Alice"],
        anchors=AnchorAssignments(speaker_a_indices=[99], speaker_b_indices=[100]),
    )

    assert len(decisions) == 3
    speakers = [decision.speaker for decision in decisions]
    assert speakers[0] == speakers[2] != speakers[1]


# ----------------------------------------------------------------------
# 4. Empty input -> zero decisions
# ----------------------------------------------------------------------


def test_empty_input_returns_zero_speaker_decisions() -> None:
    assert DualSpeakerAttributor().assign([]) == []
    assert DualSpeakerAttributor().assign([], explicit_speakers=[]) == []


def test_empty_input_does_not_break_strict_zip_wrapper() -> None:
    result = SpeakerAttributor().assign([])

    assert result.utterances == []
    assert result.detected_names == {}


# ----------------------------------------------------------------------
# 5. Malformed blend-catalog entries never crash the picker
# ----------------------------------------------------------------------


def test_malformed_blend_catalog_entries_are_skipped(tmp_path: Path) -> None:
    catalog = blend_catalog_path(tmp_path)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        json.dumps(
            {
                "version": 1,
                "voices": [
                    "not a dict",
                    None,
                    {"name": ""},
                    {"name": "missing-paths"},
                    {"name": "bad-weight", "path_a": "a.wav", "path_b": "b.wav", "weight": "abc"},
                    {"name": "none-weight", "path_a": "a.wav", "path_b": "b.wav", "weight": None},
                    {"name": 123, "path_a": "a.wav", "path_b": "b.wav"},
                    {"name": "bad-mode", "path_a": "a.wav", "path_b": "b.wav", "mode": 42},
                ],
            }
        ),
        encoding="utf-8",
    )

    # Must not raise (previously: KeyError on voice["path_a"], TypeError on
    # float(None), and uncaught errors from blend_references).
    assert blend_voice_choices(tmp_path) == []


def test_corrupt_blend_catalog_json_is_treated_as_empty(tmp_path: Path) -> None:
    catalog = blend_catalog_path(tmp_path)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text("{not valid json", encoding="utf-8")

    assert blend_voice_choices(tmp_path) == []


# ----------------------------------------------------------------------
# 6. Static emotion control mapping (no pipeline instantiation)
# ----------------------------------------------------------------------


def test_controls_for_emotion_needs_no_classifier_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(self, *args, **kwargs):
        raise AssertionError("classifier must not be instantiated for a static mapping")

    monkeypatch.setattr(GoEmotionsClassifier, "__init__", _boom)

    settings = apply_emotion_settings(VoiceSettings(), "joy")

    assert settings.exaggeration == 0.68
    assert GoEmotionsClassifier.controls_for_emotion("joy")["pause_ms"] == 150
    assert GoEmotionsClassifier.controls_for_emotion("unknown")["cfg_weight"] == 0.5


# ----------------------------------------------------------------------
# 7. Real-engine smoke output validation
# ----------------------------------------------------------------------


def _import_smoke_module():
    with _stubbed_heavy_imports():
        import importlib

        return importlib.import_module("the_oracle.real_engine_smoke")


def test_smoke_validation_rejects_silent_output(tmp_path: Path) -> None:
    smoke = _import_smoke_module()
    output = tmp_path / "out.flac"
    output.write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="silent"):
        smoke._validate_smoke_output(output, np.zeros(24000, dtype=np.float32))


def test_smoke_validation_rejects_empty_output(tmp_path: Path) -> None:
    smoke = _import_smoke_module()
    output = tmp_path / "out.flac"
    output.write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="no audio"):
        smoke._validate_smoke_output(output, np.zeros(0, dtype=np.float32))


def test_smoke_validation_rejects_missing_output_file(tmp_path: Path) -> None:
    smoke = _import_smoke_module()
    samples = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)

    with pytest.raises(RuntimeError, match="missing or empty"):
        smoke._validate_smoke_output(tmp_path / "does_not_exist.flac", samples)


def test_smoke_validation_accepts_real_signal(tmp_path: Path) -> None:
    smoke = _import_smoke_module()
    output = tmp_path / "out.flac"
    output.write_bytes(b"fake-flac-bytes")
    samples = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)

    smoke._validate_smoke_output(output, samples)  # must not raise


# ----------------------------------------------------------------------
# 8. Grammar: ellipses survive aggressive mode
# ----------------------------------------------------------------------


def test_aggressive_grammar_preserves_ellipses() -> None:
    corrector = GrammarCorrector(use_language_tool=False)

    assert corrector.correct("Wait... really.. yes....", aggressive=True) == "Wait... really. yes..."
    assert corrector.correct("Hmm... ok", aggressive=True) == "Hmm... ok"


def test_aggressive_grammar_still_collapses_stray_double_dots() -> None:
    corrector = GrammarCorrector(use_language_tool=False)

    assert corrector.correct("No.. way", aggressive=True) == "No. way"


# ----------------------------------------------------------------------
# 9. Punctuation: trailing comma
# ----------------------------------------------------------------------


def test_punctuation_trailing_comma_becomes_period_not_comma_period() -> None:
    restorer = PunctuationRestorer(use_model=False)

    assert restorer.restore("hello,") == "hello."
    assert restorer.restore("well;") == "well."
    assert restorer.restore("why me,") == "why me?"


def test_punctuation_legacy_single_line_behaviour_unchanged() -> None:
    restorer = PunctuationRestorer(use_model=False)

    assert restorer.restore("hello there\ngoodbye") == "hello there goodbye."


# ----------------------------------------------------------------------
# 10. Malformed directives are stripped, never spoken
# ----------------------------------------------------------------------


def test_malformed_directives_are_stripped_not_spoken() -> None:
    text, overrides = parse_directives("Hello (tone:) [pause=] (whisper loudly) world")

    assert text == "Hello world"
    assert overrides == {}
    assert "(tone:)" not in text
    assert "[pause=]" not in text


def test_unclosed_directives_are_stripped_not_spoken() -> None:
    text, _ = parse_directives("Hello [pause=500")
    assert text == "Hello"

    text, _ = parse_directives("Say (tone: sad")
    assert text == "Say"


def test_legitimate_parenthetical_prose_survives_directive_cleanup() -> None:
    text, overrides = parse_directives("He (whispered softly) to her.")

    assert text == "He (whispered softly) to her."
    assert overrides == {}


def test_valid_directives_still_parse() -> None:
    text, overrides = parse_directives("(tone: sad) I missed you. [pause=500]")

    assert text == "I missed you."
    assert overrides["emotion"] == "sadness"
    assert overrides["pause_ms"] == 500


# ----------------------------------------------------------------------
# 11. Emotion batch count mismatch falls back per item
# ----------------------------------------------------------------------


class _MisalignedBatchClassifier:
    """Batch API returns fewer results than inputs (the escaping bug)."""

    def classify(self, text: str) -> EmotionResult:
        return EmotionResult("neutral", 0.55)

    def classify_batch(self, texts: list[str]) -> list[EmotionResult]:
        return [EmotionResult("joy", 0.9)]  # wrong length on purpose


class _ExplodingBatchClassifier:
    def classify(self, text: str) -> EmotionResult:
        return EmotionResult("sadness", 0.61)

    def classify_batch(self, texts: list[str]) -> list[EmotionResult]:
        raise RuntimeError("batch backend exploded")


def _inferer_with(classifier) -> EmotionInferer:
    inferer = EmotionInferer.__new__(EmotionInferer)
    inferer.classifier = classifier
    return inferer


def test_infer_batch_falls_back_when_batch_count_mismatches() -> None:
    inferer = _inferer_with(_MisalignedBatchClassifier())

    predictions = inferer.infer_batch(["one", "two", "three"])

    assert len(predictions) == 3
    assert all(prediction.label == "neutral" for prediction in predictions)


def test_infer_batch_falls_back_when_batch_raises() -> None:
    inferer = _inferer_with(_ExplodingBatchClassifier())

    predictions = inferer.infer_batch(["one", "two"])

    assert len(predictions) == 2
    assert all(prediction.label == "sadness" for prediction in predictions)


def test_goemotions_classify_batch_misaligned_pipeline_falls_back() -> None:
    classifier = GoEmotionsClassifier(use_transformers=False)
    # A pipeline backend returning one prediction for two inputs.
    classifier._pipeline = lambda texts, **kwargs: [[{"label": "joy", "score": 0.9}]]

    results = classifier.classify_batch(["i am happy", "i am sad"])

    assert len(results) == 2


# ----------------------------------------------------------------------
# 12. Markdown blockquotes are spoken (user-approved behaviour change)
# ----------------------------------------------------------------------


def test_markdown_blockquote_content_is_ingested_as_text(tmp_path: Path) -> None:
    source = tmp_path / "scene.md"
    source.write_text(
        "# Scene\n\n> Alice: the quoted line\n\nBob: a normal line\n",
        encoding="utf-8",
    )

    document = ingest_text_file(source)
    texts = [(segment.text, segment.explicit_speaker) for segment in document.segments]

    assert ("the quoted line", "Alice") in texts
    assert ("a normal line", "Bob") in texts


def test_blockquote_markers_are_stripped_but_words_kept(tmp_path: Path) -> None:
    source = tmp_path / "letter.md"
    source.write_text("# Letter\n\n> Dear reader, this is quoted.\n", encoding="utf-8")

    document = ingest_text_file(source)

    assert any("Dear reader, this is quoted." in segment.text for segment in document.segments)
    assert all(">" not in segment.text for segment in document.segments)


# ----------------------------------------------------------------------
# 13. preserve_newlines on the public text APIs
# ----------------------------------------------------------------------


def test_normalize_text_preserve_newlines() -> None:
    assert normalize_text("first line\n\nsecond line", preserve_newlines=True) == "First line\n\nSecond line"
    # Legacy default collapses newlines and capitalises only the first
    # letter of the whole string, unchanged for existing callers.
    assert normalize_text("first line\n\nsecond line") == "First line second line"


def test_grammar_correct_preserve_newlines() -> None:
    corrector = GrammarCorrector(use_language_tool=False)

    assert (
        corrector.correct("hello world\n\ngoodbye world", preserve_newlines=True)
        == "Hello world\n\nGoodbye world"
    )
    assert corrector.correct("hello world\n\ngoodbye world") == "Hello world goodbye world"


def test_punctuation_restore_preserve_newlines() -> None:
    restorer = PunctuationRestorer(use_model=False)

    assert (
        restorer.restore("hello there\n\ngoodbye now", preserve_newlines=True)
        == "hello there.\n\ngoodbye now."
    )
    assert restorer.restore("hello there\n\ngoodbye now") == "hello there goodbye now."


# ----------------------------------------------------------------------
# 14. Adjacency-based folding for casts over MAX_SPEAKERS
# ----------------------------------------------------------------------


def test_extra_speaker_folds_into_most_adjacent_voice_not_ab() -> None:
    """25 distinct speakers: 'Zed' only converses with Character02 (voice C),
    so Zed must fold into C -- not into A/B as the old code did."""
    lines: list[tuple[str | None, str]] = [("Character00", "Opening line here.")]
    for index in range(1, MAX_SPEAKERS):
        lines.append((f"Character{index:02d}", f"Line from character {index}."))
    lines.extend(
        [
            ("Character02", "Hey Zed, are you there?"),
            ("Zed", "Hello there, friend."),
            ("Character02", "How are you doing?"),
            ("Zed", "Doing well, thanks."),
            ("Character02", "Good to hear it."),
        ]
    )
    texts = [text for _, text in lines]
    labels = [speaker for speaker, _ in lines]

    decisions = DualSpeakerAttributor().assign(texts, explicit_speakers=labels)

    assert len(decisions) == len(lines)
    assert {decision.speaker for decision in decisions} <= set("ABCDEFGHIJKLMNOPQRSTUVWX")
    zed_voices = {decisions[index].speaker for index, label in enumerate(labels) if label == "Zed"}
    character02_voices = {
        decisions[index].speaker for index, label in enumerate(labels) if label == "Character02"
    }
    assert character02_voices == {"C"}
    # Zed conversed only with Character02 -> folds into C, never A/B.
    assert zed_voices == {"C"}


def test_extra_adjacent_only_to_last_voice_folds_there() -> None:
    """Zed's only neighbouring turn belongs to Character23 (voice X), so
    adjacency folding must land Zed on X -- the old code could only ever
    pick A or B."""
    lines: list[tuple[str | None, str]] = [("Character00", "Opening line here.")]
    for index in range(1, MAX_SPEAKERS):
        lines.append((f"Character{index:02d}", f"Line from character {index}."))
    lines.append(("Zed", "A lone line at the end."))
    texts = [text for _, text in lines]
    labels = [speaker for speaker, _ in lines]

    decisions = DualSpeakerAttributor().assign(texts, explicit_speakers=labels)

    assert decisions[-1].speaker == "X"
