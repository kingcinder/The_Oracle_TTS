from unittest.mock import patch

from the_oracle.models.project import Utterance, VoiceSettings
from the_oracle.pipeline import OraclePipeline, RenderSettings, SpeakerSettings
from the_oracle.smoke import _SmokeEmotionClassifier, _write_reference
from the_oracle.speaker_attribution.assign import SpeakerAttributor
from the_oracle.speaker_attribution.heuristics import (
    MAX_SPEAKERS,
    DualSpeakerAttributor,
    canonical_speaker_label,
)


def _utterances(lines: list[tuple[str | None, str]]) -> list[Utterance]:
    return [
        Utterance(index=index, original_text=text, explicit_speaker=speaker)
        for index, (speaker, text) in enumerate(lines)
    ]


def test_explicit_speaker_markers_map_to_a_and_b() -> None:
    utterances = [
        Utterance(index=0, original_text="Alice: Hello there"),
        Utterance(index=1, original_text="Bob: Hi back"),
    ]

    result = SpeakerAttributor().attribute(utterances)

    assert result.utterances[0].speaker == "A"
    assert result.utterances[1].speaker == "B"


def test_unlabeled_short_lines_fall_back_to_alternation() -> None:
    utterances = [
        Utterance(index=0, original_text="Hi."),
        Utterance(index=1, original_text="Hello."),
        Utterance(index=2, original_text="Need help?"),
        Utterance(index=3, original_text="Always."),
    ]

    result = SpeakerAttributor().attribute(utterances)

    assert [item.speaker for item in result.utterances] == ["A", "B", "A", "B"]


def test_group_conversation_gets_distinct_voice_per_character() -> None:
    """A group conversation (3+ named speakers) gives each character their
    own voice key instead of folding everyone into A/B."""
    utterances = _utterances([
        ("Alice", "I called the meeting."),
        ("Bob", "Thanks for the agenda."),
        ("Carol", "Let's start with the budget."),
        ("Alice", "Good idea, Carol."),
        ("Dan", "I have the numbers ready."),
        ("Bob", "Perfect, go ahead."),
        ("Carol", "And the headcount?"),
        ("Dan", "Flat for the quarter."),
    ])

    result = SpeakerAttributor().attribute(utterances)

    speakers = [item.speaker for item in result.utterances]
    assert speakers == ["A", "B", "C", "A", "D", "B", "C", "D"]
    # Every detected character maps back to a display name.
    assert result.detected_names["A"] == "Alice"
    assert result.detected_names["B"] == "Bob"
    assert result.detected_names["C"] == "Carol"
    assert result.detected_names["D"] == "Dan"


def test_group_conversation_respects_literal_a_and_b_labels() -> None:
    """Literal A/B labels keep their voices; extra names get new keys."""
    utterances = _utterances([
        ("A", "Status update."),
        ("Bob", "The build is green."),
        ("A", "Ship it."),
        ("Carol", "Wait, I found a bug."),
    ])

    result = SpeakerAttributor().attribute(utterances)

    assert [item.speaker for item in result.utterances] == ["A", "B", "A", "C"]


def test_unlabeled_lines_between_named_turns_inherit_speaker() -> None:
    """Unmarked lines after a labelled turn belong to that speaker."""
    utterances = _utterances([
        ("Alice", "First turn."),
        (None, "Second line of the same turn."),
        ("Bob", "Reply."),
        (None, "And more from Bob."),
    ])

    result = SpeakerAttributor().attribute(utterances)

    assert [item.speaker for item in result.utterances] == ["A", "A", "B", "B"]
    assert result.utterances[1].speaker_source == "continuation_after_turn"


def test_monologue_forces_single_narrator_voice() -> None:
    utterances = _utterances([
        ("Alice", "One."),
        ("Bob", "Two."),
        ("Carol", "Three."),
    ])

    result = SpeakerAttributor().attribute(utterances, monologue=True)

    assert [item.speaker for item in result.utterances] == ["A", "A", "A"]
    assert all(item.speaker_source == "monologue" for item in result.utterances)


def test_attribution_is_deterministic_across_runs() -> None:
    """Embeddings use zlib.crc32, not per-process-randomised hash(), so the
    same document attributes identically in every run."""
    lines = [
        ("Narrator", "It was a dark and stormy night."),
        ("Hero", "We must hurry."),
        ("Villain", "You'll never escape."),
        ("Hero", "We'll see about that."),
        ("Narrator", "And so the chase began."),
        ("Villain", "The storm is my ally."),
    ]
    first = SpeakerAttributor().attribute(_utterances(lines))
    second = SpeakerAttributor().attribute(_utterances(lines))

    assert [item.speaker for item in first.utterances] == [item.speaker for item in second.utterances]
    assert [item.speaker_confidence for item in first.utterances] == [
        item.speaker_confidence for item in second.utterances
    ]


def test_prose_colon_labels_are_not_speakers() -> None:
    """Common prose colon-labels (Note:, See:, Warning:) must never become
    phantom speakers."""
    assert canonical_speaker_label("Note") is None
    assert canonical_speaker_label("See") is None
    assert canonical_speaker_label("Warning") is None
    assert canonical_speaker_label("Time") is None
    assert canonical_speaker_label("Alice") == "alice"
    assert canonical_speaker_label("Speaker A") == "a"
    assert canonical_speaker_label("A") == "a"
    assert canonical_speaker_label("ALICE") == "alice"


def test_max_speakers_constant_is_twenty_four() -> None:
    assert MAX_SPEAKERS == 24


def test_over_max_speakers_folds_extras_by_adjacency() -> None:
    """With more distinct speakers than voices, extras fold into the voice
    they interact with most instead of failing."""
    names = [f"Character{index}" for index in range(MAX_SPEAKERS + 3)]
    lines: list[tuple[str | None, str]] = []
    # A long alternating conversation between the first two characters, with
    # every extra character chiming in once alongside them.
    lines.append((names[0], "Opening line."))
    for index in range(2, len(names)):
        lines.append((names[index], f"Extra {index} chimes in."))
        lines.append((names[0], f"Reply to extra {index}."))
    lines.append((names[1], "Closing line."))

    result = SpeakerAttributor().attribute(_utterances(lines))

    speakers = {item.speaker for item in result.utterances}
    assert len(speakers) <= MAX_SPEAKERS
    assert speakers <= set("ABCDEFGHIJKLMNOPQRSTUVWX")
    # Every extra still landed on exactly one of the 24 voices.
    assert all(item.speaker in speakers for item in result.utterances)


def test_alternation_keeps_speaker_across_continuations() -> None:
    """Chat-style alternation keeps the same voice when a line continues the
    previous turn (no terminal punctuation / lowercase continuation)."""
    attributor = DualSpeakerAttributor()
    lines = [
        "I can't believe it,",
        "not after everything we did.",
        "What are we going to do?",
        "We'll figure something out.",
    ]
    decisions = attributor.assign(lines)

    # Lines 0-1 are one speaker's continuation; line 2 is a new turn (ends
    # with '?'), and line 3 answers it back on the first speaker's voice.
    assert [decision.speaker for decision in decisions] == ["A", "A", "B", "A"]


def test_pipeline_auto_provisions_voices_for_detected_cast(tmp_path) -> None:
    """End-to-end: a group document with 4 named characters renders with a
    distinct voice per character even when only A/B references are configured.
    The plan records detected_names and auto-provisions C/D voice profiles
    that reuse the first configured reference."""
    source = tmp_path / "cast.txt"
    source.write_text(
        "Alice: I called the meeting.\n"
        "Bob: Thanks for the agenda.\n"
        "Carol: Let's start with the budget.\n"
        "Alice: Good idea, Carol.\n"
        "Dan: I have the numbers ready.\n"
        "Bob: Perfect, go ahead.\n",
        encoding="utf-8",
    )
    ref = _write_reference(tmp_path / "ref.wav", 220.0)
    settings = RenderSettings(model_variant="standard", language="en", correction_mode="off")
    speakers = {
        "A": SpeakerSettings(reference_path=str(ref), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(ref), voice_settings=VoiceSettings()),
    }
    with patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(source, tmp_path / "out", speakers, settings)

    assert [u.speaker for u in plan.utterances] == ["A", "B", "C", "A", "D", "B"]
    assert plan.detected_names == {"A": "Alice", "B": "Bob", "C": "Carol", "D": "Dan"}
    # C and D were auto-provisioned with the first configured reference.
    assert set(plan.voice_profiles) >= {"A", "B", "C", "D"}
    assert plan.voice_profiles["C"].primary_reference == ref
    assert plan.voice_profiles["D"].primary_reference == ref


def test_pipeline_monologue_uses_single_narrator_voice(tmp_path) -> None:
    """With monologue enabled, every utterance in the plan belongs to Speaker A
    (the narrator) regardless of explicit labels."""
    source = tmp_path / "monologue.txt"
    source.write_text(
        "Alice: One.\nBob: Two.\nCarol: Three.\n",
        encoding="utf-8",
    )
    ref = _write_reference(tmp_path / "ref.wav", 220.0)
    settings = RenderSettings(
        model_variant="standard",
        language="en",
        correction_mode="off",
        monologue=True,
    )
    speakers = {
        "A": SpeakerSettings(reference_path=str(ref), voice_settings=VoiceSettings()),
    }
    with patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(source, tmp_path / "out", speakers, settings)

    assert all(u.speaker == "A" for u in plan.utterances)
    assert all(u.speaker_source == "monologue" for u in plan.utterances)
