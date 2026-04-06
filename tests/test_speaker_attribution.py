from the_oracle.models.project import Utterance
from the_oracle.speaker_attribution.assign import SpeakerAttributor


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


def test_partial_explicit_markers_propagate_to_unlabeled_turns() -> None:
    utterances = [
        Utterance(index=0, original_text="We need to clean this transcript.", explicit_speaker="Alice"),
        Utterance(index=1, original_text="I already started the parser rewrite."),
        Utterance(index=2, original_text="Good. I'll take the attribution pass next.", explicit_speaker="Bob"),
        Utterance(index=3, original_text="I'll own the edge cases too."),
    ]

    result = SpeakerAttributor().attribute(utterances)

    assert [item.speaker for item in result.utterances] == ["A", "A", "B", "B"]
    assert result.utterances[1].speaker_source == "partial_explicit_sequence"
    assert result.utterances[3].speaker_source == "partial_explicit_sequence"


def test_direct_name_address_pushes_assignment_to_other_speaker() -> None:
    utterances = [
        Utterance(index=0, original_text="Alice: I finished the parser."),
        Utterance(index=1, original_text="Thanks, Alice. I'll validate the result."),
        Utterance(index=2, original_text="Good. Send it over, Bob."),
        Utterance(index=3, original_text="Doing it now."),
    ]

    result = SpeakerAttributor().attribute(utterances)

    assert [item.speaker for item in result.utterances] == ["A", "B", "A", "B"]


def test_question_answer_pairs_prefer_turn_switching() -> None:
    utterances = [
        Utterance(index=0, original_text="Did you finish the parser cleanup?"),
        Utterance(index=1, original_text="Yeah, the quote extraction is in place."),
        Utterance(index=2, original_text="Can you verify the sequence model too?"),
        Utterance(index=3, original_text="Already did. The smoke tests are green."),
    ]

    result = SpeakerAttributor().attribute(utterances)

    assert [item.speaker for item in result.utterances] == ["A", "B", "A", "B"]


def test_explicit_anchor_allows_same_speaker_continuation_before_reply() -> None:
    utterances = [
        Utterance(index=0, original_text="I split the quoted spans.", explicit_speaker="Alice"),
        Utterance(index=1, original_text="and I repaired the attribution bridge too."),
        Utterance(index=2, original_text="Good. I'll validate the pipeline now.", explicit_speaker="Bob"),
        Utterance(index=3, original_text="Send me the manifest when the render finishes."),
    ]

    result = SpeakerAttributor().attribute(utterances)

    assert [item.speaker for item in result.utterances] == ["A", "A", "B", "B"]


def test_self_identification_creates_latent_name_anchors() -> None:
    utterances = [
        Utterance(index=0, original_text="I'm Alice."),
        Utterance(index=1, original_text="Bob here."),
        Utterance(index=2, original_text="Can you send the render log, Alice?"),
        Utterance(index=3, original_text="Already sent it."),
    ]

    result = SpeakerAttributor().attribute(utterances)

    assert [item.speaker for item in result.utterances] == ["A", "B", "B", "A"]


def test_non_adjacent_consistency_prefers_same_speaker_style_cluster() -> None:
    utterances = [
        Utterance(index=0, original_text="I catalogued the quote spans and normalized the punctuation."),
        Utterance(index=1, original_text="Good. I will verify the render path."),
        Utterance(index=2, original_text="I catalogued the quote spans and normalized the punctuation again for the appendix."),
    ]

    result = SpeakerAttributor().attribute(utterances)

    assert result.utterances[0].speaker == result.utterances[2].speaker


def test_lexical_signature_refinement_keeps_sparse_thematic_speaker_grouping() -> None:
    utterances = [
        Utterance(index=0, original_text="The checksum ledger and manifest hashes are reconciled."),
        Utterance(index=1, original_text="Good. I'll review it."),
        Utterance(index=2, original_text="The checksum ledger also matched the archive hashes on replay."),
        Utterance(index=3, original_text="All right. Ship it."),
        Utterance(index=4, original_text="The manifest hashes stayed stable after the recovery pass."),
    ]

    result = SpeakerAttributor().attribute(utterances)

    assert [item.speaker for item in result.utterances] == ["A", "B", "A", "B", "A"]
