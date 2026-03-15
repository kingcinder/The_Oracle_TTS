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
