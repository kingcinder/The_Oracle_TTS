"""Compatibility wrapper exposing utterance-aware speaker assignment APIs."""

from __future__ import annotations

from dataclasses import dataclass

from the_oracle.models.project import Utterance
from the_oracle.speaker_attribution.heuristics import DualSpeakerAttributor


@dataclass(slots=True)
class SpeakerAttributionResult:
    utterances: list[Utterance]
    detected_names: dict[str, str]


class SpeakerAttributor:
    def __init__(self) -> None:
        self._impl = DualSpeakerAttributor()

    def assign(self, utterances: list[Utterance]):
        texts = [utterance.original_text for utterance in utterances]
        explicit = [utterance.explicit_speaker for utterance in utterances]
        decisions = self._impl.assign(texts, explicit_speakers=explicit)
        detected_names: dict[str, str] = {}
        for utterance, decision in zip(utterances, decisions, strict=True):
            utterance.speaker = decision.speaker
            utterance.speaker_confidence = decision.confidence
            utterance.speaker_source = decision.reason
            if utterance.explicit_speaker:
                detected_names[decision.speaker] = utterance.explicit_speaker
        return SpeakerAttributionResult(utterances=utterances, detected_names=detected_names)

    def attribute(self, utterances: list[Utterance]):
        return self.assign(utterances)


def assign_speakers(utterances: list[Utterance]):
    return SpeakerAttributor().assign(utterances)


attribute_speakers = assign_speakers
infer_speakers = assign_speakers
label_speakers = assign_speakers
