"""Compatibility wrapper around the lightweight dual-speaker heuristics."""

from __future__ import annotations

from dataclasses import dataclass

from dualvoice_studio.models.project import Utterance
from dualvoice_studio.speaker_attribution.heuristics import AnchorAssignments, DualSpeakerAttributor as HeuristicAttributor


@dataclass(slots=True)
class SpeakerAttributionResult:
    utterances: list[Utterance]
    detected_names: dict[str, str]


class DualSpeakerAttributor:
    def __init__(self) -> None:
        self._impl = HeuristicAttributor()

    def attribute(self, utterances: list[Utterance], anchors: dict[str, list[int]] | None = None) -> SpeakerAttributionResult:
        anchor_assignments = None
        if anchors and anchors.get("A") and anchors.get("B"):
            anchor_assignments = AnchorAssignments(speaker_a_indices=anchors["A"], speaker_b_indices=anchors["B"])
        decisions = self._impl.assign(
            [utterance.original_text for utterance in utterances],
            explicit_speakers=[utterance.explicit_speaker for utterance in utterances],
            anchors=anchor_assignments,
        )
        for utterance, decision in zip(utterances, decisions, strict=True):
            utterance.speaker = decision.speaker
            utterance.speaker_confidence = decision.confidence
            utterance.speaker_source = decision.reason
        return SpeakerAttributionResult(utterances=utterances, detected_names={})
