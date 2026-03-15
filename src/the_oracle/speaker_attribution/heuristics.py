"""Dual-speaker attribution strategies with deterministic fallbacks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z']+")


@dataclass(slots=True)
class AnchorAssignments:
    speaker_a_indices: list[int]
    speaker_b_indices: list[int]


@dataclass(slots=True)
class SpeakerDecision:
    speaker: str
    confidence: float
    reason: str


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _embed_text(text: str, dimensions: int = 64) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in _tokenize(text):
        bucket = hash(token) % dimensions
        sign = -1.0 if (hash(token + "_sign") % 2) else 1.0
        vector[bucket] += sign
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class DualSpeakerAttributor:
    """Infers A/B speaker labels without ever emitting a third speaker."""

    def assign(
        self,
        texts: Iterable[str],
        explicit_speakers: Iterable[str | None] | None = None,
        anchors: AnchorAssignments | None = None,
    ) -> list[SpeakerDecision]:
        utterances = list(texts)
        explicit = list(explicit_speakers or [None] * len(utterances))

        if anchors and anchors.speaker_a_indices and anchors.speaker_b_indices:
            return self._assign_from_anchors(utterances, anchors)
        if any(explicit):
            explicit_result = self._assign_from_explicit(explicit)
            if explicit_result:
                return explicit_result
        if self._looks_like_chat_lines(utterances):
            return [SpeakerDecision("A" if index % 2 == 0 else "B", 0.68, "alternating_chat_lines") for index, _ in enumerate(utterances)]
        return self._assign_from_binary_clustering(utterances)

    def _assign_from_explicit(self, explicit_speakers: list[str | None]) -> list[SpeakerDecision] | None:
        seen: dict[str, str] = {}
        next_speaker = "A"
        decisions: list[SpeakerDecision] = []
        for raw_name in explicit_speakers:
            if raw_name is None:
                return None
            normalized = raw_name.strip().lower()
            if normalized not in seen:
                seen[normalized] = next_speaker
                next_speaker = "B" if next_speaker == "A" else "A"
            decisions.append(SpeakerDecision(seen[normalized], 0.99, "explicit_marker"))
        return decisions

    def _looks_like_chat_lines(self, utterances: list[str]) -> bool:
        if len(utterances) < 4:
            return False
        lengths = [len(item.split()) for item in utterances]
        return max(lengths) <= 40 and sum(lengths) / len(lengths) <= 18

    def _assign_from_anchors(self, utterances: list[str], anchors: AnchorAssignments) -> list[SpeakerDecision]:
        vectors = [_embed_text(text) for text in utterances]
        centroid_a = np.mean([vectors[index] for index in anchors.speaker_a_indices], axis=0)
        centroid_b = np.mean([vectors[index] for index in anchors.speaker_b_indices], axis=0)
        results: list[SpeakerDecision] = []
        for vector in vectors:
            score_a = _cosine_similarity(vector, centroid_a)
            score_b = _cosine_similarity(vector, centroid_b)
            speaker = "A" if score_a >= score_b else "B"
            confidence = min(0.98, 0.5 + abs(score_a - score_b))
            results.append(SpeakerDecision(speaker, confidence, "anchor_propagation"))
        return results

    def _assign_from_binary_clustering(self, utterances: list[str]) -> list[SpeakerDecision]:
        vectors = np.array([_embed_text(text) for text in utterances], dtype=np.float32)
        if len(vectors) <= 1:
            return [SpeakerDecision("A", 1.0, "single_utterance")]

        distances = np.array(
            [[1.0 - _cosine_similarity(vectors[i], vectors[j]) for j in range(len(vectors))] for i in range(len(vectors))],
            dtype=np.float32,
        )
        start_a, start_b = np.unravel_index(np.argmax(distances), distances.shape)
        centroid_a = vectors[start_a]
        centroid_b = vectors[start_b]
        labels = np.zeros(len(vectors), dtype=np.int32)

        for _ in range(6):
            for index, vector in enumerate(vectors):
                score_a = _cosine_similarity(vector, centroid_a)
                score_b = _cosine_similarity(vector, centroid_b)
                labels[index] = 0 if score_a >= score_b else 1
            if np.all(labels == 0) or np.all(labels == 1):
                return [SpeakerDecision("A" if index % 2 == 0 else "B", 0.52, "low_confidence_alternation") for index in range(len(utterances))]
            centroid_a = vectors[labels == 0].mean(axis=0)
            centroid_b = vectors[labels == 1].mean(axis=0)

        cluster_for_a = labels[0]
        results: list[SpeakerDecision] = []
        for index, vector in enumerate(vectors):
            score_a = _cosine_similarity(vector, centroid_a)
            score_b = _cosine_similarity(vector, centroid_b)
            assigned_cluster = labels[index]
            speaker = "A" if assigned_cluster == cluster_for_a else "B"
            margin = abs(score_a - score_b)
            confidence = min(0.9, max(0.51, 0.55 + margin))
            if margin < 0.05:
                speaker = "A" if index % 2 == 0 else "B"
                confidence = 0.5
                reason = "cluster_tie_alternation"
            else:
                reason = "binary_clustering"
            results.append(SpeakerDecision(speaker, confidence, reason))
        return results
