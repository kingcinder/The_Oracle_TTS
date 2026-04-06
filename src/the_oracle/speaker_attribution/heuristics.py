"""Dual-speaker attribution strategies with deterministic sequence modeling."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z']+")
FIRST_PERSON_RE = re.compile(r"\b(i|i'm|im|me|my|mine)\b", re.IGNORECASE)
SECOND_PERSON_RE = re.compile(r"\b(you|your|yours|you're|youre)\b", re.IGNORECASE)
QUESTION_WORD_RE = re.compile(r"^(who|what|when|where|why|how|did|do|does|can|could|would|will|are|is)\b", re.IGNORECASE)
RESPONSE_CUE_RE = re.compile(r"^(yes|yeah|yep|no|nope|ok|okay|right|sure|fine|thanks|thank you|got it|understood|on it)\b", re.IGNORECASE)
SELF_IDENTIFY_TEMPLATES = (
    "i'm {name}",
    "im {name}",
    "this is {name}",
    "{name} here",
    "it is {name}",
)
COMMON_TITLE_WORDS = {"mr", "mrs", "ms", "dr", "sir"}
STOPWORD_LIKE_NAMES = {"speaker", "voice", "scene", "chapter", "title", "note"}
SIGNATURE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "can",
    "did",
    "do",
    "for",
    "go",
    "got",
    "have",
    "i",
    "if",
    "im",
    "i'm",
    "in",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "now",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "will",
    "with",
    "you",
    "your",
}
NAME_CAPTURE_PATTERN = r"[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,2}"
SELF_IDENTIFY_NAME_RE = re.compile(
    rf"\b(?:i am|i'm|im|this is|it is)\s+(?P<name>{NAME_CAPTURE_PATTERN})\b",
    re.IGNORECASE,
)
NAME_HERE_RE = re.compile(rf"\b(?P<name>{NAME_CAPTURE_PATTERN})\s+here\b", re.IGNORECASE)


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


def _stable_hash_bytes(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def _bucket_for(value: str, dimensions: int) -> tuple[int, float]:
    digest = _stable_hash_bytes(value)
    bucket = int.from_bytes(digest[:8], "little") % dimensions
    sign = -1.0 if (digest[8] & 1) else 1.0
    return bucket, sign


def _embed_text(text: str, dimensions: int = 96) -> np.ndarray:
    vector = np.zeros(dimensions + 8, dtype=np.float32)
    tokens = _tokenize(text)
    for token in tokens:
        bucket, sign = _bucket_for(token, dimensions)
        vector[bucket] += sign
    for left, right in zip(tokens, tokens[1:], strict=False):
        bucket, sign = _bucket_for(f"{left}>{right}", dimensions)
        vector[bucket] += 0.7 * sign

    token_count = max(1, len(tokens))
    question_flag = 1.0 if "?" in text else 0.0
    exclaim_flag = 1.0 if "!" in text else 0.0
    quote_flag = 1.0 if '"' in text or "'" in text else 0.0
    first_person = len(FIRST_PERSON_RE.findall(text)) / token_count
    second_person = len(SECOND_PERSON_RE.findall(text)) / token_count
    short_flag = 1.0 if token_count <= 6 else 0.0
    long_flag = min(1.0, token_count / 40.0)
    uppercase_ratio = sum(1 for char in text if char.isupper()) / max(1, sum(1 for char in text if char.isalpha()))
    vector[dimensions:] = np.asarray(
        [
            question_flag,
            exclaim_flag,
            quote_flag,
            first_person,
            second_person,
            short_flag,
            long_flag,
            uppercase_ratio,
        ],
        dtype=np.float32,
    )
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize_name(raw_name: str | None) -> str | None:
    if raw_name is None:
        return None
    candidate = re.sub(r"[\[\]():\-–—]+", " ", raw_name).strip().lower()
    candidate = re.sub(r"\s+", " ", candidate)
    if not candidate:
        return None
    if candidate.startswith("speaker "):
        candidate = candidate[len("speaker ") :].strip()
    return candidate or None


def _display_name(raw_name: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw_name.strip())
    return cleaned


def _explicit_name_map(explicit_speakers: list[str | None]) -> dict[str, str]:
    names: dict[str, str] = {}
    for raw_name in explicit_speakers:
        normalized = _normalize_name(raw_name)
        if normalized and normalized not in names:
            names[normalized] = _display_name(raw_name or normalized)
    return names


def _mentions_name(text: str, name: str) -> bool:
    normalized_name = _normalize_name(name)
    if not normalized_name:
        return False
    name_tokens = [token for token in normalized_name.split() if token not in COMMON_TITLE_WORDS]
    if not name_tokens:
        return False
    text_tokens = set(_tokenize(text))
    return all(token in text_tokens for token in name_tokens if token not in STOPWORD_LIKE_NAMES)


def _vocative_mentions_name(text: str, name: str) -> bool:
    normalized_name = _normalize_name(name)
    if not normalized_name:
        return False
    name_tokens = [token for token in normalized_name.split() if token not in COMMON_TITLE_WORDS and token not in STOPWORD_LIKE_NAMES]
    if not name_tokens:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(token) for token in name_tokens) + r"\b"
    return bool(
        re.search(rf"^\s*(?:thanks|thank you|hey|hi|hello)?\s*,?\s*{pattern}[,!:]", text, re.IGNORECASE)
        or re.search(rf"[,!:\-]\s*{pattern}[.!?]*\s*$", text, re.IGNORECASE)
    )


def _self_identifies(text: str, name: str) -> bool:
    normalized_name = _normalize_name(name)
    if not normalized_name:
        return False
    lowered = " ".join(_tokenize(text))
    return any(template.format(name=normalized_name) in lowered for template in SELF_IDENTIFY_TEMPLATES)


def _extract_self_identified_name(text: str) -> str | None:
    for pattern in (SELF_IDENTIFY_NAME_RE, NAME_HERE_RE):
        match = pattern.search(text)
        if not match:
            continue
        candidate = match.group("name").strip()
        normalized = _normalize_name(candidate)
        if normalized and normalized not in STOPWORD_LIKE_NAMES:
            return candidate
    return None


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokenize(text) if len(token) >= 3 and token not in SIGNATURE_STOPWORDS]


def _is_reply_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if RESPONSE_CUE_RE.search(stripped):
        return True
    return len(_tokenize(stripped)) <= 5 and not stripped.endswith("?")


def _is_continuation(prev_text: str, current_text: str) -> bool:
    previous = prev_text.strip()
    current = current_text.strip()
    if not previous or not current:
        return False
    if previous.endswith((",", ";", ":", "-", "—", "–")):
        return True
    if not previous.endswith((".", "!", "?", '"', "'", "”", "’")):
        return True
    return bool(current[:1] and current[:1].islower())


def _confidence_from_margin(margin: float, *, locked: bool) -> float:
    if locked:
        return 0.99
    bounded = max(0.0, min(1.0, abs(margin)))
    return round(min(0.94, max(0.51, 0.57 + (bounded * 0.3))), 4)


class DualSpeakerAttributor:
    """Infers A/B speaker labels using deterministic anchors and sequence smoothing."""

    def assign(
        self,
        texts: Iterable[str],
        explicit_speakers: Iterable[str | None] | None = None,
        anchors: AnchorAssignments | None = None,
    ) -> list[SpeakerDecision]:
        utterances = list(texts)
        explicit = list(explicit_speakers or [None] * len(utterances))
        if not utterances:
            return []
        if anchors and anchors.speaker_a_indices and anchors.speaker_b_indices:
            return self._assign_from_anchor_indices(utterances, anchors)

        explicit_map = _explicit_name_map(explicit)
        latent_anchors, latent_names = self._discover_latent_name_anchors(utterances)
        combined_names = dict(explicit_map)
        for normalized, display in latent_names.items():
            if normalized not in combined_names and len(combined_names) < 2:
                combined_names[normalized] = display

        if combined_names:
            explicit_result = self._assign_from_explicit_or_partial(utterances, explicit, combined_names, latent_anchors)
            if explicit_result is not None:
                return explicit_result

        if self._looks_like_chat_lines(utterances):
            return self._assign_by_alternation(utterances, reason="alternating_chat_lines", confidence=0.69)
        return self._assign_from_binary_clustering(utterances)

    def _assign_from_explicit_or_partial(
        self,
        utterances: list[str],
        explicit_speakers: list[str | None],
        explicit_map: dict[str, str],
        latent_anchors: list[tuple[int, str]] | None = None,
    ) -> list[SpeakerDecision] | None:
        ordered_names = list(explicit_map)
        if len(ordered_names) > 2:
            return None

        speaker_for_name = {ordered_names[0]: "A"}
        if len(ordered_names) == 2:
            speaker_for_name[ordered_names[1]] = "B"

        locked: dict[int, str] = {}
        known_names: dict[str, str] = {}
        for index, raw_name in enumerate(explicit_speakers):
            normalized = _normalize_name(raw_name)
            if not normalized or normalized not in speaker_for_name:
                continue
            assigned = speaker_for_name[normalized]
            locked[index] = assigned
            known_names[assigned] = explicit_map[normalized]

        for index, raw_name in latent_anchors or []:
            normalized = _normalize_name(raw_name)
            if not normalized or normalized not in speaker_for_name or index in locked:
                continue
            assigned = speaker_for_name[normalized]
            locked[index] = assigned
            known_names.setdefault(assigned, explicit_map[normalized])

        if not locked:
            return None

        vectors = [_embed_text(text) for text in utterances]
        centroid_a, centroid_b = self._build_centroids_from_locked(vectors, locked)
        return self._sequence_assign(
            utterances,
            vectors,
            centroid_a,
            centroid_b,
            locked=locked,
            known_names=known_names,
            unlocked_reason="partial_explicit_sequence",
        )

    def _discover_latent_name_anchors(self, utterances: list[str]) -> tuple[list[tuple[int, str]], dict[str, str]]:
        anchors: list[tuple[int, str]] = []
        names: dict[str, str] = {}
        for index, text in enumerate(utterances):
            candidate = _extract_self_identified_name(text)
            normalized = _normalize_name(candidate)
            if not candidate or not normalized:
                continue
            anchors.append((index, candidate))
            names.setdefault(normalized, candidate)
        if not anchors or len(names) > 2:
            return [], {}
        return anchors, names

    def _looks_like_chat_lines(self, utterances: list[str]) -> bool:
        if len(utterances) < 4:
            return False
        lengths = [len(item.split()) for item in utterances]
        average = sum(lengths) / len(lengths)
        return max(lengths) <= 42 and average <= 18

    def _assign_by_alternation(self, utterances: list[str], *, reason: str, confidence: float) -> list[SpeakerDecision]:
        return [
            SpeakerDecision("A" if index % 2 == 0 else "B", confidence, reason)
            for index, _ in enumerate(utterances)
        ]

    def _assign_from_anchor_indices(self, utterances: list[str], anchors: AnchorAssignments) -> list[SpeakerDecision]:
        vectors = [_embed_text(text) for text in utterances]
        locked = {index: "A" for index in anchors.speaker_a_indices} | {index: "B" for index in anchors.speaker_b_indices}
        centroid_a, centroid_b = self._build_centroids_from_locked(vectors, locked)
        return self._sequence_assign(
            utterances,
            vectors,
            centroid_a,
            centroid_b,
            locked=locked,
            known_names={},
            unlocked_reason="anchor_propagation",
        )

    def _assign_from_binary_clustering(self, utterances: list[str]) -> list[SpeakerDecision]:
        vectors = [_embed_text(text) for text in utterances]
        if len(vectors) == 1:
            return [SpeakerDecision("A", 1.0, "single_utterance")]

        labels, centroid_a, centroid_b = self._cluster_vectors(vectors)
        if labels is None:
            return self._assign_by_alternation(utterances, reason="low_confidence_alternation", confidence=0.54)

        locked = {index: "A" if label == labels[0] else "B" for index, label in enumerate(labels)}
        # Only keep the most confident seeds from the cluster initialization so
        # the sequence model can still correct ambiguous local assignments.
        seeded: dict[int, str] = {}
        for index, vector in enumerate(vectors):
            score_a = _cosine_similarity(vector, centroid_a)
            score_b = _cosine_similarity(vector, centroid_b)
            margin = abs(score_a - score_b)
            if margin >= 0.18:
                seeded[index] = locked[index]
        return self._sequence_assign(
            utterances,
            vectors,
            centroid_a,
            centroid_b,
            locked=seeded,
            known_names={},
            unlocked_reason="binary_sequence_model",
        )

    def _cluster_vectors(self, vectors: list[np.ndarray]) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        stacked = np.asarray(vectors, dtype=np.float32)
        distances = np.asarray(
            [[1.0 - _cosine_similarity(stacked[i], stacked[j]) for j in range(len(stacked))] for i in range(len(stacked))],
            dtype=np.float32,
        )
        start_a, start_b = np.unravel_index(np.argmax(distances), distances.shape)
        if start_a == start_b:
            return None, stacked[0], stacked[-1]

        centroid_a = stacked[start_a]
        centroid_b = stacked[start_b]
        labels = np.zeros(len(stacked), dtype=np.int32)
        for _ in range(8):
            for index, vector in enumerate(stacked):
                score_a = _cosine_similarity(vector, centroid_a)
                score_b = _cosine_similarity(vector, centroid_b)
                labels[index] = 0 if score_a >= score_b else 1
            if np.all(labels == 0) or np.all(labels == 1):
                return None, centroid_a, centroid_b
            centroid_a = stacked[labels == 0].mean(axis=0)
            centroid_b = stacked[labels == 1].mean(axis=0)
        return labels, centroid_a, centroid_b

    def _build_centroids_from_locked(
        self,
        vectors: list[np.ndarray],
        locked: dict[int, str],
    ) -> tuple[np.ndarray, np.ndarray]:
        a_vectors = [vectors[index] for index, speaker in locked.items() if speaker == "A" and 0 <= index < len(vectors)]
        b_vectors = [vectors[index] for index, speaker in locked.items() if speaker == "B" and 0 <= index < len(vectors)]

        if a_vectors and b_vectors:
            return np.mean(a_vectors, axis=0), np.mean(b_vectors, axis=0)

        if a_vectors and not b_vectors:
            centroid_a = np.mean(a_vectors, axis=0)
            remaining = [vector for index, vector in enumerate(vectors) if locked.get(index) != "A"]
            if remaining:
                distances = sorted(remaining, key=lambda vector: 1.0 - _cosine_similarity(vector, centroid_a), reverse=True)
                return centroid_a, np.mean(distances[: max(1, min(3, len(distances)))], axis=0)
            return centroid_a, -centroid_a

        if b_vectors and not a_vectors:
            centroid_b = np.mean(b_vectors, axis=0)
            remaining = [vector for index, vector in enumerate(vectors) if locked.get(index) != "B"]
            if remaining:
                distances = sorted(remaining, key=lambda vector: 1.0 - _cosine_similarity(vector, centroid_b), reverse=True)
                return np.mean(distances[: max(1, min(3, len(distances)))], axis=0), centroid_b
            return -centroid_b, centroid_b

        return self._cluster_vectors(vectors)[1:]

    def _local_scores(
        self,
        texts: list[str],
        vectors: list[np.ndarray],
        centroid_a: np.ndarray,
        centroid_b: np.ndarray,
        known_names: dict[str, str],
    ) -> list[tuple[float, float]]:
        scores: list[tuple[float, float]] = []
        for text, vector in zip(texts, vectors, strict=True):
            score_a = _cosine_similarity(vector, centroid_a)
            score_b = _cosine_similarity(vector, centroid_b)
            for speaker, name in known_names.items():
                same_index = 0 if speaker == "A" else 1
                if _vocative_mentions_name(text, name):
                    if same_index == 0:
                        score_a -= 0.34
                        score_b += 0.30
                    else:
                        score_b -= 0.34
                        score_a += 0.30
                elif _mentions_name(text, name):
                    if same_index == 0:
                        score_a -= 0.24
                        score_b += 0.24
                    else:
                        score_b -= 0.24
                        score_a += 0.24
                if _self_identifies(text, name):
                    if same_index == 0:
                        score_a += 0.42
                        score_b -= 0.24
                    else:
                        score_b += 0.42
                        score_a -= 0.24
            scores.append((score_a, score_b))
        return scores

    def _pairwise_same_evidence(
        self,
        text_a: str,
        text_b: str,
        vector_a: np.ndarray,
        vector_b: np.ndarray,
        index_a: int,
        index_b: int,
    ) -> float:
        evidence = 0.0
        if index_b - index_a == 1 and _is_continuation(text_a, text_b):
            evidence += 0.44
        similarity = _cosine_similarity(vector_a, vector_b)
        if similarity >= 0.8:
            evidence += min(0.24, (similarity - 0.8) * 0.9)
        if _extract_self_identified_name(text_a) and _extract_self_identified_name(text_a) == _extract_self_identified_name(text_b):
            evidence += 0.7
        return evidence

    def _pairwise_diff_evidence(
        self,
        text_a: str,
        text_b: str,
        index_a: int,
        index_b: int,
    ) -> float:
        evidence = 0.0
        if index_b - index_a == 1:
            if text_a.strip().endswith("?"):
                evidence += 0.2
            if _is_reply_like(text_b):
                evidence += 0.12
            if len(_tokenize(text_a)) <= 10 and len(_tokenize(text_b)) <= 10 and not _is_continuation(text_a, text_b):
                evidence += 0.06
        name_a = _extract_self_identified_name(text_a)
        name_b = _extract_self_identified_name(text_b)
        if name_a and name_b and _normalize_name(name_a) != _normalize_name(name_b):
            evidence += 0.8
        return evidence

    def _build_signature_weights(
        self,
        texts: list[str],
        labels: list[int],
        locked: dict[int, str],
    ) -> dict[str, dict[str, float]]:
        counts = {"A": {}, "B": {}}  # type: ignore[var-annotated]
        totals = {"A": 0, "B": 0}
        for index, (text, label) in enumerate(zip(texts, labels, strict=True)):
            speaker = "A" if label == 0 else "B"
            if index in locked and locked[index] != speaker:
                continue
            for token in _content_tokens(text):
                counts[speaker][token] = counts[speaker].get(token, 0) + 1
                totals[speaker] += 1

        weights: dict[str, dict[str, float]] = {"A": {}, "B": {}}
        vocabulary = set(counts["A"]) | set(counts["B"])
        if not vocabulary:
            return weights

        for token in vocabulary:
            a = counts["A"].get(token, 0)
            b = counts["B"].get(token, 0)
            log_ratio = np.log((a + 1.0) / max(1.0, totals["A"] + len(vocabulary))) - np.log(
                (b + 1.0) / max(1.0, totals["B"] + len(vocabulary))
            )
            if abs(float(log_ratio)) < 0.25:
                continue
            if log_ratio > 0:
                weights["A"][token] = float(log_ratio)
            else:
                weights["B"][token] = float(-log_ratio)
        return weights

    def _apply_signature_weights(
        self,
        texts: list[str],
        local_scores: list[tuple[float, float]],
        signature_weights: dict[str, dict[str, float]],
    ) -> list[tuple[float, float]]:
        if not signature_weights["A"] and not signature_weights["B"]:
            return local_scores
        adjusted: list[tuple[float, float]] = []
        for text, (score_a, score_b) in zip(texts, local_scores, strict=True):
            tokens = _content_tokens(text)
            if not tokens:
                adjusted.append((score_a, score_b))
                continue
            signal_a = sum(signature_weights["A"].get(token, 0.0) for token in tokens) / len(tokens)
            signal_b = sum(signature_weights["B"].get(token, 0.0) for token in tokens) / len(tokens)
            adjusted.append((score_a + (0.22 * signal_a) - (0.08 * signal_b), score_b + (0.22 * signal_b) - (0.08 * signal_a)))
        return adjusted

    def _graph_refine_labels(
        self,
        texts: list[str],
        vectors: list[np.ndarray],
        local_scores: list[tuple[float, float]],
        labels: list[int],
        locked: dict[int, str],
    ) -> list[int]:
        if len(texts) < 3:
            return labels

        speakers = ("A", "B")
        label_state = list(labels)
        same_matrix = np.zeros((len(texts), len(texts)), dtype=np.float32)
        diff_matrix = np.zeros((len(texts), len(texts)), dtype=np.float32)
        for left in range(len(texts)):
            for right in range(left + 1, len(texts)):
                same = self._pairwise_same_evidence(texts[left], texts[right], vectors[left], vectors[right], left, right)
                diff = self._pairwise_diff_evidence(texts[left], texts[right], left, right)
                same_matrix[left, right] = same_matrix[right, left] = same
                diff_matrix[left, right] = diff_matrix[right, left] = diff

        for _ in range(4):
            changed = False
            for index in range(len(texts)):
                if index in locked:
                    continue
                scores = [local_scores[index][0], local_scores[index][1]]
                for speaker_index, speaker in enumerate(speakers):
                    scores[speaker_index] += self._adjacent_locked_bias(index, speaker, locked)
                    if index > 0:
                        scores[speaker_index] += 0.45 * self._transition_score(
                            speakers[label_state[index - 1]],
                            speaker,
                            texts[index - 1],
                            texts[index],
                        )
                    if index + 1 < len(texts):
                        scores[speaker_index] += 0.45 * self._transition_score(
                            speaker,
                            speakers[label_state[index + 1]],
                            texts[index],
                            texts[index + 1],
                        )
                    for other_index, other_label in enumerate(label_state):
                        if other_index == index:
                            continue
                        same = float(same_matrix[index, other_index])
                        diff = float(diff_matrix[index, other_index])
                        if same == 0.0 and diff == 0.0:
                            continue
                        if other_label == speaker_index:
                            scores[speaker_index] += same - diff
                        else:
                            scores[speaker_index] += diff - same
                best = 0 if scores[0] >= scores[1] else 1
                if best != label_state[index]:
                    label_state[index] = best
                    changed = True
            if not changed:
                break
        return label_state

    def _decode_sequence(
        self,
        texts: list[str],
        local_scores: list[tuple[float, float]],
        locked: dict[int, str],
    ) -> list[int]:
        speakers = ("A", "B")
        dp = [[float("-inf"), float("-inf")] for _ in texts]
        backtrack = [[0, 0] for _ in texts]

        for speaker_index, speaker in enumerate(speakers):
            score = local_scores[0][speaker_index] + self._adjacent_locked_bias(0, speaker, locked)
            if 0 in locked and locked[0] != speaker:
                score = -1e9
            elif 0 in locked:
                score += 1e6
            if speaker == "A":
                score += 0.02
            dp[0][speaker_index] = score

        for index in range(1, len(texts)):
            for current_index, current_speaker in enumerate(speakers):
                local = local_scores[index][current_index] + self._adjacent_locked_bias(index, current_speaker, locked)
                if index in locked and locked[index] != current_speaker:
                    dp[index][current_index] = -1e9
                    continue
                if index in locked and locked[index] == current_speaker:
                    local += 1e6
                best_prev = 0
                best_score = float("-inf")
                for previous_index, previous_speaker in enumerate(speakers):
                    score = dp[index - 1][previous_index]
                    score += self._transition_score(previous_speaker, current_speaker, texts[index - 1], texts[index])
                    score += local
                    if score > best_score:
                        best_score = score
                        best_prev = previous_index
                dp[index][current_index] = best_score
                backtrack[index][current_index] = best_prev

        state = 0 if dp[-1][0] >= dp[-1][1] else 1
        labels = [state]
        for index in range(len(texts) - 1, 0, -1):
            state = backtrack[index][state]
            labels.append(state)
        labels.reverse()
        return labels

    def _recompute_centroids(
        self,
        vectors: list[np.ndarray],
        labels: list[int],
        centroid_a: np.ndarray,
        centroid_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        a_vectors = [vector for vector, label in zip(vectors, labels, strict=True) if label == 0]
        b_vectors = [vector for vector, label in zip(vectors, labels, strict=True) if label == 1]
        if not a_vectors or not b_vectors:
            return centroid_a, centroid_b

        new_a = _normalize_vector(np.mean(a_vectors, axis=0))
        new_b = _normalize_vector(np.mean(b_vectors, axis=0))
        blended_a = _normalize_vector((0.7 * new_a) + (0.3 * centroid_a))
        blended_b = _normalize_vector((0.7 * new_b) + (0.3 * centroid_b))
        return blended_a, blended_b

    def _transition_score(self, prev_speaker: str, next_speaker: str, prev_text: str, next_text: str) -> float:
        score = 0.14 if prev_speaker != next_speaker else -0.06
        if prev_text.strip().endswith("?"):
            score += 0.12 if prev_speaker != next_speaker else -0.14
        if _is_reply_like(next_text):
            score += 0.08 if prev_speaker != next_speaker else -0.10
        if _is_continuation(prev_text, next_text):
            score += 0.18 if prev_speaker == next_speaker else -0.18
        if QUESTION_WORD_RE.search(next_text.strip()):
            score += 0.05 if prev_speaker != next_speaker else -0.04
        return score

    def _adjacent_locked_bias(self, index: int, speaker: str, locked: dict[int, str]) -> float:
        if not locked:
            return 0.0
        bias = 0.0
        previous = [candidate for candidate in locked if candidate < index]
        following = [candidate for candidate in locked if candidate > index]
        if previous:
            nearest_previous = max(previous)
            if index - nearest_previous == 1:
                bias += 0.24 if locked[nearest_previous] == speaker else -0.08
        if following:
            nearest_following = min(following)
            if nearest_following - index == 1:
                bias += 0.24 if locked[nearest_following] == speaker else -0.08
        return bias

    def _sequence_assign(
        self,
        texts: list[str],
        vectors: list[np.ndarray],
        centroid_a: np.ndarray,
        centroid_b: np.ndarray,
        *,
        locked: dict[int, str],
        known_names: dict[str, str],
        unlocked_reason: str,
    ) -> list[SpeakerDecision]:
        speakers = ("A", "B")
        current_centroid_a = centroid_a
        current_centroid_b = centroid_b
        local_scores: list[tuple[float, float]] = []
        labels: list[int] = []
        signature_weights = {"A": {}, "B": {}}

        for _ in range(3):
            local_scores = self._local_scores(texts, vectors, current_centroid_a, current_centroid_b, known_names)
            local_scores = self._apply_signature_weights(texts, local_scores, signature_weights)
            labels = self._decode_sequence(texts, local_scores, locked)
            labels = self._graph_refine_labels(texts, vectors, local_scores, labels, locked)
            signature_weights = self._build_signature_weights(texts, labels, locked)
            current_centroid_a, current_centroid_b = self._recompute_centroids(
                vectors,
                labels,
                current_centroid_a,
                current_centroid_b,
            )

        decisions: list[SpeakerDecision] = []
        for index, label in enumerate(labels):
            speaker = speakers[label]
            alt_label = 1 - label
            margin = local_scores[index][label] - local_scores[index][alt_label]
            reason = "explicit_marker" if locked.get(index) == speaker else unlocked_reason
            decisions.append(
                SpeakerDecision(
                    speaker=speaker,
                    confidence=_confidence_from_margin(margin, locked=index in locked),
                    reason=reason,
                )
            )
        return decisions
