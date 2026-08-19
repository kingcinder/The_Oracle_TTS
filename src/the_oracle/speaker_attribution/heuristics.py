"""Speaker attribution: name-aware mapping, multi-voice audiobook support,
monologue mode, and deterministic fallbacks.

The render pipeline supports up to ``MAX_SPEAKERS`` (24) distinct voices, so
attribution maps an arbitrary cast of named speakers onto that many keys:

* Literal ``A``/``B`` (or ``Speaker A``/``Speaker B``, ``1``/``2``) labels map
  directly to A/B.
* A two-person conversation maps by first appearance (``Alice`` then ``Bob``
  -> A then B), matching how scripts are usually written.
* A group conversation (three or more distinct speakers) gives each distinct
  speaker their own voice key (A, B, C, ... up to 24). If a document contains
  more than 24 distinct speakers, the extras fold into whichever existing
  voice they interact with most (turn adjacency) so the render always fits
  the available cast.
* ``monologue`` mode forces every line onto one narrator voice (A).

Unlabeled lines inside an otherwise-labeled document are inferred from their
nearest labeled turns (a run of unmarked lines after ``Alice:`` belongs to
Alice until the next label). Documents with no labels at all fall back to
alternation when they look like chat, prose narration when they look like
prose, and deterministic content clustering otherwise.

All embeddings are deterministic (``zlib.crc32``), unlike Python's
per-process-randomised builtin ``hash``, so attribution is reproducible across
runs -- important because attribution feeds cache keys and review tables.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z']+")
# Maximum number of distinct character voices a render can carry. Matches the
# GUI's cast panel and the manifest validator.
MAX_SPEAKERS: int = 24
# Voice keys are letters A..X (up to MAX_SPEAKERS).
VOICE_KEYS: tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(MAX_SPEAKERS))

# Common prose colon-labels that are NOT speaker names ("Note:", "See:",
# "Warning:"). A line matching one of these at the colon position is prose, not
# a dialogue turn, so the ingest layer must not turn it into a speaker marker.
NON_SPEAKER_LABELS: frozenset[str] = frozenset(
    {
        "note",
        "notes",
        "warning",
        "warn",
        "caution",
        "danger",
        "info",
        "information",
        "summary",
        "result",
        "results",
        "answer",
        "answers",
        "question",
        "questions",
        "example",
        "examples",
        "see",
        "seealso",
        "time",
        "date",
        "location",
        "place",
        "address",
        "status",
        "error",
        "errors",
        "todo",
        "fixme",
        "important",
        "attention",
        "tip",
        "tips",
        "hint",
        "hints",
        "usage",
        "install",
        "installation",
        "intro",
        "introduction",
        "overview",
        "conclusion",
        "conclusions",
        "reference",
        "references",
        "url",
        "link",
        "links",
        "http",
        "https",
        "www",
        "ftp",
        "email",
        "phone",
        "contact",
        "copyright",
        "license",
        "key",
        "keys",
        "value",
        "values",
        "name",
        "names",
        "id",
        "ids",
        "the",
        "this",
        "that",
        "these",
        "those",
        "there",
        "their",
        "they",
        "them",
        "we",
        "you",
        "your",
        "our",
        "his",
        "her",
        "its",
        "he",
        "she",
        "it",
        "speaker",
        "speakers",
        "chapter",
        "section",
        "part",
        "act",
        "scene",
        "title",
        "subtitle",
        "author",
        "last",
        "next",
        "previous",
        "first",
        "second",
        "third",
        "one",
        "two",
        "three",
        "left",
        "right",
        "top",
        "bottom",
        "front",
        "back",
        "inside",
        "outside",
        "above",
        "below",
        "start",
        "end",
        "begin",
        "stop",
        "now",
        "so",
        "but",
        "and",
        "or",
        "well",
        "ok",
        "okay",
        "yes",
        "no",
        "hello",
        "hi",
        "hey",
        "then",
        "else",
        "other",
        "others",
        "another",
        "more",
        "less",
        "many",
        "much",
        "some",
        "any",
        "all",
        "every",
        "each",
        "both",
        "neither",
        "either",
        "plus",
        "minus",
        "times",
        "vs",
        "versus",
    }
)


@dataclass(slots=True)
class AnchorAssignments:
    speaker_a_indices: list[int]
    speaker_b_indices: list[int]


@dataclass(slots=True)
class SpeakerDecision:
    speaker: str
    confidence: float
    reason: str


def canonical_speaker_label(raw: str | None) -> str | None:
    """Canonical identity of an explicit speaker label, or None if the string
    does not look like a speaker label at all.

    Normalises "Speaker A" / "speaker a" / "A" / "1" into the same canonical
    key (``a`` / ``b`` / ``1`` ...), folds whitespace, and rejects common
    prose colon-labels ("Note", "See", "Warning", ...) so those never become
    phantom speakers.
    """
    if not raw:
        return None
    compact = re.sub(r"\s+", " ", raw.strip())
    if not compact or len(compact) > 48:
        return None
    lowered = compact.lower()
    match = re.fullmatch(r"speaker[\s:]*([a-z0-9]+)", lowered)
    if match:
        return match.group(1)
    if re.fullmatch(r"[a-z0-9]{1,2}", lowered):
        return lowered
    if lowered in NON_SPEAKER_LABELS:
        return None
    if not re.search(r"[a-z]", lowered):
        return None
    return lowered


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _stable_hash(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))


def _embed_text(text: str, dimensions: int = 64) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in _tokenize(text):
        bucket = _stable_hash(token) % dimensions
        sign = -1.0 if (_stable_hash(token + "_sign") % 2) else 1.0
        vector[bucket] += sign
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class DualSpeakerAttributor:
    """Infers speaker labels for up to MAX_SPEAKERS distinct voices.

    Monologue mode assigns every line to the narrator voice (A). Otherwise
    named speakers map deterministically onto voice keys (A..X), unlabeled
    lines inherit from their neighbours, and unlabeled-only documents use the
    chat/prose/clustering fallbacks.
    """

    def assign(
        self,
        texts: Iterable[str],
        explicit_speakers: Iterable[str | None] | None = None,
        anchors: AnchorAssignments | None = None,
        *,
        monologue: bool = False,
    ) -> list[SpeakerDecision]:
        utterances = list(texts)
        explicit = list(explicit_speakers or [None] * len(utterances))

        if monologue:
            return [SpeakerDecision("A", 1.0, "monologue") for _ in utterances]
        if anchors and anchors.speaker_a_indices and anchors.speaker_b_indices:
            return self._assign_from_anchors(utterances, anchors)

        labeled = self._assign_from_labels(explicit)
        if labeled is not None:
            return labeled
        if self._looks_like_prose(utterances):
            return [SpeakerDecision("A", 0.6, "prose_narration") for _ in utterances]
        if self._looks_like_chat_lines(utterances):
            return self._assign_alternating(utterances)
        return self._assign_from_binary_clustering(utterances)

    # ------------------------------------------------------------------
    # Label-aware paths
    # ------------------------------------------------------------------

    def _assign_from_labels(self, explicit_speakers: list[str | None]) -> list[SpeakerDecision] | None:
        """Assign voices when at least one usable label is present.

        Mixed documents (some lines labelled, some not) are fully resolved
        here: unlabeled lines inherit from their nearest labeled neighbours.
        Returns None when no usable label exists anywhere in the document.
        """
        canonical = [canonical_speaker_label(label) for label in explicit_speakers]
        mapping = self._map_labels_to_voices(canonical)
        if mapping is None:
            return None

        decisions: list[SpeakerDecision] = []
        for index, label in enumerate(canonical):
            if label is not None and label in mapping:
                decisions.append(SpeakerDecision(mapping[label], 0.98, "explicit_marker"))
                continue
            speaker, confidence, reason = self._infer_unlabeled(canonical, index, mapping)
            decisions.append(SpeakerDecision(speaker, confidence, reason))
        return decisions

    @staticmethod
    def _map_labels_to_voices(canonical: list[str | None]) -> dict[str, str] | None:
        """Canonical-label -> voice mapping for up to MAX_SPEAKERS voices."""
        usable = [label for label in canonical if label is not None]
        if not usable:
            return None
        counts = Counter(usable)
        distinct = list(dict.fromkeys(usable))  # first-appearance order

        # Literal single-voice documents: everything is speaker A.
        if len(distinct) == 1:
            return {distinct[0]: "A"}

        # Honour literal A/B (or 1/2) labels directly when both are present.
        mapping: dict[str, str] = {}
        if "a" in counts and "b" in counts:
            mapping = {"a": "A", "b": "B"}
        elif "1" in counts and "2" in counts:
            mapping = {"1": "A", "2": "B"}

        # Plain two-person conversation: first appearance -> A, second -> B.
        if not mapping and len(distinct) == 2:
            return {distinct[0]: "A", distinct[1]: "B"}

        # Group conversation (three or more speakers): give every distinct
        # speaker their own voice key (A..X, up to MAX_SPEAKERS). Literal
        # A/B/1/2 labels keep their direct mapping; everyone else takes the
        # next free key in order of first appearance, so the cast is stable.
        # Speakers beyond the available keys stay unmapped here and are folded
        # into the nearest interacting voice below.
        if not mapping:
            for index, label in enumerate(distinct):
                if index >= len(VOICE_KEYS):
                    break
                mapping[label] = VOICE_KEYS[index]
        else:
            used = set(mapping.values())
            free = [key for key in VOICE_KEYS if key not in used]
            next_index = 0
            for label in distinct:
                if label in mapping:
                    continue
                if next_index >= len(free):
                    break
                mapping[label] = free[next_index]
                next_index += 1
        return DualSpeakerAttributor._fold_remaining(canonical, mapping, distinct)

    @staticmethod
    def _fold_remaining(
        canonical: list[str | None],
        mapping: dict[str, str],
        distinct: list[str],
    ) -> dict[str, str]:
        """Fold unassigned speakers into the voice they converse with most."""
        unassigned = [label for label in distinct if label not in mapping]
        if not unassigned:
            return mapping

        for label in unassigned:
            adjacency: Counter[str] = Counter()
            positions = [i for i, value in enumerate(canonical) if value == label]
            for position in positions:
                for neighbour in (position - 1, position + 1):
                    if 0 <= neighbour < len(canonical):
                        other = canonical[neighbour]
                        if other is not None and other in mapping:
                            adjacency[mapping[other]] += 1
            if adjacency["A"] != adjacency["B"]:
                mapping[label] = "A" if adjacency["A"] > adjacency["B"] else "B"
            else:
                # No adjacency signal: join the less-loaded voice to balance
                # the cast, falling back to A.
                load_a = sum(1 for value in mapping.values() if value == "A")
                load_b = sum(1 for value in mapping.values() if value == "B")
                mapping[label] = "B" if load_a > load_b else "A"
        return mapping

    @staticmethod
    def _infer_unlabeled(
        canonical: list[str | None],
        index: int,
        mapping: dict[str, str],
    ) -> tuple[str, float, str]:
        """Infer the voice of an unlabeled line from its labelled neighbours."""
        above: str | None = None
        above_dist = 0
        for offset in range(1, index + 1):
            candidate = canonical[index - offset]
            if candidate is not None and candidate in mapping:
                above = mapping[candidate]
                above_dist = offset
                break
        below: str | None = None
        below_dist = 0
        for offset in range(1, len(canonical) - index):
            candidate = canonical[index + offset]
            if candidate is not None and candidate in mapping:
                below = mapping[candidate]
                below_dist = offset
                break

        if above is not None and below is not None and above == below:
            return above, 0.78, "between_same_speaker"
        if above is not None:
            # A run of unmarked lines after a labelled turn belongs to that
            # speaker until the next label (transcript style).
            return above, 0.68, "continuation_after_turn"
        if below is not None:
            return below, 0.62, "next_labeled_turn"
        return "A", 0.5, "no_label_context"

    # ------------------------------------------------------------------
    # Unlabeled fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_prose(utterances: list[str]) -> bool:
        if len(utterances) < 2:
            return False
        lengths = [len(item.split()) for item in utterances]
        return sum(lengths) / len(lengths) > 25

    @staticmethod
    def _looks_like_chat_lines(utterances: list[str]) -> bool:
        if len(utterances) < 4:
            return False
        lengths = [len(item.split()) for item in utterances]
        return max(lengths) <= 40 and sum(lengths) / len(lengths) <= 18

    def _assign_alternating(self, utterances: list[str]) -> list[SpeakerDecision]:
        """Alternate A/B, keeping a speaker's voice across line continuations
        (previous line lacks terminal punctuation, or the line starts with a
        continuation like a lowercase word or an opening quote)."""
        decisions: list[SpeakerDecision] = []
        previous_speaker: str | None = None
        previous_text: str | None = None
        for text in utterances:
            if previous_speaker is None:
                speaker = "A"
            elif self._is_continuation(previous_text, text):
                speaker = previous_speaker
            else:
                speaker = "B" if previous_speaker == "A" else "A"
            decisions.append(SpeakerDecision(speaker, 0.68, "alternating_chat_lines"))
            previous_speaker = speaker
            previous_text = text
        return decisions

    @staticmethod
    def _is_continuation(previous: str | None, current: str) -> bool:
        if previous is None:
            return False
        if not re.search(r"[.!?]['\"]?\s*$", previous):
            return True  # previous line did not end a sentence
        if re.match(r"^[\"'(\[{]", current):
            return True  # opening quote/paren: continuation of the turn
        if current[:1].islower() and len(current) < 60:
            return True  # short lowercase continuation
        return False

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
                return self._assign_alternating(utterances)
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
