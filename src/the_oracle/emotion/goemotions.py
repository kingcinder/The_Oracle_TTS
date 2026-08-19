"""GoEmotions-style classification with a transformer primary path and lexical fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass


# Every entry is matched as a whole word (\b boundaries), so inflected
# forms must be listed explicitly to keep the lexical fallback effective:
# "laughing" needs "laugh" and "laughing" both present, "raging" needs
# "rage" and "raging", and so on.
LEXICON = {
    "anger": {"furious", "angry", "angrily", "annoyed", "annoying", "annoys", "snapped", "yell", "yelled", "yelling", "rage", "raging", "mad", "irritated", "irritating"},
    "fear": {"afraid", "scared", "scare", "scares", "scary", "terrified", "terrifying", "worry", "worried", "worries", "worrying", "panic", "panicked", "panicking", "nervous", "dread", "dreaded", "dreadful"},
    "joy": {"happy", "happily", "joy", "joyful", "laugh", "laughs", "laughed", "laughing", "smile", "smiles", "smiled", "smiling", "delighted", "delightful", "glad", "yay", "cheerful", "cheering", "excited", "exciting"},
    "sadness": {"sad", "sadly", "sadder", "sadness", "upset", "hurt", "hurting", "cry", "cries", "cried", "crying", "grief", "grieving", "mourn", "mourned", "mourning", "tears", "tearful", "heartbroken"},
    "surprise": {"surprised", "surprising", "surprises", "surprise", "suddenly", "unexpected", "astonished", "astonishing", "wow", "wowser", "amazed", "amazing"},
}
SUPPORTED_EMOTIONS = [*LEXICON.keys(), "curiosity", "neutral"]

# Precompiled whole-word patterns (one per emotion) so classify() does not
# recompile regexes on every utterance.
_LEXICON_PATTERNS: dict[str, re.Pattern[str]] = {
    label: re.compile(rf"\b(?:{'|'.join(re.escape(word) for word in words)})\b")
    for label, words in LEXICON.items()
}


@dataclass(slots=True)
class EmotionResult:
    label: str
    confidence: float


class GoEmotionsClassifier:
    def __init__(
        self,
        model_name: str = "SamLowe/roberta-base-go_emotions",
        use_transformers: bool = True,
    ) -> None:
        self.model_name = model_name
        self._pipeline = self._try_load_pipeline(model_name) if use_transformers else None

    def _try_load_pipeline(self, model_name: str):
        try:
            from transformers import pipeline
        except Exception:
            return None
        try:
            return pipeline("text-classification", model=model_name, top_k=1)
        except Exception:
            return None

    def classify(self, text: str) -> EmotionResult:
        if self._pipeline is not None:
            try:
                prediction = self._pipeline(text, truncation=True)[0][0]
                return EmotionResult(prediction["label"], float(prediction["score"]))
            except Exception:
                pass

        return self._classify_lexical(text)

    def classify_batch(self, texts: list[str]) -> list[EmotionResult]:
        """Classify many utterances with a single model call when possible.

        The transformers pipeline batches tokenization and inference across the
        whole list (one forward pass instead of one per utterance), which is
        the dominant win for long documents. Any item the model rejects (or a
        wholesale model failure) falls back to the deterministic lexical path,
        so results are never lost and the fallback is exercised per item.
        """
        if self._pipeline is None:
            return [self._classify_lexical(text) for text in texts]
        try:
            # top_k=1 makes the pipeline return one list per input text, so
            # the outer list aligns 1:1 with ``texts``.
            predictions = self._pipeline(list(texts), truncation=True, batch_size=64)
        except Exception:
            return [self._classify_lexical(text) for text in texts]
        results: list[EmotionResult] = []
        for text, item in zip(texts, predictions, strict=True):
            try:
                prediction = item[0]
                results.append(EmotionResult(prediction["label"], float(prediction["score"])))
            except Exception:
                results.append(self._classify_lexical(text))
        return results

    @staticmethod
    def _classify_lexical(text: str) -> EmotionResult:
        lowered = text.lower()
        # Whole-word matching so short lexicon tokens like "mad" don't fire on
        # substrings of unrelated words ("made", "madam"); inflected and
        # exclamation forms are listed explicitly in LEXICON.
        for label, pattern in _LEXICON_PATTERNS.items():
            if pattern.search(lowered):
                return EmotionResult(label, 0.62)
        if lowered.endswith("?"):
            return EmotionResult("curiosity", 0.58)
        return EmotionResult("neutral", 0.55)

    def controls_for_emotion(self, label: str) -> dict[str, float | int]:
        mapping = {
            "anger": {"cfg_weight": 0.35, "exaggeration": 0.78, "temperature": 0.86, "pause_ms": 130},
            "curiosity": {"cfg_weight": 0.45, "exaggeration": 0.55, "temperature": 0.8, "pause_ms": 180},
            "fear": {"cfg_weight": 0.4, "exaggeration": 0.62, "temperature": 0.82, "pause_ms": 220},
            "joy": {"cfg_weight": 0.42, "exaggeration": 0.68, "temperature": 0.82, "pause_ms": 150},
            "sadness": {"cfg_weight": 0.38, "exaggeration": 0.45, "temperature": 0.76, "pause_ms": 260},
            "surprise": {"cfg_weight": 0.34, "exaggeration": 0.8, "temperature": 0.88, "pause_ms": 160},
        }
        return mapping.get(label, {"cfg_weight": 0.5, "exaggeration": 0.5, "temperature": 0.8, "pause_ms": 180})
