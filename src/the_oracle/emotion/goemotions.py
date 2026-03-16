"""GoEmotions-style classification with a transformer primary path and lexical fallback."""

from __future__ import annotations

from dataclasses import dataclass


LEXICON = {
    "anger": {"furious", "angry", "annoyed", "snapped", "yelled", "rage"},
    "fear": {"afraid", "scared", "terrified", "worried", "panic", "nervous"},
    "joy": {"happy", "joy", "laugh", "smiled", "delighted", "glad"},
    "sadness": {"sad", "upset", "hurt", "cry", "grief", "mourn"},
    "surprise": {"surprised", "suddenly", "unexpected", "astonished"},
}
SUPPORTED_EMOTIONS = [*LEXICON.keys(), "curiosity", "neutral"]


@dataclass(slots=True)
class EmotionResult:
    label: str
    confidence: float


class GoEmotionsClassifier:
    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions") -> None:
        self.model_name = model_name
        self._pipeline = self._try_load_pipeline(model_name)

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

        lowered = text.lower()
        for label, words in LEXICON.items():
            if any(word in lowered for word in words):
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
