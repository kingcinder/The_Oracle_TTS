from __future__ import annotations

from dataclasses import dataclass

from the_oracle.emotion.goemotions import GoEmotionsClassifier
from the_oracle.models.project import Utterance, VoiceSettings


@dataclass(slots=True)
class EmotionPrediction:
    label: str
    score: float


class EmotionInferer:
    def __init__(
        self,
        model_name: str = "SamLowe/roberta-base-go_emotions",
        use_transformers: bool = True,
    ) -> None:
        self.classifier = GoEmotionsClassifier(model_name=model_name, use_transformers=use_transformers)

    def infer_batch(self, texts: list[str]) -> list[EmotionPrediction]:
        # Batch through the classifier: one transformers call for the whole
        # list instead of one forward pass per utterance. Falls back to the
        # per-item classify path if the classifier lacks the batched API.
        classify_batch = getattr(self.classifier, "classify_batch", None)
        if callable(classify_batch):
            results = classify_batch(list(texts))
            return [
                EmotionPrediction(label=result.label, score=result.confidence) for result in results
            ]
        predictions: list[EmotionPrediction] = []
        for text in texts:
            result = self.classifier.classify(text)
            predictions.append(EmotionPrediction(label=result.label, score=result.confidence))
        return predictions


def apply_emotion_settings(base: VoiceSettings, emotion: str) -> VoiceSettings:
    settings = VoiceSettings.from_mapping(base)
    for key, value in GoEmotionsClassifier().controls_for_emotion(emotion).items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings


def annotate_emotions(utterances: list[Utterance], inferer: EmotionInferer, speaker_defaults: dict[str, VoiceSettings]) -> None:
    predictions = inferer.infer_batch([utterance.text_for_tts() for utterance in utterances])
    for utterance, prediction in zip(utterances, predictions, strict=True):
        utterance.emotion = prediction.label
        utterance.emotion_score = prediction.score
        utterance.emotion_confidence = prediction.score
        base = speaker_defaults.get(utterance.speaker, VoiceSettings())
        utterance.engine_settings = apply_emotion_settings(base, prediction.label)
