from __future__ import annotations

from dataclasses import dataclass

from the_oracle.emotion.goemotions import GoEmotionsClassifier
from the_oracle.models.project import Utterance, VoiceSettings


@dataclass(slots=True)
class EmotionPrediction:
    label: str
    score: float


class EmotionInferer:
    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions") -> None:
        self.classifier = GoEmotionsClassifier(model_name=model_name)

    def infer_batch(self, texts: list[str]) -> list[EmotionPrediction]:
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
