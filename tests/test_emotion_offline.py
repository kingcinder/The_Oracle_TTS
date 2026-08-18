"""Tests for the fully-offline emotion classification mode."""

from __future__ import annotations

from the_oracle.emotion.goemotions import GoEmotionsClassifier
from the_oracle.emotion.infer import EmotionInferer


def test_offline_classifier_skips_transformers_pipeline() -> None:
    classifier = GoEmotionsClassifier(use_transformers=False)
    assert classifier._pipeline is None


def test_offline_classifier_hits_lexicon() -> None:
    classifier = GoEmotionsClassifier(use_transformers=False)
    assert classifier.classify("I am furious about this").label == "anger"
    assert classifier.classify("What a wonderful surprise").label == "surprise"


def test_offline_classifier_matches_inflected_forms() -> None:
    classifier = GoEmotionsClassifier(use_transformers=False)
    assert classifier.classify("She was laughing loudly").label == "joy"
    assert classifier.classify("He started raging at me").label == "anger"
    assert classifier.classify("I kept crying all night").label == "sadness"
    assert classifier.classify("I am worrying about the exam").label == "fear"


def test_offline_classifier_does_not_match_substrings() -> None:
    classifier = GoEmotionsClassifier(use_transformers=False)
    assert classifier.classify("I made a decision.").label == "neutral"
    assert classifier.classify("The madam arrived.").label == "neutral"


def test_offline_classifier_question_and_neutral() -> None:
    classifier = GoEmotionsClassifier(use_transformers=False)
    assert classifier.classify("What is that?").label == "curiosity"
    assert classifier.classify("The table is round.").label == "neutral"


def test_offline_inferer_batch() -> None:
    inferer = EmotionInferer(use_transformers=False)
    predictions = inferer.infer_batch(["I am so happy", "I am scared"])
    assert [prediction.label for prediction in predictions] == ["joy", "fear"]
