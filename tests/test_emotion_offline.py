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


def test_classify_batch_matches_per_item_results() -> None:
    """The batched path must agree with per-item classify, one label each,
    in the same order, with the same fallback semantics."""
    classifier = GoEmotionsClassifier(use_transformers=False)
    texts = ["I am furious about this", "What is that?", "The table is round.", "She was laughing loudly"]
    batched = classifier.classify_batch(texts)
    expected = [classifier.classify(text) for text in texts]
    assert [result.label for result in batched] == [result.label for result in expected]
    assert [result.confidence for result in batched] == [result.confidence for result in expected]


def test_classify_batch_empty_input() -> None:
    classifier = GoEmotionsClassifier(use_transformers=False)
    assert classifier.classify_batch([]) == []


def test_classify_batch_falls_back_all_when_model_call_fails() -> None:
    """A wholesale model failure (the whole call raises) falls back every
    item to the lexical path rather than dropping any utterance."""

    class _ExplodingPipeline:
        def __call__(self, texts, **kwargs):
            raise RuntimeError("gpu died")

    classifier = GoEmotionsClassifier(use_transformers=False)
    classifier._pipeline = _ExplodingPipeline()
    results = classifier.classify_batch(["I am scared", "boom", "I am happy"])
    assert [result.label for result in results] == ["fear", "neutral", "joy"]


def test_classify_batch_falls_back_on_malformed_item() -> None:
    """If one returned item is malformed (not a usable prediction dict), that
    item falls back to the lexical path while healthy items keep their model
    labels."""

    class _SporadicPipeline:
        def __call__(self, texts, **kwargs):
            results = []
            for text in texts:
                if "terrified" in text:
                    results.append([{}])  # malformed: no label/score
                else:
                    results.append([{"label": "joy", "score": 0.9}])
            return results

    classifier = GoEmotionsClassifier(use_transformers=False)
    classifier._pipeline = _SporadicPipeline()
    results = classifier.classify_batch(["I am scared", "I am terrified", "I am happy"])
    assert results[0].label == "joy"
    assert results[1].label == "fear"  # lexical fallback for the malformed item
    assert results[2].label == "joy"
