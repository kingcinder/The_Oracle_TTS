from __future__ import annotations

import sys
import types

from the_oracle.text_repair.punctuation import PunctuationRestorer


def test_legacy_punctuation_model_maps_grouped_entities_for_transformers_five(monkeypatch) -> None:
    captured: dict[str, object] = {}
    punctuation_module = types.ModuleType("deepmultilingualpunctuation.punctuationmodel")

    def pipeline(task: str, model: str, **kwargs):
        captured.update(task=task, model=model, **kwargs)
        return object()

    punctuation_module.pipeline = pipeline

    class PunctuationModel:
        def __init__(self) -> None:
            self.pipe = punctuation_module.pipeline("ner", "fixture", grouped_entities=False)

    package = types.ModuleType("deepmultilingualpunctuation")
    package.PunctuationModel = PunctuationModel
    package.punctuationmodel = punctuation_module
    monkeypatch.setitem(sys.modules, "deepmultilingualpunctuation", package)
    monkeypatch.setitem(sys.modules, "deepmultilingualpunctuation.punctuationmodel", punctuation_module)

    restorer = PunctuationRestorer()

    assert restorer._model is not None
    assert captured == {"task": "ner", "model": "fixture", "aggregation_strategy": "none"}
    assert punctuation_module.pipeline is pipeline
