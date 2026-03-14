"""Automatic punctuation restoration with optional model support."""

from __future__ import annotations

import re


QUESTION_PREFIXES = ("who", "what", "when", "where", "why", "how", "did", "do", "does", "is", "are", "can", "could", "would", "will")


class PunctuationRestorer:
    def __init__(self) -> None:
        self._model = self._try_load_punctuator()

    def _try_load_punctuator(self):
        try:
            from deepmultilingualpunctuation import PunctuationModel  # type: ignore
        except Exception:
            return None
        try:
            return PunctuationModel()
        except Exception:
            return None

    def restore(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return cleaned
        if self._model is not None:
            try:
                restored = self._model.restore_punctuation(cleaned).strip()
                if restored:
                    cleaned = restored
            except Exception:
                pass
        if cleaned[-1] not in ".!?":
            prefix = cleaned.split()[0].lower()
            cleaned += "?" if prefix in QUESTION_PREFIXES else "."
        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
        cleaned = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", cleaned)
        return cleaned
