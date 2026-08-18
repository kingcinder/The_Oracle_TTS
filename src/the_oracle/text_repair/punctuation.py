"""Automatic punctuation restoration with optional model support."""

from __future__ import annotations

import logging
import re


_LOG = logging.getLogger(__name__)

QUESTION_PREFIXES = ("who", "what", "when", "where", "why", "how", "did", "do", "does", "is", "are", "can", "could", "would", "will")


class PunctuationRestorer:
    def __init__(self, *, use_model: bool = True) -> None:
        # The optional punctuation model pulls in transformers/PyTorch. The
        # desktop GUI uses the deterministic fallback to keep native ML code
        # out of the Qt process; the CLI retains the historical default.
        self._model = self._try_load_punctuator() if use_model else None

    def _try_load_punctuator(self):
        try:
            from deepmultilingualpunctuation import PunctuationModel  # type: ignore
        except Exception as exc:
            _LOG.warning("deepmultilingualpunctuation not available, punctuation restoration disabled: %s", exc)
            return None
        try:
            return PunctuationModel()
        except Exception as exc:
            _LOG.warning("PunctuationModel failed to initialise, punctuation restoration disabled: %s", exc)
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
            except Exception as exc:
                _LOG.debug("PunctuationModel failed on segment, using fallback: %s", exc)
        if cleaned[-1] not in ".!?":
            prefix = cleaned.split()[0].lower()
            cleaned += "?" if prefix in QUESTION_PREFIXES else "."
        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
        cleaned = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", cleaned)
        return cleaned
