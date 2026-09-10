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

    def restore(self, text: str, *, preserve_newlines: bool = False) -> str:
        """Restore punctuation, adding terminal punctuation when missing.

        ``preserve_newlines`` keeps paragraph structure: each line is
        punctuated independently and newlines (including blank lines between
        paragraphs) survive instead of being collapsed to spaces. Defaults
        to False, preserving the historical single-line output for existing
        callers.
        """
        if preserve_newlines:
            lines = [self._restore_line(line) for line in text.split("\n")]
            # Mirror .strip() on the ends without collapsing interior
            # paragraph breaks.
            while lines and not lines[0]:
                lines.pop(0)
            while lines and not lines[-1]:
                lines.pop()
            return "\n".join(lines)
        return self._restore_line(text)

    def _restore_line(self, text: str) -> str:
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
            terminator = "?" if prefix in QUESTION_PREFIXES else "."
            if cleaned[-1] in ",;:":
                # A trailing comma/semicolon/colon is an unfinished thought,
                # not a finished sentence: replace it with the terminator
                # instead of appending (which produced "hello,.").
                cleaned = cleaned[:-1].rstrip() + terminator
            else:
                cleaned += terminator
        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
        cleaned = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", cleaned)
        return cleaned
