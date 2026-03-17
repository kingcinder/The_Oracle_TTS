"""SymSpell-backed spelling correction with conservative token-level behavior."""

from __future__ import annotations

import logging
from importlib import resources
import re


TOKEN_RE = re.compile(r"\b[a-zA-Z']+\b")
_LOG = logging.getLogger(__name__)


class SpellCorrector:
    def __init__(self) -> None:
        self._sym_spell = self._try_load_symspell()

    def _try_load_symspell(self):
        try:
            from symspellpy import SymSpell, Verbosity
        except Exception as exc:
            _LOG.warning("symspellpy not available, spelling correction disabled: %s", exc)
            return None

        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        try:
            dictionary_path = resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
            sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)
            sym_spell._verbosity = Verbosity.CLOSEST
            return sym_spell
        except Exception as exc:
            _LOG.warning("SymSpell dictionary failed to load, spelling correction disabled: %s", exc)
            return None

    def correct(self, text: str, aggressive: bool = False) -> str:
        if not text.strip():
            return text
        if self._sym_spell is None:
            return text
        from symspellpy import Verbosity

        max_distance = 2 if aggressive else 1

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if len(token) < 4 or token[0].isupper():
                return token
            try:
                suggestions = self._sym_spell.lookup(token.lower(), Verbosity.CLOSEST, max_edit_distance=max_distance)
            except Exception:
                return token
            if not suggestions:
                return token
            suggestion = suggestions[0].term
            if suggestion == token.lower():
                return token
            return suggestion

        return TOKEN_RE.sub(replace, text)
