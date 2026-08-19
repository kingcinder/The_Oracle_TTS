"""SymSpell-backed spelling correction with conservative token-level behavior."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
from importlib import resources
from pathlib import Path


TOKEN_RE = re.compile(r"\b[a-zA-Z']+\b")
_LOG = logging.getLogger(__name__)

# Cache directory: ~/.cache/the-oracle/symspell/
_CACHE_DIR = Path.home() / ".cache" / "the-oracle" / "symspell"


def _cache_path() -> Path:
    """Return the pickle cache path for the loaded SymSpell dictionary."""
    # Version-stamped so a symspellpy upgrade invalidates the cache.
    try:
        from symspellpy import __version__ as symspell_version
    except ImportError:
        symspell_version = "unknown"
    version_hash = hashlib.sha256(symspell_version.encode()).hexdigest()[:12]
    return _CACHE_DIR / f"freq_dict_{version_hash}.pkl"


def _load_from_cache() -> object | None:
    """Try to load a pre-pickled SymSpell instance. Returns None on any failure."""
    path = _cache_path()
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301 – trusted local cache
        _LOG.debug("Loaded SymSpell from cache: %s", path)
        return obj
    except Exception as exc:
        _LOG.debug("Cache load failed, will rebuild: %s", exc)
        return None


def _save_to_cache(sym_spell: object) -> None:
    """Persist a loaded SymSpell instance to disk for fast reload."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path()
        # Write to a temp file then atomically rename to avoid partial writes.
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(sym_spell, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
        _LOG.debug("Cached SymSpell dictionary: %s", path)
    except Exception as exc:
        _LOG.debug("Failed to cache SymSpell: %s", exc)


class SpellCorrector:
    def __init__(self) -> None:
        self._sym_spell = self._try_load_symspell()

    def _try_load_symspell(self):
        # 1. Try the pickle cache (fast path: ~0.1s vs 3.3s).
        cached = _load_from_cache()
        if cached is not None:
            return cached

        # 2. Fall back to loading from the symspellpy dictionary file.
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
            # Cache for next time.
            _save_to_cache(sym_spell)
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
