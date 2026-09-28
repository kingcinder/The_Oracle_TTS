"""The SymSpell dictionary loads once per process (U4.3's efficiency win).

cProfile over the deterministic smoke render (2026-09-28): the 19 MB
frequency-dictionary pickle is deserialized in ~0.39s and the smoke run
constructs OraclePipeline twice — 0.79s of a 2.31s run (34%) paying for the
same read-only object. The loaded dictionary is read-only after load
(``lookup()`` is the only post-load use; ``_verbosity`` is set at load), so
a process-level memo is behavior-identical while cutting every construction
after the first to zero.

**Mutation contract:** reverting SpellCorrector to per-instance loading must
fail test_two_correctors_load_the_dictionary_once; removing the failure
memoization must fail test_failed_load_is_also_memoized.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle.text_repair import spelling


@pytest.fixture(autouse=True)
def _fresh_memo():
    spelling.reset_symspell_cache_for_tests()
    yield
    spelling.reset_symspell_cache_for_tests()


def test_two_correctors_load_the_dictionary_once(monkeypatch) -> None:
    calls: list[int] = []

    def counting_loader():
        calls.append(1)
        return object()  # sentinel "loaded dictionary"

    monkeypatch.setattr(spelling, "_try_load_symspell", counting_loader)
    first = spelling.SpellCorrector()
    second = spelling.SpellCorrector()
    assert len(calls) == 1, f"dictionary loaded {len(calls)} times for two correctors"
    assert first._sym_spell is second._sym_spell


def test_failed_load_is_also_memoized(monkeypatch) -> None:
    """A machine without symspellpy (or with a broken cache) must not retry —
    and re-log a warning — on every pipeline construction."""
    calls: list[int] = []

    def failing_loader():
        calls.append(1)
        return None

    monkeypatch.setattr(spelling, "_try_load_symspell", failing_loader)
    first = spelling.SpellCorrector()
    second = spelling.SpellCorrector()
    assert len(calls) == 1, "a failed load was retried"
    assert first._sym_spell is None and second._sym_spell is None


def test_real_cache_yields_a_shared_working_corrector() -> None:
    """Integration sanity on the real machine state: two correctors share the
    object (whatever it is — None included on machines without the cache),
    and the corrector still works end to end."""
    first = spelling.SpellCorrector()
    second = spelling.SpellCorrector()
    assert first._sym_spell is second._sym_spell
    # Behavior unchanged: a correct, capitalized, or short token passes through.
    text = "Hello world this is a test"
    assert first.correct(text) == first.correct(text)
