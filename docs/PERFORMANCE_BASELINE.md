# Performance Baseline — Representative Render

Unit U4.3 (consumer-market-readiness campaign, 2026-09-28). Purpose: a recorded,
reproducible baseline of one representative render so future units can compare
against numbers instead of impressions.

## Method

- Command profiled: `scripts/smoke_render.py` — the real Oracle pipeline driven by
  the deterministic engine (`_DeterministicChatterboxEngine`) and
  `_SmokeEmotionClassifier` from `src/the_oracle/smoke.py`. Two output formats
  plus a cache-reuse pass; two `OraclePipeline` constructions.
- Profiler: `cProfile` wrapping the script entry (no monkeypatching, no synthetic
  micro-benchmarks). Wall-clock checked alongside each profile run.
- Deterministic by construction: same stems, same "Cache reused on second pass:
  True" line, rc=0 before and after the change, so the comparison is apples to
  apples.
- Machine context: Quadro K600 workstation-class host, 32 GB RAM, project venv
  on Python 3.12.

## Baseline numbers (BEFORE)

| Metric | Value |
| --- | --- |
| Wall, cold cache | 2.038 s |
| Wall, warm cache | 2.133 s |
| cProfile total | 2.309 s |
| `_pickle.load` cumulative | 0.786 s (34% of profiled time) |

The pickle load was the top non-render cost: the 19 MB SymSpell frequency
dictionary cache (`~/.cache/the-oracle/symspell/freq_dict_b23a6a8439c0.pkl`,
19,302,189 bytes) was deserialized **twice** — once per `SpellCorrector`
instance — at ~0.393 s each. Two `OraclePipeline` constructions per render each
build a `TextRepairer` → `SpellCorrector`, and the dictionary is read-only after
load, so the second deserialization bought nothing.

## Top win landed: load the SymSpell dictionary once per process

`src/the_oracle/text_repair/spelling.py` now memoizes the dictionary load at
module level (`_load_symspell_once`), caching **both** outcomes — success and
failure — so a broken environment also fails fast instead of retrying per
instance. `_try_load_symspell` is a module-level function with its body
unchanged (cache pickle fast path → bundled fallback, `_verbosity=CLOSEST`,
cache save, warnings). `SpellCorrector` behavior is otherwise identical; the
dictionary is only ever read after load.

Proof: RED-first tests in `tests/test_spelling_load_once.py` (loader call count,
shared instance, failure memoization); mutation checks M1 (per-instance load)
and M2 (retry failed loads) each caught by the suite; byte-identical reverts.

## Numbers after (AFTER)

| Metric | Value |
| --- | --- |
| Wall, cold cache | 1.82 s |
| Wall, warm cache | 1.78 s |
| cProfile total | 1.83 s |
| Load path (`_try_load_symspell` incl. `_load_from_cache`) | ct 0.412 s, ncalls 1 (was 2 × ~0.393 s) |

**Net: ~0.48 s (~21%) off the representative render**, with the dictionary now
deserialized exactly once per process regardless of how many pipelines are
built. No behavior change: full suite green after the change (1452 passed,
16 warnings, 261 s); smoke render rc=0 with identical output ("Stem count: 4",
"Cache reused on second pass: True").

## Second candidate observed, not taken

`multiprocessing` `pool.join` in `OraclePipeline._generate`
(`src/the_oracle/pipeline.py:347`) accounted for 0.751 s cumulative. Bounded
follow-up candidate for a later unit — likely needs worker-pool reuse or a
shorter join strategy, and touches the render path rather than an
initialization cost, so it was out of U4.3's "no behavior change" scope.

## Reproducing

```bash
./.venv/bin/python -m cProfile -o /tmp/u43_profile.out scripts/smoke_render.py
./.venv/bin/python -m pstats /tmp/u43_profile.out
# interactive: sort cumulative, look for _pickle.load / _try_load_symspell / pool.join
```

Wall numbers are plain `time` wraps of the same command; expect small absolute
drift machine to machine, but the ratios (pickle load dominating cold init,
single load after the fix) should hold.
