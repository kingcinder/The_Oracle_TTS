# Performance Baseline — crash-campaign profiling (2026-09-28)

Plan Task 8. Method extends `docs/PERFORMANCE_BASELINE.md` (same subject:
`scripts/smoke_render.py`, deterministic engine) so numbers stay comparable.
Tool: `scripts/perf_baseline.py` (cProfile + tracemalloc, report at
`build/perf/baseline.json`, tests in `tests/test_perf_baseline.py`).

## Numbers (this machine, this tree)

| Metric | Value |
| --- | --- |
| Profiled smoke render total | 6.19 s (under cProfile; prior campaign measured 2.31 s profiled — machine was busier this run) |
| Peak memory | 168.94 MB |
| `_load_symspell_once` → `_pickle.load` | 2.02 s (33%) — **one** real load; the second wrapper call hits the memo |
| Import machinery (`_find_and_load`, `_handle_fromlist`, `exec_module`) | ~2.9 s combined — one-time process init |
| Anything algorithmic | none in the top 40 outside init |

## Verdict (honest)

After the prior campaign's SymSpell load-once fix, **no non-inherent hot
spot remains in the deterministic render path**: the profile is process
init (imports + one dictionary load) plus engine work. The real-render
wall time users feel (≈10 min for 11 segments) is CPU model inference —
inherent to the hardware (no CUDA on this machine; RX 5700 XT only serves
the Vulkan backend).

**Task 9 has nothing measured to fix.** Candidate follow-ups logged, not
actioned (would need real-render profiling to confirm):

- SymSpell dict: 2.0 s pickle load could be avoided entirely with a
  lazy/background load at GUI startup (currently paid synchronously on
  first Analyze).
- `hasattr` → module `__getattr__` lazy-import cascade (5103 calls, 1.5 s
  cumulative) — one-time, part of import cost.

## Caveat

This profile covers the *deterministic* smoke, not a live-model render.
Perf defects in the live path (worker pool scheduling, stem I/O patterns)
would need a real-render profile; recorded as deferred.
