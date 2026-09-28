# Consumer changelog draft — consumer-market-readiness campaign (2026-09-28)

> Status: DRAFT for the next release's changelog. Merge the `Unreleased`
> block below into `CHANGELOG.md` when the release is cut. Every claim here
> is backed by the campaign records: the ticked units in
> `docs/superpowers/plans/2026-09-25-consumer-market-readiness.md` and their
> `JUNO_FIXES.log` entries. Internal names (units, commits, file paths) are
> deliberately absent from the customer-facing text.

## Unreleased

### Performance

- **Renders get to work sooner.** Each render used to re-read a 19 MB
  spell-checking dictionary once per pipeline — twice in a typical session —
  before it could begin. The dictionary now loads once for the whole
  session, cutting about **20 % of the profiled startup work** (2.31 s →
  1.84 s on the project's deterministic benchmark) and roughly **10 % off
  real-world startup time** (2.0 s → 1.8 s on the reference machine). The
  profiled method and full numbers live in `docs/PERFORMANCE_BASELINE.md`.

### Fixed

- **Fixed the crash that could close the app seconds after a render.** For
  weeks a rare segfault could take the whole window down about five seconds
  after clicking Render — sporadically, and only when the main window was
  busy at exactly the wrong moment, which is why it resisted every attempt
  to catch it. With crash reporting armed, a scripted session finally caught
  the culprit in the act: a background render thread racing the interface
  while loading a module for the first time. That race is eliminated, a
  regression test now guards the pattern, and the same fix removed two
  other first-import sites on background render paths.
- **The Health Check no longer freezes silently.** On machines whose model
  cache was incomplete, the doctor could sit without output for up to thirty
  minutes trying a network download it should never attempt — and its
  child process is now correctly offline, so this class of silent hang is
  fixed at the root.
- **Subtitle files with mixed encodings import cleanly.** A single cue
  saved in a different encoding than the rest of the file no longer turns
  the converted script into garbled text; each cue is recovered
  individually, and undecodable ones show a visible placeholder instead of
  corrupting the render.

### Changed

- **Under-the-hood reliability work across the interface.** The main
  window's rendering, preview, and recording surfaces were separated into
  independently-owned modules with test-enforced ownership records, so the
  reliability fixes above stay fixed — and background pipelines, workers,
  and dialogs now follow one audited teardown pattern.

### Trust & privacy (shipped earlier today in 1.3.1)

Recap of the campaign's foundation, already released in **1.3.1**: local,
private crash reporting you opt into (reports never leave your disk),
offline license activation with nothing gated behind it, rotating logs that
can no longer grow without bound, a doctor that warns before long first-run
downloads, and visible, deterministic self-healing when a take is retried.

---

## Notes for the editor (not for publication)

- **Tone rules observed:** present tense, benefit-first, no internal
  identifiers; every "Fixed" item corresponds to a real, reproduced defect
  with a test or a captured fingerprint behind it — nothing aspirational.
- **The speedup number** is from the U4.3 baseline
  (`docs/PERFORMANCE_BASELINE.md`): profiled total 2.309 s → 1.835 s
  (20.5 %) and wall-clock 2.038 s → 1.82 s (10.7 %) on
  `scripts/smoke_render.py`. The draft publishes both honestly instead of
  quoting the larger profiled percentage alone; it is a startup-phase win
  (dictionary memoization), not end-to-end render time.
- **Excluded as not-yet-merged:** the cache-stem resilience fix ("a missing
  cache stem is a cache miss") sits on `wip/crash-eradication` alongside the
  segfault fix; add it under *Fixed* when that branch merges.
- **Excluded as invisible to customers:** record-integrity nets, manifest
  consolidation, plan/journal hygiene, hardware-claim reconciliation
  (owner decision pending — if ratified, the store copy change belongs in
  release notes, not the changelog).
- The segfault fix ships from the `wip/crash-eradication` branch; cut the
  release only after it merges to `main`.
