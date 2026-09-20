# Changelog

Notable user-facing changes to The Oracle. The version number itself lives in
one place (`src/the_oracle/__init__.py`); see `scripts/release.py --check`.

## [Unreleased]

### Changed

- (nothing yet)

## [1.2.0] — 2026-09-20

### Added

**Input linting and repair**
- `the-oracle check-input` lints a dialogue file's formatting without
  rendering; `--fix` corrects in place (timestamped backup kept), `--json`
  emits a machine-readable report for CI, and `--check-refs` verifies every
  suggested `--speaker-ref` path exists and is readable audio.
- `the-oracle fix-folder` batch-finds and fixes formatting across a folder
  (recursing into subfolders) with `--json`, `--dry-run`, and `--no-backup`
  modes.
- GUI: an input-formatting warning popup with a side-by-side preview diff —
  fix-rule labels color-coded per rule, changed rows red/green, a rule filter
  for selective applies, per-file checkboxes in the batch folder tree, a
  remember-my-choice auto-accept for trusted files, and preview dialogs that
  remember their size and position between sessions.
- Every fix keeps a timestamped backup; Settings → *Restore most recent
  input-file backup* undoes the latest one.

**Subtitles**
- SRT and WebVTT subtitle files are auto-detected and converted to dialogue
  scripts — in the CLI render path, the GUI file picker/Analyze flow, and the
  batch scanner. Conversion never overwrites: an existing converted script is
  reused. Cue timings set the per-turn pause durations, and the subtitle cast
  is suggested as ready-to-paste `--speakerA/B-ref` flags.

**Voices**
- `the-oracle voices` lists the default Seashells reference clips usable in
  `--speaker-ref` flags.
- Unknown or rejected speaker labels get the exact flag or rename needed,
  in the CLI report and the GUI popups.

**Offline installs**
- `scripts/build_offline_bundle.py` builds a fully offline install bundle
  (pinned wheels, seeded model cache, platform installers).
- The `.oracle_offline` marker is enforced at every model-loading entry
  point: no process can silently touch the network.
- `the-oracle setup-vulkan` refuses with a clear message on an offline
  install (the Vulkan GGUF model is not part of the bundle), and the
  LanguageTool grammar warm-up never attempts its download offline — the
  render falls back to local fixes instead.

**Packaging and releases**
- `scripts/release.py` verifies the single-source version invariant
  (`--check`), rewrites the README/STATE release banners (`--sync-banners`),
  and builds a versioned sdist + wheel with a `sha256sum -c`-compatible
  checksum manifest — fully offline.

### Changed

- The doctor gate is read-only and idempotent: running it twice produces
  byte-identical reports, and its readiness verdicts reflect capabilities
  verified in that run rather than artifacts left by previous ones.
- CI now runs the Windows installer end to end (temp profile, launchers,
  GUI), builds audio.cpp so the Vulkan suite never silently skips, fails if
  two consecutive doctor runs disagree, and runs the deterministic smoke
  render as a first-class signal on every push.
- `scripts/fresh_clone_acceptance.py` validates the full acceptance path
  (suite, doctor, wrapper dispatch) from a clean `git archive` tree.

### Fixed

- The GUI Cancel-button enum-vs-widget identity bug family (full sweep, no
  remaining instances).
- GUI startup crashes during prewarm and Recording Studio use (player and
  QThread lifetime bugs).
- Windows: platform-correct tests, byte-identical backups, and offline
  bundle launcher quoting for paths containing spaces.
- The deterministic smoke render no longer leaves a duplicate "(1)" versioned
  output behind its cache-reuse proof.
- GUI settings files that are missing or corrupt JSON now raise a clear
  error instead of failing obscurely.

## [1.1.1]

- CUDA inference through the PyTorch/Chatterbox path with NVIDIA suitability
  detection, CPU fallback, and selectable device index (GUI/CLI pickers,
  persisted selection, installer runtime choice, doctor diagnostics).
- Guided first-run onboarding (inference wizard, Recording Studio setup) with
  complete persistence.
- First-run workspace persistence: recent reference paths, trusted input
  files, and window workspace layout survive restarts.
- Pain-point `~` markers (`syncronized~lockstep`) are stripped from synthesis
  in every correction mode while staying visible in the review table.

## Earlier

- Tags `alpha-0.1` and `v0.2.0` predate the pyproject-packaged releases;
  see `git log` and `JUNO_FIXES.log` for that history.
