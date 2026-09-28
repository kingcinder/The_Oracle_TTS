# Changelog

Notable user-facing changes to The Oracle. The version number itself lives in
one place (`src/the_oracle/__init__.py`); see `scripts/release.py --check`.

## [Unreleased]

### Changed

- **The self-healing guarantee now covers the worker-pool path and the Live
  sidebar.** A multi-task seeded render that heals one hiccup reproduces
  byte-identically when the spawn pool runs the synthesis (the configured seed
  crosses the pool boundary through its initializer; retry notes ride home
  from whichever worker drained them), and the persistent Live column counts
  self-heals for the session — during the render and after it ends.

### Fixed

- Subtitle files that mix encodings (one UTF-8 cue exported into a
  mostly-CP1252 file — the classic two-editors concatenation) now recover
  **each cue on its own encoding** instead of turning the UTF-8 cue into
  deterministic mojibake. Clean whole-file UTF-8 and CP1252 files decode
  byte-identically to before; undecodable cues degrade to visible replacement
  marks instead of blocking the conversion.

## [1.3.1] — 2026-09-28

### Added

- **Release artifact checksums are tracked in the repo.** Every release build
  now stores a byte-identical copy of its `checksums-<version>.sha256` manifest
  in `release_checksums/`, so a published artifact is verifiable from a fresh
  clone rather than only from the gitignored build folder — and an existing
  record for a version is never rewritten silently: a rebuild producing
  different hashes is a refusal with instructions, not an overwrite.
- **The extraction safety nets read one validated ownership record.**
  `scripts/patch_surface_manifest.json` is the single record behind the
  MainWindow extraction campaign's nets — which moved names each `gui_*`
  module owns (an app_gui-level patch of them is a silent no-op), where the
  settings-payload policy functions may be referenced, and the writer manifest
  that gates stem/preview cache writes — with every section validated loudly
  at load, so a malformed record fails the suite instead of quietly shrinking
  a net to match nothing.
- **Local, private crash reporting.** Opt in with `the-oracle privacy-opt-in`;
  reports stay in this install's `crash_reports/` folder (sanitized, capped at
  20 × 32 KiB), are never uploaded, and the doctor surfaces them for review.
  `PRIVACY.md` documents the whole data posture, every claim pinned by a test.
- **Offline licensing.** Activate a purchased license with
  `the-oracle activate <token>`; `the-oracle license-status` and `machine-id`
  report the install's state. Nothing existing is gated — the community edition
  is everything that ships today. An expired or absent license degrades to
  community; nothing is ever locked away.
- **Log rotation.** The application log now caps at 5 MiB × 3 backups instead
  of growing without bound.
- The doctor now announces *before* a first-run model download that can take
  several minutes (previously silent, indistinguishable from a hang) and names
  `--skip-model-init` as the way out.
- **Visible, deterministic self-healing.** When the engine-output gate rejects
  a take and the one-shot fresh-seed retry heals it, the render now says so:
  progress and the completion summary report the retry, and the healed take is
  bit-for-bit reproducible (a re-render serves the same cached audio).
- `the-oracle sweep-cache` runs a project's stem cache through the
  servable-stem gate on demand — dry run by default, `--apply` deletes,
  `--json` for scripts — reporting every purged stem with its reason.

### Fixed

- The Live sidebar no longer sticks on a finished or failed job: renders,
  previews, and preview teardown all return it to idle (and the mirror-drift
  scan now covers every dialog-dismissal shape so it stays that way).

## [1.3.0] — 2026-09-21

### Added

**Self-healing synthesis**
- A one-shot fresh-seed retry when the engine-output sanitizer rejects a
  generation (flat silence, DC tone, NaNs): the utterance re-synthesizes once
  at `seed + 1` instead of failing the render. On the batched Vulkan path
  only the rejected requests re-run in a fresh subprocess — successful audio
  is kept and progress accounting is untouched. A second rejection still
  fails loudly so the stem is never cached.

**Input-salvage diagnostics (doctor)**
- The doctor now scans `Input/` for subtitle files and reports which would
  take the CP1252 fallback, which would convert, and — critically — which
  would be **blocked** by bytes CP1252 cannot decode (they pass the lossy
  pre-check but fail the strict conversion re-decode, surfacing only as the
  GUI's "Could not convert"). Read-only; never creates `Input/`.

**Release tooling**
- `release.py --check` verifies the CHANGELOG too: exactly one
  `## [<version>] — <YYYY-MM-DD>` section for the current version, dated the
  release day, so a version bump cannot land without its user-facing notes.
- `release.py --sync-changelog` performs that day's edit as one command:
  retitles the `[Unreleased]` body into the dated section and re-inserts an
  empty `[Unreleased]` placeholder. Idempotent.
- The doctor surfaces release-metadata drift (version, pyproject, banners,
  changelog) by reusing the release check's probes.
- The committed patch-surface net also catches string-form patch targets
  (`setattr("the_oracle.app_gui.X", ...)`, `patch("app_gui.X")`,
  `patch.multiple`), so future extraction slices cannot be silently
  undermined by either patch spelling.

### Fixed

- The GUI suite's intermittent shutdown-time `RuntimeError: cannot join
  current thread` — a GC-triggered destructor self-join in
  `language_tool_python`; defused at construction.
- The stem cache can no longer serve pre-hardening degenerate entries: all
  three read routes validate content (with the sanctioned pause-only silence
  exception), and the batched path banks pause silence itself instead of
  asking the engine to synthesize empty text.

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

## [1.1.1] — 2026-09-14

> This section was written retrospectively: CHANGELOG.md itself was created
> only in the 1.2.0 cycle, so the pre-1.2.0 sections summarize the shipped
> feature inventory rather than day-of release notes — the bullets below
> include features that first shipped in the earlier V1.01/V1.10 releases
> (workspace persistence, the CUDA/onboarding waves) which have no sections
> of their own.

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
