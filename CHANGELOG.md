# Changelog

Notable user-facing changes to The Oracle. The version number itself lives in
one place (`src/the_oracle/__init__.py`); see `scripts/release.py --check`.

## [1.3.2] — 2026-10-05

### Added

- **The trust features got GUI surfaces.** Crash-report consent and review,
  and license activation, are now in-app dialogs alongside the existing CLI
  flags (`the-oracle privacy-opt-in`, `the-oracle activate`).
- The doctor's cache audit gained the disk side: every project `previews/`
  directory is scanned for files outside the gated preview owner's naming
  scheme — runtime evidence of a writer that bypassed
  `OraclePipeline.render_preview`.
- Release tooling: a crash-hunt reproduction harness (exit-code
  classification, per-backend render loops, offscreen GUI launch loop) whose
  `--acceptance` gate requires zero non-zero exits, zero crash records, and —
  reading the kernel log — zero native crashes inside the run window.

### Changed

- **The self-healing guarantee now covers the worker-pool path and the Live
  sidebar.** A multi-task seeded render that heals one hiccup reproduces
  byte-identically when the spawn pool runs the synthesis (the configured seed
  crosses the pool boundary through its initializer; retry notes ride home
  from whichever worker drained them), and the persistent Live column counts
  self-heals for the session — during the render and after it ends.
- **Renders start faster.** The 19 MB SymSpell frequency dictionary is
  deserialized once per process instead of once per corrector instance
  (~21% off the profiled synthesis total, ~11% off wall clock on the
  representative smoke render).

### Fixed

- Subtitle files that mix encodings (one UTF-8 cue exported into a
  mostly-CP1252 file — the classic two-editors concatenation) now recover
  **each cue on its own encoding** instead of turning the UTF-8 cue into
  deterministic mojibake. Clean whole-file UTF-8 and CP1252 files decode
  byte-identically to before; undecodable cues degrade to visible replacement
  marks instead of blocking the conversion.
- **The months-old render-click segfault is root-caused and fixed.** The
  render worker's lazy imports (SRT export, FLAC tagging) first-imported
  `mutagen` inside the QThread, racing the GUI main thread through
  shiboken6's signature import hook; everything the worker call graph needs
  now preloads on the main thread before any worker starts. Relatedly, a
  missing cache stem is now a cache miss, not corruption — the Vulkan path
  no longer logs false "System error" warnings and deletes nothing.

## [Unreleased]

### Added

- **Crash capture is now armed on every launch path, not on a consent
  transition.** With crash consent on, native-crash capture arms at the console
  entry point (every command, and the GUI, routes through it), at both GUI
  launch paths, in the spawned clean render child
  (`python -m the_oracle.render_subprocess`), in the doctor process and its
  probe children, and at the smoke and perf-baseline entry points. A session
  that *started* with consent already on used to run unarmed, which is why the
  08:06:51 GUI segfault left `crash_reports/native-crash.txt` at 0 bytes. Every
  arm keeps the same contract: idempotent, and fail-closed — with consent off
  it creates no file and no directory.
- **The crash-hunt acceptance gate fails when the net is unarmed.** For
  `--mode acceptance`, a fresh probe child through the real entry point reports
  whether the launch path arms capture under the current consent state; the
  gate fails on the regression shape (consent on, net unarmed) and fails closed
  when the probe itself cannot run, because a readiness check that silently
  does not run *is* the regression it guards. Consent-off stays visible but
  passing.
- Release tooling gained a checked suite count: `certify_suite_counts.py`
  checks a commit out into a fresh worktree, runs the full suite, and compares
  it against the counts recorded in `suite_counts_baseline.json`, so drift, an
  unrecorded commit, or a silent wrong-tree load each fail loudly. The one
  commit that can never carry its own recording — the commit that holds it,
  since a tree cannot contain a hash of its own content — is answered by
  comparison instead of by measuring: when its tree is identical to a recorded
  commit's apart from the baseline file, it inherits those counts and says so.
- A mutation probe and one hermetic GUI certifier family replace two hand-run
  disciplines. `mutation_probe.py` applies a source mutation, runs the named
  proof command, and reverts it under hash guards with per-run bytecode
  isolation, so the whole-second-mtime stale-bytecode trap cannot make a
  measurement silently wrong. The certifiers are one harness,
  `certify_gui.py --aspects contrast,chrome,studio,wizard`, with the recording
  studio and inference tour covered for the first time.
- The doctor names mixed-encoding subtitle files instead of folding them into
  the fallback count: files whose guarded-UTF-8 reading differs from their
  CP1252 reading report as "mixed-encoding, recovered per cue", which states
  the per-cue salvage behavior as the report's contract.
- **The first real Studio-gated capability: `the-oracle render-report`.** It
  measures a *finished* render — peak/RMS level and clipping, silence ratio,
  sample rate and channel count, utterance and speaker counts, and whether the
  plan's declared durations agree with the audio actually assembled — from the
  assembled FLAC and its `render_plan.json`, so it never re-renders and cannot
  alter a deliverable. `--json` for machines, `--out` to also write the report
  (atomically; beside the render when no path is given). The gate is
  deliberately **additive**: nothing that shipped before is gated, so the
  community edition loses no workflow, and the owner moves or drops the gate by
  editing one `policy.require(status, STUDIO_ENTITLEMENT)` call site. Every
  licence state is covered against the real verifier
  (`tests/test_render_report.py`, `tests/test_license_state_render.py`): studio
  runs it, community and expired licences get a refusal naming Studio, and the
  core entitlements keep working.
- The dependency audit and the secrets scan are pinned, reproducible steps
  rather than one-off commands: `pip-audit` and `detect-secrets` in the `dev`
  extra with a checked-in baseline, so a new advisory or a committed credential
  shows up as a diff instead of as a memory of having looked.

### Changed

- **The fast test loop can no longer report green without rendering a theme.**
  The GUI certification gates leave the slow marker, so `-m "not slow"` runs
  them, and CI runs one named step per aspect per operating system — a failed
  job names which certification broke.
- CI now tells the truth about what it covers. The Vulkan job's model-cache
  root moved to step level: declared at job level it used the `runner` context,
  which made GitHub reject the entire workflow file, so no job on any operating
  system had been running. Four Windows tests stopped assuming POSIX separators
  and resolved paths the way the records do. The Python support range is
  asserted as one range across pyproject, the doctor, the installer and the test
  matrix — the guard used to pin the *runner* to 3.12, which was red on the
  matrix's own 3.11 rows.

### Fixed

- The theme certifiers no longer write into the repository or the user's
  config: the startup and prewarm timing logs and the native-crash dump handle
  are contained, and the crash-records startup review is parked so its modal
  cannot block a headless sweep.
- The theme sweep certifies all eight registered sections instead of the five
  its tuple listed — the three that had gained chrome were registered but
  unverified — and measures the hidden Extra Voices section for what it
  legitimately is.
- **The shipped licence registry trusted a published test key.** `keys.py`
  carried exactly one entry, `k1`, whose public key is RFC 8032 §7.1 TEST 1's
  published vector — its seed is printed in the RFC and was hard-coded
  throughout this repository's tests, so anyone who read the spec could mint a
  token every install accepted as a genuine licence. The entry is gone, and the
  registry is deliberately **empty until the owner registers the production
  key**: an empty registry fails closed (every token reports `unknown_key` and
  the install runs as community), and `scripts/release.py` refuses to build
  artifacts around an empty *or* forgeable registry, reading the tree being
  packaged rather than the running installation. The licensing suite now runs
  against an injected test-only key, so no test can pass by trusting a
  published seed.
- **Ed25519 verification accepted a universal forgery.** A public key that
  decodes to the identity point made `(R = [s]B, S = s)` verify for *any*
  message — a signature valid everywhere, from a 32-byte key. Small-order
  public keys and R are now refused, and non-canonical point encodings
  (`y >= p`) are rejected at decode. The RFC 8032 §7.1 vector set (TEST 1, 2, 3,
  1024 and SHA(abc)) and malformed-input coverage — per-vector one-bit
  mutations over R/S/message/key, non-canonical S, wrong and empty lengths, and
  a totality check that verification never raises — pin it.
- **Ctrl+hover help is owned by its main window.** Each window has its own
  helper, parented to that window and released before teardown. The Control key
  state comes from the window's key handlers; tooltip filters are scoped to the
  window and tooltip-owning controls. Linux/offscreen churn completed 50
  window rebuilds with zero application-scope filters, and the six-theme
  chrome certification passed 10 consecutive runs. Windows crash certification
  remains open.
- The release checksum manifest is written with LF endings whatever platform
  produced it. Text-mode translation on Windows wrote CRLF, the trailing `\r`
  became part of the filename `sha256sum -c` tried to open, and a manifest from
  a Windows release run could never be verified by the runbook's own check
  (measured on a real runner: exit 1).

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
