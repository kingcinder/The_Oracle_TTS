# The Oracle — State (completeness manifest)

**Current release: V1.3.0 (self-healing synthesis + input-salvage diagnostics + verified release tooling)**

This file is the repo's self-designated completeness record. It is the
authoritative context for the Omega meta-skill loop (searched, never
rewritten by the loop).

## Done

- **Subtitle-target naming and speaker-ref hint wording each have one owner (2026-09-18)**:
  `src/the_oracle/subtitle_targets.py` owns where a subtitle's companion files go (the
  converted `<stem>.srt.txt`/`<stem>.vtt.txt`/`<stem>.txt` script, and the render's `.srt`
  sidecar) — previously spelled out at six call sites, two of which had silently dropped
  the non-subtitle fallback. `src/the_oracle/speaker_ref_hints.py` owns the `--speaker-ref`
  flag form, which keys are additional voices, every sentence, the line assembly, and the
  shape-tolerance for the two forms the advice arrives in. The two display surfaces had
  genuinely drifted — the CLI told the user to edit a rejected label "to a name or
  'Speaker X'" while the GUI stopped at "edit the label in the file" — and now render
  byte-identical advice apart from the bullet they are allowed to choose. Tests pin the
  *ownership* (source scans that fail if any module re-spells a literal, each with a
  vacuity guard) rather than only the output, because a rendered copy would pass an
  output-only test on the day it was written.

- **The GUI's hint logic now delegates instead of duplicating (2026-09-18)**:
  `MainWindow._speaker_ref_hint_lines` is a wrapper that calls the owner, and
  `RenderWorker` uses the naming owner. Both were verified by a real CLI run
  (`.venv/bin/the-oracle check-input` on a subtitle) and the offscreen GUI popup test.

- **Every doctor PASS line is now accounted for (2026-09-18)**: each check was classified
  by what its verdict is actually built on. Execution-backed in-run: `python`,
  `chatterbox_import`, `perth`, `qt`, `entrypoint`, `deterministic_smoke`, `turbo`, and
  `chatterbox_init` when not skipped. Existence is the property: `ffmpeg` (resolved from
  PATH), `voice_sources` (how many reference clips exist). Documented and honest as-is:
  `real_engine_smoke` is an import/prerequisite check that never runs the render, and says
  so in code and in its separately-falsified `output_exists`. Verified rather than
  assumed: the deterministic smoke deletes its project before its first pass
  (`smoke.py:189`), so its reuse verdict is earned in-run. Experimental result: parking all
  of `build/` and comparing every verdict field moves **nothing**, so no PASS is graded
  from the doctor's own leftovers.
- **`vulkan_backend.ok` was graded from a filename (2026-09-18)**: it was
  `bool(binary) and model_file_exists and device_available` — pure existence, while
  `find_audiocpp_binary()` returns anything that `exists()`. With `ORACLE_AUDIOCPP_CLI`
  pointing at a real file that is `chmod +x` but is not a program, the doctor reported
  `ok: True` beside `audio_cpp_devices: []` and `error: ""` — a PASS next to contradicting
  evidence the same run had collected by executing that binary, and beside its own RDNA1
  caveat. Fixed at the root: the device probe now reports `{ran, devices, detail}`, `ok`
  requires `binary_runs`, and the reason is carried into the human report (as a WARN — this
  line never gated `overall_ready`, so readiness is unchanged). A pre-existing test that
  had *pinned* the defect (`error == ""` with the comment "not executability") was
  corrected. New guard: `tests/test_doctor_report_is_history_independent.py` requires every
  verdict to survive removing `build/`, and fails on the historical defect's shape.

- **The launch path is validated from a clean checkout, repeatably (2026-09-18)**: the
  clean-tree comparison had been done by hand twice and nothing pinned it, so it now
  lives in `scripts/fresh_clone_acceptance.py`: `git archive HEAD` gives a tree with no
  `.venv` and none of the git-ignored build output a working tree accumulates, and both
  legs run the full suite, the doctor gate, and a nine-row matrix of the documented
  entry points (each asserted to dispatch to a specific `manage_install.py` subcommand
  by reading its `usage:` line). Paths into the tree under test are substituted rather
  than registered; the only registered measurements are the same two durations the
  doctor idempotence check allows; the remaining 18 deltas are per-field registrations
  with reasons. An unregistered delta exits 1. It also asserts the clean tree loaded its
  *own* code and refuses to report otherwise, because this interpreter's editable
  install otherwise resolves to this working tree. Wired into the test matrix as
  `--only doctor,wrappers` on Linux.

- **`./oracle` was not executable (2026-09-18)**: the README's first instruction
  (`./oracle install`, lines 30-33 / 263-266) and the doctor's own CUDA hint
  (`doctor.py:827`) both invoke `./oracle`, but it was tracked mode 100644, so a fresh
  clone died with exit 126. Every `./*.sh` the README invokes was already 100755; this
  was the only miss. Fixed with a mode-only change (`chmod +x oracle`), pinned by a test
  that reads `git ls-files -s` for every tracked file the README invokes as `./<path>`.
  Found by the new acceptance script's wrapper matrix, which went 5/9 -> 9/9.

- **The Vulkan smoke can no longer skip silently (2026-09-18)**: the Vulkan backend
  smoke had skipped in every CI run the suite ever had — its guard is the git-ignored
  native `audio.cpp` build, which no runner provisioned — so a green suite proved
  nothing about that backend and nothing objected to the skip. Two root causes, two
  fixes. Skips are now always reported with their nodeid and reason at the end of every
  run, and any skip fails the session when `ORACLE_FAIL_ON_SKIP=1` (logic in
  `tests/skip_audit.py`, only wired by `tests/conftest.py`). A new `vulkan-smoke` CI job
  builds `audio.cpp` (its `ggml-vulkan` CMakeLists requires `glslc`), installs Mesa's
  software Vulkan ICD so a GPU-less runner can report a device, verifies `vulkaninfo`,
  fetches the model into a cache outside the clone, and runs
  `scripts/vulkan_ci_smoke.py`, which asserts the smoke's own gates before running the
  suite under strict skips — so the job cannot pass while the test would still skip.
  Proven live: strict skips pass with 46/0 skips on this fully provisioned tree and fail
  on a clean archive copy naming the exact reason; the orchestrator refuses with
  `the Vulkan smoke would still skip` on that same copy. 1009 passed (this tree) /
  1008 passed + 1 skip (pristine), the skip now loud rather than silent.

- **The deterministic smoke's reuse proof leaves no duplicate (2026-09-18)**: the
  smoke's second render exists to prove the first pass's stems are reused, and it
  rendered under the dialogue's own name — but the export policy never overwrites
  an existing render, so the proof deposited `smoke_dialogue (1).flac` beside the
  real output (and `(2)` on repeats), an artifact nothing reads. The second pass is
  a probe, so it now renders under `cache_reuse_probe.flac`
  (`REUSE_PROBE_FILENAME`) and deletes that file immediately;
  `SmokeRenderResult.second_output_path` is gone with it. The verdict itself is
  unchanged (`compute_incremental_changes` compares only utterance hashes), and
  the tests no longer pin the duplicate: they assert the project dir holds exactly
  `["smoke_dialogue.flac"]` and now also require the pass-2 timings to show
  `cache_hit` true and `synthesize_seconds == 0` for every utterance — reuse that
  actually happened rather than reuse inferred from a plan comparison. Falsified
  by giving the probe pass a different reference: the verdict flips to False and
  `cache_hit` to `[False, True, False, True]`. 957 passed this tree / 956 + 1
  skipped clean copy, which leaves exactly one FLAC. Details in `JUNO_FIXES.log`.

- **The doctor is proven read-only, for every check including future ones
  (2026-09-18)**: the two gate regressions in this class were caught *after* the
  fact by comparing two reports, so a third check could reintroduce one and stay
  unnoticed unless its write happened to change a compared field. Now
  `tests/helpers.py` offers a reusable whole-tree content snapshot
  (`repo_tree_snapshot` / `snapshot_differences`) and
  `tests/test_doctor_read_only.py` runs the real doctor and fails if anything
  outside `build/doctor_deterministic_smoke/` — the deterministic smoke's own
  project, which is that check's subject — was created, changed or removed. Git-
  ignored paths are deliberately included, since that is where both regressions
  lived. The guard parks an existing `build/` (same-filesystem rename) so every run
  starts from the fresh-clone state; without that it silently passed, because a
  check that writes only what is missing writes nothing on a machine that already
  has it. Proven by mutation: the original defect is caught naming its three
  created files, a new check caching into `Output/logs` is caught too, and a
  vacuous snapshot is caught by its own test. Files ≤1 MiB are hashed in full;
  larger ones (3.35 GB native build clone) by size+mtime, so two snapshots cost
  ~1.5s. 957 passed this tree (193s) / 956 + 1 skipped clean copy with no `build/`
  at all. Details in `JUNO_FIXES.log`.

- **Gate idempotence enforced in CI (2026-09-18)**: the doctor's verdict must not
  depend on how many times it has been run — it regressed twice in that exact
  class (a readiness PASS citing a smoke FLAC it never produced; a voice-source
  count inflated by the previous run's own writes) and nothing in CI compared two
  consecutive runs. `scripts/doctor_idempotence.py` now runs the doctor twice in
  one mode and fails if the two `--json` reports differ, naming each differing
  field with both values and printing a diff with measurements masked. Only
  `chatterbox_init.seconds` and `deterministic_smoke.runtime_seconds` (the two
  durations) may differ; the registry is pinned by a test, a container path
  cannot be registered to excuse its children, and a renamed measurement fails as
  drift. Wired into the CI test matrix as one shell-agnostic step (so both Linux
  and Windows run it) in the same `--skip-model-init --ci` mode the other doctor
  steps use. Proven by re-introducing the historical defect: the checker exits 1
  with `voice_sources.fallback_clip_count: 0 != 2`. It asserts agreement only —
  it does not become a second readiness gate. 949 passed this tree / 948 + 1
  skipped clean copy. Details in `JUNO_FIXES.log`.

- **Windows install proven by execution, not by inspection (2026-09-18)**: the
  suite pinned what the installer *writes*; nothing had proven those files
  *execute*. New CI job `windows-install-smoke` (windows-latest, own job, out of
  the push/PR matrix gate because it downloads the pinned checkpoint) runs
  `scripts/windows_install_smoke.py`: isolated environment (throwaway profile
  with a **space in its path**, HF/pip caches named for the cache step, launcher
  dir on `PATH`, `QT_QPA_PLATFORM=offscreen`, offline flags deliberately
  removed), `manage_install install --pytorch-runtime cpu`, then the managed
  `.cmd` launcher executed for real and the **Start Menu entry** launched until
  `launch_gui()` reports `mainwindow_built`, killed via `taskkill /T /F /PID`.
  Cache key follows `models/pins.py`, so a pin bump cannot reuse a stale
  checkpoint. `tests/test_windows_install_smoke.py` (14 tests, any host) pins the
  environment, the GUI-wait paths, the CI contract and the drift check that
  launcher paths come from `manage_install` itself; a dropped PowerShell
  continuation is caught by mutation. Also fixed here: `make_layout()` leaked a
  deleted temp profile into `os.environ["APPDATA"]` (restored in a `finally`,
  with a two-case regression test). Honest boundary: the job has not yet run on
  a Windows runner; what is proven locally is its contract and logic. 930 passed
  this tree / 929 + 1 skipped clean copy. Details in `JUNO_FIXES.log`.

- **Offline guarantee enforced at every model-loading entry point (2026-09-18)**:
  the `.oracle_offline` marker was honoured only by the generated launchers and
  `manage_install run_gui()`, so the console script (`the-oracle render` / `gui`),
  `scripts/doctor.py` and `scripts/real_engine_smoke.py` loaded models with
  offline resolution off. `huggingface_hub` reads `HF_HUB_OFFLINE` at **import**
  time, so a late `os.environ` write is a no-op — measured: 0.0s offline vs 23.0s
  (5 retries × 8s) for an unsatisfied resolution, and 0.0s once the import-time
  constant is forced. New `src/the_oracle/offline.py` is the single owner
  (marker name, variable set, `apply_offline_environment()`), wired into
  `cli.main`, `doctor.main`, `real_engine_smoke.main`, and `manage_install.run_gui`.
  `tests/test_offline_guarantee.py` has one test per entry point plus seeded-cache
  resolution in a fresh process with `socket.connect` forbidden; each proven
  load-bearing by mutation. Real runs: offline flag with a closed endpoint →
  real-engine smoke render succeeded (23.4s); doctor full model init identical
  with and without the marker. 916 passed this tree / 915 + 1 skipped clean copy.
  Three adjacent gaps reported, not fixed: the Vulkan `.gguf` and the
  LanguageTool snapshot are not in the offline bundle, and `download_models.py`
  / `build_offline_bundle.py` intentionally stay online. Details in
  `JUNO_FIXES.log`.

- **Install boundary pinned by tests (2026-09-18)**: the harness that drove a
  whole install with `subprocess.run` intercepted and the user's
  `HOME`/XDG/HF roots redirected into scratch — previously an uncommitted
  scratch script — is now committed as `tests/install_recorder.py`, with
  `tests/test_install_boundary.py` asserting the exact commands and files of
  both install paths: the five pip installs, the desktop integration calls,
  exactly one full (non-CI) doctor run last, the created-file set and its
  contents, offline wheel args with no index URL or `https://` anywhere, the
  seeded `refs/main` pins plus the `.oracle_offline` marker, `python -m venv`
  first on a fresh repo, and the Windows Start Menu branch. Assertions were
  verified load-bearing by mutation (a bumped package pin fails 5 of 9 tests;
  dropping `--no-index` fails the offline network-proof test). 906 passed on
  this tree, 905 passed + 1 skipped on a clean `git archive HEAD` copy.

- **Install verifies once (2026-09-18, launch-readiness mission goal 2)**:
  `install()` used to verify twice — `bootstrap()` ran the doctor, then
  `install()` ran it again — and each doctor run constructs the Chatterbox
  model, so a fresh `./install_oracle_tts.sh` paid for two identical model
  loads (process-boundary proof: 2 -> 1 recorded `doctor.py` invocations).
  `install()` now calls `bootstrap(skip_doctor=True, ...)` and keeps its own
  full (non-CI) `run_doctor()` after the launchers are registered, so the
  entrypoint is still a blocking check. Two tests added, both proven to fail
  without the fix. Consequence, accepted: a failing final verification now
  leaves the already-written launchers in place (previously bootstrap's
  identical run aborted first); `uninstall` removes them.
  - Doctor gate re-confirmed trustworthy after the change: byte-identical
    across consecutive runs with a clean `build/` (no
    `build/real_engine_smoke/inputs` created, `fallback=0`).
  - Environmental red, not a defect: the doctor's fresh-shell entrypoint
    probe resolves the stale `/home/cody/.local/bin/the-oracle` (old
    `/home/cody/The_Oracle_TTS` checkout, outside the repo), so non-CI
    `doctor` exits 1 on this machine while `--ci` exits 0. A real
    `bootstrap`/`install` rewrites that launcher before verifying. Suites:
    897 passed (this tree) / 896 passed + 1 skipped (pristine tree).
    Nothing pushed. Details in `JUNO_FIXES.log`.

- **Launch-readiness pass (2026-09-17)**: aligned the working copy to
  upstream first — local `main` was 34 commits behind and 0 ahead, and the
  "CI failing" premise was stale (run 35039888506 on `main` is green as of
  2026-09-16). The uncommitted ingest-refactor work from an interrupted
  architecture pass was preserved on branch `wip/ingest-refactor-hints`;
  local `main` was then fast-forwarded a320ab2 -> 96f5081. Two real bugs
  fixed: `manage_install.py` `update()` dropped `--offline-bundle` on its
  no-venv fallback (an offline target would silently install from the
  network), and the offline bundle's Windows `install.bat` passed a
  backslash-terminated directory (`%~dp0`) as `"...\bundle\"`, whose
  trailing backslash-quote PowerShell parses as an escaped quote. Verified:
  full suite 886 passed / 0 failed, doctor green via both the manager and
  the wrapper, all five shell wrappers dispatch to the right subcommand,
  deterministic smoke render passes with cache reuse. Nothing pushed.
  Details in `JUNO_FIXES.log`.

- **Fully offline install (2026-09-14)**: every model pinned to an exact HF
  commit SHA in `src/the_oracle/models/pins.py` (newest commits as of
  2026-09-14); `scripts/build_offline_bundle.py` builds a self-contained
  bundle (repo snapshot + pinned models in HF-cache layout + all wheels for
  Linux/Windows); `install --offline-bundle <dir>` installs with zero
  network (pip `--no-index --find-links`, seeded HF cache with `refs/main`
  pointed at the pins); managed launchers export `HF_HUB_OFFLINE=1` on
  offline installs so no model fetch can touch the network; delete
  `.oracle_offline` in the install dir to go back online.
  `scripts/download_models.py` now reads the same central pins.

- **Systematic-debugging pass on the transformer timestamps (2026-09-13,
  from the four-dimension audit)**: three root-caused and test-pinned fixes
  in `ingest_transformer.py`. (1) Unbracketed chat-export timestamps
  (`2024-01-01 10:00 Alice: hi`, `10:00 AM Bob: hello`) were never fixed —
  the timestamp rule only matched the bracketed `[...]` form; a new
  `_UNBRACKETED_TIMESTAMP_RE` branch (date+time / time-only, optional
  AM/PM) now drops the prefix, with the bracketed rule's remainder-is-a-real
  speaker-turn safety check, in both `analyze_text` and the transform loop.
  (2) The transform loop re-emitted `_SPEAKER_RE`-matching prose lines via
  parsed label/text, silently corrupting `At 10:00 the bell rang` →
  `At 10: 00 ...` with no fix recorded; already-canonical lines now pass
  through byte-identical unless a bullet was stripped. (3) analyze/transform
  disagreement class: check-input/fix-folder initially reported the new
  form clean because analyze lacked the branch — a regression test now pins
  their agreement. 3 new tests; README/Input-README corrected (the messy
  sample exercises text-repair, not the transformer). Full suite: 870
  passed.

- **Repo cleanup (2026-09-12)**: pruned `NVR_research_backup/` (176 MB of
  unrelated NVR-firmware research, was gitignored) and all stale runtime
  output in `Output/` (191 MB, nothing newer than Sep 10), plus `build/` and
  `.pytest_cache/` artifacts (~370 MB total freed). Filed the two previously
  untracked dialogue samples (`fransisco help.txt`, the intentionally-messy
  `stream_of_consciousness_dialogue_with_typos.txt` — a useful transformer
  test case) and updated Input/README's sample list. .gitignore dropped the
  dead NVR and legacy-backup rules. Full suite (867) and deterministic smoke
  re-verified green after the prune; main pushed to origin through `fcb5745`.

- **Whole-program debug session (2026-09-12)**: full-suite baseline (867
  passed), then live end-to-end exercise of every CLI subcommand through real
  dispatch — check-input (clean/messy/missing/JSON/`--check-refs` with real
  WAV probing), fix-folder (dry-run, JSON apply with backups, missing-folder
  exit 2), voices (+JSON), and the render flag matrix. The flag-conflict
  guard, SRT auto-convert, and graceful missing-model handling all verified.
  A **real render succeeded end-to-end**: an .srt input was auto-converted,
  speakers attributed (Winston→A, Julia→B), two utterances synthesized by
  Chatterbox, assembled FLAC + render_plan.json written, exit 0. Dev tools
  green: deterministic smoke (cache reuse verified), doctor (the one FAIL is
  the environmental turbo-checkpoint prefetch; WARNs are hardware reality —
  1 GiB Quadro below VRAM minimum — correctly reported, not crashed on),
  GUI theme certifier (all 6 themes legibility-certified). Zero product
  bugs found; three initial probe "failures" were probe mistakes (wrong flag
  names / dispatch pattern), corrected and re-verified.

- **Front-end live debug session (2026-09-12)**: drove the remaining
  unexercised GUI flows offscreen (recording studio/wizard lifecycles,
  preview-dialog geometry persistence, template menu rebuild idempotency,
  close-with-studio-open). Two probe FAILs were diagnosed as probe artifacts,
  not product bugs: `_start_recording_wizard` correctly no-ops when no studio
  is open (passing with one open), and geometry persistence is deliberately
  gated on `_app_settings_ready` (probe now sets it; round-trip verified).
  All 227 GUI-suite tests pass; no code changes needed; main at `a2bb961`.

- **Ingestion transformer (2026-09-11, this campaign)**: new
  `src/the_oracle/ingest_transformer.py` analyzes an input script before use
  and automatically corrects non-canonical speaker-turn formats. Detected and
  fixed in place (canonical `Label: dialogue` output the existing ingester
  attributes correctly): dash/pipe separators (`A - text`, `Alice | text`),
  bracketed labels (`[A]: text`, `[Speaker A] text`), bulleted/quote-marked
  turns (`- A: text`, `> A: text`), period separators (`A. text`), orphan
  label lines (dialogue on the next line), chat-export timestamp prefixes,
  and non-UTF-8 encodings (UTF-16/CP1252 transcoded to UTF-8). Prose is
  protected: `Note:`/`See:` labels and parenthetical em-dash sentences are
  never rewritten (spaced-dash-inside-text guard + document-level label
  evidence). The GUI's Analyze path (`prepare_project`) now runs the
  transformer check first: a warning popup explains every issue and offers a
  one-click in-place fix with a timestamped backup; unfixable issues are
  explained with a continue/cancel choice. Accepting the fix opens a
  **Preview Fixed Text** dialog first: a unified diff (original vs fixed,
  `-`/`+` marked lines) via `preview_fixed_text()`, and nothing is written
  until the user accepts; cancelling in the preview leaves the file
  untouched. Accepting the fix re-runs Analyze automatically: the same
  click's `prepare_plan` reads the corrected file from disk and populates
  the review table in one step (no second click needed), and the status
  panel records the correction (count, file, backup path) so the automatic
  re-analysis is visible rather than silent. Transforms are idempotent and verified end-to-end against the
  real `TextIngestor` attribution. The CLI render path runs the same check:
  issues are always reported on stderr, and `--fix-input` corrects the
  fixable ones in place (timestamped backup) before the pipeline loads;
  without the flag the file is never modified and the output includes a
  re-run hint. 27 module tests (`tests/test_ingest_transformer.py`), 5
  GUI-flow tests (`tests/test_ingest_transformer_gui.py`), and 6 CLI tests
  (`tests/test_cli.py`); full suite 727 passing.

- **Batch folder fix (2026-09-11, this campaign)**: File → Batch Fix Input
  Folder... scans every `.txt`/`.md` file in a chosen folder (non-recursive;
  `*.bak-*` fix backups and non-text extensions skipped) via
  `analyze_folder()`, computes all rewrites up front via
  `preview_folder_fixes()` (nothing written), and shows **one combined
  Preview Batch Fix dialog**: each file's unified diff grouped under a
  `=== filename ===` header, unfixable warnings listed at the bottom.
  Accepting applies every fix in one pass (`apply_folder_fixes()`), each
  original backed up, and the status panel reports the total plus per-file
  fix counts and backup paths; cancelling leaves the whole folder
  untouched. A folder with no problems gets an informational popup and no
  preview; a warnings-only folder gets an explanatory popup with no fix
  offered. The batch is idempotent (re-running finds nothing) and verified
  end-to-end through the real `TextIngestor`. 8 module tests
  (`tests/test_ingest_transformer_batch.py`) + 3 GUI tests; full suite 740
  passing.

- **check-input CLI subcommand (2026-09-11, this campaign)**:
  `the-oracle check-input FILE [--fix]` lints a dialogue file's formatting
  without loading the render pipeline (no LanguageTool download, no model).
  Exit codes: 0 = no issues (or all fixed with `--fix`), 1 = issues found
  (fixable ones listed with their exact rewrites, warnings marked `!`),
  2 = file missing/unreadable. Report-only by default; `--fix` corrects
  fixable issues in place (timestamped backup), then re-analyzes so the
  exit code reflects what remains. Warnings are never rewritten, and the
  re-run hint is suppressed when nothing is fixable. 6 tests in
  `tests/test_cli.py`; full suite 746 passing.

- **--check-input-json (2026-09-11, this campaign)**: `the-oracle render
  --check-input-json` emits the input-formatting report as exactly one JSON
  document on stdout instead of human-readable stderr text — `file`,
  `fixable_count`, `warning_count`, and an `issues` list (`line`,
  `snippet`, `fixable`, `description`); with `--fix-input` the same
  document gains `fixed_count` and `backup` (pre-fix issue list plus what
  was applied, in one object so consumers can always parse stdout as a
  single document). Missing/unreadable files report an `error` key rather
  than emitting nothing. 6 tests in `tests/test_cli.py`; full suite 752
  passing. The document builder is shared with `check-input --json`
  (below) so the two schemas can never drift.

- **check-input --json (2026-09-11, this campaign)**:
  `the-oracle check-input FILE [--fix] [--json]` emits the same
  machine-readable report the render path produces — one JSON document on
  stdout, nothing else printed in json mode so CI can parse stdout
  cleanly. Exit codes are unchanged and the document always agrees with
  them: 0 = clean (or fixed with `--fix`, where the document describes the
  file's **post-fix** state and gains `fixed_count`/`backup`), 1 = issues
  found, 2 = missing/unreadable (or fix failure) with an `error` key.
  `speaker_refs`/`rejected_labels` are always present (stable schema).
  5 tests in `tests/test_cli.py`, including a schema-parity pin against
  `_input_json_document`; full suite 804 passing.

- **--speaker-ref suggestions (2026-09-11, this campaign)**: when the
  transformer reports a file, the report now suggests the exact voice flags
  for the script's cast. `suggest_speaker_refs(text)` reads the cast from
  the **post-fix** text (so a speaker only visible after a transform is
  included) and computes the mapping via the pipeline's own
  `DualSpeakerAttributor._map_labels_to_voices` — first-appearance order
  onto keys A..X, so a suggestion can never drift from what a render does.
  A/B use `--speakerA-ref`/`--speakerB-ref`; keys C+ are marked `new` (the
  flags a user must ADD). `rejected_labels(text)` lists labels that fail
  speaker validation entirely ("Chapter:", "Note:") — no reference audio
  can attribute those; the user must edit the label. Surfaced in:
  `check-input` (stderr hints, `use`/`add` markers), the render path's
  human report (stderr), and `--check-input-json` (stable `speaker_refs` +
  `rejected_labels` keys, always present). 6 tests in `tests/test_cli.py`;
  full suite 758 passing.

- **SRT auto-detection (2026-09-11, this campaign)**: the CLI render path
  converts `.srt` subtitle inputs into dialogue scripts before analysis
  (`src/the_oracle/srt_ingest.py` + `cli._maybe_convert_srt`). Detection is
  strict: only files with the `.srt` extension whose content parses as
  complete SubRip cues convert; a `.srt` file without valid cues (and any
  other file) passes through untouched, and missing/unreadable files are
  reported by the ordinary error paths. Conversion strips timestamps, cue
  indexes, HTML markup (`<i>`, `<font>`), and ASS overrides (`{\\an8}`);
  a `Name:` prefix on a cue line names the speaker (validated by the
  ingester's own `canonical_speaker_label`, so `Note:`-style prose prefixes
  stay spoken); cues without names continue the previous speaker, falling
  back to `Narrator`; dashed cue lines split into separate turns with
  alternation to the other voice of the pair (and never re-merge within a
  cue); consecutive same-speaker cues merge into one turn. The converted
  script is written as a sibling `<name>.srt.txt` (subsequent runs reuse it;
  the subtitle file itself is never modified) and replaces the input for
  the transformer check and pipeline; the cast round-trips through the real
  `TextIngestor`. 12 module tests (`tests/test_srt_ingest.py`) + 3 CLI-path
  tests; full suite 773 passing.

- **--fix-input-interactive (2026-09-11, this campaign)**: the render path's
  terminal equivalent of the GUI popup + preview. Shows the issue summary
  and the rule-labeled diff of the exact rewrite on stderr, then prompts
  `Apply these fixes before rendering? [y/N]` before writing (backup kept).
  Any non-yes answer — or EOF/ctrl-D — leaves the file untouched and aborts
  the render with exit code 2 *before OraclePipeline() loads*. In
  non-interactive runs (piped stdin, CI) the flag degrades to a report-only
  check with a stderr notice, so automation never blocks on a prompt.
  Prompt function is injectable for tests. 7 tests in `tests/test_cli.py`;
  full suite 785 passing. Combining it with `--fix-input` is now an
  **explicit error** (helpful two-line explanation of each flag's
  behavior), checked first in `handle_render`'s fast-fail validation —
  before missing-`--input`/`--outdir` and speaker-ref checks — instead of
  silently favoring the interactive review. 2 more tests; full suite 824
  passing.

- **Rule-labeled preview diffs (2026-09-11, this campaign)**:
  `transform_text_detailed(text)` returns per-fix provenance (`LineFix`:
  output line number, rule name, original text) alongside the fixed text;
  `transform_text` is a thin wrapper preserving the old API. Rules: dash,
  period, bracket, timestamp, orphan, bullet, encoding (`RULE_LABELS` maps
  them to display names). `labeled_fixed_diff(original, fixed, line_fixes)`
  builds the unified diff with each rewritten `+` line prefixed by its rule
  label — e.g. `+ [dash/pipe separator] A: Hello there.` — attributing by
  *position* (hunk headers are walked, not text-matched), so identical
  source lines fixed by different rules label correctly. The GUI's
  single-file preview uses this via `preview_fixed_text` (now 5-tuple with
  `line_fixes`) and `FolderFix.line_fixes`. 5 module tests + updated
  batch-GUI assertions.

- **Recursive batch fix with folder tree (2026-09-11, this campaign)**:
  `analyze_folder` now recurses into subfolders (`rglob`), skipping hidden
  files/directories, `_BATCH_SKIP_DIRS` (`__pycache__`, `node_modules`,
  `venv`, `site-packages`), non-text extensions, and `*.bak-*` backups;
  results sort by path relative to the scanned folder so the tree reads
  naturally. The **Preview Batch Fix** dialog is now a two-pane layout: a
  folder tree on the left (root = folder name, children = fixable files by
  relative path with fix counts) and a rule-labeled diff pane on the right
  showing the selected file (first file preselected); unfixable warnings
  listed beneath. Apply/Cancel semantics unchanged (all-or-nothing, backups
  kept, per-file status-panel lines with relative paths). 4 module tests +
  2 GUI tests; full suite 799 passing.

- **Per-file batch checkboxes (2026-09-11, this campaign)**: every file
  row in the Preview Batch Fix tree gained an include-checkbox (all
  ticked by default), so a batch can be applied selectively instead of
  all-or-nothing. The apply button live-updates to "Apply Fixes to N of
  M File(s) (K fix(es))" as ticks change and disables when nothing is
  ticked; Apply returns only the ticked fixes and the caller applies
  exactly those (unticked files untouched, no backups created for them).
  The dialog now returns the included-fix list (was a bool). 1 new GUI
  test (exclusion end-to-end, default-all-ticked, live label/enable);
  full suite 805 passing.

- **WebVTT (.vtt) support (2026-09-11, this campaign)**: the subtitle
  converter now accepts WebVTT alongside SubRip — same module
  (`srt_ingest.py`), same convert-not-overwrite contract. Parser additions:
  `WEBVTT` header, `MM:SS.mmm` clocks without hours, cue settings after
  the end time, cue identifiers, and `NOTE`/`STYLE`/`REGION` metadata
  blocks (skipped). **WebVTT voice spans** (`<v Winston>`) are lifted to
  `Winston:` prefixes before generic markup stripping — the demo caught
  them being deleted as tags, silencing the speaker. Detection is by
  content everywhere: `analyze_input_file` flags a valid `.vtt` as one
  fixable issue, `fix_input_file` writes the sibling `<name>.vtt.txt`
  script (subtitle never modified), the CLI pre-flight accepts `.vtt`
  (converting or reusing the script), and the batch scan includes `.vtt`
  files. The conversion moment now also **suggests the cast's voice
  flags**: the CLI pre-flight prints `_print_speaker_ref_hints` for the
  converted script (both convert and reuse paths), the render path's
  human report includes the hints (clean files too — the cast is the
  useful content there), and the GUI fix flow adds them to the post-fix
  "File Corrected" popup and the status panel, computed from the
  post-transform text so a subtitle conversion's cast is suggested even
  before the user accepts the fix. Full suite 820 passing.

- **Color-coded preview rows (2026-09-11, this campaign)**: the
  side-by-side Preview Fixed Text dialog highlights changed rows — soft
  red (70-alpha) on the original pane for `changed`/`removed` rows, soft
  green on the corrected pane for `changed`/`added` rows — via
  `QTextEdit.ExtraSelection` full-width line backgrounds (no rich text,
  theme-safe alpha). Row→line mapping is positional from
  `side_by_side_diff_rows`, so highlights are exact even with repeated
  lines. 1 GUI test asserts the highlight counts inside the live dialog;
  full suite 821 passing.

- **Remembered preview dialog geometry (2026-09-11, this campaign)**: the
  Preview Fixed Text dialog's size and position persist between sessions
  via the app-settings file (`preview_dialog_geometry`, a base64 Qt
  geometry blob). Restore happens before show (falling back to the
  900x560 default when absent); save happens on dialog close whenever
  app-settings persistence is enabled. The Qt blob encodes position
  relative to the virtual desktop, so multi-monitor moves survive; a
  corrupt blob fails `restoreGeometry` harmlessly. Schema hygiene added
  in `_normalize_app_settings` (non-string/empty values dropped). 1 GUI
  test (save → settings key → fresh-dialog restore); full suite 822
  passing.

- **Rule filter in the batch preview (2026-09-11, this campaign)**:
  `transform_text_detailed` gained ``exclude_rules`` — lines a matching
  rule would rewrite pass through byte-identical (and record no
  ``LineFix``); the whole-document ``srt`` rule can be excluded too.
  `preview_folder_fixes` forwards the filter and drops files whose fixes
  are all excluded (their rewrite would be a no-op — writing the file
  with its own content plus a pointless backup). The **Preview Batch Fix**
  dialog gained a checkbox row of the batch's present rules with human
  labels and per-rule occurrence tooltips; unticking rules live-updates
  the apply-button count/enablement, and Apply **recomputes** the batch
  against the filter (the precomputed rewrites include every rule, so
  applying them as-is would ignore the filter) intersected with the
  per-file checkboxes. Example: untick everything but ``chat-export
  timestamp`` to accept only timestamp fixes. 3 module tests + 1 GUI
  test; full suite 828 passing.

- **Rule names in the CLI reports (2026-09-11, this campaign)**:
  `FormatIssue` gained a ``rule`` field (the transformer rule name —
  `dash`, `period`, `bracket`, `timestamp`, `bullet`, `orphan`, `srt`,
  `encoding` — empty for warnings), populated by `analyze_text` and the
  file-level analyzers. The CLI human reports (`check-input` and the
  render path's format check) tag each fixable line as
  ``line N [rule]:``; both JSON documents (`check-input --json` and the
  render path's `--check-input-json`, via the shared
  `_input_issue_json`) gained a stable ``rule`` key (`null` for
  warnings/file-level issues without a rule), so CI can filter or group
  findings by rule. 2 tests; full suite 830 passing.

- **Color-coded rule labels (2026-09-11, this campaign)**: the ``[rule
  label]`` text in both preview dialogs is tinted per rule — a curated
  hue table (`RULE_COLORS`, exposed via `rule_color()`: dash green,
  bracket blue, timestamp orange, orphan crimson, …) with a deterministic
  FNV-1a fallback for future rules, so a mixed set of fixes reads as
  distinct colors at a glance. A shared `_color_rule_labels_in_view`
  helper appends foreground extra selections (must run **after** the row
  backgrounds — `setExtraSelections` replaces the list); used by the
  side-by-side dialog's right pane and the batch dialog's diff pane
  (re-applied per file selection). 2 module tests + 1 GUI test; full
  suite 833 passing.

- **Cue-timing pauses (2026-09-11, this campaign)**: the subtitle
  converter now preserves the one thing timings are for — silence. The
  gap between a cue's end and the next cue's start becomes a
  ``[pause=N]`` directive on the following turn when the gap is audible
  (≥ 400 ms; subtitle cues routinely butt together with sub-100 ms gaps
  that would only be directive noise), clamped to the pacing engine's
  2000 ms domain. Merged same-speaker turns carry the largest notable
  gap of their constituent cues; within a dashed cue only the first
  split turn inherits the gap (the split turns are simultaneous in the
  source). The directives flow through the pipeline's existing
  `parse_directives`/`apply_directives` path (verified end-to-end:
  stripped from spoken text, applied after punctuation scaling, pause_ms
  lands on the utterance). 2 tests; full suite 835 passing.

- **Subtitle conversion in the GUI (2026-09-11, this campaign)**: the
  input file picker now offers ``*.srt``/``*.vtt`` alongside txt/md, and
  picking a subtitle converts it immediately (sibling script via the
  CLI's `_maybe_convert_srt`, subtitle untouched, reuse of an existing
  script) and puts the **script** in the input field. The Analyze path
  (`prepare_project`) applies the same conversion to a subtitle path
  that was typed, remembered, or loaded from a project — before the
  transformer check — and re-points the field for the run. Both paths
  log the conversion to the status panel; an unreadable file reports
  and aborts instead of loading something unusable. 3 GUI tests; full
  suite 838 passing.

- **voices subcommand (2026-09-11, this campaign)**:
  `the-oracle voices` lists the default Seashells reference clips in the
  exact order the render path picks them — the first two lines are
  annotated ``(default Speaker A)``/``(default Speaker B)`` — with
  `--json` emitting a `[{label, path}]` array so scripts can wire paths
  into `--speaker-ref` flags programmatically. Backed by the same
  `default_voice_choices` the GUI's Default Voices list and the render
  fallback use, so the listing can never drift from actual behavior.
  Exit 1 (no crash) when no clips exist. 2 tests; full suite 840
  passing.

- **check-input --check-refs (2026-09-12, this campaign)**: `check-input`
  now accepts the render path's reference flags (`--speakerA-ref`,
  `--speakerB-ref`, repeatable `--speaker-ref KEY=PATH`) and, with
  `--check-refs`, validates each: the path must exist, be a real file, and
  decode as readable audio via soundfile's header probe (also rejects
  zero-frame files). Human mode prints one `[OK]`/`[BAD]` line per voice
  (a voice with no path is `unset` = OK); a bad reference fails the check
  with exit 1, alongside — not instead of — the formatting report.
  `--json` adds stable-schema `checked_refs` (always present in JSON mode,
  empty without the flag) plus `refs_ok`. The validator reuses the render
  path's `_validate_speaker_ref_paths`, so pre-flight and check can never
  disagree about what counts as a usable reference. 5 tests; full suite
  845 passing.

- **fix-folder subcommand (2026-09-12, this campaign)**: the batch folder
  fixer is now scriptable from the CLI: `the-oracle fix-folder FOLDER`
  scans recursively (same skip rules as the GUI batch: hidden, backups,
  VCS/dependency dirs) and applies every fixable rewrite with per-file
  timestamped backups (`--no-backup` to skip). `--dry-run` previews
  without writing; `--json` emits one stable-schema document (`fixes`
  with per-line rule/original detail, `warnings`, `applied` with backup
  paths, `dry_run` flag) so pipelines can parse folder scans. Exit codes:
  0 clean/applied, 1 warnings remain, 2 folder unreadable. Subtitle files
  follow the convert-not-overwrite contract (sibling `.srt.txt`/`.vtt.txt`
  written, subtitle untouched). Backed by the same `preview_folder_fixes`/
  `apply_folder_fixes` the GUI batch dialog uses — rule filter
  (`exclude_rules`) is GUI-only for now, logged for later. 7 tests; full
  suite 852 passing.

- **Rejected-label rename hints (2026-09-12, this campaign)**: when the
  format report flags a label the engine rejects (`Note:`, `See:`, ...),
  the suggestion now names the exact `--speaker-ref` flag that label would
  need *after* being renamed to an accepted form — e.g. `! 'Note' is not
  accepted ... rename the label, then provide --speaker-ref C=PATH`. The
  would-be key comes from feeding the whole cast (valid labels
  canonicalized, rejected ones as lowercase rename candidates) through the
  pipeline's own voice mapper in first-appearance order, so the flag
  matches what a render assigns post-rename. Surfaced in `check-input`
  human hints (stderr), `check-input --json` (`rejected_labels` entries
  gained `label` + `rename_flag`; string entries became objects — a
  schema change, consumers reading raw strings must migrate), the render
  path's report, and the GUI warning/post-fix popups + status panel via
  `_speaker_ref_hint_lines`. 3 tests; full suite 855 passing.

- **Settings: restore most recent input-file backup (2026-09-12, this
  campaign)**: a fix regretted can now be undone from Settings → "Restore
  most recent input-file backup". Every single-file fix (preview-accepted
  and trusted auto-fix) records `{file, backup, stamp}` into a persisted
  `recent_format_backups` list (most recent first, capped at 20; schema
  hygiene in gui_settings tolerates malformed entries). The restore action
  confirms, copies the backup text back over the corrected file, consumes
  the record, re-points the input field when the restored file is the
  current/remembered input, and logs to the status panel. Missing backup
  files are pruned with a clear error; declining leaves everything
  untouched. Scope: single-file fixes only — the batch folder fixer's
  per-file backups are not (yet) recorded here, logged for later. 5 GUI
  tests; full suite 860 passing.

- **GUI bug audit (2026-09-12)**: a targeted sweep of signal wiring, modal
  flows, thread lifecycles, and table state found and fixed three bugs:
  (1) the reference picker connected BOTH `currentIndexChanged` and
  `activated` to `_handle_reference_selection`, so picking "Custom Voice
  Reference Audio..." ran `_pick_audio()` twice — two stacked modal file
  dialogs; now `activated` alone (which also keeps the first-click behavior
  the second connection was added for). (2) `_handle_row_action` (the table's
  +/- control) repopulated the table from the plan WITHOUT
  `_sync_plan_from_table()` first, silently discarding unsaved edits in
  other rows' Repaired/Speaker/Emotion cells on every add/remove; now syncs
  before mutating. (3) the review table's Index, Original Text, and Duration
  columns were created with default editable flags (only Status was
  protected), so users could type into computed columns; now read-only
  (Repaired Text stays editable). Verified-correct during the sweep:
  thread close-path coverage (all 7 thread types joined/disconnected),
  enum-vs-enum dialog comparisons, `_output_name_edited` preservation,
  blockSignals discipline on repopulations, and lambda late-binding (all
  captures use default-arg binding). 4 regression tests; full suite 864
  passing.

- **GUI error-path audit (2026-09-12)**: second pass over runtime-error
  families (handler signatures, exception gaps, dangling refs, media
  lifecycles, wizard cleanup). One bug fixed: `load_gui_settings` let raw
  `FileNotFoundError`/`JSONDecodeError` escape, so clicking a template
  that had been deleted or was corrupt JSON crashed the menu click with
  an unhandled traceback instead of the intended error dialog — it now
  raises `GUISettingsError`, which every caller already catches
  (`load_app_settings` was already safe with its own fallback). Verified
  correct in the same sweep: all eight worker `run()` methods are fully
  try-wrapped with failed-signal emission; both wizards emit `completed`
  on window-X close (`closeEvent` → `_finish(False)`), so no dangling
  `_inference_wizard`/`_recording_wizard` ref blocks reopening; media
  status handlers defer player stops out of signal emission; slot
  signatures match their signals (zero-arg slots on value-bearing signals
  are legal Qt). 3 tests; full suite 867 passing.

- **SRT-aware transformer (2026-09-11, this campaign)**: the transformer's
  own APIs now convert subtitles, complementing the CLI's `_maybe_convert_srt`
  pre-flight. `analyze_input_file` flags a structurally-valid SubRip file
  (detection is by content, not extension) as one fixable issue — "ingested
  directly, every cue would be read as narration" — and offers the
  conversion; `transform_text_detailed` handles SRT as a single whole-file
  `srt` rule (rule label "SRT subtitle"); `fix_input_file` is the one
  non-in-place case: it writes the converted script to a sibling
  `<name>.srt.txt` (subtitle file backed up, never modified) and returns
  the *written path* as its new first element (was the fixed text; all
  callers updated — the CLI render path re-points `--input` at the
  converted script after `--fix-input`/`--fix-input-interactive`, and the
  GUI flow renders the script too); `analyze_folder` includes `.srt` in
  batch scans. The GUI popup/preview/trusted-file flow, `check-input
  --fix`, and the JSON report all handle SRT inputs through the same APIs.
  5 module tests; full suite 796 passing.

- **Remember-my-choice / trusted files (2026-09-11, this campaign)**: the
  single-file fix preview dialog offers a checkbox, "Remember this choice
  and fix this file automatically in the future". Ticking it + accepting
  adds the file's resolved path to `trusted_format_files` in the app
  settings (`gui_settings.py` schema pass-through with string hygiene); a
  later Analyze/Render on the same file auto-corrects it silently (backup
  still kept, status-panel line says "trusted file"), no popup or preview.
  Trust is per-file: other files keep the full popup + preview flow.
  Settings > **Forget remembered auto-fix approvals** clears the list.
  SECURITY-RELEVANT FIX found by the new tests: the popup's Cancel branch
  compared `clickedButton() is StandardButton.Cancel`, which can never be
  true (widget vs. enum) — clicking Cancel silently proceeded to Analyze.
  Now uses `box.standardButton(clicked)`. 5 GUI tests; full suite 791
  passing.

- **Side-by-side preview dialog (2026-09-11, this campaign)**: the
  single-file **Preview Fixed Text** dialog is now a two-pane layout —
  original on the left (with `-` markers on rewritten rows), corrected text
  on the right (with `+` markers and the rule label as a suffix, e.g.
  `+ A: Hello there. [dash/pipe separator]`), headers `Original` /
  `Corrected (with fix rule)`, and synchronized scrolling (either pane's
  scrollbar drives the other). Alignment comes from
  `side_by_side_diff_rows(original, fixed, line_fixes)` in the transformer:
  a `SequenceMatcher` opcode walk producing `DiffRow(kind=same/changed/
  removed/added, rule)` rows — rewrites pair line-for-line as `changed`,
  exact regardless of script length. The batch dialog keeps the unified
  labeled diff. 2 GUI tests (pane contents + behavioral scroll coupling);
  full suite 786 passing.

- **V1.01 GUI release** (2026-09-08, this campaign):
  - Pain-point `~` markers (`word~word`) in input scripts are replaced by a
    normal spoken word boundary at the single synthesis chokepoint
    (`Utterance.text_for_tts`) in every correction mode — including Verbatim —
    so the annotated junctions (e.g. `syncronized~lockstep` in
    `Input/What is, reality.txt`) no longer cause engine hang-ups/
    mispronunciations, while unattached `~` (e.g. `~42`) is still read
    verbatim and the review table keeps the annotation.
  - Theme system (`src/the_oracle/gui_themes.py`): six stylized themes
    (Studio Light, Night Deck, Sephiroth, Pulp Science Fiction, Tape Deck,
    Ocean Depths), each a full design-token set (surfaces, typography,
    geometry) machine-contrast-checked at import time (WCAG AA body,
    3.0:1 large/on-accent). Default stays Studio Light. Theme menu in the
    menu bar; the choice persists.
  - Voice-modifier sliders renamed to match their mechanics (Identity Lock,
    Emphasis Punch, Delivery Variety, Emotion Depth, Human Drift, Breath
    After This Speaker) and given exact-mechanics tooltips: engine knob,
    domain, formula-level behavior (CFG dual-guess sampling, emotion
    preset blend weights, the per-0.1 Human Drift deltas, punctuation
    pause multipliers). Hybridize Voices is its own labeled section inside
    each speaker panel (Hybrid Second Voice / Voice Dominance / Hybridize
    Mode).
  - Every GUI section (Shared Render Settings, Speaker A/B, Status/Errors,
    Live, extra cast voices) is a collapsible section with a per-section
    Section Size slider redistributing its splitter share; three nested
    splitters (main/sections/lower) are resizable by handle or slider.
  - Workspace persistence in `app_settings.json`: theme, last input file
    (defaults to `Input/What is, reality.txt` on fresh installs when no
    remembered file exists), configurable default Input/Output folders,
    generic output filename warning preference, ALL options + slider positions
    (shared and per-speaker), splitter sizes, section shares, collapses, and
    window size.
  - File → Save Profile… / Load Profile… (same payload as Settings →
    Save/Load Settings).
  - One-stop manager wrappers `./oracle` (Linux/macOS) and `oracle.ps1`
    (Windows) with Install / Start / Update / Uninstall actions
    delegating to `scripts/manage_install.py` (new `update` action keeps
    user data); release version now 1.1.1.
  - First-run inference onboarding (`src/the_oracle/inference_wizard.py`):
    hardware-aware CUDA/CPU/Vulkan discovery, unsuitable-GPU explanations,
    a dependency-ordered Continue-driven GUI tutorial, live control
    highlighting with beside-the-tooltip placement, and Settings replay modes
    for the full tour, hardware discovery only, and main GUI tour only. The
    chosen inference path is applied to the real picker and tutorial completion
    or dismissal is persisted.
  - First-open Recording Studio onboarding (`src/the_oracle/recording_wizard.py`):
    microphone and supported sample-rate selection, shared Input/ teleprompter
    script selection (bundled What is, reality.txt first), configurable
    Seashells/ destination, generic-name caution with disable checkbox,
    persisted last-used recording choices, and a staged
    dependency-ordered guide with detailed mic placement, room, plosive,
    breath, enunciation, emotional delivery, audition, and speaker-assignment
    instructions. Settings can replay the guide at any time.
  - Certification: `scripts/certify_gui_themes.py` builds the real window
    under all six themes offscreen and asserts no truncation (geometry
    sweep with a populated table), WCAG legibility, working section
    sliders, collapse behavior, and a full persistence round-trip; full
    suite at 499 passing tests, deterministic smoke render green.

- Chatterbox-only render pipeline: `standard`, `multilingual`, and `turbo`
  variants on PyTorch, with CPU/system-DRAM as the guaranteed fallback, an
  opt-in CUDA device mode for suitable NVIDIA GPUs, and an opt-in Vulkan
  backend via audio.cpp for AMD RDNA1-class GPUs (vendored RDNA1 device-lost
  fix). CUDA remains the existing PyTorch/Chatterbox path rather than a
  second synthesis engine: hardware discovery reports card name, VRAM, driver
  visibility, runtime availability, and rejects cards below the 4 GiB
  suitability floor with an actionable reason.
- Cross-platform bootstrap/install/doctor/run/uninstall for Linux and
  Windows; managed launcher + desktop integration.
- Desktop GUI: review table, per-row preview, repair, live progress panel,
  profiles/templates, saved project manifests, Ctrl+hover help.
- CLI render flow with saved project manifests and deterministic smoke
  render path for repo-local verification.
- Batched Vulkan rendering (`--request-sequence`, bounded 32-request
  groups, live per-request progress) with truthful timing logs.
- Voice catalog: bundled generic voices (`Seashells/generic/` — 5
  English + 4 Chinese references, Apache-2.0 / CC BY 4.0 attributed,
  English-first in the picker), curated `Seashells/` defaults, recent
  custom clips.
- Voice blending: deterministic derived reference from two clips with a
  preference weight and mix/alternate/layer modes; persisted in projects
  and profiles; honored by both backends.
- Saved blend voices: **Save Blend As...** creates a named voice that
  appears under "Saved Blends" in the picker (catalog
  `Profiles/blend_voices.json`, derived clips `Profiles/.blends/`).
- Correction Mode **Verbatim (no changes)**: true passthrough of the
  source text (no spelling/grammar/punctuation edits).
- Fidelity fixes (commit `d77c278`): assembly no longer drops chunks of
  chunked utterances (parallel loader keyed by position, regression-tested);
  monologue renders no longer require/construct an unused speaker profile.
- Test suite: 499 passing tests after CUDA, workspace-persistence, and guided-onboarding coverage.
- Test suite: 454 passing tests on the `feature/voice-craft-and-

  recording-studio` tree (416 on `main`) including the new assembly/
  blend/monologue/pacing/parity/recorder coverage.
- Voice-craft + Recording Studio (branch
  `feature/voice-craft-and-recording-studio`, **uncommitted pending review**):
  punctuation-aware pacing (turn pause scaled by terminal punctuation;
  chunk seams breathe ~35%, the full turn pause applies only after an
  utterance's final chunk), timbre-locked emotion (per-line emotion moves
  emphasis/pause only; temperature/CFG stay per-speaker so the voice
  character is consistent), perceptual voice sliders (Voice Lock, Emotional
  Punch, Delivery Variety, Emotion Strength, Human Drift, Breath After This
  Speaker, Which Voice Wins), and the Custom Voice Recording Studio window
  (teleprompter fed from `Input/`, mic + supported-sample-rate pickers,
  auto-incrementing `Seashell_No_x.wav` saving into `Seashells/`, live
  level meter, auto-audition of each take, Listen again, Use for Speaker
  A/B quick-assign that refreshes the voice pickers on close).
- **CPU (PyTorch) ⇄ Vulkan parity** for the voice-craft decisions:
  `tests/test_backend_parity.py` plans a mixed-punctuation dialogue with
  `inference_backend=pytorch` vs `vulkan` and asserts per-utterance pauses
  and engine settings are identical on both, with timbre-lock holding (one
  temperature per speaker); the Vulkan/engine-path suites pass
  (84 tests in the batching/backend/synthesis/render-worker files). Pacing
  and emotion live in shared plan/assembly code, never in the engine call,
  so both inference backends behave identically by construction.

## Next

- **Consumer-market readiness campaign is written down; all contract decisions
  resolved (2026-09-25)**: the design record is
  `docs/superpowers/specs/2026-09-25-consumer-market-readiness-design.md` — unit
  decomposition U1 diagnostics (crash/logging/privacy) → U2 licensing → U3 GUI
  surfaces → U4 refinement (measurement-first) → U5 release — plus resolutions of
  every open decision: crypto = `pynacl` Option A with offline-bundle evidence,
  no machine-locked seats in v1, trial minting vendor-side, no crash transport,
  fail-closed consent, caps as designed, next-session crash notice. The execution
  plan is `docs/superpowers/plans/2026-09-25-consumer-market-readiness.md`
  (bite-sized TDD steps per unit, mutation-proven pins, one bounded unit per
  session). The two scoped unit bullets below remain the design contracts; the
  campaign executes them slice by slice.

- **Crash/telemetry/privacy unit is scoped, not built** (2026-09-26):
  `docs/CRASH_TELEMETRY_DESIGN.md` is the design contract — local-first
  crash capture (excepthook + threading hook + Qt message handler +
  faulthandler, which finally gives the blocked-on-repro segfault watch item
  a data path), a redaction sanitizer whose core rule is *drop what it
  cannot classify*, fail-closed consent (unreadable = opted out) stored
  separately from app_settings.json, log rotation for the currently
  unbounded `utils/logging.py`, and **zero built-in network transport** —
  the user shares a report deliberately. PRIVACY.md lands last, after its
  claims are pinned by tests. Rollout §11; open decisions §12.
- **Licensing/anti-piracy unit is scoped, not built** (2026-09-26):
  `docs/LICENSING_DESIGN.md` is the design contract — module layout
  (`the_oracle/licensing/`, Qt-free, offline by construction), an ORACLE1
  signed-token model (Ed25519 verify keys embedded, private key never ships;
  pynacl vs vendored pure-Python is the open crypto decision), editions
  where **community = today's full feature set** (nothing existing is gated
  during this unit), the doctor's `licensing` check contract, and the
  offline-guarantee pins the unit must land (activation with zero network
  I/O, no heavyweight imports, no network remedy strings). Seven-step
  rollout in §8; four open decisions in §9, all owner-customer.
- **The docs surface exists.** CHANGELOG.md, docs/UPGRADING.md, and
  docs/TROUBLESHOOTING.md landed (8915feb) with the `gui_smoke_prewarm` prune;
  README links all three. The offline audit's two recommended fixes also
  landed (fd302d7): `setup-vulkan` refuses on an offline install and the
  LanguageTool warm download skips when the marker is present.
- **The venv is checked against the declared pins.** The 2026-09-20 incident —
  an out-of-band pip event that replaced transformers/huggingface_hub/chatterbox
  with incompatible majors and broke two offline-guarantee tests — is now a gate
  rather than a watch item: the doctor's `dependency_pins` check compares every
  pinned requirement in `pyproject.toml` (runtime plus each optional group)
  against the installed distribution metadata and reports missing or
  mismatching packages, so drift surfaces in the doctor report instead of as a
  mysterious test failure days later.

- **Release metadata is single-sourced.** `__version__` in `src/the_oracle/__init__.py` is the
  only tracked version literal; `pyproject.toml` reads it via `[tool.setuptools.dynamic]` and
  `scripts/release.py` enforces the invariant (`--check`), rewrites the README/STATE banners
  from it (`--sync-banners`), and builds versioned sdist+wheel with a `checksums-<version>.sha256`
  manifest via the in-venv PEP 517 hooks (offline). A copy of that manifest is written into
  the tracked `release_checksums/` folder, so a published artifact is verifiable from a clone
  rather than only from the gitignored build folder; an existing record for a version is never
  rewritten silently (a rebuild producing different hashes is a refusal with instructions, not
  an overwrite). `--check` also requires CHANGELOG.md to
  carry exactly one `## [<version>] — <YYYY-MM-DD>` section for the current version, dated
  the release day — so a version bump cannot land without its changelog entry, and a release
  is: bump version → `--sync-changelog` (retitles the `[Unreleased]` body into the dated
  section) → `--sync-banners` → commit → build → commit the tracked
  `release_checksums/checksums-<version>.sha256`. Undated `--check` therefore passes only on
  the release day; `tests/test_release.py` pins the heading shape, drift, duplicates,
  injected-date semantics, and the sync-changelog mode. Verified end-to-end
  on the real repo; `tests/test_release.py` pins it. Setuptools' `build_meta` mutates
  `sys.argv` permanently — any future in-process hook caller must not read `sys.argv` after
  the first hook call.
- **The MainWindow extraction, slices 1-6 landed.** Slice 1: the transformer popups and
  preview dialogs live in `src/the_oracle/gui_ingest.py` (~500 lines), with the Qt classes
  the tests patch injected from MainWindow at call time. Slice 2: the format-health
  bookkeeping (trusted-file approvals, backup records) lives in `gui_settings.py` — which
  already normalized both schemas on load, so one file now owns the full payload
  lifecycle; `app_gui.py` holds zero references to the two schema keys. Slice 3 (the
  cluster slice 2's review had deferred pending a widget-read injection design): the
  settings-payload POLICY — default/current payload builders, the vulkan-only
  audio_cpp knob-persistence rule, cast resolution, blend decode — now lives in
  `gui_settings.py` as pure functions fed by a frozen `WidgetSnapshot` the window builds
  in one method (`_widget_snapshot`), with `PayloadDefaults` supplied by the window so
  gui_settings never imports pipeline. Slice 4 (render/preview cluster): `gui_render.py`
  owns `RenderWorker`, `PreviewWorker`, `RenderProgressDialog` and the isolated-render
  child-environment builder, with `OraclePipeline` split-owned between window assembly
  (app_gui) and the workers' direct non-subprocess fallback (gui_render). Slice 5
  (Vulkan cluster): `gui_vulkan.py` owns the device-row text and model-path parsing.
  Slice 6 (sidebar/LivePanel chrome): `gui_chrome.py` owns `LivePanel` and
  `build_live_section`, the Live column's collapsible/resizable section chrome; app_gui
  re-imports both, so MainWindow's construction, the two progress handlers that drive it,
  and the existing test import keep resolving to the identical objects, while the splitter
  assembly and the `_register_section("live", ...)` persistence wiring stay in app_gui
  where the layout state lives. All six slices required zero edits to existing tests,
  which is the seam proof. What the review predicted held: the later clusters carry
  wholesale-patched module names (thread classes, workers, dialogs), and the committed
  patch-surface net is what kept that surface intact — it now enforces three rules
  (MOVED_OWNERS, SPLIT_OWNED, and PARTIAL_OWNED, added for the `QHSectionGroup`
  construction split slice 6 created) plus string-form patch targets, with the
  moved-owner record itself in `scripts/patch_surface_manifest.json` so extraction
  tooling can read what has already been moved without importing the test.
- **The parked `wip/ingest-refactor-hints` branch is finished and merged** (591efa0, fast-forward). Its two still-valid owners landed: `srt_ingest.ensure_subtitle_script` owns the convert-not-overwrite subtitle policy (both `cli._maybe_convert_srt` and `app_gui._convert_subtitle_input` are presentation wrappers; the GUI's old strict-decode copy — which blocked CP1252 subtitles the CLI converted fine — is gone), and the speaker-ref report data lives in `ingest_transformer.speaker_ref_report_for_file` instead of a CLI private the GUI imported. Ownership is pinned by source scans in `tests/test_subtitle_conversion_has_one_owner.py`, both pins mutation-proven. Worktree note: testing a linked worktree against the shared venv requires PYTHONPATH shadowing (the editable install resolves `the_oracle` to the main checkout); verified by probe before trusting any worktree result.
- The stream-of-consciousness sample is now a permanent fixture
  (`tests/fixtures/stream_of_consciousness_dialogue_with_typos.txt`) pinning the
  transformer's spelling-blindness: format-clean file, typos survive every transform
  untouched. Typos in *content* are render-time text repair's domain, not the format
  gate's -- if anyone asks why check-input says "no issues" on a file named
  "_with_typos", that is the tested contract, not a bug.
- The same audit question is worth asking of the *GUI*'s readiness surfaces (the
  onboarding/status panels), which report capability to the user but are only covered by
  smoke tests today.
- Watch the first real CI run of the new `Fresh Clone Acceptance (Linux)` step; the
  whole path is exercised locally (both legs, all three checks, 18 deltas registered,
  exit 0), but the step itself has not run on a runner yet.
- Consider giving the `.ps1` entry points a dispatch matrix of their own on Windows;
  today the acceptance script's matrix is `.sh`-only, which is why its CI step is
  Linux-gated, and `test_review_fixes_scripts.py` can only statically inspect `oracle.ps1`.
- Watch the first real run of the new `vulkan-smoke` CI job (it is gated off pull
  requests, so `workflow_dispatch` triggers it): what is proven locally is the job's
  declared contract and the orchestrator's preflight/gate/refusal logic, but no GitHub
  runner has yet built `audio.cpp` and opened the smoke's device gate. Expect the usual
  one round of dependency fixes on first execution.
- Re-render `Input/What is, reality.txt` on the Vulkan backend with generic
  voices as live verification of the assembly fix (hardware-dependent).

## Noticed, not yet actioned

- **Hardware-floor claim vs. documented floors (2026-09-25)**: the sales claim
  driving this campaign is "minimum 3 GB of free and available VRAM/DRAM"; the
  repo documents a **4 GiB Chatterbox CUDA suitability floor** (the Inference
  Setup tour and the CUDA picker both reject cards below it — README.md:276,284,
  STATE.md:801), while CPU/system DRAM is the guaranteed fallback. Both may be
  true (different resources), but they are not yet reconciled in one place and
  sale copy must not overclaim. Owned by plan step U4.5: evidence pass first,
  owner decision after. No code or copy was changed — flagged, not silently
  aligned.

- **`.ps1` executable bits are inconsistent (2026-09-18)**: `bootstrap_oracle_tts.ps1`,
  `doctor_oracle_tts.ps1` and `oracle.ps1` are tracked 100644 while
  `install_oracle_tts.ps1`, `run_oracle_tts.ps1` and `uninstall_oracle_tts.ps1` are
  100755. This is *not* fixed alongside the `./oracle` defect because unlike that one
  it has no failure mode: PowerShell is invoked as `powershell -File .\script.ps1` (as
  the Windows CI job does) and never by a shebang, and `git` on Windows does not track
  the bit at all. Making them consistent would be tidy, not a fix, and would touch six
  files' modes for no behavioural gain. The acceptance script's wrapper matrix is
  `.sh`-only for the same reason it is Linux-gated.

- **`pkg_resources` deprecation warning (2026-09-26)**: the suite emits 16 warnings,
  including `pkg_resources.resource_filename` coming from a dependency (not Oracle code
  — repo-wide scan finds no `pkg_resources` import in `src/` or `scripts/`). Harmless
  today; worth one look when the pinned dependency that imports it is next touched, since
  a future setuptools major could turn it into an error.

- **Enum-vs-widget identity audit (2026-09-11, prompted by the Cancel-button
  bug)**: the full GUI was swept for the `clickedButton() is
  StandardButton.X` bug family — no further instances. Verified correct:
  the two `result == StandardButton.Cancel` checks after `exec()` (exec
  returns the enum, so that comparison is valid), `QMessageBox.question`
  result checks, `_confirm_delete`'s `result == QMessageBox.Ok`, the
  `clicked is fix_button` widget-vs-widget comparison (valid — both are
  the same QPushButton instance), `checkState(0) == Qt.Checked`, the
  `EndOfMedia` `getattr(status, "EndOfMedia")` pattern in both playback
  handlers, and the wizards' signal-based button wiring (no identity
  comparisons at all). One fragile pattern exists only in tests
  (`buttons()[0]` positional indexing in four transformer GUI tests —
  brittle if Qt reorders buttons, but test-only).

- **RESOLVED 2026-09-20**: `tests/test_grammar_language_tool.py` used to
  leak temp dirs (found 2026-09-11 during the orphan sweep): each test
  created its cache via `tempfile.mkdtemp(prefix="oracle_lt_cache_")` and
  never removed it — ~140 `oracle_lt_cache_*` directories accumulated in
  `/tmp` across sessions. All four cache sites now use pytest's auto-cleaned
  `tmp_path`, the accumulated dirs were swept, and a before/after run probe
  showed zero new leaks (JUNO_FIXES.log 2026-09-20).

- Serpent-circle inventory scan updated (2026-09-08, in
  `~/.agents/skills/serpent-circle/`): the bloat scan and language
  histogram now honor `.gitignore` — untracked+gitignored residue
  (bytecode, `.venv/`, vendored clones) is treated as repo-declared
  retention, while tracked junk and non-ignored strays stay flagged. This
  lets the omega loop terminate **CONVERGED** instead of NO-PROGRESS on a
  fresh campaign; the change lives in the skill (outside this repo), so
  it is documented here rather than committed here.

## Deferred (intentional)

- **`Output/render_plan.json.bak` kept — omega residue finding dispositioned
  KEEP (2026-09-26)**: serpent-circle's inventory has flagged this untracked
  file twice (campaign 7, cycle 2; fresh-campaign preview). It is a render-plan
  backup for "What is, reality" (2026-09-21, 27 KiB) in the user's Output
  folder — the user's own artifact, not repo debris. Disposition: KEEP, do not
  clean, do not track. Deleting a `.bak` on the user's behalf is exactly the
  irreversible tidy-up this repo's rules forbid, and Output/ is runtime data
  that never ships. Any future omega cycle treats this as a documented keep,
  not an open finding.
- **`JUNO_FIRST_PROMPT.txt` kept — omega residue finding dispositioned KEEP
  (2026-09-26)**: flagged by the same serpent-circle inventories. Untracked,
  currently 0 bytes, sibling of the user's `JUNO_BRIEF.md` working set (the
  JUNO_FIXES.log journal is the tracked member of that set). An empty file
  looks like debris to an inventory scan; it is the user's placeholder prompt
  file. Disposition: KEEP, do not clean, do not track. Same rule as above —
  user-owned working files are outside the loop's authority.
- **GUI native crash — root cause found & fixed (2026-09-08)**: kernel-log
  `python: segfault at 0 ip 0000000000000000` (execution through a
  null/corrupted function pointer → use-after-free signature). Two lifetime
  bugs in the Recording Studio (added by the voice-craft campaign) matched
  the signature exactly and were fixed:
  1. `_stop_playback` called `stop()` + `deleteLater()` on the audition
     `QMediaPlayer` from inside its own `mediaStatusChanged` handler —
     tearing the QtMultimedia FFmpeg backend down mid-emission is the
     canonical player use-after-free. Fixed: one persistent player per
     dialog (lazy, created on first audition, reused like MainWindow's
     preview player); EndOfMedia now defers the stop via a zero-timer and
     never deletes the player.
  2. The `RecordStudioWorker` QThread was `deleteLater`'d from the
     `captured`/`failed` slots (fired from `run()`'s final lines while the
     thread still exits) and `closeEvent` never joined it — destroying a
     live QThread is a hard Qt abort. Fixed: teardown moved to the
     `finished` handler; dialog close and MainWindow close now do a bounded
     `wait()` and refuse to close if the thread cannot stop.
  3. Same deferred-stop discipline applied to MainWindow's preview player:
     EndOfMedia defers the stop via a zero-timer (never stop/delete from
     inside `mediaStatusChanged`), one persistent player is reused across
     previews, and window close stops the player before teardown.
  4. A full-file audit of every QObject teardown site found one more
     in-handler hazard: `PrewarmThread` emitted `ready`/`failed` from the
     last lines of `run()` and `_handle_prewarm_ready`/`_handle_prewarm_failed`
     called `thread.deleteLater()` with no `finished` connection — the same
     QThread-destroyed-while-running race. Fixed to match every other worker
     (teardown only from `finished` → `_cleanup_prewarm_thread`).
  Regression tests: `tests/test_recording_studio.py` (EndOfMedia handler
  safety, single-player reuse, finished-based worker teardown, close waits
  for worker, MainWindow preview-player deferral and close-stop, prewarm
  teardown via finished for both ready and failed paths). The pre-voice-craft Render-click crash report remains
  separate and still blocked-on-repro; repro launcher: `bash
  /tmp/gui_crash_catcher.sh` (recreated 2026-09-11 after a machine restart
  wiped /tmp). Findings: `.serpent-circle/04-debug/root-causes.md`.
  **Update 2026-09-11: the crash is CONFIRMED LIVE** — journalctl shows four
  more null-pointer segfaults (Sep 09 x3, Sep 10 x1), one in libQt6Widgets,
  and the Sep 10 one lands ~5 s after a logged `render_click`
  (`gui_action_timing.json` entry 182). The Sep 10 session's re-render
  succeeded after relaunch, so the trigger is state-dependent
  (render-after-render / preview-playback-active are candidate conditions;
  see root-causes.md). Static re-review of the current render flow found
  all worker teardowns correct; a real repro under the catcher is still the
  designated next step.
- **audio.cpp punctuation normalization is NOT patched**: its replacement
  table (`:`→`,`, `;`→`, `, dashes, quotes) exactly mirrors the installed
  Chatterbox Python reference `punc_norm`, so diverging would reduce
  model fidelity, not improve it. "Inflection at punctuation" is canonical
  model behavior, identical on both backends.
- `turbo` variant on Vulkan (PyTorch-only by design, rejected clearly).
- AMD ROCm remains intentionally out of scope for this release: AMD systems
  use CPU or the existing Vulkan/audio.cpp path. CUDA support is now present
  for NVIDIA through the PyTorch path, including GUI/CLI pickers, persisted
  device selection, manifest round-tripping, installer runtime selection
  (`auto`, `cpu`, `cuda`), and doctor diagnostics. A live CUDA render still
  requires compatible NVIDIA hardware, driver, CUDA-enabled PyTorch wheels,
  model availability, and cannot be certified on this CPU-only CI host.
- `Seashells/` runtime caches and `Output/` renders stay gitignored;
  `.venv/` bytecode is documented retention (see `.omega/CHANGELIST.md`).
- Recording Studio extras (deferred): a waveform/trim editor for cropping a
  take before saving, punch-in/overdub re-recording from a chosen prompt
  line while keeping earlier audio, and per-Seashell metadata sidecars
  (recorded-in sample rate, speaker tag) surfaced in the voice picker.