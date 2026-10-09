# The Oracle — State (completeness manifest)

**Current release: V1.3.2 (local private crash reporting + offline licensing + visible self-healing renders)**

This file is the repo's self-designated completeness record. It is the
authoritative context for the Omega meta-skill loop (searched, never
rewritten by the loop).

**Records-hygiene doctrine (2026-10-05, amended 2026-10-08):** a commit that closes work must
amend the STATE entries that work closes in the same commit — a Deferred
entry gets its RESOLVED block (with the repro/fingerprint/fix facts), a
Noticed entry is marked resolved-verbatim or rewritten only as far as
verification proves. Finding the entries is mechanical, not memory: when
closing work, grep Noticed/Deferred for the work's own name and keywords,
and amend every hit in the same commit — the entry the author forgot
exists is exactly the one a later audit finds. Leaving a closed entry for
a later reconciliation pass is how the 2026-10-05 records audit arose,
and is treated as a broken record, not a pending chore. An entry that
re-verification proves still true stays untouched; its holding commit is
cited when the entry is next touched.

## Done

- **Crash-eradication & performance campaign (2026-09-28, branch
  `wip/crash-eradication`)**: evidence-first campaign per the spec/plan in
  `docs/superpowers/{specs,plans}/2026-09-28-crash-eradication-*`. Armed the
  fail-closed crash capture (consent was off — no record could ever have
  existed), built a TDD'd reproduction harness (`scripts/crash_hunt.py`:
  exit-code classifier for 245/139 sigsegv and 137 oom, per-backend render
  loops, offscreen GUI launch loop, crash-record harvest; `scripts/gui_drive.py`
  interactive drive; `scripts/perf_baseline.py` cProfile+tracemalloc). Fixed
  the "Cached stem ... unreadable (System error)" cascade at its root
  (`73de246`): the batched Vulkan read path probes with no `exists()` check, so
  an ordinary cross-backend cache miss (hashes key on `inference_backend`) was
  logged as corruption with a pointless unlink — live-reproduced (16 false
  warnings) and mutation-proven silent-miss fix with a corruption-still-warns
  vacuity guard. QtMultimedia-as-crash-trigger hypothesis disproved by probe;
  the real exit-245/139 root cause (mutagen first-import in the render worker
  thread vs. shiboken6's import hook) was concurrently reproduced and fixed in
  `ed6ef8b` — this campaign's capture+harness are its evidence net. Perf
  baseline: no non-inherent hot spot remains (SymSpell load-once already
  landed; remainder is one-time init) — recorded, nothing optimized blind.
  **GATE PASSED**: acceptance = 5/5 pytorch + 3/3 vulkan renders + 3/3 GUI
  launches, 0 non-zero exits, 0 new crash records; full suite
  `ORACLE_FAIL_ON_SKIP=1`: 1508 passed / 0 skipped. Honest boundary:
  `VK_ERROR_DEVICE_LOST` not reproduced in 4 Vulkan renders (intermittent
  RDNA1 hang; already surfaced as a clean dialog). Details:
  `docs/superpowers/findings/2026-09-28-crash-evidence.md`.
  Extraction-campaign follow-on (2026-09-28): the writer-manifest policy has
  one owner too (commit b5b9683) — `scripts/patch_surface_manifest.json`
  gained a validated `writer_manifest` section (write functions, stem/preview
  target-name classifiers, per-kind gated owner sets, blindness floors of 4
  stem sites / 1 preview site), and `tests/test_stem_cache_write_path.py` now
  reads its rule data through the manifest's loud-failure loader family
  instead of hand-typed constants, with a vacuity guard pinning the loaded
  policy's self-consistency (non-empty sets, disjoint target classifiers,
  `pipeline.py` in both owner sets); `tests/test_app_gui_patch_surface.py`
  is the loader-side reader whose `load_*` family validates every manifest
  section. Slice-time suite 1518 passed / 1 failed (the 1 was the
  then-untracked U4.2 sweep net on the parallel session side); re-verified
  2026-10-05 against the settled tree, full suite 1529 passed / 0 failed.
  Suite-of-record progression since, every increment landed via
  commit_slices with slice-time gates: 1541 passed / 0 failed at the
  2026-10-05 records reconciliation (`cea2584`), then +4 faulthandler
  launch-arming tests (`4d59854`), +9 gui_cast owner-net tests
  (`0533e97`), +1 journal-checker attribution pin (`f128e10`),
  +3 render-subprocess arming tests (`725ddf0`), +9 doctor
  stale-resolved-records tests (`ec5a7ba`), +2 records RESOLVED-entry
  net tests (`212a613`) — and, landed on the parallel session's side,
  the GUI-launch arming, crash_hunt acceptance gate, and entry-point
  arming stream (`b1fb2d5`, `311e570`, `ec136c8`) with its 5 new pins
  plus the in-flight GUI-legibility pass's test additions —
  **1592 passed / 0 failed (2026-10-08, full suite on the settled tree,
  the parallel session's journaled verification; the legibility pass
  and the extraction-campaign's own uncommitted recording-owner net
  were in the tree), the current suite of record.**
  (The 1554 figure still visible in JUNO's 2026-10-07 entry was the
  standing count at `0533e97` — correct as written there, and a
  historical note only.)

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
- **The MainWindow extraction campaign — six slices landed (2026-09-19 to
  2026-09-21; commits d4388e3, a70aef3, 9a90281, 69989f4, a397943, 5a297a0),
  then a seventh follow-on (2026-09-27; commit 5d26f1a)** (the
  JUNO journal's *numbered* extraction slices stop at four — "Fourth" is
  gui_render — because the gui_vulkan and gui_chrome entries were never
  numbered there; this record settles the count at six for the campaign
  proper, the seventh being the Vulkan thread-cluster slice that
  moved the half the 2026-09-20 damage assessment kept in app_gui), each its
  own revertable commit. The original "zero edits to existing tests" seam
  claim was already inaccurate when written — gui_render repointed one test,
  and the thread-cluster slice deliberately touched three more — so it is
  corrected here. Slice 1: the transformer
  popups and preview dialogs live in `src/the_oracle/gui_ingest.py`, with the Qt classes the
  tests patch injected from MainWindow at call time. Slice 2: format-health
  bookkeeping (trusted-file approvals, backup records) lives in `gui_settings.py`,
  which already normalized both schemas, so one file owns the full payload
  lifecycle. Slice 3 (the slice the review deferred pending a widget-read
  injection design): the settings-payload policy — default/current payload
  builders, the vulkan-only audio_cpp knob-persistence rule, cast resolution,
  blend decode — lives in `gui_settings.py` as pure functions fed by a frozen
  `WidgetSnapshot` the window builds in one method (`_widget_snapshot`), with
  `PayloadDefaults` supplied by the window so gui_settings never imports
  pipeline. Slice 4: `gui_render.py` owns `RenderWorker`, `PreviewWorker`,
  `RenderProgressDialog` and the isolated-render child-environment builder, with
  `OraclePipeline` split-owned between window assembly (app_gui) and the workers'
  direct fallback (gui_render). Slice 5: `gui_vulkan.py` owns the device-row text
  and model-path parsing (and, since the 2026-09-27 thread-cluster slice, the
  four Vulkan thread classes, with `VulkanPreflightThread`'s policy dependency
  crossing a `preflight_report=` constructor seam). Slice 6: `gui_chrome.py` owns `LivePanel` and
  `build_live_section`; app_gui re-imports both so construction, the progress
  handlers, and the existing test import resolve to identical objects, while the
  splitter  assembly and `_register_section("live", ...)` stay in app_gui where the
  layout state lives. The campaign's two safety nets are themselves committed
  hardening — `tests/test_app_gui_patch_surface.py` (the patch-surface net) and
  `tests/test_payload_policy_ownership.py` (the payload-policy net) — and the
  later slices ran both green as their pre-flight gate before moving a line.
  The patch-surface net enforces MOVED_OWNERS (the ownership record now lives
  in `scripts/patch_surface_manifest.json`, validated and read by the
  scanner), SPLIT_OWNED, PARTIAL_OWNED (added when slice 6 created the
  `QHSectionGroup` split-readership hazard: a patch of `app_gui.QHSectionGroup`
  is not a no-op but a *partial* one), and string-form patch targets
  (`monkeypatch.setattr("the_oracle.app_gui.X", ...)`, `patch("app_gui.X")`,
  `patch.multiple("the_oracle.app_gui", ...)`) — so no future slice can
  silently undermine the tests the earlier slices depended on. SPLIT_OWNED is
  the concept slice 4 introduced: a name with *split readership* — today
  `OraclePipeline`, constructed by MainWindow (and the prewarm thread) from
  app_gui globals AND by the render workers' direct fallback from gui_render
  globals. An app_gui-level patch of a split-owned name stays legitimate for
  window-assembly tests but is a silent no-op on the worker path, where the
  net's WORKER_PATH_TESTS scope must patch at gui_render level; the map guard
  asserts the name exists on both sides so neither half can vanish. The
  payload-policy net pins slice 3's design instead: exactly one
  `_widget_snapshot`, payload-widget reads only from its explicitly sanctioned
  readers, the policy functions referenced only from their owner modules
  (gui_settings owns, app_gui delegates), and no hand-rolled copy of the
  17-key schema outside the owners.
- **The TTS hardening slices landed (2026-09-26; commits 2db5360, 9790b3d,
  3808064).** *Retry visibility + determinism*: the one-shot engine retry no
  longer heals silently — engines record a `SynthesisRetryNote` at successful
  recovery (`utils/audio.record_synthesis_retry`; three call sites including per
  healed request in the Vulkan batch), the pipeline drains notes per-process
  (`synthesize_task`, spawn-safe) and in-process on the batched path, notes ride
  `RenderProgress.retry_note` through the asdict-JSON GUI boundary, both progress
  handlers log them live, and the completion summary names the total via
  `plan.metadata["synthesis_retries"]`. Determinism is pinned at both layers:
  the healed draw equals a direct synthesis at the retry seed, and two cold
  seeded renders that each heal one hiccup are byte-identical (stems, output,
  note streams); a warm re-render serves the healed stems note-free — the healed
  take is the cached one. *Pause-writer single ownership*: `render_preview` no
  longer hand-rolls its silence buffer — pause-only previews route through
  `_write_pause_only_stem` (byte-identical output proven), an AST drift pin bans
  `np.zeros` in `render_preview`, and the writer manifest gained a preview-side
  gate whose target classifier catches the inline `cache.preview_path(...)`
  write shape the old name-only scan could not see. *Cache sweep*:
  `the-oracle sweep-cache` runs a project's stem cache through the servable-stem
  gate on demand (dry run by default, `--apply` deletes, `--json` for scripts),
  reports every purged hash with its reason, and the purge list IS the
  re-synthesis set — e2e-proven on a deterministic render. The sweep's pause
  decision is by canonical shape, not `allow_silence`: content alone cannot
  separate a sanctioned pause stem from a degenerate spoken entry, and exempting
  all silence would silently serve the entries the sweep exists to catch. The
  sweep also forced a no-mkdir existence probe (`ProjectCache.stem_cache_dir_for`)
  because the constructor eagerly creates the layout — sweeping a misspelled
  path would otherwise conjure an empty cache and report a clean run.
- **The consumer-market campaign's diagnostics unit (U1) is built and landed
  (2026-09-25)** — the design contract was written first
  (`docs/CRASH_TELEMETRY_DESIGN.md`: local-first crash capture via excepthook +
  threading hook + Qt message handler + faulthandler, which finally gives the
  blocked-on-repro segfault watch item a data path; a redaction sanitizer whose
  core rule is *drop what it cannot classify*; fail-closed consent stored
  separately from app_settings.json; log rotation for the previously unbounded
  `utils/logging.py`; **zero built-in network transport** — the user shares a
  report deliberately; PRIVACY.md deliberately lands last, after its claims are
  pinned by tests — rollout §11, open decisions §12), then three slices built it:
  - **U1.1 — log rotation + repo-local default log.** `configure_logging`
    installs a `RotatingFileHandler` (5 MiB × 3, read from module constants at
    call time), `default_log_file()` returns the repo-local `logs/oracle.log`
    (directory created on demand), and `logs/` is gitignored; FD-clean
    reconfigure semantics unchanged and pinned with rotation in place. Tests
    `tests/test_logging_rotation.py` (4). Full suite at this state: 1315
    passed / 0 failed.
  - **U1.2 — the crash core (handlers, sanitizer, consent, persistence).**
    `the_oracle/crash/`: fail-closed consent (unreadable/missing/malformed =
    opted out; stored separately from app_settings.json so settings corruption
    can never flip it on), the drop-what-it-cannot-classify sanitizer, the §4
    record schema, capped atomic persistence (20 records), sys/threading
    excepthooks that chain to the previous hook and never re-raise, and
    consent-gated faulthandler (enable-time consent — C-level code cannot be
    wrapped, and the docstring says so honestly). One `install()` call in
    cli.main; the Qt message handler waits for the GUI slice since app_gui.py
    is the concurrent engine thread's in-flight surface. Native segfaults now
    leave `crash_reports/native-crash.txt` — the blocked-on-repro watch item
    has its data path. 26 new tests; M-CONSENT and M-SANITIZE caught live with
    sha256-identical reverts. Lesson pinned in JUNO_FIXES.log: pytest's
    threadexception plugin owns threading.excepthook inside test bodies —
    identity tests pin sys.excepthook and invoke the threading hook directly.
  - **U1.3 — consent CLI + doctor crash check; §12 decisions confirmed.**
    `the-oracle privacy-status` / `privacy-opt-in` / `privacy-opt-out [--purge]`:
    the opt-in is the only path that can enable capture (and arms faulthandler),
    opt-out disarms immediately (fire-time consent) and keeps reports unless
    `--purge` is explicit — a flag, not a prompt, because the CLI refuses
    interactive prompts on non-TTY (repo precedent); the confirm dialog belongs
    to the still-pending GUI slice. The doctor's `crash_reports` check treats
    opted-out as valid, is excluded from `overall_ready`, infers writability
    with os.access instead of a write probe (read-only pin), and surfaces a
    native-crash dump as the segfault watch item's data. All four §12 decisions
    CONFIRMED and recorded in the design doc (no transport, caps as implemented,
    separate consent store, next-session GUI prompt). 15 new tests; M-CLI-CONSENT
    and M-DOC-CRASH caught live, reverts sha256-identical. Real bug the net
    caught: `bundle.list_records`/cap enforcement sorted by filename, but
    same-second records share a timestamp prefix and fell through to the random
    uuid suffix — "newest" and "oldest dropped" could both be arbitrary;
    ordering is now mtime-based. Full suite 1369 passed / 0 failed.
- **The licensing unit's core, doctor check, and offline pins are built** (2026-09-25,
  rollout steps 1–5 of `docs/LICENSING_DESIGN.md`; step 6 GUI slice and step 7
  privacy policy remain — campaign U3). The crypto decision is MADE: vendored verify-only
  pure-Python Ed25519 (`the_oracle/_ed25519.py`, pinned to the RFC 8032 §7.1
  test vectors — no new dependency, offline bundle unchanged; the pynacl swap
  path is one module). ORACLE1 canonical-JSON tokens verify through typed
  states (malformed/unknown_key/bad_signature/expired/machine_mismatch/
  store_error); the verify-key registry's commissioning key IS the RFC vector
  key, so registry corruption fails the spec pin; the machine fingerprint is
  hash-only and fail-open; the token store is atomic repo-local with a
  verify-before-save refusal gate; editions keep **community = today's full
  feature set** (nothing existing is gated). Vendor-only `scripts/license_sign.py`
  (env-var seed, wheel-exclusion pinned); `the-oracle activate` / `machine-id` /
  `license-status` wired. The doctor's `licensing` check: unlicensed = ok and
  deliberately excluded from `overall_ready` (a license state is not a broken
  install), every remedy offline-safe. 62 new tests (including the §7 key-rotation pin); six mutations (M1–M6,
  contracts stated in the test files) applied live, caught, reverted
  sha256-identical. Full suite 1316 passed / 0 failed.
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
- **Slice commits are tooled, not hand-rolled.** `scripts/commit_slices.py` lands a session's
  verified-but-uncommitted work as separate index-level slice commits: a slice file lists each
  slice's explicit paths (never `git add -A` — a concurrent session's dirty files cannot ride
  along), `!` verify commands prove a slice green after its commit before the next slice is
  touched (a failure stops the run with the failed slice committed and revertable, the journal
  unwritten), and the run ends with the house-format journal entry and its own commit.
  `--check-journal` validates `JUNO_FIXES.log` against the house format — per-entry date,
  field, and commit-citation rules (a dateish-but-malformed line fails loudly instead of
  silently skipping; truncated hashes would break `git rev-parse` verification, the same net
  `tests/test_record_integrity.py` enforces over the whole file) plus whole-file vacuity
  floors — and the journal-commit path refuses a malformed entry before any write. Parsing,
  refusals, end-to-end commits in a disposable repo, and verify semantics are pinned in
  `tests/test_commit_slices.py`.
- **The parked `wip/ingest-refactor-hints` branch is finished and merged** (591efa0, fast-forward). Its two still-valid owners landed: `srt_ingest.ensure_subtitle_script` owns the convert-not-overwrite subtitle policy (both `cli._maybe_convert_srt` and `app_gui._convert_subtitle_input` are presentation wrappers; the GUI's old strict-decode copy — which blocked CP1252 subtitles the CLI converted fine — is gone), and the speaker-ref report data lives in `ingest_transformer.speaker_ref_report_for_file` instead of a CLI private the GUI imported. Ownership is pinned by source scans in `tests/test_subtitle_conversion_has_one_owner.py`, both pins mutation-proven. Worktree note: testing a linked worktree against the shared venv requires PYTHONPATH shadowing (the editable install resolves `the_oracle` to the main checkout); verified by probe before trusting any worktree result.
- The stream-of-consciousness sample is now a permanent fixture
  (`tests/fixtures/stream_of_consciousness_dialogue_with_typos.txt`) pinning the
  transformer's spelling-blindness: format-clean file, typos survive every transform
  untouched. Typos in *content* are render-time text repair's domain, not the format
  gate's — if anyone asks why check-input says "no issues" on a file named
  "_with_typos", that is the tested contract, not a bug.
- **2026-10-09: adversarial pass over 8e932bd (child-QTimer startup,
  teardown sweep, section chrome) found and fixed three substantiated
  defects, each pinned.** (1) The wizard-launch child QTimer started with
  Qt's default 0 ms interval, silently dropping the 250 ms first-frame
  delay its own startup comment documents (`_wizard_launch_timer.start(250)`
  now carries it; `test_wizard_launch_timer_keeps_its_250ms_delay` pins it).
  (2) The autouse teardown sweep ran AFTER `monkeypatch.undo()`, so
  `closeEvent`'s final `_persist_workspace_layout()` wrote test payloads —
  pytest tmp paths and all — into the developer's real
  `~/.config/the_oracle/app_settings.json` (reproduced: one passing GUI test
  overwrote it; the file had been found corrupted in full-suite runs). The
  sweep fixture now takes `monkeypatch` explicitly so it finalizes while the
  XDG isolation is still active; pinned by
  `test_teardown_fixture_finalizes_before_config_isolation_is_undone`, and a
  stack-traced full run confirmed zero writes to the real path. The polluted
  file was reset to normalized defaults (theme preserved). (3) The sweep
  `deleteLater()`ed a window even when `closeEvent` had REFUSED the close
  (a GUI-owned thread still running) — destroying a live QThread aborts the
  process (reproduced: SIGABRT, exit 134, "QThread: Destroyed while thread
  is still running"). The sweep now honours the refusal and reclaims the
  window on a later pass; pinned by
  `test_teardown_never_destroys_a_window_whose_close_is_refused`.
  Section-chrome round-trips (new paths splitter, Review/Extra-Voices
  sections, pre-change schema, corrupt payloads) probed clean through the
  real startup path — no fix needed there. Validation: focused GUI nets
  227/227; full suite 1664 passed, exit 0, real config byte-identical
  across the run.

## Next

- **2026-10-08: the tenth extraction slice is EXECUTED — the ingest-tools
  handler cluster moved to the new `gui_ingest_tools` owner, seam-proven by
  mutation.** The move followed the plan recorded below exactly: the three
  flow bodies (`_run_ingest_transformer_check`, `_batch_fix_input_folder`,
  `_restore_most_recent_format_backup` — the remaining delegators
  `_show_fix_preview_dialog`/`_show_batch_fix_preview_dialog` and the
  trusted-file/backup wrappers stay on MainWindow, which remains the owner
  of app-settings state) moved verbatim into
  `src/the_oracle/gui_ingest_tools.py` as free functions taking the window;
  every patch-coupled Qt name crosses the boundary as an INJECTION parameter
  resolved from app_gui globals at call time (`message_box_cls=`,
  `file_dialog_cls=` — the VulkanPreflightThread precedent; the module
  imports no Qt class at all, which keeps the patch-couple net's bare-set
  rule satisfied with no new exemption). `QFileDialog` was measured before
  shaping the seam: the suite's 7 `QFileDialog` patches bind through
  `app_gui.QFileDialog` (the shared Qt class object, not a module global),
  so they survive any import spelling — but it crosses as a parameter
  anyway, for symmetry with `message_box_cls=`. `_convert_subtitle_input`
  and `_speaker_ref_hint_lines` deliberately stayed (both are source-pinned
  by `tests/test_subtitle_conversion_has_one_owner.py` and
  `tests/test_naming_and_wording_have_one_owner.py` respectively). The
  MOVED_OWNERS record gains `the_oracle.gui_ingest_tools` (3 names). The
  seam is mutation-proven: with the seam intact, an app_gui-level
  `QMessageBox` rebind reaches the moved body's `critical()` (probe exit 0);
  with a bare-global resolution mutated in, the identical rebind is bypassed
  and the probe fails (exit 1) — the exact Vulkan-slice harm made concrete —
  then reverted byte-identically. Verified: cluster net 34/34
  (test_ingest_transformer_gui), one-owner/naming nets 27/27, patch-surface
  12/12 over the amended manifest, payload-policy + import-direction 11/11,
  neighbor GUI nets 126/126; app_gui.py 4099 → 3910 lines.

- **2026-10-08: the tenth extraction slice is picked and armed — the
  ingest-tools handler cluster — pre-flight gates green; the move itself is
  held until the parallel GUI-legibility session's app_gui.py hunks land.**
  Picked from the current 156-method MainWindow (4099 lines) by the
  campaign's own criteria, measured live rather than assumed: the cluster is
  `_run_ingest_transformer_check` (81 lines), `_restore_most_recent_format_backup`
  (69), `_batch_fix_input_folder` (64), `_show_fix_preview_dialog` (40),
  `_convert_subtitle_input` (39), `_show_batch_fix_preview_dialog`,
  `_speaker_ref_hint_lines`, and the trusted-file/backup delegators
  (`_input_file_is_trusted` / `_remember_trusted_input_file` /
  `_remember_format_backup`) — ~350 lines total, at app_gui L1297–1335 and
  L3145–3468. It won on patch-coupling: 2 test files reference it
  (test_ingest_transformer_gui.py, test_subtitle_conversion_has_one_owner.py)
  against 4+ for every other candidate (project lifecycle, Vulkan window
  handlers, settings appliers). Its policy layer already has owners —
  gui_ingest (slice 1) owns the popups/diffs and already receives
  `dialog_cls=`/`message_box_cls=` at call time; gui_settings (slice 2) owns
  trusted-file approvals and backup records — so the window methods are pure
  orchestration, the lowest-risk remaining cluster. The harm-class inventory
  for the move design (measured, not guessed): 17 bare `QMessageBox.*`
  resolutions, 1 bare `QFileDialog.getExistingDirectory`, plus the two
  sanctioned call-time seam sites; test_ingest_transformer_gui.py patches
  `app_gui.QMessageBox`/`app_gui.QDialog` and patches `QFileDialog` 7 times,
  so moved bodies resolving those names as bare globals would be the exact
  Vulkan-slice silent-no-op harm — the move must follow the
  VulkanPreflightThread `preflight_report=` seam precedent (window wiring
  passes the bare names from app_gui globals at call time) or keep thin
  MainWindow wrappers over a Qt-free core, and the patch-surface manifest's
  MOVED_OWNERS gains the new owner. Pre-flight gates ran green on the current
  tree (62 passed / 0 failed, exit 0): test_app_gui_patch_surface.py,
  test_payload_policy_ownership.py, test_gui_import_direction.py, plus the
  cluster's own baseline (test_ingest_transformer_gui.py,
  test_subtitle_conversion_has_one_owner.py). Conflict map kept honest: the
  cluster's line regions do NOT overlap the legibility session's active
  zones (build ~L344–475, ctrl-help ~L716–1041, persistence ~L2500–2560),
  but both sessions' hunks would share app_gui.py, and that session's plan
  explicitly holds staging on mixed-hunk files — so the slice starts here
  and the move executes after their app_gui.py hunks land or on explicit
  coordination.

- **2026-09-29: the omega gate is open — tree clean at `fc7d78c`, full suite
  1529 passed / 0 failed.** The parallel actor's U44 unit landed as-authored
  (`486a59e`, owner-instructed) and the two red nets were settled per the
  V1.3.2 checklist's Step 0 (`fc7d78c`): the export_flac FLAC re-bind hoisted
  to module level, the 1697156 PID token classified in the record-integrity
  net. The campaign cycle runs from this state.
- **Consumer-market readiness campaign — U1 diagnostics landed; U2–U5 remain
  (designed 2026-09-25, executing slice by slice)**: the design record is
  `docs/superpowers/specs/2026-09-25-consumer-market-readiness-design.md` — unit
  decomposition U1 diagnostics (crash/logging/privacy, **landed** — its records
  live in Done) → U2 licensing → U3 GUI surfaces → U4 refinement
  (measurement-first) → U5 release — plus resolutions of every open decision:
  crypto = `pynacl` Option A with offline-bundle evidence, no machine-locked
  seats in v1, trial minting vendor-side, no crash transport, fail-closed
  consent, caps as designed, next-session crash notice. The execution plan is
  `docs/superpowers/plans/2026-09-25-consumer-market-readiness.md` (bite-sized
  TDD steps per unit, mutation-proven pins, one bounded unit per session). The
  built units' design contracts are recorded in Done and their design docs;
  what remains of them is the next bullet.
- **U3 landed (2026-09-28) — the GUI surfaces are built**: `gui_crash.py`
  (first-run consent, next-session crash review, D8 startup branch, and the
  Qt message handler U1.2 deliberately deferred — Qt fatal/critical join the
  consent-gated pipeline) and `gui_license.py` (offline paste-a-token
  activation surfacing the CLI/doctor's typed states; About panel reading
  `current_license()`), wired through a new Help menu. Commit `4db0852`, 21
  click-through-fake tests; the tests caught a real bug in the new activation
  dialog (DialogCode attribute on an injected factory crashed after the token
  was saved). With this, the "remainder of the built units" bullet above is
  fully closed: U1, U2 steps 1–5, U3, and step 7 (PRIVACY.md) have all landed.
  U2 needs the crypto-decision
  ratification flagged in "Noticed" (Option B's own record exists; the spec
  still says pynacl).
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

- **faulthandler net arms only on consent transitions — a normal GUI relaunch
  runs unarmed (found 2026-09-28, off the 08:06:51 GUI segfault)**:
  `enable_faulthandler_catch` is invoked only when consent *changes* —
  first-run accept (`gui_crash.py`), Help-menu re-enable (`app_gui.py`,
  `_open_crash_privacy_dialog`), and `the-oracle privacy-opt-in` (`cli.py`).
  Nothing arms it on a plain `the-oracle gui` launch when consent is already
  on, so the crash-evidence doc's Vulkan watch-item promise ("next occurrence
  lands in `crash_reports/` with capture armed") is currently false for GUI
  relaunches — proven by the 08:06:51 null-call segfault
  (`python[1697156]: segfault at 0 ip 0 … error 14` in the kernel log) leaving
  `crash_reports/native-crash.txt` at 0 bytes. Caveat kept honest: that process
  ran the worktree *including* the other session's uncommitted WIP
  (`app_gui.py`, `gui_chrome.py`, `gui_vulkan.py` dirty), so the segfault itself
  is not evidence against the committed campaign. **Resolved 2026-10-08:**
  `handlers.arm_native_capture` (idempotent, fail-closed) is now armed on
  every GUI launch — called at the top of `gui_crash.maybe_run_startup_flow`
  (covers every `MainWindow` startup, including the already-consented relaunch
  branch that previously returned unarmed) and at the top of `app_gui.launch_gui`
  before `MainWindow()` is built (covers the window-build phase and direct
  entries that skip `cli.main`). Two tests pin it: a plain relaunch with
  consent on leaves the net armed, and a source pin orders the `launch_gui`
  arm before construction. Changes are in the worktree, uncommitted. The
  regression is additionally gated end-to-end: `scripts/crash_hunt.py`'s
  acceptance mode now runs a capture-readiness probe (a fresh child through
  the real `cli.main`) and FAILS the gate when consent is on but the net is
  unarmed — a probe that cannot run also fails the gate, fail-closed. The
  segfault itself is now **closed as unreproducible** (2026-10-08): its exact
  dirty bytes landed only at `2ba2392` (08:56, 50 min post-crash, further
  evolved — never verbatim-shipped), the Sep 28 kernel log is gone to journal
  rotation, and the committed tree survived all three repro angles with the
  net armed (10/10 passive Vulkan-restore launches, a full
  analyze→preview→playback cycle, 40 rapid playback stop/start cycles; 0
  kernel traps, `native-crash.txt` still 0 bytes). Watch stands: the next
  armed kernel trap auto-captures its stack. Details:
  crash-evidence §D, investigation-of-record paragraph.
  **RESOLVED 2026-10-07** — armed exactly as the fix shape prescribed:
  `cli.main` (the single console entry point — every command and the GUI
  route through it) calls `arm_native_capture()` beside `install()`, so every
  launch path comes up armed, not only consent transitions. Idempotence lives
  in the one owner (`enable_faulthandler_catch` early-returns on an existing
  dump handle — no second open, no leak when a launch arm and a
  consent-transition arm land in one session); consent-off launches get False
  and no capture machinery (enable-time contract unchanged; a later opt-in
  still arms). The fix also corrected a latent `install()` hazard the new
  in-suite launches exposed: remember-once semantics held `root_override`/
  `log_file_override` via `setdefault`, so a no-root install poisoned a later
  explicitly-targeted install in the same process — explicit roots now always
  win. Four new tests (fail-closed arming, idempotence, consented
  relaunch-arms through real `cli.main`, one-owner pin with vacuity guard);
  mutation-proven (removing the launch arm call fails 2, removing the
  idempotence early-return fails 1), reverts byte-identical.
  **Closed 2026-10-08: the last unarmed process class** —
  `render_subprocess.main` (the GUI's spawned clean render interpreter,
  `python -m the_oracle.render_subprocess`) now arms the same way beside its
  job dispatch, so a native segfault inside the child itself — the exact
  class the GUI delegates there to escape — lands in this checkout's
  `crash_reports/` too; the parent's arm cannot see across the process
  boundary. Same contract: idempotent, fail-closed without consent. Three
  new tests (`test_render_subprocess_arming.py`: consented arm before a real
  job run, consent-off creates nothing, one-owner order pin); mutation-proven
  (arm removal fails 2, consent-gate removal fails 1).

- **RESOLVED 2026-09-27** — `test_gui_vulkan_imports_nothing_from_app_gui`
  only saw absolute spellings (found 2026-09-27): that guard checked
  `alias.name == "app_gui"` / `node.module == "app_gui"` bare, so it would
  pass green over `from the_oracle import app_gui`, `from . import app_gui`,
  and every `gui_*` sibling import in `gui_vulkan`. Closed by the Vulkan
  thread-cluster extraction slice (holding commit 5d26f1a): the test now reuses the
  `tests/test_gui_import_direction.py` scanner (`_scan_imports` with a
  gui_vulkan-specific predicate — gui_utils stays legal, app_gui does not),
  with full spelling coverage and per-form proofs; the lazy function-level
  `from the_oracle import app_gui` mutation the old check would have missed
  is now caught (mutation-proven in the slice).

- **Campaign crypto decision D1 vs. the built licensing unit (2026-09-25)**:
  the campaign design resolved token crypto to `pynacl==1.5.0` (Option A),
  but the licensing unit concurrently built in this checkout chose vendored
  pure-Python Ed25519 (Option B) and recorded that as DECIDED in
  `docs/LICENSING_DESIGN.md` §3, STATE, and JUNO_FIXES (no new dependency,
  offline bundle unchanged, 61 mutation-proven tests). The spec's D1 fallback
  clause anticipated Option B "with its own record" — that record now exists.
  Owner ratification asked for (accept Option B and repoint the campaign spec,
  or hold U2 and revert to pynacl); nothing in the spec was changed
  unilaterally. **RATIFIED 2026-09-28: owner accepted Option B; the campaign
  spec's D1 now records the ratification and points at the built unit.** Also noted: one transient full-suite failure while the
  licensing files were being written mid-run did not reproduce (their suites
  pass: 61, plus 11 in the two files outside the six I first ran).

- **Hardware-floor claim vs. documented floors (2026-09-25)**: the sales claim
  driving this campaign is "minimum 3 GB of free and available VRAM/DRAM"; the
  repo documents a **4 GiB Chatterbox CUDA suitability floor** (the Inference
  Setup tour and the CUDA picker both reject cards below it — README.md:276,284,
  STATE.md:801), while CPU/system DRAM is the guaranteed fallback. Both may be
  true (different resources), but they are not yet reconciled in one place and
  sale copy must not overclaim. RESOLVED 2026-09-28 by U4.5's evidence pass
  (4bcaf6d; plan box carries the full digest): the 3 GB figure is an overclaim on the
  VRAM side (the enforced floor is 4 GiB, render-time enforced too) and an
  unenforced assertion on the DRAM side (nothing in the code measures DRAM;
  CPU is unconditionally available).  Decision presented to the owner: drop
  the numeric claim from sale copy in favor of README's exact framing
  (optional NVIDIA 4+ GiB VRAM for CUDA; CPU/DRAM guaranteed fallback); DRAM
  guidance stays a recommendation until U4.3 measures the real CPU memory
  profile. **RATIFIED 2026-09-28: claim dropped.** No sale copy exists in-repo
  (the "3 GB" figure lives only in campaign records), so the ratification
  required no file change beyond this record and the spec's item 8.

- **`.ps1` executable bits are inconsistent (2026-09-18)**: `bootstrap_oracle_tts.ps1`,
  `doctor_oracle_tts.ps1` and `oracle.ps1` are tracked 100644 while
  `install_oracle_tts.ps1`, `run_oracle_tts.ps1` and `uninstall_oracle_tts.ps1` are
  100755. This is *not* fixed alongside the `./oracle` defect because unlike that one
  it has no failure mode: PowerShell is invoked as `powershell -File .\script.ps1` (as
  the Windows CI job does) and never by a shebang, and `git` on Windows does not track
  the bit at all. Making them consistent would be tidy, not a fix, and would touch six
  files' modes for no behavioural gain. The acceptance script's wrapper matrix is
  `.sh`-only for the same reason it is Linux-gated.

- **`pkg_resources` deprecation warning (2026-09-25)**: the suite emits 18 warnings
  including `pkg_resources.resource_filename` coming from a dependency (not Oracle code
  — repo-wide scan finds no `pkg_resources` import in `src/` or `scripts/`). Harmless
  today; worth one look when the pinned dependency that imports it is next touched, since
  a future setuptools major could turn it into an error. 2026-09-28: culprit pinned —
  `perth`'s `perth_net/__init__.py` line 1 does `from pkg_resources import
  resource_filename` at module level (verified in `.venv`), so the warning fires whenever
  the watermarker is imported. The fix belongs upstream (or a `warnings.filterwarnings`
  in the suite if it ever becomes an error); no Oracle code change warranted.

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
  showed zero new leaks (JUNO_FIXES.log 2026-09-20; holding commit fd302d7).

- Serpent-circle inventory scan updated (2026-09-08, in
  `~/.agents/skills/serpent-circle/`): the bloat scan and language
  histogram now honor `.gitignore` — untracked+gitignored residue
  (bytecode, `.venv/`, vendored clones) is treated as repo-declared
  retention, while tracked junk and non-ignored strays stay flagged. This
  lets the omega loop terminate **CONVERGED** instead of NO-PROGRESS on a
  fresh campaign; the change lives in the skill (outside this repo), so
  it is documented here rather than committed here.

- **2026-10-09 (adversarial pass over 8e932bd, noticed not actioned): the
  last context-less deferred callback on MainWindow.**
  `open_recording_studio` still arms
  `QTimer.singleShot(150, self._start_recording_wizard)` — the exact
  strong-reference pattern that commit converted for the three startup
  timers. Harm today is bounded: it pins the window wrapper for 150 ms and
  `_start_recording_wizard` early-returns when the studio set is empty
  (MainWindow.closeEvent closes and discards every studio first), so the
  observed replay is benign. If it is ever touched, give the call the
  context overload `QTimer.singleShot(150, self, ...)` or a child timer,
  like the startup three.

## Deferred (intentional)

- **The 2026-10-08 native-crash audit closed every script-entry capture
class**: the doctor's probe children and the three real-MainWindow script
entries now arm the faulthandler catch like every launch path does
(scripts/crash_hunt.py GUI-leg children, windows_install_smoke,
vulkan_ci_smoke, fresh_clone_acceptance were already covered through
cli.main; runpy-in-process smoke/baseline scripts need no second handle —
same process, same file). Mutation contract: tests/test_entry_point_arming.py
fails per-site on any arm removal or reorder.

- **Recording Studio takes and the user's teleprompter script kept —
  dispositioned KEEP, gitignored (2026-09-28)**: four Recording Studio takes
  (`Seashells/Seashell_No_2.wav`, two `Seashells/*fransisco*.wav` takes and
  their v5 cleaned render) and `Input/recording prompt.txt` (the teleprompter
  working script) are the user's personal recording artifacts, not catalog
  voice references (the tracked `Seashells/*.wav` references stay tracked).
  Ignored alongside the `Seashell_No_1.wav` precedent so the release tooling's
  dirty-tree gate stays meaningful for repo files without ever touching the
  user's files. Disposition: KEEP, do not clean, do not track.
- **`Output/render_plan.json.bak` kept — omega residue finding dispositioned
  KEEP (2026-09-25)**: serpent-circle's inventory has flagged this untracked
  file twice (campaign 7, cycle 2; fresh-campaign preview). It is a render-plan
  backup for "What is, reality" (2026-09-21, 27 KiB) in the user's Output
  folder — the user's own artifact, not repo debris. Disposition: KEEP, do not
  clean, do not track. Deleting a `.bak` on the user's behalf is exactly the
  irreversible tidy-up this repo's rules forbid, and Output/ is runtime data
  that never ships. Any future omega cycle treats this as a documented keep,
  not an open finding.
- **`JUNO_FIRST_PROMPT.txt` kept — omega residue finding dispositioned KEEP
  (2026-09-25)**: flagged by the same serpent-circle inventories. Untracked,
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
  **RESOLVED 2026-09-28 (amended 2026-10-05 after the records audit): the
  repro happened and the crash is root-caused and fixed.** The scripted
  render → preview → render with playback active reproduced it rc=139;
  `crash_reports/native-crash.txt` named the fingerprint — first import of
  `mutagen` inside the render worker thread racing the main thread through
  shiboken6's import hook — and the fix (`ed6ef8b`: worker-call-graph
  main-thread preloads + mutation-proven regression net
  `test_gui_render_import_safety.py`) closed the lineage this evidence names:
  post-fix kernel crashes = 0, the scripted repro passes green, and the
  campaign's 11-process acceptance gate is green (full write-up:
  `docs/superpowers/findings/2026-09-28-crash-evidence.md` §A).
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