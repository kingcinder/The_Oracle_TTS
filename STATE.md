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
  **Suite of record 2026-10-11: 1799 passed / 0 failed / 0 skipped, exit 0,
  465 s** — the full suite run on the tree of the three slices below
  (`d5cc554` the GUI work, `58cf7c9` the certifier family, `8c863a3`
  mutation_probe) on this machine -- the tree those three slices were
  committed on. The 1592 figure above stays correct as the count at
  `ec136c8`; this entry records the increment the campaign's twelve landings
  produced, measured rather than summed.

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

- **2026-10-10: the theme certifier's section sweep now covers every
  registered section — `paths`/`table`/`extra_voices` added, and the sweep
  loop made correct for them.** `scripts/certify_gui_themes.py` pinned a
  five-key tuple (`shared`, `speaker_a`, `speaker_b`, `status`, `live`) while
  `_section_registry` had grown to eight keys: the three sections the
  2026-10-08 legibility pass gave chrome to were registered but unswept, so a
  regression visible only in their chrome passed this certifier. Extending
  the tuple alone does NOT work, and each reason is measured rather than
  assumed: (1) `extra_voices` is hidden by design until a cast beyond A/B
  exists and the redistributer deliberately skips hidden panes, so its slider
  legitimately cannot move splitter space (pinned by
  `test_hidden_extra_voices_pane_does_not_break_size_slider`) — a hidden pane
  is now exempt from the drive, its slider-while-shown behavior staying
  certified by the sibling `certify_section_chrome.py`, which shows the pane;
  (2) the incumbent loop drove one slider and then collapsed that section
  before moving on, which measures leftover state instead of launch geometry
  (a collapse clips its pane to 30 px and the splitter keeps that size on
  expand — the failure the sibling certifier reproduced on the Review pane,
  floor-locked at [174, 104]) — the sweep is now two passes, every slider
  driven from launch geometry first, then every collapse; (3) a
  single-direction slide has a no-op regime in each branch (75 matches the
  launch geometry when the pane already sits there, 30 clamps silently to a
  content floor), so the drive now requests both directions and passes iff
  one moves, restoring the launch geometry after each drive. Collapse restore
  is asserted too, because the persistence step that follows must snapshot an
  uncollapsed layout. Acceptance: the full six-theme run exits 0 (all six
  report contrast body and on-accent ratios, no truncation, legibility
  certified, persistence verified), and the coverage is mutation-proven —
  popping `paths` out of the registry before the sweep fails the run with six
  `missing section paths` findings (exit 1), reverted byte-identically; the
  five certifier-pinning test files pass 37/37 (exit 0), and the run leaves
  no parked crash-records store behind.

- **2026-10-10: the six-theme section-chrome probe is persisted as a tracked
  CI gate — `scripts/certify_section_chrome.py`, two workflow steps, two
  pins.** **(SUPERSEDED 2026-10-10: this script was merged into
  `scripts/certify_gui.py`; the pins named in this entry were repointed, not
  dropped — see the merge entry below.)** The legibility pass's offscreen verification lived in a one-off
  /tmp probe; it now lands as a standalone certifier that builds the REAL
  MainWindow for all six themes and checks the three chrome'd sections
  (registry presence, chrome titles, non-degenerate geometry, size sliders
  moving splitter space, collapse clip/restore, the hidden-by-design Extra
  Voices behavior, the populated Review-table viewport, and the paths
  splitter persistence round-trip). CI runs it as a named step on both
  operating systems (pinned by
  `test_workflow_runs_the_section_chrome_sweep_on_both_oses`), and the
  entry arms the faulthandler net before its first Qt construction (pinned
  by `test_certify_section_chrome_arms_before_mainwindow`; both pins
  mutation-proven). Persisting it surfaced four real problems, each fixed
  with its mechanism recorded: (1) the D8 startup crash flow is
  machine-state modal — on a fresh checkout (no consent decision recorded)
  it opens first-run consent and blocks any unattended sweep; the script
  stubs `_maybe_run_crash_startup_flow` and `_start_inference_wizard` per
  window before the first event pump (both timers connect inside
  `_on_gui_shown`, so instance attributes set right after construction are
  what they capture — the test_app_gui_profiles idiom). (2) A settings
  backup/restore dance cannot survive a SIGSEGV (finally never runs; two
  probe runs really did clobber the developer's real
  `~/.config/the_oracle/app_settings.json` before the isolation landed —
  restored from the oldest /tmp backup, only theme/geometry had differed):
  the run now points XDG_CONFIG_HOME AND APPDATA at a temp dir before any
  the_oracle import (the isolate_user_config contract — XDG alone still
  touches %APPDATA% on Windows), so even a crash leaves nothing to
  restore; real settings verified byte-identical across clean runs. (3)
  The single-direction slider check had a no-op regime in each branch —
  75% matches the geometry when a pane already sits at ~75% (the slider
  only re-syncs from splitterMoved), 30% clamps silently to the Review
  table's 140 px + 34 px chrome = 174 floor — so the check now drives
  both directions and passes iff one moves (a dead slider moves neither).
  (4) Interleaving slider checks with collapse checks measured leftover
  state, not the launch layout: a collapse clips its pane to 30 px and the
  splitter keeps that size on expand, and on a clean HEAD checkout that
  left the Review splitter exactly floor-locked [174, 104] — no slider
  value moves a splitter already at its combined minimums (caught only by
  validating the artifact against a `git archive HEAD` clean tree, not
  the working tree). Checks now run in two passes (all geometry/slider
  checks from launch geometry, then the collapse checks), and windows are
  torn down on the conftest discipline (stop the startup timers, honor
  closeEvent's refusal, deleteLater + DeferredDelete flush) — before it
  the probe segfaulted 2 of 3 runs (exit 139, faulthandler silent, the
  gui_tooltips teardown class), after it 8/8 clean full sweeps across
  both trees. Noticed, not actioned: the shared `_assert_armed_before`
  textual pin matches the arm call left inside a comment (`pass  #
  crash_handlers.arm_native_capture()` passes it) — pre-existing for all
  arming pins, deleting the line is caught; hardening the shared helper
  is a separate unit of work. Validation: the sweep itself 8/8 exit 0
  after the fixes (6 working-tree, 2 clean-HEAD); arming/chrome/smoke/
  record-integrity nets 43 passed; full suite 1670 passed / 0 failed /
  0 skipped, exit 0 — that number includes the concurrent session's
  in-flight recording-wizard-timer and apply_theme work (its own 4 tests,
  left uncommitted for its owner).

- **2026-10-10 (second landing): both certifiers now park the
  crash-records store, so the D8 startup modal can never block them.** The
  D8 branch opens a modal whenever `bundle.list_records` is non-empty — a
  QDialog.exec no headless sweep can answer — and the earlier offscreen
  sessions had worked around it by hand-moving `crash_reports/` out of the
  repo root (once leaving the store misplaced). `bundle` (the store owner)
  gained `park_records` / `restore_parked_records`: the directory is
  renamed to `crash_reports.certifier-park-<pid>` in the repo root (in the
  root, not /tmp, on purpose — a SIGSEGV skips every `finally`, and a park
  a later run CAN find is a park it can bring home), and the restore
  merges rather than renames, so a record written DURING the parked window
  (a genuine crash in the sweep itself, the Qt fatal handler) survives
  beside the parked originals; a name colliding on both sides is
  byte-compared (identical duplicate dropped, differing copy preserved as
  `.parked`) so parking may hide evidence but never destroy it. The next
  run's `park_records` recovers any crashed predecessor's park first — the
  mislaid-evidence failure mode self-heals. `scripts/certify_gui_themes.py`
  also gained the per-window `_maybe_run_crash_startup_flow` stub at all
  three of its MainWindow sites (the park kills the records-review branch;
  a fresh checkout with no consent decision would still block on the
  first-run ask — both layers needed). Pinned by
  tests/test_crash_records_parking.py (7 tests: hide + byte-identical
  restore, no-store noop, raise-path restore, crashed-predecessor
  recovery, collision preservation, both-certifiers source scans with an
  order pin, and a vacuity guard on the scan literals). Acceptance on the
  REAL store: a synthetic record planted in the repo's crash_reports —
  the exact shape that blocked an unparked sweep in the earlier sessions —
  and both certifiers completed exit 0 with the record byte-identical
  afterwards and zero park dirs left; the synthetic record was then
  removed (native-crash.txt untouched). `.gitignore` gained the park
  pattern so a crashed run's leftover is untracked noise, not a dirty
  tree. Validation: parking/crash/arming/chrome nets 80 passed; full
  suite 1680 passed / 0 failed / 0 skipped, exit 0 — including the
  concurrent session's newer in-flight tests, as before.

- **2026-10-10: suite-count drift is a check now, not a memory —
  `scripts/certify_suite_counts.py` + `scripts/suite_counts_baseline.json`.** The
  counts were already what caught this repo's skip drift twice (the Vulkan
  smoke's skip count moving 1 → 3 as more audio-gated tests were added), but both
  times a human compared two runs in their head, and a number only a human
  remembers is not a check. `python scripts/certify_suite_counts.py [REF]`
  checks a commit out into a fresh detached worktree, asserts the tree under test
  imports its OWN code (the editable install points at the working tree, so a
  silent wrong-tree load would report counts for code that never ran), runs the
  full suite, and compares it against the counts recorded for that commit: exit 0
  match, 1 drift (both values, plus any skip id that appeared or disappeared), 2
  could not run — and an unrecorded commit is 2, never a pass (the one exception,
  where a recorded commit checks out the same suite, is the inheritance below).
  `--record` writes
  the counts for the resolved sha and prints every field it changed, so
  re-baselining is visible rather than reflexive; `--capture-output` keeps the
  run's raw bytes so a mismatch can be read instead of re-run.
  **A worktree and not a `git archive`, because measurement said so**: an archive
  has no `.git`, and four files of this suite need one — an archive of HEAD
  reports 1 failed / 7 skipped (`test_commit_slices`' landed-fix pin clones
  `file://HEAD` under `check=True`; `test_record_integrity` ×3 and `test_release`
  ×1 skip without a repository) where the worktree reports 0 failed / 3 skipped.
  Recorded for `40b8ddbd6058`: **1684 passed / 3 skipped / 0 failed / exit 0**,
  the 3 being the audio-gated smokes the absent native build explains; three
  independent fresh-worktree runs of that commit agree, and the verify path exits
  0 reporting "reproduces its recorded suite counts".
  **Re-recorded 2026-10-11 for the pushed tip `dbee3289628`: 1796 passed / 3
  skipped / 0 failed / exit 0** — the same 3 audio-gated skips, and the count
  moved 1684 → 1796 because the campaign's own committed tests and gates ride
  that tip, and the entry is committed and pushed — it rode the recording commit
  that is now the branch tip, with `dbee328` as its parent, so the reference point
  reaches a clone rather than living in one working tree. An entry keys a commit,
  so the commit that *carries* the entry can never be recorded — a tree cannot
  contain a hash of its own content — and **that gap is closed by inheritance
  rather than by a run** (2026-10-11, this change): when nothing is recorded
  under the target's sha, the check looks for a recorded commit whose tree is
  identical apart from `scripts/suite_counts_baseline.json`, because the suite
  reads that file in exactly one place — test_suite_counts.py's shape audit — and
  both files pass it. So `certify_suite_counts.py` with no args at the carrier tip
  now exits 0, reporting `PASS (inherited)` with `dbee328` as the witness and
  running no suite and building no worktree (the comparison is a two-tree
  `git diff`, and "could not compare" is never read as "same tree"); a commit
  that really changed the suite still exits 2, and a baseline file whose entries
  the audit would reject inherits nothing — the guard, pinned from both sides.
  `--record` still measures: it never inherits a copied entry. 7 new tests, 38 in
  the file. The ancestors under the recorded commit stay
  unrecorded on purpose too, so an older commit gets recorded on demand rather
  than by pre-emptively running six full suites.
  **Re-verification is not deterministic yet, measured:** three fresh-worktree
  full-suite runs at this tip gave one green (the recorded 1796) and two failures,
  both the chrome aspect of the merged certifier dying with `rc=-11` after 1795
  passed. The entry is from the green run; the crash is logged under Noticed
  rather than folded into this record. Drift was proven to fail
  rather than assumed: `--pytest-args "-k test_skip_audit"` exits 1 naming
  `passed 1684 → 27` and `skipped 3 → 0`. Skip-id comparison is gated on the audit
  being complete, so a truncated list can neither invent drift nor hide behind
  one (see the Noticed entry on the ledger); the counts are compared either way.
  Refusals and paths leave no half-state either, both fixed after measurement:
  the not-added path no longer "removes" a worktree it never created (it had
  shipped a spurious WARNING naming an unregistered path), and `--scratch` now
  creates the directory it was pointed at instead of dying in a
  `FileNotFoundError` traceback whose exit code 1 collided with the code reserved
  for drift; both `--record` branches are exercised live (a first record prints
  every field it wrote, a repeat says "already matches; nothing changed"). Pinned
  by `tests/test_suite_counts.py` (31 tests: summary parsing, the audit's skip
  ids, count/exit-code/id drift in both directions, ids compared as a set not a
  list, baseline round-trip and shape, the unrecorded-commit refusal with no
  leftover scratch, a scratch directory that does not exist yet, worktree
  acquisition at a named ref with clone-ability and removal reporting, and both
  fresh-tree checks'  import assertions agreeing).

- **2026-10-10: the theme-switch probe is retired and its measurement is a
  tracked, CI-wired certifier — `scripts/certify_theme_switch.py`.** The
  investigation ran as a temp harness (`tests/_theme_switch_probe.py`, v5 = an
  interleaved A/B of the restyle gate) and its findings landed as `bf591a5`; the
  harness itself was untracked draft litter, and its source has since been
  deleted (a stale `__pycache__` entry is all that remains). The measurement is
  now permanent, splitting the switch cost into the three terms the probe
  identified plus the one that was ours: **structure** (354 widgets, depth 13 on
  the real MainWindow — the term nothing here can shrink), **propagation**
  (16 210 change events inside one `setStyleSheet`, 18 506 by the end of the
  pump: ~83% of the fan-out is synchronous inside the call), **rule matching**
  (59 rules fan out 16 210 change events against 10 806 for a 2-rule stylesheet on
  the same tree), and **ours** (the app's own Ctrl+hover filter is invoked 18 506
  times during a real `apply_theme` switch and finds the bulk-restyle scope open
  on EVERY one of them, where the ungated `setStyleSheet` path — what the code did
  before the gate — finds it closed for all 18 506). Only the gate is asserted as
  a contract; the rest are floors and relationships, and **timings are printed but
  never asserted** (of this machine: 619 ms ungated / 531 ms gated / 435 ms for the
  2-rule sheet / 0 ms for a redundant re-apply), pinned by
  `test_timings_are_never_asserted` so a slow runner cannot fail the step.
  Mutation-proven live, each reverted byte-identically (`gui_themes.py` hashed
  before and after every mutation, the worktree each time returning to HEAD's own
  bytes): dropping the `with bulk_restyle()` from `apply_theme` fails with
  "saw 18506 of its 18506 events … with the bulk-restyle scope CLOSED (must be
  0)"; dropping the redundant-restyle early return fails with "re-applying the
  current theme fanned out 16196 events"; gutting `build_stylesheet` to two rules
  fails four ways (the rule floor, the rule-matching separation, "not one gated
  round performed a genuine switch", and the now-vacuous gate floor — the last two
  after a live `StatisticsError` in the timings path was found and fixed, a crash
  that would have read as a finding). Where the unit pins stop is stated in the
  file itself: the suite drives `apply_theme` with a duck-typed app stand-in and a
  synthetic event, so this certifier is the only place the scope is checked across
  a REAL Qt fan-out. CI gains one step per OS beside the GUI certification
  sweep (`.github/workflows/ci.yml`, "Theme Switch Restyle Split", Linux +
  Windows; **amended 2026-10-10: the neighbour step was renamed to "GUI
  Six-Theme Certification" when the two theme certifiers merged into
  `scripts/certify_gui.py`**).
  Pinned by `tests/test_theme_switch_certification.py` (23 tests: every verdict
  driven just over its line, timings-never-asserted, the terms named in the
  contract, the config redirect ahead of every `the_oracle` import, the paths drain
  and the no-repo-log entry points, the two CI steps, and a subprocess run that
  exits 0 having left the real repo tree AND the real config root byte-identical).
  **Not resolved here, on purpose**: the open Noticed item about
  `certify_gui_themes.py` being unwired — that certifier is still invoked by
  nothing but a human, so this landing adds a second CI-wired theme artifact
  without closing that gap.

- **2026-10-10: the design-token and section-chrome certifications are local
  gates now — `pytest` runs both, so a theme regression fails a developer's
  run instead of the gate.** **(SUPERSEDED IN SHAPE THE SAME DAY, NOT IN
  SUBSTANCE: the two certifiers merged into `scripts/certify_gui.py` with
  `--aspects`, so the two gates here now drive its two aspects —
  `test_gui_theme_certification.py` runs `--aspects contrast` (~8 s) and
  `test_gui_sections_chrome.py` `--aspects chrome` (~14 s), against the 49 s
  the two separate scripts cost together. Everything else below still holds:
  the verdict is every theme's own summary line, the parsers stay pinned, and
  the hermeticity assertion is unchanged. See the merge entry.)** Both certifiers were CI-shaped: named workflow
  steps a runner or a human's shell invoked, with the suite touching
  `certify_section_chrome.py` only for a *different* claim (the hermeticity
  contract runs it and asserts its exit code as a side effect) and
  `certify_gui_themes.py` — the six-theme design-token / contrast /
  no-truncation / persistence certification — executed by NOTHING but a human
  (the Noticed entry below, now amended). Each certification's own module
  gained one slow test that runs its script as a real subprocess exactly as CI
  does: `test_design_token_certification_runs_locally_and_passes` and
  `test_section_chrome_certification_runs_locally_and_passes`. Exit 0 is
  necessary but not sufficient, because a certifier that printed nothing and
  returned 0 would look exactly like a clean sweep: every one of the six themes
  must appear in the certifier's OWN summary lines — parsed for the theme
  certifier into the (body, on-accent) ratios it reports, which must meet the
  4.5/3.0 the certification promises, and compared as a key set against
  `THEMES` for both — and the closing summary line must be present. Each parse
  is pinned by a unit test that runs a truncated and an empty report through it
  and requires FEWER keys, so the gate cannot pass vacuously.
  **The theme certifier had to be made hermetic first, and that was the real
  work**: running it per-developer-run was unsafe as it stood, measured — the
  run rewrote the developer's REAL app settings file once per theme (a backup
  copy and restore in `finally`, so a SIGSEGV mid-run lost the user's theme,
  layout and last input — the same class the sibling's hermeticity work was
  about) and wrote `Output/logs/gui_startup_timing.json` and
  `gui_prewarm_timing.json` into the repo. It now carries the sibling's three
  isolation layers — `XDG_CONFIG_HOME`/`APPDATA` redirected to a throwaway
  `mkdtemp` before any `the_oracle` import, `app_gui.ensure_repo_default_paths`
  repointed at it so every `Output/logs` write drains there, and
  `crash_handlers.install(root=<throwaway>)` before the launch-path arm with the
  user's EXISTING consent decision mirrored byte-for-byte (never created from
  nothing) — and the backup/restore of the real settings is deleted rather than
  kept as a second net, because isolation has no window in which a crash
  matters. Verified around a full run: rc=0, six themes, the real repo tree AND
  the real app-config root byte-identical (content and mtime), and the
  throwaway removed afterwards.
  Mutation-proven locally, each reverted byte-identically (`sha256` of every
  touched file compared against HEAD after the restore, and the *runtime* value
  checked, not just the file hash — see the caveat below): (1) `tape_deck`'s
  `text_muted` set to its own panel colour fails at IMPORT, from `gui_themes`'
  own `_certify` ("text_muted on panel = 1.00:1 … panel_alt = 1.15:1 … bg =
  1.16:1"), i.e. that whole class was already local before this landing — worth
  knowing, and the reason the gate's own teeth are proven by the other two;
  (2) collapse no longer clipping the pane to its header fails BOTH new gates,
  the certifier naming "did not clip when collapsed" per section per theme;
  (3) the Theme menu dropping one of its six entries — `list(THEMES)[:5]` —
  passes the fast half of the certification module (5 passed) and no test in
  `tests/` names the theme menu at all (grepped), while the new gate fails
  naming "Theme menu should list 6 themes, got ['Studio Light', 'Night Deck',
  'Sephiroth', 'Pulp Science Fiction', 'Tape Deck']". That third mutation is
  the coverage this landing buys: the contract is now enforced on a developer's
  run and on every push, not on no push.
  **Cost, stated rather than discovered later**: +36 s for the theme
  certification run and +13 s for the section-chrome one, the latter because
  `test_certifier_hermeticity.py` already runs that same 13 s sweep for its own
  claim — deliberate duplication, disclosed in the new test's docstring: if the
  hermeticity contract is re-scoped or deleted, the chrome gate must not vanish
  with it.  Both new tests carry the `slow` marker the repo uses for real-window
  work; the README's documented local command is a bare `pytest`, which runs
  them, while `-m "not slow"` deliberately does not.
  **AMENDED 2026-10-11**: the markers are gone — the fast subset runs them now,
  and pyproject's marker description records that the certifications are
  deliberately outside the marker's scope. The sentence above describes how it
  stood for one day; see the entry above for the measured numbers.
  **Measured caveat, worth keeping**: replaying a mutation exposed that
  Python's pyc invalidation is (source mtime, source size) — restoring a
  same-length file inside the same second as the mutated compile leaves the
  cached bytecode looking current, so a second gate run loaded the MUTATED
  module while every file hash said clean. It was caught by asserting the
  runtime value (`THEMES['tape_deck'].text_muted`) instead of trusting the
  diff, and the invalid run was thrown away and redone with mutated runs
  isolating bytecode via `PYTHONPYCACHEPREFIX`. Nothing in the repo is wrong
  here — it is stdlib invalidation — but a mutation run that restores quickly
  can lie, which is the same lesson as the vacuous-pass assertions above.
  **AMENDED 2026-10-11**: that hand-repair is the default now —
  `scripts/mutation_probe.py` runs a mutation under fresh per-run bytecode
  isolation with hash guards, and its own tests reproduce this collision on
  purpose (see the entry above).

- **2026-10-10: the two GUI certifiers are merged into one parametrized
  six-theme harness — `scripts/certify_gui.py`; `certify_gui_themes.py` and
  `certify_section_chrome.py` are deleted.** Both built their own windows and
  repeated the same six-theme loop; the successor builds one window per theme
  and runs both check families against it, selected by `--aspects
  contrast,chrome` (default both). The suite's two local gates run one aspect
  each (contrast from `test_gui_theme_certification.py`, chrome from
  `test_gui_sections_chrome.py`) and CI runs the default as ONE step per OS
  ("GUI Six-Theme Certification"; the two "Section Chrome Six-Theme Sweep"
  steps are gone).
  **Measured, not assumed.** `scripts/certify_gui.py` (both aspects) is 14 s
  and green; the scripts it replaces cost 33 s + 13 s = 49 s on the same tree
  for the same 13 windows. The speedup is the teardown discipline, and that
  was proven rather than guessed: adding the inference-wizard stub to the old
  theme certifier — the one change that was NOT teardown — made it SEGFAULT
  (rc=139) at 22 s on this machine, because it destroyed windows with plain
  `close()` + `processEvents()` and left them pinned, replaying every deferred
  startup callback on each pump. The merged harness uses the
  `_destroy_window` discipline both old docstrings describe. That crash left
  its parked records store in the repo root; the next harness run merged it
  home — the self-healing path the parking entry above describes, exercised
  live rather than argued.
  **Coverage parity is proven by mutation against a baseline taken BEFORE the
  files were deleted.** With collapse made not to clip (`gui_sections.py`),
  the merged harness reports 48 failures and the old theme certifier's 48
  failure messages are byte-identical to it (`diff` empty), with none of the
  old chrome certifier's 18 missing: the union, not the intersection. Every
  check message was carried over verbatim so this comparison stays possible.
  **The certification entry is the pin this was asked for.** One explicit
  `ThemeCertification` per theme (body bar, on-accent bar, the sections it
  must render) — literals, never a comprehension over `THEMES` — audited in
  both directions BEFORE a window is built: a theme with no entry fails naming
  it, proven by appending a real seventh theme (`THEMES["seventh_probe"]`),
  which fails in 2 s with **zero bytes of stdout certified**; and an entry
  naming a theme that does not exist fails naming that too. Section drift is
  closed the same way — a registry key no entry covers fails the sweep, proven
  by dropping `live` from `ALL_SECTIONS`; the old script carried that
  expectation as a COMMENT, which is exactly how paths/table/extra_voices rode
  uncertified for a while. The usage path is pinned as well: an unknown aspect
  exits 2 having certified nothing, and `--help` lists both aspects. One
  assertion was ADDED while merging: the contrast aspect now checks
  `window._current_theme == key`, so the printed ratios certify the theme
  actually on screen — probed first (a fresh window really does set it), so it
  is a real check and not a false failure.
  Pins repointed rather than dropped: the arming pins collapsed into
  `test_certify_gui_arms_before_mainwindow`, the parking scan's `CERTIFIERS`
  tuple and docstring name the one harness, the hermeticity contract points at
  it (its source anchors were written against the section-chrome file and hold
  verbatim, because the isolation layers were copied whole), and the CI-step
  pin names the new steps. 74 tests pass / exit 0 across everything that names
  either retired file, including the sibling theme-switch certifier and the
  records net. Entries above that mention the deleted files as background are
  left as dated history except where they described present state, which is
  amended.
  **Not done, on purpose**: no Windows run — the checks are OS-independent but
  `scripts/certify_gui.py` has only ever executed on Linux, which its docstring
  says rather than implies.

## Next


- **2026-10-11: the certifier family's hermeticity preamble is ONE module
  now — `scripts/certifier_isolation.py` owns the throwaway root, both
  environment redirects and the reap, and the copies that lived in the two
  certifiers are gone.** The two preambles were byte-for-byte twins: a
  `tempfile.mkdtemp`, the same two `os.environ` writes, and the same reap loop
  (five attempts, `0.1 s` pause, the same survivor warning) registered the same
  way with `atexit` — the policy had two owners and therefore two places to be
  right, which is the duplication this landing removes. The helper is
  deliberately Qt-free and `the_oracle`-free and stays so (it must be importable
  before the config root is known and reachable at interpreter exit, when the
  C++ application may already be gone), so the ONE rule callers cannot delegate
  is unchanged and still theirs: the redirect has to happen before the first
  `the_oracle` import. Both call sites keep their own root prefix
  (`oracle_gui_cert_cfg_`, `oracle_theme_switch_cfg_`) so a leaked root still
  names the sweep that leaked it, and the two source-scanning pins that named
  the removed lines (`tests/test_certifier_hermeticity.py`,
  `tests/test_theme_switch_certification.py`) were repointed, not deleted.
  **New owner-side net, `tests/test_certifier_isolation.py` (8 tests):** both
  variables asserted against the real reader (`platform_support.user_config_root`)
  rather than against the strings; the package-free claim proven in a FRESH
  interpreter (this process has `the_oracle` loaded for other reasons); the reap
  idempotent (a reaped root is success, not an error); a chmod-locked root
  reported on stderr with `False` rather than swallowed; `atexit` measured, not
  assumed, by a child that dies of `SystemExit(2)`; and the dedupe itself pinned
  by AST SHAPE — a re-grown `mkdtemp`/`rmtree`/`atexit.register`/
  `os.environ[...] = ...` in either certifier fails, because a textual scan would
  flag the prose that documents the rule they no longer implement.
  **Six mutations, all caught, every restore sha256-identical** (the probe from
  this morning's landing, first use here): dropping `APPDATA` from the tuple,
  making the reap return `True` while the root survives, unhooking `atexit`,
  re-growing a `mkdtemp` in `certify_gui.py`, importing `the_oracle` from the
  helper, and generifying a root prefix. **One of those six SURVIVED the first
  pass and the finding is the useful part**: the redirect test looped over
  `isolation.THROWAWAY_ENV_VARS`, so deleting a member from that tuple deleted it
  from the test's own expectation too and all eight tests stayed green — the
  expected set was being read off the code under test. Fixed by naming the two
  variables in the test and re-run: caught, naming that test. **End-to-end, on the
  refactored scripts**: both real sweeps rc=0 (`certify_gui.py`, and
  `certify_theme_switch.py --quiet`), zero new roots in `TMPDIR`, zero survivor
  warnings, and `git status --porcelain` byte-identical before and after (the
  repo writes the redirect exists to prevent did not happen). The argparse
  usage-error exit was exercised by accident and counted anyway — a bad flag
  (`--no-open`, rc=2) is exactly the `SystemExit` path the `atexit` half covers,
  and it left no root. Cluster 56 passed / 0 failed, rc=0 (84 s); parking +
  entry-point-arming pins 19 passed; fast-subset/skip-audit/mutation-probe 36
  passed. Suite 1791 → 1799. **Not done, on purpose**: the 28 `oracle_gui_cert_cfg_*`
  and 14 `oracle_theme_switch_cfg_*` roots already in /tmp are left in place
  (inert, and deleting them is not this task), and the two non-certifier
  redirectors stay as they are — `u42_crash_repro.py` points at a repo-local
  path by design, and u44's un-reaped roots are the standing Noticed entry next
  door. Checked rather than assumed: those four files were the only other
  `mkdtemp`/`XDG_CONFIG_HOME` sites, and none carries this preamble. Nothing
  committed; these edits ride the same uncommitted tree. **LANDED 2026-10-11:
  committed as `58cf7c9` (the certifier-family slice).**

- **2026-10-11: the GUI certification gates are IN the fast subset — the
  `slow` marker is off them, so `-m "not slow"` can no longer report green
  without certifying a theme.** Four gates, each the local half of a named CI
  step: contrast (9.86 s), studio+wizard (11.21 s), chrome (14.09 s) and the
  theme-switch restyle run (10.32 s) — 45.5 s of certification per fast run, read
  off `--durations`. Before this, the fast subset collected 1519 and deselected
  271, and all four gates were inside that deselection while a bare `pytest` (the
  README's documented command, and what CI runs) kept running them: a developer
  working between changes had a green loop that proved nothing about any theme.
  After: the subset collects 1524, deselects 267, and passes 1524 / 0 failed,
  exit 0 in 228 s — 1519 + the 4 gates + 1 new pin, exactly. The marker keeps its
  job (model-heavy and real-window sweeps), and pyproject's marker description
  now says the certifications are deliberately outside its scope. **The contract
  is pinned by ASKING PYTEST, not by scanning sources**: tests/test_fast_subset.py
  runs `pytest --collect-only -q -m "not slow"` in a subprocess and requires the
  four node ids in its own report, with a >1000-line floor so a truncated parse
  cannot read as agreement. A source scan would be a second implementation of
  "what the subset runs", which is the doctrine next door in
  tests/test_skip_audit.py. **Mutation-proven with the new mutation probe** (both
  restores sha256-identical — the probe's first real use on a change it did not
  introduce): re-marking the chrome gate fails naming that node id, and pointing
  the pin's own expression at `-m "slow"` fails the 269-line vacuity guard.
  README's Development section now documents the fast loop and why the
  certifications live in it. **Not done, on purpose**: the certifier-hermeticity
  module stays `slow`-marked (hermeticity is not a certification verdict, and its
  end-to-end run duplicates a gate that now runs in the subset anyway), and no
  conftest hook was added to refuse marker-deselection — a hook would also fire
  on legitimate `-k` subsets, so the exact command is pinned instead. The suite
  count moves to 1791 (1782 before this session's three landings). Nothing
  committed; these edits ride the same uncommitted tree. **LANDED 2026-10-11:
  committed as `58cf7c9` (the certifier-family slice).**

- **2026-10-11: the hand-done mutation discipline is a TOOL now —
  `scripts/mutation_probe.py` applies, runs and reverts a source mutation with
  hash guards and per-run bytecode isolation, so the same-second stale-bytecode
  trap cannot invalidate a measurement again.** It exists because of the caveat
  recorded in the 2026-10-10 local-gate entry below: CPython keys cached
  bytecode on (source mtime in WHOLE SECONDS, source size), so a mutation
  restored at the same length inside the same second leaves the `.pyc` looking
  current — the next run executes bytes no file contains while every hash a tool
  can read says clean, and that invalid run was caught by luck. **The design
  makes the hand-repair the default**: every child runs with
  `PYTHONPYCACHEPREFIX` pointed at a FRESH throwaway root per run (fresh, not
  merely isolated — the same-second key collides inside a private cache too),
  so it reads neither the repo's `__pycache__` nor an earlier phase's entry, and
  nested processes inherit the variable, which is what makes probing `pytest`
  cover pytest's own subprocesses. On top of that, the cache paths considered
  are the DEFAULT repo-side `.pyc` AND the prefixed one, because
  `importlib.util.cache_from_source` follows the CALLING process's prefix — a
  real bug in this tool's first draft, where the tests' helper used that call
  and therefore named the isolated root while under a probe: the isolation
  mutation "passed for the wrong reason". A matching cache is dropped right
  after every rewrite (`forget_hazardous_bytecode`), and whether it was dropped
  is in the report. **Hash guards**: an optional `expect_sha256` per file, an
  unambiguous occurrence count, a refusal when the replacement changes nothing,
  an all-or-nothing apply (a half-applied spec is rolled back), a restore whose
  sha256 is compared against the original rather than assumed, and — because
  this is a shared checkout — a file that changed under the probe is REFUSED
  and left exactly as found rather than clobbered. **Verdicts**: exit 0 CAUGHT,
  1 SURVIVED, 2 nothing concluded (a guard refusal, an unusable spec, a
  timeout). The last one earned a refinement by being driven at the real repo:
  the first real run mutated `oracle_light`'s `text_muted` DARKER and
  legitimately reported SURVIVED (darker text on a white panel raises contrast,
  so the check should pass); the second pointed it toward the panel and broke
  the module at import, so pytest exited 4 — and under `--expect-rc 1` that is
  now INVALID ("a collection error is not a caught mutation"), not "the check
  does not cover this line". **Verified**: 8 tests in
  `tests/test_mutation_probe.py`, green plain and under
  `PYTHONPYCACHEPREFIX`, including a reproduction of the trap in BOTH directions
  (a naive child running the pre-mutation bytes while the file says otherwise,
  and every hash on disk saying clean while the interpreter runs the reverted
  ones) and a test that keeps the collision LIVE through the probe's own run by
  re-keying the stale `.pyc` header to the file as it is then — without that,
  the mutation's size change quietly destroyed the collision and the isolation
  claim proved nothing; the real theme mutation came back CAUGHT with a
  byte-identical restore (sha256 both sides); a wrong `expect_sha256` and a
  malformed spec each exit 2 naming the file and both digests; and the tool's
  own guards were mutated (shared cache root, `forget_hazardous_bytecode`
  disabled, conflict guard disabled, hash guard disabled — caught by the right
  assertion each, restores sha256-identical), with the honest nuance that the
  no-isolation mutation is invisible from INSIDE another probe: a nested probe
  has already given its child a prefix, so it is only load-bearing when run
  directly. **Cost, stated rather than assumed**: a fresh cache root per run
  recompiles what the child imports — medians 3.03 s isolated vs 2.18 s on the
  repo cache for this file's own tests, three alternating rounds, so ~0.85 s per
  child run here. **Limits**: there is no witness step, so a mutation the check
  simply never reads reports SURVIVED — correct, but the caller still has to
  choose a mutation that changes observable behaviour (the direction-wrong
  theme mutation above is exactly that); the tool does not find the test that
  covers a line; and no Windows execution from here. The 8 new tests will move
  the recorded suite count, which is a commit-keyed baseline
  (`scripts/suite_counts_baseline.json`) and is therefore untouched by
  uncommitted work. Nothing committed; these files ride the same uncommitted
  tree. **LANDED 2026-10-11: committed as `8c863a3` (the mutation-probe
  slice); the baseline's own count moved with it and was re-verified, not
  assumed.**

- **2026-10-11: the GUI certification is one NAMED CI STEP PER ASPECT, so
  the job list says which certification broke.** Before this, `ci.yml` ran the
  merged harness as a single step per OS, so a contrast regression and a chrome
  regression were the same red "GUI Six-Theme Certification" with the reason
  somewhere in a log. Now `ci.yml` carries four steps per OS — `contrast`,
  `chrome`, `studio`, `wizard`, the harness's own order — each named
  `GUI Six-Theme Certification - <aspect> (<OS>)` and each selecting its aspect
  by name (`--aspects <aspect>`), so a failure is attributable from the job
  list alone. **The cost is measured, not assumed**: as separate processes the
  four aspects take 8.1 + 13.8 + 9.3 + 8.9 = 40.1 s, against 17.5 s for one
  process running all four (16.6 s on the re-measure), because every aspect
  rebuilds the six-theme window loop (and the dialogs) it needs — ~+23 s per
  OS against a 491 s suite. Attribution is what the time buys. **The drift is
  pinned in both directions**:
  `tests/test_gui_sections_chrome.py::test_workflow_runs_each_gui_certification_aspect_as_its_own_named_step`
  reads the harness's `ASPECTS` tuple off the source (importing it would create
  the throwaway config root and redirect the config environment — the isolation
  that makes the sweep safe to run on a developer's machine — so the test would
  become part of what it grades) and requires the workflow's certification
  steps to be EXACTLY those aspects x {Linux, Windows}, each with the right
  interpreter, `if:` guard, shell, and its own `--aspects <name>`; an aspect
  added to the harness therefore fails the suite until it has a step, and a step
  for an aspect that no longer exists fails too. **Mutation-proven three ways**
  (each restore sha256-identical): the wizard (Linux) step losing its
  `--aspects wizard` fails on "must select its own aspect by name"; deleting
  the wizard (Windows) step fails naming it in `missing=[...]`; adding a fifth
  aspect to `ASPECTS` fails on the pin's own literal list. **Verified**: the
  workflow parses, the `test` job's 25 steps carry no duplicate name, and the
  eight certification steps sit in the harness's order;
  `tests/test_gui_sections_chrome.py` 9 passed / 0 failed (the slow
  `--aspects chrome` run included); the affected cluster — chrome, design-token,
  certifier hermeticity, entry-point arming, crash-record parking, smoke render,
  theme-switch, doctor idempotence, fresh-clone acceptance, Windows install
  smoke, Vulkan CI smoke — 166 passed / 0 failed, exit 0; the workflow's own
  default invocation still exits 0 in 16.6 s leaving no throwaway root. The
  harness docstring now says one step per aspect rather than one step, and the
  pin's docstring says why the aspect belongs in the step NAME. **Not done, on
  purpose**: no Windows execution from here — this machine is Linux, so the
  eight steps are wired and pinned but only the Linux four have ever run.
  Nothing committed; these edits ride the same uncommitted tree. **LANDED
  2026-10-11: committed as `58cf7c9` (the certifier-family slice).**

- **2026-10-10: the merged GUI harness grows two dialog aspects — `studio`
  and `wizard` — so the six-theme certification covers the recording studio
  and the inference tour, not only MainWindow.** Designed in the harness's own
  idiom (certify facts a real window supplies) and measured before anything
  was asserted. For each of the six themes: (1) the THEMED SURFACE — the
  dialog carries no stylesheet of its own, an application stylesheet is really
  installed, the dominant opaque pixel is one of that theme's surface tokens
  (`bg`/`panel`/`panel_alt`), the theme's token colors cover at least 70% of
  the dialog's painted pixels, and at least two distinct token colors appear.
  Measured with the census's stride-3 sample: the studio paints **97%** theme
  colors and the wizard **94%**, both led by `panel_alt` over `panel`,
  `border` and `bg`. A full pixel scan of the same dialogs reads ~86%,
  because the grid steps over most glyph antialiasing (blended by definition,
  never a flat token), so the floor is set against the sampled number. The
  census skips
  alpha-0 pixels — a correction with a measurement behind it: `QWidget.grab()`
  leaves ~13% of the studio dialog transparent, and counted as black those
  pixels look exactly like an unthemed surface. (2) The STUDIO's own facts:
  the teleprompter lists the text script it was given and never the audio
  decoy beside it, Font Size 24 reaches the prompt area, and no visible widget
  is zero-sized or clipped. (3) The WIZARD's own facts: all nine tour stages
  render a heading and an explanation, every stage that names a highlight
  target RESOLVES to a real widget on the actual MainWindow (eight do — the
  tour exists to point at real controls, and a renamed control would otherwise
  leave it pointing at nothing), the last stage offers Finish, the dialog
  stays non-modal, and the first and last stages pass the surface certificate.
  Wiring: both run inside the theme loop while that theme's window is alive
  (the tour needs a real window to resolve against), then the window is torn
  down as before; the report gains a line per theme and a summary per aspect.
  The module docstring and the ci.yml step comment now describe four aspects,
  and CI's default invocation runs all of them (18 s here, up from ~10 s).
  **Mutation-proven three ways, each restore byte-identical (sha256 both
  sides)**: a dialog-local stylesheet on the studio (it also broke the font
  check) fails the studio aspect on the dialog painting itself;
  `setModal(True)` on the wizard fails on modality; `_target` returning
  nothing fails on the highlight targets, naming the stage and the control.
  **Verified**: the default all-aspects run exit 0 with both dialog readings
  for all six themes; the certifier cluster 84 passed / 0 failed, exit 0;
  tests/test_gui_theme_certification.py 10 passed (the new dialog-aspect test
  11.7 s, the new parser form proof instant). Full suite on the final bytes:
  1782 passed / 0 failed, exit 0 (491.5 s) — the +2 over the standing 1780
  baseline is exactly the two new tests in that file. Three things worth
  naming: the aspects lean on the
  tour's private `_stages`/`_continue`/`_target` names (a rename fails loudly,
  which is the point), and the studio's construction enumerates the machine's
  real input devices once per theme — a cost and a source of ALSA noise on
  stderr, not a defect. Third, the /tmp pile grew by two during this task
  (26 → 28 `oracle_gui_cert_cfg_*` roots; the 23:39 and 23:40 ones hold full
  run content) and they are NOT a reap regression — probed rather than argued:
  a run killed before its `finally` leaves exactly such a root (SIGKILL mid
  studio aspect), while the passing, failing, piped-to-`head`, and
  hard-pipe-close (`| true`, rc 120) paths all reap clean; the mutation runs
  were piped, and a process the tool kills on timeout cannot reap anything.
  The pile is inert and, as recorded in the entry below, left in place.
  Nothing committed; these edits ride the same
  uncommitted tree. **LANDED 2026-10-11: committed as `58cf7c9` (the
  certifier-family slice).**

- **2026-10-10: the certifier family stops littering /tmp — the throwaway
  config root is now reaped on EVERY exit path, after the audio stack
  settles, and the old "the audio teardown races the removal" cause is
  replaced by the measured one.** Found while auditing this machine's /tmp:
  22 `oracle_gui_cert_cfg_*` roots and 14 `oracle_theme_switch_cfg_*` roots,
  one per run — and the theme-switch ones still held the WHOLE hermetic run
  inside them (`the_oracle/app_settings.json`,
  `Output/logs/gui_*_timing.json`, `Input/`, `Profiles/`, `Seashells/`).
  (This session's own red-state probe runs, which deliberately exercised the
  leaking paths before the fix, brought the first count to 26; the totals are
  a snapshot, the shapes are the finding.) Measured per exit path instead of
  assumed: `certify_gui.py --aspects contrast` and the heavy
  `--aspects chrome` were ALREADY clean (both reach the sweep's `finally`),
  while `--help` (argparse's `SystemExit`) and `--aspects legibility` (the
  rc=2 usage error) left one root each, every run, containing nothing but the
  PulseAudio client's `pulse/cookie` — which that client writes under
  `$XDG_CONFIG_HOME` while it LOADS, i.e. at import, before any QApplication
  exists (a phase probe with an atexit checkpoint showed it present at
  import, absent from every teardown phase). `certify_theme_switch.py` had no
  cleanup at all. So the old comment blaming a teardown-time write that races
  the removal described a race nothing could reproduce; the litter was the
  paths no `finally` covered. **The fix**: each script owns one Qt-free
  `_reap_config_root()` — bounded retries, because the cookie write is a
  load-time write a single `rmtree` can lose, and a root that survives the
  retries is REPORTED on stderr rather than vanishing into `ignore_errors` —
  registered with `atexit` so every exit path is covered, and called where
  the audio stack has settled: the sweep's `finally` in `certify_gui.py`
  (replacing the bare `rmtree`) and right after the window teardown in
  `certify_theme_switch.py`. Both docstrings now state the measured cause.
  **Pinned in both certifier test files** on a scratch `TMPDIR` (the
  throwaway root's location is `TMPDIR`'s business, so a run is graded
  against its own litter rather than the shared /tmp): the two fast paths
  above and both slow end-to-end runs assert the scratch parent is empty.
  Red first, measured: both pins failed on the real leaked roots before the
  fix (`oracle_gui_cert_cfg_r3cdrew2`, `oracle_theme_switch_cfg_c7hpgw46`).
  **Verified**: the certifier cluster 65 passed / 0 failed, exit 0 (both slow
  runs included); all four exit paths re-run by hand with `TMPDIR` in a
  scratch dir give `leftover=''` and the same rc as before (0 / 2 / 0 / 0);
  and the honest-failure branch was exercised deliberately — a chmod-locked
  root produced the stderr warning after five attempts at rc 0, not a crash.
  Scope checked rather than assumed: the other two scripts that make scratch
  directories (`certify_suite_counts.py`, `fresh_clone_acceptance.py`) already
  reap theirs in a `finally`, so the leak was confined to these two, and the
  only remaining non-reaper is the u44 harness (logged under Noticed). The
  roots already sitting in /tmp were left in place (inert, and deleting them
  is not this task); `rm -rf /tmp/oracle_gui_cert_cfg_*
  /tmp/oracle_theme_switch_cfg_*` clears them. Nothing is committed; these
  edits ride the same uncommitted tree as the extraction campaign. **LANDED
  2026-10-11: committed as `58cf7c9` (the certifier-family slice).**
  **AMENDED 2026-10-11: the per-script `_reap_config_root()` this entry landed
  is now the shared `scripts/certifier_isolation.py` — same policy, same
  reap-on-every-exit-path behaviour, one owner** (see the certifier-isolation
  entry in Next). The measurement above is what the helper's docstring now
  records — the litter was the exits no `finally` covered, and the cookie write
  is a LOAD-time write a single `rmtree` can lose. Reachability facts unchanged;
  only the address of the implementation moved.

- **2026-10-10: the cast-bar cluster (extraction slice 13) is EXECUTED — the
  ten MainWindow cast-bar methods' bodies moved into `gui_cast` behind four
  call-boundary injections, seam-proven by mutation on a real window, and the
  `moved_owners` gui_cast listing plus the pruned latent `SpeakerSettings`
  adjudication land with it.** The move followed the manifest's recorded
  disposal name-for-name. `src/the_oracle/gui_cast.py` gains ten module-level
  functions (`cast_keys`, `speaker_names`, `_all_speaker_groups`,
  `apply_cast`, `_update_speaker_group_title`, `_refresh_cast_bar`,
  `open_cast_manager`, `_sync_extra_speaker_groups`, `_speaker_settings`,
  `_speaker_config_from_payload`), GENERATED from the app_gui source rather
  than transcribed and proved statement-for-statement against the originals
  by AST dump after the documented substitutions — `self` → `main_window`,
  the four injected names (`speaker_group_cls`, `settings_from_group_fn`,
  `apply_settings_fn`, `normalize_keys_fn`), and
  `CastManagementDialog(self)` → `CastManagementDialog(main_window)`; every
  docstring, comment and blank line travelled byte-for-byte. MainWindow keeps
  each of the ten as a thin delegate whose body imports its owner function
  FUNCTION-LOCALLY (the settled entry-point convention), so app_gui holds no
  module-level binding for any of them and gui_cast is the single patch
  point. **The four injections are the recorded seam, not a convenience.**
  Each name a staying app_gui site still reads (`SpeakerGroup` — `_build_ui`;
  `_speaker_settings_from_group` — `_widget_snapshot`;
  `_apply_speaker_settings_to_group` — `_restore_speaker_section`;
  `normalize_cast_keys` — `_load_project_into_ui`/`prepare_project`) crosses
  resolved from app_gui's globals at call time, so an app_gui-level rebind
  still reaches the moved half of the flow; a module-global read inside
  gui_cast would have made it a silent no-op. **Seam-proven by mutation on a
  REAL MainWindow** (built through `tests/test_app_gui_profiles._build_window`),
  14/14: an app_gui-level `SpeakerGroup` rebind is the class the moved
  `_sync_extra_speaker_groups` constructs (and a gui_cast-level rebind is NOT
  — the designed PARTIAL_OWNED split made visible); app_gui-level
  `_speaker_settings_from_group` / `apply_settings_fn` / `normalize_keys_fn`
  rebinds are what the moved `_speaker_settings` / `apply_cast` run; and the
  moved `open_cast_manager` builds gui_cast's `CastManagementDialog`, which
  app_gui no longer names in any runtime position. The probe is EPHEMERAL (a
  script, not a test) for the slice-12 reason: the patch-couple census is
  derived FROM this suite, and the patch-surface net refuses app_gui-level
  patches of the three PARTIAL_OWNED names outright, so a durable
  app_gui-level rebind would itself redden the nets (measured — an early
  durable version did exactly that). The durable pins therefore cover the
  shape (delegate + function-local import, the four keyword injections at the
  delegate boundary, no injected name read as a gui_cast global) and the
  owner-side behavior (inject recording fakes, watch them run). **Record
  changes.** `moved_owners.the_oracle.gui_cast` now lists `_CastRow` and
  `CastManagementDialog` — the dialog's only app_gui runtime reader was the
  moved `open_cast_manager`, and its remaining `__init__` mention is an
  annotation (the file has `from __future__ import annotations`), exactly the
  concession the pre-flight predicted; app_gui's re-export is gone and the
  pre-flight note was rewritten to the EXECUTED state with its reasoning kept.
  `latent_split_readership.gui_cast.SpeakerSettings` is PRUNED: app_gui keeps
  only annotation uses of it, so no runtime read remains on that side (raw set
  and record re-checked equal, no new pairs). The now-unused
  `speaker_config_from_payload` import left app_gui, and
  `payload_policy.policy_owners` gained `the_oracle.gui_cast` as a delegates
  owner — the decode delegate genuinely lives there now, and the payload net's
  referenced-only-from-owners rule is what forced that honest record edit
  rather than a seam. **Verified.** Net cluster 39 passed (patch-surface incl.
  the offender and latent nets, payload policy, import-direction, cast-owner
  with 16 of its own); wider targeted batch 218 passed (profiles, srt,
  ingest-transformer gui, tooltips, theme certification, stem-writer
  manifest, recording owner, entry-point arming, record-integrity); full suite
  **1777 passed / 0 failed** (491.2 s, exit 0 preserved). Nothing is
  committed; the slice rides the same uncommitted slice-12 + lazy-sections
  tree as its two predecessors. **LANDED 2026-10-11: committed as `d5cc554`
  (the GUI slice).**
  Next-ranked candidates: input/path pickers
  (107 lines, 38 of them source-pinned to stay by the one-owner net), then
  workspace-layout persistence (98 lines — but slice 6's record keeps
  `_register_section` in app_gui "where the layout state lives", so that one
  needs the decision revisited first, not just a seam).

- **2026-10-10: the entry-point delegate convention is SETTLED — the two
  slice-11/12 delegates now import their owner functions function-locally,
  making their `moved_owners` listings literally true (rule 1); the cast
  slice's entry points follow it.** Adjudicated the Noticed item that the
  `gui_menu.build_menu` / `gui_tooltips.register_ctrl_help_descriptions`
  listings contradicted rule 1's "nothing in app_gui reads anymore"
  predicate: both were bound at app_gui module level and the staying
  delegates called them by bare name, so an app_gui-level patch WAS live for
  the delegate call (app_gui held its own reference) while the net's remedy —
  repoint at the owner — was inert. Chosen resolution (the item's first
  candidate): function-local imports in the delegate bodies (the slice-10
  gui_ingest_tools style), so app_gui keeps NO module-level binding, an
  app_gui-level patch is a silent no-op, and the owner module resolves at
  call time — the single patch point the net's printed remedy assumes. The
  alternative (narrowing rule 1's text) was rejected: it would leave the net
  flagging app_gui-level intercepts that genuinely work while its remedy
  stays inert. **Measured, then verified behaviorally on the settled
  bytes.** Before the change the net's own predicate
  (`_app_gui_patchable_reads`) contained both names; after it, neither. A
  Python probe proved dispatch, not just text: an app_gui attribute set to a
  raising sentinel is never read, an owner-module recorder intercepts, and
  the two documented seams stay live through the delegates — `app_gui.THEMES`
  reaches `gui_menu.build_menu` (the slice-12 seam) and `app_gui.cuda_reason`
  reaches `register_ctrl_help_descriptions` (the slice-11 seam).
  `install_ctrl_hover_help` keeps its module-level import (still read in
  `__init__`). Record amended in the manifest: the gui_menu and gui_tooltips
  notes carry the convention, and the gui_cast note's open "entry-point
  delegate bookkeeping" sentence is replaced by it. Nets re-ran green on the
  settled bytes: patch-surface (incl. the offender and latent nets — raw 91
  pairs unchanged, raw == sanctioned), import-direction, tooltips,
  cast-owner, recording-owner, writer-manifest — 79 passed; real-window
  suites test_gui_window_lifetime (7) and test_gui_sections_lazy (5) green,
  so both delegates ran for real during MainWindow construction. Nothing is
  committed; the edit rides the uncommitted slice-12 + lazy-sections tree.
  **LANDED 2026-10-11: committed as `d5cc554` (the GUI slice).**

- **2026-10-10: the cast-bar cluster's split-readership constraint is
  RESOLVED in the record — the cluster is now a movable extraction candidate,
  with the per-name disposal written into the manifest's `gui_cast` note.**
  Method: an AST read-site sweep of the ten cluster methods
  (`_all_speaker_groups`, `_refresh_cast_bar`, `apply_cast`,
  `_update_speaker_group_title`, `_sync_extra_speaker_groups`,
  `_speaker_settings`, `_speaker_config_from_payload`, `open_cast_manager`,
  `cast_keys`, `speaker_names`), intersecting runtime-position reads
  (annotations are not reads) with app_gui module-level bindings — the union
  is EXACTLY the seven recorded names, so the constraint list was complete —
  and every per-name site claim was checked by a verifier before the note was
  written (exit 0). Disposal recorded: (1) `CastManagementDialog` —
  `open_cast_manager` is its only runtime app_gui reader (the `__init__` use
  is an annotation), so once the cluster moves no app_gui runtime reader
  remains and the moving slice LISTS it in `moved_owners.the_oracle.gui_cast`
  beside `_CastRow`, app_gui-level patches becoming offenders to repoint at
  gui_cast; (2) `SpeakerSettings` and `speaker_config_from_payload` need no
  seam — each is runtime-read only by the moving bodies and each is already
  moved-owned elsewhere in the record (models.settings / gui_settings), so
  app_gui-level patches are already offenders and the moved bodies resolve
  gui_cast's imports; (3) `SpeakerGroup`, `_speaker_settings_from_group`,
  `_apply_speaker_settings_to_group` and `normalize_cast_keys` keep
  legitimate app_gui-level patches (staying readers: `_build_ui` panels A/B;
  `_widget_snapshot`; `_restore_speaker_section`; `_load_project_into_ui` +
  `prepare_project`), so none may be listed and the slice must INJECT each at
  the call boundary from app_gui's globals (`speaker_group_cls` /
  `settings_from_group_fn` / `apply_settings_fn` + `normalize_keys_fn` — the
  find_audiocpp_binary pattern); those four stay unlistable permanently
  (their staying readers are window-assembly and persistence code that does
  not travel), the three PARTIAL_OWNED entries stay as written, and
  `normalize_cast_keys` gains none. Verified on the settled bytes: the
  structural diff is exactly the one note (owner names and order untouched),
  and the readers re-ran green — patch-surface net (incl. the latent net),
  cast-owner, recording-owner, writer-manifest: 56 passed / 0 failed, exit 0.
  Nothing is committed; the note rides the same uncommitted manifest as the
  latent record. **LANDED 2026-10-11: committed as `d5cc554` (the GUI
  slice).**
  The cluster's entry-point delegate bookkeeping now follows
  the settled entry-point delegate convention (2026-10-10, the entry above):
  thin delegates left in app_gui import their owner function
  function-locally; next-ranked candidates remain input/path pickers, then
  this cluster (now unblocked).

- **2026-10-10: the patch-couple net gains the LATENT split-readership
  rule — every name both a staying app_gui function body and a gui_* module
  read in runtime position is now flagged and adjudicated, patched or not.**
  The census net could only see names the suite already rebinds; a name read
  on both sides is a split waiting for its first app_gui-level patch — and if
  that first patch is a fix under time pressure, the silently-missed half IS
  the bug. `tests/test_app_gui_patch_surface.py` gains `_RuntimeReads`
  (runtime-position Name loads; annotations are not reads — a name used
  solely as an annotation has no runtime effect), `_app_gui_patchable_reads`
  (names bound at app_gui module level AND read inside some function body: a
  MainWindow method or a module-level function; module-level reads ran at
  import and are excluded, and builtins are never module bindings), 
  `_latent_split_names`, the net
  `test_latent_split_readership_pairs_are_adjudicated` (raw set equals the
  record EXACTLY; floors, anchors, stale entries must prune), a synthetic
  form-proof test (a real split is flagged; annotation-only, function-local
  and module-level-only reads are not; the census exclusion passes through),
  and five loud-failure cases for the new policy fields in
  `test_a_malformed_manifest_fails_loudly`. The record lives in
  `patch_couple_policy`: `latent_split_readership` (91 pairs across 16
  gui_* modules), `latent_anchor_names` (`THEMES`, `QWidget`,
  `save_gui_settings`) and `min_latent_pairs` (60, against 91 raw). Census
  names are excluded by construction — the two nets partition the space, so
  no pair is adjudicated twice. The 91 adjudications classify as 50 PySide6
  platform imports, 18 stdlib imports, 12 package-sibling imports (repoint
  at the owner's module), 10 same-module definitions (rebind at the owner
  for both sides — gui_cast's SpeakerGroup trio, gui_settings'
  load/save/GUISettingsError, gui_themes' THEMES/DEFAULT_THEME,
  gui_utils' normalize_cast_keys), and 1 app_gui-defined wrapper (the vulkan
  `_vulkan_prerequisite_missing` delegate, whose reason says to rebind inside
  gui_vulkan). Every reason is per pair; the platform/stdlib clause names the
  escalation: the census net picks the pair up automatically the moment a
  test rebinds it. **Proven three ways.** (1) Forcing function: before the
  record existed, BOTH nets refused the manifest (the loader's loud rule),
  after it the file is 14 passed / exit 0. (2) Perturbation: dropping
  `gui_themes.py.THEMES` fails naming it unadjudicated; planting a fake
  `gui_menu.py.QSpacerItem` fails naming it stale (both fire in one run);
  an absent anchor fails with "the latent scan lost QSpacerItem"; every
  perturbation was restored byte-identical (`cmp`) and the file re-ran green.
  (3) Form proofs, above. Neighbor manifest readers re-ran: writer-manifest
  23 passed, cast/recording owners 19 passed, import-direction green; the
  payload-policy net's 2 failures are the pre-existing concurrent-refactor
  pair (relocated widget assignments, `_monologue_enabled`) listed in the
  slice-12 entry, untouched by this task. **Caveat for the next session**:
  the 91-pair set is measured mid-refactor — one pair (gui_tooltips' QWidget
  read) already disappeared between two measurements minutes apart. When the
  lazy-sections refactor lands, expect the net to fail on the delta until
  each new pair is adjudicated and each vanished one pruned: that is the rule
  working, not a regression. **RETIRED by re-measurement (2026-10-10, after
  the refactor settled in the tree): the latent set is 90 pairs across 16
  gui_* modules with raw == record EXACTLY — zero unadjudicated, zero stale
  — so there is no delta left to adjudicate and nothing to prune. The one
  vanished pair is the one the caveat named: gui_tooltips.py's QWidget read
  is annotation-only now (`dict[QWidget, str]`, `widget: QWidget`,
  `QWidget | None`), so no runtime read remains on that side and its entry
  was already pruned. The seam census side re-measured in the same pass: 13
  raw bare-use pairs == 13 sanctioned, gui_cast.py's five intact. Floors and
  anchors all clear (16 modules vs floor 13, 90 vs floor 60, census 21 names
  vs floor 15, and THEMES / QWidget / save_gui_settings all still seen). Both
  patch-surface rules re-ran green with cast-owner, payload-policy and
  import-direction (39 passed). Boundary for whoever commits this tree: the
  refactor's code is IN the working tree (gui_sections.py / gui_tooltips.py
  lazy machinery present, every lazy suite green, no writes for the hour
  before the measurement) but still UNCOMMITTED — as is the latent record
  itself, which is absent at HEAD — so committing re-measures the same 90, it
  does not move the numbers. **LANDED 2026-10-11: committed as `d5cc554`**,
  so the re-measure the sentence promises now has its answer — the numbers
  were not re-measured by a second commit because the manifest landed in the
  same slice as the code it records.

- **2026-10-10: the twelfth extraction slice is EXECUTED — the whole menu bar
  moved to a new `gui_menu` owner behind a THEMES injection seam, seam-proven
  by mutation on a real window; the shared tree went mid-refactor under a
  concurrent session immediately afterward, so the full-suite rerun is
  attributed, not green.** The move followed the armed plan below exactly:
  `_build_menu`'s body (186 lines) moved VERBATIM into
  `src/the_oracle/gui_menu.py` as the module-level `build_menu(main_window,
  *, themes: Mapping[str, ThemeTokens]) -> None` — every action attribute,
  menu and callback rides the window (the gui_ingest_tools direction rule;
  the module imports nothing from app_gui). MainWindow keeps `_build_menu`
  as the thin delegate with its exact call-site signature, calling
  `build_menu(self, themes=THEMES)` so the mapping resolves from app_gui's
  globals — the single resolution point its own `_apply_theme_selection`
  also reads; gui_menu binds no THEMES global. The extraction was generated
  from the source programmatically, not transcribed, and proved byte-
  identical against `git show HEAD:...` (186 lines each; 129 `self.` →
  `main_window.` substitutions; AST name count == textual count); the old
  body was then deleted structurally (156 MainWindow methods held; app_gui
  3665 → 3491 lines at that point — the file has since grown to 3885/175
  through the concurrent refactor, not this slice). MOVED_OWNERS gains
  `the_oracle.gui_menu` (1 name: `build_menu`; `_build_menu` stays a method
  and is NOT listed); no bare-set exemption was needed — neither `QAction`
  (imported by gui_menu) nor `THEMES` (injected) is in the suite's
  app_gui-rebind census.
  **The seam is mutation-proven on a real window.** Probe: build through the
  suite's own `_build_window`, with `app_gui.THEMES` rebound to a mapping
  carrying a sentinel theme. Seam intact: the Theme menu contains
  `sentinel_probe` / `SENTINEL-PROBE-NAME` (exit 0). With gui_menu mutated
  to bind its own module-global `THEMES` and ignore the parameter, the
  identical rebind is bypassed — six real themes, sentinel absent, exit 1.
  Reverted and verified byte-identical (sha256 of the whole file, both
  sides), probe green again on the restored bytes.
  **Verification before the tree moved on**: the four pre-flight gate files
  ran green — 40 passed / 0 failed, exit 0 (test_app_gui_patch_surface,
  test_payload_policy_ownership, test_gui_import_direction,
  test_gui_tooltips); the five-file menu-consumer cluster (profiles,
  recording studio, window lifetime, theme cert, sections chrome) 124 passed
  with one hermeticity-snapshot failure — the concurrent session's write to
  `gui_chrome.py` at 02:29:57 UTC landing inside that window — and that file
  passed 7/7 in isolation ~45 s later on the same bytes; the mutation probes
  (which build real windows) pass.
  **The full-suite attempt is attributed, not green, and none of it is slice
  12.** The concurrent lazy-sections refactor began landing in app_gui.py /
  gui_sections.py / gui_tooltips.py / gui_cast.py / gui_chrome.py (writes
  from 19:09 through 19:48 local, continuing); the full-suite run (449 s)
  returned 33 failed / 1732 passed, every failure theirs: 24 stamped by a
  SyntaxError from a merged-line save artifact in their app_gui hunk
  (`layout.setContentsMargins(0, 0, 0, 0)        self.live_panel =
  LivePanel(lazy=...)`), 7 their own new `test_app_gui_lazy_sections.py`
  feature tests against their incomplete implementation, and 1 the doctor
  read-only net catching their write to app_gui.py inside its own window.
  To restore parse-ability I split that one merged line — disclosed as my
  only touch of their hunk (one newline; all tokens preserved) — and did not
  chase their later churn: the post-repair rerun of the affected nets gives
  99 failed / 89 passed, 91 of 97 failure sections the same mid-flight cause
  (`gui_sections.QHSectionGroup.content_built` is a Qt signal while their
  app_gui hunk calls it as a method), the rest their window-assembly
  relocations (`_monologue_enabled` is theirs — absent at HEAD) and their
  gui_tooltips.py IndentationError (L533, written 02:47:58 UTC mid-rerun).
  **State for the next session**: app_gui.py parses but MainWindow cannot be
  constructed until their refactor lands — do NOT read the 33/99 failure
  lists as slice-12 regressions. Once it lands, rerun the four gates + the
  five-file cluster (the seam probe is ephemeral: sentinel THEMES rebind
  through the suite's window builder). **RETIRED by rerun (2026-10-10, after
  the lazy-sections refactor settled in the tree): both groups are green on
  the settled bytes.** The four pre-flight gate files
  (test_app_gui_patch_surface, test_payload_policy_ownership,
  test_gui_import_direction, test_gui_tooltips) are **42 passed / 0 failed,
  exit 0**; the recorded 40 was measured against HEAD, whose patch-surface
  file carries 12 tests against today's 14, and the two added cases are the
  latent readership rule and its form proof landing in that same gate file
  after the reading — so the delta is fully accounted for, not drift. The
  five-file menu-consumer cluster (profiles 83, recording studio 19, window
  lifetime 7, theme cert 8, sections chrome 9) is **126 passed / 0 failed,
  exit 0**, and the single hermeticity-snapshot failure of the recorded run
  is GONE — that failure was the concurrent session's write to gui_chrome.py
  landing inside the snapshot window, and the tree has been quiet for over an
  hour (no src/tests writes since 21:50 local apart from this session's own
  files). MainWindow construction is restored: every window-building suite in
  the cluster drives a real window green, and this session's cast-bar seam
  probe built one through the suite's own `_build_window`.
  **PROMOTED (2026-10-10, this session): the ephemeral seam probe is now a
  permanent net — `tests/test_gui_menu_themes_seam.py` (2 tests) — and the
  ownership record paid the price.** The rebound leg rebinds
  `app_gui.THEMES` to the real mapping plus one sentinel `ThemeTokens` (built
  with `replace`, so window construction still sees a complete token set),
  builds through the suite's `_build_window`, and asserts BOTH that the Theme
  menu's own action labels carry the sentinel's display name and that the
  menu's key set equals the rebound mapping EXACTLY — so a hard-coded look
  list in the moved body fails too; a second baseline leg builds with nothing
  rebound and asserts the menu is exactly the real mapping with no sentinel,
  so the first leg cannot pass on a menu that would carry the probe anyway.
  The promotion is also what moves the name: the patch-couple census is
  derived FROM this suite, so this file makes `THEMES` patch-coupled by
  construction — measured before writing anything, not assumed: **census
  21 → 22 names; latent raw 90 → 89 pairs (the pair `gui_themes.py.THEMES`
  leaves the latent set, taking the `THEMES` anchor with it); seam raw
  13 → 14 pairs (that same pair joins and needs adjudication)**. Three record
  edits, each load-bearing and mutation-checked: the pair moves from
  `latent_split_readership.gui_themes` (entry PRUNED — the latent net rejects
  a stale reason) to `sanctioned_exemptions.gui_themes` with the split spelled
  out (gui_themes DEFINES THEMES and its own stylesheet/theme-application
  helpers read that definition; the window's menu path is the exception the
  census exists for, since the delegate resolves the app_gui global and hands
  it to `build_menu(themes=...)`); and `latent_anchor_names` swaps `THEMES`
  for `Path` (read by eight gui_* modules' latent pairs), because **a name
  the census owns can never stay a LATENT anchor** — that reasoning now sits
  in the manifest's `patch_couple_policy` description beside the anchor list,
  not only here. **Mutation proofs, run on the real bytes**: manifest
  mutation A (drop the gui_themes sanction) reddens the seam net naming
  `gui_themes.py.THEMES`; B (restore the latent entry AND the `THEMES`
  anchor) reddens the latent net on the lost anchor; C (restore only the
  latent entry) reddens it as a stale adjudication; D (identity) passes as the
  control. Source mutation (gui_menu binding its own module-global `THEMES`
  and iterating it, ignoring the parameter — the exact harm the seam
  forecloses) makes BOTH the new permanent test and the seam net fail, and
  the file was restored byte-identical (sha256 compared on both sides) with
  the nets green again on the restored bytes. **Verified**: the new file plus
  the four pre-flight gate files 16 passed / 0 failed, exit 0 (14 of those
  are the gate file itself). One earlier detail is superseded by design: the
  pick-time line "neither `QAction` … nor `THEMES` (injected) is in the
  suite's app_gui-rebind census" was true when measured and still holds for
  `QAction`; `THEMES` left that state here, deliberately, because the census
  is derived from the suite.
  Nothing is committed: gui_menu.py is
  untracked and the delegate hunk rides the same uncommitted app_gui.py as
  their half-applied refactor — separate the two before any commit takes
  this slice. Next-ranked candidates unchanged: input/path pickers, then the
  cast-bar cluster (manifest split-readership notes must be resolved first).
  **LANDED 2026-10-11: committed as `d5cc554` — and the separation this
  paragraph asked for did NOT happen.** The GUI work landed as one slice
  (`d5cc554`) covering the menu extraction, the cast-bar extraction, the
  deferred-sections refactor and the manifest together, so this entry's
  "separate the two before any commit takes this slice" is recorded here as
  guidance that was not followed, not as a step that was taken.

- **2026-10-10: the twelfth extraction slice is picked and armed — the menu
  bar (`_build_menu`) — pre-flight gates green; the tree is settled at
  `ad033cb`, so this pick is NOT held.** Picked from the current 156-method
  MainWindow (3665 lines) by the campaign's own criteria, measured live
  rather than assumed: the cluster is `_build_menu` (187 lines, L568–754),
  the single biggest remaining method — it builds the File / Settings /
  Theme / Help menus plus the menu-bar Recording Studio action, owns 24
  public action attributes (`new_action` … `about_action`), the
  `inference_wizard_menu` and `templates_menu` menus and the
  `_theme_actions` dict, and wires the three `textEdited`
  workspace-persistence connections; called once from `__init__` (L269)
  between `_build_ui()` and `_register_ctrl_help_descriptions()`. It won on
  patch-coupling — graded by the patch-surface net's own AST census,
  measured live at 21 app_gui-rebound names (the eleventh entry's "34" is
  not reproducible, and is not a census shift: no test file changed between
  `4c66ede` and `ad033cb`, so the scanner returns the same 21 names at both
  points — a stale miscount in that entry, not a change in the suite) —
  resolving EXACTLY ZERO
  coupled names, and it carries the strongest drive-from-window proof base
  of the field. Field, re-measured with the same scanner (and CORRECTING
  the eleventh slice's arm record, which called all four candidates "0
  coupled … none has drive-from-window coverage"): the cast-bar cluster
  (108 lines — `_all_speaker_groups`, `_refresh_cast_bar`, `apply_cast`,
  `_update_speaker_group_title`, `_sync_extra_speaker_groups`,
  `_speaker_settings`, `_speaker_config_from_payload`, `open_cast_manager`,
  `cast_keys`, `speaker_names`) resolves 0 coupled names but builds/reads
  SEVEN app_gui-global cast/settings names (`CastManagementDialog`,
  `SpeakerGroup`, `SpeakerSettings`, `_apply_speaker_settings_to_group`,
  `_speaker_settings_from_group`, `normalize_cast_keys`,
  `speaker_config_from_payload`), and the manifest's gui_cast note
  explicitly forbids moving four of them — MainWindow still constructs/reads
  them from app_gui globals (the split-readership hazard) — so this cluster
  is the RISKIEST, not the safest, and it is the one candidate with
  genuinely no window coverage (only `test_gui_settings.py` references its
  names, and that file builds no window); workspace layout persistence (98
  lines — `_register_section`, `_persist_workspace_layout`,
  `_apply_workspace_layout`) resolves 0 coupled names but reads
  `QHSectionGroup`, the manifest's recorded PARTIAL_OWNED split-readership
  hazard (gui_chrome builds the Live column's from its own globals), and
  slice 6's record keeps `_register_section` in app_gui "where the layout
  state lives" — a recorded decision this slice will not contradict;
  input/path pickers (107 lines) resolves 0 coupled names but 38 of its
  lines are `_convert_subtitle_input`, source-pinned to stay by the
  one-owner net, and its remaining ~69 lines mix pickers with layout-row
  building. Drive-from-window proof base for the pick, all through a real
  MainWindow: `test_app_gui_profiles.py` drives menu-created action state
  (`download_vulkan_model_action.isEnabled()` and its disable path,
  `remember_backend_action.setChecked(False)` with the persisted-payload
  assertions, `output_filename_warning_action.setChecked(...)` with the
  warning behavior, `replay_full_wizard_action.text()` pinned to "Replay
  Entire Setup & Tutorial"); `test_gui_tooltips.py::
  test_main_window_registers_ctrl_help` asserts the Ctrl+hover registry
  entries for the `new_action` and `download_vulkan_model_action` the menu
  creates; `test_recording_studio.py` pins `recording_studio_action.text()`;
  `scripts/certify_gui_themes.py` (run by `test_gui_theme_certification.py` —
  **a claim that was false when this sentence was written and became true on
  2026-10-10**, when that certification was made a local gate; see the Done
  entry on the landing)
  builds real windows and asserts the File menu carries "Save Profile…"/
  "Load Profile…" and that `window._theme_actions` holds six entries — the
  theme-menu loop; `test_gui_window_lifetime.py` builds real windows and
  asserts they can be destroyed, which the menu's C-level `triggered`
  lambdas are the recorded reason to need. Owner: MINTS
  `src/the_oracle/gui_menu.py` — no existing owner covers the menu bar
  (gui_chrome.py's docstring scopes it to the Live sidebar column), and
  minting is the campaign's own precedent (slices 1, 2, 4, 5, 10).
  Harm-class inventory (same scope-aware scan): ZERO census-coupled free
  names; the body's only module globals are `QAction` (PySide6.QtGui) and
  `THEMES` (gui_themes), NEITHER patch-coupled, so the patch-couple net
  needs NO new bare-set exemption. Design: `THEMES` crosses as an INJECTION
  parameter on the delegate call (`build_menu(self, themes=THEMES)`), so
  app_gui stays the single resolution point for a mapping its own staying
  `_apply_theme_selection` also reads — foreclosing a latent app_gui-level
  `THEMES` split (the tenth slice's inject-for-symmetry stance); `QAction`
  is imported by the new module, and no test patches `app_gui.QAction`.
  Every widget, action attribute, and window-method callback the body
  touches rides `main_window` (the gui_ingest_tools direction rule), so
  there is no other seam. Sibling import: gui_menu → gui_themes is a
  within-layer sibling import with precedent (gui_chrome → gui_render /
  gui_sections); the import-direction net forbids only gui_render from
  looking up, and gui_menu is not gui_render. MOVED_OWNERS gains
  `the_oracle.gui_menu` (1 name: `build_menu`); MainWindow keeps `_build_menu`
  as a thin delegate with its exact call-site signature (the gui_ingest_tools
  delegate precedent), and it stays a payload-policy sanctioned reader, so
  that record needs no change. Pre-flight gates ran green on the committed
  tree (39 passed / 0 failed, exit 0): test_app_gui_patch_surface.py,
  test_payload_policy_ownership.py, test_gui_import_direction.py,
  test_gui_tooltips.py; cluster baseline (theme certifier + window
  lifetime) 10 passed / 0 failed, exit 0. Conflict map: the tree is CLEAN
  at `ad033cb` (the eleventh slice's own commit; the only untracked file is
  a parallel session's temp probe, `tests/_theme_switch_probe.py`, not part
  of this campaign — **that probe is retired as of 2026-10-10; the measurement
  was promoted to the tracked `scripts/certify_theme_switch.py`**), so there are no shared hunks to wait on and the move
  can execute immediately. Next-ranked candidates on record: input/path
  pickers (smaller, mixed concerns, one source-pinned method), then the
  cast-bar cluster (biggest coupling risk — the manifest's recorded
  split-readership notes must be resolved first).

- **2026-10-10: the eleventh extraction slice is EXECUTED — the Ctrl+hover
  help description table moved to the existing `gui_tooltips` owner,
  seam-proven by mutation.** The move followed the plan recorded below
  exactly: the 273-line body of `MainWindow._register_ctrl_help_descriptions`
  (L775–1047) moved VERBATIM into `src/the_oracle/gui_tooltips.py` as the new
  module-level `register_ctrl_help_descriptions(main_window, *,
  cuda_reason_fn)` — the widgets and actions ride the window (the
  gui_ingest_tools direction rule; the module still imports nothing from
  app_gui). MainWindow keeps `_register_ctrl_help_descriptions` as a thin
  delegate with its exact call-site signature, passing the bare app_gui
  `cuda_reason` name so it resolves from app_gui's globals at call time: the
  PyTorch-device sentence is composed from the CALLER's globals, the
  `_vulkan_prerequisite_missing` `find_binary=` precedent generalized. The
  call stays behind the `_ctrl_help is None` guard, so a missing help
  registry still never computes the reason (the function object binds
  eagerly; the call does not). The moved body is byte-identical to the
  original modulo the required re-indent and the two substitutions (`self.`
  → `main_window.`, `cuda_reason()` → `cuda_reason_fn()`), proven by a script
  diffing it against `git show HEAD:...` rather than by eye. MOVED_OWNERS
  gains `the_oracle.gui_tooltips` (1 name); no bare-set exemption was needed —
  the patch-couple net's raw bare set still equals the sanctioned set,
  because the injection is a call-boundary parameter and gui_tooltips binds
  no coupled name. The seam is mutation-proven on a real window: with the
  seam intact, an app_gui-level `cuda_reason` rebind reaches the moved text
  (the registered PyTorch-device description ends with the sentinel; probe
  exit 0); with a bare-global `cuda_reason` resolution mutated into
  gui_tooltips, the identical rebind is bypassed and the probe fails (exit
  1) — the exact Vulkan-slice harm made concrete — then reverted
  byte-identically (proven by a content equality check). Verified: pre-flight
  gates 39/39
  (test_app_gui_patch_surface, test_payload_policy_ownership,
  test_gui_import_direction, test_gui_tooltips), neighbor GUI nets 453/453,
  full suite 1680 passed / 0 failed (exit 0, 353 s) — the cluster's existing
  drive-from-window move-proof (`test_main_window_registers_ctrl_help`)
  passed unchanged, alongside the file's 15 other poller/registry nets. app_gui.py
  3927 → 3665 lines; gui_tooltips.py 225 → 512; the MainWindow method count is
  unchanged at 156 (the delegate stays a method).

- **2026-10-10: the eleventh extraction slice is picked and armed — the
  Ctrl+hover help description table — pre-flight gates green; the tree is
  settled, so this pick is NOT held.** Picked from the current 156-method
  MainWindow (3927 lines) by the campaign's own criteria, measured live
  rather than assumed: the cluster is `_register_ctrl_help_descriptions`
  (273 lines, L775–1047) — the (widget, description) table that feeds the
  Ctrl+hover help registry (4 `register_many` calls / 37 pairs, 3
  `register_action`, 8 single `register`s), called once from `_build_ui`
  (L270) right after `install_ctrl_hover_help` stores `self._ctrl_help`.
  It won on patch-coupling, graded by the patch-surface net's own AST
  census (34 app_gui names rebound across the suite): the cluster resolves
  exactly ONE coupled name — `cuda_reason`, rebound by
  `test_app_gui_profiles.py:1939`, which the PyTorch device description at
  L868 calls to compose its "4 GiB Chatterbox VRAM floor" sentence.
  Field, measured the same way: `_build_menu` (187 lines) and the three
  small clusters (cast bar sync 108 lines, input/path pickers 106 — minus
  `_convert_subtitle_input`, which stays, source-pinned by the one-owner
  net — and workspace layout persistence 98, constrained by slice 6's
  record that layout state stays in app_gui) all show 0 coupled names but
  none has drive-from-window coverage; every sizeable cluster with real
  coupling was graded and rejected: project lifecycle+profiles (330 lines,
  `QMessageBox` via test_ingest_transformer_gui), render/preview window
  handlers (364, `RenderWorker`/`PreviewWorker`/`QMessageBox` across 3
  files), Vulkan window handlers (345, the four thread classes +
  `_vulkan_preflight_report`, 3 files), settings appliers+knobs (220,
  `ChatterboxEngine`/`cuda_reason`, 3 files), inference wizard wiring
  (115, four agent globals, 3 files), prewarm (87, `OraclePipeline`, 3
  files). Its policy layer already has an owner, the tenth-slice risk
  argument: gui_tooltips owns the registry itself
  (`CtrlHoverHelp.register`/`register_action`/`register_many`/
  `description_for`, `install_ctrl_hover_help`), so the description table
  is that owner's content and the slice extends an existing module rather
  than minting one. Harm-class inventory for the move design (measured,
  not guessed): exactly one bare global to seam — `cuda_reason` crosses as
  an INJECTION parameter resolved from app_gui globals at call time (the
  VulkanPreflightThread `preflight_report=` / gui_ingest_tools
  `message_box_cls=` precedent, so the existing app_gui-level rebind keeps
  reaching the moved text); NO Qt class resolves bare in the body (verified
  by the same scope-aware scan — the widget/action references all ride the
  window, which crosses as the `main_window` value per the gui_cast /
  gui_ingest_tools direction rule), and injection means no own-binding, so
  the patch-couple net's bare-set rule needs NO new exemption (gui_tooltips
  binds no coupled name today). MOVED_OWNERS gains
  `the_oracle.gui_tooltips` with the new function name; MainWindow keeps
  `_register_ctrl_help_descriptions` as a thin delegate with its exact
  call-site signature (the gui_ingest_tools delegate precedent). The
  move-proof already exists: `test_gui_tooltips.py::
  test_main_window_registers_ctrl_help` drives the registration through
  the real MainWindow and asserts the registry contents — buttons,
  combos, speaker spin boxes, menu actions, the path-row label/button
  pairs, and the label↔field sharing — so the moved table must pass it
  unchanged, alongside the file's 15 other poller/registry nets. Pre-flight gates
  ran green on the settled tree (39 passed / 0 failed, exit 0):
  test_app_gui_patch_surface.py, test_payload_policy_ownership.py,
  test_gui_import_direction.py, and the cluster's own baseline
  (test_gui_tooltips.py) — run on the committed tree, not the working one.
  Conflict map: the tree is CLEAN at `b35aaa9` (the parallel session's
  recording-timer/apply_theme work landed there and carries no suite
  number of its own, so none is claimed here), so unlike the tenth
  slice's pick there are no shared hunks to wait on — the move can execute
  immediately; next-ranked candidate on record is `_build_menu` (187
  lines, 0 coupled names, but no drive-from-window pin, so its proof base
  is thinner).

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

- **The merged GUI certifier's chrome aspect can SEGFAULT inside a full-suite run
  in a fresh worktree (found 2026-10-11, while re-recording the suite-count
  baseline).** Three fresh-worktree full-suite runs at `dbee3289628`: one green
  (1796 passed / 3 skipped / 0 failed — the entry the baseline now records) and
  two failing at
  `tests/test_gui_sections_chrome.py::test_section_chrome_certification_runs_locally_and_passes`
  with `rc=-11` (SIGSEGV in the certifier subprocess) after 1795 passed. The gate
  is not obviously broken on its own: the same aspect passes 3/3 standalone in a
  fresh worktree of that same commit, and working-tree runs are green — so the
  difference is the context it runs in (a full suite of ~1800 tests, on a shared
  checkout other sessions are also driving), not anything the certification
  checks. Left unfixed on purpose as a bounded unit of its own, not a footnote to
  a baseline record: the crashing tree is deleted by the tool before its
  `crash_reports/` can be read, so the next step is one `--keep` run to capture
  the faulting frame. Consequences today: the commit-keyed baseline's
  re-verification can fail non-deterministically (a count-preserving crash reads
  as drift), and `tests/test_fast_subset.py` lists this gate as of `58cf7c9`, so a
  local `pytest -m "not slow"` loop can segv for the same reason.

- **`scripts/mutation_probe.py` carries a THIRD copy of the reap policy
  (found 2026-10-11, while factoring the certifier family's preamble into
  `scripts/certifier_isolation.py`).** Its `_reap_scratch` is the same doctrine
  by its own description — bounded retries, a survivor reported on stderr
  instead of swallowed, registered with `atexit` so the exits that skip `finally`
  do not plant a root for good — and it reaps the tool's per-run throwaway
  bytecode root and its cache roots. Logged rather than folded in: the request
  named the GUI certification scripts, and the probe is a standalone tool whose
  reap has its own wording, its own failure path and tests that pin both, so
  moving it under the helper is a separate bounded unit with its own red-state
  check. Its docstring now names the helper's function rather than the
  per-script name that refactor removed.

- **`scripts/u44_accessibility_harness.py` still plants both of its throwaway
  roots in /tmp (found 2026-10-10, while reaping the certifier family's
  litter)**: the harness mkdtemps a config root at import (`u44-config-*`) and
  a sandbox per run (`u44-sandbox-*`) and removes neither — exactly the shape
  the two certifiers just lost, in a tool that is not a certifier (a read-only
  audit harness), which is why it is logged rather than fixed with them. Not
  actioned here (the request named the certifier family, one bounded unit);
  the fix is the same two lines — a Qt-free reaper registered with `atexit`,
  called after the harness's own teardown — and it should land with whichever
  test file ends up covering that harness.
  **AMENDED 2026-10-11: that Qt-free reaper now EXISTS as**
  `scripts/certifier_isolation.py` (`reap_isolated_root`, with the certifier
  family's own tests in `tests/test_certifier_isolation.py`), so the remedy is
  no longer "write two lines" but "call the owner" — the entry is still open
  because u44 is not a certifier and its redirects are `setdefault` at module
  scope, i.e. a caller-side decision this refactor deliberately did not make
  for it.

- **The slice-12 menu extraction still has no owner net — only its THEMES
  seam is pinned (found 2026-10-10, while promoting that probe)**:
  `gui_menu.py` is the one moved module with no `test_gui_*_owner.py` file at
  all. Its `build_menu` (187 lines, 24 action attributes, the
  `inference_wizard_menu` / `templates_menu` menus and three `textEdited`
  connections) is covered today only by the import-direction net, the
  patch-couple census and the GUI certifier, and the new
  `test_gui_menu_themes_seam.py` deliberately pins the one seam the extraction
  was mutation-proven on rather than the module's shape. Not actioned here
  (the request was the probe promotion); the work is
  a `test_gui_menu_owner.py` on the cast/recording owner-test pattern —
  identity (app_gui holds no module-level `build_menu` binding, every
  construction site resolves the owner function function-locally), the
  `moved_owners` listing's currency, and behavior probes against a stub main
  window for the action attributes and those three connections.

- **The skip audit can be emptied mid-run, so `ORACLE_FAIL_ON_SKIP=1` can exit 0
  with tests skipped (found 2026-10-10, while recording the suite-count
  baseline)**: the audit every run prints — and the strict CI gate that reads the
  same `_SKIPS` ledger — are only as complete as the last file that skipped.
  `tests/test_skip_audit.py` carries an autouse `_clean_audit` fixture that calls
  `conftest.pytest_sessionstart(...)`, whose whole job is `_SKIPS.clear()`, before
  and after each of its tests, so every skip recorded earlier in the session is
  dropped from the ledger both of them read. Measured rather than read off the
  code: the same three tests skipping both times,
  `test_engine_output_sanitizer.py test_skip_audit.py test_vulkan_backend.py`
  prints "1 test(s) did not run here" against pytest's own "3 skipped", while
  the same selection without `test_skip_audit.py` prints all 3. Consequence, also
  measured: `ORACLE_FAIL_ON_SKIP=1` over
  `test_engine_output_sanitizer.py test_skip_audit.py` — 2 skips, both wiped —
  exits **0**, where `test_engine_output_sanitizer.py` alone exits 1 and names
  both skips in the failure block. So the gate whose own docstring exists so that
  "a skip fails the job instead of hiding in it" can be defeated by test
  ordering: a strict job whose only skips fall in files sorted before
  `test_skip_audit.py` reports green while covering nothing — the exact failure
  `tests/skip_audit.py` was written about. Not actioned here (the request was the
  drift check, and that check now reports the shortfall instead of trusting the
  list); the fix is for that fixture to save and restore the ledger rather than
  clear it, or to move the session reset somewhere a test cannot reach
  mid-session.

- **`fresh_clone_acceptance.py --scratch` dies on a path that does not exist yet
  (found 2026-10-10, checking whether the new count check's own repair applied
  there)**: `--scratch /tmp/somewhere/new` raises `FileNotFoundError` out of
  `tempfile.mkdtemp(dir=...)` and exits **1** — the code that script's docstring
  reserves for "a check failed or a delta was unexpected" — so a user reads a
  crash as a finding. Verified live rather than read off the code:
  `--only wrappers --scratch /tmp/nope_xyz` traces back from
  `scripts/fresh_clone_acceptance.py:625`. Not actioned here (the request was the
  count check, and this is a two-line fix in another script that has its own test
  file); the fix is to `mkdir(parents=True, exist_ok=True)` the scratch parent,
  exactly as `scripts/certify_suite_counts.py` now does.

- **`certify_gui_themes.py` is not wired into CI, and a sibling docstring
  says it is (found 2026-10-10, while extending its section sweep)**:
  **AMENDED 2026-10-10 (same day): the certifier is executed by the test suite
  on every push and every local run now** — see the Done entry on the
  local-gate landing. Neither fix proposed below was taken: no `ci.yml` step
  was added, but the pytest step CI already runs executes the certifier on both
  operating systems, and `test_gui_theme_certification.py` really does run it.
  The "enforced on no push" sentence below is therefore historical, as is the
  parenthetical claim above this entry that the suite only source-scans it.
  **AMENDED AGAIN, 2026-10-10: closed for real** — the merged
  `scripts/certify_gui.py` is a NAMED CI step on both operating systems
  ("GUI Six-Theme Certification"), so `ci.yml` now wires the certification's
  content directly as well as through the suite, and the sibling docstring
  quoted below no longer exists to make the claim.
  **FURTHER AMENDED 2026-10-11**: the name now carries the aspect — one named
  step per aspect per OS ("GUI Six-Theme Certification - <aspect>"), four per
  OS, so the sentence above is true of four steps rather than one; see the Done
  entry for the measured cost of the split.
  `scripts/certify_section_chrome.py`'s module docstring states "Standalone
  like certify_gui_themes.py — CI runs it as a named step on both operating
  systems", but `.github/workflows/ci.yml` carries no step for
  `certify_gui_themes.py` — its only two "Six-Theme" steps invoke
  `certify_section_chrome.py`. The theme certifier is executed by nothing but
  a human: the test suite only source-scans it
  (`tests/test_crash_records_parking.py`, `tests/test_entry_point_arming.py`),
  and no runner or acceptance script shells out to it. Its no-truncation /
  legibility / persistence contract is therefore enforced on no push. Not
  actioned here (the request was the section-coverage extension, not CI
  wiring); the fix is either a pair of workflow steps for it or a corrected
  sentence in the sibling's docstring.

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

- **2026-10-09 (adversarial pass over 8e932bd → ACTIONED 2026-10-10): the
  last context-less deferred callback on MainWindow.** `open_recording_studio`
  armed `QTimer.singleShot(150, self._start_recording_wizard)`; it now arms
  `self._recording_wizard_launch_timer`, a single-shot child of the window
  created in `__init__` (`start(150)` restarts, so a second studio open moves
  the deadline instead of doubling it), and the teardown sweep stops it with
  the other startup timers. Pinned by three tests: child/interval/armed
  (`test_recording_wizard_timer_is_a_child_of_the_window`), no-outlive
  (`test_recording_wizard_timer_does_not_outlive_the_window`), and the
  end-to-end launch
  (`test_open_recording_studio_launches_the_guide_from_the_child_timer`).
  **Rationale corrected while actioning**: the original "pins the window
  wrapper for 150 ms" claim does not reproduce on this PySide6 build — a
  probe showed `QTimer.singleShot(msec, bound_method)` holds the receiver
  *weakly* (receiver collected after `del`, the call silently skipped), so
  the conversion buys explicit child ownership over an undocumented overload
  detail, not a fixed leak. The lifetime pin therefore guards against
  lambda/app-level regressions, not the original helper verbatim.

- **2026-10-10 (found while validating the conversion, FIXED): every
  redundant `apply_theme` re-set cost ~0.45 s and starved timers.** Qt
  re-polishes the whole widget tree on every `QApplication.setStyleSheet`
  call: measured 0.45–0.50 s synchronous on a fully built MainWindow (first
  set on an unstyled app is ~0.08 s). `_on_gui_shown` re-applied the app
  theme unconditionally, so any process holding a previously themed window
  (every multi-window test run) blocked ~0.5 s in the next window's first
  event pump, and any sub-second timer scheduled around it missed its
  deadline — exactly why the new 150 ms recording-guide launch test failed
  only in combined runs. `apply_theme` now returns early when
  `app.styleSheet()` already equals the built css; pinned by
  `test_apply_theme_skips_a_redundant_restyle` in
  tests/test_gui_theme_certification.py.

- **2026-10-10 (genuine theme switch blocks ~0.45 s → investigated and
  ACTIONED the same day: −137 ms / −27%).** Different css (the redundant
  early-return does not apply) and the whole stall is synchronous inside
  `QApplication.setStyleSheet`. Probe breakdown on a fully built MainWindow
  (491 widgets incl. combo popups/menus, offscreen, `tests/_theme_switch_probe.py`
  temp harness — **retired 2026-10-10: this same split is now the tracked,
  CI-wired `scripts/certify_theme_switch.py`**): (1) **~330 ms is Qt's full-tree unpolish/repolish, ∝ widget
  count × depth** — hiding the window or disabling updates changes nothing
  (441–457 vs 450 ms), a bare app with the same css is ~0 ms, flat
  400-siblings costs +78 ms while the same 400 at depth 26 costs +733 ms;
  inherent to Qt, only fewer/shallower widgets would cut it. (2) **~130 ms is
  QSS rule matching, ∝ rules** — a 2-rule css on the same tree is ~311 ms
  vs ~441–471 for the 59-rule stylesheet (a duplicated 118-rule css adds
  ~79 ms); a selector diet is a legibility-owned follow-up, not taken here.
  (3) **~137 ms was ours**: the app-wide Ctrl-hover event filter re-entered
  Python ~20.6k times per switch (PaletteChange ×10.9k, FontChange ×10.5k,
  StyleChange ×5.9k storm) even though no branch in it can ever fire for an
  event generated inside `setStyleSheet` (no user input is processed until
  the call returns to the event loop). Fixed with a `bulk_restyle()` scope
  in gui_themes — `apply_theme` opens it around `setStyleSheet`,
  `CtrlHoverHelp.eventFilter` short-circuits as its first line, unwound in
  `finally`. Interleaved A/B in one process: gate-off mean 510 ms vs
  gate-on mean 373 ms. Pinned by `test_event_filter_short_circuits_during_a_bulk_restyle`
  (tests/test_gui_tooltips.py), the `scope_during_set == [True, True, True]`
  assertion and `test_bulk_restyle_scope_closes_when_setstylesheet_raises`
  (tests/test_gui_theme_certification.py). Remaining ~370 ms is Qt-inherent
  for this tree; of the two open levers named here, BOTH are now taken —
  lazy-building collapsed section content (landed 2026-10-10, slice 12+;
  the n×depth term) and the css selector diet (landed 2026-10-10, see its
  own entry below: 59 → 51 rules, pixel-identical, but wall-clock-neutral —
  the ~130 ms does not scale down with small rule cuts). Widget-level
  `setStyleSheet` calls at widget
  construction (gui_sections/gui_cast/gui_chrome/gui_widgets) stream their
  own smaller storms ungated — unmeasured, separate unit of work, still
  open.

- **2026-10-10 (css selector diet, TAKEN and MEASURED: 59 → 51 rules,
  pixel-identical across all six themes, wall-clock-neutral).** The lever
  offered ~130 ms (rule matching ∝ rules); the safe cut is eight merges of
  blocks with byte-identical declaration bodies — `QLabel` (which `QWidget`
  already matched), the two wizard-muted-label triples, `QCheckBox:disabled`
  and `QMenu::item:disabled` into `QLabel:disabled`,
  `QPushButton:disabled` into the inputs-disabled group (its border line
  duplicated the base `QPushButton` rule; the pseudo-class specificity keeps
  the merged block winning regardless of its new, earlier position),
  `QPushButton:pressed` into `QSplitter::handle`, the two scrollbar
  handle-hover twins, and `QMenuBar::item:selected` into
  `QMenu::item:selected`. Equivalence proven per theme by a pixel-comparison
  probe over a widget zoo covering every merged STATE (disabled
  buttons/inputs/labels/checkboxes, a pressed button, a popped-up menu with
  a disabled item and a separator) against `git show HEAD`'s builder: all
  six IDENTICAL — the probe caught one real bug in my own first pass
  (`QMenuBar::item` accidentally joined the selected-colors list, which
  would have painted the whole menubar selected). **The honest number:
  nothing measurable.** Interleaved A/B on one built MainWindow, alternating
  the two css texts inside one process (11 rounds each, every call a genuine
  switch): pre-diet median 532.8 ms vs post-diet 545.3 ms, delta +12.5 ms —
  inside the ±35 ms run spread; the certifier's cross-run 767 → 652 ms was
  variance, not the diet. The ~130 ms term therefore does NOT scale down
  with small rule cuts: removing 8 of 59 rules is invisible, while
  DUPLICATING all 59 added ~79 ms — the cost lives in the matched-set size
  at this tree's widget count, and the only sub-floor cut (40) would change
  rendering. Left as a guard, not a gain: `RULE_BUDGET = 51` pinned by
  `test_stylesheet_stays_within_its_rule_budget`
  (tests/test_gui_theme_certification.py), pairing the certifier's
  MIN_REAL_RULES floor (40, against gutting) with a ceiling (against
  creep). Certifier green after the diet: 51 rules, all contracts held.

- **2026-10-10 (window-lifetime leak from the lazy-sections slice, FIXED:
  dead section wrappers were pinning every closed MainWindow).** The four
  `test_gui_window_lifetime` failures plus the payload-policy pair and the
  chrome-ownership AST pin (7 deterministic, full-suite and isolated) all
  traced to the deferred-section machinery, verified by gc-referrer probes.
  Root cause: `QHSectionGroup`'s content factory (`partial(self._build_…)`)
  and the `content_built` restore/tooltip hooks are non-QObject Python
  callables; PySide holds them in C-level connection structures the GC
  cannot trace, and the section wrappers outlive their deleted C++ side, so
  every factory → bound method → window edge kept the destroyed window's
  Python wrapper reachable forever (probe: 8 bound methods referr the
  post-teardown wrapper; releasing factory + connections on the live window
  collects it, which is exactly the fix). New
  `QHSectionGroup.release_content_factory()` drops both; `MainWindow`
  `closeEvent` calls it for every registered section on the accepted-close
  path (every refusal path returns above it). The other two contract fixes
  are adjudications the slice-12 entry's caveat asked for: `_monologue_enabled`
  (reads `monologue_check` only when the shared section is built, else the
  saved slice) joins the payload record's `sanctioned_readers`, the payload
  liveness scan's assembly-method set gains `_build_shared_content` /
  `_build_paths_content` (widget construction moved there), and the
  chrome-ownership AST pin accepts `_register_lazy_section` (which delegates
  to `_register_section`) for the Live column. Full suite 1767 passed /
  0 failed after the fixes.

- **2026-10-10 (widget-construction `setStyleSheet` storms, MEASURED and
  GATED in gui_sections/gui_cast: the "storms" are one change event per
  call — the gate is an invariant, not a speedup).** The theme-switch entry's
  open item, measured on a themed app with the real app-wide Ctrl+hover
  filter installed (the production state: `MainWindow.__init__` installs it
  at L320 BEFORE `_build_ui`, so every construction storm is filter-visible).
  Exact attribution by tracing `QWidget.setStyleSheet` from inside the call
  (caller file:line + events streamed while inside): `gui_sections`'s
  collapse toggle 1 call / 1 event / 0.06 ms; `gui_cast`'s three panel
  dividers 3 / 3 / 0.17 ms; the cast row key label 1 / 1 / 0.06 ms — a full
  eager SpeakerGroup's entire setStyleSheet footprint is **11 events /
  ~0.6 ms**, and a whole MainWindow construction streams 10 217 change
  events of which only a handful come from these calls (the rest is Qt's
  inherent polish traffic from creating ~350 widgets under a styled app —
  NOT setStyleSheet storms; the unparented micro-measure over-reads at 3–4
  events/call because parenting changes propagation). **Gated anyway**: the
  three sites now run inside `with bulk_restyle():`, which costs nothing at
  this scale and guarantees no app-wide filter is ever re-entered by a
  widget-level restyle — a future heavier per-widget style inherits the gate
  instead of silently taxing every filter. Pinned by the new
  tests/test_widget_restyle_gate.py, mutation-proven in both halves (the
  AST net and the runtime trace pin each failed when the toggle's gate was
  reverted, restored byte-identically, sha256-verified): the AST scan
  demands every `setStyleSheet` in the two modules sit inside a
  `bulk_restyle()` with (vacuity floor 3, synthetic self-check incl. the
  def-boundary case where a `with` outside a nested def must NOT count), and
  the runtime pin constructs a real eager SpeakerGroup + _CastRow with a
  traced `QWidget.setStyleSheet`, asserting all 5+ in-scope calls found the
  gate OPEN. **Noticed, not actioned** (out of the requested scope):
  `gui_chrome.py` L74 (Live header) and `gui_widgets.py` L68
  (PerceptualSlider caption — 7 calls per slider panel, the largest
  construction-time share) carry the same ungated pattern; extending the
  same net to those two modules is the obvious next slice.

- **2026-10-10 (dialog teardown/config-isolation audit, ACTIONED): all five
  non-MainWindow windows audited against MainWindow's guarantees.**
  RecordingStudioDialog (closeEvent bounded-wait + refusal for its take
  worker; MainWindow.closeEvent joins studio workers before destruction),
  RecordingStudioSetupWizard (parented to its studio, deleteLater on
  finish), InferenceSetupWizard (already in the teardown sweep, context
  singleShot), CastManagementDialog (modal exec, parented, no threads,
  `_cast_dialogs` tracked with finished+finally discard), and
  RenderProgressDialog (display-only, parented, workers owned and waited by
  MainWindow) — all clean; every test-side builder isolates user config.
  The only gap was the last two context-less `QTimer.singleShot(0,
  bound_method)` EndOfMedia deferrals (studio `_on_playback_status` and
  MainWindow `_on_preview_playback_status`) — converted to the context
  overload (pending stop dies with its owner; live behavior identical),
  and pinned against regressions by
  tests/test_qt_timer_ownership.py (AST scan: every `QTimer.singleShot` in
  src must pass an explicit receiver).

- **2026-10-10 (certify_section_chrome hermeticity, ACTIONED): the sweep
  wrote two repo logs on every run — and the crash channel could write a
  third.** Measured (repo-tree + real-config snapshot either side of a real
  consented-machine run): config was already clean (XDG/APPDATA redirect),
  but `MainWindow.__init__` writes `Output/logs/gui_startup_timing.json` on
  every construction and the startup prewarm writes
  `Output/logs/gui_prewarm_timing.json` — exactly those two files changed
  (12 windows across 6 themes). Latent third channel:
  `arm_native_capture()` resolved the repo root, so a consent-ON machine
  opens/creates `<repo>/crash_reports/native-crash.txt` (invisible to a
  snapshot once the file exists; a fresh checkout gets it created).
  Fixed in scripts/certify_section_chrome.py **(now `scripts/certify_gui.py`,
  which absorbed the file on 2026-10-10 with these same layers — the
  contract, its test and the source anchors below moved with them)**:
  `app_gui.ensure_repo_default_paths`
  repointed at the throwaway (`_hermetic_paths`, the same seam the suite
  patches), and `handlers.install(root=_CONFIG_ROOT)` before the unchanged
  bare arm so the native dump + exception/Qt-fatal records resolve into the
  throwaway — with the user's existing consent mirrored byte-for-byte
  (`shutil.copyfile`, never fabricated: absent/false consent stays
  fail-closed) or a consented machine would run silently unarmed while the
  source-level arming pin still passed. Post-fix snapshot: repo 0/0/0,
  config 0/0/0, rc=0, 11.3 s. Pinned by tests/test_certifier_hermeticity.py
  (behavioral subprocess snapshot + source pins for the channels a snapshot
  cannot see: arm-root/consent-mirror order, redirect-before-import,
  no configure_logging/launch_gui, app_gui seam drift); mutation-proven
  live M1–M4 (drop paths redirect → behavioral red naming
  Output/logs/gui_startup_timing.json; drop XDG redirect → red naming real
  config; drop install(root) → source red; add configure_logging → source
  red), each reverted byte-identical. **Noticed, not actioned:**
  `certify_gui_themes.py` has the same three channels (it also builds real
  MainWindows; its config  redirect and record park are in place, but the
  timing logs and the bare arm are unredirected) — separate unit of work,
  same fix shape. Out of contract (stated): OS caches outside the app config
  root (fontconfig/Qt under `HOME`) and the /tmp litter a SIGSEGV leaves
  behind.

- **2026-10-10: the `moved_owners` entry-point inconsistency is RESOLVED
  in the working tree — the listings are now literally true, and the
  slice-10 function-local delegate convention is the campaign's settled
  entry-point convention (landed 2026-10-11 in `d5cc554`, the
  GUI slice that carries the corrected listings and the settled
  convention).** Finding (found 2026-10-10
  while resolving the cast cluster's record):
  `the_oracle.gui_menu.build_menu` and
  `the_oracle.gui_tooltips.register_ctrl_help_descriptions` were listed as
  moved-owned, yet app_gui bound both at module level and the staying
  delegates called them by bare name, so an app_gui-level patch WAS live for
  the delegate call while the net's rule-1 offender message pointed the
  author at the owner module, where the delegate's call did not look
  (app_gui held its own reference); slice 10's function-local imports were
  the older clean spelling. Fix — the finding's first candidate: each
  delegate body imports its owner function function-locally, so app_gui
  keeps no module-level binding, an app_gui-level patch is a silent no-op,
  and the owner module is the single call-time resolution point. Verified
  behaviorally (a raising app_gui-level sentinel is never read; an
  owner-module recorder intercepts; the app_gui.THEMES and app_gui.cuda_reason
  seams stay live through the delegates) and by the net's own predicate
  (`_app_gui_patchable_reads` no longer contains either name). Manifest
  notes amended (gui_menu, gui_tooltips, gui_cast); the `## Next` entry
  carries the full trail.

- **The settled entry-point fix has a dormant older-names analogue (found
  2026-10-10 by the same AST measurement)**: earlier moved-owner listings
  also keep live bare reads in staying app_gui bodies — `RenderSettings`
  (`_render_settings`, `_payload_defaults`, the project load/restore flows),
  `_device_row_text` (the device-combo refreshes), and the gui_settings
  payload names in their `_*_payload` delegate methods
  (`current_gui_settings_payload`, `default_gui_settings_payload`,
  `cast_request_from_payload`, `PayloadDefaults`) — so an app_gui-level
  patch of those names is live for app_gui's sites while the net's
  repoint-at-the-owner remedy is inert for those sites. No test patches any
  of them today (the rebind census shares no name with moved_owners), and
  unlike the two settled entry points these reads are not uniformly
  delegate-shaped (builders and consumers are mixed in), so the conversion
  is not mechanical. Disposition open: separate adjudication
  (function-local reads, split-readership entries, or an accepted-policy
  note).

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