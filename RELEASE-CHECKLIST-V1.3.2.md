# V1.3.2 cut-day checklist

Written on cut-day-eve (2026-09-28) because the cut is **blocked today**: the
parallel actor's work is not landed — `src/the_oracle/app_gui.py`,
`src/the_oracle/gui_vulkan.py`, `tests/helpers.py`,
`tests/test_gui_vulkan_owner.py` are still dirty (their Vulkan-seam
extraction, U4.2 import-hook helper, owner-net updates) and
`scripts/u44_accessibility_harness.py` + `tests/test_worker_import_sweep.py`
are untracked (the sweep net is currently red on *their own uncommitted*
`export_flac.py` lazy-import change). Every step below runs only once that
clears.

## Step 0 — precondition

- `git status --short` must print **nothing**. The release gate
  (`release.py`'s `tree_is_dirty`) refuses ANY uncommitted change, including
  untracked files. Leftover untracked files from either actor = stop and
  resolve first (`commit_slices.py` and `tests/test_commit_slices.py` must be
  committed or deliberately removed — never left untracked on cut day).
- Full suite: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -q`
  — expect ~1560 passed / 0 failed. `tests/test_worker_import_sweep.py` must
  pass (it passes once the parallel actor commits their `export_flac.py`
  preload change). A release run tolerates no known-failing test: any
  failure = stop and root-cause. The perf-baseline cgroup caveat applies
  only while that file is untracked; by cut day it must be committed.

## Step 1 — notes coverage audit (both directions)

The `[Unreleased]` block is already pre-written from the committed
post-1.3.1 window (`0e68df6..HEAD`):

- **Added:** in-app crash-consent/review + license-activation dialogs
  (`4db0852`); doctor `previews/` disk-side audit via
  `ProjectCache.is_gated_preview_name` (`9c3eeb2`); crash-hunt harness +
  `--acceptance` gate incl. kernel-log scan (`30dd91b`, `8fa824d`, `e187e86`,
  `89d5403`, `2d28788`, `0445a72`).
- **Changed:** self-healing covers the spawn-pool path + Live sidebar tally
  (machinery committed with `c3bd892`; determinism pins `d121680`); SymSpell
  dictionary loaded once per process — ~21% off profiled total, ~11% off
  wall clock, stated honestly (`9fa5d6e`).
- **Fixed:** mixed-encoding per-cue subtitle salvage (`f6b9418`); the
  render-click segfault root-caused and fixed via main-thread preloads
  (`ed6ef8b`), and a missing cache stem is a miss, not corruption
  (`73de246`).

On cut day: walk `git log 0e68df6..HEAD` and confirm every user-facing
commit is represented (the parallel actor's landed seam/U4.4 work likely
adds bullets), and every bullet still matches a commit. Add what landed
since; prune nothing silently.

## Step 2 — stamp the release

1. `.venv/bin/python scripts/release.py --sync-changelog` — retitles
   `[Unreleased]` into `## [1.3.2] — <today>` and re-inserts an empty
   placeholder (idempotent).
2. `.venv/bin/python scripts/release.py --sync-banners` — README/STATE
   banners to V1.3.2.
3. Bump `src/the_oracle/__init__.py` `__version__` to `"1.3.2"`.
4. `.venv/bin/python scripts/release.py --check` — must hold. Note: the
   hardened gate (duplicate `[Unreleased]`, misordered sections, empty
   shipped section; `scripts/release.py` + `tests/test_release.py`, still
   uncommitted) is recommended to commit BEFORE cut day; otherwise the cut
   runs on the old single-section gate and the hardening stays pending.

## Step 3 — release commit

House shape, modeled on `0e68df6`:

    git add CHANGELOG.md README.md STATE.md src/the_oracle/__init__.py
    git commit -m "Release V1.3.2: <headline user-facing changes>"

## Step 4 — build, checksums, journal

1. `.venv/bin/python scripts/release.py` — full run: dirty-tree refusal,
   suite, PEP 517 sdist+wheel (offline), `checksums-1.3.2.sha256`, tracked
   copy in `release_checksums/`. Commit that tracked copy (the script prints
   the path). Never rewrite an existing record for the version.
2. Verify `sha256sum -c` against the tracked manifest from an unrelated
   downloads directory (the fresh-clone verifiability premise).
3. Append the `JUNO_FIXES.log` release entry (record the numbers from THIS
   run's suite, not an earlier one) as a separate docs commit, house style.

## Explicitly out of scope for the cut

- The Vulkan-seam extraction and U4.2 import-sweep net are the parallel
  actor's slices to land; the LivePanel tally reset lines are already
  committed (`c3bd892`), so their `app_gui.py` diff no longer carries them.
- `scripts/commit_slices.py` + tests: commit before cut day (Step 0
  forbids untracked files) or drop deliberately.
