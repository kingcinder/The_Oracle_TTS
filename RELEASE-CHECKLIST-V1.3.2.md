# V1.3.2 cut-day checklist

Written on cut-day-eve (2026-09-28) because the cut is **blocked today**.
Landed since this was drafted: the Vulkan seam (`2ba2392`), the
patch-couple net (`82b5f61`), and the crash-evidence STATE.md updates
(`6a50358`). Still blocking: `src/the_oracle/audio/export_flac.py:44`
carries a function-level `from mutagen.flac import FLAC` in *committed*
code (the module-level preload at lines 20–21 already exists — the
settle-edit is to hoist or drop the lazy import); the bare `1697156` PID
token sits in committed `STATE.md:1138` and `JUNO_FIXES.log:371` (the
record-integrity net is red on both); `scripts/commit_slices.py` +
`tests/test_commit_slices.py` are uncommitted; and
`scripts/u44_accessibility_harness.py` + `tests/test_worker_import_sweep.py`
are untracked. Every step below runs only once Step 0 clears.

## Step 0 — precondition

- `git status --short` must print **nothing**. The release gate
  (`release.py`'s `tree_is_dirty`) refuses ANY uncommitted change, including
  untracked files. Leftover untracked files from either actor = stop and
  resolve first (`commit_slices.py` and `tests/test_commit_slices.py` must be
  committed or deliberately removed — never left untracked on cut day).
- Journal integrity: `.venv/bin/python scripts/commit_slices.py
  --check-journal` must exit 0 ("house format holds") before anything else
  runs. It validates every `JUNO_FIXES.log` entry in the WORKING TREE copy
  (calendar-date field, no empty fields, hash-shaped citations pure-hex
  7–40 digits — resolvability itself stays the record-integrity net's job
  inside the full suite). A release must not extend a malformed record.
- Gate discipline: the cut later relies on the tool's proof machinery
  (Step 4's journal entry is meant to run under `--verify` with `!`
  commands), so cut day enforces it here: any slice landing performed today
  goes through `commit_slices.py` with `--verify` and at least one `!`
  command — without the flag, `!` lines are parsed but silently skipped.
  The machinery's own proof (`tests/test_commit_slices.py`, 16 tests) is
  inside the full-suite count below; if it is deselected or skipped, stop —
  the gates the cut trusts are unproven.
- Full suite: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -q`
  — expect ~1530 passed / 0 failed (1529 on the settled tree per the
  2026-09-29 record). `tests/test_worker_import_sweep.py` must pass (green
  since `fc7d78c` hoisted the function-level `mutagen.flac` import in
  `export_flac.py` to the module-level `FLAC` binding). A release run
  tolerates no known-failing test: any failure = stop and root-cause. The
  perf-baseline cgroup caveat applies only while that file is untracked; by
  cut day it must be committed.

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
4. `.venv/bin/python scripts/release.py --check` — must hold; the cut runs
   on the hardened gate (duplicate `[Unreleased]`, misordered sections,
   empty shipped section; `scripts/release.py` + `tests/test_release.py`,
   landed in `efa7f03`).

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
3. Append the `JUNO_FIXES.log` release entry as a separate docs commit,
   house style — record the numbers from THIS run's suite, not an earlier
   one. Route it through the tool so the entry is refused before it touches
   the record and the committed runbook is re-checked after the commit.

   The tool's commit order is: slice commits (one per `# ---` header), then
   each slice's `!` verify commands, then the journal append + commit last.
   So the slice MUST commit real content — a slice listing only a clean
   `JUNO_FIXES.log` dies on the empty-commit refusal BEFORE the journal
   phase ever runs (the pre-write refusal cannot save it; there is nothing
   to append into a commit). The working shape proven live: the slice
   commits the runbook change itself, and the journal phase appends the
   entry citing that commit's real hash. Scratch slice file
   `release_journal.slices` (delete it afterwards — Step 0 forbids
   untracked leftovers):

       # --- release journal ---
       RELEASE-CHECKLIST-V1.3.2.md
       ! .venv/bin/python scripts/commit_slices.py --check-journal

   then:

       .venv/bin/python scripts/commit_slices.py release_journal.slices \
           --verify \
           --journal-entry 1:'<the release entry text>' \
           --suite-note "Full suite: <N> passed / 0 failed (release build)."

   Mechanics, in run order: the slice commit lands the runbook text (the
   edit below must exist in the worktree first — e.g. this very cut-day
   correction); the `!` verify then runs `--check-journal` over the record
   AS IT STANDS BEFORE the new entry — proving the record is clean before
   it grows, not after; the journal phase then refuses a malformed entry
   BEFORE appending (the pre-write refusal), appends, and commits
   `JUNO_FIXES.log` with the house message, citing the slice's real hash.
   The `!` line runs only under `--verify`; without the flag it is silently
   skipped, so the flag is not optional here. If you journal by hand
   instead, running `--check-journal` immediately after the commit is the
   minimum gate: a malformed entry in the record is a release defect, not
   a style nit (the record-integrity net reads this file too).

## Explicitly out of scope for the cut

- The Vulkan-seam extraction and U4.2 import-sweep net are the parallel
  actor's slices to land; the LivePanel tally reset lines are already
  committed (`c3bd892`), so their `app_gui.py` diff no longer carries them.
- `scripts/commit_slices.py` + tests: commit before cut day (Step 0
  forbids untracked files) or drop deliberately.
