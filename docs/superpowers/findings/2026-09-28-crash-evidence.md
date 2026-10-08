# Crash Evidence — findings (2026-09-28)

Evidence collected by `scripts/crash_hunt.py` per plan Task 5. Crash capture
armed at 04:40 (`privacy-opt-in`; `crash_reports/` exists, faulthandler armed —
`native-crash.txt` present and empty = no native crash since arming).

## COMPLETE crash inventory (user: "many many crashes")

Channels swept: kernel journal (master list — every native crash logs here),
`/var/crash` apport dumps, `.serpent-circle/04-debug/root-causes.md`,
`Output/logs/`, `crash_reports/`.

### A. Qt native crashes — THE "many" (7 in kernel log, Sep 27–28)

**Resolved-by `ed6ef8b`** — the commit that closed the live killer (worker-
thread first-imports vs shiboken6's import hook, generation 2 below); the
2026-09-08 lifetime fixes are generation 1, recorded in STATE's Deferred
entry for this lineage.

| When | Signature |
|---|---|
| Sep 09 16:41:59, 16:44:45, 16:55:17 | null-ip segfault ×2 + libQt6Widgets GPF |
| Sep 10 13:49:25 | null-ip, 5 s after `render_click` (action-timing cross-ref) |
| **Sep 27 16:40:42** | GPF in libQt6Core |
| **Sep 27 16:41:06** | segfault in libQt6Widgets @ offset 0x5dc49 (the original Render-click signature) |
| **Sep 27 16:47:56** | GPF libQt6Widgets @ 0x5dc49 |
| **Sep 27 16:48:17** | null-ip — same second as render_child.log's `Queuing segment 1/11` (the user's crash-retry visible in logs) |
| **Sep 27 16:48:29** | GPF libQt6Widgets @ 0x5d9b5 |
| **Sep 27 16:48:42** | GPF libQt6Widgets @ 0x5dc49 |
| **Sep 28 01:18:03** | jump into libQt6Core non-exec page (error 15) |

Six of these fired in **8 minutes during the user's Sep 27 evening render
session** — six GUI deaths in one sitting. Root cause family, traced through
three generations of fixes:
1. Recording Studio QMediaPlayer/QThread lifetime UAFs — fixed 2026-09-08.
2. **Worker-thread first-imports vs shiboken6's import hook** — reproduced
   rc=139 with fingerprint (mutagen first-import vs Qt main thread) and FIXED
   2026-09-28 05:52 in `ed6ef8b` (module-level preloads + regression net
   `test_gui_render_import_safety.py`, mutation-proven).
3. **Post-fix: kernel crashes since 05:52 = 0**; the other session's scripted
   repro passes green; this campaign's acceptance (11 processes) green.

The crash-hunt acceptance gate now scans the kernel log too (`kernel_crashes`
field) — the channel where every historical crash lived but exit codes and
`crash_records` never looked. Scan live-validated: 7 in the Sep 27→now window,
  0 since the fix (re-checked 2026-09-28: no kernel trap through 08:06:51,
  when a GUI segfault occurred on the worktree *including* the other session's
  uncommitted WIP — attribution and the capture gap in section D below; the
  committed campaign commits remain kernel-clean).

### B. Apport SIGABRT dumps (2 — RESOLVED 2026-09-28)

**Resolved-by `6a50358`** — the commit that recorded this resolution
(instrumentation, not crashes; the verdict is section D's SIGABRT bullet).

- `doctor.py --ci`, Sep 28 01:38:53, SIGABRT while blocked in
  `poll(1800000ms)` = the `--model-timeout` model-init probe wait; frame
  layout shows faulthandler's handler re-raising an externally-sent SIGABRT.
  **Sender identified: the debug sweep's own `timeout -s ABRT` instrumentation**
  (`docs/DEBUG_SWEEP-2026-09-28.md:25` — "Faulthandler stack captured via
  `PYTHONFAULTHANDLER=1 timeout -s ABRT`"; its reproduction runs ended rc=124).
  Corroboration from the core dump itself: ProcStatus carries
  `Pid=3724243 / PPid=3724241 / NSpgid=3724241 / NSsid=3724240` — a foreground
  job of an interactive session whose parent led the job's process group — and
  the core's environment block contains `_=/usr/bin/timeout` (bash exports `_`
  as the last launched command; a direct shell invocation would name bash).
  GNU `timeout` delivers its expiry signal to the wrapped child from exactly
  that parent; the GDB pthread_kill→raise chain is that delivery passing
  through faulthandler's SIGABRT handler. The poll still had its full 1800 s
  budget: the wrapper fired seconds into the probe wait, not at the doctor's
  own `--model-timeout` deadline. Not an app crash — the sweep's own capture
  mechanism, seen from the inside.
- **Second victim the first report hid**: `apport.log` at 01:42:22 records
  another signal-6 delivery whose cmdline is the model-init probe's exact
  command (`import perth … ChatterboxTTS.from_pretrained(device='cpu')`) —
  apport skipped writing it ("report already exists and unseen"), so only one
  dump ever landed in `/var/crash`. Same sweep, same mechanism (the probe
  command most plausibly wrapped directly to capture the child's own stack).
- `python3 -m unittest discover` in a deleted `/tmp` sandbox, Sep 28 04:45:26,
  SIGABRT — system python (not our venv), C.UTF-8 env; ProcStatus
  `State: I (idle)`, and its pid sits far below the doctor run's (the counter
  wrapped in between, consistent with the chrome-cdp crash-loop churning
  pids). Not The Oracle's suite (ours is pytest under `.venv`, never system
  `python3 -m unittest`); apport.log shows a same-shaped second delivery at
  05:22:25, again deduplicated away. External-ABRT-wrapper pattern like the
  doctor dump — an agent/harness bounding a stuck test run. No Oracle code
  involved; recorded here only because it surfaced in the same crash sweep.
- ffmpeg crash (Sep 26) is `/opt/bcam-nvr` (different project). Desktop app
  crashes (xfdesktop, ChatGPT, parole, nm-applet) are unrelated noise.

### C. Vulkan `VK_ERROR_DEVICE_LOST` (intermittent, RDNA1)

Not reproduced in 4 Vulkan renders (1 evidence + 3 acceptance). Hardware/
driver-level hang on the RX 5700 XT; GUI contains it as a clean Render Failed
dialog (session survives). Watch item: next occurrence lands in
`crash_reports/` with capture armed — **caveat found 2026-09-28, RESOLVED
2026-10-08 (holding commits 4d59854, 725ddf0): arming no longer depends on
consent transitions** — the net arms
on every GUI launch when consent is already on (`arm_native_capture` at the
top of `gui_crash.maybe_run_startup_flow` and in `app_gui.launch_gui` before
`MainWindow()`; see section D).

### D. Post-campaign events (2026-09-28 morning) — two new findings

- **08:06:51 GUI segfault, capture net not armed.** Kernel:
  `python[1697156]: segfault at 0 ip 0 … error 14` (null function-pointer
  call); apport.log logged signal 11 for `the-oracle gui` but wrote no report
  ("executable does not belong to a package"). `crash_reports/native-crash.txt`
  is 0 bytes. Two honest caveats: (1) the process ran the worktree *including*
  the other session's uncommitted WIP (`app_gui.py`, `gui_chrome.py`,
  `gui_vulkan.py` dirty at the time) — it is not evidence against the 13
  committed campaign commits; (2) the net produced nothing because
  `enable_faulthandler_catch` is invoked only on consent transitions
  (first-run accept in `gui_crash.py`, Help-menu re-enable in `app_gui.py`,
  `privacy-opt-in` in `cli.py`) — never on a plain GUI launch with consent
  already on. The section-C watch item's "capture armed" promise was
  therefore false for GUI relaunches. **Resolved 2026-10-08:** arming now
  happens on every GUI launch — `handlers.arm_native_capture` (idempotent,
  fail-closed) is called at the top of `gui_crash.maybe_run_startup_flow`
  (the already-consented relaunch branch that previously returned unarmed)
  and in `app_gui.launch_gui` before `MainWindow()` is built, completing the
  launch-arming series (`4d59854` CLI dispatch, `725ddf0` render subprocess).
  Pinned by `tests/test_gui_crash.py::test_plain_relaunch_with_consent_on_arms_faulthandler`
  (relaunch → armed net, no dialog) and a source pin ordering the
  `launch_gui` arm before window construction; full suite 1562 passed / 0
  failed with the fix in the tree. The segfault's root cause itself remains
  unknown (the process ran a dirty worktree; section-C watch applies).
  **Investigation of record (2026-10-08, post-WIP-commit):** the crashed
  process's dirty `app_gui.py`/`gui_chrome.py`/`gui_vulkan.py` state landed
  only at `2ba2392` (08:56, 50 min AFTER the crash, via the policy-extraction
  slice), after further evolution — the exact bytes that segfaulted never
  verbatim-shipped, and the Sep 28 kernel log is gone to journal rotation
  (boots now reach only Oct 3), so the `ip 0` trap line survives only as the
  quotation in this section. Reproduction on the COMMITTED tree offscreen
  with the net armed: Phase A 10/10 passive 90s launches with the remembered
  Vulkan backend restored, clean; Phase B a full analyze→preview→isolated-
  synthesis→QMediaPlayer playback cycle (+60.5s click, playback +130.5s),
  clean exit, EndOfMedia deferred-stop verified; Phase C 40 rapid
  stop/start playback cycles (120 status transitions), clean. Kernel log
  over the whole repro window: 0 trap lines; `native-crash.txt` still 0
  bytes. Verdict: NOT reproducible on the committed tree under these three
  angles; every known in-process native-crash family this GUI owns was
  already fixed or subprocess-isolated (§A `ed6ef8b`, render/preview
  `run_in_subprocess=True`). Classified closed-as-unreproducible with the
  surviving watch: the next kernel trap on an armed build auto-captures its
  Python-side stack in `native-crash.txt`. Repro drivers:
  `build/crash_hunt/repro_0806*.py` (disposable, not committed).
  **Attribution tightening (2026-10-08, second pass — git-object
  archaeology):** the reflog bounds the crashed worktree far tighter than
  "dirty": the other session committed `b5b9683`/`9c3eeb2` 49–58 s after the
  crash and `c3bd892` (LivePanel tally, in `app_gui.py`/`gui_chrome.py`) 6 min
  after — so the crashed process imported files the other actor was editing
  at that very moment, and the reconstructable in-flight diff
  (`0445a72..2ba2392`) is now characterizable: pure QLabel text plumbing, a
  cumulative counter, and policy-body moves behind constructor injection —
  zero signal `connect`/`emit` changes, zero object-lifetime changes, zero
  native-dispatch surface, i.e. nothing in the recoverable portion plausibly
  produces an `ip 0` instruction-fetch fault. No dangling Sep-28 git objects
  exist (fsck: only Sep 08–21 residue), and the rotated apport logs carry no
  Sep-28 entries, so the unreconstructable residue is exactly: unsaved editor
  states between the crash and the 08:56 landing. Final attribution: the
  recoverable in-flight diff is EXONERATED by inspection; the residual
  suspects are (a) the unrecoverable unsaved states, or (b) the same
  Qt/RDNA1 native class as §A/§C — with 0 kernel traps since the 05:52 fix
  across every committed-tree launch (thousands, incl. the repro campaign)
  pointing at the dirty-mid-edit state as the differentiator. No further
  evidence source exists on this machine; the watch (armed net) is the
  closure path for any recurrence.
- **The SIGABRT family (section B) resolves as instrumentation, not crashes**:
  every externally-sent SIGABRT on this machine that day traces to a
  `timeout -s ABRT` wrapper used to capture stacks of hung processes.

## Runs performed

| # | Mode | Backend | Runs | Result |
|---|------|---------|------|--------|
| 1 | render | pytorch | 2 | 2/2 exit 0, no crash records |
| 2 | gui (offscreen) | — | 3 | 3/3 launched to `mainwindow_built`, no crash records |
| 3 | render | vulkan | 1 | exit 0, no crash records |

**Not reproduced in the harness:** Qt SIGSEGV (exit 245) and Vulkan
`VK_ERROR_DEVICE_LOST`. Both were reported interactively (user screenshots).
Honest boundary: the harness launches the GUI offscreen and kills it after
`mainwindow_built` — it does not drive interactive preview/render flows, and
the Vulkan run succeeded once (device-lost is intermittent, user reports it as
the common failure). More evidence needed: interactive GUI drive + repeated
Vulkan runs (≥3), per plan Task 6.

## Symptom classes (from user screenshots)

1. **Qt process death, exit 245 (= SIGSEGV, 256−11)** — terminal's last output
   is `qt.multimedia.ffmpeg ... vaapi` init lines. Hypothesis tested: Qt
   Multimedia playback itself segfaults the process. **DISPROVED**: a minimal
   QMediaPlayer probe (`build/qt_media_probe.py`) played a stem wav 8× on both
   offscreen and xcb with volume up — no crash (rc=0, 3/3 and 2/2 attempts).
   The vaapi lines are just loud init noise; Qt prints them on every process
   that touches QtMultimedia (our probe included). The real trigger is still
   unknown — candidate: interactive flows inside the GUI process (preview
   dialog + multimedia + model-adjacent imports), per `pipeline.py:394` and
   `app_gui.py:3402` comments that this combination "has been observed to
   segfault".
2. **Vulkan render failure** — `VK_ERROR_DEVICE_LOST` during buffer init on
   RX 5700 XT (RDNA1, radv driver). GPU-hang class; not reproduced in one
   harness run. The GUI already surfaces it as a "Render Failed" dialog
   (session survives). Acceptable fix per spec: clean failure + no session
   degradation; retry/fallback decided in Task 6.
3. **Both backends crash** (user report) — not yet reproduced; CLI pytorch
   renders were clean.

## Bonus bug found and FIXED (Task 7)

False "Cached stem ... unreadable (System error); deleting and re-synthesizing"
warnings on ordinary cache misses — reproduced live in the harness Vulkan run
(16 warnings) and present in the user's `Output/logs/render_child.log`
(2026-09-27, 15 warnings) vs **zero** from sequential (PyTorch) runs.

Root cause: `pipeline.py:836` (batched path) and `:1656` (fast path) probe
`_load_servable_stem` with no `exists()` check; a miss (Vulkan hashes differ
from PyTorch — `inference_backend` is in the chunk hash) hit `sf.read` on a
missing file → libsndfile ENOENT → "System error" → corruption warning +
pointless unlink. Fixed in `_load_cached_stem` (silent miss for missing path,
corruption warning preserved). Fix commit: `73de246`, mutation-proven,
1494-test suite green.

Not a root cause of the *crashes* — it's a log-integrity/perf symptom
(miss handling itself was always correct: re-synthesize).

## Environment notes

- Suite green baseline: **1494 passed** (this tree, before/after Task 7 fix).
- Concurrent activity observed during evidence collection: another session
  ran the full suite (doctor read-only guard parked/restored `build/`
  mid-collection — evidence files survived) and several unrelated commits
  landed on the branch base. No interference with results.
- `Output/cache/utterances/` contains files from Sep 15–21 that open cleanly
  today — the Sep 27 "unreadable" victims were re-synthesized then (mtimes
  16:50:34), consistent with the false-warning root cause above.

## Task 6 outcome (closed 2026-09-28)

- **Bucket (a) Qt SIGSEGV 245/139 — ROOT-CAUSED AND FIXED** (commit
  `ed6ef8b`, concurrent session on this branch): reproduced via scripted
  render → preview → render with playback active (rc=139);
  `crash_reports/native-crash.txt` named the fingerprint — first import of
  `mutagen` inside the render worker thread while the main thread ran Qt
  through shiboken6's import hook (not thread-safe). Fix: worker-call-graph
  imports preloaded at module level; regression net
  `tests/test_gui_render_import_safety.py` (RED first, mutation-proven ×2).
  Post-fix: repro pass rc=0, four candidate states clean.
- **Buckets (b)/(c)/(d): not reproduced.** Vulkan DEVICE_LOST did not fire
  across 4 total Vulkan renders (1 evidence + 3 acceptance); the GUI already
  surfaces it as a clean Render Failed dialog (session survives). No Python
  exception records, no OOM kills (137) observed in any run.
- My independent interactive drive (`scripts/gui_drive.py`, 6 cycles
  offscreen + xcb) was clean even before the fix — consistent with the
  fix's own finding that the race needs playback-active render sequencing.

## Acceptance (plan Task 11 — PASSED)

`scripts/crash_hunt.py --mode acceptance`: 5/5 pytorch renders, 3/3 vulkan
renders, 3/3 GUI launches, **0 non-zero exits, 0 new crash records,
verdict_pass=true** (`build/crash_hunt/acceptance.json`); all 8 renders
produced FLACs. Full suite under `ORACLE_FAIL_ON_SKIP=1`: **1508 passed /
0 failed / 0 skipped**.
