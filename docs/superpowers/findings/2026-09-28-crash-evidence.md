# Crash Evidence — findings (2026-09-28)

Evidence collected by `scripts/crash_hunt.py` per plan Task 5. Crash capture
armed at 04:40 (`privacy-opt-in`; `crash_reports/` exists, faulthandler armed —
`native-crash.txt` present and empty = no native crash since arming).

## COMPLETE crash inventory (user: "many many crashes")

Channels swept: kernel journal (master list — every native crash logs here),
`/var/crash` apport dumps, `.serpent-circle/04-debug/root-causes.md`,
`Output/logs/`, `crash_reports/`.

### A. Qt native crashes — THE "many" (7 in kernel log, Sep 27–28)

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
  0 since the fix.

### B. Apport SIGABRT dumps (2, origin UNRESOLVED — stated honestly)

- `doctor.py --ci`, Sep 28 01:38:53, SIGABRT while blocked in
  `poll(1800000ms)` = the `--model-timeout` model-init probe wait; frame
  layout shows faulthandler's handler re-raising an externally-sent SIGABRT.
  Overlaps the other session's BUG-1 (silent model-init download, doc in
  `docs/DEBUG_SWEEP-2026-09-28.md`). Sender of the signal: not established
  (no oomd kill in journal, no `timeout -s ABRT` in repo, no bash-history
  entry).
- `python3 -m unittest discover` in a deleted `/tmp` sandbox, Sep 28 04:45:26,
  SIGABRT — system python (not our venv), C.UTF-8 env: likely another
  project's or a harness' test run, not The Oracle's suite.
- ffmpeg crash (Sep 26) is `/opt/bcam-nvr` (different project). Desktop app
  crashes (xfdesktop, ChatGPT, parole, nm-applet) are unrelated noise.

### C. Vulkan `VK_ERROR_DEVICE_LOST` (intermittent, RDNA1)

Not reproduced in 4 Vulkan renders (1 evidence + 3 acceptance). Hardware/
driver-level hang on the RX 5700 XT; GUI contains it as a clean Render Failed
dialog (session survives). Watch item: next occurrence lands in
`crash_reports/` with capture armed.

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
