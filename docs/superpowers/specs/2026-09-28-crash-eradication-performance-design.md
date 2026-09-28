# Crash-Eradication & Performance Campaign — Design

**Date:** 2026-09-28
**Status:** approved by user (design gate), pending spec review
**Scope decision:** full repo sweep — every crash class reproduced, root-caused,
and fixed with a test; performance defects and other bugs found along the way
are also fixed (not merely logged).

## Problem statement

The Oracle crashes in daily use. The user reports three symptom classes,
confirmed with screenshots:

1. **Qt process death by SIGSEGV** — terminal shows "The child process exited
   normally with status 245" (256 − 11 = SIGSEGV). The codebase already
   documents this class (`pipeline.py:394`, `gui_render.py` — native
   torch/Chatterbox/Qt interplay) and mitigates it by isolating renders in
   child processes. Crashes still reach the user, so either a code path
   bypasses the isolation or the isolation has a gap.
2. **Vulkan render failure** — Render Failed dialog with
   `VK_ERROR_DEVICE_LOST` during `vkCmd...buffer memset` on an AMD Radeon
   RX 5700 XT (RDNA1, documented experimental). amdgpu queue hang.
3. **Both backends crash** (user-confirmed) — the failure is not exclusively
   the RDNA1 GPU hang.

Additionally, `crash_reports/` does not exist: local crash capture
(sys.excepthook / threading.excepthook / faulthandler, the proven `crash/`
unit) was never consent-enabled, so no Python-side crash records exist.
Logs show a fourth symptom worth root-causing: a render rejected **every**
cached stem as unreadable ("Error opening ... System error") and
re-synthesized the entire cache — a perf-damaging cascade.

The test suite (1401 tests) collects and has been green, meaning the crashes
live in paths the suite does not exercise: real model init, real renders,
live Qt event loops with native interplay.

Out of scope: `/home/cody/llama.cpp/run model.sh` (status 127 / port 8080) —
user-confirmed unrelated external script.

## Approach: evidence-first (option A)

No fix without a captured root cause (systematic-debugging Iron Law).
Performance profiling is used as a bug-finding tool, not premature
optimization.

### Phase 1 — Evidence net (no fixes)

- Arm local crash capture: `the-oracle privacy-opt-in` (writes fail-closed
  consent, installs excepthooks + faulthandler into `crash_reports/`).
  Reports stay local; nothing is uploaded.
- Build a reproduction harness script (repo-local, e.g.
  `scripts/crash_hunt.py`) that:
  - runs repeated real renders on both backends (PyTorch CPU, Vulkan),
  - drives the GUI offscreen (`QT_QPA_PLATFORM=offscreen`) through
    Analyze → Render → Preview cycles,
  - watches subprocess exit codes and records every non-zero exit
    (245/139 SIGSEGV, OOM kills) with captured output and, where
    available, the faulthandler stack,
  - writes findings to a repo-local report the plan's tasks can assert on.
- Run the harness until at least N (≥ 3) reproducible crashes with stacks
  are captured. If a suspected crash cannot be reproduced, gather more
  data rather than guessing.

### Phase 2 — Classify and root-cause

Bucket every captured crash:

- **(a) Qt-process SIGSEGV (exit 245)** — find which code path touches
  native torch/Chatterbox inside the Qt process despite the documented
  isolation; close the gap at the source, not by adding `try/except`.
- **(b) Vulkan `VK_ERROR_DEVICE_LOST`** — determine whether it hangs
  forever, wedges the driver, or cascades; the fix must make a failed
  Vulkan render terminate cleanly with a useful error and not degrade the
  rest of the session (retry/backoff or graceful backend-failover policy
  decided during root-cause work).
- **(c) Python-side exceptions reaching the excepthook** — root-cause each
  individually.
- **(d) Resource kills (OOM)** — confirm via memory profiling; fix the
  allocation/leak, not the symptom.

Per crash: pattern analysis (compare working vs broken paths), one
hypothesis at a time, minimal change, failing test written first
(TDD), full suite after each fix. If 3 fixes fail on the same bucket,
stop and question the architecture with the user before a 4th.

### Phase 3 — Performance as bug-finding (python-performance-optimization skill)

- cProfile on render paths and GUI startup; py-spy (if installable) for
  live sampling; memory profiling for leak-driven slowdowns.
- Fix profiled hot spots and slow-path defects — including the
  "every cached stem unreadable → full re-synthesis" cascade (both a
  perf bug and a symptom: why did the open fail?).
- Guard rails: profile before optimizing; optimize only measured hot
  paths; every perf change keeps the suite green.

### Phase 4 — Full repo sweep

Everything noticed during phases 1–3 that is not a crash gets fixed too
(user chose the full sweep over log-only), each with its own test. One
bounded unit of work at a time; anything genuinely out of reach goes to
"Noticed, not yet actioned" in `STATE.md`.

## Error handling & verification

- Fixes land one at a time; the suite (1401 tests) must be green after
  every fix, plus new regression tests pinned to each root cause.
- Final acceptance: a stress loop — N consecutive renders across both
  backends plus offscreen GUI cycles — with zero non-zero exits and no
  new crash records, before declaring the app no longer crashes.
- Update `STATE.md` (authoritative status file) and `JUNO_FIXES.log`
  after each unit; commit per task.

## Risks

- RDNA1 Vulkan hangs are partly hardware/firmware-level; "fixed" may mean
  robust isolation + clean failure + documented fallback rather than
  eliminating the GPU hang itself. This boundary will be reported honestly.
- Some crashes may only reproduce with the real model loaded; the harness
  must budget for slow model init.
- Full-sweep scope can expand indefinitely; the crash-free stress loop is
  the hard acceptance gate, sweep work proceeds after it.
