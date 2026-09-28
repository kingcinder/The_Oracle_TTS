# Exhaustive Debug Sweep — 2026-09-28

Method: search-and-document first (no fixes in place), then fix-all, then a
straggler re-sweep. Probes: static AST sweeps, bulk exception-site audit,
whole-package import/compile probe, thread→widget discipline scan, backend
CLI/doctor probes, offscreen GUI combo-crash matrix (25 adversarial
sequences), real-worker close-during-flight crash replication (20 reps),
front-end/back-end boundary protocol checks.

## Result: 1 real bug found

### BUG-1 — Doctor `--ci` can hang for 30 minutes (silent, looks like a freeze)

- **Where:** `scripts/doctor.py` — `_chatterbox_probe` (call at line 1236,
  probe child spawned via `_run_python_probe` → `_run_command`, line 94).
- **What:** the doctor spawns a child Python that runs
  `ChatterboxTTS.from_pretrained(device="cpu")` **before the repo's offline
  environment is applied** (`the_oracle.offline.apply_offline_environment()`
  is a CLI-startup concern; the doctor never sets `HF_HUB_OFFLINE=1` in its
  probe env — `_probe_environment` only sets `HF_HUB_DISABLE_TELEMETRY=1`).
- **Trigger (reproduced live):** on this machine the Chatterbox HF cache is
  partial (250 MB of `models--ResembleAI--chatterbox`); the child wedges
  inside `huggingface_hub → hf_hub_download → xet_get →
  _download_to_tmp_and_move` attempting a **live network model download**.
  Faulthandler stack captured via `PYTHONFAULTHANDLER=1 timeout -s ABRT`.
- **Why it is catastrophic-by-configuration:** `--model-timeout` defaults to
  **1800 s**, so `the-oracle doctor` (and any scripted/CI doctor run) sits
  hung with zero output for up to 30 minutes; a user sees a silent freeze.
- **Secondary defect (same site):** a *doctor* — whose own design says
  checks must be offline-safe — initiating a multi-gigabyte model download
  contradicts the offline-first contract (CRASH/licensing docs pin
  "no network I/O in diagnostics"; the doctor currently *causes* network I/O
  whenever the model cache is incomplete).
- **Reproduction:**
  - `./.venv/bin/python scripts/doctor.py --ci` → hangs (rc=124 at 90 s and
    at 25 s timeouts), zero output even with `-u`.
  - `--skip-model-init` → completes rc=0; `--model-timeout 25` → completes
    rc=1 with a clean timed-out probe entry. The hang is precisely the
    model-init probe's download.
  - `env HF_HUB_OFFLINE=1 ... from_pretrained` → fails fast with
    `LocalEntryNotFoundError` (the intended offline behavior; the probe then
    reports a structured failure like any other).
- **Fix landed (Phase 2 — revised after the pins were read):** the ledger's
  original fix direction (force `HF_HUB_OFFLINE` in `_probe_environment`)
  turned out to **violate a deliberate pin**: `test_offline_guarantee.py`
  pins that a dev checkout's doctor keeps `HF_HUB_OFFLINE` unset (a
  bootstrap-by-doctor first download is designed behavior; CI always pairs
  `--ci` with `--skip-model-init`, and offline installs already fail fast
  via the marker-gated launchers). The real defect was therefore the
  **silence**: a potentially 30-minute first-run download with zero output,
  indistinguishable from a freeze. The fix announces on stderr **before**
  spawning the probe — naming the bound (`--model-timeout`, with its current
  value) and the skip (`--skip-model-init`) — and stays silent on skipped
  runs. Pinned by two new tests in `tests/test_review_fixes_scripts.py`
  (ordering proven by reading stderr at spawn time); mutation-proven
  (announcement removed → named test fails; revert byte-identical).
- **Not part of the bug:** the doctor's CHANGELOG-date and 4-GiB VRAM
  findings in the report output are its designed checks firing correctly.

## Explicitly cleared (no bug)

- **GUI combo matrix (Tier A, 25 combos × fresh subprocess, offscreen):**
  double-render, render→new_project, preview double/mid-flight, late
  `_finish/_fail_preview`, vulkan preflight/setup failure interleaves,
  profile save/load/reset churn, studio open during project ops, backup
  double-restore, close during every busy state. **Zero native crashes,
  zero unhandled exceptions in product code.** Nonzero exits in the log are
  probe artifacts (modal dialogs that correctly fired and were recorded by
  the harness instead of dismissed; harness-side signature mistakes) — each
  annotated in /tmp/combo_probe/out_*.txt.
- **Real-worker close-during-flight (Tier B, 20 reps):** genuine
  RenderWorker/PreviewWorker QThreads blocked mid-flight against a
  blocking fake pipeline, closed immediately and at +150 ms. Zero aborts,
  zero `native-crash.txt` artifacts: `closeEvent`'s bounded thread-wait +
  refusal path holds. The 2026-09-25 U1.2 faulthandler instrumentation
  (live in the probe children) would have recorded any native crash.
- **Thread→widget discipline (AST scan of every `run()` in QThread/Thread
  subclasses):** no widget access from worker threads; all UI updates ride
  signals. (Single scan hit `gui_vulkan.py:196 stream.close()` is a
  subprocess pipe, not a widget.)
- **121 broad `except Exception` sites audited line-by-line:** 27
  bare-swallow sites each classified as guarded teardown (proc/thread/stream
  cleanup), fallback-with-notice (grammar→local, goemotions→lexical,
  LanguageTool unloadable→deterministic corrector), or telemetry that must
  never raise (crash handlers per contract, timing/log persistence). None
  is a silent user-facing failure path.
- **4 production `assert`s** (vulkan_backend 834, vulkan_setup 139,
  gui_tooltips 55, gui_vulkan 210): each documents a genuinely impossible
  state after explicit guards; `python -O` would strip them without
  changing behavior (conditions re-checked below each).
- **Boundary protocol:** `render_subprocess` progress/error lines are
  flushed, whole-line JSON; the GUI reader drains then parses — no
  partial-line parses (pin exists in the concurrency suite).
- **Import/compile health:** every `the_oracle.*` module imports; compileall
  clean; `python -m the_oracle --help` rc=0.
- **No `shell=True` anywhere; no mutable default arguments anywhere; no
  QThread `.terminate()` misuse** (only subprocess pool/process terminate
  in bounded cleanup paths).
- **Resource leaks:** every spawned probe/worker subprocess is reaped in
  `finally` with kill→wait escalation; no orphaned children observed after
  the sweep's timeouts.

## Phase 3 — straggler re-sweep (2026-09-28, post-fix)

Re-swept the BUG-1 disease class (unbounded subprocess waits) across all
32 `subprocess.run/check_output/call` sites:

- **Already bounded:** every probe and engine call that can legitimately
  stall carries a timeout (device_support 5 s, vulkan probe 15 s, audio.cpp
  invocations self.timeout with per-synthesis scaling, licensing machine-id
  5 s, doctor probes user-bound). ✓
- **Bounded by design:** git queries, release pytest/build/pip invocations
  (the operator waits by design), fresh-clone acceptance runs. ✓
- **Guarded teardown:** gui_render's Windows `taskkill /T /F` (no timeout)
  sits inside a kill-tree escalation that ends in `process.wait(timeout=5)`
  + SIGKILL; Windows-only, teardown-only. ✓
- **Hardening candidates — logged, not actioned** (per repo rules: adjacent
  findings are recorded, not fixed in passing):
  1. `utils/audio.py:260` `write_audio_ffmpeg` and `audio/export_flac.py:54`
     run ffmpeg with no timeout — a wedged ffmpeg binary would hang a
     render/export. Both are interactive, GUI-cancelable paths with no
     observed occurrence; a fix needs a measured long-audio encode bound.
  2. The doctor's `--model-timeout` default of 1800 s remains generous for
     a genuine first download; the new announcement mitigates the silence.
     If a future regression wants a hard cap, `--ci` could clamp it.

Re-verification after the fix: doctor/offline pins 68 passed (incl. the
three read-only/idempotence/history pins); full suite 1403/0; live doctor
run announces the download and completes its report when the probe is
bounded (`--model-timeout 20`). The GUI combo matrix and Tier B were NOT
re-run: the fix touches only `scripts/doctor.py` stderr behavior, no GUI,
worker, or engine code.

## Honest limits of this sweep

- Offscreen probes cannot exercise real GPU/Vulkan devices, real audio
  hardware, or real media backends (QMediaPlayer was faked per house test
  convention). Crash-report telemetry from real sessions remains the
  authority there (that is what U1.2/U1.3 built).
- Modal dialogs were recorded, not interacted with; dialog *contents* were
  not validated, only that flows complete.
- The sweep ran on Linux; Windows/macOS-specific paths (cmd wrappers,
  MachineGuid fingerprint) are covered by their unit pins, not live runs.
