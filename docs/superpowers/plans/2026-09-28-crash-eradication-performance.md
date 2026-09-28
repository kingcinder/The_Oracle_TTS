# Crash-Eradication & Performance Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce, root-cause, and fix every crash class in The Oracle (Qt SIGSEGV 245, Vulkan DEVICE_LOST, Python-side exceptions, resource kills), then profile and fix performance defects, until a crash-free stress loop passes.

**Architecture:** Evidence-first: arm the existing fail-closed crash-capture unit, build a repo-local reproduction harness that drives renders and the offscreen GUI while recording exit codes and crash records, classify captured crashes into buckets, and fix one bucket at a time with a failing test first (systematic-debugging Iron Law). Performance profiling (cProfile, memory) runs as a bug-finding tool in Phase 3; the full-repo sweep is Phase 4. Acceptance gate: stress loop with zero non-zero exits.

**Tech Stack:** Python 3.11/3.12, pytest, PySide6 (offscreen), torch 2.6.0+cpu, audio.cpp/Vulkan (optional), cProfile/tracemalloc.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-28-crash-eradication-performance-design.md`.
- Baseline suite: 1401 tests collect; full suite must be green after every fix: `.venv/bin/python -m pytest -q`.
- Run tests with `ORACLE_FAIL_ON_SKIP=1` only when verifying no new skips; normal runs allow the known `audiocpp_cli` environmental skip when `build/` is absent.
- No fix without a captured root cause and a failing test written first (TDD, systematic-debugging Phase 4).
- One hypothesis/fix at a time; if 3 fixes fail in one bucket, stop and discuss architecture with the user.
- Commit per task; never `git add -A` (user has untracked WAVs and a modified STATE.md — leave them alone).
- Crash capture reports stay local in `crash_reports/` (gitignored); nothing is uploaded.
- Do not touch `/home/cody/llama.cpp` (user-confirmed unrelated).
- Offline guarantee: never let new code import `huggingface_hub` in a way that resolves online when `.oracle_offline` exists (use `the_oracle.offline.apply_offline_environment` patterns already in place).

---

### Task 1: Arm local crash capture

**Files:**
- No code changes (operational step using existing, tested CLI).
- Verify: `crash_reports/crash_consent.json`.

**Interfaces:**
- Consumes: `the-oracle privacy-opt-in` (cli.py:1304 — writes consent, arms faulthandler).
- Produces: consent ON for this install; `sys.excepthook`/`threading.excepthook`/`faulthandler` active in all subsequent `the-oracle` runs.

- [ ] **Step 1: Confirm current state**

Run: `.venv/bin/the-oracle crash-status`
Expected: consent off (or missing), directory absent.

- [ ] **Step 2: Opt in**

Run: `.venv/bin/the-oracle privacy-opt-in`
Expected: "Local crash reporting enabled."

- [ ] **Step 3: Verify persistence**

Run: `.venv/bin/python -c "from the_oracle import crash; from pathlib import Path; root=Path('.').resolve(); print(crash.read_consent(root), crash.crash_dir(root).exists())"`
Expected: `True True`

---

### Task 2: Reproduction harness — failing tests first

**Files:**
- Create: `scripts/crash_hunt.py`
- Test: `tests/test_crash_hunt.py`

**Interfaces:**
- Consumes: `the_oracle.crash.list_records`, subprocess exit codes, `QT_QPA_PLATFORM=offscreen`.
- Produces: `scripts/crash_hunt.py:main(argv) -> int`; `scripts/crash_hunt.py:classify_exit(code: int) -> str`; `scripts/crash_hunt.py:class CrashReport` (fields: `kind`, `command`, `returncode`, `stderr_tail`, `crash_record_path`); report JSON written to `build/crash_hunt/report.json` with keys `runs`, `failures`, `crash_records`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crash_hunt.py
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "crash_hunt.py"
spec = importlib.util.spec_from_file_location("oracle_crash_hunt", SCRIPT)
crash_hunt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crash_hunt)
classify_exit, main = crash_hunt.classify_exit, crash_hunt.main


def test_classify_exit_codes():
    assert classify_exit(0) == "ok"
    assert classify_exit(245) == "sigsegv"      # 256 - 11
    assert classify_exit(139) == "sigsegv"      # shell-style 128+11
    assert classify_exit(137) == "oom_kill"     # SIGKILL
    assert classify_exit(1) == "error"
    assert classify_exit(-9) == "oom_kill"


def test_report_written(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(crash_hunt, "_render_command",
                        lambda: ["python", "-c", "raise SystemExit(0)"])
    rc = main(["--runs", "1", "--outdir", str(tmp_path)])
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["runs"] == 1
    assert report["failures"] == []


def test_sigsegv_child_is_recorded(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(crash_hunt, "_render_command",
                        lambda: ["python", "-c", "import os; os._exit(245)"])
    rc = main(["--runs", "1", "--outdir", str(tmp_path)])
    assert rc == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["failures"][0]["kind"] == "sigsegv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_crash_hunt.py -v`
Expected: FAIL (ModuleNotFoundError: scripts.crash_hunt)

- [ ] **Step 3: Write the harness skeleton (minimal to pass tests)**

```python
# scripts/crash_hunt.py
"""Reproduce crashes: run render/GUI loops, classify exit codes, collect crash records."""
from __future__ import annotations
import argparse, json, subprocess, sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def classify_exit(code: int) -> str:
    if code == 0:
        return "ok"
    if code in (245, 139):
        return "sigsegv"
    if code in (137, -9):
        return "oom_kill"
    return "error"


@dataclass
class CrashReport:
    kind: str
    command: list[str]
    returncode: int
    stderr_tail: str
    crash_record_path: str | None = None


def _render_command() -> list[str]:
    return [sys.executable, "-m", "the_oracle.cli", "render"]  # refined in Task 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--outdir", default=str(REPO_ROOT / "build" / "crash_hunt"))
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    for i in range(args.runs):
        proc = subprocess.run(_render_command(), cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=3600)
        kind = classify_exit(proc.returncode)
        if kind != "ok":
            failures.append(asdict(CrashReport(kind, _render_command(),
                                               proc.returncode,
                                               proc.stderr[-4000:])))
    report = {"runs": args.runs, "failures": failures, "crash_records": []}
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_crash_hunt.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/crash_hunt.py tests/test_crash_hunt.py
git commit -m "feat: crash-hunt reproduction harness skeleton with exit classification"
```

---

### Task 3: Harness drives a real render and harvests crash records

**Files:**
- Modify: `scripts/crash_hunt.py`
- Test: `tests/test_crash_hunt.py` (extend)

**Interfaces:**
- Consumes: CLI render (`the-oracle render --input ... --outdir ...` with an Input/ sample), `the_oracle.crash.list_records`.
- Produces: `_render_command()` returning the full render argv using `Input/What is, reality.txt` and a temp outdir; `harvest_crash_records(root, before: set) -> list[str]` returning paths of records created since `before`.

- [ ] **Step 1: Write failing tests for harvest + real command shape**

```python
def test_render_command_uses_real_input():
    cmd = crash_hunt._render_command()
    assert "--input" in cmd and "--outdir" in cmd
    assert any("the_oracle" in part or part.endswith("the-oracle") for part in cmd)


def test_harvest_detects_new_records(tmp_path: Path):
    from the_oracle import crash
    root = tmp_path
    before = set(crash.list_records(root))
    # simulate a crash record landing
    (crash.crash_dir(root)).mkdir(parents=True, exist_ok=True)
    crash.crash_dir(root).joinpath("crash-20260928-000000-aaaaaaaa.json").write_text("{}")
    new = crash_hunt.harvest_crash_records(root, before)
    assert len(new) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_crash_hunt.py -v`
Expected: FAIL (ImportError: harvest_crash_records)

- [ ] **Step 3: Implement**

`_render_command()` builds:
```python
[sys.executable, "-m", "the_oracle.cli", "render",
 "--input", str(REPO_ROOT / "Input" / "What is, reality.txt"),
 "--outdir", str(REPO_ROOT / "build" / "crash_hunt" / "out")]
```
`harvest_crash_records(root, before)` returns `[p for p in crash.list_records(root) if p not in before]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_crash_hunt.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/crash_hunt.py tests/test_crash_hunt.py
git commit -m "feat: crash-hunt drives real renders and harvests crash records"
```

---

### Task 4: Offscreen GUI loop mode in the harness

**Files:**
- Modify: `scripts/crash_hunt.py`
- Test: `tests/test_crash_hunt.py` (extend)

**Interfaces:**
- Consumes: `QT_QPA_PLATFORM=offscreen`, `the_oracle.app_gui.launch_gui` pattern, `Output/logs/gui_launch_timing.json` (`mainwindow_built` event).
- Produces: `_gui_command()` → `[sys.executable, "-m", "the_oracle.cli", "gui"]`; env override adding `QT_QPA_PLATFORM=offscreen`; `--mode gui|render` flag; GUI loop runs launches for `--runs` iterations, each killed after `mainwindow_built` appears or 120s timeout, exit code classified by `classify_exit`.

- [ ] **Step 1: Write failing tests**

```python
def test_gui_mode_uses_offscreen(tmp_path, monkeypatch):
    env = crash_hunt._gui_env()
    assert env["QT_QPA_PLATFORM"] == "offscreen"


def test_gui_mode_flag_accepted(tmp_path, monkeypatch):
    assert "gui" in crash_hunt._gui_command()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_crash_hunt.py -v`
Expected: FAIL (ImportError: _gui_env)

- [ ] **Step 3: Implement `--mode gui`**

Launch the GUI child with `subprocess.Popen`, poll for `mainwindow_built` in `Output/logs/gui_launch_timing.json` (fresh mtime), then terminate the process group; classify the exit. Stale log from a prior launch must not count (compare mtime before launch).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_crash_hunt.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/crash_hunt.py tests/test_crash_hunt.py
git commit -m "feat: crash-hunt offscreen GUI launch loop"
```

---

### Task 5: Evidence collection run (no fixes)

**Files:**
- Create (record only): `docs/superpowers/findings/2026-09-28-crash-evidence.md` (directory new).

**Interfaces:**
- Consumes: Tasks 1–4 (capture armed, harness ready).
- Produces: a findings doc listing each captured crash: bucket (a) Qt SIGSEGV / (b) Vulkan DEVICE_LOST / (c) Python exception / (c) Python exception / (d) resource kill, with stack/record path, repro command, and frequency.

- [ ] **Step 1: Run render loop (PyTorch CPU)**

Run: `.venv/bin/python scripts/crash_hunt.py --mode render --runs 5`
Record return code and `build/crash_hunt/report.json`.

- [ ] **Step 2: Run render loop (Vulkan)**

Add `--backend vulkan` (extend `_render_command` with `--inference-backend vulkan`) and run `--runs 3`.
Record results.

- [ ] **Step 3: Run GUI loop**

Run: `.venv/bin/python scripts/crash_hunt.py --mode gui --runs 3`
Record results.

- [ ] **Step 4: Harvest crash records**

Run: `.venv/bin/the-oracle crash-status` and dump any records: `.venv/bin/python -c "from the_oracle import crash; [print(p) for p in crash.list_records('.')]"`
Read each record's traceback.

- [ ] **Step 5: Write the findings doc**

Table: `| bucket | repro | stack/record | frequency |`. If a reported crash (user's screenshots) did NOT reproduce, say so explicitly and note what extra data is needed (Iron Law: do not fix unreproduced crashes by guessing).
Commit:

```bash
git add docs/superpowers/findings/2026-09-28-crash-evidence.md
git commit -m "docs: crash evidence collected from harness runs"
```

---

### Task 6: Root-cause loop — one task per captured crash (repeat until buckets empty)

**Files:**
- Varies per crash; each fix follows this fixed procedure.

**Interfaces:**
- Consumes: findings doc from Task 5.
- Produces: one regression test + one fix per root cause; findings doc updated with `fixed` marker.

For **each** entry in the findings doc, in bucket order (a → d):

- [ ] **Step 1: Write the failing repro test** — smallest possible reproduction of THIS crash (subprocess exit, exception, or hang), in a new `tests/test_crash_<slug>.py`.

```python
# Example shape for a Qt-process SIGSEGV isolation gap:
def test_render_isolation_gap_closes(tmp_path):
    """If code path X touches torch inside the Qt process, this must go through
    the isolated child instead. Repro: <exact steps from findings doc>."""
    ...
```

- [ ] **Step 2: Run it, confirm it fails for the right reason**

Run: `.venv/bin/python -m pytest tests/test_crash_<slug>.py -v`
Expected: FAIL with the captured root-cause symptom, not a typo/ImportError.

- [ ] **Step 3: Form one hypothesis, make the minimal fix**

State the hypothesis in a code comment or commit message. Fix at the source (e.g., route the leaking path through `render_subprocess`; make Vulkan failure terminate the child cleanly; close the allocation causing OOM). No `try/except` swallow, no symptom patches.

- [ ] **Step 4: Verify test passes + full suite green**

Run: `.venv/bin/python -m pytest tests/test_crash_<slug>.py -v && .venv/bin/python -m pytest -q`
Expected: both pass.

- [ ] **Step 5: Mutation-check the test** — revert the fix temporarily, confirm the test fails, restore, confirm green.

- [ ] **Step 6: Commit**

```bash
git add <fix files> tests/test_crash_<slug>.py
git commit -m "fix: <root cause> isolated and pinned by regression test"
```

**Vulkan bucket special rule (bucket b):** the RDNA1 `VK_ERROR_DEVICE_LOST` hang is partly hardware-level. Acceptable fix shapes, in preference order: (1) audio.cpp child already exits → ensure the GUI renders a clean, actionable failure and the session survives (no wedge, no cascade into subsequent actions); (2) bounded timeout + one retry with backoff; (3) documented auto-fallback to PyTorch with user-visible notice. If the GPU hang itself proves unfixed-at-software-level, record that boundary honestly in the findings doc — the *session* must still never crash.

**Bucket discipline:** after 3 failed fix attempts on one crash, STOP, write up the architectural question, and ask the user (spec §Error handling).

---

### Task 7: Stem-cache "unreadable → full re-synthesis" cascade

**Files:**
- Modify: `src/the_oracle/pipeline.py` (`_load_cached_stem` ~line 462, `render_child.log` evidence)
- Test: `tests/test_stem_cache_write_path.py` (extend) or new `tests/test_cache_open_cascade.py`

**Interfaces:**
- Consumes: observed log: every cached WAV rejected with `Error opening '...': System error` though all 82 files exist (evidence: `Output/logs/render_child.log` 2026-09-27 16:49).
- Produces: root-caused fix + regression test; `_load_cached_stem` behavior contract unchanged for genuinely corrupt files (delete + re-synthesize).

- [ ] **Step 1: Investigate before fixing** (Phase 1 of systematic-debugging)

Check: (a) does `soundfile.open` still fail on those exact files today? Run:
```bash
.venv/bin/python -c "import soundfile as sf; sf.info('Output/cache/utterances/4dc2af08a0d99c6d3a58e124db02c944e4dcdd695951caba09d1e1e128e1d58a.wav')"
```
(b) git-log the `_load_servable_stem` gate (JUNO_FIXES.log 2026-09-20 stem-cache write-path audit) — was the cascade caused by a sample-rate mismatch, a gate change, or a real OS-level open failure (path length, permissions, too-many-open-files)? Write the hypothesis down in the findings doc.

- [ ] **Step 2: Write the failing test** capturing the confirmed root cause only.

- [ ] **Step 3: Run to verify failure.**

Run: `.venv/bin/python -m pytest tests/test_cache_open_cascade.py -v`

- [ ] **Step 4: Minimal fix at the source.**

- [ ] **Step 5: Verify — new test + full suite.**

Run: `.venv/bin/python -m pytest tests/test_cache_open_cascade.py -v && .venv/bin/python -m pytest -q`

- [ ] **Step 6: Commit.**

```bash
git add src/the_oracle/pipeline.py tests/test_cache_open_cascade.py
git commit -m "fix: root-cause the stem-cache unreadable cascade (<actual cause>)"
```

---

### Task 8: Performance baseline — profile before optimizing

**Files:**
- Create: `scripts/perf_baseline.py`
- Test: `tests/test_perf_baseline.py`
- Create (record): `docs/superpowers/findings/2026-09-28-perf-baseline.md`

**Interfaces:**
- Consumes: cProfile over the deterministic smoke render path (`the_oracle.smoke`), tracemalloc peak-memory measurement.
- Produces: `scripts/perf_baseline.py:profile_render(profile_path, mem_path) -> dict` with keys `top_functions` (list of `{name, cumtime, calls}`), `peak_memory_mb`; JSON files under `build/perf/`; findings doc with the ranked hot-list.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_perf_baseline.py
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "perf_baseline.py"
spec = importlib.util.spec_from_file_location("oracle_perf_baseline", SCRIPT)
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)

def test_top_n_orders_by_cumtime():
    rows = [{"name": "a", "cumtime": 1.0}, {"name": "b", "cumtime": 9.0}]
    assert pb.top_n(rows, 1)[0]["name"] == "b"

def test_profile_render_returns_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(pb, "_run_smoke", lambda: None)
    result = pb.profile_render(tmp_path / "p.prof", tmp_path / "m.json")
    assert "top_functions" in result and "peak_memory_mb" in result
```

- [ ] **Step 2: Run to verify failure.**

Run: `.venv/bin/python -m pytest tests/test_perf_baseline.py -v`

- [ ] **Step 3: Implement** — cProfile around `_run_smoke()` (wraps `the_oracle.smoke` deterministic render, which runs without live model generation), `pstats` → top-40 by cumtime, tracemalloc peak.

- [ ] **Step 4: Run tests to pass, then run the real profile.**

Run: `.venv/bin/python -m pytest tests/test_perf_baseline.py -v && .venv/bin/python scripts/perf_baseline.py`

- [ ] **Step 5: Write the ranked hot-list findings doc** (only measured facts; no optimizations yet).

- [ ] **Step 6: Commit.**

```bash
git add scripts/perf_baseline.py tests/test_perf_baseline.py docs/superpowers/findings/2026-09-28-perf-baseline.md
git commit -m "feat: performance baseline profiler and measured hot-list"
```

---

### Task 9: Fix top performance defects (measured, one at a time)

**Files:**
- Varies per hot spot from Task 8's ranked list.

**Interfaces:**
- Consumes: ranked hot-list.
- Produces: for each of the top defects that is a genuine defect (not inherent model compute): a benchmark-style test or complexity pin + fix; findings doc updated with before/after numbers.

- [ ] **Step 1: Pick the #1 non-inherent hot spot.** Write a failing test showing the defect (e.g., O(n²) repeated file reads, uncached recomputation, quadratic string building).
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Fix.**
- [ ] **Step 4: Verify test + full suite + re-run `perf_baseline.py`; record before/after.**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python scripts/perf_baseline.py`

- [ ] **Step 5: Commit.**

```bash
git add <files>
git commit -m "perf: fix <measured hot spot> (<x>% faster on baseline)"
```

Repeat Steps 1–5 while genuine defects remain on the list; log anything arguable as "Noticed, not yet actioned" in STATE.md instead.

---

### Task 10: Full-repo sweep — remaining bugs found along the way

**Files:**
- Varies; each gets its own micro-cycle.

- [ ] **Step 1: Compile the sweep list** from: findings docs, `STATE.md` "Noticed, not yet actioned" entries, and warnings emitted during a full-suite run (`-W error::DeprecationWarning` probe as a discovery tool only).
- [ ] **Step 2: For each item (one at a time): failing test → minimal fix → full suite green → mutation check → commit.** Per item:
  1. Write the smallest failing test that reproduces the item (`pytest <file> -v` → FAIL for the right reason).
  2. Fix at the source — one hypothesis, one change, no `try/except` swallowing.
  3. `pytest <file> -v && pytest -q` → both green.
  4. Temporarily revert the fix, confirm the test fails, restore, confirm green.
  5. `git add <fix files> <test file>` + `git commit -m "fix: <item> pinned by regression test"`.
- [ ] **Step 3: Cap and report honestly.** If the list grows faster than it shrinks past ~10 items, checkpoint with the user (spec: bounded units of work).

---

### Task 11: Acceptance — crash-free stress loop

**Files:**
- Modify: `scripts/crash_hunt.py` (add `--acceptance` mode)
- Test: `tests/test_crash_hunt.py` (extend)

**Interfaces:**
- Consumes: all fixes from Tasks 6–10.
- Produces: acceptance report `build/crash_hunt/acceptance.json`: `{render_pytorch: {runs, non_zero}, render_vulkan: {runs, non_zero}, gui: {runs, non_zero}, new_crash_records: N}` — all zero/non-zero counts must be 0 and `new_crash_records == 0`.

- [ ] **Step 1: Write the failing acceptance test** (harness function `_acceptance_summary(report)` returns pass/fail from the three loops' results; test feeds a synthetic all-clean report → pass, and one with a 245 → fail).

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `--acceptance`** running: 5× PyTorch renders, 3× Vulkan renders (skip with explicit `vulkan_unavailable` marker if audio.cpp build missing), 3× GUI launches.

- [ ] **Step 4: Run the real acceptance.**

Run: `.venv/bin/python scripts/crash_hunt.py --acceptance`
Expected: exit 0, all zeros.

- [ ] **Step 5: Full suite under strict skips.**

Run: `ORACLE_FAIL_ON_SKIP=1 .venv/bin/python -m pytest -q`
Expected: 0 failed, 0 skipped (this tree has `build/` present).

- [ ] **Step 6: Update STATE.md (campaign entry: what changed, what's deferred) and JUNO_FIXES.log; commit.**

```bash
git add STATE.md JUNO_FIXES.log docs/superpowers/plans/2026-09-28-crash-eradication-performance.md
git commit -m "docs: record the crash-eradication campaign results in the journals"
```

Note: STATE.md is currently modified in the working tree (user's edit) — read it, append only the new entry on top of its current content, and commit only if the user confirms; otherwise leave STATE.md uncommitted and say so.
