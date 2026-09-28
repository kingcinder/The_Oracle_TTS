"""Crash-hunt reproduction harness tests (plan Task 2/3/4).

The harness classifies child exit codes (SIGSEGV = 245/139, OOM = 137),
drives real renders and offscreen GUI launches, and harvests crash records.
"""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "crash_hunt.py"
spec = importlib.util.spec_from_file_location("oracle_crash_hunt", SCRIPT)
crash_hunt = importlib.util.module_from_spec(spec)
# dataclasses resolve annotations through sys.modules; register before exec.
sys.modules[spec.name] = crash_hunt
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
                        lambda: [sys.executable, "-c", "raise SystemExit(0)"])
    rc = main(["--runs", "1", "--outdir", str(tmp_path)])
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["runs"] == 1
    assert report["failures"] == []


def test_sigsegv_child_is_recorded(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(crash_hunt, "_render_command",
                        lambda: [sys.executable, "-c", "import os; os._exit(245)"])
    rc = main(["--runs", "1", "--outdir", str(tmp_path)])
    assert rc == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["failures"][0]["kind"] == "sigsegv"


# --- Task 3: real render command + crash-record harvest ---


def test_render_command_uses_real_input():
    cmd = crash_hunt._render_command()
    assert "--input" in cmd and "--outdir" in cmd
    assert any("the_oracle" in part or part.endswith("the-oracle") for part in cmd)
    input_path = Path(cmd[cmd.index("--input") + 1])
    assert input_path.exists(), f"input file must exist for a real repro: {input_path}"


def test_harvest_detects_new_records(tmp_path: Path):
    from the_oracle import crash

    root = tmp_path
    before = set(crash.list_records(root))
    # simulate a crash record landing between snapshots
    crash.crash_dir(root).mkdir(parents=True, exist_ok=True)
    crash.crash_dir(root).joinpath("crash-20260928-000000-aaaaaaaa.json").write_text("{}")
    new = crash_hunt.harvest_crash_records(root, before)
    assert len(new) == 1


def test_harvest_ignores_preexisting_records(tmp_path: Path):
    from the_oracle import crash

    root = tmp_path
    crash.crash_dir(root).mkdir(parents=True, exist_ok=True)
    old = crash.crash_dir(root).joinpath("crash-20260927-000000-bbbbbbbb.json")
    old.write_text("{}")
    before = set(crash.list_records(root))
    assert crash_hunt.harvest_crash_records(root, before) == []


# --- Task 4: offscreen GUI loop mode ---


def test_gui_env_forces_offscreen():
    env = crash_hunt._gui_env()
    assert env["QT_QPA_PLATFORM"] == "offscreen"


def test_gui_command_launches_gui():
    cmd = crash_hunt._gui_command()
    assert "gui" in cmd
    assert any("the_oracle" in part or part.endswith("the-oracle") for part in cmd)


def test_gui_launch_waits_for_mainwindow_built(tmp_path: Path, monkeypatch):
    """A GUI child that writes mainwindow_built counts as a clean launch."""
    timing_log = tmp_path / "gui_launch_timing.json"

    def fake_child(timing_file: Path, baseline_mtime: float | None, timeout_s: float) -> int:
        # child process: reports a successful launch, then exits 0
        timing_file.write_text(json.dumps({"events": [["mainwindow_built", 1.0]]}))
        return 0

    monkeypatch.setattr(crash_hunt, "_gui_command", lambda: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(crash_hunt, "_gui_child", fake_child)
    outcome = crash_hunt._run_gui_once(tmp_path, timeout_s=10)
    assert outcome == "ok"


def test_gui_mode_flag_drives_gui_loop(tmp_path: Path, monkeypatch):
    """--mode gui must run the GUI cycle, not the render loop."""
    calls = []
    monkeypatch.setattr(crash_hunt, "_run_gui_once",
                        lambda timing_dir, timeout_s=120.0: calls.append(timing_dir) or "ok")
    rc = main(["--mode", "gui", "--runs", "2", "--outdir", str(tmp_path)])
    assert rc == 0
    assert len(calls) == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["runs"] == 2 and report["failures"] == []


def test_gui_launch_timeout_is_classified(tmp_path: Path, monkeypatch):
    """A child that never reports mainwindow_built must not count as a clean launch."""
    monkeypatch.setattr(crash_hunt, "_gui_command", lambda: [sys.executable, "-c", "pass"])

    def fake_child(timing_file: Path, baseline_mtime: float | None, timeout_s: float) -> int:
        return 0  # exits without ever writing the timing log

    monkeypatch.setattr(crash_hunt, "_gui_child", fake_child)
    outcome = crash_hunt._run_gui_once(tmp_path, timeout_s=2)
    assert outcome == "launch_timeout"
