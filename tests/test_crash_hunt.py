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
