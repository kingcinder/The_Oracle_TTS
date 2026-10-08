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
                        lambda backend="pytorch": [sys.executable, "-c", "raise SystemExit(0)"])
    rc = main(["--runs", "1", "--outdir", str(tmp_path)])
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["runs"] == 1
    assert report["failures"] == []


def test_sigsegv_child_is_recorded(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(crash_hunt, "_render_command",
                        lambda backend="pytorch": [sys.executable, "-c", "import os; os._exit(245)"])
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


# --- Task 11: acceptance summary ---


def test_kernel_crash_scan_parses_journal(monkeypatch):
    """Kernel segfault lines during the run window must count as crashes."""
    fake_output = (
        "Sep 27 16:41:06 codypc kernel: python[10010]: segfault at 12 ip 00007af2 in libQt6Widgets.so.6\n"
        "Sep 27 16:47:56 codypc kernel: traps: python[10825] general protection fault ip:795f\n"
        "Sep 27 16:48:00 codypc kernel: something harmless\n"
    )
    monkeypatch.setattr(crash_hunt, "_journal_kernel_log", lambda since: fake_output)
    assert crash_hunt._kernel_crash_count("2026-09-27 16:40:00") == 2


def test_kernel_crash_scan_unavailable_is_none(monkeypatch):
    def boom(since):
        raise FileNotFoundError

    monkeypatch.setattr(crash_hunt, "_journal_kernel_log", boom)
    assert crash_hunt._kernel_crash_count("2026-09-27 16:40:00") is None


def test_acceptance_fails_on_kernel_crash():
    summary = {
        "render_pytorch": {"runs": 5, "non_zero": 0},
        "render_vulkan": {"runs": 3, "non_zero": 0},
        "gui": {"runs": 3, "non_zero": 0},
        "new_crash_records": 0,
        "kernel_crashes": 1,
    }
    verdict, reasons = crash_hunt._acceptance_summary(summary)
    assert verdict is False
    assert any("kernel" in reason for reason in reasons)


def test_acceptance_kernel_scan_unavailable_stays_visible_but_passes():
    summary = {
        "render_pytorch": {"runs": 5, "non_zero": 0},
        "render_vulkan": {"runs": 3, "non_zero": 0},
        "gui": {"runs": 3, "non_zero": 0},
        "new_crash_records": 0,
        "kernel_crashes": None,
    }
    verdict, reasons = crash_hunt._acceptance_summary(summary)
    assert verdict is True
    assert any("kernel" in reason for reason in reasons)  # degraded scan is visible


def test_acceptance_summary_all_clean_passes():
    summary = {
        "render_pytorch": {"runs": 5, "non_zero": 0},
        "render_vulkan": {"runs": 3, "non_zero": 0},
        "gui": {"runs": 3, "non_zero": 0},
        "new_crash_records": 0,
    }
    verdict, reasons = crash_hunt._acceptance_summary(summary)
    assert verdict is True
    assert reasons == []


def test_acceptance_summary_fails_on_sigsegv():
    summary = {
        "render_pytorch": {"runs": 5, "non_zero": 1},
        "render_vulkan": {"runs": 3, "non_zero": 0},
        "gui": {"runs": 3, "non_zero": 0},
        "new_crash_records": 0,
    }
    verdict, reasons = crash_hunt._acceptance_summary(summary)
    assert verdict is False
    assert any("render_pytorch" in reason for reason in reasons)


def test_acceptance_summary_fails_on_new_crash_record():
    summary = {
        "render_pytorch": {"runs": 5, "non_zero": 0},
        "render_vulkan": {"runs": 3, "non_zero": 0},
        "gui": {"runs": 3, "non_zero": 0},
        "new_crash_records": 2,
    }
    verdict, reasons = crash_hunt._acceptance_summary(summary)
    assert verdict is False
    assert any("crash record" in reason for reason in reasons)


def test_acceptance_summary_handles_vulkan_unavailable():
    summary = {
        "render_pytorch": {"runs": 5, "non_zero": 0},
        "render_vulkan": {"runs": 0, "non_zero": 0, "vulkan_unavailable": "audio.cpp build missing"},
        "gui": {"runs": 3, "non_zero": 0},
        "new_crash_records": 0,
    }
    verdict, reasons = crash_hunt._acceptance_summary(summary)
    assert verdict is True, "an explicitly-skipped backend must not fail acceptance"
    assert any("vulkan_unavailable" in reason for reason in reasons)  # skip is visible


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


# --- Task 5: --backend vulkan ---


def test_backend_vulkan_flag_extends_render_command():
    cmd = crash_hunt._render_command(backend="vulkan")
    assert "--inference-backend" in cmd
    assert cmd[cmd.index("--inference-backend") + 1] == "vulkan"


def test_backend_default_is_pytorch():
    cmd = crash_hunt._render_command()
    assert "--inference-backend" not in cmd or "pytorch" in cmd


# --- capture-readiness gate (faulthandler launch-arming regression) ---


def _clean_summary(**extra) -> dict:
    summary = {
        "render_pytorch": {"runs": 5, "non_zero": 0},
        "render_vulkan": {"runs": 3, "non_zero": 0},
        "gui": {"runs": 3, "non_zero": 0},
        "new_crash_records": 0,
    }
    summary.update(extra)
    return summary


def test_acceptance_fails_when_consent_on_but_net_unarmed():
    """The 08:06:51 regression shape: consent on, launch path unarmed. The
    gate must name it and fail — never pass silently again."""
    verdict, reasons = crash_hunt._acceptance_summary(
        _clean_summary(capture_readiness={"consent": True, "armed": False})
    )
    assert verdict is False
    assert any("capture_readiness" in r and "unarmed" in r for r in reasons)


def test_acceptance_passes_when_consent_on_and_net_armed():
    verdict, reasons = crash_hunt._acceptance_summary(
        _clean_summary(capture_readiness={"consent": True, "armed": True})
    )
    assert verdict is True
    assert reasons == []


def test_acceptance_consent_off_is_visible_but_passes():
    """Consent-off installs are honestly unarmed by contract — visible, not fatal."""
    verdict, reasons = crash_hunt._acceptance_summary(
        _clean_summary(capture_readiness={"consent": False, "armed": False})
    )
    assert verdict is True
    assert any("capture_readiness" in r for r in reasons)


def test_acceptance_probe_error_fails_closed():
    """A readiness check that cannot run must fail the gate, not skip: a
    silently-skipped check is the regression it exists to catch."""
    verdict, reasons = crash_hunt._acceptance_summary(
        _clean_summary(capture_readiness={"probe_error": "probe child exit 1: boom"})
    )
    assert verdict is False
    assert any("capture_readiness" in r for r in reasons)


def test_capture_readiness_probe_child_reports_launch_arming():
    """Live probe child through the real cli.main: the launch path must arm
    exactly when consent is on (fail-closed contract), so the regression
    shape (consent on, armed False) is detected end-to-end, not just in
    synthetic summaries."""
    from the_oracle import crash
    from the_oracle.offline import repo_root

    readiness = crash_hunt._capture_readiness(repo_root())
    assert "probe_error" not in readiness, readiness
    assert readiness["armed"] == readiness["consent"], readiness


def test_acceptance_mode_consults_capture_readiness(tmp_path: Path, monkeypatch):
    """--mode acceptance must actually run the readiness probe and let its
    verdict through — a wiring drop would green-light the gate while the
    check silently never executes."""
    class _FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(crash_hunt, "_vulkan_available", lambda: False)
    monkeypatch.setattr(crash_hunt, "_run_gui_once", lambda timing_dir, timeout_s=120.0: "ok")
    monkeypatch.setattr(
        crash_hunt, "_capture_readiness", lambda root: {"consent": True, "armed": False}
    )
    monkeypatch.setattr(crash_hunt.subprocess, "run", lambda *a, **k: _FakeProc())
    rc = crash_hunt.main(["--mode", "acceptance", "--outdir", str(tmp_path)])
    assert rc == 1, "the unarmed-net regression must fail the gate"
    payload = json.loads((tmp_path / "acceptance.json").read_text())
    assert payload["capture_readiness"] == {"consent": True, "armed": False}
    assert any("unarmed" in reason for reason in payload["reasons"])


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
