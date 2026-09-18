"""Tests for the Windows install smoke and the CI job that runs it.

The smoke itself executes ``.cmd`` launchers through cmd.exe and kills a live
Qt process, so it can only *run* on Windows — that is what the CI job is for.
What is testable here, on any host, is everything around that: the isolated
environment it builds, that its launcher paths come from the installer's own
rules rather than a second copy, the wait logic that decides whether the GUI
came up (including the failure paths), and the CI job's declared contract.

No test installs anything, launches a GUI, or touches a real user profile.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def smoke():
    return _load("oracle_windows_install_smoke", "windows_install_smoke.py")


@pytest.fixture()
def layout(smoke, tmp_path: Path, monkeypatch):
    """A layout whose paths are the *Windows* ones, on any host."""
    manage = _load("oracle_manage_install_for_layout", "manage_install.py")
    from the_oracle import platform_support

    monkeypatch.setattr(manage, "is_windows", lambda: True)
    monkeypatch.setattr(manage, "is_linux", lambda: False)
    monkeypatch.setattr(platform_support, "is_windows", lambda: True)
    monkeypatch.setattr(platform_support, "is_linux", lambda: False)

    profile = tmp_path / "oracle smoke profile"
    profile.mkdir()
    return smoke.make_layout(profile, profile / "hf", profile / "pip-cache", manage=manage)


# --- the isolated environment -------------------------------------------------


def test_environment_isolates_the_profile_and_caches(smoke, layout, tmp_path: Path) -> None:
    env = smoke.build_environment(
        layout,
        base={"PATH": r"C:\Windows\System32", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        platform="nt",
    )

    profile = str(layout.profile)
    assert env["USERPROFILE"] == profile
    assert env["HOME"] == profile
    assert env["APPDATA"] == str(layout.profile / "AppData" / "Roaming")
    assert env["LOCALAPPDATA"] == str(layout.profile / "AppData" / "Local")
    assert env["HF_HOME"] == str(layout.hf_home)
    assert env["PIP_CACHE_DIR"] == str(layout.pip_cache)
    # A CI runner has no desktop; the GUI has to come up offscreen.
    assert env["QT_QPA_PLATFORM"] == "offscreen"
    # This job covers the online install; an inherited offline flag would make
    # the installer's doctor fail for the wrong reason.
    assert "HF_HUB_OFFLINE" not in env
    assert "TRANSFORMERS_OFFLINE" not in env
    # The launcher directory must be on PATH, or the doctor's entrypoint probe
    # fails on a machine whose launcher dir the user has not added yet.
    assert env["PATH"].split(";") == [str(layout.launcher_dir), r"C:\Windows\System32"]


def test_environment_keeps_the_base_environment(smoke, layout) -> None:
    env = smoke.build_environment(layout, base={"SystemRoot": r"C:\Windows", "PATH": ""}, platform="nt")

    assert env["SystemRoot"] == r"C:\Windows"
    assert env["PATH"] == str(layout.launcher_dir)


# --- paths come from the installer, not a second copy -------------------------


def test_layout_uses_the_installers_own_launcher_paths(smoke, layout, tmp_path: Path) -> None:
    appdata = layout.profile / "AppData" / "Roaming"
    assert layout.managed_launcher == appdata / "Python" / "Scripts" / "the-oracle.cmd"
    assert layout.start_menu_entry == (
        appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "The Oracle.cmd"
    )
    # Both directories must exist before the launchers are written.
    assert layout.managed_launcher.parent.is_dir()
    assert layout.start_menu_entry.parent.is_dir()
    # The profile path carries a space on purpose (quoting exercise).
    assert " " in str(layout.profile)


def test_layout_does_not_leak_the_temp_profile_into_the_environment(
    smoke, tmp_path: Path, monkeypatch
) -> None:
    """Resolving the launcher paths must not edit the shared process environment.

    ``make_layout`` has to put the temp profile's ``APPDATA`` into ``os.environ``
    to resolve the installer's Windows rules. Leaving it there outlives the only
    run that needs it: the profile is deleted when the test ends, so a later
    test that resolves an APPDATA-derived path gets a dangling one.
    """
    manage = _load("oracle_manage_install_for_leak", "manage_install.py")
    from the_oracle import platform_support

    monkeypatch.setattr(manage, "is_windows", lambda: True)
    monkeypatch.setattr(platform_support, "is_windows", lambda: True)
    monkeypatch.setattr(platform_support, "is_linux", lambda: False)

    monkeypatch.setenv("APPDATA", "C:\\existing\\Roaming")
    smoke.make_layout(tmp_path / "a", tmp_path / "a-hf", tmp_path / "a-pip", manage=manage)
    assert os.environ["APPDATA"] == "C:\\existing\\Roaming"

    monkeypatch.delenv("APPDATA", raising=False)
    smoke.make_layout(tmp_path / "b", tmp_path / "b-hf", tmp_path / "b-pip", manage=manage)
    assert "APPDATA" not in os.environ


def test_layout_agrees_with_the_installer_module(smoke, tmp_path: Path, monkeypatch) -> None:
    """A drift check: the smoke must not invent its own location rules."""
    manage = _load("oracle_manage_install_for_agreement", "manage_install.py")
    from the_oracle import platform_support

    monkeypatch.setattr(manage, "is_windows", lambda: True)
    monkeypatch.setattr(platform_support, "is_windows", lambda: True)
    monkeypatch.setattr(platform_support, "is_linux", lambda: False)
    profile = tmp_path / "profile"
    profile.mkdir()
    # Resolve the installer's own expectation under the same APPDATA the smoke
    # passes in, rather than inheriting one the smoke left behind.
    monkeypatch.setenv("APPDATA", str(profile / "AppData" / "Roaming"))

    layout = smoke.make_layout(profile, tmp_path / "hf", tmp_path / "pip", manage=manage)

    assert layout.managed_launcher == Path(manage.managed_launcher_path())
    assert layout.start_menu_entry == Path(manage.start_menu_launcher_path())


# --- the GUI wait ------------------------------------------------------------


def _fake_process(*, returncode: int | None) -> SimpleNamespace:
    return SimpleNamespace(pid=4242, returncode=returncode, poll=lambda: returncode)


def test_wait_accepts_a_live_process_reporting_a_main_window(smoke, tmp_path: Path) -> None:
    log = tmp_path / "gui_launch_timing.json"
    log.write_text(
        json.dumps({"events": [["qt_app_created", 0.1], ["mainwindow_built", 2.6]]}), encoding="utf-8"
    )

    smoke.wait_for_gui_window(log, process=_fake_process(returncode=None), timeout=5.0)


def test_wait_polls_until_the_window_appears(smoke, tmp_path: Path) -> None:
    """A slow GUI start must not be mistaken for a failure."""
    log = tmp_path / "gui_launch_timing.json"
    ticks = {"n": 0}

    def sleep(_seconds: float) -> None:
        ticks["n"] += 1
        if ticks["n"] == 2:
            log.write_text(json.dumps({"events": [["mainwindow_built", 3.0]]}), encoding="utf-8")

    clock = lambda: float(ticks["n"])  # noqa: E731 - a monotonic fake, advanced by sleep
    smoke.wait_for_gui_window(
        log, process=_fake_process(returncode=None), timeout=10.0, sleep=sleep, clock=clock
    )
    assert ticks["n"] == 2


def test_wait_reports_a_dead_process_with_its_exit_code(smoke, tmp_path: Path) -> None:
    with pytest.raises(smoke.SmokeFailure, match="exited with code 1"):
        smoke.wait_for_gui_window(
            tmp_path / "missing.json", process=_fake_process(returncode=1), timeout=5.0
        )


def test_wait_reports_a_timeout(smoke, tmp_path: Path) -> None:
    ticks = {"n": 0}

    def clock() -> float:
        ticks["n"] += 1
        return float(ticks["n"])

    with pytest.raises(smoke.SmokeFailure, match="did not report mainwindow_built"):
        smoke.wait_for_gui_window(
            tmp_path / "missing.json",
            process=_fake_process(returncode=None),
            timeout=2.0,
            sleep=lambda _s: None,
            clock=clock,
        )


def test_wait_rejects_a_stale_window_report_from_a_dead_process(smoke, tmp_path: Path) -> None:
    log = tmp_path / "gui_launch_timing.json"
    log.write_text(json.dumps({"events": [["mainwindow_built", 1.0]]}), encoding="utf-8")

    with pytest.raises(smoke.SmokeFailure, match="had already exited"):
        smoke.wait_for_gui_window(log, process=_fake_process(returncode=3), timeout=5.0)


def test_launch_timing_events_tolerates_unusable_logs(smoke, tmp_path: Path) -> None:
    assert smoke.launch_timing_events(tmp_path / "absent.json") == []
    garbage = tmp_path / "garbage.json"
    garbage.write_text("not json", encoding="utf-8")
    assert smoke.launch_timing_events(garbage) == []
    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text(json.dumps({"events": "nope"}), encoding="utf-8")
    assert smoke.launch_timing_events(wrong_shape) == []


def test_kill_command_takes_down_the_process_tree(smoke) -> None:
    assert smoke.kill_tree_command(4242) == ["taskkill", "/T", "/F", "/PID", "4242"]


@pytest.mark.skipif(os.name == "nt", reason="asserts the refusal on non-Windows hosts")
def test_refuses_to_run_off_windows(smoke, tmp_path: Path) -> None:
    assert smoke.main(["--profile", str(tmp_path)]) == 2


# --- the CI job's declared contract -------------------------------------------


def _logical_commands(script: str) -> list[str]:
    """Join PowerShell backtick continuations so a command can be asserted whole."""
    commands: list[str] = []
    pending = ""
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith("`"):
            pending += stripped[:-1].rstrip() + " "
            continue
        commands.append((pending + stripped).strip())
        pending = ""
    if pending:
        commands.append(pending.strip())
    return commands


def test_workflow_runs_the_windows_install_smoke() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["windows-install-smoke"]

    assert job["runs-on"] == "windows-latest"
    steps = job["steps"]
    uses = [step.get("uses", "") for step in steps]
    assert "actions/checkout@v4" in uses
    assert "actions/setup-python@v5" in uses
    assert "actions/cache@v4" in uses

    cache = next(step for step in steps if step.get("uses") == "actions/cache@v4")
    assert "${{ runner.temp }}/oracle-smoke/hf" in cache["with"]["path"]
    # The cache key follows the pins, so a revision bump can never reuse a stale
    # checkpoint.
    assert "hashFiles('src/the_oracle/models/pins.py')" in cache["with"]["key"]

    ffmpeg = next(step for step in steps if step.get("name") == "Ensure ffmpeg Is Available")
    assert ffmpeg["shell"] == "pwsh"
    assert "choco install ffmpeg" in ffmpeg["run"]

    smoke_step = next(
        step for step in steps if step.get("name") == "Install End To End And Exercise The Launchers"
    )
    assert smoke_step["shell"] == "pwsh"
    run = smoke_step["run"]
    # The invocation has to be one continued command: if a continuation is
    # dropped, the step silently runs the script with no arguments and the
    # flags below land in a separate, meaningless command.
    invocation = next(
        command
        for command in _logical_commands(run)
        if command.startswith("python scripts/windows_install_smoke.py")
    )
    # Exact flag/value pairs: a renamed or dropped flag must fail here rather
    # than silently leaving the model cache outside CI's cache step.
    assert r'--profile "${{ runner.temp }}\oracle smoke profile"' in invocation
    assert r'--hf-home "${{ runner.temp }}\oracle-smoke\hf"' in invocation
    assert r'--pip-cache "${{ runner.temp }}\oracle-smoke\pip-cache"' in invocation
    # A space in the profile path is deliberate: quoting must survive it.
    assert " " in invocation.split("--profile ")[1].split('"')[1]
    # PowerShell line continuations must be well formed, or the step runs
    # `python script.py` with no arguments and fails confusingly.
    continuation = [line for line in run.splitlines() if line.rstrip().endswith("`")]
    assert continuation, "the smoke invocation should be continued across lines"
    assert not run.splitlines()[-1].rstrip().endswith("`"), "the last line must not continue"
