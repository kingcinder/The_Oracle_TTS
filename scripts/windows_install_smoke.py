#!/usr/bin/env python3
"""End-to-end install smoke for Windows: a temp user profile, then the real launchers.

The suite covers what the installer *writes* (``tests/test_install_boundary.py``
pins the exact ``the-oracle.cmd`` and Start Menu contents). What it cannot cover
is whether those files *execute*: that needs cmd.exe on Windows, and the GUI the
Start Menu entry starts runs an event loop that has to be killed rather than
awaited. This script closes that gap, which is why it is a script rather than a
test — it needs a Windows shell, a throwaway profile and process control.

Run by ``.github/workflows/ci.yml`` (the ``windows-install-smoke`` job)::

    python scripts/windows_install_smoke.py --profile "%RUNNER_TEMP%\\oracle profile"

It works against a temp profile only: ``USERPROFILE``/``HOME``/``APPDATA``/
``LOCALAPPDATA`` are redirected there, the pinned model cache and pip cache go
to explicitly named directories so CI can cache them, and nothing reads or
writes the runner's real user profile. Steps, in order:

1. build that environment (the profile's launcher directory goes on ``PATH``,
   because that is the state a user who followed the install instructions is in),
2. run ``manage_install.py install``: venv, dependencies, managed ``.cmd``
   launcher, Start Menu entry, then the installer's own doctor,
3. execute the managed launcher for real (``the-oracle.cmd --help``),
4. launch the **Start Menu entry** — which ``call``s the managed launcher with
   ``gui`` — wait for the GUI to report a built main window, then kill it.

Every failure path prints the captured output and exits non-zero.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: Written by ``app_gui.launch_gui()`` immediately before it enters the Qt event
#: loop, so its presence proves the GUI built a main window rather than merely
#: starting a process.
GUI_LAUNCH_LOG = REPO_ROOT / "Output" / "logs" / "gui_launch_timing.json"

#: The marker event ``launch_gui()`` records once ``MainWindow()`` exists.
MAINWINDOW_EVENT = "mainwindow_built"


class SmokeFailure(RuntimeError):
    """A step of the smoke did not do what the installer promises."""


@dataclass(frozen=True)
class Layout:
    """The isolated locations this smoke uses, and the launchers it expects."""

    profile: Path
    hf_home: Path
    pip_cache: Path
    managed_launcher: Path
    start_menu_entry: Path

    @property
    def launcher_dir(self) -> Path:
        return self.managed_launcher.parent


def _load_manage_install(name: str = "oracle_manage_install_smoke"):
    """Load ``scripts/manage_install.py`` so the launcher paths come from it.

    The smoke must exercise the installer's own idea of where the launchers go,
    not a second copy of those rules that can drift out of step.
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / "manage_install.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_environment(
    layout: Layout,
    *,
    base: Mapping[str, str],
    platform: str | None = None,
) -> dict[str, str]:
    """The child environment: temp profile, caches, launcher dir on ``PATH``.

    ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` are removed on purpose — this
    job covers the online install path, and an inherited offline flag would make
    the doctor's model init fail for the wrong reason.
    """
    env = dict(base)
    env["USERPROFILE"] = str(layout.profile)
    env["HOME"] = str(layout.profile)
    env["APPDATA"] = str(layout.profile / "AppData" / "Roaming")
    env["LOCALAPPDATA"] = str(layout.profile / "AppData" / "Local")
    env["HF_HOME"] = str(layout.hf_home)
    env["PIP_CACHE_DIR"] = str(layout.pip_cache)
    # No interactive desktop on a CI runner: the GUI has to come up offscreen.
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONUNBUFFERED"] = "1"
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(name, None)
    separator = ";" if (platform or os.name) == "nt" else os.pathsep
    entries = [str(layout.launcher_dir)]
    entries += [entry for entry in base.get("PATH", "").split(separator) if entry]
    env["PATH"] = separator.join(entries)
    return env


def make_layout(profile: Path, hf_home: Path, pip_cache: Path, *, manage=None) -> Layout:
    """Resolve the profile's launcher paths through the installer's own rules.

    ``manage`` is injectable so a test can force the Windows view of the paths
    on a non-Windows host instead of duplicating those rules here. Nothing
    outside the returned ``Layout`` is left modified: the environment the rules
    need to run is put back exactly as it was found.
    """
    manage = manage or _load_manage_install()
    # The installer resolves launcher locations from APPDATA, so the temp profile
    # has to be visible in the environment while those rules run -- but only
    # while they run. The process environment is shared: a deleted temp profile
    # left in it would be resolved by whatever runs next.
    previous_appdata = os.environ.get("APPDATA")
    os.environ["APPDATA"] = str(profile / "AppData" / "Roaming")
    try:
        managed = Path(manage.managed_launcher_path())
        start_menu = Path(manage.start_menu_launcher_path())
    finally:
        if previous_appdata is None:
            os.environ.pop("APPDATA", None)
        else:
            os.environ["APPDATA"] = previous_appdata
    for directory in (profile, hf_home, pip_cache, managed.parent, start_menu.parent):
        directory.mkdir(parents=True, exist_ok=True)
    return Layout(
        profile=profile,
        hf_home=hf_home,
        pip_cache=pip_cache,
        managed_launcher=managed,
        start_menu_entry=start_menu,
    )


def kill_tree_command(pid: int) -> list[str]:
    """``taskkill`` argv that takes down the cmd -> python -> Qt tree."""
    return ["taskkill", "/T", "/F", "/PID", str(pid)]


def run_streaming(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    tail_lines: int = 60,
) -> tuple[int, str]:
    """Run a step, echo its output live (CI logs), and keep the tail for errors."""
    process = subprocess.Popen(
        list(argv),
        cwd=str(cwd),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    captured: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line.rstrip("\n"))
    return process.wait(), "\n".join(captured[-tail_lines:])


def launch_timing_events(log_path: Path) -> list[str]:
    """Event names recorded by ``launch_gui()``, or ``[]`` if not there yet."""
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    events = payload.get("events")
    if not isinstance(events, list):
        return []
    names: list[str] = []
    for event in events:
        if isinstance(event, (list, tuple)) and event:
            names.append(str(event[0]))
    return names


def wait_for_gui_window(
    log_path: Path,
    *,
    process: subprocess.Popen,
    timeout: float,
    poll: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Wait for a *live* process to report a built main window.

    Raises if the process dies first: that is the interesting failure (the
    launcher started and the GUI crashed), so it must not be masked by a
    timeout, and the caller's captured output explains why.
    """
    deadline = clock() + timeout
    while True:
        if MAINWINDOW_EVENT in launch_timing_events(log_path):
            if process.poll() is not None:
                raise SmokeFailure(
                    f"the GUI reported {MAINWINDOW_EVENT} but had already exited "
                    f"(code {process.returncode})"
                )
            return
        if process.poll() is not None:
            raise SmokeFailure(
                f"the Start Menu entry exited with code {process.returncode} before the "
                f"GUI reported {MAINWINDOW_EVENT}"
            )
        if clock() >= deadline:
            raise SmokeFailure(
                f"the GUI did not report {MAINWINDOW_EVENT} within {timeout:.0f}s"
            )
        sleep(poll)


def _step(description: str) -> None:
    print(f"\n=== {description} ===", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, required=True, help="Throwaway user profile directory.")
    parser.add_argument("--hf-home", type=Path, default=None, help="Model cache (cacheable; defaults under the profile).")
    parser.add_argument("--pip-cache", type=Path, default=None, help="pip cache directory (cacheable).")
    parser.add_argument("--gui-timeout", type=float, default=300.0, help="Seconds to wait for the GUI window.")
    args = parser.parse_args(argv)

    if os.name != "nt":
        print(
            "FAIL: this smoke exercises the Windows .cmd launcher and Start Menu entry; "
            "it only runs on Windows.",
            file=sys.stderr,
        )
        return 2

    profile = args.profile.expanduser().resolve()
    hf_home = (args.hf_home or profile / "hf").expanduser().resolve()
    pip_cache = (args.pip_cache or profile / "pip-cache").expanduser().resolve()
    layout = make_layout(profile, hf_home, pip_cache)
    env = build_environment(layout, base=os.environ)

    print("Profile:           ", layout.profile)
    print("Launcher:          ", layout.managed_launcher)
    print("Start Menu entry:  ", layout.start_menu_entry)
    print("Model cache:       ", layout.hf_home)

    _step("install (venv, dependencies, launchers, doctor)")
    status, tail = run_streaming(
        [sys.executable, str(SCRIPTS_DIR / "manage_install.py"), "install", "--pytorch-runtime", "cpu"],
        env=env,
        cwd=REPO_ROOT,
    )
    if status != 0:
        raise SmokeFailure(f"the installer exited {status}:\n{tail}")
    for expected in (layout.managed_launcher, layout.start_menu_entry):
        if not expected.is_file():
            raise SmokeFailure(f"the installer did not create {expected}")

    _step(f"execute the managed launcher: {layout.managed_launcher.name} --help")
    status, tail = run_streaming(
        ["cmd", "/d", "/c", str(layout.managed_launcher), "--help"],
        env=env,
        cwd=REPO_ROOT,
    )
    if status != 0:
        raise SmokeFailure(
            f"the managed launcher exited {status} (expected 0):\n{tail}"
        )
    if "usage" not in tail.lower():
        raise SmokeFailure(f"the managed launcher printed no usage output:\n{tail}")

    _step(f"launch the GUI through the Start Menu entry: {layout.start_menu_entry.name}")
    # A previous run's log must not be able to satisfy the wait.
    GUI_LAUNCH_LOG.unlink(missing_ok=True)
    # Redirect the child's output to a file rather than a pipe: the GUI runs
    # until it is killed, so nobody is draining a pipe and a chatty startup
    # could fill the buffer and block the very process we are waiting on.
    gui_output = layout.profile / "gui-launch-output.txt"
    with gui_output.open("w", encoding="utf-8", errors="replace") as sink:
        process = subprocess.Popen(
            ["cmd", "/d", "/c", str(layout.start_menu_entry)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_gui_window(GUI_LAUNCH_LOG, process=process, timeout=args.gui_timeout)
            print(f"PASS: the GUI built its main window via the Start Menu entry (pid {process.pid})")
        except SmokeFailure as exc:
            raise SmokeFailure(f"{exc}\n--- Start Menu entry output ---\n{_tail_of(gui_output)}")
        finally:
            if process.poll() is None:
                subprocess.run(kill_tree_command(process.pid), check=False)

    print("\nPASS: Windows install smoke complete — installer, .cmd launcher and Start Menu entry all executed.")
    return 0


def _tail_of(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return "<no output captured>"


def _run(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except SmokeFailure as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
