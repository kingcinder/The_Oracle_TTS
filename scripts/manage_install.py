#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from the_oracle.platform_support import (  # noqa: E402
    is_linux,
    is_windows,
    managed_launcher_dir,
    managed_launcher_path,
    repo_bootstrap_display,
    venv_entrypoint_path,
    venv_python_path,
)


PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cpu"
MANAGED_WRAPPER_MARKER = "ORACLE_TTS_WRAPPER"
LINUX_DESKTOP_MARKER = "ORACLE_TTS_DESKTOP"
WINDOWS_START_MENU_MARKER = "ORACLE_TTS_START_MENU"
SUPPORTED_PYTHON_MIN = (3, 11)
SUPPORTED_PYTHON_MAX = (3, 13)


def info(message: str) -> None:
    print(f"[INFO] {message}")


def passed(message: str) -> None:
    print(f"PASS: {message}")


def fail(message: str, exit_code: int = 1) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return exit_code


def ensure_supported_python() -> None:
    version = sys.version_info[:3]
    if SUPPORTED_PYTHON_MIN <= version < SUPPORTED_PYTHON_MAX:
        return
    raise SystemExit(
        fail(
            "The Oracle requires Python 3.11 or 3.12. "
            f"Current interpreter: {sys.executable} ({sys.version.split()[0]})."
        )
    )


def run_command(args: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(args, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    return env


def ensure_venv() -> Path:
    python_path = venv_python_path(REPO_ROOT)
    if python_path.exists():
        passed(f"Reusing project venv at {python_path.parent.parent}")
        return python_path
    run_command([sys.executable, "-m", "venv", str(REPO_ROOT / ".venv")])
    passed(f"Created project venv at {REPO_ROOT / '.venv'}")
    return python_path


def install_dependencies(venv_python: Path, *, include_dev: bool = False) -> None:
    env = build_env()
    run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools<81", "wheel"], env=env)
    run_command(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--index-url",
            PYTORCH_INDEX_URL,
            "torch==2.6.0",
            "torchaudio==2.6.0",
            "torchvision==0.21.0",
        ],
        env=env,
    )
    extras = ".[ml,dev]" if include_dev else ".[ml]"
    run_command([str(venv_python), "-m", "pip", "install", "-e", extras], env=env)
    run_command(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "librosa==0.11.0",
            "s3tokenizer",
            "diffusers==0.29.0",
            "resemble-perth==1.0.1",
            "conformer==0.3.2",
            "safetensors==0.5.3",
            "spacy-pkuseg",
            "pykakasi==2.3.0",
            "pyloudnorm",
            "omegaconf",
        ],
        env=env,
    )
    run_command([str(venv_python), "-m", "pip", "install", "--no-deps", "chatterbox-tts==0.1.6"], env=env)
    passed("Installed The Oracle runtime bundle into the project venv")


def managed_wrapper_contents() -> str:
    entrypoint = venv_entrypoint_path(REPO_ROOT, "the-oracle")
    if is_windows():
        return (
            "@echo off\r\n"
            f"REM {MANAGED_WRAPPER_MARKER}\r\n"
            f"set \"REPO_ROOT={REPO_ROOT}\"\r\n"
            f"set \"VENV_ENTRYPOINT={entrypoint}\"\r\n"
            "if not exist \"%VENV_ENTRYPOINT%\" (\r\n"
            "  echo the-oracle is not installed in %REPO_ROOT%\\.venv\r\n"
            f"  echo Run {repo_bootstrap_display()} first.\r\n"
            "  exit /b 1\r\n"
            ")\r\n"
            "\"%VENV_ENTRYPOINT%\" %*\r\n"
        )
    return (
        "#!/usr/bin/env bash\n"
        f"# {MANAGED_WRAPPER_MARKER}\n"
        "set -Eeuo pipefail\n\n"
        f'REPO_ROOT="{REPO_ROOT}"\n'
        f'VENV_ENTRYPOINT="{entrypoint}"\n\n'
        'if [[ ! -x "$VENV_ENTRYPOINT" ]]; then\n'
        '  printf \'the-oracle is not installed in %s\\n\' "$REPO_ROOT/.venv" >&2\n'
        f'  printf \'Run {repo_bootstrap_display()} first.\\n\' >&2\n'
        "  exit 1\n"
        "fi\n\n"
        'exec "$VENV_ENTRYPOINT" "$@"\n'
    )


def install_managed_wrapper() -> Path:
    wrapper_path = managed_launcher_path()
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(managed_wrapper_contents(), encoding="utf-8", newline="" if is_windows() else "\n")
    if not is_windows():
        wrapper_path.chmod(0o755)
    passed(f"Installed managed wrapper at {wrapper_path}")
    return wrapper_path


def desktop_file_path() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "applications" / "the-oracle.desktop"


def desktop_dir() -> Path:
    """The user's Desktop directory, honoring the freedesktop user-dirs spec.

    Reads ``XDG_DESKTOP_DIR`` from ``~/.config/user-dirs.dirs`` (the
    localized Desktop location on GNOME/KDE) and falls back to ``~/Desktop``
    when the setting or file is absent.
    """
    user_dirs = Path.home() / ".config" / "user-dirs.dirs"
    try:
        for line in user_dirs.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("XDG_DESKTOP_DIR="):
                value = stripped.split("=", 1)[1].strip().strip('"')
                if value:
                    # expandvars covers "$HOME/..."; expanduser also handles a
                    # literal "~/..." some user-dirs variants write.
                    return Path(os.path.expandvars(value)).expanduser()
    except OSError:
        pass
    return Path.home() / "Desktop"


def desktop_shortcut_path() -> Path:
    """Where the desktop launcher shortcut lives (same filename as the
    applications-menu entry so both are clearly The Oracle's)."""
    return desktop_dir() / "the-oracle.desktop"


def start_menu_launcher_path() -> Path:
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return root / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "The Oracle.cmd"


def desktop_entry_contents(*, trusted: bool = False) -> str:
    """The [Desktop Entry] body shared by the applications-menu entry and the
    desktop shortcut, so the two can never drift apart.

    ``Exec`` points at the absolute managed-launcher path rather than the
    bare ``the-oracle`` command, so the entry still works when
    ``~/.local/bin`` is missing from PATH (a fresh login may not have it).
    """
    # The freedesktop Desktop Entry spec only recognizes double-quote quoting
    # for Exec arguments (single quotes are reserved characters), so use the
    # full path wrapped in double quotes instead of shlex.quote.
    exec_line = f'Exec="{managed_launcher_path()}" gui'
    lines = [
        "[Desktop Entry]",
        "Name=The Oracle",
        "Comment=Chatterbox-based two-speaker TTS",
        exec_line,
        "Terminal=false",
        "Type=Application",
        "Categories=AudioVideo;Utility;",
        "StartupNotify=true",
    ]
    if trusted:
        # GNOME only launches desktop files marked trusted; the applications-
        # menu copy does not need this key.
        lines.append("Trusted=true")
    lines.append(f"# {LINUX_DESKTOP_MARKER}")
    lines.append("")
    return "\n".join(lines)


def refresh_desktop_database(applications_dir: Path) -> None:
    """Best-effort refresh so the entry shows up in the app list immediately.

    ``update-desktop-database`` ships with desktop-file-utils; when absent the
    entry still appears after the next session login.
    """
    if not shutil.which("update-desktop-database"):
        return
    try:
        subprocess.run(["update-desktop-database", str(applications_dir)], check=False)
    except OSError:
        pass


def mark_desktop_shortcut_trusted(shortcut_path: Path) -> None:
    """Best-effort GNOME trust grant for a desktop .desktop file.

    Nautilus/GNOME Shell key trust off the gio ``metadata::trusted``
    attribute rather than the ``Trusted=true`` key (which is the KDE
    convention). Setting both covers every desktop. Silently skips when
    ``gio`` is unavailable; a missing trust mark only means Nautilus shows
    an "Allow Launching" prompt, never a failure.
    """
    if not shutil.which("gio"):
        return
    try:
        subprocess.run(["gio", "set", str(shortcut_path), "metadata::trusted", "true"], check=False)
    except OSError:
        pass


def install_desktop_shortcut() -> Path:
    """Put a launcher on the user's Desktop.

    The desktop copy is marked ``Trusted=true``, made executable, and (when
    ``gio`` exists) granted GNOME's ``metadata::trusted`` attribute so it runs
    on double-click instead of prompting about an untrusted launcher. Returns
    the created path.
    """
    shortcut_path = desktop_shortcut_path()
    shortcut_path.parent.mkdir(parents=True, exist_ok=True)
    shortcut_path.write_text(desktop_entry_contents(trusted=True), encoding="utf-8")
    shortcut_path.chmod(0o755)
    mark_desktop_shortcut_trusted(shortcut_path)
    passed(f"Installed desktop shortcut at {shortcut_path}")
    return shortcut_path


def install_desktop_launcher() -> None:
    if is_windows():
        launcher_path = start_menu_launcher_path()
        launcher_path.parent.mkdir(parents=True, exist_ok=True)
        launcher_path.write_text(
            (
                "@echo off\r\n"
                f"REM {WINDOWS_START_MENU_MARKER}\r\n"
                f"call \"{managed_launcher_path()}\" gui %*\r\n"
            ),
            encoding="utf-8",
            newline="",
        )
        passed(f"Installed Start Menu launcher at {launcher_path}")
        return
    if not is_linux():
        return
    launcher_path = desktop_file_path()
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_path.write_text(desktop_entry_contents(), encoding="utf-8")
    passed(f"Installed desktop entry at {launcher_path}")
    refresh_desktop_database(launcher_path.parent)
    install_desktop_shortcut()


def remove_if_managed(path: Path, marker: str) -> None:
    if not path.exists():
        return
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return
    if marker not in content:
        return
    path.unlink()
    passed(f"Removed managed file {path}")


def run_doctor(skip_model_init: bool = False, ci_mode: bool = False) -> int:
    venv_python = venv_python_path(REPO_ROOT)
    if not venv_python.exists():
        return fail(f"Oracle TTS venv is missing at {REPO_ROOT / '.venv'}")
    args = [str(venv_python), str(REPO_ROOT / "scripts" / "doctor.py"), "--repo-root", str(REPO_ROOT)]
    if skip_model_init:
        args.append("--skip-model-init")
    if ci_mode:
        args.append("--ci")
    completed = subprocess.run(args, cwd=REPO_ROOT, check=False)
    return completed.returncode


def run_gui() -> int:
    entrypoint = venv_entrypoint_path(REPO_ROOT, "the-oracle")
    if not entrypoint.exists():
        return fail(f"The Oracle is not bootstrapped yet. Run {repo_bootstrap_display()} first.")
    env = build_env()
    if is_linux():
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            return fail(f"No GUI display detected. Set DISPLAY or WAYLAND_DISPLAY, then rerun {REPO_ROOT / 'run_oracle_tts.sh'}.")
        env.setdefault("LTP_JAR_DIR_PATH", str(Path.home() / ".cache" / "language_tool_python" / "LanguageTool-6.8-SNAPSHOT"))
        if not env.get("QT_QPA_PLATFORM") and os.environ.get("DISPLAY"):
            env["QT_QPA_PLATFORM"] = "xcb"
    completed = subprocess.run([str(entrypoint), "gui"], cwd=REPO_ROOT, env=env, check=False)
    return completed.returncode


def bootstrap(skip_doctor: bool = False, *, include_dev: bool = False) -> int:
    ensure_supported_python()
    passed(f"Using Python {sys.version.split()[0]} from {sys.executable}")
    venv_python = ensure_venv()
    install_dependencies(venv_python, include_dev=include_dev)
    install_managed_wrapper()
    if str(managed_launcher_dir()) not in os.environ.get("PATH", "").split(os.pathsep):
        info(f"Note: {managed_launcher_dir()} is not on this shell PATH.")
    if skip_doctor:
        return 0
    doctor_status = run_doctor()
    if doctor_status != 0:
        print("FAIL: Bootstrap verification found one or more blocking issues.", file=sys.stderr)
        return doctor_status
    passed("Bootstrap complete.")
    return 0


def install() -> int:
    status = bootstrap()
    if status != 0:
        return status
    install_desktop_launcher()
    doctor_status = run_doctor()
    if doctor_status != 0:
        return doctor_status
    passed("Install complete.")
    return 0


def uninstall() -> int:
    remove_if_managed(managed_launcher_path(), MANAGED_WRAPPER_MARKER)
    if is_windows():
        remove_if_managed(start_menu_launcher_path(), WINDOWS_START_MENU_MARKER)
    elif is_linux():
        remove_if_managed(desktop_file_path(), LINUX_DESKTOP_MARKER)
        remove_if_managed(desktop_shortcut_path(), LINUX_DESKTOP_MARKER)
    venv_dir = REPO_ROOT / ".venv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        passed(f"Removed virtualenv {venv_dir}")
    info("User projects, settings, and cached voices remain in place.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-platform install manager for The Oracle.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Create the venv and install runtime dependencies.")
    bootstrap_parser.add_argument("--skip-doctor", action="store_true")
    bootstrap_parser.add_argument("--include-dev", action="store_true")
    subparsers.add_parser("install", help="Bootstrap and register desktop/start-menu launchers.")
    doctor_parser = subparsers.add_parser("doctor", help="Run install diagnostics.")
    doctor_parser.add_argument("--skip-model-init", action="store_true")
    doctor_parser.add_argument("--ci", action="store_true")
    subparsers.add_parser("run", help="Launch the GUI.")
    subparsers.add_parser("uninstall", help="Remove managed launchers and the local venv.")

    args = parser.parse_args(argv)
    if args.command == "bootstrap":
        return bootstrap(skip_doctor=args.skip_doctor, include_dev=args.include_dev)
    if args.command == "install":
        return install()
    if args.command == "doctor":
        return run_doctor(skip_model_init=args.skip_model_init, ci_mode=args.ci)
    if args.command == "run":
        return run_gui()
    if args.command == "uninstall":
        return uninstall()
    return fail(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
