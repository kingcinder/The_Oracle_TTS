"""Platform-aware filesystem helpers for install and runtime tooling."""

from __future__ import annotations

import os
import sys
from pathlib import Path


WINDOWS_INVALID_PATH_CHARS = set('<>:"/\\|?*')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def is_windows() -> bool:
    return os.name == "nt"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def venv_bin_dir(repo_root: str | Path) -> Path:
    root = Path(repo_root).expanduser().resolve()
    return root / ".venv" / ("Scripts" if is_windows() else "bin")


def venv_python_path(repo_root: str | Path) -> Path:
    return venv_bin_dir(repo_root) / ("python.exe" if is_windows() else "python")


def venv_entrypoint_path(repo_root: str | Path, name: str) -> Path:
    bin_dir = venv_bin_dir(repo_root)
    candidates = (
        [bin_dir / f"{name}.exe", bin_dir / f"{name}.cmd", bin_dir / name, bin_dir / f"{name}-script.py"]
        if is_windows()
        else [bin_dir / name]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def user_config_root() -> Path:
    if is_windows():
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata)
        return Path.home() / "AppData" / "Roaming"
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    return Path(xdg_config) if xdg_config else Path.home() / ".config"


def app_config_dir(app_name: str = "the_oracle") -> Path:
    return user_config_root() / app_name


def managed_launcher_dir() -> Path:
    if is_windows():
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Python" / "Scripts"
        return Path.home() / "AppData" / "Roaming" / "Python" / "Scripts"
    xdg_bin = os.environ.get("XDG_BIN_HOME")
    return Path(xdg_bin) if xdg_bin else Path.home() / ".local" / "bin"


def managed_launcher_path(name: str = "the-oracle") -> Path:
    suffix = ".cmd" if is_windows() else ""
    return managed_launcher_dir() / f"{name}{suffix}"


def path_entries(path_value: str | None = None) -> list[str]:
    raw = os.environ.get("PATH", "") if path_value is None else path_value
    return [entry for entry in raw.split(os.pathsep) if entry]


def repo_python_display() -> str:
    return r".\.venv\Scripts\python.exe" if is_windows() else "./.venv/bin/python"


def repo_bootstrap_display(repo_root: str | Path | None = None) -> str:
    if is_windows():
        return r".\bootstrap_windows.cmd"
    return "./bootstrap_oracle_tts.sh"


def repo_run_display(repo_root: str | Path | None = None) -> str:
    if is_windows():
        return r".\run_windows.cmd"
    return "./run_oracle_tts.sh"


def invalid_windows_path_parts(path: str | Path) -> list[str]:
    candidate = Path(path)
    invalid: list[str] = []
    for part in candidate.parts:
        if part in {candidate.anchor, ".", ".."}:
            continue
        stem = Path(part).stem.upper().rstrip(" .")
        if any(char in WINDOWS_INVALID_PATH_CHARS for char in part):
            invalid.append(part)
            continue
        if stem in WINDOWS_RESERVED_NAMES:
            invalid.append(part)
            continue
        if part.endswith((" ", ".")):
            invalid.append(part)
    return invalid
