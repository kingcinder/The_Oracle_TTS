"""Tests for scripts/manage_install.py desktop integration (app list + desktop shortcut).

Uses the same importlib loader convention as test_doctor_vulkan.py so the
script's functions are tested without executing its __main__ path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MANAGE_INSTALL_PATH = SCRIPTS_DIR / "manage_install.py"


def _load_manage_install():
    spec = importlib.util.spec_from_file_location("oracle_manage_install", MANAGE_INSTALL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def manage(tmp_path: Path, monkeypatch):
    module = _load_manage_install()
    # Force Linux behavior and a fake home so nothing touches the real user.
    # Redirect $HOME too: desktop_dir() resolves "$HOME/..." via expandvars,
    # which reads the environment rather than Path.home().
    monkeypatch.setattr(module, "is_linux", lambda: True)
    monkeypatch.setattr(module, "is_windows", lambda: False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)  # no update-desktop-database
    return module


def test_desktop_file_path_uses_xdg_data_home(manage, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert manage.desktop_file_path() == tmp_path / "xdg" / "applications" / "the-oracle.desktop"


def test_desktop_dir_honors_user_dirs(manage, tmp_path, monkeypatch) -> None:
    config = tmp_path / ".config"
    config.mkdir()
    (config / "user-dirs.dirs").write_text('XDG_DESKTOP_DIR="$HOME/My Desktop"\n', encoding="utf-8")
    assert manage.desktop_dir() == tmp_path / "My Desktop"


def test_desktop_dir_falls_back_to_home_desktop(manage, tmp_path) -> None:
    assert manage.desktop_dir() == tmp_path / "Desktop"


def test_desktop_shortcut_path_sits_on_desktop(manage, tmp_path) -> None:
    assert manage.desktop_shortcut_path() == tmp_path / "Desktop" / "the-oracle.desktop"


def test_desktop_entry_contents_uses_absolute_launcher(manage) -> None:
    contents = manage.desktop_entry_contents()
    launcher = str(manage.managed_launcher_path())
    # The freedesktop Exec key requires double-quote quoting for arguments.
    assert f'Exec="{launcher}" gui' in contents
    assert manage.LINUX_DESKTOP_MARKER in contents
    assert "Trusted=true" not in contents  # applications-menu copy is not trusted


def test_desktop_entry_contents_trusted_variant_for_shortcut(manage) -> None:
    contents = manage.desktop_entry_contents(trusted=True)
    assert "Trusted=true" in contents
    assert manage.LINUX_DESKTOP_MARKER in contents


def test_install_desktop_launcher_writes_entry_and_shortcut(manage, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    manage.install_desktop_launcher()

    entry = tmp_path / "xdg" / "applications" / "the-oracle.desktop"
    shortcut = tmp_path / "Desktop" / "the-oracle.desktop"
    assert entry.exists()
    assert shortcut.exists()
    assert manage.LINUX_DESKTOP_MARKER in entry.read_text(encoding="utf-8")
    shortcut_text = shortcut.read_text(encoding="utf-8")
    assert "Trusted=true" in shortcut_text
    assert manage.LINUX_DESKTOP_MARKER in shortcut_text
    # Shortcut must be executable for GNOME to launch it.
    assert shortcut.stat().st_mode & 0o111


def test_mark_desktop_shortcut_trusted_runs_gio_when_available(manage, tmp_path, monkeypatch) -> None:
    """GNOME trust is granted via gio metadata::trusted when gio exists; a
    missing gio is a silent no-op, never a failure."""
    shortcut = tmp_path / "Desktop" / "the-oracle.desktop"
    shortcut.parent.mkdir(parents=True, exist_ok=True)
    shortcut.write_text(manage.desktop_entry_contents(trusted=True), encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(manage.shutil, "which", lambda _name: "/usr/bin/gio")
    monkeypatch.setattr(manage.subprocess, "run", lambda args, **kwargs: calls.append(list(args)))

    manage.mark_desktop_shortcut_trusted(shortcut)

    assert calls == [["gio", "set", str(shortcut), "metadata::trusted", "true"]]

    # gio absent: no attempt, no error.
    calls.clear()
    monkeypatch.setattr(manage.shutil, "which", lambda _name: None)
    manage.mark_desktop_shortcut_trusted(shortcut)
    assert calls == []


def test_install_desktop_launcher_is_windows_only_start_menu(manage, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(manage, "is_linux", lambda: False)
    monkeypatch.setattr(manage, "is_windows", lambda: True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    manage.install_desktop_launcher()

    launcher = tmp_path / "appdata" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "The Oracle.cmd"
    assert launcher.exists()
    assert "the-oracle" in launcher.read_text(encoding="utf-8")
    assert not (tmp_path / "Desktop").exists()  # no desktop shortcut on Windows


def test_uninstall_removes_managed_desktop_files(manage, tmp_path, monkeypatch) -> None:
    """Uninstall removes both the app-list entry and the desktop shortcut
    (and only files carrying The Oracle's marker)."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(manage.shutil, "rmtree", lambda _path: None)  # never delete the real venv

    # Create the managed files plus an unrelated desktop file that must stay.
    entry = tmp_path / "xdg" / "applications" / "the-oracle.desktop"
    shortcut = tmp_path / "Desktop" / "the-oracle.desktop"
    entry.parent.mkdir(parents=True, exist_ok=True)
    shortcut.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(manage.desktop_entry_contents(), encoding="utf-8")
    shortcut.write_text(manage.desktop_entry_contents(trusted=True), encoding="utf-8")
    unrelated = tmp_path / "Desktop" / "user-notes.txt"
    unrelated.write_text("not a launcher", encoding="utf-8")

    manage.uninstall()

    assert not entry.exists()
    assert not shortcut.exists()
    assert unrelated.exists()  # unmanaged files are left alone
