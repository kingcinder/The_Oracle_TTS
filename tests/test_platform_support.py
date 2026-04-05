from pathlib import Path

from the_oracle import platform_support


def test_app_config_dir_uses_xdg_on_non_windows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(platform_support, "is_windows", lambda: False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert platform_support.app_config_dir() == tmp_path / "the_oracle"


def test_app_config_dir_uses_appdata_on_windows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(platform_support, "is_windows", lambda: True)
    monkeypatch.setenv("APPDATA", str(tmp_path))

    assert platform_support.app_config_dir() == tmp_path / "the_oracle"
    assert platform_support.managed_launcher_dir() == tmp_path / "Python" / "Scripts"


def test_venv_entrypoint_path_switches_per_platform(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(platform_support, "is_windows", lambda: True)
    assert platform_support.venv_entrypoint_path(tmp_path, "the-oracle").name == "the-oracle.exe"

    monkeypatch.setattr(platform_support, "is_windows", lambda: False)
    assert platform_support.venv_entrypoint_path(tmp_path, "the-oracle").as_posix().endswith("/.venv/bin/the-oracle")


def test_invalid_windows_path_parts_flags_reserved_characters() -> None:
    assert platform_support.invalid_windows_path_parts("Input/What is, reality?.txt") == ["What is, reality?.txt"]
    assert platform_support.invalid_windows_path_parts("Input/portable_name.txt") == []
