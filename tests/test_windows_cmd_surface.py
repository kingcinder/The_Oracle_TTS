from pathlib import Path


def test_windows_cmd_wrappers_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("bootstrap_windows.cmd", "doctor_windows.cmd", "run_windows.cmd"):
        assert (repo_root / name).is_file()


def test_windows_cmd_wrappers_use_repo_venv_python() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("bootstrap_windows.cmd", "doctor_windows.cmd", "run_windows.cmd"):
        content = (repo_root / name).read_text(encoding="utf-8")
        assert r".venv\Scripts\python.exe" in content
