from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_DIR = REPO_ROOT / "packaging"


@pytest.mark.parametrize("name", ["_launcher.sh", "install.sh", "uninstall.sh"])
def test_desktop_install_chain_scripts_are_executable(name: str) -> None:
    script = PACKAGING_DIR / name
    assert script.stat().st_mode & 0o111, f"desktop install chain cannot execute {script}"


@pytest.mark.parametrize(
    ("wrapper", "target"),
    [
        ("install.sh", "install_oracle_tts.sh"),
        ("uninstall.sh", "uninstall_oracle_tts.sh"),
    ],
)
def test_packaging_wrappers_resolve_root_script_independent_of_cwd(
    tmp_path: Path, wrapper: str, target: str
) -> None:
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    capture = tmp_path / "argv.txt"
    fake_bash = stub_bin / "bash"
    fake_bash.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"\n",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)

    result = subprocess.run(
        ["/bin/bash", str(PACKAGING_DIR / wrapper), "--sentinel"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{stub_bin}:{os.environ['PATH']}",
            "CAPTURE_ARGS": str(capture),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    actual_args = capture.read_text(encoding="utf-8").splitlines()
    assert Path(actual_args[0]).resolve() == REPO_ROOT / target
    assert actual_args[1:] == ["--sentinel"]
