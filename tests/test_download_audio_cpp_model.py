"""Deterministic tests for scripts/download_audio_cpp_model.sh (offline).

The script wraps audio.cpp's tools/model_manager_v2.py. These tests stub that
manager with a tiny fake so the export line and install layout are verified
without any network access.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "download_audio_cpp_model.sh"

FAKE_MANAGER = """#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


KNOWN = {"chatterbox_q8_0", "chatterbox_f16"}


def _info(package: str) -> dict:
    file_name = "chatterbox-f16.gguf" if package == "chatterbox_f16" else "chatterbox-q8_0.gguf"
    return {
        "family": "chatterbox",
        "id": package,
        "display_name": "Chatterbox GGUF",
        "format": "gguf",
        "precision": "f16" if package == "chatterbox_f16" else "q8_0",
        "default": package != "chatterbox_f16",
        "target_directory": "Chatterbox-GGUF",
        "files": [f"Chatterbox-GGUF/{file_name}"],
        "strip_prefix": "Chatterbox-GGUF",
        "download": {"kind": "huggingface_snapshot", "repo": "stub", "revision": "main", "gated": False},
    }


def _resolve(package: str) -> dict:
    if package not in KNOWN:
        print(f"ERROR: unknown package: {package}", file=sys.stderr)
        raise SystemExit(2)
    return _info(package)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    info_p = sub.add_parser("info")
    info_p.add_argument("package")
    info_p.add_argument("--json", action="store_true")
    install_p = sub.add_parser("install")
    install_p.add_argument("package")
    install_p.add_argument("--models-root", default="models")
    install_p.add_argument("--dry-run", action="store_true")
    install_p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.command == "info":
        print(json.dumps(_resolve(args.package)))
        return 0

    info = _resolve(args.package)
    target = Path(args.models_root) / info["target_directory"]
    file_name = info["files"][0].split("/")[-1]
    print(f"target {target}")
    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)
        (target / file_name).write_bytes(b"stub-model")
    print(f"installed {args.package} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


@pytest.fixture()
def fake_audio_cpp(tmp_path: Path) -> Path:
    """A minimal audio.cpp checkout with a stub model manager."""
    tools = tmp_path / "audio.cpp" / "tools"
    tools.mkdir(parents=True)
    (tools / "model_manager_v2.py").write_text(FAKE_MANAGER, encoding="utf-8")
    return tmp_path / "audio.cpp"


def _run_script(audio_cpp: Path, *args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "AUDIOCPP_DIR": str(audio_cpp), **(extra_env or {})}
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_script_installs_model_and_prints_export_line(fake_audio_cpp: Path) -> None:
    result = _run_script(fake_audio_cpp, extra_env={"AUDIOCPP_MODELS_ROOT": str(fake_audio_cpp / "models")})

    assert result.returncode == 0, result.stdout + result.stderr
    expected = fake_audio_cpp / "models" / "Chatterbox-GGUF" / "chatterbox-q8_0.gguf"
    assert expected.exists()
    assert f'export ORACLE_AUDIOCPP_MODEL="{expected}"' in result.stdout


def test_script_prints_correct_line_when_model_package_override(fake_audio_cpp: Path) -> None:
    result = _run_script(
        fake_audio_cpp,
        extra_env={"AUDIOCPP_MODEL_PACKAGE": "chatterbox_f16", "AUDIOCPP_MODELS_ROOT": str(fake_audio_cpp / "models")},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    expected = fake_audio_cpp / "models" / "Chatterbox-GGUF" / "chatterbox-f16.gguf"
    assert f'export ORACLE_AUDIOCPP_MODEL="{expected}"' in result.stdout
    assert expected.exists()


def test_script_fails_cleanly_on_unknown_package(fake_audio_cpp: Path) -> None:
    result = _run_script(
        fake_audio_cpp,
        extra_env={"AUDIOCPP_MODEL_PACKAGE": "does_not_exist", "AUDIOCPP_MODELS_ROOT": str(fake_audio_cpp / "models")},
    )

    assert result.returncode != 0
    assert "unknown package" in result.stderr


def test_script_dry_run_does_not_download(fake_audio_cpp: Path) -> None:
    result = _run_script(fake_audio_cpp, "--dry-run", extra_env={"AUDIOCPP_MODELS_ROOT": str(fake_audio_cpp / "models")})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "not downloaded" in result.stdout
    assert not (fake_audio_cpp / "models" / "Chatterbox-GGUF" / "chatterbox-q8_0.gguf").exists()


def test_script_rejects_user_supplied_models_root_flag(fake_audio_cpp: Path) -> None:
    """The script already passes --models-root itself; a user-supplied one would
    download to a different directory than the printed export line points at, so
    it must fail fast before any download happens."""
    result = _run_script(fake_audio_cpp, "--models-root", "/somewhere/else")

    assert result.returncode != 0
    assert "AUDIOCPP_MODELS_ROOT" in result.stderr
    # Nothing was installed before the guard fired.
    assert not (fake_audio_cpp / "models" / "Chatterbox-GGUF").exists()


def test_script_rejects_models_root_equals_form(fake_audio_cpp: Path) -> None:
    result = _run_script(fake_audio_cpp, "--models-root=/somewhere/else")

    assert result.returncode != 0
    assert "AUDIOCPP_MODELS_ROOT" in result.stderr


def test_script_fails_clearly_when_audio_cpp_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no_audio_cpp"
    result = _run_script(missing, extra_env={"AUDIOCPP_MODELS_ROOT": str(missing / "models")})

    assert result.returncode != 0
    assert "model manager not found" in result.stderr
