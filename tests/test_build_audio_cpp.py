"""Deterministic tests for scripts/build_audio_cpp.sh (offline).

The real build clones audio.cpp and compiles with CMake plus the Vulkan SDK,
which is far too heavy for tests. These tests stub the audio.cpp checkout
with a fake build_linux.sh (which produces the expected binary at the
documented path), a fake model manager (the same stub used for
download_audio_cpp_model.sh tests), and a no-op patch script (via the
PATCH_AUDIOCPP_GGML_SH override), so the --with-model / SKIP_MODEL_DOWNLOAD
wiring is verified without any network access or compilation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_audio_cpp.sh"

from tests.test_download_audio_cpp_model import FAKE_MANAGER  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("cmake") is None,
    reason="cmake is not installed (the real build requires it)",
)

FAKE_BUILD_LINUX = """#!/usr/bin/env bash
# Minimal stand-in for audio.cpp's scripts/build_linux.sh: accepts the flags
# audio.cpp's helper would receive and produces the binary the build script
# expects at its documented path. FAKE_NO_BINARY=1 simulates a build that
# silently produced no binary.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "fake build_linux.sh ran with: $*"
if [[ "${FAKE_NO_BINARY:-0}" != "1" ]]; then
  mkdir -p "$ROOT/build/linux-vulkan-release/bin"
  printf '#!/bin/sh\\necho fake audiocpp_cli\\n' > "$ROOT/build/linux-vulkan-release/bin/audiocpp_cli"
  chmod +x "$ROOT/build/linux-vulkan-release/bin/audiocpp_cli"
fi
"""

FAKE_PATCH = """#!/usr/bin/env bash
echo "fake patch_audio_cpp_ggml.sh (no-op)"
"""


@pytest.fixture()
def fake_audio_cpp(tmp_path: Path) -> tuple[Path, Path]:
    """(audio.cpp root, no-op patch script path).

    The checkout has a .git dir so the clone step is skipped, a stub
    build_linux.sh, the same fake model manager the download tests use, and a
    no-op patch script that would otherwise rewrite scripts/patches/ artifacts
    or fail on the missing ggml source tree.
    """
    root = tmp_path / "audio.cpp"
    (root / ".git").mkdir(parents=True)
    scripts = root / "scripts"
    scripts.mkdir()
    build_linux = scripts / "build_linux.sh"
    build_linux.write_text(FAKE_BUILD_LINUX, encoding="utf-8")
    build_linux.chmod(0o755)
    tools = root / "tools"
    tools.mkdir()
    (tools / "model_manager_v2.py").write_text(FAKE_MANAGER, encoding="utf-8")
    patch = tmp_path / "patch_audio_cpp_ggml.sh"
    patch.write_text(FAKE_PATCH, encoding="utf-8")
    patch.chmod(0o755)
    return root, patch


def _run_script(
    audio_cpp: Path,
    patch: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "AUDIOCPP_DIR": str(audio_cpp),
        "PATCH_AUDIOCPP_GGML_SH": str(patch),
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _expected_binary(audio_cpp: Path) -> Path:
    return audio_cpp / "build" / "linux-vulkan-release" / "bin" / "audiocpp_cli"


def _expected_model(audio_cpp: Path) -> Path:
    return audio_cpp / "models" / "Chatterbox-GGUF" / "chatterbox-q8_0.gguf"


def test_default_build_skips_model_download(fake_audio_cpp: tuple[Path, Path]) -> None:
    audio_cpp, patch = fake_audio_cpp
    result = _run_script(audio_cpp, patch)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _expected_binary(audio_cpp).exists()
    # No model was fetched unless --with-model is passed.
    assert not _expected_model(audio_cpp).exists()
    assert "download_audio_cpp_model.sh" in result.stdout  # next-steps pointer


def test_with_model_builds_and_downloads_model(fake_audio_cpp: tuple[Path, Path]) -> None:
    audio_cpp, patch = fake_audio_cpp
    result = _run_script(audio_cpp, patch, "--with-model")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _expected_binary(audio_cpp).exists()
    model = _expected_model(audio_cpp)
    assert model.exists()
    assert f'export ORACLE_AUDIOCPP_MODEL="{model}"' in result.stdout
    assert "Build and model complete" in result.stdout


def test_skip_model_download_env_overrides_with_model(fake_audio_cpp: tuple[Path, Path]) -> None:
    audio_cpp, patch = fake_audio_cpp
    result = _run_script(audio_cpp, patch, "--with-model", extra_env={"SKIP_MODEL_DOWNLOAD": "1"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert _expected_binary(audio_cpp).exists()
    assert not _expected_model(audio_cpp).exists()


def test_unknown_argument_fails(fake_audio_cpp: tuple[Path, Path]) -> None:
    audio_cpp, patch = fake_audio_cpp
    result = _run_script(audio_cpp, patch, "--bogus")

    assert result.returncode != 0
    assert "unknown argument" in result.stderr
    assert not _expected_binary(audio_cpp).exists()


def test_help_lists_with_model_flag(fake_audio_cpp: tuple[Path, Path]) -> None:
    audio_cpp, patch = fake_audio_cpp
    result = _run_script(audio_cpp, patch, "--help")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--with-model" in result.stdout
    assert "SKIP_MODEL_DOWNLOAD" in result.stdout
    # Help exits before any build or download happens.
    assert not _expected_binary(audio_cpp).exists()
    assert not _expected_model(audio_cpp).exists()


def test_with_model_refuses_when_binary_missing(fake_audio_cpp: tuple[Path, Path]) -> None:
    """--with-model must not fetch a model when the build produced no binary:
    the one-shot flow only makes sense if the CLI actually exists."""
    audio_cpp, patch = fake_audio_cpp
    result = _run_script(audio_cpp, patch, "--with-model", extra_env={"FAKE_NO_BINARY": "1"})

    assert result.returncode != 0
    assert "expected binary not found" in result.stderr
    assert not _expected_model(audio_cpp).exists()
