from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from the_oracle import vulkan_prebuilt


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_packager():
    spec = importlib.util.spec_from_file_location(
        "oracle_package_vulkan_binary", REPO_ROOT / "scripts" / "package_vulkan_binary.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_package_manifest_records_pinned_commit_and_hash(tmp_path: Path) -> None:
    packager = _load_packager()
    binary = tmp_path / "audiocpp_cli"
    binary.write_bytes(b"vulkan binary")
    binary.chmod(0o755)
    artifacts = packager.package(binary, tmp_path / "out", version="1.5.0")
    payload = json.loads(artifacts.manifest.read_text(encoding="utf-8"))

    assert payload["audio_cpp_commit"] == packager.AUDIOCPP_PINNED_COMMIT
    assert payload["artifacts"][0]["name"] == artifacts.archive.name
    assert payload["artifacts"][0]["sha256"] == hashlib.sha256(artifacts.archive.read_bytes()).hexdigest()


def test_prebuilt_install_verifies_hash_and_extracts_binary(tmp_path: Path) -> None:
    packager = _load_packager()
    binary = tmp_path / "audiocpp_cli"
    binary.write_bytes(b"vulkan binary")
    binary.chmod(0o755)
    artifacts = packager.package(binary, tmp_path / "out", version="1.5.0")
    destination = tmp_path / "installed" / "audiocpp_cli"

    installed = vulkan_prebuilt.install_from_manifest(artifacts.manifest, artifacts.archive, destination)
    assert installed == destination
    assert destination.read_bytes() == b"vulkan binary"
    assert destination.stat().st_mode & 0o111


def test_prebuilt_install_refuses_changed_archive(tmp_path: Path) -> None:
    packager = _load_packager()
    binary = tmp_path / "audiocpp_cli"
    binary.write_bytes(b"vulkan binary")
    binary.chmod(0o755)
    artifacts = packager.package(binary, tmp_path / "out", version="1.5.0")
    artifacts.archive.write_bytes(b"tampered")

    with pytest.raises(vulkan_prebuilt.PrebuiltArtifactError, match="SHA-256"):
        vulkan_prebuilt.install_from_manifest(
            artifacts.manifest, artifacts.archive, tmp_path / "installed" / "audiocpp_cli"
        )


def test_prebuilt_download_verifies_manifest_before_install(tmp_path: Path) -> None:
    packager = _load_packager()
    binary = tmp_path / "audiocpp_cli"
    binary.write_bytes(b"vulkan binary")
    binary.chmod(0o755)
    artifacts = packager.package(binary, tmp_path / "out", version="1.5.0")
    files = {"manifest.json": artifacts.manifest.read_bytes(), artifacts.archive.name: artifacts.archive.read_bytes()}

    def fetch(url: str, target: Path) -> None:
        target.write_bytes(files[Path(url).name])

    destination = tmp_path / "installed" / "audiocpp_cli"
    assert vulkan_prebuilt.download_and_install(
        "https://updates.example/manifest.json",
        f"https://updates.example/{artifacts.archive.name}",
        destination,
        cache_dir=tmp_path / "cache",
        fetch=fetch,
    ) == destination
