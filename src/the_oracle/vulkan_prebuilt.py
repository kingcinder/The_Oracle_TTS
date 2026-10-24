"""Checksum-verified installation of a CI-built audio.cpp Vulkan binary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
from pathlib import Path
from typing import Any


class PrebuiltArtifactError(ValueError):
    """A prebuilt artifact is missing, tampered with, or unsafe to extract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrebuiltArtifactError(f"could not read prebuilt manifest: {exc}") from exc
    if payload.get("format") != 1 or not isinstance(payload.get("artifacts"), list):
        raise PrebuiltArtifactError("unsupported prebuilt manifest format")
    return payload


def _safe_extract(archive: Path, destination: Path) -> tuple[Path, Path]:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        binary_members = []
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                raise PrebuiltArtifactError(f"unsafe path in prebuilt archive: {member.name}")
            if member.isfile() and relative.name == "audiocpp_cli":
                binary_members.append(member)
        if len(binary_members) != 1:
            raise PrebuiltArtifactError("prebuilt archive must contain exactly one audiocpp_cli file")
        staging = destination.parent / f".{destination.name}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        handle.extractall(staging, filter="data")
    extracted = staging / binary_members[0].name
    if not extracted.is_file():
        candidates = list(staging.rglob("audiocpp_cli"))
        extracted = candidates[0] if len(candidates) == 1 else extracted
    if not extracted.is_file():
        shutil.rmtree(staging, ignore_errors=True)
        raise PrebuiltArtifactError("prebuilt archive did not extract audiocpp_cli")
    return extracted, staging


def install_from_manifest(manifest: Path, archive: Path, destination: Path) -> Path:
    payload = _load_manifest(manifest)
    item = next((entry for entry in payload["artifacts"] if entry.get("name") == archive.name), None)
    if item is None:
        raise PrebuiltArtifactError(f"archive is not listed in prebuilt manifest: {archive.name}")
    if archive.stat().st_size != item.get("size") or _sha256(archive) != item.get("sha256"):
        raise PrebuiltArtifactError(f"prebuilt archive SHA-256 mismatch: {archive.name}")
    extracted, staging = _safe_extract(archive, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.copy2(extracted, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    shutil.rmtree(staging, ignore_errors=True)
    return destination
