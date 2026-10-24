"""Validate a static update manifest without downloading or replacing files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """The manifest or selected artifact is not trustworthy."""


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read update manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ManifestError("unsupported update manifest format")
    if not isinstance(payload.get("version"), str) or not payload["version"]:
        raise ManifestError("update manifest has no version")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError("update manifest has no artifacts")
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ManifestError("update manifest contains an invalid artifact")
        if not _SHA256.fullmatch(str(item.get("sha256", ""))):
            raise ManifestError(f"invalid SHA-256 for {item['name']}")
        if not isinstance(item.get("size"), int) or item["size"] < 0:
            raise ManifestError(f"invalid size for {item['name']}")
    return payload


def verify_artifact(payload: dict[str, Any], directory: Path, name: str) -> Path:
    item = next((entry for entry in payload["artifacts"] if entry["name"] == name), None)
    if item is None:
        raise ManifestError(f"artifact is not listed in the manifest: {name}")
    path = (directory / name).resolve()
    if path.parent != directory.resolve():
        raise ManifestError("artifact name must not escape the artifact directory")
    if not path.is_file():
        raise ManifestError(f"artifact is missing: {path}")
    if path.stat().st_size != item["size"]:
        raise ManifestError(f"artifact size mismatch for {name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        raise ManifestError(f"artifact SHA-256 mismatch for {name}")
    return path


def update_available(current_version: str, payload: dict[str, Any]) -> bool:
    return str(payload["version"]) != str(current_version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--current-version", default=None)
    args = parser.parse_args(argv)
    try:
        payload = load_manifest(args.manifest)
        if args.current_version is not None:
            print("update available" if update_available(args.current_version, payload) else "already current")
        if args.artifact is not None:
            if args.artifact_dir is None:
                parser.error("--artifact-dir is required with --artifact")
            verify_artifact(payload, args.artifact_dir, args.artifact)
            print(f"verified: {args.artifact}")
        print(json.dumps({"version": payload["version"], "rollback": payload.get("rollback", "")}, indent=2))
        return 0
    except ManifestError as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
