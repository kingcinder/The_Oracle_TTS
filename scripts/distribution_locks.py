"""Create hash-locked wheel requirements and release manifests."""

import hashlib
import json
from pathlib import Path

from packaging.utils import parse_wheel_filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lock(
    wheels_dir: Path,
    lock_path: Path,
    manifest_path: Path,
    *,
    platform: str,
    python_version: str,
) -> None:
    wheels = sorted(wheels_dir.glob("*.whl"), key=lambda path: path.name)
    if not wheels:
        raise ValueError(f"no wheels found in {wheels_dir}")
    records: list[dict[str, str]] = []
    for wheel in wheels:
        name, version, _build, _tags = parse_wheel_filename(wheel.name)
        records.append(
            {
                "file": wheel.name,
                "name": str(name),
                "version": str(version),
                "sha256": _sha256(wheel),
            }
        )

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated from the release wheel set; do not edit by hand.",
        f"# platform={platform} python={python_version}",
        "--require-hashes",
        "--no-index",
        f"--find-links wheels/{platform}",
    ]
    lines.extend(
        f"{record['name']}=={record['version']} --hash=sha256:{record['sha256']}"
        for record in records
    )
    lock_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "app": "the-oracle", "python_version": python_version, "platforms": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("schema", 1)
    manifest.setdefault("app", "the-oracle")
    manifest.setdefault("python_version", python_version)
    manifest.setdefault("platforms", {})
    manifest["platforms"][platform] = {"lockfile": lock_path.name, "wheels": records}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
