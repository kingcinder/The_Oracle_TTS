"""Package the pinned audio.cpp Vulkan CLI for unsigned CI publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

AUDIOCPP_PINNED_COMMIT = "748c5e28f6a7228b8f38ad7142ca97d29584544b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PackageResult:
    archive: Path
    manifest: Path


def package(binary: Path, outdir: Path, *, version: str) -> PackageResult:
    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        raise ValueError(f"Vulkan binary is missing or not executable: {binary}")
    outdir.mkdir(parents=True, exist_ok=True)
    archive = outdir / f"the-oracle-audiocpp-vulkan-{version}-linux-x86_64.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(binary, arcname="usr/libexec/audiocpp_cli")
    manifest = outdir / "audiocpp-vulkan-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": 1,
                "version": version,
                "audio_cpp_commit": AUDIOCPP_PINNED_COMMIT,
                "artifacts": [
                    {"name": archive.name, "sha256": _sha256(archive), "size": archive.stat().st_size}
                ],
                "unsigned": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return PackageResult(archive=archive, manifest=manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    result = package(args.binary, args.outdir, version=args.version)
    print(json.dumps({"archive": str(result.archive), "manifest": str(result.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
