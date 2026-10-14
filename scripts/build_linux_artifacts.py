"""Build unsigned Linux review artifacts without the production-key gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import release  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(repo: Path, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    version = release.read_version(repo)
    artifacts = release.build_artifacts(repo, outdir, version)
    subprocess.run(
        [sys.executable, "-m", "piplicenses", "--format=json", "--with-urls", "--output-file", str(outdir / "sbom.json")],
        cwd=repo,
        check=True,
    )
    manifest = outdir / f"checksums-{version}.sha256"
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in [*artifacts, outdir / "sbom.json"]),
        encoding="utf-8",
        newline="\n",
    )
    return [*artifacts, outdir / "sbom.json", manifest]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("release_artifacts/linux-unsigned"))
    args = parser.parse_args(argv)
    repo = REPO_ROOT
    paths = build(repo, (repo / args.outdir).resolve() if not args.outdir.is_absolute() else args.outdir)
    print(json.dumps({"unsigned": True, "artifacts": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
