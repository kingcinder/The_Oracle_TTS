"""Build unsigned, self-contained Linux x86_64 rehearsal artifacts.

The frozen application contains the Python runtime, Qt runtime, and CPU
runtime installed in the build environment. Models are deliberately excluded;
the application acquires them only after EULA acceptance and serial activation.
The owner supplies signing keys later, so this script emits unsigned artifacts
plus SHA-256 and static update manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = REPO_ROOT / "packaging" / "linux" / "the-oracle.spec"


class BuildError(RuntimeError):
    """The frozen artifact cannot be produced safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_host() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise BuildError("frozen Linux artifacts require a Linux x86_64 builder")


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise BuildError(f"required build tool is missing: {name}")
    return path


def write_checksums(outdir: Path, artifacts: list[Path], version: str) -> Path:
    manifest = outdir / f"checksums-{version}.sha256"
    lines = [f"{_sha256(path)}  {path.name}\n" for path in sorted(artifacts, key=lambda item: item.name)]
    manifest.write_text("".join(lines), encoding="utf-8", newline="\n")
    return manifest


def write_update_manifest(outdir: Path, artifacts: list[Path], version: str) -> Path:
    manifest = outdir / "update-manifest.json"
    payload = {
        "format": 1,
        "version": version,
        "artifacts": [
            {"name": path.name, "sha256": _sha256(path), "size": path.stat().st_size}
            for path in sorted(artifacts, key=lambda item: item.name)
        ],
        "rollback": "Install the previous verified artifact; no automatic replacement is performed.",
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def _copy_tree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, symlinks=True)
        else:
            shutil.copy2(child, target, follow_symlinks=False)


def stage_appdir(appdir: Path, frozen_dist: Path) -> Path:
    """Stage an AppDir from PyInstaller's one-folder output."""
    frozen = frozen_dist / "the-oracle"
    if not frozen.is_dir():
        raise BuildError(f"PyInstaller output is missing: {frozen}")
    if appdir.exists():
        shutil.rmtree(appdir)
    runtime = appdir / "usr" / "lib" / "the-oracle"
    _copy_tree_contents(frozen, runtime)
    (appdir / "usr" / "bin").mkdir(parents=True, exist_ok=True)
    launcher = appdir / "usr" / "bin" / "the-oracle"
    launcher.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'exec "$(dirname "$0")/../lib/the-oracle/the-oracle" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    applications = appdir / "usr" / "share" / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "packaging" / "linux" / "the-oracle.desktop", applications)
    icons = appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
    icons.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "packaging" / "icons" / "oracle-tts-install.png", icons / "the-oracle.png")
    apprun = appdir / "AppRun"
    apprun.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "# Per-user settings and models stay in ~/.config/the-oracle and ~/.cache;\n"
        "# uninstall removes this application tree only.\n"
        'HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"\n'
        'exec "$HERE/usr/bin/the-oracle" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    apprun.chmod(apprun.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return appdir


def _build_deb(appdir: Path, outdir: Path, version: str) -> Path:
    root = outdir / "deb-root"
    if root.exists():
        shutil.rmtree(root)
    _copy_tree_contents(appdir, root / "opt" / "the-oracle")
    (root / "usr" / "bin").mkdir(parents=True, exist_ok=True)
    (root / "usr" / "share" / "applications").mkdir(parents=True, exist_ok=True)
    link = root / "usr" / "bin" / "the-oracle"
    link.symlink_to("/opt/the-oracle/usr/bin/the-oracle")
    shutil.copy2(
        appdir / "usr" / "share" / "applications" / "the-oracle.desktop",
        root / "usr" / "share" / "applications",
    )
    control = root / "DEBIAN"
    control.mkdir(parents=True, exist_ok=True)
    (control / "control").write_text(
        "Package: the-oracle\n"
        f"Version: {version}\n"
        "Section: sound\nPriority: optional\nArchitecture: amd64\n"
        "Maintainer: The Oracle\n"
        "Description: The Oracle local text-to-speech application\n"
        " Frozen application; models are downloaded after activation.\n",
        encoding="utf-8",
        newline="\n",
    )
    deb = outdir / f"the-oracle_{version}_amd64.deb"
    subprocess.run([_require_tool("dpkg-deb"), "--build", "--root-owner-group", str(root), str(deb)], check=True)
    return deb


def build(repo: Path, outdir: Path, *, version: str, formats: tuple[str, ...] = ("appimage", "deb")) -> list[Path]:
    ensure_host()
    pyinstaller = _require_tool("pyinstaller")
    if "appimage" in formats:
        _require_tool("appimagetool")
    outdir.mkdir(parents=True, exist_ok=True)
    frozen_dist = outdir / "pyinstaller-dist"
    workpath = outdir / "pyinstaller-build"
    subprocess.run(
        [pyinstaller, "--noconfirm", "--clean", "--distpath", str(frozen_dist), "--workpath", str(workpath), str(SPEC)],
        cwd=repo,
        check=True,
    )
    appdir = stage_appdir(outdir / "AppDir", frozen_dist)
    artifacts: list[Path] = []
    if "appimage" in formats:
        appimage = outdir / f"the-oracle-{version}-x86_64.AppImage"
        subprocess.run([_require_tool("appimagetool"), str(appdir), str(appimage)], cwd=repo, check=True)
        artifacts.append(appimage)
    if "deb" in formats:
        artifacts.append(_build_deb(appdir, outdir, version))
    checksums = write_checksums(outdir, artifacts, version)
    update_manifest = write_update_manifest(outdir, artifacts, version)
    return [*artifacts, checksums, update_manifest]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("release_artifacts/linux-frozen"))
    parser.add_argument("--version", default=None)
    parser.add_argument("--format", choices=("appimage", "deb", "both"), default="both")
    args = parser.parse_args(argv)
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from release import read_version

    version = args.version or read_version(REPO_ROOT)
    formats = ("appimage", "deb") if args.format == "both" else (args.format,)
    paths = build(REPO_ROOT, args.outdir.resolve(), version=version, formats=formats)
    print(json.dumps({"unsigned": True, "artifacts": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
