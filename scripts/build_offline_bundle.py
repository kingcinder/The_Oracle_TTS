#!/usr/bin/env python3
"""Build a fully offline install bundle for The Oracle.

The bundle contains EVERYTHING needed to install and launch the program with
no network access, on Linux or Windows:

    <output>/
        manifest.json          # pins, versions, platform wheel sets
        repo.tar.gz            # git archive of the built commit
        hf_cache/              # pre-downloaded HF hub cache (models--<org>--<name>/),
                               # every model pinned to the_oracle.models.pins
        wheels/
            linux/             # every wheel the installer needs on Linux
            windows/           # every wheel the installer needs on Windows
        install.sh             # extract + run the offline install on Linux
        install.bat            # extract + run the offline install on Windows

Build on a machine WITH internet, copy the bundle to the target (USB, drive,
network share), and run install.sh / install.bat there. The installer is
invoked as ``install --offline-bundle <dir>``: it creates the venv from the
bundled wheels, seeds the local Hugging Face cache from the bundled models,
and the managed launcher then runs with HF_HUB_OFFLINE=1 so no model fetch
can ever touch the network.

Usage:
    python3 scripts/build_offline_bundle.py --output dist/oracle-offline-bundle
    python3 scripts/build_offline_bundle.py --output dist/bundle --platform windows --pytorch cpu
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from collections.abc import Iterable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(SRC_ROOT), str(REPO_ROOT / "scripts")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from the_oracle.models.pins import (  # noqa: E402
    CHATTERBOX_MULTILINGUAL_PATTERNS,
    CHATTERBOX_REPO,
    CHATTERBOX_STANDARD_PATTERNS,
    GO_EMOTIONS_REPO,
    HELPER_MODEL_PATTERNS,
    MODEL_PINS,
    PUNCTUATION_REPO,
    TURBO_ALLOW_PATTERNS,
    TURBO_REPO_ID,
)

import manage_install  # noqa: E402

#: Torch's unconditional runtime dependencies. The torch wheels are downloaded
#: with ``--no-deps``: pip evaluates torch's ``platform_system == "Linux"``
#: nvidia markers against the *building* host, so a dependency-resolving
#: cross-platform download of torch fails on a Linux builder even when the
#: target is Windows and needs none of those packages. These few deps are the
#: same on every platform.
TORCH_RUNTIME_DEPS = (
    "filelock",
    "typing-extensions",
    "networkx",
    "jinja2",
    "MarkupSafe",
    "fsspec",
    "sympy",
    "mpmath",
)

#: Packages that require torch (or torchaudio) themselves. Their wheels are
#: bundled with ``--no-deps``, exactly as manage_install installs them: pip
#: evaluates torch's ``platform_system == "Linux"`` nvidia markers against the
#: *building* host, so a resolving cross-platform download of any package that
#: requires torch fails on a Linux builder even when the target is Windows and
#: needs none of those packages. Newer PyPI torch releases also require
#: ``setuptools<82``, which conflicts with the pinned bootstrap setuptools.
TORCH_CONSUMER_NAMES = (
    "chatterbox-tts",
    "conformer",
    "deepmultilingualpunctuation",
    "s3tokenizer",
)

#: Distributions whose declared requirements are NOT the ones this project
#: installs. manage_install installs chatterbox-tts with --no-deps and then
#: rewrites its metadata (scripts/patch_chatterbox_metadata.py) to the audited
#: stack, so following its published pins would drag in transformers==5.2.0,
#: diffusers==0.29.0, unused Gradio, and the GPL-3.0 pykakasi that the patch
#: deliberately removes.
CLOSURE_IGNORED_DISTRIBUTIONS = frozenset({"chatterbox-tts"})


def _requirement_name(requirement: str) -> str:
    """Distribution name of a requirement string (``name[extra]>=1`` -> ``name``)."""
    return re.split(r"[<>=!~\[\s;]", requirement.strip(), maxsplit=1)[0].strip().lower()


def _target_marker_environment(platform: str, python_version: str) -> dict[str, str]:
    """Marker environment for a wheel built for the *target*, not this host."""
    if platform == "windows":
        environment = {
            "sys_platform": "win32",
            "platform_system": "Windows",
            "platform_machine": "AMD64",
            "os_name": "nt",
        }
    else:
        environment = {
            "sys_platform": "linux",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "os_name": "posix",
        }
    environment.update(
        {
            "python_version": python_version,
            "python_full_version": f"{python_version}.0",
            "implementation_name": "cpython",
            "platform_python_implementation": "CPython",
            "extra": "",
        }
    )
    return environment


def torch_dependency_pins(wheels_dir: Path, *, platform: str, python_version: str) -> list[str]:
    """The exact dependency pins the bundled torch wheel declares for the target.

    torch pins its runtime dependencies exactly (``sympy==1.13.1``), so the
    unconstrained ``TORCH_RUNTIME_DEPS`` set silently downloaded a newer sympy
    that torch then refused to install with, failing the offline install on the
    target machine. Reading the pins out of the wheel's own METADATA keeps the
    bundle in step with whatever torch version is bundled, and evaluating the
    markers against the *target* drops torch's Linux-only nvidia requirements.
    """
    try:
        from packaging.markers import Marker
        from packaging.requirements import Requirement
    except ImportError:  # pragma: no cover - packaging is a hard project dependency
        log("WARNING: packaging unavailable; falling back to the loose torch dep list")
        return list(TORCH_RUNTIME_DEPS)

    environment = _target_marker_environment(platform, python_version)
    pins: list[str] = []
    for wheel in sorted(wheels_dir.glob("torch-*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if not metadata_names:
                continue
            metadata = archive.read(metadata_names[0]).decode("utf-8", "replace")
        for line in metadata.splitlines():
            if not line.startswith("Requires-Dist:"):
                continue
            requirement = Requirement(line.split(":", 1)[1].strip())
            if requirement.marker is not None:
                if "extra" in str(requirement.marker):
                    continue
                if not requirement.marker.evaluate(environment):
                    continue
            pins.append(str(requirement).split(";")[0].strip())
    if not pins:
        log("WARNING: no torch metadata found; falling back to the loose torch dep list")
        return list(TORCH_RUNTIME_DEPS)
    return pins


def wheel_set_index(wheels_dir: Path) -> dict[str, set[str]]:
    """Map canonical distribution name -> versions present as ``*.whl`` files."""
    from packaging.utils import canonicalize_name

    index: dict[str, set[str]] = {}
    for wheel in wheels_dir.glob("*.whl"):
        parts = wheel.name.split("-")
        if len(parts) < 2:
            continue
        index.setdefault(canonicalize_name(parts[0]), set()).add(parts[1])
    return index


def missing_target_requirements(wheels_dir: Path, *, platform: str, python_version: str) -> list[str]:
    """Requirements the wheels declare for the target that the set does not contain.

    pip evaluates environment markers against the machine doing the download, so
    a Linux builder silently drops Windows-only requirements (``tqdm`` needs
    ``colorama; platform_system == "Windows"``). The target then resolves the
    install from the bundle and fails on the missing package. This walks every
    bundled wheel's metadata with the *target* marker environment and returns
    whatever is missing, so the caller can fetch it and repeat until closed.
    """
    from packaging.requirements import Requirement
    from packaging.specifiers import InvalidSpecifier
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version

    environment = _target_marker_environment(platform, python_version)
    present = wheel_set_index(wheels_dir)
    missing: dict[str, str] = {}
    for wheel in sorted(wheels_dir.glob("*.whl")):
        if canonicalize_name(wheel.name.split("-")[0]) in CLOSURE_IGNORED_DISTRIBUTIONS:
            continue
        try:
            with zipfile.ZipFile(wheel) as archive:
                metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                if not metadata_names:
                    continue
                metadata = archive.read(metadata_names[0]).decode("utf-8", "replace")
        except zipfile.BadZipFile:
            continue
        for line in metadata.splitlines():
            if not line.startswith("Requires-Dist:"):
                continue
            try:
                requirement = Requirement(line.split(":", 1)[1].strip())
            except Exception:  # noqa: BLE001 - a malformed third-party line must not abort the build
                continue
            if "extra" in str(requirement.marker or ""):
                continue
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            name = canonicalize_name(requirement.name)
            versions = present.get(name)
            if versions:
                try:
                    if not requirement.specifier or any(
                        Version(version) in requirement.specifier for version in versions
                    ):
                        continue
                except (InvalidSpecifier, InvalidVersion):
                    continue
            missing[name] = str(requirement).split(";")[0].strip()
    return sorted(missing.values())


def torch_constraints(wheels_dir: Path) -> Path:
    """Pin resolution to the torch wheels this bundle already downloaded.

    Every package that requires ``torch`` (deepmultilingualpunctuation,
    s3tokenizer, conformer, ...) otherwise lets pip choose PyPI's default
    wheel, whose ``platform_system == "Linux"`` nvidia pins are evaluated
    against the *building* host and can never be satisfied for a cross-platform
    Windows target. Constraining to the exact version of the wheel on disk —
    e.g. ``torch==2.6.0+cpu``, a local version PyPI does not publish — leaves
    the bundled cpu wheel as the only candidate, and it declares no nvidia
    dependencies at all.
    """
    pins: list[str] = []
    for wheel in sorted(wheels_dir.glob("*.whl")):
        parts = wheel.name.split("-")
        if len(parts) >= 2 and parts[0] in {"torch", "torchaudio", "torchvision"}:
            pins.append(f"{parts[0]}=={parts[1]}")
    if not pins:
        pins = list(manage_install.TORCH_PACKAGES)
    path = wheels_dir.parent / "torch-constraints.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pins) + "\n", encoding="utf-8")
    return path


def log(message: str) -> None:
    print(f"[bundle] {message}", flush=True)


def run(args: list[str], **kwargs) -> None:
    log("+ " + " ".join(args))
    subprocess.run(args, check=True, **kwargs)


def project_requirements() -> list[str]:
    """Install-time requirements: base deps + ml extra (what ``.[ml]`` pulls)."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    reqs: list[str] = list(project.get("dependencies", []))
    reqs.extend(project.get("optional-dependencies", {}).get("ml", []))
    return reqs


def _link_or_copy(source: str, destination: str) -> str:
    """Hardlink when the filesystem allows it, copy otherwise."""
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _mirror_repo(source_repo: Path, destination_repo: Path) -> None:
    """Copy one hub-cache repo, copying each blob once and linking its snapshots.

    ``copytree`` would materialize every snapshot entry as well as its blob,
    roughly doubling a model cache that is kept as links (a snapshot entry is a
    link to ``blobs/<etag>`` in every hub cache). Blobs are copied once here and
    snapshot entries are hardlinked to the copied blob, falling back to a copy
    only where the filesystem cannot link.
    """
    source_blobs = source_repo / "blobs"
    destination_blobs = destination_repo / "blobs"
    destination_blobs.mkdir(parents=True, exist_ok=True)
    by_inode: dict[tuple[int, int], str] = {}
    if source_blobs.is_dir():
        for blob in source_blobs.iterdir():
            if not blob.is_file():
                continue
            stat = blob.stat()
            by_inode[(stat.st_dev, stat.st_ino)] = blob.name
            _link_or_copy(str(blob), str(destination_blobs / blob.name))

    for path in source_repo.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source_repo)
        if relative.parts[0] == "blobs":
            continue
        target = destination_repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        stat = path.stat()  # follows the snapshot's link to its blob
        blob_name = by_inode.get((stat.st_dev, stat.st_ino))
        copied = destination_blobs / blob_name if blob_name else None
        if copied is not None and copied.is_file():
            _link_or_copy(str(copied), str(target))
        else:
            shutil.copy2(path, target)


def dereference_cache_symlinks(hf_cache: Path) -> int:
    """Make every cached file a real file, because the hub cache uses symlinks.

    ``snapshot_download`` links cached files into ``snapshots/<rev>/`` with
    relative symlinks. The offline installer seeds the local cache with
    ``copytree(symlinks=True)``, and creating a symlink on Windows needs
    Developer Mode or an elevated token, so a symlinked snapshot entry would
    break an ordinary user's install. Each link is replaced by its content;
    ``_link_or_copy`` keeps it a hardlink where the filesystem allows, so the
    bundle still occupies a single copy of each model shard.
    """
    replaced = 0
    for path in sorted(hf_cache.rglob("*")):
        if not path.is_symlink():
            continue
        target = path.resolve()
        if not target.is_file():
            raise SystemExit(f"broken model-cache symlink in bundle: {path}")
        path.unlink()
        _link_or_copy(str(target), str(path))
        replaced += 1
    return replaced


def seed_cache(source: Path, hf_cache: Path, repo_ids: Iterable[str] | None = None) -> int:
    """Hardlink an existing hub cache into the bundle before downloading.

    A builder that already has the pinned models on disk (a developer box, a
    previous bundle, the CI model cache) should not pull another ~13 GB. Only
    the repos in *repo_ids* (default: every pinned repo) are linked into
    *hf_cache*; ``snapshot_download`` then finds those files and only fetches
    the ones that are genuinely missing.

    Restricting the seed to the pins matters: a machine's hub cache also holds
    unrelated models (chat models, other TTS checkpoints), and seeding those
    would copy gigabytes of weights the bundle never uses.
    """
    if not source.is_dir():
        return 0
    wanted = {"models--" + repo_id.replace("/", "--") for repo_id in (MODEL_PINS if repo_ids is None else repo_ids)}
    hf_cache.mkdir(parents=True, exist_ok=True)
    seeded = 0
    for dirname in sorted(wanted):
        repo_dir = source / dirname
        if not repo_dir.is_dir():
            continue
        destination = hf_cache / dirname
        if destination.exists():
            continue
        _mirror_repo(repo_dir, destination)
        seeded += 1
    if seeded:
        log(f"seeded {seeded} repo(s) from {source} (no download needed for those files)")
    return seeded


def download_models(hf_cache: Path, *, seed_from: Path | None = None) -> dict[str, str]:
    """Pre-download every pinned model into hub-cache layout. Returns sizes."""
    from huggingface_hub import snapshot_download

    jobs: list[tuple[str, list[str]]] = [
        (GO_EMOTIONS_REPO, HELPER_MODEL_PATTERNS[GO_EMOTIONS_REPO]),
        (PUNCTUATION_REPO, HELPER_MODEL_PATTERNS[PUNCTUATION_REPO]),
        (CHATTERBOX_REPO, sorted(set(CHATTERBOX_STANDARD_PATTERNS) | set(CHATTERBOX_MULTILINGUAL_PATTERNS))),
        (TURBO_REPO_ID, TURBO_ALLOW_PATTERNS),
    ]
    if seed_from is not None:
        seed_cache(seed_from, hf_cache, [repo_id for repo_id, _ in jobs])

    sizes: dict[str, str] = {}
    for repo_id, patterns in jobs:
        revision = MODEL_PINS[repo_id]
        log(f"downloading {repo_id} @ {revision[:12]} ({len(patterns)} file patterns)")
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=patterns,
            cache_dir=str(hf_cache),
        )
        total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        sizes[repo_id] = f"{total / 1e9:.2f} GB"
        log(f"cached {repo_id}: {sizes[repo_id]}")

    replaced = dereference_cache_symlinks(hf_cache)
    if replaced:
        log(f"replaced {replaced} symlinked cache entry(ies) with real files (Windows installers cannot create symlinks)")
    return sizes


def download_wheels(wheels_dir: Path, *, platform: str, python_version: str, pytorch: str) -> None:
    """Download every wheel the installer needs into wheels_dir."""
    wheels_dir.mkdir(parents=True, exist_ok=True)
    plat_args: list[str] = []
    if platform == "windows":
        plat_args = [
            "--platform", "win_amd64",
            "--python-version", python_version,
            "--implementation", "cp",
            "--abi", f"cp{python_version.replace('.', '')}",
            "--only-binary", ":all:",
        ]
        running = f"{sys.version_info[0]}.{sys.version_info[1]}"
        if python_version != running:
            log(f"WARNING: cross-downloading Windows wheels for cp{python_version} "
                f"while running Python {running}; pure-Python wheels are unaffected, "
                f"compiled ones follow --python-version.")

    torch_index = manage_install.PYTORCH_CUDA_INDEX_URL if pytorch == "cuda" else manage_install.PYTORCH_CPU_INDEX_URL
    log(f"downloading torch wheels ({pytorch}) for {platform} from {torch_index}")
    run([sys.executable, "-m", "pip", "download", "--dest", str(wheels_dir),
         "--index-url", torch_index, *plat_args, "--no-deps", *manage_install.TORCH_PACKAGES])
    torch_deps = torch_dependency_pins(wheels_dir, platform=platform, python_version=python_version)
    log(f"downloading torch runtime dependencies for {platform}: {', '.join(torch_deps)}")
    run([sys.executable, "-m", "pip", "download", "--dest", str(wheels_dir),
         *plat_args, *torch_deps])

    # Build the pinned Perth source revision into a wheel now so offline
    # installs never need a VCS checkout or build backend network access.
    run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels_dir),
         manage_install.PERTH_GIT_PACKAGE])

    all_reqs = [
        *project_requirements(),
        *manage_install.CHATTERBOX_ENGINE_PACKAGES,
        manage_install.CHATTERBOX_TTS_PACKAGE,
        *manage_install.BOOTSTRAP_PACKAGES,
    ]
    resolving = [req for req in all_reqs if _requirement_name(req) not in TORCH_CONSUMER_NAMES]
    torch_consumers = [req for req in all_reqs if _requirement_name(req) in TORCH_CONSUMER_NAMES]

    log(f"downloading {len(resolving)} PyPI requirements (+transitives) for {platform}")
    # --find-links + --constraint keep the resolution pointed at the bundled cpu
    # torch family should a torch requirement ever slip through the exclusion
    # above (constraints alone cannot save it: pip reads torch's nvidia pins
    # from whichever wheel it considers, and evaluates them on this host).
    constraints = torch_constraints(wheels_dir)
    run([sys.executable, "-m", "pip", "download", "--dest", str(wheels_dir),
         "--find-links", str(wheels_dir),
         "--constraint", str(constraints),
         *plat_args, *resolving])

    # Bundled without dependency resolution, matching the order manage_install
    # installs them in: torch first, then these, so their torch requirement is
    # already satisfied. chatterbox-tts is additionally metadata-patched by the
    # installer, so its published pins never apply.
    log(f"downloading {len(torch_consumers)} torch-consuming packages (no deps) for {platform}")
    run([sys.executable, "-m", "pip", "download", "--dest", str(wheels_dir),
         *plat_args, "--no-deps", *torch_consumers])

    # Close the target's dependency closure. pip drops requirements whose markers
    # only hold on the target platform (colorama for tqdm on Windows, for one),
    # and the offline install then fails on the missing wheel. Each pass fetches
    # what the bundled metadata still asks for; newly fetched wheels can bring
    # their own target-only requirements, so repeat until nothing is missing.
    for _ in range(6):
        missing = missing_target_requirements(wheels_dir, platform=platform, python_version=python_version)
        if not missing:
            break
        log(f"downloading {len(missing)} requirement(s) only {platform} needs: {', '.join(missing)}")
        run([sys.executable, "-m", "pip", "download", "--dest", str(wheels_dir),
             *plat_args, "--no-deps", *missing])
    else:  # pragma: no cover - a cycle in third-party metadata
        print(f"WARNING: {platform} dependency closure did not settle after 6 passes", file=sys.stderr)

    count = len(list(wheels_dir.glob("*.whl")))
    log(f"{platform}: {count} wheels")


def write_launchers(bundle_dir: Path) -> None:
    (bundle_dir / "install.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# Offline install of The Oracle from this bundle. No network needed.\n"
        "set -Eeuo pipefail\n"
        'BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'INSTALL_ROOT="${1:-$HOME/The_Oracle_TTS}"\n'
        'mkdir -p "$INSTALL_ROOT"\n'
        'tar -xzf "$BUNDLE_DIR/repo.tar.gz" -C "$INSTALL_ROOT"\n'
        'exec "$INSTALL_ROOT/install_oracle_tts.sh" --offline-bundle "$BUNDLE_DIR"\n',
        encoding="utf-8",
    )
    (bundle_dir / "install.bat").write_text(
        "@echo off\r\n"
        "REM Offline install of The Oracle from this bundle. No network needed.\r\n"
        'set "BUNDLE_DIR=%~dp0"\r\n'
        # %~dp0 always ends with a backslash; strip it so the quoted
        # --offline-bundle argument does not end in a backslash-quote, which
        # PowerShell reads as an escaped quote and mangles the path with.
        'if "%BUNDLE_DIR:~-1%"=="\\" set "BUNDLE_DIR=%BUNDLE_DIR:~0,-1%"\r\n'
        'if "%~1"=="" ( set "INSTALL_ROOT=%USERPROFILE%\\The_Oracle_TTS" ) else ( set "INSTALL_ROOT=%~1" )\r\n'
        'mkdir "%INSTALL_ROOT%" 2>nul\r\n'
        'tar -xzf "%BUNDLE_DIR%\\repo.tar.gz" -C "%INSTALL_ROOT%"\r\n'
        'powershell -ExecutionPolicy Bypass -File "%INSTALL_ROOT%\\install_oracle_tts.ps1" --offline-bundle "%BUNDLE_DIR%"\r\n',
        encoding="utf-8",
    )
    (bundle_dir / "install.sh").chmod(0o755)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a fully offline install bundle for The Oracle.")
    parser.add_argument("--output", type=Path, required=True, help="Bundle directory to create.")
    parser.add_argument("--platform", choices=["linux", "windows", "both"], default="both",
                        help="Which platform wheel sets to download (default: both).")
    parser.add_argument("--pytorch", choices=["cpu", "cuda"], default="cpu",
                        help="Which PyTorch wheel family to bundle (default: cpu).")
    parser.add_argument("--python-version", default=f"{sys.version_info[0]}.{sys.version_info[1]}",
                        help="Target Python version for cross-downloaded wheels (default: running interpreter).")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads (wheels only).")
    parser.add_argument("--model-cache", type=Path, default=None,
                        help="Existing HF hub cache to seed the bundle from; only files missing "
                             "from it are downloaded (default: download everything).")
    parser.add_argument("--skip-wheels", action="store_true", help="Skip wheel downloads (models only).")
    args = parser.parse_args(argv)

    bundle_dir: Path = args.output
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        print(f"FAIL: output directory is not empty: {bundle_dir}", file=sys.stderr)
        return 1
    bundle_dir.mkdir(parents=True, exist_ok=True)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()

    log(f"archiving repo @ {commit[:12]}")
    with open(bundle_dir / "repo.tar.gz", "wb") as fh:
        subprocess.run(["git", "archive", "--format=tar.gz", "HEAD"], cwd=REPO_ROOT,
                       stdout=fh, check=True)

    model_sizes: dict[str, str] = {}
    if not args.skip_models:
        model_sizes = download_models(bundle_dir / "hf_cache", seed_from=args.model_cache)

    platforms = ["linux", "windows"] if args.platform == "both" else [args.platform]
    if not args.skip_wheels:
        for platform in platforms:
            try:
                download_wheels(bundle_dir / "wheels" / platform, platform=platform,
                                python_version=args.python_version, pytorch=args.pytorch)
            except subprocess.CalledProcessError:
                print(f"FAIL: wheel download failed for platform={platform}. "
                      f"The bundle cannot be fully offline for {platform}.", file=sys.stderr)
                return 1

    write_launchers(bundle_dir)

    manifest = {
        "app": "the-oracle",
        "repo_commit": commit,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_pins": dict(MODEL_PINS),
        "model_sizes": model_sizes,
        "platforms": platforms,
        "pytorch": args.pytorch,
        "python_version": args.python_version,
        "wheel_counts": {
            platform: len(list((bundle_dir / "wheels" / platform).glob("*.whl")))
            for platform in platforms if (bundle_dir / "wheels" / platform).is_dir()
        },
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"bundle complete: {bundle_dir}")
    log(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
