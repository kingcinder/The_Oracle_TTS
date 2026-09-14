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
import shutil
import subprocess
import sys
import tomllib
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


def download_models(hf_cache: Path) -> dict[str, str]:
    """Pre-download every pinned model into hub-cache layout. Returns sizes."""
    from huggingface_hub import snapshot_download

    jobs: list[tuple[str, list[str]]] = [
        (GO_EMOTIONS_REPO, HELPER_MODEL_PATTERNS[GO_EMOTIONS_REPO]),
        (PUNCTUATION_REPO, HELPER_MODEL_PATTERNS[PUNCTUATION_REPO]),
        (CHATTERBOX_REPO, sorted(set(CHATTERBOX_STANDARD_PATTERNS) | set(CHATTERBOX_MULTILINGUAL_PATTERNS))),
        (TURBO_REPO_ID, TURBO_ALLOW_PATTERNS),
    ]
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
         "--index-url", torch_index, *plat_args, *manage_install.TORCH_PACKAGES])

    pypi_reqs = [
        *project_requirements(),
        *manage_install.CHATTERBOX_ENGINE_PACKAGES,
        manage_install.CHATTERBOX_TTS_PACKAGE,
        *manage_install.BOOTSTRAP_PACKAGES,
    ]
    log(f"downloading {len(pypi_reqs)} PyPI requirements (+transitives) for {platform}")
    run([sys.executable, "-m", "pip", "download", "--dest", str(wheels_dir),
         *plat_args, *pypi_reqs])

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
        'if "%~1"=="" ( set "INSTALL_ROOT=%USERPROFILE%\\The_Oracle_TTS" ) else ( set "INSTALL_ROOT=%~1" )\r\n'
        'mkdir "%INSTALL_ROOT%" 2>nul\r\n'
        'tar -xzf "%BUNDLE_DIR%repo.tar.gz" -C "%INSTALL_ROOT%"\r\n'
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
        model_sizes = download_models(bundle_dir / "hf_cache")

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
