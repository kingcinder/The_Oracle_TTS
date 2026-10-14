"""Repo-local directory and filename helpers for The Oracle GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil

from the_oracle.platform_support import app_data_dir


PORTABLE_MARKER = ".oracle-portable"


@dataclass(frozen=True, slots=True)
class OraclePaths:
    repo_root: Path
    input_dir: Path
    output_dir: Path
    profile_dir: Path
    voice_dir: Path


def _portable_mode(root: Path) -> bool:
    value = os.environ.get("ORACLE_PORTABLE", "").strip().lower()
    return value in {"1", "true", "yes", "on"} or (root / PORTABLE_MARKER).is_file()


def _migrate_tree(source: Path, destination: Path) -> None:
    """Copy a legacy repo-local tree without deleting or overwriting files."""
    if not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def ensure_repo_default_paths(repo_root: str | Path) -> OraclePaths:
    root = Path(repo_root).expanduser().resolve()
    installed_root = Path(__file__).resolve().parents[2]
    data_root = root
    if root == installed_root and not _portable_mode(root):
        data_root = app_data_dir()
        for name in ("Input", "Output", "Profiles", "Seashells"):
            _migrate_tree(root / name, data_root / name)
    paths = OraclePaths(
        repo_root=root,
        input_dir=data_root / "Input",
        output_dir=data_root / "Output",
        profile_dir=data_root / "Profiles",
        voice_dir=data_root / "Seashells",
    )
    for path in (paths.input_dir, paths.output_dir, paths.profile_dir, paths.voice_dir):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def normalize_output_filename(value: str) -> str:
    cleaned = Path(value.strip()).name
    if not cleaned:
        return ""
    suffix = Path(cleaned).suffix.lower()
    if suffix == ".flac":
        return cleaned
    stem = Path(cleaned).stem
    return f"{stem}.flac" if stem else ""


def default_output_filename(input_path: str | Path) -> str:
    candidate = Path(str(input_path).strip())
    stem = candidate.stem.strip()
    return f"{stem}.flac" if stem else ""


def resolve_output_filename(
    input_path: str | Path,
    output_dir: str | Path,
    default_output_dir: str | Path,
    requested_name: str,
) -> str:
    normalized = normalize_output_filename(requested_name)
    if normalized:
        return normalized
    if Path(output_dir).expanduser().resolve() == Path(default_output_dir).expanduser().resolve():
        return default_output_filename(input_path)
    return ""
