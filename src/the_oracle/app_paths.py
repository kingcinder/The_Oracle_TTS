"""Repo-local directory and filename helpers for The Oracle GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OraclePaths:
    repo_root: Path
    input_dir: Path
    output_dir: Path
    profile_dir: Path
    voice_dir: Path


def ensure_repo_default_paths(repo_root: str | Path) -> OraclePaths:
    root = Path(repo_root).expanduser().resolve()
    paths = OraclePaths(
        repo_root=root,
        input_dir=root / "Input",
        output_dir=root / "Output",
        profile_dir=root / "Profiles",
        voice_dir=root / "Seashells",
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
