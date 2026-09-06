"""Reference voice catalog helpers for repo-local defaults and recent custom clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VoiceChoice:
    label: str
    path: str


VOICE_FILE_PATTERNS = ("*.wav", "*.flac", "*.mp3")


def _sorted_voice_paths(directory: Path, pattern: str) -> list[Path]:
    """Voice files for a directory, English-conditioned clips first.

    The bundled library mixes languages (see Seashells/generic/ATTRIBUTION.md);
    the default render path is English, so English-conditioned references must
    lead the picker (and be the CLI's fallback default) rather than sort
    behind other languages alphabetically.
    """
    paths = list(directory.glob(pattern))
    if directory.name == "generic":
        return sorted(
            paths,
            key=lambda p: (0 if p.stem.startswith(("english", "generic")) else 1, str(p)),
        )
    return sorted(paths)


def default_voice_choices(repo_root: str | Path, limit: int = 10) -> list[VoiceChoice]:
    root = Path(repo_root)
    candidates = [
        # Bundled generic voice library first (repo-shipped, third-party
        # demo clips from audio.cpp, see Seashells/generic/ATTRIBUTION.md),
        # then the user's curated clips, then build-time fallbacks.
        root / "Seashells" / "generic",
        root / "Seashells",
        root / ".engine-setup" / "chatterbox_runtime_check" / "refs",
        root / ".engine-setup" / "chatterbox_runtime_check_final" / "refs",
        root / "build" / "real_engine_smoke" / "inputs",
        root / "build" / "smoke_render",
        root / "build" / "project_manifest_verify",
        root / "build" / "project_manifest_verify_20260314",
    ]
    choices: list[VoiceChoice] = []
    seen: set[str] = set()
    for directory in candidates:
        if not directory.exists():
            continue
        for pattern in VOICE_FILE_PATTERNS:
            for path in _sorted_voice_paths(directory, pattern):
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                choices.append(VoiceChoice(label=_label_for_path(path), path=resolved))
                if len(choices) >= limit:
                    return choices
    return choices


def voice_catalog_audit(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    generic_dir = root / "Seashells" / "generic"
    seashell_dir = root / "Seashells"
    fallback_dirs = [
        root / ".engine-setup" / "chatterbox_runtime_check" / "refs",
        root / ".engine-setup" / "chatterbox_runtime_check_final" / "refs",
        root / "build" / "real_engine_smoke" / "inputs",
        root / "build" / "smoke_render",
        root / "build" / "project_manifest_verify",
        root / "build" / "project_manifest_verify_20260314",
    ]
    generic_clips = _count_voice_files(generic_dir)
    seashell_clips = _count_voice_files(seashell_dir)
    fallback_clips = sum(_count_voice_files(directory) for directory in fallback_dirs)
    primary_source = "generic" if generic_clips else "seashells" if seashell_clips else "build_fallbacks" if fallback_clips else "none"
    return {
        "ok": bool(generic_clips or seashell_clips or fallback_clips),
        "primary_source": primary_source,
        "seashell_dir": str(seashell_dir),
        "generic_clip_count": generic_clips,
        "seashell_clip_count": seashell_clips,
        "fallback_clip_count": fallback_clips,
        "default_voice_assessment": (
            "Bundled generic voices in Seashells/generic."
            if generic_clips
            else "Repo-local reference clips in Seashells."
            if seashell_clips
            else "Build-time smoke/reference clips, not curated production voices."
            if fallback_clips
            else "No local reference clips found."
        ),
        "better_local_assets_available": True,
        "better_local_assets_detail": (
            "Higher-quality local reference clips can be preloaded by dropping them into Seashells."
            " Chatterbox itself does not expose a separate packaged voice library here."
        ),
        "voice_mixing_low_risk": False,
        "voice_mixing_detail": (
            "Current Chatterbox rendering conditions each speaker from one reference/conditioning payload."
            " Blending multiple voices would require changing the voice-profile and render pipeline."
        ),
    }


def _label_for_path(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip() or "Reference"
    return stem.title()


def _count_voice_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for pattern in VOICE_FILE_PATTERNS for _ in directory.glob(pattern))
