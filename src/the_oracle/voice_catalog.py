"""Reference voice catalog helpers for repo-local defaults and recent custom clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VoiceChoice:
    label: str
    path: str
    # "file" = a plain reference clip; "blend" = a saved named voice blend
    # (see save_blend_voice). The picker renders blends under their own
    # "Saved Blends" section.
    kind: str = "file"


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
    # Named blend voices (user-saved) follow the bundled/curated files so a
    # plain file is always the first (English-conditioned) default, then
    # truncate to the limit.
    seen.update(choice.path for choice in choices)
    choices.extend(blend for blend in blend_voice_choices(root / "Profiles") if blend.path not in seen)
    return choices[:limit]


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


def blend_catalog_path(profiles_dir: str | Path) -> Path:
    """The JSON catalog of saved named blend voices under a Profiles dir."""
    return Path(profiles_dir).expanduser() / "blend_voices.json"


def blend_clips_dir(profiles_dir: str | Path) -> Path:
    """Directory holding the derived (deterministic) blend reference wavs."""
    return Path(profiles_dir).expanduser() / ".blends"


def _load_blend_catalog(catalog_path: Path) -> dict[str, Any]:
    if not catalog_path.exists():
        return {"version": 1, "voices": []}
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt catalog must never crash the voice picker; treat as empty.
        return {"version": 1, "voices": []}
    if not isinstance(payload, dict) or not isinstance(payload.get("voices"), list):
        return {"version": 1, "voices": []}
    return payload


def save_blend_voice(
    profiles_dir: str | Path,
    name: str,
    path_a: str | Path,
    path_b: str | Path,
    weight: float = 0.5,
    mode: str = "mix",
) -> VoiceChoice:
    """Save (or upsert by name) a named blend voice and return its picker entry.

    The derived reference clip is materialized into ``.blends/`` by the same
    deterministic ``blend_references`` the render pipeline uses, so picking
    the saved voice is byte-identical to configuring the blend inline.
    Saving the same name again replaces the entry.
    """
    from the_oracle.audio.blend import blend_references

    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Blend voice name must not be empty.")
    source_a = Path(path_a).expanduser()
    source_b = Path(path_b).expanduser()
    for source in (source_a, source_b):
        if not source.exists():
            raise FileNotFoundError(f"Blend source does not exist: {source}")
    derived = blend_references(
        source_a,
        source_b,
        weight_a=float(weight),
        mode=mode,
        out_dir=blend_clips_dir(profiles_dir),
    )
    catalog_path = blend_catalog_path(profiles_dir)
    payload = _load_blend_catalog(catalog_path)
    entry = {
        "name": clean_name,
        "path_a": str(source_a),
        "path_b": str(source_b),
        "weight": float(max(0.0, min(1.0, float(weight)))),
        "mode": mode,
    }
    payload["voices"] = [voice for voice in payload["voices"] if voice.get("name") != clean_name] + [entry]
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return VoiceChoice(label=clean_name, path=str(derived), kind="blend")


def blend_voice_choices(profiles_dir: str | Path) -> list[VoiceChoice]:
    """The saved named blend voices, as picker entries (derived wavs resolved).

    Entries whose source clips no longer exist are skipped, never fatal: a
    saved voice stays selectable only while its inputs are on disk. Malformed
    entries (missing paths, non-numeric weights, bad modes) are skipped the
    same way: a corrupt catalog must never crash the voice picker.
    """
    from the_oracle.audio.blend import blend_references

    catalog_path = blend_catalog_path(profiles_dir)
    payload = _load_blend_catalog(catalog_path)
    choices: list[VoiceChoice] = []
    seen: set[str] = set()
    for voice in payload.get("voices", []):
        if not isinstance(voice, dict):
            continue
        name = voice.get("name")
        path_a = voice.get("path_a")
        path_b = voice.get("path_b")
        if not name or not isinstance(name, str) or not path_a or not path_b:
            continue
        try:
            weight = float(voice.get("weight", 0.5))
        except (TypeError, ValueError):
            weight = 0.5
        mode = voice.get("mode", "mix")
        if not isinstance(mode, str):
            mode = "mix"
        try:
            derived = blend_references(
                path_a,
                path_b,
                weight_a=weight,
                mode=mode,
                out_dir=blend_clips_dir(profiles_dir),
            )
        except Exception:
            # Missing clips, unreadable audio, unknown blend mode, missing
            # optional audio deps: skip the entry, never crash the picker.
            continue
        resolved = str(derived)
        if resolved in seen:
            continue
        seen.add(resolved)
        choices.append(VoiceChoice(label=name, path=resolved, kind="blend"))
    return choices


def remove_blend_voice(profiles_dir: str | Path, name: str) -> bool:
    """Remove a saved blend voice by name. Returns True when it existed."""
    catalog_path = blend_catalog_path(profiles_dir)
    payload = _load_blend_catalog(catalog_path)
    before = len(payload["voices"])
    payload["voices"] = [voice for voice in payload["voices"] if voice.get("name") != name]
    if len(payload["voices"]) == before:
        return False
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def _count_voice_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for pattern in VOICE_FILE_PATTERNS for _ in directory.glob(pattern))
