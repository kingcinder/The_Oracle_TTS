"""Reference voice catalog helpers for repo-local defaults and recent custom clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VoiceChoice:
    label: str
    path: str


def default_voice_choices(repo_root: str | Path, limit: int = 10) -> list[VoiceChoice]:
    root = Path(repo_root)
    candidates = [
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
        for path in sorted(directory.glob("*.wav")):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            choices.append(VoiceChoice(label=_label_for_path(path), path=resolved))
            if len(choices) >= limit:
                return choices
    return choices


def _label_for_path(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip() or "Reference"
    return stem.title()
