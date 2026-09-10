"""Qt-free helpers for the Oracle desktop GUI.

Everything in this module is importable without PySide6 so it stays
unit-testable in headless CI. The Qt widgets live in ``app_gui.py`` and
delegate the pure logic here:

- :class:`CastModel` — the ordered speaker cast (keys, optional character
  names, per-speaker settings payloads) backing the cast-management dialog.
- :func:`sanitize_recording_filename` / :func:`recording_target_path` — keep
  user-typed recording names confined to the chosen output folder.
- :func:`normalize_cast_keys` / :func:`next_speaker_key` — speaker-key
  bookkeeping clamped to the engine's voice capacity.
- :func:`kill_process_tree` — SIGKILL a subprocess's whole process group so a
  timed-out download can never block a worker thread on inherited pipes.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The engine's speaker-attribution stage addresses at most 24 distinct voices
# (speaker_attribution/heuristics.py: MAX_SPEAKERS, keys A..X). The GUI mirrors
# that capacity so the cast can never promise a voice the engine can't render.
# Kept as a literal rather than imported so this module stays dependency-light
# (importing the engine constant would drag in NumPy); the review-fix tests
# assert it still matches the engine value whenever the engine is importable.
MAX_CAST_SPEAKERS = 24

_SPEAKER_KEY_RE = re.compile(r"[A-X]\Z")
_UNSAFE_FILENAME_CHARS_RE = re.compile(r'[<>:"|?*\x00-\x1f]')


def speaker_keys() -> list[str]:
    """All voice keys the engine can address, in order (A..X)."""
    return [chr(ord("A") + index) for index in range(MAX_CAST_SPEAKERS)]


def normalize_cast_keys(keys: Any) -> list[str]:
    """Coerce a persisted/loaded cast to an ordered list of valid voice keys.

    Unknown keys are dropped, duplicates collapse (first occurrence wins),
    the list is capped at :data:`MAX_CAST_SPEAKERS`, and an empty result
    falls back to the single narrator ``["A"]`` so callers always get a
    usable cast.
    """
    seen: list[str] = []
    if isinstance(keys, (list, tuple)):
        for key in keys:
            text = str(key).strip().upper()
            if _SPEAKER_KEY_RE.fullmatch(text) and text not in seen:
                seen.append(text)
                if len(seen) >= MAX_CAST_SPEAKERS:
                    break
    return seen or ["A"]


def next_speaker_key(used_keys: Any) -> str | None:
    """First free voice key for ``used_keys``, or None when the cast is full."""
    used = {str(key).strip().upper() for key in (used_keys or [])}
    for key in speaker_keys():
        if key not in used:
            return key
    return None


def sanitize_recording_filename(name: str | None, *, default_stem: str = "Seashell_No_1") -> str:
    """Make a user-typed recording name safe and confined to its folder.

    Directory components (``/``, ``\\``, ``..``) are stripped so the name can
    never escape the chosen output folder, filesystem-hostile characters are
    removed, and a ``.wav`` extension is enforced. An empty/degenerate result
    falls back to ``default_stem``.
    """
    text = (name or "").strip().replace("\\", "/")
    # Drop any directory components: only the final path segment survives.
    base = text.rsplit("/", 1)[-1].strip()
    base = _UNSAFE_FILENAME_CHARS_RE.sub("", base).strip()
    base = base.strip(".")
    if not base or base in {".", ".."}:
        base = default_stem
    if not base.lower().endswith(".wav"):
        base = f"{base}.wav"
    return base


def recording_target_path(
    folder: str | Path,
    filename: str | None,
    *,
    default_stem: str = "Seashell_No_1",
) -> Path:
    """Join a sanitized recording name onto ``folder``.

    Because the name carries no separators after sanitizing, the result can
    never escape ``folder``.
    """
    return Path(folder) / sanitize_recording_filename(filename, default_stem=default_stem)


def kill_process_tree(process: "subprocess.Popen[Any] | None") -> None:
    """SIGKILL a subprocess and its whole process group (POSIX).

    Used when a worker has already timed out waiting: SIGTERM is not enough
    because a timed-out download's real work happens in grandchildren that
    inherited the pipes. On Windows falls back to ``Popen.kill``. Never
    raises.
    """
    if process is None:
        return
    try:
        if process.poll() is not None:
            return
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - exercised on Windows installations
            process.kill()
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


@dataclass
class CastMember:
    """One speaker row in the cast-management dialog."""

    key: str
    name: str = ""
    settings: dict[str, Any] = field(default_factory=dict)

    def display_label(self) -> str:
        return f"{self.key} ({self.name})" if self.name else self.key


@dataclass
class CastModel:
    """Ordered, Qt-free model of the speaker cast.

    The main window and the cast-management dialog exchange casts through
    this model so the dialog never needs live widget access. ``settings``
    holds the per-speaker payload dicts (reference_path, voice_settings,
    emotion_reference_paths, blend_*, name) as produced/consumed by the GUI
    settings layer.
    """

    members: list[CastMember] = field(default_factory=list)

    @classmethod
    def from_parts(
        cls,
        keys: Any,
        names: dict[str, str] | None = None,
        settings_map: dict[str, dict[str, Any]] | None = None,
    ) -> "CastModel":
        names = names or {}
        settings_map = settings_map or {}
        return cls(
            members=[
                CastMember(
                    key=key,
                    name=str(names.get(key, "") or ""),
                    settings=dict(settings_map.get(key, {}) or {}),
                )
                for key in normalize_cast_keys(keys)
            ]
        )

    def keys(self) -> list[str]:
        return [member.key for member in self.members]

    def names(self) -> dict[str, str]:
        return {member.key: member.name for member in self.members}

    def settings_map(self) -> dict[str, dict[str, Any]]:
        return {member.key: dict(member.settings) for member in self.members}

    def add(self) -> CastMember | None:
        """Append a new speaker row; None when the engine's cast is full."""
        key = next_speaker_key(self.keys())
        if key is None:
            return None
        member = CastMember(key=key)
        self.members.append(member)
        return member

    def remove(self, key: str) -> bool:
        """Remove a speaker row. The narrator (A) can never be removed: the
        graceful path back to monologue is removing every *other* speaker."""
        if str(key).strip().upper() == "A":
            return False
        before = len(self.members)
        self.members = [member for member in self.members if member.key != key]
        return len(self.members) < before

    def rename(self, key: str, name: str) -> bool:
        for member in self.members:
            if member.key == key:
                member.name = str(name or "").strip()
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Persist the cast order and names (settings travel separately)."""
        return {
            "cast": self.keys(),
            "names": {member.key: member.name for member in self.members if member.name},
        }

    @classmethod
    def from_dict(cls, data: Any, settings_map: dict[str, dict[str, Any]] | None = None) -> "CastModel":
        if not isinstance(data, dict):
            return cls.from_parts([])
        cast = data.get("cast")
        names = data.get("names")
        # Backwards compatibility: very old payloads only have the speakers
        # dict, so fall back to its keys in sorted order.
        if cast is None and settings_map:
            cast = sorted(settings_map)
        return cls.from_parts(
            cast,
            names if isinstance(names, dict) else {},
            settings_map,
        )
