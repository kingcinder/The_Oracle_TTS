"""Versioned GUI settings profiles and reusable local templates."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


GUI_SETTINGS_VERSION = 1


class GUISettingsError(ValueError):
    """Raised when a GUI settings profile is invalid or incompatible."""


def user_config_dir() -> Path:
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config) if xdg_config else Path.home() / ".config"
    return base / "dualvoice_studio"


def template_dir() -> Path:
    path = user_config_dir() / "templates"
    path.mkdir(parents=True, exist_ok=True)
    return path


def recent_references_path() -> Path:
    path = user_config_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path / "recent_reference_clips.json"


def save_gui_settings(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_normalize_payload(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    return destination


def load_gui_settings(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return _normalize_payload(payload)


def save_template(name: str, payload: dict[str, Any]) -> Path:
    destination = template_dir() / f"{_safe_name(name)}.json"
    return save_gui_settings(destination, payload)


def load_template(name: str) -> dict[str, Any]:
    return load_gui_settings(template_dir() / f"{_safe_name(name)}.json")


def list_templates() -> list[str]:
    return sorted(path.stem for path in template_dir().glob("*.json"))


def load_recent_reference_paths(limit: int = 10) -> list[str]:
    path = recent_references_path()
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [str(item) for item in payload if isinstance(item, str)]
    return entries[:limit]


def remember_recent_reference_path(path_value: str, limit: int = 10) -> None:
    normalized = str(Path(path_value).expanduser())
    existing = [item for item in load_recent_reference_paths(limit=limit * 2) if item != normalized]
    updated = [normalized, *existing][:limit]
    recent_references_path().write_text(json.dumps(updated, indent=2, ensure_ascii=True), encoding="utf-8")


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"version", "project", "speakers"}
    missing = sorted(required - set(payload))
    if missing:
        raise GUISettingsError(f"GUI settings profile is missing required fields: {', '.join(missing)}")
    if payload["version"] != GUI_SETTINGS_VERSION:
        raise GUISettingsError(f"Unsupported GUI settings version {payload['version']}; expected {GUI_SETTINGS_VERSION}.")
    speakers = payload["speakers"]
    if not isinstance(speakers, dict) or set(speakers) != {"A", "B"}:
        raise GUISettingsError("GUI settings profile must contain both speakers 'A' and 'B'.")
    normalized = {
        "version": GUI_SETTINGS_VERSION,
        "name": str(payload.get("name", "")),
        "project": dict(payload["project"]),
        "device_mode": str(payload.get("device_mode", "cpu")),
        "speakers": {},
    }
    for speaker, config in speakers.items():
        normalized["speakers"][speaker] = {
            "reference_path": str(config.get("reference_path", "")),
            "voice_settings": dict(config.get("voice_settings", {})),
            "emotion_reference_paths": dict(config.get("emotion_reference_paths", {})),
        }
    return normalized


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value).strip("_")
    return safe or "template"
