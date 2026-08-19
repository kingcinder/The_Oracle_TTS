"""Versioned GUI settings profiles and reusable local templates."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from the_oracle.correction_modes import normalize_correction_mode
from the_oracle.platform_support import app_config_dir


GUI_SETTINGS_VERSION = 1
_SUPPORTED_DEVICE_MODES = {"cpu"}
_SUPPORTED_INFERENCE_BACKENDS = {"pytorch", "vulkan"}
_LOG = logging.getLogger(__name__)


class GUISettingsError(ValueError):
    """Raised when a GUI settings profile is invalid or incompatible."""


def user_config_dir() -> Path:
    return app_config_dir("the_oracle")


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


def app_settings_path() -> Path:
    """Path of the app-level settings file (backend memory + audio.cpp paths).

    Unlike profiles/templates, this file is written automatically (not by a
    Save dialog) and read at every app launch, so the chosen inference backend
    and the resolved ``ORACLE_AUDIOCPP_CLI``/``ORACLE_AUDIOCPP_MODEL`` paths
    survive across sessions without any manual wiring.
    """
    path = user_config_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path / "app_settings.json"


def _normalize_path_string(value: Any) -> str:
    """Coerce a persisted filesystem path to a plain string ("" when absent)."""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_remember_backend(value: Any) -> bool:
    """Coerce the remember-backend flag to a bool (default: True).

    ``bool("false")`` is True, so a hand-edited file that spells the flag as
    a string would otherwise flip the semantics; string values are parsed
    textually and anything else is truthy-checked.
    """
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _normalize_app_settings(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce an app-settings payload to the supported schema.

    A hand-edited or older file must never crash the app at launch: every
    field is normalized (unsupported backend -> pytorch, bad device index ->
    None, non-bool remember flag -> True) and unknown keys are dropped.
    """
    data = payload if isinstance(payload, dict) else {}
    return {
        "version": 1,
        "remember_backend": _normalize_remember_backend(data.get("remember_backend", True)),
        "inference_backend": _normalize_inference_backend(data.get("inference_backend", "pytorch")),
        "audio_cpp_device": _normalize_audio_cpp_device(data.get("audio_cpp_device")),
        "audio_cpp_threads": _normalize_audio_cpp_threads(data.get("audio_cpp_threads")),
        "audio_cpp_timeout": _normalize_audio_cpp_timeout(data.get("audio_cpp_timeout")),
        "audio_cpp_max_batch": _normalize_audio_cpp_max_batch(data.get("audio_cpp_max_batch")),
        "audio_cpp_cli": _normalize_path_string(data.get("audio_cpp_cli")),
        "audio_cpp_model": _normalize_path_string(data.get("audio_cpp_model")),
    }


def load_app_settings() -> dict[str, Any]:
    """Load the persisted app settings (backend memory + audio.cpp paths).

    Returns the normalized default payload when the file is missing or
    unreadable, so a fresh install or a corrupted file simply starts with
    defaults instead of crashing the GUI.
    """
    path = app_settings_path()
    if not path.exists():
        return _normalize_app_settings({})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _normalize_app_settings({})
    return _normalize_app_settings(payload)


def save_app_settings(payload: dict[str, Any]) -> Path:
    """Persist the app settings, normalizing so the file always round-trips."""
    path = app_settings_path()
    path.write_text(json.dumps(_normalize_app_settings(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    return path


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


def _normalize_device_mode(value: str) -> str:
    """Coerce device_mode to a supported value.

    Only "cpu" is a verified execution path.  Any other value in a saved
    settings file (e.g. an old "vulkan" entry) is silently replaced with
    "cpu" so round-tripped files stay clean and users are not left in an
    unverified state.
    """
    candidate = str(value).strip().lower()
    if candidate not in _SUPPORTED_DEVICE_MODES:
        _LOG.debug(
            "Unsupported device_mode %r in settings file, replacing with 'cpu'.",
            value,
        )
        return "cpu"
    return candidate


def _normalize_audio_cpp_device(value: Any) -> int | None:
    """Coerce audio_cpp_device to a non-negative int (Vulkan device index) or None.

    None / missing means "let audio.cpp pick"; anything non-numeric or
    negative is replaced with None so a hand-edited file cannot produce an
    invalid RenderSettings at load time.
    """
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _normalize_audio_cpp_threads(value: Any) -> int | None:
    """Coerce audio_cpp_threads to a positive int or None.

    None / missing means "audio.cpp's own default"; anything non-numeric or
    non-positive is replaced with None.
    """
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _normalize_audio_cpp_timeout(value: Any) -> int | None:
    """Coerce audio_cpp_timeout to a positive int (seconds) or None.

    None / missing means "the engine's default (600s)"; anything non-numeric
    or non-positive is replaced with None so a hand-edited file cannot produce
    an invalid RenderSettings at load time.
    """
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _normalize_audio_cpp_max_batch(value: Any) -> int | None:
    """Coerce audio_cpp_max_batch to a positive int (request count) or None.

    None / missing means "the engine's default (32)"; anything non-numeric
    or non-positive is replaced with None so a hand-edited file cannot produce
    an invalid RenderSettings at load time.
    """
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _normalize_inference_backend(value: str) -> str:
    """Coerce inference_backend to a supported value.

    The Vulkan backend is opt-in; "vulkan" is kept when present so saved
    settings round-trip faithfully. Any other value (e.g. an old or typo'd
    entry) is silently replaced with the default "pytorch".
    """
    candidate = str(value).strip().lower()
    if candidate not in _SUPPORTED_INFERENCE_BACKENDS:
        _LOG.debug(
            "Unsupported inference_backend %r in settings file, replacing with 'pytorch'.",
            value,
        )
        return "pytorch"
    return candidate


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"version", "project", "speakers"}
    missing = sorted(required - set(payload))
    if missing:
        raise GUISettingsError(f"GUI settings profile is missing required fields: {', '.join(missing)}")
    if payload["version"] != GUI_SETTINGS_VERSION:
        raise GUISettingsError(f"Unsupported GUI settings version {payload['version']}; expected {GUI_SETTINGS_VERSION}.")
    speakers = payload["speakers"]
    if not isinstance(speakers, dict) or not speakers:
        raise GUISettingsError("GUI settings profile must contain at least one speaker.")
    invalid_keys = [key for key in speakers if not re.fullmatch(r"[A-X]", str(key))]
    if invalid_keys:
        raise GUISettingsError(
            "GUI settings profile contains invalid speaker keys: "
            f"{', '.join(map(str, invalid_keys))}. Voices are A..X (up to 24)."
        )
    if len(speakers) > 24:
        raise GUISettingsError("GUI settings profile can carry at most 24 speaker voices.")
    project = dict(payload["project"])
    normalized_project = {
        "model_variant": str(project.get("model_variant", "standard")),
        "correction_mode": normalize_correction_mode(str(project.get("correction_mode", "moderate"))),
        "loudness_preset": str(project.get("loudness_preset", "light")),
        "crossfade_ms": int(project.get("crossfade_ms", 20)),
        "inference_backend": _normalize_inference_backend(project.get("inference_backend", "pytorch")),
        "audio_cpp_device": _normalize_audio_cpp_device(project.get("audio_cpp_device")),
        "audio_cpp_threads": _normalize_audio_cpp_threads(project.get("audio_cpp_threads")),
        "audio_cpp_timeout": _normalize_audio_cpp_timeout(project.get("audio_cpp_timeout")),
        "audio_cpp_max_batch": _normalize_audio_cpp_max_batch(project.get("audio_cpp_max_batch")),
        "output_dir": str(project.get("output_dir", "")),
        "output_filename": str(project.get("output_filename", "")),
    }
    for key, value in project.items():
        if key not in normalized_project:
            normalized_project[key] = value
    normalized = {
        "version": GUI_SETTINGS_VERSION,
        "name": str(payload.get("name", "")),
        "project": normalized_project,
        # Always normalise device_mode so stale values in old settings files
        # do not leave the app in an unverified execution state.
        "device_mode": _normalize_device_mode(payload.get("device_mode", "cpu")),
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
