"""Versioned save/load support for editable Chatterbox projects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from the_oracle.models.project import RenderPlan
from the_oracle.pipeline import RenderSettings, SpeakerSettings


PROJECT_MANIFEST_VERSION = 1


class ProjectManifestError(ValueError):
    """Raised when a saved project manifest is missing required fields or is incompatible."""


@dataclass(slots=True)
class SavedProject:
    manifest_version: int
    title: str
    artist: str
    input_path: str
    output_path: str
    engine: str
    model_variant: str
    render_settings: RenderSettings
    speaker_settings: dict[str, SpeakerSettings]
    plan: RenderPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "title": self.title,
            "artist": self.artist,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "engine": self.engine,
            "model_variant": self.model_variant,
            "render_settings": _render_settings_to_dict(self.render_settings),
            "speaker_settings": {speaker: _speaker_settings_to_dict(settings) for speaker, settings in self.speaker_settings.items()},
            "utterances": [_editable_utterance_state(item) for item in self.plan.utterances],
            "render_plan": self.plan.to_dict(),
        }


def _render_settings_to_dict(settings: RenderSettings) -> dict[str, Any]:
    return {
        "correction_mode": settings.correction_mode,
        "model_variant": settings.model_variant,
        "language": settings.language,
        "export_stems": settings.export_stems,
        "loudness_preset": settings.loudness_preset,
        "pause_between_turns_ms": settings.pause_between_turns_ms,
        "crossfade_ms": settings.crossfade_ms,
        "device_mode": settings.device_mode,
        "metadata": dict(settings.metadata),
    }


def _speaker_settings_to_dict(settings: SpeakerSettings) -> dict[str, Any]:
    return {
        "reference_path": settings.reference_path,
        "voice_settings": settings.voice_settings.to_dict() if hasattr(settings.voice_settings, "to_dict") else dict(settings.voice_settings),
        "emotion_reference_paths": dict(settings.emotion_reference_paths),
    }


def _editable_utterance_state(utterance) -> dict[str, Any]:
    return {
        "index": utterance.index,
        "speaker": utterance.speaker,
        "repaired_text": utterance.repaired_text,
        "emotion": utterance.emotion,
        "manual_speaker_override": utterance.manual_speaker_override,
        "manual_text_override": utterance.manual_text_override,
        "manual_emotion_override": utterance.manual_emotion_override,
    }


def build_saved_project(plan: RenderPlan, render_settings: RenderSettings, speaker_settings: dict[str, SpeakerSettings]) -> SavedProject:
    return SavedProject(
        manifest_version=PROJECT_MANIFEST_VERSION,
        title=plan.title,
        artist=plan.metadata.get("artist", "The Oracle"),
        input_path=plan.source_path,
        output_path=plan.output_dir,
        engine="chatterbox",
        model_variant=render_settings.model_variant,
        render_settings=render_settings,
        speaker_settings=speaker_settings,
        plan=plan,
    )


def save_project_manifest(path: str | Path, project: SavedProject) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(project.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
    return destination


def load_project_manifest(path: str | Path) -> SavedProject:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return saved_project_from_dict(payload)


def saved_project_from_dict(payload: dict[str, Any]) -> SavedProject:
    required_fields = {
        "manifest_version",
        "title",
        "artist",
        "input_path",
        "output_path",
        "engine",
        "model_variant",
        "render_settings",
        "speaker_settings",
        "render_plan",
    }
    missing = sorted(field for field in required_fields if field not in payload)
    if missing:
        raise ProjectManifestError(f"Project manifest is missing required fields: {', '.join(missing)}")
    if payload["manifest_version"] != PROJECT_MANIFEST_VERSION:
        raise ProjectManifestError(
            f"Unsupported project manifest version {payload['manifest_version']}; expected {PROJECT_MANIFEST_VERSION}."
        )
    if payload["engine"] != "chatterbox":
        raise ProjectManifestError(f"Unsupported engine '{payload['engine']}'. Only 'chatterbox' is supported.")

    speakers_payload = payload["speaker_settings"]
    if not isinstance(speakers_payload, dict) or set(speakers_payload) != {"A", "B"}:
        raise ProjectManifestError("Project manifest must contain speaker settings for both 'A' and 'B'.")

    render_settings = RenderSettings(**payload["render_settings"])
    speaker_settings = {
        speaker: SpeakerSettings(
            reference_path=value["reference_path"],
            voice_settings=value.get("voice_settings", {}),
            emotion_reference_paths=value.get("emotion_reference_paths", {}),
        )
        for speaker, value in speakers_payload.items()
    }
    plan = RenderPlan.from_dict(payload["render_plan"])
    return SavedProject(
        manifest_version=payload["manifest_version"],
        title=payload["title"],
        artist=payload["artist"],
        input_path=payload["input_path"],
        output_path=payload["output_path"],
        engine=payload["engine"],
        model_variant=payload["model_variant"],
        render_settings=render_settings,
        speaker_settings=speaker_settings,
        plan=plan,
    )
