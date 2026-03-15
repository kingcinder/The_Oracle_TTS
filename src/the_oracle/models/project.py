"""Dataclasses for analysis, review, render planning, and Chatterbox voice profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from the_oracle.utils.hashing import hash_payload


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class VoiceSettings:
    variant: str = "standard"
    language: str = "en"
    cfg_weight: float = 0.5
    exaggeration: float = 0.5
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    min_p: float = 0.05
    top_p: float = 1.0
    top_k: int = 1000
    norm_loudness: bool = True
    pause_ms: int = 180
    crossfade_ms: int = 20
    emotion_intensity: float = 1.0
    naturalness: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | "VoiceSettings" | None) -> "VoiceSettings":
        if isinstance(payload, cls):
            return payload
        if payload is None:
            return cls()
        filtered = {field_name: payload[field_name] for field_name in cls.__dataclass_fields__ if field_name in payload}
        return cls(**filtered)


@dataclass(slots=True)
class CorrectionRecord:
    stage: str
    before: str
    after: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class VoiceProfile:
    name: str
    speaker: str
    reference_audio: list[Path] = field(default_factory=list)
    neutral_reference: Path | None = None
    emotion_references: dict[str, Path] = field(default_factory=dict)
    engine_params: VoiceSettings = field(default_factory=VoiceSettings)
    conditioning_cache_id: str = ""
    normalized_reference_audio: list[Path] = field(default_factory=list)
    reference_audio_hash: str = ""

    @property
    def primary_reference(self) -> Path:
        if self.neutral_reference:
            return self.neutral_reference
        if self.reference_audio:
            return self.reference_audio[0]
        raise ValueError(f"Voice profile {self.speaker} has no reference audio configured.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "speaker": self.speaker,
            "reference_audio": [str(path) for path in self.reference_audio],
            "neutral_reference": str(self.neutral_reference) if self.neutral_reference else None,
            "emotion_references": {key: str(value) for key, value in self.emotion_references.items()},
            "engine_params": self.engine_params.to_dict(),
            "conditioning_cache_id": self.conditioning_cache_id,
            "normalized_reference_audio": [str(path) for path in self.normalized_reference_audio],
            "reference_audio_hash": self.reference_audio_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VoiceProfile":
        return cls(
            name=payload.get("name", ""),
            speaker=payload.get("speaker", ""),
            reference_audio=[Path(path) for path in payload.get("reference_audio", [])],
            neutral_reference=Path(payload["neutral_reference"]) if payload.get("neutral_reference") else None,
            emotion_references={key: Path(value) for key, value in payload.get("emotion_references", {}).items()},
            engine_params=VoiceSettings.from_mapping(payload.get("engine_params")),
            conditioning_cache_id=payload.get("conditioning_cache_id", ""),
            normalized_reference_audio=[Path(path) for path in payload.get("normalized_reference_audio", [])],
            reference_audio_hash=payload.get("reference_audio_hash", ""),
        )


@dataclass(slots=True)
class Utterance:
    index: int
    original_text: str
    repaired_text: str = ""
    speaker: str = "A"
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    emotion_score: float = 0.0
    speaker_confidence: float = 0.0
    speaker_source: str = "unknown"
    explicit_speaker: str | None = None
    source_line: int | None = None
    original_start_line: int = 0
    original_end_line: int = 0
    duration_seconds: float | None = None
    pause_after_ms: int = 180
    parameters: dict[str, Any] = field(default_factory=dict)
    corrections: list[CorrectionRecord] = field(default_factory=list)
    manual_speaker_override: bool = False
    manual_text_override: bool = False
    manual_emotion_override: bool = False
    chunk_hash: str = ""
    cache_key: str = ""
    engine_settings: VoiceSettings = field(default_factory=VoiceSettings)

    def text_for_tts(self) -> str:
        return self.repaired_text or self.original_text

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["corrections"] = [item.to_dict() for item in self.corrections]
        payload["engine_settings"] = self.engine_settings.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Utterance":
        corrections = [CorrectionRecord(**item) for item in payload.get("corrections", [])]
        engine_settings = VoiceSettings.from_mapping(payload.get("engine_settings"))
        return cls(**{**payload, "corrections": corrections, "engine_settings": engine_settings})


@dataclass(slots=True)
class RenderPlan:
    title: str
    source_path: str
    output_dir: str
    engine: str
    correction_mode: str
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, str] = field(default_factory=dict)
    utterances: list[Utterance] = field(default_factory=list)
    voice_profiles: dict[str, VoiceProfile] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source_path": self.source_path,
            "output_dir": self.output_dir,
            "engine": self.engine,
            "correction_mode": self.correction_mode,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "utterances": [utterance.to_dict() for utterance in self.utterances],
            "voice_profiles": {speaker: profile.to_dict() for speaker, profile in self.voice_profiles.items()},
            "hashes": self.hashes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RenderPlan":
        return cls(
            title=payload["title"],
            source_path=payload["source_path"],
            output_dir=payload["output_dir"],
            engine=payload["engine"],
            correction_mode=payload["correction_mode"],
            created_at=payload.get("created_at", utc_now_iso()),
            metadata=payload.get("metadata", {}),
            utterances=[Utterance.from_dict(item) for item in payload.get("utterances", [])],
            voice_profiles={speaker: VoiceProfile.from_dict(item) for speaker, item in payload.get("voice_profiles", {}).items()},
            hashes=payload.get("hashes", {}),
        )

    def update_hashes(self) -> None:
        utterance_payload = [utterance.to_dict() for utterance in self.utterances]
        profiles_payload = {speaker: profile.to_dict() for speaker, profile in self.voice_profiles.items()}
        self.hashes["utterances"] = hash_payload(utterance_payload)
        self.hashes["voice_profiles"] = hash_payload(profiles_payload)
        self.hashes["plan"] = hash_payload(
            {
                "title": self.title,
                "source_path": self.source_path,
                "engine": self.engine,
                "correction_mode": self.correction_mode,
                "metadata": self.metadata,
                "utterances": utterance_payload,
                "voice_profiles": profiles_payload,
            }
        )


@dataclass(slots=True)
class PreparedProject:
    config: dict[str, Any]
    utterances: list[Utterance]
    detected_names: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RenderResult:
    final_output: Path
    render_plan: RenderPlan
    stems: list[Path]
    changed_keys: list[str]
