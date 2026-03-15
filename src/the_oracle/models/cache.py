"""Project cache helpers for stems, plans, references, and conditioning assets."""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from the_oracle.models.project import RenderPlan
from the_oracle.utils.hashing import build_chunk_hash, hash_file, hash_payload


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)
    duration = len(audio) / float(source_rate)
    source_positions = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_length = max(1, int(round(duration * target_rate)))
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    if audio.size == 0:
        return audio
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    active = np.where(np.abs(mono) >= threshold)[0]
    if active.size == 0:
        return audio
    return audio[int(active[0]) : int(active[-1]) + 1]


@dataclass(slots=True)
class CachedReference:
    original_path: str
    normalized_path: str
    original_hash: str
    sample_rate: int


@dataclass(slots=True)
class CachePaths:
    project_root: Path
    cache_dir: Path
    stems_dir: Path
    logs_dir: Path
    voice_dir: Path

    @classmethod
    def build(cls, output_dir: Path, project_name: str) -> "CachePaths":
        safe_name = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in project_name).strip("_") or "project"
        project_root = output_dir / f"{safe_name}_oracle"
        return cls(
            project_root=project_root,
            cache_dir=project_root / "cache",
            stems_dir=project_root / "stems",
            logs_dir=project_root / "logs",
            voice_dir=project_root / "voices",
        )

    def ensure(self) -> None:
        for path in (self.project_root, self.cache_dir, self.stems_dir, self.logs_dir, self.voice_dir):
            path.mkdir(parents=True, exist_ok=True)


class ProjectCache:
    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir)
        self.cache_dir = self.project_dir / "cache"
        self.stem_cache_dir = self.cache_dir / "utterances"
        self.reference_cache_dir = self.cache_dir / "references"
        self.conditioning_cache_dir = self.cache_dir / "conditioning"
        self.preview_dir = self.project_dir / "previews"
        self.log_dir = self.project_dir / "logs"
        for path in (
            self.project_dir,
            self.cache_dir,
            self.stem_cache_dir,
            self.reference_cache_dir,
            self.conditioning_cache_dir,
            self.preview_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def stem_path(self, chunk_hash: str) -> Path:
        return self.stem_cache_dir / f"{chunk_hash}.wav"

    def preview_path(self, speaker: str, utterance_index: int) -> Path:
        safe_speaker = "".join(character for character in speaker if character.isalnum()) or "speaker"
        return self.preview_dir / f"preview_{safe_speaker}_{utterance_index:04d}.wav"

    def conditioning_path(self, cache_id: str) -> Path:
        return self.conditioning_cache_dir / f"{cache_id}.pkl"

    def cache_reference_audio(self, source_path: str | Path, speaker: str, sample_rate: int) -> CachedReference:
        source = Path(source_path)
        original_hash = hash_file(source)
        cached_path = self.reference_cache_dir / f"{speaker}_{original_hash[:12]}_{sample_rate}.wav"
        if not cached_path.exists():
            audio, read_rate = sf.read(str(source), always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = _trim_silence(audio)
            audio = _resample_linear(audio, read_rate, sample_rate)
            sf.write(cached_path, audio, sample_rate)
        return CachedReference(str(source), str(cached_path), original_hash, sample_rate)

    def save_json(self, relative_path: str, payload: dict[str, Any]) -> Path:
        destination = self.project_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
        return destination

    def write_text(self, relative_path: str, content: str) -> Path:
        destination = self.project_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return destination

    def store_conditioning(self, speaker: str, reference_hash: str, payload: Any) -> str:
        cache_id = hash_payload({"speaker": speaker, "reference_hash": reference_hash})
        with self.conditioning_path(cache_id).open("wb") as handle:
            pickle.dump(payload, handle)
        return cache_id

    def load_conditioning(self, cache_id: str) -> Any | None:
        path = self.conditioning_path(cache_id)
        if not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def export_stem(self, stem_path: str | Path, relative_path: str) -> Path:
        destination = self.project_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stem_path, destination)
        return destination


def build_chunk_cache_key(
    *,
    speaker: str,
    repaired_text: str,
    engine_name: str,
    engine_version: str,
    engine_params: dict[str, Any],
    reference_audio_hash: str,
) -> str:
    return build_chunk_hash(
        speaker=speaker,
        repaired_text=repaired_text,
        engine_key=engine_name,
        engine_params=engine_params,
        engine_version=engine_version,
        reference_audio_hash=reference_audio_hash,
    )


def input_fingerprint(input_path: Path) -> str:
    return hash_payload({"path": str(input_path.resolve()), "sha256": hash_file(input_path)})


def write_render_plan(plan: RenderPlan, destination: Path) -> None:
    plan.update_hashes()
    destination.write_text(json.dumps(plan.to_dict(), indent=2, default=str), encoding="utf-8")


def read_previous_render_plan(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_conditioning_id(speaker: str, references: list[Path]) -> str:
    return hash_payload({"speaker": speaker, "references": [hash_file(path) for path in references]})
