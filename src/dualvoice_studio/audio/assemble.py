"""Assemble utterance stems into a continuous dialogue performance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(slots=True)
class AudioSegment:
    path: str
    sample_rate: int
    pause_after_ms: int
    duration_seconds: float


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    return audio - np.mean(audio) if audio.size else audio


def apply_fade(audio: np.ndarray, sample_rate: int, fade_ms: int = 10) -> np.ndarray:
    if audio.size == 0:
        return audio
    fade_samples = max(1, int(sample_rate * fade_ms / 1000))
    fade_samples = min(fade_samples, len(audio) // 2 or 1)
    envelope = np.ones(len(audio), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, num=fade_samples, dtype=np.float32)
    envelope[:fade_samples] = ramp
    envelope[-fade_samples:] = ramp[::-1]
    return audio * envelope


def normalize_loudness(audio: np.ndarray, preset: str = "light") -> np.ndarray:
    if audio.size == 0:
        return audio
    target_rms = {"off": None, "light": 0.11, "medium": 0.15}.get(preset, 0.11)
    if target_rms is None:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio)))) or 1.0
    gain = min(1.8, target_rms / rms)
    return np.clip(audio * gain, -1.0, 1.0)


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array, sample_rate


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, format="WAV")


def assemble_dialogue(
    segments: list[AudioSegment],
    crossfade_ms: int = 20,
    loudness_preset: str = "light",
) -> tuple[np.ndarray, int]:
    if not segments:
        return np.zeros(1, dtype=np.float32), 24000

    final_audio: np.ndarray | None = None
    final_rate = segments[0].sample_rate
    crossfade_samples = max(0, int(final_rate * crossfade_ms / 1000))

    for segment in segments:
        audio, sample_rate = load_audio(segment.path)
        if sample_rate != final_rate:
            raise ValueError("All stems must share a sample rate before assembly.")
        audio = apply_fade(remove_dc_offset(audio), sample_rate)
        if final_audio is None:
            final_audio = audio.copy()
        else:
            if crossfade_samples > 0 and len(final_audio) > crossfade_samples and len(audio) > crossfade_samples:
                head = final_audio[:-crossfade_samples]
                tail = final_audio[-crossfade_samples:]
                lead = audio[:crossfade_samples]
                mix = tail * np.linspace(1.0, 0.0, crossfade_samples) + lead * np.linspace(0.0, 1.0, crossfade_samples)
                final_audio = np.concatenate([head, mix.astype(np.float32), audio[crossfade_samples:]])
            else:
                final_audio = np.concatenate([final_audio, audio])
        pause_samples = int(final_rate * segment.pause_after_ms / 1000)
        if pause_samples > 0:
            final_audio = np.concatenate([final_audio, np.zeros(pause_samples, dtype=np.float32)])

    final_audio = normalize_loudness(final_audio, preset=loudness_preset)
    return np.asarray(final_audio, dtype=np.float32), final_rate


def assemble_stems(
    stems: list[Path],
    output_wav: Path,
    sample_rate: int,
    pause_ms: int,
    crossfade_ms: int,
    normalize_output: bool,
    normalization_preset: str,
) -> None:
    segments = [
        AudioSegment(path=str(path), sample_rate=sample_rate, pause_after_ms=pause_ms, duration_seconds=0.0)
        for path in stems
    ]
    audio, actual_rate = assemble_dialogue(
        segments,
        crossfade_ms=crossfade_ms,
        loudness_preset=normalization_preset if normalize_output else "off",
    )
    save_wav(output_wav, audio, actual_rate)
