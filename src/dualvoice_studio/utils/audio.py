from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def trim_silence(audio: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    if audio.size == 0:
        return audio
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    active = np.where(np.abs(mono) > threshold)[0]
    if active.size == 0:
        return audio
    return audio[active[0] : active[-1] + 1]


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32)
    duration = len(audio) / float(source_sr)
    source_positions = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_length = max(1, int(round(duration * target_sr)))
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    if audio.ndim == 1:
        return np.interp(target_positions, source_positions, audio).astype(np.float32)
    channels = [np.interp(target_positions, source_positions, audio[:, index]) for index in range(audio.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def apply_fade(audio: np.ndarray, sample_rate: int, fade_ms: int) -> np.ndarray:
    if fade_ms <= 0 or audio.size == 0:
        return audio
    fade_samples = min(int(sample_rate * (fade_ms / 1000.0)), len(audio) // 2)
    if fade_samples <= 0:
        return audio
    envelope = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    faded = audio.astype(np.float32).copy()
    faded[:fade_samples] *= envelope
    faded[-fade_samples:] *= envelope[::-1]
    return faded


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    return (audio - np.mean(audio)).astype(np.float32)


def normalize_loudness(audio: np.ndarray, preset: str = "light") -> np.ndarray:
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms <= 1e-6:
        return audio
    target_rms = 0.14 if preset == "medium" else 0.11
    gain = min(2.5, target_rms / rms)
    normalized = audio * gain
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized = normalized / peak * 0.99
    return normalized.astype(np.float32)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32")
    return audio, sample_rate


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def write_audio_ffmpeg(input_wav: Path, output_flac: Path, metadata: dict[str, str]) -> None:
    command = ["ffmpeg", "-y", "-i", str(input_wav)]
    for key, value in metadata.items():
        command.extend(["-metadata", f"{key}={value}"])
    command.append(str(output_flac))
    subprocess.run(command, check=True, capture_output=True)
