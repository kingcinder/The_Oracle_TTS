"""FLAC export with PySoundFile primary writing and ffmpeg fallback metadata tagging."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def _tag_with_mutagen(path: Path, metadata: dict[str, str]) -> None:
    from mutagen.flac import FLAC

    tags = FLAC(str(path))
    for key, value in metadata.items():
        if value:
            tags[key] = [str(value)]
    tags.save()


def _ffmpeg_write(path: Path, audio: np.ndarray, sample_rate: int, metadata: dict[str, str]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FLAC export failed and ffmpeg is not available.")

    temp_wav = path.with_suffix(".tmp.wav")
    sf.write(str(temp_wav), audio, sample_rate, format="WAV")
    command = [ffmpeg, "-y", "-i", str(temp_wav)]
    for key, value in metadata.items():
        if value:
            command.extend(["-metadata", f"{key}={value}"])
    command.extend(["-c:a", "flac", str(path)])
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    finally:
        temp_wav.unlink(missing_ok=True)


def write_flac(path: str | Path, audio: np.ndarray, sample_rate: int, metadata: dict[str, str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(destination), np.asarray(audio, dtype=np.float32), sample_rate, format="FLAC")
        _tag_with_mutagen(destination, metadata)
    except Exception:
        _ffmpeg_write(destination, audio, sample_rate, metadata)
    return destination


def export_flac(input_wav: str | Path, output_flac: str | Path, metadata: dict[str, str]) -> Path:
    audio, sample_rate = sf.read(str(input_wav), always_2d=False)
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return write_flac(output_flac, array, sample_rate, metadata)
