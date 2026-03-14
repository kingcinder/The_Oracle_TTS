from __future__ import annotations

from pathlib import Path

import soundfile as sf

from dualvoice_studio.utils.audio import ensure_mono, resample_audio, trim_silence
from dualvoice_studio.utils.hashing import hash_file


def normalize_reference_audio(source: Path, destination_dir: Path, target_sr: int = 24000) -> tuple[Path, str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = sf.read(source, dtype="float32")
    mono = ensure_mono(audio)
    trimmed = trim_silence(mono)
    normalized = resample_audio(trimmed, sample_rate, target_sr)
    destination = destination_dir / f"{source.stem}_{target_sr}.wav"
    sf.write(destination, normalized, target_sr)
    return destination, hash_file(destination)
