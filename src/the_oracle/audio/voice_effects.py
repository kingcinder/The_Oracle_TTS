"""Non-destructive voice tone controls for recorded reference audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import istft, resample, stft

from the_oracle.models.cache import atomic_write


def _stretch(audio: np.ndarray, rate: float, sample_rate: int) -> np.ndarray:
    """Phase-vocoder time stretch used to keep pitch edits at the same duration."""
    if len(audio) < 16 or abs(rate - 1.0) < 1e-8:
        return audio.copy()
    frame_size = min(2048, 2 ** int(np.floor(np.log2(len(audio)))))
    frame_size = max(16, frame_size)
    hop = frame_size // 4
    _, _, spectrum = stft(
        audio, fs=sample_rate, nperseg=frame_size,
        noverlap=frame_size - hop, boundary="zeros",
    )
    steps = np.arange(0, spectrum.shape[1], rate)
    magnitudes = np.abs(spectrum)
    phases = np.angle(spectrum)
    expected = 2.0 * np.pi * hop * np.arange(spectrum.shape[0]) / frame_size
    output = np.empty((spectrum.shape[0], len(steps)), dtype=np.complex128)
    phase = phases[:, 0].copy()
    for column, step in enumerate(steps):
        left = min(int(step), spectrum.shape[1] - 1)
        right = min(left + 1, spectrum.shape[1] - 1)
        fraction = step - left
        magnitude = magnitudes[:, left] * (1.0 - fraction) + magnitudes[:, right] * fraction
        if column:
            delta = phases[:, right] - phases[:, left] - expected
            delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
            phase += expected + delta
        output[:, column] = magnitude * np.exp(1j * phase)
    _, stretched = istft(
        output, fs=sample_rate, nperseg=frame_size,
        noverlap=frame_size - hop, input_onesided=True, boundary=True,
    )
    target_length = max(1, round(len(audio) / rate))
    if len(stretched) < target_length:
        stretched = np.pad(stretched, (0, target_length - len(stretched)))
    return stretched[:target_length].astype(np.float32)


def process_voice_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    pitch_semitones: float = 0.0,
    tone_db: float = 0.0,
    timbre_db: float = 0.0,
) -> np.ndarray:
    """Apply pitch, low-tone, and high-timbre adjustments to mono or multichannel audio."""
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim not in (1, 2) or sample_rate <= 0:
        raise ValueError("Audio must be mono or multichannel with a positive sample rate.")
    if not np.isfinite([pitch_semitones, tone_db, timbre_db]).all():
        raise ValueError("Voice effect settings must be finite numbers.")
    if not np.isfinite(source).all():
        raise ValueError("Audio contains non-finite samples.")
    if source.size == 0 or (pitch_semitones == 0 and tone_db == 0 and timbre_db == 0):
        return source.copy()
    channels = source[:, None] if source.ndim == 1 else source
    factor = 2.0 ** (float(pitch_semitones) / 12.0)
    rate = 1.0 / factor
    output = np.column_stack([
        resample(_stretch(channels[:, index], rate, sample_rate), len(source))
        for index in range(channels.shape[1])
    ]).astype(np.float32)
    frequencies = np.fft.rfftfreq(len(output), 1.0 / sample_rate)
    gains_db = (
        float(tone_db) / (1.0 + (frequencies / 450.0) ** 4)
        + float(timbre_db) / (1.0 + (2200.0 / np.maximum(frequencies, 1.0)) ** 4)
    )
    spectrum = np.fft.rfft(output, axis=0)
    output = np.fft.irfft(
        spectrum * (10.0 ** (gains_db / 20.0))[:, None],
        n=len(output), axis=0,
    ).astype(np.float32)
    peak = float(np.max(np.abs(output), initial=0.0))
    if peak > 0.999:
        output *= 0.999 / peak
    return output[:, 0] if source.ndim == 1 else output


def process_voice_file(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    pitch_semitones: float = 0.0,
    tone_db: float = 0.0,
    timbre_db: float = 0.0,
) -> None:
    """Process a recording and atomically write PCM WAV, safe for in-place replacement."""
    audio, sample_rate = sf.read(source_path, dtype="float32", always_2d=False)
    processed = process_voice_audio(
        audio, sample_rate, pitch_semitones=pitch_semitones,
        tone_db=tone_db, timbre_db=timbre_db,
    )
    destination = Path(destination_path)
    destination_suffix = destination.suffix.lower().lstrip(".")
    output_format = {"wav": "WAV", "flac": "FLAC"}.get(destination_suffix)
    if output_format is None:
        raise ValueError("Processed recordings must be saved as WAV or FLAC.")
    atomic_write(
        destination,
        lambda temporary: sf.write(
            str(temporary), processed, sample_rate,
            format=output_format, subtype="PCM_16",
        ),
    )
