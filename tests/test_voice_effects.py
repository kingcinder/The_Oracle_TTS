from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from the_oracle.audio.voice_effects import process_voice_audio, process_voice_file


def test_pitch_processing_changes_pitch_without_changing_duration():
    sample_rate = 24_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    source = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    processed = process_voice_audio(
        source, sample_rate, pitch_semitones=3.0, tone_db=0.0, timbre_db=0.0
    )

    peak_bin = int(np.argmax(np.abs(np.fft.rfft(processed))))
    peak_hz = peak_bin * sample_rate / len(processed)
    assert len(processed) == len(source)
    assert peak_hz == pytest.approx(523, abs=2)


def test_tone_processing_changes_spectrum_and_leaves_neutral_audio_unchanged():
    sample_rate = 24_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    source = (np.sin(2 * np.pi * 180 * time) + np.sin(2 * np.pi * 5000 * time)).astype(np.float32)

    neutral = process_voice_audio(source, sample_rate)
    brighter = process_voice_audio(
        source, sample_rate, pitch_semitones=0.0, tone_db=0.0, timbre_db=9.0
    )

    assert np.array_equal(neutral, source)
    low = np.abs(np.fft.rfft(brighter))[180]
    high = np.abs(np.fft.rfft(brighter))[5000]
    assert high > low


def test_save_as_writes_processed_copy_and_preserves_original(tmp_path):
    source_path = tmp_path / "voice.wav"
    copy_path = tmp_path / "voice_processed.wav"
    sample_rate = 24_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    source = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    sf.write(source_path, source, sample_rate, subtype="PCM_16")
    original_bytes = source_path.read_bytes()

    process_voice_file(
        str(source_path), str(copy_path), pitch_semitones=3.0
    )

    assert source_path.read_bytes() == original_bytes
    assert copy_path.is_file()
    processed, processed_rate = sf.read(copy_path, dtype="float32")
    assert processed_rate == sample_rate
    assert len(processed) == len(source)
    assert not np.array_equal(processed, source)


def test_save_replaces_source_atomically_with_processed_wav(tmp_path):
    source_path = tmp_path / "voice.wav"
    sample_rate = 24_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    source = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    sf.write(source_path, source, sample_rate, subtype="PCM_16")

    process_voice_file(source_path, source_path, pitch_semitones=3.0)

    processed, processed_rate = sf.read(source_path, dtype="float32")
    peak_bin = int(np.argmax(np.abs(np.fft.rfft(processed))))
    assert processed_rate == sample_rate
    assert peak_bin == pytest.approx(523, abs=2)
