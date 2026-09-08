from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from the_oracle.audio import recorder


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._started = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def start(self):
        self._started = True

    def stop(self):
        self._started = False


class _FakeSD:
    def __init__(self, devices=None, supported_rates=(48000, 44100)):
        self._devices = devices or [
            {"name": "First Mic", "max_input_channels": 1, "default_samplerate": 48000},
            {"name": "Studio Interface", "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "Output Only", "max_input_channels": 0, "default_samplerate": 48000},
        ]
        self._supported = supported_rates
        self.default = SimpleNamespace(device=[0, 1])
        self.streams = []

    def query_devices(self, device=None):
        if device is None:
            return self._devices
        return self._devices[device]

    def check_input_settings(self, **kwargs):
        if kwargs.get("samplerate") not in self._supported:
            raise ValueError("unsupported")

    def InputStream(self, **kwargs):
        stream = _FakeStream(**kwargs)
        self.streams.append(stream)
        return stream


@pytest.fixture
def fake_sd(monkeypatch):
    fake = _FakeSD()
    monkeypatch.setattr(recorder, "_sd", fake)
    return fake


def test_have_capture_backend_reflects_import(monkeypatch):
    monkeypatch.setattr(recorder, "_sd", object())
    assert recorder.have_capture_backend() is True
    monkeypatch.setattr(recorder, "_sd", None)
    assert recorder.have_capture_backend() is False


def test_list_input_devices_excludes_non_inputs_and_defaults_first(fake_sd):
    devices = recorder.list_input_devices()
    assert [d.name for d in devices] == ["First Mic", "Studio Interface"]
    assert devices[0].index == 0  # default input first
    assert devices[1].default_samplerate == 44100


def test_samplerates_for_device_probes_and_defaults_first(fake_sd):
    rates = recorder.samplerates_for_device(1)
    assert 44100 == rates[0]  # default first
    assert 48000 in rates
    assert 44100 in rates
    assert 22050 not in rates


def test_samplerates_raise_without_backend(monkeypatch):
    monkeypatch.setattr(recorder, "_sd", None)
    with pytest.raises(recorder.RecorderUnavailableError):
        recorder.samplerates_for_device(0)


def test_next_seashell_name_increments(tmp_path: Path):
    assert recorder.next_seashell_name(tmp_path) == "Seashell_No_1"
    (tmp_path / "Seashell_No_1.wav").write_bytes(b"")
    (tmp_path / "Seashell_No_3.wav").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert recorder.next_seashell_name(tmp_path) == "Seashell_No_4"


def test_save_recording_wav_writes_mono_pcm(tmp_path: Path):
    stereo = np.full((4800, 2), 0.4, dtype=np.float32)
    destination = recorder.save_recording_wav(tmp_path / "Seashell_No_1.wav", stereo, 48000)
    assert destination.exists()
    audio, rate = sf.read(str(destination), always_2d=False)
    assert rate == 48000
    assert audio.ndim == 1
    assert float(np.abs(audio).max()) == pytest.approx(0.4, abs=0.01)
