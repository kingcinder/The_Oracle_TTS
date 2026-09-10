"""Microphone capture and "Seashell" recording helpers for the Recording Studio.

Everything here is deliberately thin and backend-agnostic so the GUI stays
testable without a microphone: device listing, supported-sample-rate probing,
mono WAV saving, auto-incrementing Seashell names, and a stop-driven capture
loop. ``sounddevice`` is imported lazily; when the wheel is absent (or tests
inject a fake) the module-level ``_sd`` global is swapped, and every function
reports availability through :func:`have_capture_backend`.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

try:  # sounddevice ships its own PortAudio; degrade gracefully if absent.
    import sounddevice as _sd  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - depends on local install state
    _sd = None  # type: ignore[assignment]

_SeashellNameRe = re.compile(r"Seashell_No_(\d+)", re.IGNORECASE)

# Candidate sample rates tried against the selected device; whichever the
# device accepts (sd.check_input_settings) become the dropdown options.
_COMMON_SAMPLE_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000)


@dataclass(frozen=True, slots=True)
class InputDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: int


class RecorderUnavailableError(RuntimeError):
    """Raised when no capture backend (sounddevice) is importable."""


def have_capture_backend() -> bool:
    return _sd is not None


def list_input_devices() -> list[InputDevice]:
    """Input devices detected by the backend, default input device first."""
    if _sd is None:
        return []
    try:
        devices = _sd.query_devices()
        default_index = _sd.default.device[0]
    except Exception:
        return []
    inputs = [
        InputDevice(
            index=index,
            name=str(device.get("name", f"Device {index}")),
            max_input_channels=int(device.get("max_input_channels", 0)),
            default_samplerate=int(device.get("default_samplerate", 48000)),
        )
        for index, device in enumerate(devices)
        if int(device.get("max_input_channels", 0)) > 0
    ]
    return sorted(inputs, key=lambda d: (d.index != default_index, d.index))


def samplerates_for_device(device_index: int) -> list[int]:
    """Sample rates the selected input device actually supports.

    Probes the common rates with ``check_input_settings`` (one channel). The
    device's default rate leads the list when it survives the probe; if no
    common rate is accepted, the default alone is returned so the UI still has
    something valid to record at.
    """
    if _sd is None:
        raise RecorderUnavailableError("sounddevice is not installed; no capture backend available.")
    supported: list[int] = []
    default_rate: int = 48000
    try:
        info = _sd.query_devices(device_index)
        default_rate = int(info.get("default_samplerate", 48000))
    except Exception:
        pass
    for rate in _COMMON_SAMPLE_RATES:
        try:
            _sd.check_input_settings(device=device_index, channels=1, samplerate=float(rate))
        except Exception:
            continue
        if rate not in supported:
            supported.append(rate)
    if default_rate not in supported:
        supported.append(default_rate)
    # Default rate first when present.
    return sorted(supported, key=lambda r: (r != default_rate, r))


def next_seashell_name(voice_dir: str | Path) -> str:
    """Next auto name (stem only) that would not overwrite an existing file.

    Scans ``Seashell_No_<n>*`` files in ``voice_dir`` and returns
    ``Seashell_No_<max+1>`` (or ``Seashell_No_1`` when none exist yet).
    """
    directory = Path(voice_dir)
    highest = 0
    if directory.is_dir():
        for entry in directory.iterdir():
            if entry.is_file():
                match = _SeashellNameRe.match(entry.stem)
                if match:
                    highest = max(highest, int(match.group(1)))
    return f"Seashell_No_{highest + 1}"


def allocate_seashell_path(voice_dir: str | Path, suffix: str = ".wav") -> Path:
    """Atomically allocate the next Seashell recording path.

    :func:`next_seashell_name` only *suggests* a name: two recorders racing
    can both be handed the same name, and the second save then silently
    overwrites the first (a TOCTOU race). This allocator closes the race with
    ``O_EXCL`` creation — the returned path refers to an actually-created
    (empty) file, so no other allocator call, in this process or another,
    can ever hand out the same name. Retries on ``FileExistsError`` (another
    allocator won the race for that number) until it wins one.

    Callers should write the audio into the returned path (e.g. via
    :func:`save_recording_wav`, which truncates the placeholder safely).
    """
    directory = Path(voice_dir)
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        candidate = directory / f"{next_seashell_name(directory)}{suffix}"
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate


def save_recording_wav(path: str | Path, audio: np.ndarray, samplerate: int) -> Path:
    """Write captured audio to a mono WAV (PCM_16) at ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    sf.write(str(destination), samples, int(samplerate), format="WAV", subtype="PCM_16")
    return destination


def record_until_stop(
    device_index: int,
    samplerate: int,
    channels: int = 1,
    stop_event: threading.Event | None = None,
    on_level=None,
    poll_interval: float = 0.05,
) -> np.ndarray:
    """Record until ``stop_event`` is set; yield float32 mono samples.

    ``on_level`` (callable RMS) is invoked for each input block so the UI can
    show a live level meter. When ``stop_event`` is already set on entry, the
    stream is still opened and immediately stopped, returning whatever the
    device delivered during setup (typically an empty array).
    """
    if _sd is None:
        raise RecorderUnavailableError("sounddevice is not installed; no capture backend available.")
    stop_event = stop_event or threading.Event()
    chunks: list[np.ndarray] = []
    lock = threading.Lock()
    stopped = threading.Event()

    def _callback(indata, frames, _time_info, status) -> None:  # type: ignore[no-untyped-def]
        block = np.asarray(indata, dtype=np.float32).copy()
        with lock:
            chunks.append(block)
        if on_level is not None:
            level = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
            on_level(level)

    with _sd.InputStream(
        device=device_index,
        samplerate=int(samplerate),
        channels=max(1, int(channels)),
        dtype="float32",
        callback=_callback,
    ) as stream:
        stream.start()
        try:
            while not stop_event.is_set():
                time.sleep(poll_interval)
        finally:
            stream.stop()
            stopped.set()
    with lock:
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio
