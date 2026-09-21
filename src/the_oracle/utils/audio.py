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
    if audio.size == 0:
        # Empty input resamples to empty output (same channel layout); the
        # interpolation below would crash on zero sample points.
        return np.zeros(audio.shape, dtype=np.float32)
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


# --- TTS output sanitizer: one owner for engine-boundary hiccup defense ---
# Chatterbox generation occasionally hiccups (a documented model pathology,
# not an input problem): NaN/Inf samples from a diverged S3Gen pass, a
# constant DC "tone" from a degenerate decode, or a near-silent/empty stem
# when the token stream truncated right after SOT. Each failure mode used to
# fall through this boundary silently and became a one-way cached stem (the
# stem cache trusts whatever was written), so one bad sample hardened into a
# permanent defect in every later render. sanitize_engine_audio catches the
# recognizable modes at the single chokepoint both engines return through;
# unrecognizable audio passes through byte-identical.

#: A stem whose peak stays below this after DC removal is treated as empty
#: generation (truncation right after SOT), not speech.
_MIN_PEAK_AMPLITUDE = 1e-4

#: Fraction of samples at (near-)zero amplitude above which a stem is read as
#: a constant-DC tone (or its inverse: silence with a lone spike) rather than
#: speech. Real speech sits far below this duty cycle.
_MAX_ZERO_DUTY_CYCLE = 0.99


class DegenerateEngineOutput(ValueError):
    """The engine gate rejected synthesis output as degenerate.

    A subclass of ``ValueError`` because every existing caller treated these
    rejections as plain ValueErrors; the marker exists so the engines can
    distinguish a recognized one-off hiccup (worth one automatic re-synthesis
    at a fresh seed) from unrelated value errors with the same type. The
    message content is unchanged.
    """


def sanitize_engine_audio(audio: np.ndarray, *, text: str = "", sample_rate: int | None = None) -> np.ndarray:
    """Return *audio* with recognized TTS hiccups repaired or rejected.

    Repaired in place (returns a cleaned array):
      * non-finite samples (NaN/Inf) -> zeroed. A diverged pass usually
        corrupts a span, not the whole stem; zeroing is the least-destructive
        repair and cannot clip or distort the healthy remainder.
    Rejected with :class:`DegenerateEngineOutput` (a ``ValueError`` subclass;
    the caller's failure path reports it, the stem is never cached, and the
    engine retries once with a fresh seed before giving up):
      * all-non-finite audio — nothing salvageable;
      * constant-DC tone / near-total silence (a degenerate decode or an
        empty token stream) — no repair recovers speech from it.

    ``text`` and ``sample_rate`` only enrich the error messages.
    """
    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        raise DegenerateEngineOutput(_empty_error(text))
    finite = np.isfinite(array)
    if not finite.all():
        if not finite.any():
            raise DegenerateEngineOutput(
                _degenerate_error(text, sample_rate, "every sample is NaN or infinite")
            )
        array = np.where(finite, array, np.float32(0.0))
    centered = array - np.mean(array, dtype=np.float64)
    peak = float(np.max(np.abs(centered)))
    if peak < _MIN_PEAK_AMPLITUDE:
        # A perfectly flat line: DC tone (constant offset decode) or pure
        # silence. The distinction does not change the remedy — neither
        # carries speech — so one message covers both shapes.
        raise DegenerateEngineOutput(
            _degenerate_error(text, sample_rate, "the stem is a constant DC tone with no speech content")
        )
    zero_duty = float(np.count_nonzero(np.abs(centered) < _MIN_PEAK_AMPLITUDE) / array.size)
    if zero_duty > _MAX_ZERO_DUTY_CYCLE:
        raise DegenerateEngineOutput(
            _degenerate_error(text, sample_rate, "the stem is silence with no speech content")
        )
    return array


def _empty_error(text: str) -> str:
    preview = _text_preview(text)
    return f"TTS engine returned no audio for the utterance{preview}. "


def _degenerate_error(text: str, sample_rate: int | None, detail: str) -> str:
    preview = _text_preview(text)
    hint = f" at sample rate {sample_rate}" if sample_rate else ""
    return (
        f"TTS engine produced degenerate audio ({detail}){hint} for the utterance{preview}. "
        "The stem was not cached; re-render the segment (a different seed or temperature often clears it)."
    )


def _text_preview(text: str) -> str:
    stripped = " ".join((text or "").split())
    if not stripped:
        return ""
    preview = stripped[:60]
    return f" ({preview!r}{'…' if len(stripped) > 60 else ''})"


def stem_is_speech_like(
    audio: np.ndarray, *, allow_silence: bool = False, sample_rate: int | None = None
) -> bool:
    """Read-side counterpart of :func:`sanitize_engine_audio`.

    Classifies cached stem content with the same taxonomy the write-side gate
    enforces for engine output, minus the in-place repair: an entry is
    servable for spoken text only when it is finite, non-empty, and carries
    speech-shaped content (peak above the degenerate threshold, zero-duty
    cycle below the silence ceiling).

    ``allow_silence=True`` exempts pause-only stems, which are legitimately
    all-zero by construction (``_write_pause_only_stem``) — silence is only
    ever accepted where silence is the expected product.

    Honest scope note: this gate defends every entry written AFTER the
    hardening (2026-09-20) and any entry a future engine hiccup might produce.
    Stems cached before the hardening that already contain degenerate audio
    are not distinguishable from pause stems by content alone, so they are
    still servable until the utterance is re-rendered; cache keys include the
    text, seed, and engine parameters, so any edit to an affected utterance
    re-synthesizes from scratch.
    """
    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        return False
    if not np.isfinite(array).all():
        return False
    if allow_silence:
        return True
    centered = array - np.mean(array, dtype=np.float64)
    peak = float(np.max(np.abs(centered)))
    if peak < _MIN_PEAK_AMPLITUDE:
        return False
    zero_duty = float(np.count_nonzero(np.abs(centered) < _MIN_PEAK_AMPLITUDE) / array.size)
    return zero_duty <= _MAX_ZERO_DUTY_CYCLE


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
