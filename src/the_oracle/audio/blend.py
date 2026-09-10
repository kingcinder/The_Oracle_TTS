"""Voice blending: derive ONE conditioning reference from two clips.

Chatterbox conditions each speaker on a single reference payload, so
combining two voices means producing a single blended reference WAV that the
engine then treats like any other reference clip. The blend is
deterministic and cached by an input hash, so a given
(A, B, weight, mode) always yields the same derived clip and therefore the
same synthesized voice across renders.

``blend_weight`` is the proportion (0..1) of the FIRST clip, which is how
the user "preferentially selects" which voice's qualities dominate. Modes:

- ``mix``       — time-aligned weighted mix over the overlap; the tail of
                  the longer clip carries through at its own weight, so a
                  long anchor phrase keeps its ending. Loudness is
                  normalized so a 50/50 mix is not quieter than either
                  source.
- ``alternate`` — trim both clips and play them in sequence (A first when
                  weight >= 0.5, else B first) with a short gap: a blend
                  that literally alternates between the two voices.
- ``layer``     — A leads at full level while B sits underneath at
                  ``(1 - weight)`` gain: a duet/texture voice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from the_oracle.audio.assemble import normalize_loudness
from the_oracle.models.cache import atomic_write
from the_oracle.utils.audio import ensure_mono, resample_audio, trim_silence
from the_oracle.utils.hashing import hash_file, hash_payload

BLEND_MODES = ("mix", "alternate", "layer")
_BLEND_TARGET_SR = 24000
_ALTERNATE_GAP_SECONDS = 0.25


def _load_trimmed(path: Path) -> np.ndarray:
    """Load, mono-ize, silence-trim, and resample one clip to 24 kHz."""
    audio, sample_rate = sf.read(str(path), dtype="float32")
    mono = ensure_mono(audio)
    trimmed = trim_silence(mono)
    return np.asarray(resample_audio(trimmed, sample_rate, _BLEND_TARGET_SR), dtype=np.float32)


def blend_references(
    path_a: str | Path,
    path_b: str | Path,
    weight_a: float = 0.5,
    mode: str = "mix",
    out_dir: str | Path | None = None,
) -> Path:
    """Blend two reference clips into one deterministic derived reference.

    ``weight_a`` is clamped to 0..1 (1.0 = pure A, 0.0 = pure B). The result
    is cached at ``out_dir/<hash>.wav`` (default: ``<dir of A>/.blends``),
    keyed by the input files' content hashes plus weight and mode, so the
    derived clip is byte-identical across calls with the same inputs.
    """
    source_a = Path(path_a).expanduser()
    source_b = Path(path_b).expanduser()
    for source in (source_a, source_b):
        if not source.exists():
            raise FileNotFoundError(
                f"Voice blend reference does not exist: {source}. "
                "Both blend clips must be readable WAV/FLAC/MP3 files."
            )
    weight = float(max(0.0, min(1.0, weight_a)))
    if mode not in BLEND_MODES:
        raise ValueError(f"Unsupported blend mode {mode!r}; choose from {BLEND_MODES}.")

    destination_dir = Path(out_dir).expanduser() if out_dir is not None else source_a.parent / ".blends"
    destination_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hash_payload(
        {
            "a": hash_file(source_a),
            "b": hash_file(source_b),
            "weight": round(weight, 3),
            "mode": mode,
        }
    )[:16]
    destination = destination_dir / f"blend_{cache_key}_{mode}_{int(round(weight * 100))}.wav"
    if destination.exists():
        return destination

    audio_a = _load_trimmed(source_a)
    audio_b = _load_trimmed(source_b)
    if audio_a.size == 0 or audio_b.size == 0:
        raise ValueError("Both voice blend clips must contain audio after silence trimming.")

    if mode == "mix":
        length = max(audio_a.size, audio_b.size)
        blended = np.zeros(length, dtype=np.float32)
        blended[: audio_a.size] += audio_a * weight
        blended[: audio_b.size] += audio_b * (1.0 - weight)
    elif mode == "alternate":
        first, second = (audio_a, audio_b) if weight >= 0.5 else (audio_b, audio_a)
        gap = np.zeros(int(_BLEND_TARGET_SR * _ALTERNATE_GAP_SECONDS), dtype=np.float32)
        blended = np.concatenate([first, gap, second]).astype(np.float32)
    else:  # layer
        length = max(audio_a.size, audio_b.size)
        blended = np.zeros(length, dtype=np.float32)
        blended[: audio_a.size] += audio_a
        blended[: audio_b.size] += audio_b * (1.0 - weight)

    blended = normalize_loudness(np.clip(blended, -1.0, 1.0), preset="light")
    # Atomic write: parallel renders share the blend cache, so a plain
    # sf.write could leave a torn WAV for the racing reader.
    atomic_write(destination, lambda tmp: sf.write(str(tmp), blended, _BLEND_TARGET_SR, format="WAV"))
    return destination