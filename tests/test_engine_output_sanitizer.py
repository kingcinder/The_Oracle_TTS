"""Tests for the engine-boundary TTS output sanitizer.

``sanitize_engine_audio`` is the single hiccup gate both engines return
through (Chatterbox PyTorch and the audio.cpp Vulkan backend). It repairs the
repairable (NaN/Inf spans) and rejects the degenerate (constant DC tone,
near-total silence, empty output) with ``ValueError`` so the caller's failure
path reports the problem and the stem cache never stores a defective stem.
Healthy audio must pass through bit-identically — the gate must never touch a
good render.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest

# --- pipeline-adjacent imports need huggingface_hub stubbed (same pattern as
# tests/test_review_fixes_audio.py) so the engine modules import without the
# real hub installed.
_hub = types.ModuleType("huggingface_hub")
_hub.snapshot_download = lambda *a, **k: ""  # noqa: E731
_hub_errors = types.ModuleType("huggingface_hub.errors")
_hub_errors.LocalEntryNotFoundError = type("LocalEntryNotFoundError", (Exception,), {})
_hub_utils = types.ModuleType("huggingface_hub.utils")
_hub_utils.LocalTokenNotFoundError = type("LocalTokenNotFoundError", (Exception,), {})
_hub.errors = _hub_errors
_hub.utils = _hub_utils
sys.modules.setdefault("huggingface_hub", _hub)
sys.modules.setdefault("huggingface_hub.errors", _hub_errors)
sys.modules.setdefault("huggingface_hub.utils", _hub_utils)

from the_oracle.utils.audio import sanitize_engine_audio  # noqa: E402

SAMPLE_RATE = 24000


def _speech(seconds: float = 1.0, freq: float = 220.0, peak: float = 0.4) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (peak * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pass-through: healthy audio is never touched
# ---------------------------------------------------------------------------


def test_healthy_speech_passes_through_byte_identical() -> None:
    audio = _speech()
    out = sanitize_engine_audio(audio, text="A normal line.", sample_rate=SAMPLE_RATE)
    np.testing.assert_array_equal(out, audio)


def test_quiet_but_voiced_audio_passes() -> None:
    # A whispered line is quiet but still shaped like speech: well above the
    # empty threshold and far below the zero-duty degeneracy threshold.
    audio = _speech(peak=0.005)
    out = sanitize_engine_audio(audio, sample_rate=SAMPLE_RATE)
    np.testing.assert_array_equal(out, audio)


def test_normal_leading_trailing_silence_passes() -> None:
    # Real stems have silence around speech; only *near-total* silence is a
    # degenerate decode. 20% speech / 80% silence is ordinary.
    silence = np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)
    audio = np.concatenate([silence, _speech(0.1), silence])
    out = sanitize_engine_audio(audio, sample_rate=SAMPLE_RATE)
    np.testing.assert_array_equal(out, audio)


def test_int16_sourced_audio_passes() -> None:
    # WAV read through soundfile as float32 from a quiet int16 recording can
    # carry a visible DC offset; the gate measures AC peak after centering.
    audio = _speech(peak=0.02) + 0.01
    out = sanitize_engine_audio(audio, sample_rate=SAMPLE_RATE)
    np.testing.assert_array_equal(out, audio)


# ---------------------------------------------------------------------------
# Repair: non-finite spans are zeroed, healthy remainder survives
# ---------------------------------------------------------------------------


def test_nan_span_is_zeroed_and_rest_survives() -> None:
    audio = _speech()
    corrupted = audio.copy()
    corrupted[100:150] = np.nan
    out = sanitize_engine_audio(corrupted, text="hello", sample_rate=SAMPLE_RATE)
    assert np.isfinite(out).all()
    assert np.count_nonzero(out[100:150]) == 0
    np.testing.assert_array_equal(out[:100], audio[:100])
    np.testing.assert_array_equal(out[150:], audio[150:])


def test_inf_samples_are_zeroed() -> None:
    audio = _speech()
    corrupted = audio.copy()
    corrupted[10] = np.inf
    corrupted[20] = -np.inf
    out = sanitize_engine_audio(corrupted, sample_rate=SAMPLE_RATE)
    assert np.isfinite(out).all()
    assert out[10] == 0.0 and out[20] == 0.0


def test_all_nonfinite_is_rejected() -> None:
    with pytest.raises(ValueError, match="every sample is NaN or infinite"):
        sanitize_engine_audio(np.full(4800, np.nan, dtype=np.float32), text="hello")


# ---------------------------------------------------------------------------
# Reject: degenerate generation modes
# ---------------------------------------------------------------------------


def test_constant_dc_tone_is_rejected() -> None:
    # The classic degenerate decode: a flat line at some non-zero offset.
    flat = np.full(int(1.0 * SAMPLE_RATE), 0.25, dtype=np.float32)
    with pytest.raises(ValueError, match="constant DC tone"):
        sanitize_engine_audio(flat, text="hello", sample_rate=SAMPLE_RATE)


def test_pure_silence_is_rejected() -> None:
    # Flat silence lands in the DC-tone bucket (same remedy, one message).
    with pytest.raises(ValueError, match="no speech content"):
        sanitize_engine_audio(np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32), text="hello")


def test_truncated_generation_is_rejected() -> None:
    # Truncation right after SOT: a lone click inside an otherwise silent
    # stem. >99% of samples are (near-)zero after centering.
    audio = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    audio[1000] = 0.5
    with pytest.raises(ValueError, match="silence with no speech content"):
        sanitize_engine_audio(audio, text="hello", sample_rate=SAMPLE_RATE)


def test_empty_output_is_rejected() -> None:
    with pytest.raises(ValueError, match="returned no audio"):
        sanitize_engine_audio(np.zeros(0, dtype=np.float32), text="hello")


def test_rejection_messages_carry_the_text_preview() -> None:
    with pytest.raises(ValueError, match="Hello there") as excinfo:
        sanitize_engine_audio(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            text="Hello there, general  Kenobi. " * 3,
            sample_rate=SAMPLE_RATE,
        )
    assert "not cached" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Engine wiring: both engines gate their synthesize() return through it
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, audio):
        self.conds = None
        self._audio = audio

    def generate(self, **_kwargs):
        return self._audio


def _engine_with(audio, *, variant: str = "standard"):
    from the_oracle.tts_engines.chatterbox_engine import ChatterboxEngine

    engine = object.__new__(ChatterboxEngine)
    engine.variant = variant
    engine.device = "cpu"
    engine.seed = None
    class _Cond:
        def to(self, _device):
            return self

    engine._condition_cls = type("_C", (), {"load": classmethod(lambda cls, p, **k: _Cond())})
    engine._loaded_conditioning = {}
    engine._model = _FakeModel(audio)
    import threading

    engine._synthesize_lock = threading.Lock()
    return engine


def _conditioning():
    from the_oracle.tts_engines.chatterbox_engine import ChatterboxConditioning

    return ChatterboxConditioning("id", Path("cond.pt"), "hash", "A", "standard")


def test_chatterbox_engine_rejects_degenerate_generate_output(monkeypatch) -> None:
    engine = _engine_with(np.zeros((1, 4800), dtype=np.float32))
    with pytest.raises(ValueError, match="degenerate audio"):
        engine.synthesize("Hello.", _conditioning(), _voice_settings())


def test_chatterbox_engine_repairs_nan_spans_from_generate(monkeypatch) -> None:
    good = _speech(0.5)
    audio = good.copy()
    audio[200:220] = np.nan
    engine = _engine_with(audio.reshape(1, -1))
    out = engine.synthesize("Hello.", _conditioning(), _voice_settings())
    assert np.isfinite(out).all()


def test_chatterbox_engine_still_strips_pain_point_markers() -> None:
    # The sanitizer sits AFTER the existing marker strip in the boundary; the
    # long-standing annotation behavior must be untouched.
    seen: dict[str, str] = {}

    class _SpyModel:
        conds = None

        def generate(self, **kwargs):
            seen["text"] = kwargs["text"]
            return _speech(0.2).reshape(1, -1)

    engine = _engine_with(None)
    engine._model = _SpyModel()
    engine.synthesize("syncronized~lockstep", _conditioning(), _voice_settings())
    assert seen["text"] == "syncronized lockstep"


def _voice_settings():
    from the_oracle.models.project import VoiceSettings

    return VoiceSettings()


def _vulkan_engine(tmp_path: Path, wav_bytes: bytes, sample_rate: int = SAMPLE_RATE):
    import soundfile as sf

    from the_oracle.tts_engines.vulkan_backend import AudioCppVulkanEngine

    engine = AudioCppVulkanEngine.__new__(AudioCppVulkanEngine)
    engine.variant = "standard"
    engine.device = "vulkan"
    engine._timeout_override = 30.0
    engine._batch_limit_override = 32
    engine._binary_override = None
    engine._model_override = None
    engine._device_index_override = None
    engine._threads_override = None
    engine._seed_override = None
    engine._binary = None
    engine._model = None
    engine._last_sample_rate = None
    reference = tmp_path / "ref.wav"
    sf.write(str(reference), _speech(0.5), sample_rate, format="WAV")
    return engine, reference


def test_vulkan_engine_rejects_silent_wav(tmp_path, monkeypatch) -> None:
    import soundfile as sf
    from the_oracle.tts_engines.vulkan_backend import AudioCppVulkanEngine, VulkanConditioning

    # The gate under test runs BEFORE _run_command (the stubbed boundary),
    # but constructing the engine's _build_command resolves self.binary /
    # self.model first, which fails on a machine without the gitignored
    # audio.cpp build — skip like test_vulkan_backend's hardware smoke does.
    engine, reference = _vulkan_engine(tmp_path, b"")
    if engine.binary is None or engine.model is None:
        pytest.skip("audiocpp_cli binary/model not built; the Vulkan engine-gate tests need the audio.cpp build")
    with pytest.raises(ValueError, match="degenerate audio"):
        _run_vulkan_synthesize_with_wav(tmp_path, engine, reference, _silent_wav(tmp_path), monkeypatch)


def _silent_wav(tmp_path: Path) -> Path:
    import soundfile as sf

    silent = tmp_path / "silent.wav"
    sf.write(str(silent), np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, format="WAV")
    return silent


def _run_vulkan_synthesize_with_wav(
    tmp_path: Path, engine, reference: Path, wav: Path, monkeypatch
):
    """Drive AudioCppVulkanEngine.synthesize against a prepared output wav.

    _build_command emits the full audio.cpp invocation ending in --out; the
    fake _run_command rewrites the model/voice-ref/out paths to test stubs and
    copies the prepared wav into the temp path the engine's own existence
    check looks at.
    """
    import shutil
    import subprocess as _sp
    from the_oracle.tts_engines.vulkan_backend import AudioCppVulkanEngine, VulkanConditioning

    def fake_run(self, command, *, timeout=None):
        command[command.index("--model") + 1] = str(engine._model)
        command[command.index("--voice-ref") + 1] = str(reference)
        out_wav = Path(command[command.index("--out") + 1])
        command[command.index("--out") + 1] = str(out_wav)
        shutil.copy2(wav, out_wav)
        return _sp.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(AudioCppVulkanEngine, "_run_command", fake_run)
    conditioning = VulkanConditioning(cache_id="c", reference_path=reference, speaker="A")
    return engine.synthesize("Hello.", conditioning, _voice_settings())


def test_vulkan_engine_passes_healthy_wav_through(tmp_path, monkeypatch) -> None:
    import soundfile as sf

    engine, reference = _vulkan_engine(tmp_path, b"")
    if engine.binary is None or engine.model is None:
        pytest.skip("audiocpp_cli binary/model not built; the Vulkan engine-gate tests need the audio.cpp build")
    good = tmp_path / "good.wav"
    sf.write(str(good), _speech(0.5), SAMPLE_RATE, format="WAV")
    out = _run_vulkan_synthesize_with_wav(tmp_path, engine, reference, good, monkeypatch)
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) > 0.1


def test_cached_reference_type_hides_unused_import() -> None:
    # Silence linters about the conditional import in _vulkan_engine.
    from the_oracle.models.cache import CachedReference  # noqa: F401
