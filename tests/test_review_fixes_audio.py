"""Focused regression tests for the core audio pipeline review fixes.

Covers (all without torch, Qt, model downloads, hardware, or cloud):
1.  Shared crossfade overlap calculation (allocation == fill, clamped to real
    stem lengths) and honest crossfade diagnostics.
2.  Seed folded into stem cache keys.
3.  Corrupt cached stems are deleted and re-synthesized, never crashed on.
4.  Empty / directive-only utterances bank a silent stem so the turn's pause
    survives instead of the row being dropped.
5.  Hyphen chunking does not insert spaces into compound words.
6.  Individual words over the chunk limit are hard-split.
7.  Both resamplers tolerate empty audio.
8.  FLAC tagging failure warns but does not fail the export.
9.  Incremental cache diagnostics compare real hashes (empty != empty).
10. Crossfade diagnostics report the actually applied window.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from the_oracle.audio import assemble as assemble_mod
from the_oracle.audio import export_flac as flac_mod
from the_oracle.audio.assemble import AudioSegment, _crossfade_overlap, assemble_dialogue
from the_oracle.models.cache import ProjectCache, _resample_linear, build_chunk_cache_key
from the_oracle.utils import audio as audio_util_mod
from the_oracle.utils.audio import resample_audio
from the_oracle.utils.chunking import MAX_CHUNK_SIZE, chunk_utterance
from the_oracle.utils.hashing import build_chunk_hash

# --- pipeline imports need huggingface_hub stubbed (heavy dep, absent here) ---
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

from the_oracle.models.project import VoiceSettings  # noqa: E402
from the_oracle.pipeline import (  # noqa: E402
    SynthesisTask,
    _load_cached_stem,
    _write_pause_only_stem,
    compute_incremental_changes,
    synthesize_task,
)

SAMPLE_RATE = 24000


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, format="WAV")
    return path


def _sine(seconds: float, freq: float = 440.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


class _FakeEngine:
    """Minimal engine double for synthesize_task tests."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.engine_version = "test-engine-1"
        self.synthesize_calls = 0

    def synthesize(self, text, conditioning, voice_settings):
        self.synthesize_calls += 1
        return _sine(0.4, freq=440.0, sample_rate=self.sample_rate)


class _ExplodingEngine(_FakeEngine):
    def synthesize(self, text, conditioning, voice_settings):
        raise AssertionError("engine.synthesize must not be called for pause-only text")


def _make_task(text: str, **overrides) -> SynthesisTask:
    params = dict(
        utterance_index=1,
        source_index=0,
        speaker="narrator",
        text=text,
        reference_audio_hash="refhash",
        reference_path=Path("/tmp/ref.wav"),
        voice_settings=VoiceSettings(),
        model_variant="standard",
        device_mode="cpu",
        export_stems=False,
        inference_backend="pytorch",
    )
    params.update(overrides)
    return SynthesisTask(**params)


# ---------------------------------------------------------------------------
# 1. Shared crossfade overlap
# ---------------------------------------------------------------------------


class TestCrossfadeOverlap:
    def test_full_overlap_when_both_stems_long_enough(self):
        assert _crossfade_overlap(1000, 1000, 240) == 240

    def test_clamped_to_shorter_stem(self):
        assert _crossfade_overlap(100, 1000, 240) == 100
        assert _crossfade_overlap(1000, 100, 240) == 100

    def test_zero_or_negative_request_disables(self):
        assert _crossfade_overlap(1000, 1000, 0) == 0
        assert _crossfade_overlap(1000, 1000, -5) == 0

    def test_empty_stem_gives_no_overlap(self):
        assert _crossfade_overlap(0, 1000, 240) == 0


def _assemble_two(tmp_path: Path, len_a: int, len_b: int, crossfade_ms: int,
                  pause_a_ms: int = 0, freq_b: float = 660.0):
    stem_a = _write_wav(tmp_path / "a.wav", _sine(len_a / SAMPLE_RATE, 440.0))
    stem_b = _write_wav(tmp_path / "b.wav", _sine(len_b / SAMPLE_RATE, freq_b))
    segments = [
        AudioSegment(path=str(stem_a), sample_rate=SAMPLE_RATE, pause_after_ms=pause_a_ms,
                     duration_seconds=len_a / SAMPLE_RATE, segment_index=0, speaker="a"),
        AudioSegment(path=str(stem_b), sample_rate=SAMPLE_RATE, pause_after_ms=0,
                     duration_seconds=len_b / SAMPLE_RATE, segment_index=1, speaker="b"),
    ]
    diagnostics: dict = {}
    out, rate = assemble_dialogue(segments, crossfade_ms=crossfade_ms,
                                  loudness_preset="off", diagnostics=diagnostics)
    return out, rate, diagnostics


class TestAssembleCrossfade:
    def test_output_length_uses_applied_overlap(self, tmp_path: Path):
        len_a, len_b = SAMPLE_RATE, SAMPLE_RATE  # 1s each
        crossfade_ms = 20
        requested = int(SAMPLE_RATE * crossfade_ms / 1000)
        out, rate, _ = _assemble_two(tmp_path, len_a, len_b, crossfade_ms)
        assert rate == SAMPLE_RATE
        assert len(out) == len_a + len_b - requested

    def test_short_stem_clamps_overlap_no_overrun(self, tmp_path: Path):
        """A stem shorter than the requested crossfade clamps the overlap to
        the real content length instead of reading past the buffer."""
        len_a, len_b = 100, SAMPLE_RATE  # stem A is 100 samples
        crossfade_ms = 20  # requests 480 samples
        out, _, diagnostics = _assemble_two(tmp_path, len_a, len_b, crossfade_ms)
        applied = _crossfade_overlap(len_a, len_b, int(SAMPLE_RATE * crossfade_ms / 1000))
        assert applied == 100
        assert len(out) == len_a + len_b - applied
        join = diagnostics["joins"][0]
        assert join["crossfade_applied_seconds"] == pytest.approx(round(applied / SAMPLE_RATE, 6))

    def test_pause_zeros_survive_crossfade(self, tmp_path: Path):
        """The crossfade mixes stem tails only; the previous segment's pause
        region stays true silence."""
        len_a, len_b = SAMPLE_RATE, SAMPLE_RATE
        pause_ms = 300
        pause_samples = int(SAMPLE_RATE * pause_ms / 1000)
        out, _, diagnostics = _assemble_two(tmp_path, len_a, len_b, 20, pause_a_ms=pause_ms)
        seg0 = diagnostics["segments"][0]
        pause_start = int(seg0["content_end_seconds"] * SAMPLE_RATE)
        pause_region = out[pause_start:pause_start + pause_samples]
        assert len(pause_region) == pause_samples
        assert np.all(pause_region == 0.0)

    def test_crossfade_actually_mixes_tails(self, tmp_path: Path):
        """The overlap region is a blend of both stems, not a hard cut."""
        len_a, len_b = SAMPLE_RATE, SAMPLE_RATE
        crossfade_ms = 20
        overlap = int(SAMPLE_RATE * crossfade_ms / 1000)
        out, _, _ = _assemble_two(tmp_path, len_a, len_b, crossfade_ms)
        region = out[len_a - overlap:len_a]
        # Pure stem A tail would equal the sine; the blend deviates from both
        # pure stems at the edges of the ramp.
        assert not np.allclose(region[:10], _sine(len_a / SAMPLE_RATE)[len_a - overlap:len_a - overlap + 10])
        assert np.any(region != 0.0)

    def test_diagnostics_report_applied_not_requested(self, tmp_path: Path):
        len_a, len_b = 100, SAMPLE_RATE
        out, _, diagnostics = _assemble_two(tmp_path, len_a, len_b, 20)
        requested_s = 20 / 1000
        applied_s = diagnostics["segments"][1]["crossfade_applied_seconds"]
        join_applied_s = diagnostics["joins"][0]["crossfade_applied_seconds"]
        assert applied_s == pytest.approx(round(100 / SAMPLE_RATE, 6))
        assert join_applied_s == pytest.approx(round(100 / SAMPLE_RATE, 6))
        assert applied_s < requested_s
        assert diagnostics["segments"][1]["crossfade_requested_ms"] == 20


# ---------------------------------------------------------------------------
# 2. Seed in stem cache keys
# ---------------------------------------------------------------------------


class TestSeedInCacheKeys:
    def _kwargs(self, **overrides):
        kw = dict(
            speaker="narrator",
            repaired_text="hello world",
            engine_key="render",
            engine_params={"temperature": 0.8},
            engine_version="v1",
            reference_audio_hash="refhash",
        )
        kw.update(overrides)
        return kw

    def test_different_seeds_different_hashes(self):
        assert build_chunk_hash(**self._kwargs(seed=1)) != build_chunk_hash(**self._kwargs(seed=2))

    def test_seed_none_differs_from_seed_value(self):
        assert build_chunk_hash(**self._kwargs(seed=None)) != build_chunk_hash(**self._kwargs(seed=0))

    def test_same_seed_stable(self):
        assert build_chunk_hash(**self._kwargs(seed=42)) == build_chunk_hash(**self._kwargs(seed=42))

    def test_cache_key_builder_forwards_seed(self):
        k1 = build_chunk_cache_key(
            speaker="n", repaired_text="hi", engine_name="e", engine_version="v",
            engine_params={}, reference_audio_hash="r", seed=7,
        )
        k2 = build_chunk_cache_key(
            speaker="n", repaired_text="hi", engine_name="e", engine_version="v",
            engine_params={}, reference_audio_hash="r", seed=8,
        )
        assert k1 != k2


# ---------------------------------------------------------------------------
# 3. Corrupt cached stems
# ---------------------------------------------------------------------------


class TestCorruptCachedStems:
    def test_load_valid_stem(self, tmp_path: Path):
        p = _write_wav(tmp_path / "ok.wav", _sine(0.2))
        loaded = _load_cached_stem(p)
        assert loaded is not None
        audio, rate = loaded
        assert rate == SAMPLE_RATE and len(audio) == int(0.2 * SAMPLE_RATE)
        assert p.exists()

    def test_load_corrupt_stem_deletes_and_returns_none(self, tmp_path: Path):
        p = tmp_path / "corrupt.wav"
        p.write_bytes(b"this is not a wav file, just garbage bytes" * 10)
        assert _load_cached_stem(p) is None
        assert not p.exists()

    def test_synthesize_task_recovers_from_corrupt_cache(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _FakeEngine()
        task = _make_task("hello there")
        # Pre-compute the hash the task will use, then plant garbage there.
        from the_oracle.utils.hashing import build_chunk_hash as bch
        from the_oracle.pipeline import _chunk_engine_key

        chunk_hash = bch(
            speaker=task.speaker, repaired_text=task.text,
            engine_key=_chunk_engine_key(task.inference_backend, task.model_variant),
            engine_params=task.voice_settings.to_dict(), engine_version=engine.engine_version,
            reference_audio_hash=task.reference_audio_hash, seed=task.seed,
        )
        stem_path = cache.stem_path(chunk_hash)
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        stem_path.write_bytes(b"corrupt garbage" * 100)

        result = synthesize_task(task, engine, conditioning=None, project_cache=cache)
        assert engine.synthesize_calls == 1  # re-synthesized, not crashed on
        assert result.cache_hit is False
        assert result.duration_seconds == pytest.approx(0.4, abs=0.01)
        # The stem on disk is now a valid wav the next render can trust.
        assert _load_cached_stem(stem_path) is not None

    def test_synthesize_task_cache_hit_skips_engine(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _FakeEngine()
        task = _make_task("hello there")
        first = synthesize_task(task, engine, conditioning=None, project_cache=cache)
        assert first.cache_hit is False
        second = synthesize_task(task, engine, conditioning=None, project_cache=cache)
        assert second.cache_hit is True
        assert engine.synthesize_calls == 1
        assert second.duration_seconds == pytest.approx(first.duration_seconds)


# ---------------------------------------------------------------------------
# 4. Empty / directive-only utterances keep their pause
# ---------------------------------------------------------------------------


class TestPauseOnlyUtterances:
    def test_write_pause_only_stem_is_half_second_silence(self, tmp_path: Path):
        p = tmp_path / "pause.wav"
        _write_pause_only_stem(p, SAMPLE_RATE)
        audio, rate = sf.read(str(p))
        assert rate == SAMPLE_RATE
        assert len(audio) == int(0.5 * SAMPLE_RATE)
        assert np.all(np.asarray(audio) == 0.0)

    def test_empty_text_never_touches_engine(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _ExplodingEngine()
        result = synthesize_task(_make_task(""), engine, conditioning=None, project_cache=cache)
        assert result.duration_seconds == pytest.approx(0.5, abs=0.01)
        assert result.cache_hit is False

    def test_whitespace_only_text_is_pause_only(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _ExplodingEngine()
        result = synthesize_task(_make_task("   \n\t  "), engine, conditioning=None, project_cache=cache)
        assert result.duration_seconds == pytest.approx(0.5, abs=0.01)

    def test_pause_only_stem_is_cached(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _ExplodingEngine()
        task = _make_task("")
        first = synthesize_task(task, engine, conditioning=None, project_cache=cache)
        second = synthesize_task(task, engine, conditioning=None, project_cache=cache)
        assert first.cache_hit is False and second.cache_hit is True


# ---------------------------------------------------------------------------
# 5 & 6. Chunking: hyphens and hard-split words
# ---------------------------------------------------------------------------


class TestChunkingFixes:
    def test_hyphenated_compound_survives_without_spaces(self):
        text = ("This well-known fact about state-of-the-art systems is important, "
                "and it keeps going; truly it does.")
        chunks = chunk_utterance(text, 0)
        joined = " ".join(c.text for c in chunks)
        assert "well-known" in joined
        assert "state-of-the-art" in joined
        assert "well - known" not in joined
        assert "well- known" not in joined

    def test_bare_hyphen_delimiter_still_splits(self):
        # A hyphen surrounded by spaces is still a clause boundary.
        text = "first part - second part " * 40
        chunks = chunk_utterance(text, 0, max_size=60)
        assert len(chunks) > 1

    def test_single_word_over_limit_is_hard_split(self):
        word = "x" * (MAX_CHUNK_SIZE + 37)
        chunks = chunk_utterance(word, 0)
        assert len(chunks) >= 2
        assert all(len(c.text) <= MAX_CHUNK_SIZE for c in chunks)
        assert "".join(c.text for c in chunks) == word

    def test_word_exactly_at_limit_not_split(self):
        word = "y" * MAX_CHUNK_SIZE
        chunks = chunk_utterance(word, 0)
        assert "".join(c.text for c in chunks) == word
        assert all(len(c.text) <= MAX_CHUNK_SIZE for c in chunks)


# ---------------------------------------------------------------------------
# 7. Empty-audio resamplers
# ---------------------------------------------------------------------------


class TestEmptyResamplers:
    def test_resample_audio_empty_mono(self):
        out = resample_audio(np.zeros((0,), dtype=np.float32), 44100, 48000)
        assert out.shape == (0,)
        assert out.dtype == np.float32

    def test_resample_audio_empty_stereo(self):
        out = resample_audio(np.zeros((0, 2), dtype=np.float32), 44100, 48000)
        assert out.shape == (0, 2)
        assert out.dtype == np.float32

    def test_resample_audio_empty_same_rate(self):
        out = resample_audio(np.zeros((0,), dtype=np.float64), 48000, 48000)
        assert out.shape == (0,)
        assert out.dtype == np.float32

    def test_cache_resample_linear_empty(self):
        out = _resample_linear(np.zeros((0,), dtype=np.float32), 44100, 48000)
        assert out.shape == (0,)
        assert out.dtype == np.float32

    def test_resample_audio_still_works_on_real_audio(self):
        src = _sine(0.1, sample_rate=44100)
        out = resample_audio(src, 44100, 22050)
        assert len(out) == pytest.approx(len(src) / 2, abs=2)


# ---------------------------------------------------------------------------
# 8. FLAC tagging failure
# ---------------------------------------------------------------------------


class TestFlacExport:
    def test_tagging_failure_warns_but_keeps_export(self, tmp_path: Path, caplog):
        def _boom(path, metadata):
            raise RuntimeError("mutagen exploded")

        monkeypatched = pytest.MonkeyPatch()
        monkeypatched.setattr(flac_mod, "_tag_with_mutagen", _boom)
        try:
            with caplog.at_level(logging.WARNING, logger=flac_mod.LOGGER.name):
                dest = flac_mod.write_flac(tmp_path / "out.flac", _sine(0.2), SAMPLE_RATE,
                                           {"title": "test"})
        finally:
            monkeypatched.undo()
        assert dest.exists()
        assert dest.stat().st_size > 0
        assert any("tagging failed" in r.message.lower() for r in caplog.records)

    def test_audio_write_failure_still_falls_back(self, tmp_path: Path):
        # If the SoundFile write itself fails, the ffmpeg fallback path is used.
        calls = {}

        def _fail_write(*a, **k):
            raise RuntimeError("no flac encoder")

        def _fake_ffmpeg(destination, audio, sample_rate, metadata):
            calls["ffmpeg"] = True
            Path(destination).write_bytes(b"fake-flac-bytes")

        monkeypatched = pytest.MonkeyPatch()
        monkeypatched.setattr(sf, "write", _fail_write)
        monkeypatched.setattr(flac_mod, "_ffmpeg_write", _fake_ffmpeg)
        try:
            dest = flac_mod.write_flac(tmp_path / "out.flac", _sine(0.2), SAMPLE_RATE, {})
        finally:
            monkeypatched.undo()
        assert calls.get("ffmpeg") is True
        assert dest.exists()


# ---------------------------------------------------------------------------
# 9. Incremental diagnostics compare real hashes
# ---------------------------------------------------------------------------


class TestIncrementalDiagnostics:
    def test_identical_real_hashes_not_changed(self):
        old = [{"index": 0, "chunk_hash": "abc"}, {"index": 1, "chunk_hash": "def"}]
        new = [{"index": 0, "chunk_hash": "abc"}, {"index": 1, "chunk_hash": "def"}]
        assert compute_incremental_changes(old, new) == []

    def test_changed_hash_flagged(self):
        old = [{"index": 0, "chunk_hash": "abc"}]
        new = [{"index": 0, "chunk_hash": "xyz"}]
        assert compute_incremental_changes(old, new) == [0]

    def test_two_missing_hashes_never_compare_equal(self):
        """The old code compared '' != '' as False and reported cache reuse
        vacuously when hashes were never populated."""
        old = [{"index": 0, "chunk_hash": ""}, {"index": 1}]
        new = [{"index": 0, "chunk_hash": ""}, {"index": 1}]
        changed = compute_incremental_changes(old, new)
        assert 0 in changed and 1 in changed

    def test_new_row_without_old_hash_flagged(self):
        old = []
        new = [{"index": 0, "chunk_hash": "abc"}]
        assert compute_incremental_changes(old, new) == [0]

    def test_cache_key_fallback_still_works(self):
        old = [{"index": 0, "cache_key": "k1"}]
        new = [{"index": 0, "cache_key": "k1"}]
        assert compute_incremental_changes(old, new) == []
        new2 = [{"index": 0, "cache_key": "k2"}]
        assert compute_incremental_changes(old, new2) == [0]


# ---------------------------------------------------------------------------
# 2b. Seed flows into synthesize_task's hash
# ---------------------------------------------------------------------------


class TestSeededSynthesisTask:
    def test_different_seeds_use_different_cache_entries(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _FakeEngine()
        t1 = _make_task("same text", seed=1)
        t2 = _make_task("same text", seed=2)
        r1 = synthesize_task(t1, engine, conditioning=None, project_cache=cache)
        r2 = synthesize_task(t2, engine, conditioning=None, project_cache=cache)
        assert engine.synthesize_calls == 2  # no cross-seed cache poisoning
        assert r1.chunk_hash != r2.chunk_hash
        # Repeating seed 1 hits its own entry.
        r1b = synthesize_task(t1, engine, conditioning=None, project_cache=cache)
        assert r1b.cache_hit is True
        assert engine.synthesize_calls == 2
