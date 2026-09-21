"""Ownership audit: the engine-output gate is the only voice-stem writer, and
degenerate cache entries can never be served.

Two policies landed with the TTS hardening (2026-09-20) and the stem-cache
audit that followed:

1.  WRITE SIDE — every engine returns through ``sanitize_engine_audio``, so a
    degenerate stem (constant-DC tone, flat silence from an SOT-truncated
    token stream, NaN span, empty output) is repaired or rejected before it
    can reach the cache. The batched Vulkan path trusts its engine's gate by
    design: ``synthesize_batch`` is the *engine's* contract, so the pipeline
    does not re-sanitize its output (a pipeline-side re-check would silently
    excuse an engine that lost its own gate).

2.  READ SIDE — cache hits are no longer trusted on existence alone. Every
    read of a spoken-text stem re-classifies its content with the same
    taxonomy (``stem_is_speech_like``) and deletes entries it cannot serve,
    so a pre-hardening degenerate entry — or one written by any future
    writer that bypasses the gate — becomes a cache miss and re-synthesizes
    instead of hardening into every later render.

The single sanctioned exception is pause-only stems, which are legitimately
all-zero by construction (``_write_pause_only_stem``): silence is only ever
banked or served where silence is the expected product (``allow_silence``).
On the batched path, pause-only tasks bank their silent stem directly instead
of shipping empty text to audio.cpp — whose silent decode would then fail the
engine gate and take the whole batch down with it.

This file pins both sides: per-path stale-entry rejection, the pause-only
exemption, and a writer manifest — the scan of every stem/preview-shaped
write call in ``src/the_oracle`` must land in the gated owners (pipeline.py,
real_engine_smoke.py) and must not come up empty.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import soundfile as sf

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

from the_oracle.models.cache import CachedReference, ProjectCache  # noqa: E402
from the_oracle.models.project import VoiceSettings  # noqa: E402
from the_oracle.pipeline import (  # noqa: E402
    SynthesisTask,
    _chunk_engine_key,
    _load_cached_stem,
    _load_servable_stem,
    _write_pause_only_stem,
    _PAUSE_STEM_FALLBACK_SAMPLE_RATE,
    synthesize_task,
    synthesize_tasks_batched,
)
from the_oracle.utils.audio import sanitize_engine_audio, stem_is_speech_like  # noqa: E402
from the_oracle.utils.hashing import build_chunk_hash  # noqa: E402

SAMPLE_RATE = 24000
REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "the_oracle"


def _ramp(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Speech-like content under the gate's taxonomy (non-DC, non-silent)."""
    count = int(seconds * sample_rate)
    return np.linspace(-0.25, 0.25, count, dtype=np.float32)


def _zeros(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


class _FakeEngine:
    """Sequential-path engine double whose synthesize returns healthy audio."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.engine_version = "test-engine-1"
        self.synthesize_calls = 0

    def synthesize(self, text, conditioning, voice_settings):
        self.synthesize_calls += 1
        return _ramp(0.4, self.sample_rate)


class _GatedBatchStub:
    """Batched-path engine double: healthy gate output, records batch sizes."""

    engine_version = "stub-batch-v1"

    def __init__(self, audio: np.ndarray | None = None):
        self.batch_sizes: list[int] = []
        self._audio = _ramp(0.4) if audio is None else audio

    def prepare_reference(self, project_cache: ProjectCache, speaker: str, reference_path: str) -> CachedReference:
        return CachedReference(reference_path, reference_path, "refhash", SAMPLE_RATE)

    def prepare_conditioning(self, project_cache, speaker, cached_reference, settings):
        return object()  # opaque conditioning; synthesize_tasks_batched never inspects it

    def synthesize_batch(self, entries, on_request_complete=None):
        self.batch_sizes.append(len(entries))
        return [(self._audio, SAMPLE_RATE, 1.0) for _ in entries]


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


def _make_batch_task(text: str, utterance_index: int = 1, **overrides) -> SynthesisTask:
    """A task with its chunk_hash precomputed exactly as the batched path expects."""
    task = _make_task(
        text,
        utterance_index=utterance_index,
        inference_backend="vulkan",
        **overrides,
    )
    task.chunk_hash = build_chunk_hash(
        speaker=task.speaker,
        repaired_text=task.text,
        engine_key=_chunk_engine_key(task.inference_backend, task.model_variant),
        engine_params=task.voice_settings.to_dict(),
        engine_version="stub-batch-v1",
        reference_audio_hash=task.reference_audio_hash,
    )
    return task


def _sequential_hash(task: SynthesisTask, engine: _FakeEngine) -> str:
    """The chunk hash synthesize_task computes for a task without one."""
    return build_chunk_hash(
        speaker=task.speaker,
        repaired_text=task.text,
        engine_key=_chunk_engine_key(task.inference_backend, task.model_variant),
        engine_params=task.voice_settings.to_dict(),
        engine_version=engine.engine_version,
        reference_audio_hash=task.reference_audio_hash,
        seed=task.seed,
    )


def _result_for(results, utterance_index: int):
    return next(r for r in results if r.utterance_index == utterance_index)


# ---------------------------------------------------------------------------
# The read-side predicate
# ---------------------------------------------------------------------------


class TestStemIsSpeechLike:
    def test_rejects_all_zero_silence_for_spoken_text(self):
        assert stem_is_speech_like(_zeros(0.5)) is False

    def test_rejects_constant_dc_tone(self):
        tone = np.full(2400, 0.125, dtype=np.float32)
        assert stem_is_speech_like(tone) is False

    def test_accepts_speech_like_content(self):
        assert stem_is_speech_like(_ramp(0.5)) is True

    def test_silence_is_servable_only_where_silence_is_expected(self):
        assert stem_is_speech_like(_zeros(0.5), allow_silence=True) is True

    def test_rejects_non_finite_content(self):
        audio = _ramp(0.5)
        audio[10] = np.nan
        assert stem_is_speech_like(audio) is False

    def test_rejects_empty_content(self):
        assert stem_is_speech_like(np.zeros(0, dtype=np.float32)) is False


# ---------------------------------------------------------------------------
# Sequential path: stale degenerate entries are deleted, not served
# ---------------------------------------------------------------------------


class TestSequentialPathGate:
    def test_pre_hardening_silent_entry_is_deleted_and_resynthesized(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _FakeEngine()
        task = _make_task("hello there")
        stem_path = cache.stem_path(_sequential_hash(task, engine))
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pause_only_stem(stem_path, SAMPLE_RATE)  # banks exactly all-zero content
        assert _load_cached_stem(stem_path) is not None  # it is a *loadable* stale entry

        result = synthesize_task(task, engine, conditioning=None, project_cache=cache)

        assert result.cache_hit is False
        assert engine.synthesize_calls == 1
        # The degenerate entry was deleted; the replacement is servable speech-like audio.
        loaded = _load_cached_stem(stem_path)
        assert loaded is not None
        assert stem_is_speech_like(loaded[0]) is True

    def test_pause_only_cache_hit_serves_silence(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _FakeEngine()
        task = _make_task("")
        stem_path = cache.stem_path(_sequential_hash(task, engine))
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pause_only_stem(stem_path, SAMPLE_RATE)

        result = synthesize_task(task, engine, conditioning=None, project_cache=cache)

        assert result.cache_hit is True
        assert engine.synthesize_calls == 0
        loaded = _load_cached_stem(stem_path)
        assert loaded is not None
        assert float(np.max(np.abs(loaded[0]))) == 0.0  # silence, by design

    def test_healthy_entry_is_served_without_resynthesis(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _FakeEngine()
        task = _make_task("hello there")
        stem_path = cache.stem_path(_sequential_hash(task, engine))
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(stem_path), _ramp(0.4), SAMPLE_RATE, format="WAV")

        result = synthesize_task(task, engine, conditioning=None, project_cache=cache)

        assert result.cache_hit is True
        assert engine.synthesize_calls == 0


# ---------------------------------------------------------------------------
# Batched path: hits re-validated; pause-only banks silence off the engine
# ---------------------------------------------------------------------------


class TestBatchedPathGate:
    def test_spoken_cache_hit_requires_servable_content(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _GatedBatchStub()
        task = _make_batch_task("hello there")
        stem_path = cache.stem_path(task.chunk_hash)
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pause_only_stem(stem_path, SAMPLE_RATE)  # plant loadable all-zero content

        result = _result_for(
            synthesize_tasks_batched([task], engine, {}, cache), task.utterance_index
        )

        assert result.cache_hit is False
        assert engine.batch_sizes == [1]  # the engine gate produced the replacement
        loaded = _load_cached_stem(stem_path)
        assert loaded is not None
        assert stem_is_speech_like(loaded[0]) is True

    def test_pause_only_task_banks_silence_instead_of_engine_call(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _GatedBatchStub()
        task = _make_batch_task("")  # empty text never reaches audio.cpp
        stem_path = cache.stem_path(task.chunk_hash)
        assert not stem_path.exists()

        result = _result_for(
            synthesize_tasks_batched([task], engine, {}, cache), task.utterance_index
        )

        assert engine.batch_sizes == []  # the engine (and its gate) was never involved
        assert result.cache_hit is False
        assert stem_path.exists()
        loaded = _load_cached_stem(stem_path)
        assert loaded is not None
        audio, rate = loaded
        assert float(np.max(np.abs(audio))) == 0.0
        assert rate == _PAUSE_STEM_FALLBACK_SAMPLE_RATE

    def test_pause_only_cache_hit_skips_engine(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _GatedBatchStub()
        task = _make_batch_task("")
        stem_path = cache.stem_path(task.chunk_hash)
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pause_only_stem(stem_path, SAMPLE_RATE)

        result = _result_for(
            synthesize_tasks_batched([task], engine, {}, cache), task.utterance_index
        )

        assert engine.batch_sizes == []
        assert result.cache_hit is True

    def test_healthy_batch_hit_skips_engine(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        engine = _GatedBatchStub()
        task = _make_batch_task("hello there")
        stem_path = cache.stem_path(task.chunk_hash)
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(stem_path), _ramp(0.4), SAMPLE_RATE, format="WAV")

        result = _result_for(
            synthesize_tasks_batched([task], engine, {}, cache), task.utterance_index
        )

        assert engine.batch_sizes == []
        assert result.cache_hit is True


# ---------------------------------------------------------------------------
# The write gate itself (both engines route through it — see
# tests/test_engine_output_sanitizer.py for the engine wiring pins)
# ---------------------------------------------------------------------------


class TestWriteGateContract:
    def test_gate_rejects_the_shapes_the_read_gate_rejects(self):
        with pytest.raises(ValueError):
            sanitize_engine_audio(_zeros(0.5))
        with pytest.raises(ValueError):
            sanitize_engine_audio(np.full(2400, 0.125, dtype=np.float32))

    def test_gate_passes_the_shapes_the_read_gate_serves(self):
        assert sanitize_engine_audio(_ramp(0.4)).size == int(0.4 * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Writer manifest: stem/preview-shaped writes land only in the gated owners
# ---------------------------------------------------------------------------

_STEM_WRITE_RE = r"\b(save_wav|atomic_write)\(\s*(stem_path|preview_path)\b"
_EXPECTED_STEM_WRITERS = frozenset({"pipeline.py", "real_engine_smoke.py"})


def _scan_stem_write_sites() -> list[tuple[str, int, str]]:
    """Every save_wav/atomic_write call whose first argument is a stem/preview path.

    A lexical call-shape scan (variable-name based), not full dataflow: it pins
    the *known* write shapes and their owning modules. A future writer that
    renames the variable would drop out of this scan — which is exactly what
    the blindness guard below turns into a loud failure.
    """
    sites: list[tuple[str, int, str]] = []
    for path in sorted(REPO_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in {"save_wav", "atomic_write"} or not node.args:
                continue
            target = node.args[0]
            if isinstance(target, ast.Name) and target.id in {"stem_path", "preview_path"}:
                sites.append((path.name, node.lineno, node.func.id))
    return sites


class TestWriterManifest:
    def test_stem_writes_land_only_in_gated_owners(self):
        sites = _scan_stem_write_sites()
        offenders = sorted({name for name, _line, _fn in sites if name not in _EXPECTED_STEM_WRITERS})
        assert offenders == [], (
            "New modules are writing stem/preview-cache-shaped files directly. "
            "Voice stems must be written only after the engine-output gate "
            "(sanitize_engine_audio) — route the write through pipeline.py's "
            "gated paths, or extend _EXPECTED_STEM_WRITERS consciously with a "
            f"reason. Offending modules: {offenders}"
        )

    def test_writer_manifest_cannot_go_blind(self):
        sites = _scan_stem_write_sites()
        assert len(sites) >= 4, (
            f"stem-write scan found only {len(sites)} sites; the known surface "
            "is 5 (pipeline's pause/sequential/batched/preview writes and "
            "real_engine_smoke). The scan went blind — fix the scan, not this "
            "assertion."
        )
        owners = {name for name, _line, _fn in sites}
        assert owners == set(_EXPECTED_STEM_WRITERS), (
            f"stem-write owners drifted: found {sorted(owners)}, expected "
            f"{sorted(_EXPECTED_STEM_WRITERS)}"
        )


# ---------------------------------------------------------------------------
# Round trip: a rejected entry on the write side could never have been cached,
# and the read side deletes anything that nonetheless exists on disk
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_deleted_degenerate_entry_never_returns(self, tmp_path: Path):
        cache = ProjectCache(tmp_path / "proj")
        task = _make_batch_task("hello there")
        stem_path = cache.stem_path(task.chunk_hash)
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pause_only_stem(stem_path, SAMPLE_RATE)

        # Direct read-gate use (the shape the fast path and dispatch use):
        assert _load_servable_stem(stem_path) is None
        assert not stem_path.exists()

        # And the batched path re-synthesizes on the next render.
        engine = _GatedBatchStub()
        result = _result_for(
            synthesize_tasks_batched([task], engine, {}, cache), task.utterance_index
        )
        assert result.cache_hit is False
        assert stem_is_speech_like(_load_cached_stem(stem_path)[0]) is True
