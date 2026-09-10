"""Focused tests for the review-fix cycle: concurrency, cache/IO safety,
Vulkan lifecycle, manifests, recording allocation, logging, and tolerant
parsing.

Heavy third-party dependencies (soundfile, torch, PySide6, huggingface_hub)
are NOT installed in this runtime, so this module installs minimal
functional stubs for ``soundfile`` (real WAV bytes via the stdlib ``wave``
module) and ``huggingface_hub`` (enough for module import) *before* importing
``the_oracle``. The stubs are removed from ``sys.modules`` immediately after
the import so the rest of the suite keeps its baseline collection behavior
for modules that import those packages directly.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import pickle
import subprocess
import sys
import threading
import time
import types
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import-time stub setup (see module docstring).
# ---------------------------------------------------------------------------

def _install_stubs():
    previous = {}

    sf_stub = types.ModuleType("soundfile")

    def _sf_write(path, data, samplerate, format=None, subtype=None, **kwargs):
        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(samplerate))
            handle.writeframes(pcm.tobytes())

    def _sf_read(path, dtype="float32", always_2d=False, **kwargs):
        with wave.open(str(path), "rb") as handle:
            nframes = handle.getnframes()
            nchannels = handle.getnchannels()
            framerate = handle.getframerate()
            sampwidth = handle.getsampwidth()
            raw = handle.readframes(nframes)
        if sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:  # pragma: no cover - stub only handles PCM WAVs
            raise ValueError(f"stub soundfile: unsupported sampwidth {sampwidth}")
        if nchannels > 1:
            audio = audio.reshape(-1, nchannels)
        if always_2d and audio.ndim == 1:
            audio = audio[:, None]
        return audio, framerate

    class _SfInfo:
        def __init__(self, samplerate):
            self.samplerate = samplerate

    def _sf_info(path):
        with wave.open(str(path), "rb") as handle:
            return _SfInfo(handle.getframerate())

    sf_stub.write = _sf_write
    sf_stub.read = _sf_read
    sf_stub.info = _sf_info

    hf_stub = types.ModuleType("huggingface_hub")
    hf_stub.snapshot_download = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("stub huggingface_hub: downloads are disabled in tests")
    )
    hf_errors = types.ModuleType("huggingface_hub.errors")

    class LocalEntryNotFoundError(Exception):
        pass

    hf_errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    hf_utils = types.ModuleType("huggingface_hub.utils")

    class LocalTokenNotFoundError(Exception):
        pass

    hf_utils.LocalTokenNotFoundError = LocalTokenNotFoundError
    hf_stub.errors = hf_errors
    hf_stub.utils = hf_utils

    stubs = {
        "soundfile": sf_stub,
        "huggingface_hub": hf_stub,
        "huggingface_hub.errors": hf_errors,
        "huggingface_hub.utils": hf_utils,
    }
    for name, stub in stubs.items():
        previous[name] = sys.modules.get(name)
        sys.modules[name] = stub
    return previous


def _remove_stubs(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


_STUB_PREVIOUS = _install_stubs()
try:
    from the_oracle.models.cache import (
        ProjectCache,
        atomic_write,
        sanitize_speaker_component,
    )
    from the_oracle.models.project import (
        CorrectionRecord,
        RenderPlan,
        Utterance,
        VoiceSettings,
    )
    from the_oracle.pipeline import (
        RenderSettings,
        SpeakerSettings,
        SynthesisTask,
        _run_tasks_with_worker_pool,
        _time_weighted_progress,
    )
    from the_oracle.project_manifest import (
        ProjectManifestError,
        build_saved_project,
        load_project_manifest,
        save_project_manifest,
        saved_project_from_dict,
    )
    from the_oracle.tts_engines import chatterbox_engine as chatterbox_engine_module
    from the_oracle.tts_engines import vulkan_backend as vulkan_backend_module
    from the_oracle.tts_engines.chatterbox_engine import (
        ChatterboxConditioning,
        ChatterboxEngine,
    )
    from the_oracle.tts_engines.vulkan_backend import AudioCppVulkanEngine
    from the_oracle import vulkan_setup as vulkan_setup_module
    from the_oracle.vulkan_setup import (
        VulkanSetupCancelled,
        _run_script_streaming,
    )
    from the_oracle.audio import recorder as recorder_module
    from the_oracle.audio.recorder import (
        allocate_seashell_path,
        next_seashell_name,
    )
    from the_oracle.utils import logging as oracle_logging
    from the_oracle.utils.logging import configure_logging
finally:
    _remove_stubs(_STUB_PREVIOUS)
del _STUB_PREVIOUS


@pytest.fixture(autouse=True)
def _restore_logging():
    """Leave the root logger in a sane state after logging tests."""
    yield
    oracle_logging._created_file_handlers.clear()
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    logging.basicConfig(level=logging.WARNING, force=True)


def _write_wav(path: Path, samples: np.ndarray, rate: int = 24000) -> None:
    """Write a real mono PCM16 WAV without soundfile (stub is import-scoped)."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _cached_reference(speaker: str = "A", original_hash: str = "hash-a"):
    from the_oracle.models.cache import CachedReference

    return CachedReference(
        original_path="ref.wav",
        normalized_path="ref.wav",
        original_hash=original_hash,
        sample_rate=24000,
    )


# ---------------------------------------------------------------------------
# #1: atomic_write
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_success_writes_content(self, tmp_path):
        destination = tmp_path / "out.json"
        atomic_write(destination, lambda tmp: tmp.write_text('{"a": 1}', encoding="utf-8"))
        assert destination.read_text(encoding="utf-8") == '{"a": 1}'

    def test_writer_failure_leaves_destination_untouched(self, tmp_path):
        destination = tmp_path / "out.json"
        destination.write_text("original", encoding="utf-8")

        def _boom(tmp):
            tmp.write_text("partial", encoding="utf-8")
            raise RuntimeError("writer exploded")

        with pytest.raises(RuntimeError, match="writer exploded"):
            atomic_write(destination, _boom)
        assert destination.read_text(encoding="utf-8") == "original"
        leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_no_temp_files_remain_on_success(self, tmp_path):
        destination = tmp_path / "data.bin"
        atomic_write(destination, lambda tmp: tmp.write_bytes(b"data"))
        assert [p.name for p in tmp_path.iterdir()] == ["data.bin"]


# ---------------------------------------------------------------------------
# #2: atomic cache writes + speaker sanitization in cache.py
# ---------------------------------------------------------------------------


class TestProjectCacheAtomicity:
    def test_save_json_and_load_roundtrip(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        payload = {"alpha": [1, 2, 3], "beta": "text"}
        cache.save_json("state.json", payload)
        raw = (tmp_path / "project" / "state.json").read_text(encoding="utf-8")
        assert json.loads(raw) == payload

    def test_write_text_and_read(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        cache.write_text("log.txt", "hello log")
        assert (tmp_path / "project" / "log.txt").read_text(encoding="utf-8") == "hello log"

    def test_store_conditioning_roundtrip(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        payload = {"speaker_embedding": [0.1, 0.2]}
        cache_id = cache.store_conditioning("A", "hash-a", payload)
        assert cache.load_conditioning(cache_id) == payload

    def test_failed_json_write_keeps_previous_good(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        cache.save_json("state.json", {"version": 1})
        before = (tmp_path / "project" / "state.json").read_text(encoding="utf-8")

        def _failing(path, writer):
            raise OSError("disk exploded")

        import the_oracle.models.cache as cache_module

        with patch.object(cache_module, "atomic_write", side_effect=_failing):
            with pytest.raises(OSError, match="disk exploded"):
                cache.save_json("state.json", {"version": 2})
        assert (tmp_path / "project" / "state.json").read_text(encoding="utf-8") == before

    def test_speaker_component_sanitized(self):
        assert sanitize_speaker_component("Alice") == "Alice"
        nasty = sanitize_speaker_component("../../etc/evil")
        assert "/" not in nasty and "\\" not in nasty and ".." not in nasty
        assert sanitize_speaker_component("") == "speaker"

    def test_cache_reference_audio_sanitizes_speaker(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        source = tmp_path / "source.wav"
        _write_wav(source, np.zeros(240, dtype=np.float32))
        reference = cache.cache_reference_audio(source, "../../evil", 24000)
        name = Path(reference.normalized_path).name
        assert "/" not in name and ".." not in name
        assert Path(reference.normalized_path).exists()

    def test_export_stem_rejects_escape(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        with pytest.raises(ValueError, match="escapes"):
            cache.export_stem(Path("stem.wav"), "../../evil.wav")

    def test_export_stem_accepts_nested_relative(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        stem = tmp_path / "stem.wav"
        _write_wav(stem, np.zeros(120, dtype=np.float32))
        destination = cache.export_stem(stem, "stems/chunk-1.wav")
        assert destination.exists()


# ---------------------------------------------------------------------------
# #3: path confinement
# ---------------------------------------------------------------------------


class TestPathConfinement:
    def test_absolute_path_rejected(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        with pytest.raises(ValueError, match="relative"):
            cache.save_json("/etc/passwd.json", {})

    def test_parent_escape_rejected(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        with pytest.raises(ValueError, match="escapes"):
            cache.save_json("../../outside.json", {})

    def test_dotdot_that_stays_inside_allowed(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        cache.save_json("subdir/../inside.json", {"ok": True})
        assert (tmp_path / "project" / "inside.json").exists()

    def test_write_text_rejects_escape(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        with pytest.raises(ValueError, match="escapes"):
            cache.write_text("../outside.txt", "x")


# ---------------------------------------------------------------------------
# #5: turbo conditioning cache key includes norm_loudness
# ---------------------------------------------------------------------------


class _FakeConds:
    def __init__(self, path):
        self._path = Path(path)

    def save(self, path):
        Path(path).write_bytes(b"fake-conds")


class _FakeConditionModel:
    def __init__(self):
        self.calls: list[dict] = []
        self.conds = _FakeConds("unused")

    def prepare_conditionals(self, wav, exaggeration=0.5, norm_loudness=True):
        self.calls.append(
            {"wav": wav, "exaggeration": exaggeration, "norm_loudness": norm_loudness}
        )
        return _FakeConds("prepared")


class TestTurboConditioningCacheKey:
    def _engine(self, variant):
        engine = ChatterboxEngine(variant=variant, device="cpu")
        engine._model = _FakeConditionModel()
        return engine

    def test_norm_loudness_flip_recomputes_for_turbo(self, tmp_path):
        engine = self._engine("turbo")
        cache = ProjectCache(tmp_path / "project")
        loud = VoiceSettings(norm_loudness=True)
        quiet = VoiceSettings(norm_loudness=False)
        first = engine.prepare_conditioning(cache, "A", _cached_reference(), loud)
        second = engine.prepare_conditioning(cache, "A", _cached_reference(), quiet)
        assert first.cache_id != second.cache_id
        assert len(engine._model.calls) == 2
        # Same settings again must reuse the cache, not recompute.
        engine.prepare_conditioning(cache, "A", _cached_reference(), loud)
        assert len(engine._model.calls) == 2

    def test_standard_variant_key_unchanged_by_norm_loudness(self, tmp_path):
        engine = self._engine("standard")
        cache = ProjectCache(tmp_path / "project")
        first = engine.prepare_conditioning(
            cache, "A", _cached_reference(), VoiceSettings(norm_loudness=True)
        )
        second = engine.prepare_conditioning(
            cache, "A", _cached_reference(), VoiceSettings(norm_loudness=False)
        )
        assert first.cache_id == second.cache_id
        assert len(engine._model.calls) == 1


# ---------------------------------------------------------------------------
# #6: serialize conds swap + generate
# ---------------------------------------------------------------------------


class _FakeGenerateModel:
    def __init__(self):
        self.conds = None
        self.seen_conds: list = []

    def generate(self, **kwargs):
        self.seen_conds.append(self.conds)
        return np.zeros(8, dtype=np.float32)


class TestSynthesizeLock:
    def test_concurrent_synthesize_serializes_conds(self):
        engine = ChatterboxEngine(variant="standard", device="cpu")
        engine._model = _FakeGenerateModel()
        engine._loaded_conditioning = {"x": "CONDS-A"}

        conditioning = ChatterboxConditioning(
            cache_id="c", path=Path("x"), reference_hash="h", speaker="A",
            variant="standard",
        )
        finished: list[str] = []
        engine._synthesize_lock.acquire()
        worker = threading.Thread(
            target=lambda: finished.append(
                engine.synthesize("hi", conditioning, VoiceSettings()).shape
            )
        )
        try:
            worker.start()
            worker.join(timeout=3)
            assert worker.is_alive(), (
                "synthesize() must block while another holder owns the conditioning lock"
            )
        finally:
            engine._synthesize_lock.release()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert finished == [(8,)]
        assert engine._model.seen_conds == ["CONDS-A"]


# ---------------------------------------------------------------------------
# Pool tests: one failing utterance must not abort the render (#pool),
# and abandoning the stream must terminate the pool (pool lifecycle).
# ---------------------------------------------------------------------------


class _FakePoolEngine:
    """Picklable (module-level) stand-in engine for fork-pool tests."""

    engine_id = "fake"

    def __init__(self, variant="standard", device="cpu", seed=None):
        self.variant = variant
        self.device = device

    def ensure_model_ready(self):
        pass

    @property
    def engine_version(self):
        return "fake-1"

    @property
    def sample_rate(self):
        return 24000

    def prepare_reference(self, project_cache, speaker, reference_path):
        from the_oracle.models.cache import CachedReference

        return CachedReference(
            original_path=str(reference_path),
            normalized_path=str(reference_path),
            original_hash=f"hash-{speaker}",
            sample_rate=24000,
        )

    def prepare_conditioning(self, project_cache, speaker, cached_reference, settings):
        return ("conds", speaker)

    def synthesize(self, text, conditioning, settings):
        if text == "BOOM":
            raise RuntimeError("simulated worker failure")
        return np.zeros(2400, dtype=np.float32)


_fork_context = multiprocessing.get_context("fork")
from multiprocessing.pool import Pool as _BasePool


class _RecordingPool(_BasePool):
    """Pool subclass that records terminate()/close() calls."""

    instances: list["_RecordingPool"] = []

    def __init__(self, *args, **kwargs):
        self.terminated_flag = False
        self.closed_flag = False
        type(self).instances.append(self)
        super().__init__(*args, **kwargs)

    def terminate(self):
        self.terminated_flag = True
        super().terminate()

    def close(self):
        self.closed_flag = True
        super().close()


class _RecordingContext:
    def Pool(self, *args, **kwargs):
        return _RecordingPool(*args, **kwargs)


def _pool_task(index, text, tmp_path):
    return SynthesisTask(
        utterance_index=index,
        source_index=index,
        speaker="A",
        text=text,
        reference_audio_hash="hash-A",
        reference_path=tmp_path / "ref.wav",
        voice_settings=VoiceSettings(),
        model_variant="standard",
        device_mode="cpu",
        export_stems=False,
        inference_backend="pytorch",
        chunk_hash=f"testhash{index:04d}",
    )


@pytest.fixture
def _recording_pool(monkeypatch):
    _RecordingPool.instances.clear()
    monkeypatch.setattr(multiprocessing, "get_context", lambda *args, **kwargs: _RecordingContext())
    return _RecordingPool


class TestWorkerPoolFailureIsolation:
    def test_one_failing_utterance_does_not_abort_render(self, tmp_path, _recording_pool):
        ref = tmp_path / "ref.wav"
        _write_wav(ref, np.zeros(240, dtype=np.float32))
        project_dir = tmp_path / "project"
        tasks = [
            _pool_task(1, "hello", tmp_path),
            _pool_task(2, "BOOM", tmp_path),
            _pool_task(3, "world", tmp_path),
        ]
        results, mode = _run_tasks_with_worker_pool(
            tasks,
            _FakePoolEngine,
            "standard",
            "cpu",
            str(project_dir),
            worker_count=2,
            stream=False,
        )
        assert mode == "parallel"
        assert [r.utterance_index for r in results] == [1, 2, 3]
        assert results[0].error is None
        assert results[2].error is None
        assert results[1].error is not None
        assert "simulated worker failure" in results[1].error
        # Successful utterances still produced stems.
        assert Path(results[0].stem_path).exists()
        assert Path(results[2].stem_path).exists()

    def test_abandoning_stream_terminates_pool(self, tmp_path, _recording_pool):
        ref = tmp_path / "ref.wav"
        _write_wav(ref, np.zeros(240, dtype=np.float32))
        project_dir = tmp_path / "project"
        tasks = [_pool_task(i, f"text {i}", tmp_path) for i in range(1, 7)]
        stream, mode = _run_tasks_with_worker_pool(
            tasks,
            _FakePoolEngine,
            "standard",
            "cpu",
            str(project_dir),
            worker_count=2,
            stream=True,
        )
        assert mode == "parallel"
        # Start iterating (the realistic abandonment scenario), then bail out.
        # Closing a never-started generator would run no cleanup code at all,
        # so one item must be pulled before abandoning the stream.
        next(stream)
        stream.close()  # abandon with work still queued
        pool = _RecordingPool.instances[-1]
        assert pool.terminated_flag, "abandoned stream must terminate the pool"
        assert not pool.closed_flag, "terminated pool must not also be closed"


# ---------------------------------------------------------------------------
# #10: progress clamp
# ---------------------------------------------------------------------------


class TestTimeWeightedProgress:
    def test_segments_done_clamped_to_total(self):
        render_state = {
            "segments_total": 4,
            "segments_done": 7,  # over-counted; must not drive ETA negative
            "segment_avg": 2.0,
            "backend": "vulkan",
        }
        fraction, eta = _time_weighted_progress(render_state, 100.0, "Rendering")
        assert 0.0 <= fraction <= 1.0
        assert eta == pytest.approx(3.0)  # remaining=0 -> tail estimate only

    def test_normal_progress(self):
        render_state = {
            "segments_total": 4,
            "segments_done": 2,
            "segment_avg": 2.0,
            "backend": "vulkan",
        }
        fraction, eta = _time_weighted_progress(render_state, 100.0, "Rendering")
        assert 0.0 < fraction < 1.0
        assert eta == pytest.approx(2 * 2.0 + 3.0)


# ---------------------------------------------------------------------------
# #4: streaming batch cleanup on every failure path
# ---------------------------------------------------------------------------


class _FakeBatchStream:
    def __init__(self, lines):
        self._lines = list(lines)
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


class _FakeBatchProc:
    """Fake Popen for _run_batch_command_streaming."""

    def __init__(self, stdout_lines=(), stderr_lines=(), hang=True):
        self.args = ("fake-audiocpp",)
        self.pid = 4242
        self.returncode = None
        self.stdout = _FakeBatchStream(stdout_lines)
        self.stderr = _FakeBatchStream(stderr_lines)
        self._hang = hang
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


class TestStreamingBatchCleanup:
    def _engine(self, tmp_path, timeout=30):
        return AudioCppVulkanEngine(variant="standard", device="vulkan", timeout=timeout)

    def test_progress_callback_exception_still_cleans_up(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "request_0.wav").touch()
        proc = _FakeBatchProc(stdout_lines=["line\n"])
        monkeypatch.setattr(
            vulkan_backend_module.subprocess, "Popen", lambda *a, **k: proc
        )
        engine = self._engine(tmp_path)

        def _boom(index):
            raise ValueError("callback exploded")

        with pytest.raises(ValueError, match="callback exploded"):
            engine._run_batch_command_streaming(
                ("fake",), out_dir, request_count=1, timeout=30, on_request_complete=_boom
            )
        assert proc.killed, "child must be killed even when the callback raised"
        assert proc.wait_calls >= 1, "child must be reaped"
        assert proc.stdout.closed and proc.stderr.closed, "pipes must be closed"
        deadline = time.time() + 5
        while any(t.is_alive() for t in threading.enumerate()) and time.time() < deadline:
            time.sleep(0.01)

    def test_timeout_kills_and_reaps_child(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        proc = _FakeBatchProc(stdout_lines=[], hang=True)
        monkeypatch.setattr(
            vulkan_backend_module.subprocess, "Popen", lambda *a, **k: proc
        )
        engine = self._engine(tmp_path)
        with pytest.raises(RuntimeError, match="timed out"):
            engine._run_batch_command_streaming(
                ("fake",), out_dir, request_count=1, timeout=0.05,
                on_request_complete=lambda index: None,
            )
        assert proc.killed
        assert proc.wait_calls >= 1
        assert proc.stdout.closed and proc.stderr.closed


# ---------------------------------------------------------------------------
# #9: setup cancellation must surface as cancellation, never a cleanup error
# ---------------------------------------------------------------------------


class _SetupFakeProc:
    def __init__(self, lines, wait_behavior):
        self.args = ("fake-setup",)
        self.pid = 424242
        self.returncode = None
        self._lines = list(lines)
        self._wait_behavior = wait_behavior
        self.stdout = self._stream()
        self.wait_calls = 0

    def _stream(self):
        for line in self._lines:
            yield line + "\n"

    def wait(self, timeout=None):
        self.wait_calls += 1
        action = self._wait_behavior[min(self.wait_calls - 1, len(self._wait_behavior) - 1)]
        if action == "timeout":
            raise subprocess.TimeoutExpired(self.args, timeout)
        self.returncode = 0
        return self.returncode


class TestVulkanSetupCancellation:
    def test_cancellation_not_masked_by_cleanup(self, monkeypatch):
        cancel = threading.Event()
        killpg_calls: list[tuple[int, int]] = []
        real_killpg = os.killpg

        def _fake_killpg(pid, sig):
            killpg_calls.append((pid, sig))

        monkeypatch.setattr(os, "killpg", _fake_killpg)
        proc = _SetupFakeProc(["building"], wait_behavior=["timeout", "ok"])

        def _popen(*args, **kwargs):
            cancel.set()  # cancel arrives mid-stream
            return proc

        monkeypatch.setattr(vulkan_setup_module.subprocess, "Popen", _popen)
        with pytest.raises(VulkanSetupCancelled):
            _run_script_streaming("fake-script", progress=lambda line: None, cancel=cancel)
        assert proc.wait_calls == 2, "child must be reaped even when SIGTERM is ignored"
        signals = [sig for _, sig in killpg_calls]
        import signal as signal_module

        assert signals[0] == signal_module.SIGTERM
        assert signals[1] == signal_module.SIGKILL

    def test_normal_completion(self, monkeypatch):
        proc = _SetupFakeProc(["line1", "line2"], wait_behavior=["ok"])
        monkeypatch.setattr(
            vulkan_setup_module.subprocess, "Popen", lambda *a, **k: proc
        )
        seen: list[str] = []
        code, output = _run_script_streaming(
            "fake-script", progress=seen.append, cancel=None
        )
        assert code == 0
        assert output == "line1\nline2"
        assert seen == ["line1", "line2"]


# ---------------------------------------------------------------------------
# #12: timeout validation
# ---------------------------------------------------------------------------


class TestTimeoutValidation:
    def test_negative_constructor_timeout_rejected(self):
        with pytest.raises(ValueError, match="timeout"):
            AudioCppVulkanEngine(timeout=-5)

    def test_zero_constructor_timeout_rejected(self):
        with pytest.raises(ValueError, match="timeout"):
            AudioCppVulkanEngine(timeout=0)

    def test_positive_constructor_timeout_accepted(self):
        engine = AudioCppVulkanEngine(timeout=30)
        assert engine.timeout == 30

    def test_negative_env_timeout_clamped(self, monkeypatch):
        monkeypatch.setenv("ORACLE_AUDIOCPP_TIMEOUT", "-30")
        assert vulkan_backend_module._synthesis_timeout_seconds() == 1.0

    def test_zero_env_timeout_clamped(self, monkeypatch):
        monkeypatch.setenv("ORACLE_AUDIOCPP_TIMEOUT", "0")
        assert vulkan_backend_module._synthesis_timeout_seconds() == 1.0

    def test_malformed_env_timeout_falls_back(self, monkeypatch):
        monkeypatch.setenv("ORACLE_AUDIOCPP_TIMEOUT", "bogus")
        assert vulkan_backend_module._synthesis_timeout_seconds() == 600.0

    def test_valid_env_timeout_parsed(self, monkeypatch):
        monkeypatch.setenv("ORACLE_AUDIOCPP_TIMEOUT", "45")
        assert vulkan_backend_module._synthesis_timeout_seconds() == 45.0


# ---------------------------------------------------------------------------
# #8: race-free recording-name allocation
# ---------------------------------------------------------------------------


class TestSeashellAllocation:
    def test_allocate_creates_and_reserves(self, tmp_path):
        first = allocate_seashell_path(tmp_path)
        second = allocate_seashell_path(tmp_path)
        assert first != second
        assert first.exists() and second.exists()
        assert first.stem == "Seashell_No_1"
        assert second.stem == "Seashell_No_2"

    def test_allocate_skips_existing_numbers(self, tmp_path):
        (tmp_path / "Seashell_No_1.wav").touch()
        (tmp_path / "Seashell_No_2.wav").touch()
        assert allocate_seashell_path(tmp_path).stem == "Seashell_No_3"

    def test_concurrent_allocation_never_collides(self, tmp_path):
        results: list[Path] = []
        errors: list[BaseException] = []

        def _worker():
            try:
                results.append(allocate_seashell_path(tmp_path))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert not errors
        assert len({str(p) for p in results}) == 16
        assert all(p.exists() for p in results)

    def test_next_seashell_name_suggestion_unchanged(self, tmp_path):
        assert next_seashell_name(tmp_path) == "Seashell_No_1"


# ---------------------------------------------------------------------------
# #11: logging reconfiguration must not leak handlers/FDs
# ---------------------------------------------------------------------------


class TestLoggingReconfiguration:
    def test_repeated_configure_keeps_single_file_handler(self, tmp_path):
        log_a = tmp_path / "a.log"
        log_b = tmp_path / "b.log"
        configure_logging(log_a)
        first_handlers = list(logging.getLogger().handlers)
        configure_logging(log_b)
        configure_logging(log_b)
        handlers = logging.getLogger().handlers
        file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == str(log_b)
        # The old file handler was closed, not just detached: FileHandler.close()
        # nulls the stream after closing the FD, so accept either signal.
        old_file_handlers = [h for h in first_handlers if isinstance(h, logging.FileHandler)]
        assert old_file_handlers, "expected a file handler from the first configure"
        for handler in old_file_handlers:
            stream = handler.stream
            assert stream is None or stream.closed, "detached handler must be closed (FD leak)"

    def test_still_logs_after_reconfigure(self, tmp_path):
        log_file = tmp_path / "run.log"
        configure_logging(log_file)
        logging.getLogger("oracle-test").warning("hello-log")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert "hello-log" in log_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# #7: manifest atomicity with previous-good backup
# ---------------------------------------------------------------------------


def _sample_saved_project():
    plan = RenderPlan(
        title="Manifest Test",
        source_path="input.md",
        output_dir="output",
        engine="chatterbox",
        correction_mode="moderate",
    )
    settings = RenderSettings()
    speaker_settings = {"A": SpeakerSettings(reference_path="")}
    return build_saved_project(plan, settings, speaker_settings)


class TestManifestAtomicity:
    def test_save_creates_backup_of_previous(self, tmp_path):
        manifest_path = tmp_path / "project.oracle.json"
        save_project_manifest(manifest_path, _sample_saved_project())
        project = _sample_saved_project()
        project.plan.title = "Second Version"
        save_project_manifest(manifest_path, project)
        backup = tmp_path / "project.oracle.json.bak"
        assert backup.exists()
        assert json.loads(backup.read_text(encoding="utf-8"))["render_plan"]["title"] == "Manifest Test"
        loaded = load_project_manifest(manifest_path)
        assert loaded.plan.title == "Second Version"

    def test_failed_save_keeps_previous_manifest(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "project.oracle.json"
        save_project_manifest(manifest_path, _sample_saved_project())
        before = manifest_path.read_text(encoding="utf-8")

        def _boom(*args, **kwargs):
            raise OSError("disk exploded")

        import the_oracle.project_manifest as manifest_module

        monkeypatch.setattr(manifest_module, "atomic_write", _boom)
        with pytest.raises(OSError, match="disk exploded"):
            save_project_manifest(manifest_path, _sample_saved_project())
        assert manifest_path.read_text(encoding="utf-8") == before

    def test_write_render_plan_keeps_backup(self, tmp_path):
        from the_oracle.models.cache import read_previous_render_plan, write_render_plan

        destination = tmp_path / "project" / "render_plan.json"
        plan = RenderPlan(
            title="Plan One",
            source_path="a.md",
            output_dir="o",
            engine="chatterbox",
            correction_mode="moderate",
        )
        write_render_plan(plan, destination)
        plan.title = "Plan Two"
        write_render_plan(plan, destination)
        backup = tmp_path / "project" / "render_plan.json.bak"
        assert backup.exists()
        assert json.loads(backup.read_text(encoding="utf-8"))["title"] == "Plan One"
        assert read_previous_render_plan(destination)["title"] == "Plan Two"


# ---------------------------------------------------------------------------
# #13: tolerant from_dict parsing
# ---------------------------------------------------------------------------


class TestTolerantParsing:
    def test_utterance_ignores_unknown_fields(self):
        utterance = Utterance.from_dict(
            {
                "index": 0,
                "original_text": "Hello",
                "repaired_text": "Hello",
                "speaker": "A",
                "future_field": {"nested": [1, 2, 3]},
            }
        )
        assert utterance.index == 0
        assert utterance.speaker == "A"
        assert not hasattr(utterance, "future_field")

    def test_utterance_still_rejects_missing_required_field(self):
        with pytest.raises(TypeError):
            Utterance.from_dict({"index": 0})  # original_text is required

    def test_correction_record_ignores_unknown_fields(self):
        record = CorrectionRecord.from_dict(
            {"stage": "s", "before": "a", "after": "b", "confidence": 0.9}
        )
        assert (record.stage, record.before, record.after) == ("s", "a", "b")

    def test_render_plan_ignores_unknown_fields(self):
        plan = RenderPlan.from_dict(
            {
                "title": "T",
                "source_path": "s.md",
                "output_dir": "o",
                "engine": "chatterbox",
                "correction_mode": "moderate",
                "utterances": [],
                "unknown_top_level": True,
            }
        )
        assert plan.title == "T"
        assert plan.utterances == []

    def test_saved_project_ignores_unknown_render_settings(self):
        payload = _sample_saved_project().to_dict()
        payload["render_settings"]["future_knob"] = {"deep": [1]}
        project = saved_project_from_dict(payload)
        assert project.render_settings.model_variant == "standard"

    def test_saved_project_ignores_unknown_utterance_fields(self):
        project = _sample_saved_project()
        project.plan.utterances.append(
            Utterance(
                index=0,
                original_text="Hi",
                repaired_text="Hi",
                speaker="A",
            )
        )
        payload = project.to_dict()
        payload["utterances"][0]["future_field"] = "x"
        reloaded = saved_project_from_dict(payload)
        assert reloaded.plan.utterances[0].repaired_text == "Hi"

    def test_saved_project_rejects_malformed_known_render_setting(self):
        payload = _sample_saved_project().to_dict()
        payload["render_settings"]["audio_cpp_timeout"] = -5
        with pytest.raises(ValueError, match="positive"):
            saved_project_from_dict(payload)

    def test_saved_project_still_rejects_bad_speaker_key(self):
        payload = _sample_saved_project().to_dict()
        payload["speaker_settings"]["not a valid key!"] = payload["speaker_settings"]["A"]
        with pytest.raises(ProjectManifestError, match="invalid speaker keys"):
            saved_project_from_dict(payload)


# ---------------------------------------------------------------------------
# Hardened path handling: identifier validation + symlink-aware confinement
# ---------------------------------------------------------------------------


class TestHardenedCachePaths:
    def test_stem_path_rejects_malicious_hash(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        with pytest.raises(ValueError, match="chunk hash"):
            cache.stem_path("../../evil")
        with pytest.raises(ValueError, match="chunk hash"):
            cache.stem_path("a/b")

    def test_conditioning_path_rejects_malicious_id(self, tmp_path):
        cache = ProjectCache(tmp_path / "project")
        with pytest.raises(ValueError, match="conditioning cache id"):
            cache.conditioning_path("../escape")
        with pytest.raises(ValueError, match="conditioning cache id"):
            cache.conditioning_path("")

    def test_confined_path_rejects_symlink_escape(self, tmp_path):
        project = tmp_path / "project"
        cache = ProjectCache(project)
        outside = tmp_path / "outside"
        outside.mkdir()
        (project / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="escapes"):
            cache.save_json("link/evil.json", {"x": 1})
        assert not (outside / "evil.json").exists()

    def test_confined_path_allows_interior_symlink(self, tmp_path):
        project = tmp_path / "project"
        cache = ProjectCache(project)
        real = project / "realdir"
        real.mkdir()
        (project / "alias").symlink_to(real, target_is_directory=True)
        cache.save_json("alias/ok.json", {"a": 1})
        assert (real / "ok.json").exists()

    def test_reference_wav_writer_uses_explicit_format(self, tmp_path):
        # The atomic temp file has a .tmp suffix, so the writer must name the
        # format explicitly instead of relying on extension inference.
        import the_oracle.models.cache as cache_module

        seen: dict = {}
        real_write = cache_module.sf.write

        def _spy(path, data, samplerate, **kwargs):
            seen.update(kwargs)
            return real_write(path, data, samplerate, **kwargs)

        cache_module.sf.write = _spy
        try:
            cache = ProjectCache(tmp_path / "project")
            source = tmp_path / "source.wav"
            _write_wav(source, np.zeros(240, dtype=np.float32))
            cache.cache_reference_audio(source, "A", 24000)
        finally:
            cache_module.sf.write = real_write
        assert seen.get("format") == "WAV"
