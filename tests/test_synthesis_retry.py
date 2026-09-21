"""The one-shot hiccup retry: a rejected generation self-heals once, then fails loudly.

The engine output gate (:func:`the_oracle.utils.audio.sanitize_engine_audio`)
recognizes Chatterbox's degenerate-generation pathologies — flat silence, a DC
tone, NaNs, empty output. Before the retry existed, one such hiccup failed the
whole utterance (and on the batched Vulkan path, the whole batch). The design
pins here: a rejected generation retries EXACTLY once at a genuinely different
seed (seed + 1 when configured; unseeded stays unseeded, whose fresh draw is
already different), because the engines re-apply their seed on every call —
retrying at the SAME seed would deterministically reproduce the same
degenerate output. A second rejection propagates so the stem is never cached.

Everything runs on ``__new__``-shaped doubles and a stubbed command runner:
no model load, no subprocess, no GPU. The Vulkan doubles stub
``_run_command``/``_run_batch_command_streaming`` (the process boundary), not
the audio math, so the gate/retry machinery is the thing under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from the_oracle.models.project import VoiceSettings
from the_oracle.tts_engines import vulkan_backend as vb
from the_oracle.tts_engines.chatterbox_engine import ChatterboxEngine
from the_oracle.tts_engines.vulkan_backend import AudioCppVulkanEngine
from the_oracle.utils.audio import DegenerateEngineOutput, sanitize_engine_audio


def _speech_like(seconds: float = 0.2, rate: int = 24000) -> np.ndarray:
    """A ramped, non-degenerate waveform the gate accepts."""
    n = int(seconds * rate)
    return np.linspace(-0.5, 0.5, n, dtype=np.float32)


def _engine(backend: type, seed: int | None):
    """Build a ``__new__``-shaped engine with only the retry-relevant state."""
    engine = object.__new__(backend)
    engine.seed = seed
    return engine


class _StubConditioning:
    reference_path = None


_SETTINGS = VoiceSettings()


# --------------------------------------------------------------------------
# ChatterboxEngine (PyTorch path)
# --------------------------------------------------------------------------


def test_chatterbox_retry_succeeds_after_one_rejection() -> None:
    engine = _engine(ChatterboxEngine, seed=7)
    calls: list[int | None] = []

    def flaky_once(text, conditioning, settings, seed):
        calls.append(seed)
        if len(calls) == 1:
            raise DegenerateEngineOutput("flat silence")
        return _speech_like()

    engine._generate_once = flaky_once  # type: ignore[method-assign]
    audio = engine.synthesize("hello", _StubConditioning(), _SETTINGS)

    assert audio.shape[0] > 0
    assert calls == [7, 8], "retry must advance a configured seed exactly once"


def test_chatterbox_unseeded_retry_stays_unseeded() -> None:
    engine = _engine(ChatterboxEngine, seed=None)
    calls: list[int | None] = []

    def flaky_once(text, conditioning, settings, seed):
        calls.append(seed)
        if len(calls) == 1:
            raise DegenerateEngineOutput("flat silence")
        return _speech_like()

    engine._generate_once = flaky_once  # type: ignore[method-assign]
    engine.synthesize("hello", _StubConditioning(), _SETTINGS)

    assert calls == [None, None], "unseeded retry must stay unseeded (fresh randomness)"


def test_chatterbox_second_rejection_propagates() -> None:
    engine = _engine(ChatterboxEngine, seed=7)
    calls: list[int | None] = []

    def always_degenerate(text, conditioning, settings, seed):
        calls.append(seed)
        raise DegenerateEngineOutput("flat silence")

    engine._generate_once = always_degenerate  # type: ignore[method-assign]
    with pytest.raises(DegenerateEngineOutput):
        engine.synthesize("hello", _StubConditioning(), _SETTINGS)

    assert calls == [7, 8], "exactly one retry: two draws, both seeded 7 then 8"


def test_chatterbox_unrelated_valueerror_never_retries() -> None:
    engine = _engine(ChatterboxEngine, seed=None)
    calls: list[int] = []

    def broken_model(text, conditioning, settings, seed):
        calls.append(seed)
        raise RuntimeError("model exploded")

    engine._generate_once = broken_model  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        engine.synthesize("hello", _StubConditioning(), _SETTINGS)

    assert len(calls) == 1, "non-gate failures must propagate without a retry"


# --------------------------------------------------------------------------
# AudioCppVulkanEngine (single-utterance subprocess path)
# --------------------------------------------------------------------------


class _VulkanHarness:
    """Minimal engine state for the Vulkan retry paths, no model/binaries.

    ``seed`` is a read-only class property on the real engine (constructor
    override -> env fallback); tests patch it per-test with the harness seed.
    """

    def __init__(self, seed: int | None):
        self.seed = seed


def _vulkan_engine(monkeypatch, seed: int | None) -> AudioCppVulkanEngine:
    engine = object.__new__(AudioCppVulkanEngine)
    monkeypatch.setattr(
        AudioCppVulkanEngine, "seed", property(lambda self: seed)
    )
    return engine


def test_vulkan_single_retry_uses_fresh_seed(tmp_path, monkeypatch) -> None:
    engine = _vulkan_engine(monkeypatch, seed=11)
    seeds_in_commands: list[int | None] = []
    calls = {"n": 0}

    def fake_once(self, text, conditioning, settings, seed=vb._USE_CONFIGURED_SEED):
        from the_oracle.tts_engines import vulkan_backend as vb

        resolved = engine.seed if seed is vb._USE_CONFIGURED_SEED else seed
        seeds_in_commands.append(resolved)
        calls["n"] += 1
        if calls["n"] == 1:
            raise DegenerateEngineOutput("DC tone")
        return _speech_like()

    monkeypatch.setattr(AudioCppVulkanEngine, "_synthesize_once", fake_once)
    audio = engine.synthesize("hello", _StubConditioning(), _SETTINGS)

    assert audio.shape[0] > 0
    assert seeds_in_commands == [11, 12]


def test_vulkan_single_second_rejection_propagates(tmp_path, monkeypatch) -> None:
    engine = _vulkan_engine(monkeypatch, seed=5)
    calls = {"n": 0}

    def always_degenerate(self, text, conditioning, settings, seed=vb._USE_CONFIGURED_SEED):
        calls["n"] += 1
        raise DegenerateEngineOutput("DC tone")

    monkeypatch.setattr(AudioCppVulkanEngine, "_synthesize_once", always_degenerate)
    with pytest.raises(DegenerateEngineOutput):
        engine.synthesize("hello", _StubConditioning(), _SETTINGS)
    assert calls["n"] == 2, "exactly one retry"


# --------------------------------------------------------------------------
# AudioCppVulkanEngine.synthesize_batch (batched subprocess path)
# --------------------------------------------------------------------------


class _FakeBatchRunner:
    """Stands in for the batch subprocess: scriptable degenerate requests.

    ``degenerate_rounds`` maps a run ordinal (0 = first pass, 1 = retry pass)
    to the set of request indexes that come back degenerate on that run.
    """

    def __init__(self, degenerate_rounds: dict[int, set[int]], rate: int = 24000):
        self.degenerate_rounds = degenerate_rounds
        self.rate = rate
        self.runs: list[list[str]] = []
        self.progress_fired: list[int] = []

    def __call__(self, engine, entries, on_request_complete, retry_seed):
        run_ordinal = len(self.runs)
        self.runs.append([text for text, _, _ in entries])
        outputs: list[tuple[np.ndarray, int, float] | None] = [None] * len(entries)
        rejected: dict[int, DegenerateEngineOutput] = {}
        for index, (text, _, _) in enumerate(entries):
            if index in self.degenerate_rounds.get(run_ordinal, set()):
                rejected[index] = DegenerateEngineOutput(f"request {index} degenerate")
                continue
            outputs[index] = (_speech_like(rate=self.rate), self.rate, 0.0)
            if on_request_complete is not None:
                self.progress_fired.append(index)
                on_request_complete(index)
        return outputs, rejected


def _batch_engine(monkeypatch, runner: _FakeBatchRunner, seed: int | None):
    engine = _vulkan_engine(monkeypatch, seed)
    # The batch cap check reads self.batch_limit; satisfy it at the instance.
    engine._batch_limit_override = 32
    # Model provisioning (binary discovery, GGUF checks) is out of scope here.
    monkeypatch.setattr(AudioCppVulkanEngine, "ensure_model_ready", lambda self: None)

    def fake_uncapped(self, entries, on_request_complete, retry_seed=vb._USE_CONFIGURED_SEED):
        if retry_seed is not vb._USE_CONFIGURED_SEED:
            # Route the retry pass through the harness so the test can see the
            # seed the retry subprocess would carry.
            runner.retry_seed_seen = retry_seed
        return runner(engine, entries, on_request_complete, retry_seed)

    monkeypatch.setattr(AudioCppVulkanEngine, "_synthesize_batch_uncapped", fake_uncapped)
    return engine


def test_batch_retry_runs_only_rejected_requests(monkeypatch) -> None:
    runner = _FakeBatchRunner(degenerate_rounds={0: {1}})
    engine = _batch_engine(monkeypatch, runner, seed=100)
    entries = [
        (f"utterance {i}", _StubConditioning(), _SETTINGS) for i in range(3)
    ]

    results = engine.synthesize_batch(entries, on_request_complete=lambda i: None)

    assert runner.runs[0] == ["utterance 0", "utterance 1", "utterance 2"]
    assert runner.runs[1] == ["utterance 1"], (
        "the retry subprocess must carry only the gate-rejected request"
    )
    assert runner.retry_seed_seen == 101, "batch retry must advance the seed"
    assert [r is not None for r in results] == [True, True, True]
    # Slot 0 and 2 keep their FIRST-pass audio objects (not fresh copies).
    assert results[0] is not None and results[2] is not None


def test_batch_retry_preserves_result_order_and_progress_count(monkeypatch) -> None:
    runner = _FakeBatchRunner(degenerate_rounds={0: {0, 2}})
    engine = _batch_engine(monkeypatch, runner, seed=None)
    entries = [
        (f"utterance {i}", _StubConditioning(), _SETTINGS) for i in range(3)
    ]
    progress: list[int] = []
    results = engine.synthesize_batch(entries, on_request_complete=progress.append)

    assert runner.runs[1] == ["utterance 0", "utterance 2"]
    assert results[0] is not None and results[1] is not None and results[2] is not None
    # Progress fired once per successful first-pass request, in order; the
    # retry pass must NOT re-fire (progress maps batch-local indexes to tasks).
    assert progress == [1]
    assert runner.progress_fired == [1]


def test_batch_second_rejection_fails_loudly(monkeypatch) -> None:
    runner = _FakeBatchRunner(degenerate_rounds={0: {1}, 1: {0}})
    engine = _batch_engine(monkeypatch, runner, seed=7)
    entries = [
        (f"utterance {i}", _StubConditioning(), _SETTINGS) for i in range(3)
    ]

    with pytest.raises(DegenerateEngineOutput):
        engine.synthesize_batch(entries)

    assert len(runner.runs) == 2, "retry runs exactly once"


def test_batch_second_rejection_names_lowest_index(monkeypatch) -> None:
    runner = _FakeBatchRunner(degenerate_rounds={0: {1, 2}, 1: {0, 1}})
    engine = _batch_engine(monkeypatch, runner, seed=7)
    entries = [
        (f"utterance {i}", _StubConditioning(), _SETTINGS) for i in range(4)
    ]

    # Retry pass carries original indexes [1, 2]; the harness labels its
    # rejections by run-local index, so run-local 0 == original 1 — the
    # lowest ORIGINAL index among the still-rejected. That one must be raised.
    with pytest.raises(DegenerateEngineOutput, match="request 0"):
        engine.synthesize_batch(entries)


def test_batch_no_rejections_keeps_single_pass(monkeypatch) -> None:
    runner = _FakeBatchRunner(degenerate_rounds={})
    engine = _batch_engine(monkeypatch, runner, seed=3)
    entries = [
        (f"utterance {i}", _StubConditioning(), _SETTINGS) for i in range(2)
    ]

    results = engine.synthesize_batch(entries)

    assert len(runner.runs) == 1
    assert all(r is not None for r in results)


def test_gate_marker_is_valueerror_subclass() -> None:
    """The retry keys on the gate's marker; unrelated ValueErrors must not retry."""
    assert issubclass(DegenerateEngineOutput, ValueError)
    with pytest.raises(DegenerateEngineOutput):
        sanitize_engine_audio(np.zeros(24000, dtype=np.float32), text="hello")
