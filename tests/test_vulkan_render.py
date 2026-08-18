"""Slow pipeline-level test: full Vulkan-backend render of a chunked dialogue.

Mirrors the conventions of ``tests/test_smoke_render.py``: a deterministic
drop-in engine (here ``AudioCppVulkanEngine``) is patched into the pipeline,
references are written with the smoke helper, and truth is asserted against
the rendered FLAC, the render plan metadata, and the stem cache.

The point of the namespace assertions: the Vulkan backend must hash every
stem (including chunked ones) under the ``vulkan:chatterbox:*`` cache key so
it can never collide with a PyTorch-rendered stem for the same text.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.slow

from the_oracle.models.cache import CachedReference, ProjectCache
from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import OraclePipeline, RenderProgress, RenderSettings, SpeakerSettings, _chunk_engine_key
from the_oracle.smoke import _SmokeEmotionClassifier, _write_reference
from the_oracle.utils.chunking import chunk_utterance
from the_oracle.utils.hashing import build_chunk_hash, hash_file, hash_payload

VULKAN_ENGINE_VERSION = "deterministic-vulkan-v1"

# Two long utterances (both well past the 250-char chunking threshold) so the
# render exercises the chunked synthesis path end to end. Each sentence stays
# under the chunk ceiling, so chunking splits on sentence boundaries.
LONG_DIALOGUE = """Speaker A: The Vulkan backend routes every overlong utterance through audio.cpp, and when a single line exceeds the conservative chunking threshold the pipeline splits it into smaller pieces before synthesis even begins. This first long sentence exists purely to push the utterance past the limit so the render exercises the chunked synthesis path end to end. It has enough words to comfortably exceed two hundred fifty characters without any effort at all, which guarantees the splitter engages.
Speaker B: Every word of the second line is chosen so that this utterance also crosses the same two hundred fifty character ceiling and gets split into its own sequence of smaller chunks before reaching the engine. The renderer must then reassemble these chunks in order, preserve the pause after the turn, and write each stem into the cache under a hash that only the Vulkan backend can ever produce. That is the entire point of the namespace isolation this test exists to verify.
"""


class _FakeVulkanConditioning:
    """Lightweight conditioning payload, shaped like the pipeline expects."""

    def __init__(self, cache_id: str, speaker: str, reference_hash: str, variant: str) -> None:
        self.cache_id = cache_id
        self.speaker = speaker
        self.reference_hash = reference_hash
        self.variant = variant


class _DeterministicVulkanEngine:
    """Deterministic drop-in replacement for AudioCppVulkanEngine.

    Injected via ``patch("the_oracle.pipeline.AudioCppVulkanEngine", ...)`` so
    the pipeline sees it in place of the real audio.cpp-shipping engine. It
    implements the same duck-typed surface the pipeline touches during render:
    engine_id, sample_rate, engine_version, ensure_model_ready,
    prepare_reference, prepare_conditioning, synthesize, synthesize_batch.
    """

    engine_id = "vulkan"
    sample_rate = 24000
    engine_version = VULKAN_ENGINE_VERSION
    # Records every constructor kwargs the pipeline passes, so tests can
    # assert RenderSettings.audio_cpp_device/threads/timeout/max_batch reach
    # the engine.
    recorded_ctor_kwargs: list[dict[str, int | None]] = []
    # Records how synthesis was invoked: batch_calls holds one entry per
    # synthesize_batch call (the number of utterances in that batch);
    # single_synthesize_calls counts per-utterance synthesize calls. The batch
    # render test asserts the pipeline NEVER falls back to per-utterance
    # synthesis: exactly one batch call covering every cache-missing stem.
    batch_calls: list[int] = []
    single_synthesize_calls: int = 0

    def __init__(
        self,
        variant: str = "standard",
        device: str | None = None,
        *,
        device_index: int | None = None,
        threads: int | None = None,
        timeout: int | None = None,
        batch_limit: int | None = None,
    ) -> None:
        self.variant = variant
        self.device = device or "vulkan"
        self.device_index = device_index
        self.threads = threads
        self.timeout = timeout
        self.batch_limit = batch_limit
        type(self).recorded_ctor_kwargs.append(
            {
                "device_index": device_index,
                "threads": threads,
                "timeout": timeout,
                "batch_limit": batch_limit,
            }
        )

    def ensure_model_ready(self) -> None:
        pass

    def prepare_reference(
        self,
        project_cache: ProjectCache,
        speaker: str,
        reference_path: str,
    ) -> CachedReference:
        # Mirror the real engine: audio.cpp reads the wav itself, so the
        # normalized path is the original path.
        source = Path(reference_path)
        return CachedReference(
            original_path=str(source),
            normalized_path=str(source),
            original_hash=hash_file(source),
            sample_rate=self.sample_rate,
        )

    def prepare_conditioning(
        self,
        project_cache: ProjectCache,
        speaker: str,
        cached_reference: CachedReference,
        settings: VoiceSettings,
    ) -> _FakeVulkanConditioning:
        cache_id = hash_payload(
            {
                "backend": "vulkan",
                "speaker": speaker,
                "reference_hash": cached_reference.original_hash,
            }
        )
        return _FakeVulkanConditioning(
            cache_id=cache_id,
            speaker=speaker,
            reference_hash=cached_reference.original_hash,
            variant=self.variant,
        )

    def _deterministic_audio(self, text: str, conditioning: _FakeVulkanConditioning, settings: VoiceSettings) -> np.ndarray:
        seed = hash_payload(
            {
                "text": text,
                "speaker": conditioning.speaker,
                "reference_hash": conditioning.reference_hash,
                "variant": conditioning.variant,
                "settings": settings.to_dict(),
            }
        )
        base = int(seed[:8], 16)
        duration_seconds = 0.22 + 0.02 * max(1, len(text.split()))
        samples = max(1, int(round(self.sample_rate * duration_seconds)))
        time_axis = np.arange(samples, dtype=np.float32) / np.float32(self.sample_rate)
        frequency = 180.0 + float(base % 240)
        phase = np.float32((base % 360) * np.pi / 180.0)
        amplitude = np.float32(0.12 + ((base >> 8) % 25) / 500.0)
        envelope = np.linspace(1.0, 0.7, num=samples, dtype=np.float32)
        audio = amplitude * np.sin((2.0 * np.pi * frequency * time_axis) + phase) * envelope
        return np.asarray(audio, dtype=np.float32)

    def synthesize(self, text: str, conditioning: _FakeVulkanConditioning, settings: VoiceSettings) -> np.ndarray:
        type(self).single_synthesize_calls += 1
        return self._deterministic_audio(text, conditioning, settings)

    def synthesize_batch(
        self,
        entries: list[tuple[str, _FakeVulkanConditioning, VoiceSettings]],
        on_request_complete=None,
    ) -> list[tuple[np.ndarray, int, float]]:
        type(self).batch_calls.append(len(entries))
        outputs: list[tuple[np.ndarray, int, float]] = []
        for index, (text, conditioning, settings) in enumerate(entries):
            # Mirror the real engine: fire the live-progress callback as each
            # request "lands", before the batch returns.
            if on_request_complete is not None:
                on_request_complete(index)
            outputs.append(
                (
                    self._deterministic_audio(text, conditioning, settings),
                    self.sample_rate,
                    250.0,  # wall_ms, parsed from [TIMING] by the real engine
                )
            )
        return outputs


def _expected_chunk_hashes(plan, backend: str, engine_version: str) -> set[str]:
    """Recompute the stem hashes a render would build for one backend.

    Mirrors the task-building loop in ``OraclePipeline.render``: chunk each
    utterance with ``chunk_utterance`` and hash each chunk with the backend's
    cache key, the profile's engine params, and the resolved reference hash.
    """
    expected: set[str] = set()
    for utterance in plan.utterances:
        profile = plan.voice_profiles[utterance.speaker]
        text = utterance.text_for_tts()
        chunks = chunk_utterance(text, utterance.index)
        for chunk in chunks:
            # Render uses the full text for the single-chunk case, chunk text otherwise.
            chunk_text = text if len(chunks) == 1 and chunk.is_single_chunk else chunk.text
            expected.add(
                build_chunk_hash(
                    speaker=utterance.speaker,
                    repaired_text=chunk_text,
                    engine_key=_chunk_engine_key(backend, "standard"),
                    engine_params=profile.engine_params.to_dict(),
                    engine_version=engine_version,
                    reference_audio_hash=profile.reference_audio_hash,
                )
            )
    return expected


def test_vulkan_render_chunked_dialogue_stems_stay_in_vulkan_namespace(tmp_path: Path) -> None:
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(LONG_DIALOGUE, encoding="utf-8")
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    speaker_settings = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=VoiceSettings()),
    }
    render_settings = RenderSettings(
        model_variant="standard",
        language="en",
        loudness_preset="off",
        inference_backend="vulkan",
    )

    # Force the text-repair helpers onto their built-in fallback paths so the
    # render stays deterministic and fast (no HuggingFace punctuator / language
    # tool model load) -- the same convention the doctor's deterministic smoke
    # uses. The repair pipeline still runs its regex/spelling fallbacks.
    with (
        patch("the_oracle.pipeline.AudioCppVulkanEngine", _DeterministicVulkanEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.text_repair.grammar.GrammarCorrector._try_load_language_tool", return_value=None),
        patch("the_oracle.text_repair.punctuation.PunctuationRestorer._try_load_punctuator", return_value=None),
    ):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, render_settings)
        output_path = pipeline.render(plan, render_settings)

    # The render itself completed and produced real audio.
    assert output_path.exists()
    audio, sample_rate = sf.read(output_path, always_2d=False)
    assert sample_rate == 24000
    assert len(audio) > 1000

    # The plan truthfully records the Vulkan backend.
    assert plan.metadata["inference_backend"] == "vulkan"
    assert "via audio.cpp Vulkan" in plan.metadata["comment"]
    # The Vulkan backend never uses the multiprocessing worker pool.
    assert plan.metadata["synthesis_mode"] == "sequential"

    # Chunking must actually have engaged: more stems than source utterances.
    vulkan_expected = _expected_chunk_hashes(plan, "vulkan", VULKAN_ENGINE_VERSION)
    pytorch_expected = _expected_chunk_hashes(plan, "pytorch", VULKAN_ENGINE_VERSION)
    assert len(vulkan_expected) > len(plan.utterances), "dialogue did not chunk; lengthen the fixture"

    # Every stem on disk is exactly the vulkan-namespaced set...
    project_cache = ProjectCache(plan.output_dir)
    stem_names = {path.stem for path in project_cache.stem_cache_dir.glob("*.wav")}
    assert stem_names == vulkan_expected
    # ...and none of the pytorch-namespaced hashes for the same content exist:
    # the two backends can never collide in the shared stem cache.
    assert stem_names.isdisjoint(pytorch_expected)
    assert not any(project_cache.stem_path(hash_value).exists() for hash_value in pytorch_expected)


def test_vulkan_render_batches_all_misses_into_one_process(tmp_path: Path) -> None:
    """The Vulkan inline path must synthesize every cache-missing stem through
    ONE synthesize_batch call (a single audio.cpp --request-sequence process =
    one model load + shader compile), never falling back to per-utterance
    synthesize calls. The render timeline records how many processes ran."""
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(LONG_DIALOGUE, encoding="utf-8")
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    speaker_settings = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=VoiceSettings()),
    }
    render_settings = RenderSettings(
        model_variant="standard",
        language="en",
        loudness_preset="off",
        inference_backend="vulkan",
    )
    _DeterministicVulkanEngine.recorded_ctor_kwargs = []
    _DeterministicVulkanEngine.batch_calls = []
    _DeterministicVulkanEngine.single_synthesize_calls = 0

    with (
        patch("the_oracle.pipeline.AudioCppVulkanEngine", _DeterministicVulkanEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.text_repair.grammar.GrammarCorrector._try_load_language_tool", return_value=None),
        patch("the_oracle.text_repair.punctuation.PunctuationRestorer._try_load_punctuator", return_value=None),
    ):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, render_settings)
        output_path = pipeline.render(plan, render_settings)

    assert output_path.exists()
    vulkan_expected = _expected_chunk_hashes(plan, "vulkan", VULKAN_ENGINE_VERSION)
    # Chunking engaged, so there are more stems than source utterances, and
    # every one of them was synthesized by the single batch call.
    assert len(vulkan_expected) > len(plan.utterances)
    assert _DeterministicVulkanEngine.batch_calls == [len(vulkan_expected)]
    assert _DeterministicVulkanEngine.single_synthesize_calls == 0

    # The render timeline truthfully records one process serving all requests.
    timing_path = ProjectCache(plan.output_dir).project_dir / "logs" / "render_timings.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    assert timing["timeline"]["vulkan_batch_processes"] == 1
    assert timing["timeline"]["vulkan_batch_requests"] == len(vulkan_expected)

    # Every stem on disk is exactly the vulkan-namespaced set (cache writes
    # flow through the same save_wav path as single synthesis).
    project_cache = ProjectCache(plan.output_dir)
    stem_names = {path.stem for path in project_cache.stem_cache_dir.glob("*.wav")}
    assert stem_names == vulkan_expected


def test_vulkan_render_batch_cache_hits_do_not_respawn(tmp_path: Path) -> None:
    """Second render (all stems cached) must not spawn any audio.cpp process:
    the all-cached fast path short-circuits before the engine is touched."""
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(LONG_DIALOGUE, encoding="utf-8")
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    speaker_settings = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=VoiceSettings()),
    }
    render_settings = RenderSettings(
        model_variant="standard",
        language="en",
        loudness_preset="off",
        inference_backend="vulkan",
    )
    _DeterministicVulkanEngine.batch_calls = []
    _DeterministicVulkanEngine.single_synthesize_calls = 0

    def _render_once() -> None:
        with (
            patch("the_oracle.pipeline.AudioCppVulkanEngine", _DeterministicVulkanEngine),
            patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
            patch("the_oracle.text_repair.grammar.GrammarCorrector._try_load_language_tool", return_value=None),
            patch("the_oracle.text_repair.punctuation.PunctuationRestorer._try_load_punctuator", return_value=None),
        ):
            pipeline = OraclePipeline()
            plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, render_settings)
            pipeline.render(plan, render_settings)

    _render_once()
    first_render_batches = list(_DeterministicVulkanEngine.batch_calls)
    assert first_render_batches, "first render should have batched synthesis"

    _DeterministicVulkanEngine.batch_calls = []
    _DeterministicVulkanEngine.single_synthesize_calls = 0
    _render_once()

    assert _DeterministicVulkanEngine.batch_calls == []
    assert _DeterministicVulkanEngine.single_synthesize_calls == 0


def test_vulkan_render_batch_cap_splits_many_stems_into_multiple_processes(
    tmp_path: Path, monkeypatch
) -> None:
    """ORACLE_AUDIOCPP_MAX_BATCH caps each audio.cpp process: a render with
    more cache-missing stems than the cap splits into multiple batches (each
    at most the cap), the timeline records the true process/request counts,
    and all stems still land in the vulkan namespace."""
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "2")
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(LONG_DIALOGUE, encoding="utf-8")
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    speaker_settings = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=VoiceSettings()),
    }
    render_settings = RenderSettings(
        model_variant="standard",
        language="en",
        loudness_preset="off",
        inference_backend="vulkan",
    )
    _DeterministicVulkanEngine.batch_calls = []
    _DeterministicVulkanEngine.single_synthesize_calls = 0

    with (
        patch("the_oracle.pipeline.AudioCppVulkanEngine", _DeterministicVulkanEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.text_repair.grammar.GrammarCorrector._try_load_language_tool", return_value=None),
        patch("the_oracle.text_repair.punctuation.PunctuationRestorer._try_load_punctuator", return_value=None),
    ):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, render_settings)
        output_path = pipeline.render(plan, render_settings)

    assert output_path.exists()
    vulkan_expected = _expected_chunk_hashes(plan, "vulkan", VULKAN_ENGINE_VERSION)
    # Chunking produced enough stems to exceed the cap of 2.
    assert len(vulkan_expected) > 2
    assert _DeterministicVulkanEngine.single_synthesize_calls == 0
    batch_sizes = _DeterministicVulkanEngine.batch_calls
    assert len(batch_sizes) > 1, "cap=2 should split the render into multiple batches"
    assert all(size <= 2 for size in batch_sizes)
    assert sum(batch_sizes) == len(vulkan_expected)

    # The timeline truthfully records one process per batch and all requests.
    timing_path = ProjectCache(plan.output_dir).project_dir / "logs" / "render_timings.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    assert timing["timeline"]["vulkan_batch_processes"] == len(batch_sizes)
    assert timing["timeline"]["vulkan_batch_requests"] == len(vulkan_expected)

    # Every stem on disk is exactly the vulkan-namespaced set.
    project_cache = ProjectCache(plan.output_dir)
    stem_names = {path.stem for path in project_cache.stem_cache_dir.glob("*.wav")}
    assert stem_names == vulkan_expected


def test_vulkan_render_batch_cap_from_settings_splits_stems(tmp_path: Path) -> None:
    """The batch cap threaded through RenderSettings.audio_cpp_max_batch (the
    CLI flag / manifest field / GUI spin) must actually drive grouping -- no
    ORACLE_AUDIOCPP_MAX_BATCH env var needed. A render with more cache-missing
    stems than the setting splits into multiple batches, each at most the
    setting's value."""
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(LONG_DIALOGUE, encoding="utf-8")
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    speaker_settings = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=VoiceSettings()),
    }
    render_settings = RenderSettings(
        model_variant="standard",
        language="en",
        loudness_preset="off",
        inference_backend="vulkan",
        audio_cpp_max_batch=2,
    )
    _DeterministicVulkanEngine.batch_calls = []
    _DeterministicVulkanEngine.single_synthesize_calls = 0
    _DeterministicVulkanEngine.recorded_ctor_kwargs = []

    with (
        patch("the_oracle.pipeline.AudioCppVulkanEngine", _DeterministicVulkanEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.text_repair.grammar.GrammarCorrector._try_load_language_tool", return_value=None),
        patch("the_oracle.text_repair.punctuation.PunctuationRestorer._try_load_punctuator", return_value=None),
    ):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, render_settings)
        output_path = pipeline.render(plan, render_settings)

    assert output_path.exists()
    vulkan_expected = _expected_chunk_hashes(plan, "vulkan", VULKAN_ENGINE_VERSION)
    assert len(vulkan_expected) > 2
    assert _DeterministicVulkanEngine.single_synthesize_calls == 0
    batch_sizes = _DeterministicVulkanEngine.batch_calls
    assert len(batch_sizes) > 1, "audio_cpp_max_batch=2 should split the render into multiple batches"
    assert all(size <= 2 for size in batch_sizes)
    assert sum(batch_sizes) == len(vulkan_expected)
    # The engine received the setting as its constructor batch_limit.
    assert _DeterministicVulkanEngine.recorded_ctor_kwargs[0]["batch_limit"] == 2


def test_vulkan_render_emits_progress_per_request_during_batch(tmp_path: Path, monkeypatch) -> None:
    """A Vulkan render must emit a progress event as each request completes
    while the batch subprocess is still running (live per-request progress),
    not only after the whole batch returns. The progress callback therefore
    sees 'Synthesized segment N/M' events before the batch returns, the bar
    can advance during the render, and the shared results loop does NOT
    re-emit a duplicate 'ready' event for those tasks (no 1..N rewind)."""
    # Force one batch regardless of how many stems the fixture chunks into,
    # so the single-batch assertions below are not coupled to the cap.
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "64")
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(LONG_DIALOGUE, encoding="utf-8")
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    speaker_settings = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=VoiceSettings()),
    }
    render_settings = RenderSettings(
        model_variant="standard",
        language="en",
        loudness_preset="off",
        inference_backend="vulkan",
    )
    _DeterministicVulkanEngine.batch_calls = []
    _DeterministicVulkanEngine.single_synthesize_calls = 0

    progress_events: list[RenderProgress] = []

    with (
        patch("the_oracle.pipeline.AudioCppVulkanEngine", _DeterministicVulkanEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.text_repair.grammar.GrammarCorrector._try_load_language_tool", return_value=None),
        patch("the_oracle.text_repair.punctuation.PunctuationRestorer._try_load_punctuator", return_value=None),
    ):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, render_settings)
        pipeline.render(plan, render_settings, progress_callback=progress_events.append)

    # Every cache-missing stem was synthesized through ONE batch call, and the
    # pipeline emitted a live progress event per request as it landed.
    assert _DeterministicVulkanEngine.single_synthesize_calls == 0
    assert len(_DeterministicVulkanEngine.batch_calls) == 1
    vulkan_expected = _expected_chunk_hashes(plan, "vulkan", VULKAN_ENGINE_VERSION)
    synthesizing = [
        event
        for event in progress_events
        if event.stage == "Rendering segment" and "Synthesized segment" in event.detail
    ]
    assert len(synthesizing) == len(vulkan_expected)
    # Segments advance monotonically during the batch: 1..N, not a jump.
    assert [event.current_segment for event in synthesizing] == list(range(1, len(vulkan_expected) + 1))
    # No rewind: the results loop must NOT re-emit a 'Segment ... ready'
    # duplicate for live-reported tasks, so the total per-segment events are
    # exactly N (never 2N). Locks in the no-visible-rewind fix.
    segment_events = [event for event in progress_events if event.stage == "Rendering segment"]
    assert len(segment_events) == len(vulkan_expected)


def test_vulkan_render_mixed_cache_progress_never_rewinds(tmp_path: Path) -> None:
    """Mixed render (some stems cached, some synthesized fresh): the shared
    progress counter must keep current_segment/current_step strictly increasing
    across the live batch events and the results loop. Cache-hit stems are only
    emitted by the results loop, misses are emitted live during the batch; if
    the two used different counters, an early cache hit would emit a LOWER
    step/segment than the last live event and visibly rewind the GUI bar."""
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(LONG_DIALOGUE, encoding="utf-8")
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    speaker_settings = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=VoiceSettings()),
    }
    render_settings = RenderSettings(
        model_variant="standard",
        language="en",
        loudness_preset="off",
        inference_backend="vulkan",
    )
    _DeterministicVulkanEngine.batch_calls = []
    _DeterministicVulkanEngine.single_synthesize_calls = 0

    progress_events: list[RenderProgress] = []

    with (
        patch("the_oracle.pipeline.AudioCppVulkanEngine", _DeterministicVulkanEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.text_repair.grammar.GrammarCorrector._try_load_language_tool", return_value=None),
        patch("the_oracle.text_repair.punctuation.PunctuationRestorer._try_load_punctuator", return_value=None),
    ):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, render_settings)
        # Resolve reference hashes (render does this internally) so the stem
        # hashes can be computed before rendering.
        for profile in plan.voice_profiles.values():
            profile.reference_audio_hash = hash_file(profile.primary_reference)
        vulkan_expected = _expected_chunk_hashes(plan, "vulkan", VULKAN_ENGINE_VERSION)
        assert len(vulkan_expected) > 3, "fixture must chunk into several stems"
        # Seed every other stem as a cache hit so misses and hits interleave in
        # task order -- exactly the mixed case that used to rewind the bar.
        project_cache = ProjectCache(plan.output_dir)
        ordered = sorted(vulkan_expected)
        for index, chunk_hash in enumerate(ordered):
            if index % 2 == 0:
                sf.write(project_cache.stem_path(chunk_hash), np.zeros(2400, dtype=np.float32), 24000)
        seeded_hits = set(ordered[::2])
        misses = vulkan_expected - seeded_hits
        assert misses, "seeding left no misses to synthesize"

        pipeline.render(plan, render_settings, progress_callback=progress_events.append)

    rendering = [event for event in progress_events if event.stage == "Rendering segment"]
    segments = [event.current_segment for event in rendering]
    steps = [event.current_step for event in rendering]
    # Strictly increasing 1..N across live batch events AND loop emissions:
    # no rewind, no duplicate, every stem accounted for exactly once.
    assert segments == list(range(1, len(vulkan_expected) + 1))
    assert all(left < right for left, right in zip(steps, steps[1:]))
    # Only the misses reached the engine, all in one batch.
    assert _DeterministicVulkanEngine.single_synthesize_calls == 0
    assert _DeterministicVulkanEngine.batch_calls == [len(misses)]


def test_vulkan_render_forwards_device_and_threads_settings(tmp_path: Path) -> None:
    """RenderSettings.audio_cpp_device/threads/timeout/max_batch must reach
    the Vulkan engine constructor (the CLI flags and manifest fields ride
    RenderSettings, so they work without ORACLE_AUDIOCPP_DEVICE/THREADS/
    TIMEOUT/MAX_BATCH)."""
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text("Speaker A: Hello from the Vulkan test.\nSpeaker B: Confirmed.\n", encoding="utf-8")
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    speaker_settings = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=VoiceSettings()),
    }
    render_settings = RenderSettings(
        model_variant="standard",
        language="en",
        loudness_preset="off",
        inference_backend="vulkan",
        audio_cpp_device=2,
        audio_cpp_threads=6,
        audio_cpp_timeout=120,
        audio_cpp_max_batch=8,
    )
    _DeterministicVulkanEngine.recorded_ctor_kwargs = []

    with (
        patch("the_oracle.pipeline.AudioCppVulkanEngine", _DeterministicVulkanEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.text_repair.grammar.GrammarCorrector._try_load_language_tool", return_value=None),
        patch("the_oracle.text_repair.punctuation.PunctuationRestorer._try_load_punctuator", return_value=None),
    ):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, render_settings)
        pipeline.render(plan, render_settings)

    assert _DeterministicVulkanEngine.recorded_ctor_kwargs, "pipeline never constructed the Vulkan engine"
    assert _DeterministicVulkanEngine.recorded_ctor_kwargs[0] == {
        "device_index": 2,
        "threads": 6,
        "timeout": 120,
        "batch_limit": 8,
    }
    # Provenance: the render plan records the knobs that were used.
    assert plan.metadata["audio_cpp_device"] == "2"
    assert plan.metadata["audio_cpp_threads"] == "6"
    assert plan.metadata["audio_cpp_timeout"] == "120"
    assert plan.metadata["audio_cpp_max_batch"] == "8"
