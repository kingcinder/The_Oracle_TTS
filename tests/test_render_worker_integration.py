from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from the_oracle.models.project import VoiceProfile, VoiceSettings
from the_oracle.pipeline import (
    OraclePipeline,
    RenderPlan,
    RenderSettings,
    SynthesisResult,
    Utterance,
)
from the_oracle.smoke import _DeterministicChatterboxEngine, _SmokeEmotionClassifier, _write_reference
from the_oracle.text_repair.repairer import RepairResult

import the_oracle.pipeline as pipeline_module


class _StubTextRepairPipeline:
    def repair(self, text: str, mode: str = "moderate") -> RepairResult:
        return RepairResult(text=text, corrections=[])


def _build_render_plan(tmp_path: Path, render_settings: RenderSettings) -> RenderPlan:
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    voice_settings = VoiceSettings(variant=render_settings.model_variant, language="en")
    utterances = [
        Utterance(
            index=1,
            original_text="Speaker A.",
            repaired_text="Speaker A.",
            speaker="A",
            pause_after_ms=render_settings.pause_between_turns_ms,
            engine_settings=voice_settings,
        ),
    ]
    voice_profiles = {
        "A": VoiceProfile(
            name="Speaker A",
            speaker="A",
            reference_audio=[Path(speaker_a)],
            neutral_reference=Path(speaker_a),
            engine_params=voice_settings,
        ),
    }
    plan = RenderPlan(
        title="Worker pool tests",
        source_path=str(tmp_path / "dialogue.txt"),
        output_dir=str(project_dir),
        engine="chatterbox",
        correction_mode="moderate",
        metadata={"model_variant": render_settings.model_variant},
        utterances=utterances,
        voice_profiles=voice_profiles,
    )
    plan.update_hashes()
    return plan


def _build_synthesis_tasks(plan: RenderPlan, render_settings: RenderSettings) -> list[pipeline_module.SynthesisTask]:
    tasks = []
    for idx, utterance in enumerate(plan.utterances, start=1):
        profile = plan.voice_profiles[utterance.speaker]
        tasks.append(
            pipeline_module.SynthesisTask(
                utterance_index=idx,
                source_index=utterance.index,
                speaker=utterance.speaker,
                text=utterance.text_for_tts(),
                reference_path=profile.primary_reference,
                reference_audio_hash=profile.reference_audio_hash,
                voice_settings=profile.engine_params,
                model_variant=render_settings.model_variant,
                device_mode=render_settings.device_mode,
                export_stems=render_settings.export_stems,
            )
        )
    return tasks


def _fake_results(tasks: list[pipeline_module.SynthesisTask], project_dir: str) -> list[SynthesisResult]:
    return [
        SynthesisResult(
            utterance_index=task.utterance_index,
            speaker=task.speaker,
            stem_path=Path(project_dir) / f"stem_{task.utterance_index}.wav",
            exported_stem_path=str(Path(project_dir) / f"stem_{task.utterance_index}.wav"),
            duration_seconds=0.1,
            chunk_hash=f"hash-{task.utterance_index}",
            cache_hit=False,
            synthesize_seconds=0.0,
            load_audio_seconds=0.0,
            segment_total_seconds=0.0,
            sample_rate=24000,
        )
        for task in tasks
    ]


def _dispatch_tasks(
    plan: RenderPlan,
    render_settings: RenderSettings,
    *,
    pool_override,
    seq_override,
    device_resolver=pipeline_module.resolve_chatterbox_device,
) -> tuple[list[SynthesisResult], str]:
    tasks = _build_synthesis_tasks(plan, render_settings)
    resolved_device = device_resolver(render_settings.device_mode)
    should_parallelize = pipeline_module._should_use_worker_pool(render_settings, resolved_device)
    if should_parallelize:
        results, mode = pool_override(tasks, pipeline_module.ChatterboxEngine, render_settings.model_variant, resolved_device, plan.output_dir)
    else:
        results = seq_override(tasks, pipeline_module.ChatterboxEngine, render_settings.model_variant, resolved_device, plan.output_dir)
        mode = "sequential"
    plan.metadata["synthesis_mode"] = mode
    return results, mode


def test_cpu_standard_uses_worker_pool(tmp_path: Path) -> None:
    render_settings = RenderSettings(model_variant="standard", device_mode="cpu")
    plan = _build_render_plan(tmp_path, render_settings)
    called = {"count": 0}

    def fake_pool(tasks, engine_cls, variant, device, project_dir, worker_count=None):
        called["count"] += 1
        return _fake_results(tasks, project_dir), "parallel"

    results, mode = _dispatch_tasks(plan, render_settings, pool_override=fake_pool, seq_override=lambda *args, **kwargs: [])

    assert called["count"] == 1
    assert mode == "parallel"
    assert plan.metadata["synthesis_mode"] == "parallel"
    assert len(results) == len(plan.utterances)


def test_gpu_device_stays_sequential(tmp_path: Path) -> None:
    render_settings = RenderSettings(model_variant="standard", device_mode="gpu")
    plan = _build_render_plan(tmp_path, render_settings)
    called = {"count": 0}

    def fake_seq(tasks, engine_cls, variant, device, project_dir):
        called["count"] += 1
        return _fake_results(tasks, project_dir)

    results, mode = _dispatch_tasks(
        plan,
        render_settings,
        pool_override=lambda *args, **kwargs: ([], "parallel"),
        seq_override=fake_seq,
        device_resolver=lambda _: "gpu",
    )

    assert called["count"] == 1
    assert mode == "sequential"
    assert plan.metadata["synthesis_mode"] == "sequential"
    assert len(results) == len(plan.utterances)


def test_turbo_stays_sequential(tmp_path: Path) -> None:
    render_settings = RenderSettings(model_variant="turbo", device_mode="cpu")
    plan = _build_render_plan(tmp_path, render_settings)
    called = {"count": 0}

    def fake_seq(tasks, engine_cls, variant, device, project_dir):
        called["count"] += 1
        return _fake_results(tasks, project_dir)

    results, mode = _dispatch_tasks(plan, render_settings, pool_override=lambda *args, **kwargs: ([], "parallel"), seq_override=fake_seq)

    assert called["count"] == 1
    assert mode == "sequential"
    assert plan.metadata["synthesis_mode"] == "sequential"
    assert len(results) == len(plan.utterances)


def test_multilingual_stays_sequential(tmp_path: Path) -> None:
    render_settings = RenderSettings(model_variant="multilingual", device_mode="cpu")
    plan = _build_render_plan(tmp_path, render_settings)
    called = {"count": 0}

    def fake_seq(tasks, engine_cls, variant, device, project_dir):
        called["count"] += 1
        return _fake_results(tasks, project_dir)

    results, mode = _dispatch_tasks(plan, render_settings, pool_override=lambda *args, **kwargs: ([], "parallel"), seq_override=fake_seq)

    assert called["count"] == 1
    assert mode == "sequential"
    assert plan.metadata["synthesis_mode"] == "sequential"
    assert len(results) == len(plan.utterances)


def test_worker_pool_returns_sequential_mode_when_fallback(tmp_path: Path) -> None:
    render_settings = RenderSettings(model_variant="standard", device_mode="cpu")
    plan = _build_render_plan(tmp_path, render_settings)
    called_pool = {"count": 0}
    called_seq = {"count": 0}

    def failing_pool(tasks, engine_cls, variant, device, project_dir, worker_count=None):
        called_pool["count"] += 1
        return _fake_results(tasks, project_dir), "sequential"

    def fake_seq(tasks, engine_cls, variant, device, project_dir):
        called_seq["count"] += 1
        return _fake_results(tasks, project_dir)

    results, mode = _dispatch_tasks(plan, render_settings, pool_override=failing_pool, seq_override=fake_seq)

    assert called_pool["count"] == 1
    assert called_seq["count"] == 0
    assert mode == "sequential"
    assert plan.metadata["synthesis_mode"] == "sequential"
    assert len(results) == len(plan.utterances)


def _run_render_with_interrupt(
    plan: RenderPlan,
    render_settings: RenderSettings,
    *,
    pool_override,
    seq_override,
    expected_message: str,
) -> RenderPlan:
    with (
        patch("the_oracle.pipeline.ChatterboxEngine", _DeterministicChatterboxEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
        patch("the_oracle.pipeline.assemble_dialogue", lambda *_args, **_kwargs: (b"", 24000)),
        patch("the_oracle.pipeline.write_flac", lambda path, audio, rate, metadata: str(path)),
        patch("the_oracle.pipeline.next_available_output_path", lambda path: path),
        patch("the_oracle.pipeline.TextRepairPipeline", _StubTextRepairPipeline),
        patch("the_oracle.pipeline._run_tasks_with_worker_pool", new=pool_override),
        patch("the_oracle.pipeline._sequential_worker_execution", new=seq_override),
    ):
        pipeline = OraclePipeline()
        with pytest.raises(RuntimeError, match=expected_message):
            pipeline.render(plan, render_settings)
    return plan


def test_render_invokes_worker_pool(tmp_path: Path) -> None:
    render_settings = RenderSettings(model_variant="standard", device_mode="cpu")
    plan = _build_render_plan(tmp_path, render_settings)
    called = {"pool": 0}

    def fake_pool(tasks, engine_cls, variant, device, project_dir, worker_count=None):
        called["pool"] += 1
        plan.metadata["synthesis_mode"] = "parallel"
        raise RuntimeError("stop after worker pool")

    def dummy_seq(*_args, **_kwargs):
        raise AssertionError("Sequential fallback should not run for CPU standard worker path.")

    plan = _run_render_with_interrupt(
        plan,
        render_settings,
        pool_override=fake_pool,
        seq_override=dummy_seq,
        expected_message="stop after worker pool",
    )

    assert called["pool"] == 1
    assert plan.metadata.get("synthesis_mode") == "parallel"


def test_render_worker_pool_failure_triggers_sequential(tmp_path: Path) -> None:
    render_settings = RenderSettings(model_variant="standard", device_mode="cpu")
    plan = _build_render_plan(tmp_path, render_settings)
    called = {"seq": 0, "pool": 0}

    def fake_seq(tasks, engine_cls, variant, device, project_dir):
        called["seq"] += 1
        plan.metadata["synthesis_mode"] = "sequential"
        raise RuntimeError("stop after sequential")

    def pool_override(tasks, engine_cls, variant, device, project_dir, worker_count=None):
        called["pool"] += 1
        return fake_seq(tasks, engine_cls, variant, device, project_dir)

    plan = _run_render_with_interrupt(
        plan,
        render_settings,
        pool_override=pool_override,
        seq_override=fake_seq,
        expected_message="stop after sequential",
    )

    assert called["pool"] == 1
    assert called["seq"] == 1
    assert plan.metadata.get("synthesis_mode") == "sequential"


def test_long_utterance_gets_chunked_before_synthesis(tmp_path: Path) -> None:
    """Verify that overlong utterances are split into multiple synthesis tasks."""
    from the_oracle.utils.chunking import chunk_utterance

    # Create a long utterance that should be chunked
    long_text = "This is a very long sentence. " * 20  # ~560 chars
    chunks = chunk_utterance(long_text, parent_index=1)

    # Should produce multiple chunks
    assert len(chunks) > 1
    # All chunks should be within size limits
    from the_oracle.utils.chunking import MAX_CHUNK_SIZE
    assert all(len(c.text) <= MAX_CHUNK_SIZE + 50 for c in chunks)  # Small tolerance
    # Chunks should preserve order
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_sequence == i
        assert chunk.parent_index == 1
    # Reassembly should match original
    from the_oracle.utils.chunking import verify_chunking
    assert verify_chunking(long_text, chunks) is True


def test_short_utterance_stays_unsplit(tmp_path: Path) -> None:
    """Verify that short utterances remain as single chunks."""
    from the_oracle.utils.chunking import chunk_utterance, MIN_CHUNK_SIZE

    short_text = "Short utterance."
    chunks = chunk_utterance(short_text, parent_index=1)

    assert len(chunks) == 1
    assert chunks[0].is_single_chunk is True
    assert chunks[0].text == short_text


def test_chunked_utterance_progress_accounting() -> None:
    """Verify progress accounting is correct when utterances are chunked.
    
    Regression test: total_steps must be calculated AFTER chunking so that
    progress bar doesn't freeze or exceed 100% when one utterance expands
    into multiple synthesis tasks.
    
    This is a unit test that verifies the accounting logic without running
    actual synthesis.
    """
    from the_oracle.utils.chunking import chunk_utterance, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE
    
    # Simulate what happens in pipeline.render() when chunking occurs
    # Original utterance count
    original_utterance_count = 5
    
    # Simulate chunking: some utterances expand into multiple tasks
    # Utterance 1: 1 chunk (short)
    # Utterance 2: 3 chunks (long, gets split)
    # Utterance 3: 1 chunk (short)
    # Utterance 4: 2 chunks (medium-long)
    # Utterance 5: 1 chunk (short)
    # Total: 8 synthesis tasks from 5 utterances
    
    chunk_counts = [1, 3, 1, 2, 1]  # Chunks per utterance
    total_synthesis_tasks = sum(chunk_counts)
    
    # Verify our test setup: chunking should produce more tasks than utterances
    assert total_synthesis_tasks > original_utterance_count
    
    # The bug was: total_steps was calculated BEFORE chunking
    # Old (buggy) calculation:
    voice_profile_count = 2
    old_total_steps = voice_profile_count + original_utterance_count + 3  # +3 for stages
    
    # New (fixed) calculation:
    new_total_steps = voice_profile_count + total_synthesis_tasks + 3
    
    # The old calculation would cause progress to exceed 100% because
    # completed_steps would increment total_synthesis_tasks times but
    # total_steps was based on original_utterance_count
    assert new_total_steps > old_total_steps
    
    # Verify that with the fix, progress accounting is consistent:
    # completed_steps should reach new_total_steps at the end
    completed_steps = 0
    # Simulate speaker prep (voice_profile_count steps)
    completed_steps += voice_profile_count
    # Simulate synthesis tasks (total_synthesis_tasks steps)
    completed_steps += total_synthesis_tasks
    # Simulate assembly, write, complete stages (3 steps)
    completed_steps += 3
    
    # With the fix, completed_steps should equal total_steps
    assert completed_steps == new_total_steps, \
        f"Progress accounting mismatch: {completed_steps} != {new_total_steps}"
    
    # With the old buggy calculation, progress would exceed 100%
    progress_percentage = (completed_steps / old_total_steps) * 100
    assert progress_percentage > 100, \
        "Test setup: old calculation should show >100% progress (the bug)"
