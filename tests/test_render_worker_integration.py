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
