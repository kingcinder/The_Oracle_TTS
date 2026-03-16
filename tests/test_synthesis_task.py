from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

from the_oracle.models.cache import ProjectCache
from the_oracle.models.project import Utterance, VoiceSettings
from the_oracle.pipeline import (
    SynthesisTask,
    synthesize_task,
    _run_tasks_with_worker_pool,
    _worker_initialize,
    _worker_process_task,
    _worker_reset,
)
from the_oracle.smoke import _DeterministicChatterboxEngine, _write_reference
from the_oracle.utils.hashing import build_chunk_hash


def _prepare_synthesis_task_helpers(
    tmp_path: Path,
    speaker: str = "A",
    base_text: str = "Hermetic test utterance.",
) -> Tuple[
    ProjectCache,
    _DeterministicChatterboxEngine,
    ChatterboxConditioning,
    Callable[[int, str | None, bool], SynthesisTask],
]:
    project_cache = ProjectCache(tmp_path / "project")
    engine = _DeterministicChatterboxEngine()
    reference = _write_reference(tmp_path / f"{speaker}_ref.wav", 220.0)
    cached_reference = engine.prepare_reference(project_cache, speaker, str(reference))
    voice_settings = VoiceSettings()
    conditioning = engine.prepare_conditioning(project_cache, speaker, cached_reference, voice_settings)

    def build_task(
        utterance_index: int,
        source_index: int | None = None,
        text: str | None = None,
        export_stems: bool = True,
    ) -> SynthesisTask:
        actual_text = text or base_text
        utterance = Utterance(
            index=utterance_index,
            original_text=actual_text,
            repaired_text=actual_text,
            speaker=speaker,
        )
        return SynthesisTask(
            utterance_index=utterance_index,
            source_index=source_index if source_index is not None else utterance_index,
            speaker=speaker,
            text=actual_text,
            reference_path=Path(reference),
            reference_audio_hash=cached_reference.original_hash,
            voice_settings=voice_settings,
            model_variant="standard",
            device_mode="cpu",
            export_stems=export_stems,
        )

    return project_cache, engine, conditioning, build_task


def test_synthesize_task_cache_behavior(tmp_path: Path) -> None:
    project_cache, engine, conditioning, build_task = _prepare_synthesis_task_helpers(tmp_path)

    first_task = build_task(utterance_index=1, export_stems=True)
    first_result = synthesize_task(first_task, engine, conditioning, project_cache)
    expected_chunk = build_chunk_hash(
        speaker=first_task.speaker,
        repaired_text=first_task.text,
        engine_key=f"chatterbox:{first_task.model_variant}",
        engine_params=first_task.voice_settings.to_dict(),
        engine_version=engine.engine_version,
        reference_audio_hash=first_task.reference_audio_hash,
    )

    exported_path = Path(first_result.exported_stem_path)

    assert first_result.chunk_hash == expected_chunk
    assert first_result.cache_hit is False
    assert first_result.stem_path.exists()
    assert exported_path.exists()
    assert first_result.sample_rate == engine.sample_rate
    assert first_result.duration_seconds > 0.0

    second_task = build_task(utterance_index=2, export_stems=True)
    second_result = synthesize_task(second_task, engine, conditioning, project_cache)

    assert second_result.cache_hit is True
    assert second_result.chunk_hash == first_result.chunk_hash
    assert second_result.stem_path == first_result.stem_path
    assert Path(second_result.exported_stem_path).exists()


def test_worker_process_task(tmp_path: Path) -> None:
    project_cache, engine, conditioning, build_task = _prepare_synthesis_task_helpers(tmp_path)
    _worker_initialize(_DeterministicChatterboxEngine, "standard", "cpu", str(project_cache.project_dir))
    try:
        task = build_task(utterance_index=1, export_stems=True)
        result = _worker_process_task(task)
        assert result.stem_path.exists()
        assert result.cache_hit is False
        repeated = _worker_process_task(build_task(utterance_index=2, export_stems=False))
        assert repeated.cache_hit is True
    finally:
        _worker_reset()


def test_worker_pool_dispatch(tmp_path: Path) -> None:
    project_cache, engine, conditioning, build_task = _prepare_synthesis_task_helpers(tmp_path)
    tasks = [build_task(utterance_index=i, export_stems=(i == 1)) for i in range(1, 4)]
    results, mode = _run_tasks_with_worker_pool(tasks, _DeterministicChatterboxEngine, "standard", "cpu", str(project_cache.project_dir), worker_count=2)
    assert mode in {"parallel", "sequential"}
    assert len(results) == 3
    assert all(results[i].utterance_index == i + 1 for i in range(3))


def test_synthesize_task_chunk_changes_with_text(tmp_path: Path) -> None:
    project_cache, engine, conditioning, build_task = _prepare_synthesis_task_helpers(tmp_path)

    base_result = synthesize_task(build_task(utterance_index=1, text="First line.", export_stems=False), engine, conditioning, project_cache)
    alternate_result = synthesize_task(build_task(utterance_index=2, text="Different line.", export_stems=False), engine, conditioning, project_cache)

    assert base_result.chunk_hash != alternate_result.chunk_hash
    assert base_result.stem_path != alternate_result.stem_path
    assert alternate_result.cache_hit is False
