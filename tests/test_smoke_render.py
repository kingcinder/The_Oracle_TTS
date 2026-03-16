import json
from pathlib import Path
from unittest.mock import patch

import soundfile as sf
import pytest

from the_oracle.models.project import VoiceProfile
from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import OraclePipeline, RenderProgress, RenderSettings, SpeakerSettings
from the_oracle.smoke import _DeterministicChatterboxEngine, _SmokeEmotionClassifier, _write_reference, run_deterministic_smoke_render


def test_deterministic_smoke_render_runs_end_to_end(tmp_path: Path) -> None:
    result = run_deterministic_smoke_render(tmp_path, source_format="txt")

    assert result.output_path.exists()
    assert result.second_output_path.exists()
    assert result.second_output_path.name == "smoke_dialogue (1).flac"
    assert result.render_plan_path.exists()
    assert result.stem_count == 4
    assert result.cache_reused_on_second_pass is True

    audio, sample_rate = sf.read(result.output_path, always_2d=False)
    assert sample_rate == 24000
    assert len(audio) > 1000

    render_plan = json.loads(result.render_plan_path.read_text(encoding="utf-8"))
    render_timings = json.loads((result.project_dir / "logs" / "render_timings.json").read_text(encoding="utf-8"))
    render_trace = (result.project_dir / "logs" / "render_trace.log").read_text(encoding="utf-8")
    utterance_entries = [entry for entry in render_timings["entries"] if entry["type"] == "utterance"]
    assert render_plan["engine"] == "chatterbox"
    assert render_plan["metadata"]["model_variant"] == "standard"
    assert render_plan["metadata"]["cache_reused_on_second_pass"] == "True"
    assert render_plan["metadata"]["watermark"] == "Perth watermark embedded by Chatterbox"
    assert render_timings["summary"]["utterance_count"] == 4
    assert render_timings["summary"]["segment_count"] == 4
    assert render_timings["summary"]["join_count"] == 3
    assert len(render_timings["segments"]) == 4
    assert len(render_timings["joins"]) == 3
    assert utterance_entries[0]["cache_stem_path"].endswith(".wav")
    assert utterance_entries[0]["exported_stem_path"].endswith(".wav")
    assert render_timings["segments"][0]["content_start_seconds"] == 0.0
    assert render_timings["output"]["path"].endswith(".flac")
    assert "segment 1/4" in render_trace


def test_deterministic_markdown_smoke_render_runs_end_to_end(tmp_path: Path) -> None:
    result = run_deterministic_smoke_render(tmp_path, source_format="md")

    assert result.output_path.exists()
    assert result.second_output_path.exists()
    assert result.second_output_path.name == "smoke_dialogue (1).flac"
    assert result.render_plan_path.exists()
    assert result.stem_count == 4
    assert result.cache_reused_on_second_pass is True

    audio, sample_rate = sf.read(result.output_path, always_2d=False)
    assert sample_rate == 24000
    assert len(audio) > 1000

    render_plan = json.loads(result.render_plan_path.read_text(encoding="utf-8"))
    render_timings = json.loads((result.project_dir / "logs" / "render_timings.json").read_text(encoding="utf-8"))
    render_trace = (result.project_dir / "logs" / "render_trace.log").read_text(encoding="utf-8")
    assert render_plan["engine"] == "chatterbox"
    assert render_plan["metadata"]["model_variant"] == "standard"
    assert render_plan["metadata"]["cache_reused_on_second_pass"] == "True"
    assert render_timings["summary"]["utterance_count"] == 4
    assert render_timings["summary"]["segment_count"] == 4
    assert render_timings["summary"]["join_count"] == 3
    assert len(render_timings["segments"]) == 4
    assert len(render_timings["joins"]) == 3
    assert render_timings["joins"][0]["left_stem_path"].endswith(".wav")
    assert render_timings["joins"][0]["right_stem_path"].endswith(".wav")
    assert "output | path=" in render_trace


def test_render_progress_reports_stage_updates(tmp_path: Path) -> None:
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(
        "Speaker A: The Oracle is online.\n"
        "Speaker B: Confirm the signal path.\n",
        encoding="utf-8",
    )
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    speaker_settings = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=VoiceSettings()),
    }
    render_settings = RenderSettings(
        correction_mode="moderate",
        model_variant="standard",
        language="en",
        export_stems=True,
        loudness_preset="off",
        pause_between_turns_ms=120,
        crossfade_ms=10,
    )
    events: list[RenderProgress] = []

    with (
        patch("the_oracle.pipeline.ChatterboxEngine", _DeterministicChatterboxEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
    ):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, render_settings)
        output_path = pipeline.render(plan, render_settings, progress_callback=events.append)

    assert output_path.exists()
    assert events
    assert events[-1].stage == "Complete"
    assert events[-1].current_step == events[-1].total_steps
    assert any(event.stage == "Rendering segment" and event.current_segment == 1 for event in events)


def test_render_preview_creates_preview_files_for_both_speakers(tmp_path: Path) -> None:
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(
        "Speaker A: First preview.\n"
        "Speaker B: Second preview.\n",
        encoding="utf-8",
    )
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    speaker_settings = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=VoiceSettings()),
    }
    render_settings = RenderSettings(model_variant="standard", language="en", loudness_preset="off")

    with (
        patch("the_oracle.pipeline.ChatterboxEngine", _DeterministicChatterboxEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
    ):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, render_settings)
        preview_a = pipeline.render_preview(plan.utterances[0], plan.voice_profiles["A"], "standard")
        preview_b = pipeline.render_preview(plan.utterances[1], plan.voice_profiles["B"], "standard")

    assert preview_a.exists()
    assert preview_a.parent.name == "previews"
    assert preview_a.name == "preview_A_0000.wav"
    assert preview_b.exists()
    assert preview_b.parent.name == "previews"
    assert preview_b.name == "preview_B_0001.wav"


def test_render_preview_rejects_blank_reference_path_before_reading_dot(tmp_path: Path) -> None:
    utterance = type("PreviewUtterance", (), {"speaker": "A", "index": 0, "engine_settings": VoiceSettings(), "text_for_tts": lambda self: "Preview text"})()
    profile = VoiceProfile(name="Speaker A", speaker="A", neutral_reference=Path(""), engine_params=VoiceSettings())

    pipeline = OraclePipeline()

    with pytest.raises(ValueError, match="has no reference audio configured"):
        pipeline.render_preview(utterance, profile, "standard")


def test_render_preview_reports_honest_stage_progress(tmp_path: Path) -> None:
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text("Speaker A: First preview.\n", encoding="utf-8")
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_settings = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path=str(speaker_a), voice_settings=VoiceSettings()),
    }
    events: list[RenderProgress] = []

    with (
        patch("the_oracle.pipeline.ChatterboxEngine", _DeterministicChatterboxEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
    ):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, RenderSettings(model_variant="standard"))
        preview_path = pipeline.render_preview(
            plan.utterances[0],
            plan.voice_profiles["A"],
            "standard",
            progress_callback=events.append,
        )

    assert preview_path.exists()
    assert [event.stage for event in events] == [
        "Loading model",
        "Preparing reference",
        "Preparing conditioning",
        "Generating preview",
        "Complete",
    ]
    assert events[-1].current_step == events[-1].total_steps == 4
