import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import soundfile as sf
import pytest
import yaml

pytestmark = pytest.mark.slow

from the_oracle.models.project import VoiceProfile
from the_oracle.models.project import VoiceSettings
from the_oracle.models.settings import RenderSettings, SpeakerSettings; from the_oracle.pipeline import OraclePipeline, RenderProgress
from the_oracle.smoke import (
    SmokeRenderResult,
    _DeterministicChatterboxEngine,
    _SmokeEmotionClassifier,
    _write_reference,
    run_deterministic_smoke_render,
    smoke_output_problem,
)


def test_deterministic_smoke_render_runs_end_to_end(tmp_path: Path) -> None:
    result = run_deterministic_smoke_render(tmp_path, source_format="txt")

    assert result.output_path.exists()
    assert result.render_plan_path.exists()
    assert result.stem_count == 4
    assert result.cache_reused_on_second_pass is True
    # The second pass is a cache-reuse probe, and the export policy never
    # overwrites an existing render, so the probe must not leave a versioned
    # duplicate ("smoke_dialogue (1).flac") of the real output beside it.
    assert sorted(path.name for path in result.project_dir.glob("*.flac")) == ["smoke_dialogue.flac"]

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
    # The verdict above comes from comparing the two plans. Prove it empirically
    # as well: a pass that reuses the cache synthesizes nothing, so every
    # utterance must be a cache hit.
    assert [entry["cache_hit"] for entry in utterance_entries] == [True] * 4
    assert [entry["synthesize_seconds"] for entry in utterance_entries] == [0.0] * 4
    assert utterance_entries[0]["cache_stem_path"].endswith(".wav")
    assert utterance_entries[0]["exported_stem_path"].endswith(".wav")
    assert render_timings["segments"][0]["content_start_seconds"] == 0.0
    assert render_timings["output"]["path"].endswith(".flac")
    assert "segment 1/4" in render_trace


def test_deterministic_markdown_smoke_render_runs_end_to_end(tmp_path: Path) -> None:
    result = run_deterministic_smoke_render(tmp_path, source_format="md")

    assert result.output_path.exists()
    assert result.render_plan_path.exists()
    assert result.stem_count == 4
    assert result.cache_reused_on_second_pass is True
    assert sorted(path.name for path in result.project_dir.glob("*.flac")) == ["smoke_dialogue.flac"]

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
    utterances = [entry for entry in render_timings["entries"] if entry["type"] == "utterance"]
    assert [entry["cache_hit"] for entry in utterances] == [True] * 4
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
        pipeline = OraclePipeline(
            use_transformers=False,
            use_language_tool=False,
            use_punctuation_model=False,
        )
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
        pipeline = OraclePipeline(
            use_transformers=False,
            use_language_tool=False,
            use_punctuation_model=False,
        )
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

    pipeline = OraclePipeline(
        use_transformers=False,
        use_language_tool=False,
        use_punctuation_model=False,
    )

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
        pipeline = OraclePipeline(
            use_transformers=False,
            use_language_tool=False,
            use_punctuation_model=False,
        )
        plan = pipeline.prepare_plan(dialogue, tmp_path / "output", speaker_settings, RenderSettings(model_variant="standard"))
        preview_result = pipeline.render_preview(
            plan.utterances[0],
            plan.voice_profiles["A"],
            "standard",
            progress_callback=events.append,
        )

    assert preview_result.exists()
    assert [event.stage for event in events] == [
        "Loading model",
        "Preparing reference",
        "Preparing conditioning",
        "Generating preview",
        "Complete",
    ]
    assert events[-1].current_step == events[-1].total_steps == 4


# --- the output-verdict policy: one owner, applied by every caller ------------


def _script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_render.py"
    spec = importlib.util.spec_from_file_location("oracle_smoke_render_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(output_path: Path, source_format: str = "txt") -> SmokeRenderResult:
    return SmokeRenderResult(
        source_format=source_format,
        output_path=output_path,
        project_dir=output_path.parent,
        cache_reused_on_second_pass=True,
        stem_count=4,
        render_plan_path=output_path.parent / "plan.json",
        dialogue_path=output_path.parent / "dialogue.txt",
    )


def test_smoke_output_problem_reports_missing_and_empty_outputs(tmp_path: Path) -> None:
    missing = tmp_path / "gone.flac"
    assert "no output" in (smoke_output_problem(_result(missing)) or "").lower()

    empty = tmp_path / "empty.flac"
    empty.write_bytes(b"")
    assert "empty" in (smoke_output_problem(_result(empty)) or "").lower()

    real = tmp_path / "real.flac"
    real.write_bytes(b"RIFF....")
    assert smoke_output_problem(_result(real)) is None


def test_smoke_render_script_exits_zero_only_when_outputs_are_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The standalone runner verifies what the render claims, not just that
    it returned: a render that returns without producing a non-empty output
    file fails the run with the reason, like the doctor's wrapper does."""
    module = _script_module()
    real = tmp_path / "real.flac"
    real.write_bytes(b"RIFF....")

    monkeypatch.setattr(
        module,
        "run_deterministic_smoke_render",
        lambda output_root, source_format="txt": _result(real, source_format),
    )
    assert module.main(["--output-root", str(tmp_path)]) == 0

    # The healthy md leg, but a txt leg whose output vanished after render.
    def _half_broken(output_root, source_format="txt"):
        if source_format == "txt":
            return _result(tmp_path / "vanished.flac", source_format)
        return _result(real, source_format)

    monkeypatch.setattr(module, "run_deterministic_smoke_render", _half_broken)
    assert module.main(["--output-root", str(tmp_path)]) == 1
    assert "no output" in capsys.readouterr().err.lower()
    # The healthy leg's report still printed, so the failure is diagnosable.
    assert "md" in capsys.readouterr().out


def test_the_output_verdict_policy_has_one_owner() -> None:
    repo = Path(__file__).resolve().parents[1]
    owner = (repo / "src" / "the_oracle" / "smoke.py").read_text(encoding="utf-8")
    doctor = (repo / "scripts" / "doctor.py").read_text(encoding="utf-8")
    runner = (repo / "scripts" / "smoke_render.py").read_text(encoding="utf-8")
    # The verdict literals live exactly once, in the owner (vacuity guard:
    # a rename must not make this scan pass by matching nothing).
    assert owner.count("produced no output file") == 1
    assert "produced no output file" not in doctor
    assert "produced no output file" not in runner
    # Both callers delegate to the owner instead of re-implementing it.
    assert "smoke_output_problem(result)" in doctor
    assert "smoke_output_problem(result)" in runner


# --- CI: the smoke render is a named step on every push -----------------------


def test_workflow_runs_the_deterministic_smoke_render_on_both_oses() -> None:
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = {step.get("name"): step for step in workflow["jobs"]["test"]["steps"]}
    linux = steps["Deterministic Smoke Render (Linux)"]
    windows = steps["Deterministic Smoke Render (Windows)"]
    assert ".venv/bin/python" in linux["run"]
    assert "scripts/smoke_render.py" in linux["run"]
    assert ".venv\\Scripts\\python.exe" in windows["run"]
    assert "scripts/smoke_render.py" in windows["run"]
    # Gated only by operating system, so the step runs on every push and
    # pull_request -- the `on:` block needs no per-step opt-in.
    assert linux["if"] == "runner.os == 'Linux'"
    assert windows["if"] == "runner.os == 'Windows'"


# ---------------------------------------------------------------------------
# The standalone runner's own verdict, and where its policy lives
# ---------------------------------------------------------------------------


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_render.py"
WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _load_smoke_render_script():
    spec = importlib.util.spec_from_file_location("oracle_smoke_render_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _result(output_path: Path, source_format: str = "txt") -> SmokeRenderResult:
    return SmokeRenderResult(
        source_format=source_format,
        output_path=output_path,
        project_dir=output_path.parent / f"render_project_{source_format}",
        cache_reused_on_second_pass=True,
        stem_count=4,
        render_plan_path=output_path.parent / "render_plan.json",
        dialogue_path=output_path.parent / f"smoke_dialogue.{source_format}",
    )


def test_smoke_output_problem_flags_missing_and_empty_outputs(tmp_path: Path) -> None:
    healthy = tmp_path / "healthy.flac"
    healthy.write_bytes(b"RIFF-fake-audio")
    assert smoke_output_problem(_result(healthy)) is None

    missing = smoke_output_problem(_result(tmp_path / "missing.flac"))
    assert missing is not None and "no output file" in missing

    empty = tmp_path / "empty.flac"
    empty.write_bytes(b"")
    empty_problem = smoke_output_problem(_result(empty))
    assert empty_problem is not None and "empty" in empty_problem


def test_smoke_render_script_exits_zero_only_when_outputs_are_usable(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A render can return without raising yet leave no usable audio; the
    runner's exit code must catch that, not just a raised exception."""
    module = _load_smoke_render_script()
    healthy = tmp_path / "healthy.flac"
    healthy.write_bytes(b"RIFF-fake-audio")

    healthy_results = [_result(healthy, "txt"), _result(healthy, "md")]
    monkeypatch.setattr(
        module,
        "run_deterministic_smoke_render",
        lambda root, source_format="txt": healthy_results[0 if source_format == "txt" else 1],
    )
    assert module.main(["--output-root", str(tmp_path)]) == 0
    capsys.readouterr()  # discard the human-readable run
    assert module.main(["--json", "--output-root", str(tmp_path)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert [entry["source_format"] for entry in parsed] == ["txt", "md"]

    missing = tmp_path / "missing.flac"
    broken_results = [_result(missing, "txt"), _result(healthy, "md")]
    monkeypatch.setattr(
        module,
        "run_deterministic_smoke_render",
        lambda root, source_format="txt": broken_results[0 if source_format == "txt" else 1],
    )
    assert module.main(["--output-root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "no output file" in err and "txt" in err
    assert "md" not in err  # only the failing leg is named


def test_workflow_runs_the_deterministic_smoke_render_on_both_operating_systems() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = {step.get("name"): step for step in workflow["jobs"]["test"]["steps"]}
    linux = steps["Deterministic Smoke Render (Linux)"]
    windows = steps["Deterministic Smoke Render (Windows)"]
    assert "scripts/smoke_render.py" in linux["run"]
    assert ".venv/bin/python" in linux["run"]
    assert "scripts/smoke_render.py" in windows["run"]
    assert ".venv\\Scripts\\python.exe" in windows["run"]
    # Gated only by operating system, so the step runs on every push and
    # pull request; a job-level `if:` could silently remove it from pushes.
    assert linux["if"] == "runner.os == 'Linux'"
    assert windows["if"] == "runner.os == 'Windows'"


def test_the_smoke_output_verdict_policy_has_one_owner() -> None:
    repo = Path(__file__).resolve().parents[1]
    smoke_src = (repo / "src" / "the_oracle" / "smoke.py").read_text(encoding="utf-8")
    doctor_src = (repo / "scripts" / "doctor.py").read_text(encoding="utf-8")
    script_src = (repo / "scripts" / "smoke_render.py").read_text(encoding="utf-8")
    # The reason strings live once, in the owner; the count is its own
    # vacuity guard -- a rename that empties the owner fails here too.
    assert smoke_src.count("produced no output file") == 1
    assert "produced no output file" not in doctor_src
    assert "produced no output file" not in script_src
    # And both callers delegate to the owner rather than re-deriving it.
    assert "smoke_output_problem(result)" in doctor_src
    assert "smoke_output_problem(result)" in script_src
