import json
from pathlib import Path
from unittest.mock import patch

import pytest

from the_oracle.project_manifest import ProjectManifestError, build_saved_project, load_project_manifest, save_project_manifest
from the_oracle.pipeline import OraclePipeline, RenderSettings, SpeakerSettings
from the_oracle.smoke import _DeterministicChatterboxEngine, _SmokeEmotionClassifier, _write_reference
from the_oracle.models.project import RenderPlan, VoiceSettings


def _sample_project(tmp_path: Path):
    source = tmp_path / "dialogue.txt"
    source.write_text(
        "Speaker A: The Oracle is online.\n"
        "Speaker B: Confirm the signal path.\n",
        encoding="utf-8",
    )
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    settings = RenderSettings(
        correction_mode="moderate",
        model_variant="standard",
        language="en",
        export_stems=True,
        loudness_preset="off",
        pause_between_turns_ms=120,
        crossfade_ms=10,
    )
    voice = VoiceSettings(variant="standard", language="en")
    speakers = {
        "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=voice),
        "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=voice),
    }
    with patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier):
        pipeline = OraclePipeline()
        plan = pipeline.prepare_plan(source, tmp_path / "output", speakers, settings)
    plan.metadata["artist"] = "Oracle QA"
    return pipeline, plan, settings, speakers


@pytest.mark.slow  # prepare_plan runs the full repair pipeline (LanguageTool download)
def test_project_manifest_round_trip_preserves_key_fields(tmp_path: Path) -> None:
    _pipeline, plan, settings, speakers = _sample_project(tmp_path)
    manifest_path = tmp_path / "project.json"
    save_project_manifest(manifest_path, build_saved_project(plan, settings, speakers))

    loaded = load_project_manifest(manifest_path)

    assert loaded.title == plan.title
    assert loaded.artist == "Oracle QA"
    assert loaded.input_path == plan.source_path
    assert loaded.output_path == plan.output_dir
    assert loaded.engine == "chatterbox"
    assert loaded.model_variant == "standard"
    assert loaded.render_settings.correction_mode == settings.correction_mode
    assert loaded.render_settings.device_mode == "cpu"
    assert loaded.render_settings.loudness_preset == settings.loudness_preset
    assert loaded.speaker_settings["A"].reference_path == speakers["A"].reference_path
    assert loaded.speaker_settings["B"].reference_path == speakers["B"].reference_path
    assert loaded.speaker_settings["A"].voice_settings["cfg_weight"] == speakers["A"].voice_settings.cfg_weight
    assert loaded.speaker_settings["A"].voice_settings["pause_ms"] == speakers["A"].voice_settings.pause_ms


@pytest.mark.slow  # prepare_plan runs the full repair pipeline (LanguageTool download)
def test_modified_utterance_state_survives_round_trip(tmp_path: Path) -> None:
    _pipeline, plan, settings, speakers = _sample_project(tmp_path)
    plan.utterances[0].speaker = "B"
    plan.utterances[0].repaired_text = "Manual repair."
    plan.utterances[0].emotion = "joy"
    plan.utterances[0].manual_speaker_override = True
    plan.utterances[0].manual_text_override = True
    plan.utterances[0].manual_emotion_override = True

    manifest_path = tmp_path / "project.json"
    save_project_manifest(manifest_path, build_saved_project(plan, settings, speakers))

    loaded = load_project_manifest(manifest_path)
    utterance = loaded.plan.utterances[0]
    assert utterance.speaker == "B"
    assert utterance.repaired_text == "Manual repair."
    assert utterance.emotion == "joy"
    assert utterance.manual_speaker_override is True
    assert utterance.manual_text_override is True
    assert utterance.manual_emotion_override is True


def test_manifest_round_trips_audio_cpp_knobs(tmp_path: Path) -> None:
    plan = RenderPlan(
        title="t",
        source_path="in.txt",
        output_dir=str(tmp_path / "output"),
        engine="chatterbox",
        correction_mode="moderate",
    )
    settings = RenderSettings(
        inference_backend="vulkan",
        audio_cpp_device=2,
        audio_cpp_threads=6,
        audio_cpp_max_batch=16,
    )
    speakers = {
        "A": SpeakerSettings(reference_path="a.wav", voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path="b.wav", voice_settings=VoiceSettings()),
    }

    manifest_path = tmp_path / "project.json"
    save_project_manifest(manifest_path, build_saved_project(plan, settings, speakers))
    loaded = load_project_manifest(manifest_path)

    assert loaded.render_settings.inference_backend == "vulkan"
    assert loaded.render_settings.audio_cpp_device == 2
    assert loaded.render_settings.audio_cpp_threads == 6
    assert loaded.render_settings.audio_cpp_max_batch == 16


def test_manifest_without_audio_cpp_knobs_loads_with_defaults(tmp_path: Path) -> None:
    """Manifests saved before the knobs existed (no keys in render_settings)
    must still load, with the new fields defaulting to None."""
    plan = RenderPlan(
        title="t",
        source_path="in.txt",
        output_dir=str(tmp_path / "output"),
        engine="chatterbox",
        correction_mode="moderate",
    )
    speakers = {
        "A": SpeakerSettings(reference_path="a.wav", voice_settings=VoiceSettings()),
        "B": SpeakerSettings(reference_path="b.wav", voice_settings=VoiceSettings()),
    }
    payload = build_saved_project(plan, RenderSettings(), speakers).to_dict()
    payload["render_settings"].pop("audio_cpp_device", None)
    payload["render_settings"].pop("audio_cpp_threads", None)
    payload["render_settings"].pop("audio_cpp_max_batch", None)

    manifest_path = tmp_path / "project.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_project_manifest(manifest_path)

    assert loaded.render_settings.audio_cpp_device is None
    assert loaded.render_settings.audio_cpp_threads is None
    assert loaded.render_settings.audio_cpp_max_batch is None


def test_loading_incomplete_manifest_fails_clearly(tmp_path: Path) -> None:
    manifest_path = tmp_path / "broken.json"
    manifest_path.write_text(json.dumps({"manifest_version": 1, "title": "broken"}), encoding="utf-8")

    try:
        load_project_manifest(manifest_path)
    except ProjectManifestError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("Expected ProjectManifestError for incomplete manifest")


@pytest.mark.slow  # prepare_plan runs the full repair pipeline (LanguageTool download)
def test_loaded_project_renders_with_deterministic_engine(tmp_path: Path) -> None:
    pipeline, plan, settings, speakers = _sample_project(tmp_path)
    manifest_path = tmp_path / "project.json"
    save_project_manifest(manifest_path, build_saved_project(plan, settings, speakers))
    loaded = load_project_manifest(manifest_path)

    with (
        patch("the_oracle.pipeline.ChatterboxEngine", _DeterministicChatterboxEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
    ):
        output_path = pipeline.render(loaded.plan, loaded.render_settings)

    assert output_path.exists()
