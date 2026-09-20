"""Ownership tests for the settings schema extraction (models/settings.py).

RenderSettings/SpeakerSettings moved out of pipeline.py so settings-adjacent
modules (gui_settings, project_manifest) can import the schema without the
engine surface — the wrong dependency direction the WidgetSnapshot design
called out. These tests pin the move: one owner, verbatim behavior, and the
historical import path still resolving to the same objects.
"""

from __future__ import annotations

import dataclasses
import importlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SETTINGS_CLASSES = ("RenderSettings", "SpeakerSettings")


def _owner():
    return importlib.import_module("the_oracle.models.settings")


def test_owner_defines_both_settings_classes() -> None:
    owner = _owner()
    for name in SETTINGS_CLASSES:
        assert dataclasses.is_dataclass(getattr(owner, name))


def test_pipeline_reexports_are_the_same_objects() -> None:
    owner = _owner()
    pipeline = importlib.import_module("the_oracle.pipeline")
    for name in SETTINGS_CLASSES:
        assert getattr(pipeline, name) is getattr(owner, name), name


def test_settings_modules_import_from_the_owner_not_the_engine() -> None:
    """The point of the move: gui_settings/project_manifest must not go through pipeline."""
    for module_name in ("the_oracle.gui_settings", "the_oracle.project_manifest"):
        module = importlib.import_module(module_name)
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "from the_oracle.models.settings import" in source, module_name
        assert "pipeline import" not in source, module_name


def test_settings_module_never_imports_the_engine() -> None:
    """The owner module must stay engine-free: importing it must not import pipeline."""
    script = (
        "import sys; import the_oracle.models.settings; "
        "assert 'the_oracle.pipeline' not in sys.modules; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_render_settings_validation_survived_the_move() -> None:
    """Verbatim behavior: __post_init__ validation and normalization."""
    RenderSettings = _owner().RenderSettings
    assert RenderSettings().correction_mode == "moderate"
    try:
        RenderSettings(inference_backend="bogus")
    except ValueError as error:
        assert "Unsupported inference backend" in str(error)
    else:
        raise AssertionError("bogus backend accepted")
    for field_name in ("audio_cpp_device", "audio_cpp_threads", "audio_cpp_timeout", "audio_cpp_max_batch"):
        try:
            RenderSettings(**{field_name: -1})
        except ValueError:
            pass
        else:
            raise AssertionError(f"{field_name}=-1 accepted")
    try:
        RenderSettings(seed=-1)
    except ValueError as error:
        assert "seed" in str(error)
    else:
        raise AssertionError("seed=-1 accepted")


def test_speaker_settings_defaults_survived_the_move() -> None:
    SpeakerSettings = _owner().SpeakerSettings
    settings = SpeakerSettings(reference_path="x")
    assert settings.blend_weight == 0.5
    assert settings.blend_mode == "mix"
    assert settings.blend_references == []
    assert settings.emotion_reference_paths == {}
