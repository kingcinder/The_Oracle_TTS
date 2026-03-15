from pathlib import Path

import pytest

from the_oracle.gui_settings import (
    GUISettingsError,
    list_templates,
    load_gui_settings,
    load_recent_reference_paths,
    load_template,
    remember_recent_reference_path,
    save_gui_settings,
    save_template,
)


def _payload() -> dict:
    return {
        "version": 1,
        "name": "Oracle Template",
        "device_mode": "cpu",
        "project": {
            "model_variant": "standard",
            "language": "en",
            "correction_mode": "conservative",
            "loudness_preset": "light",
            "pause_between_turns_ms": 180,
            "crossfade_ms": 20,
            "output_dir": "/tmp/output",
            "output_filename": "oracle_render.flac",
        },
        "speakers": {
            "A": {
                "reference_path": "/tmp/a.wav",
                "voice_settings": {"cfg_weight": 0.5},
                "emotion_reference_paths": {},
            },
            "B": {
                "reference_path": "/tmp/b.wav",
                "voice_settings": {"cfg_weight": 0.6},
                "emotion_reference_paths": {},
            },
        },
    }


def test_gui_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    save_gui_settings(path, _payload())

    loaded = load_gui_settings(path)

    assert loaded["project"]["model_variant"] == "standard"
    assert loaded["project"]["output_dir"] == "/tmp/output"
    assert loaded["project"]["output_filename"] == "oracle_render.flac"
    assert loaded["speakers"]["A"]["reference_path"] == "/tmp/a.wav"
    assert loaded["speakers"]["B"]["voice_settings"]["cfg_weight"] == 0.6


def test_gui_template_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_template("Oracle Template", _payload())

    assert list_templates() == ["Oracle_Template"]
    assert load_template("Oracle Template")["name"] == "Oracle Template"


def test_incomplete_gui_settings_fail_clearly(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"version": 1, "project": {}}', encoding="utf-8")

    with pytest.raises(GUISettingsError, match="missing required fields"):
        load_gui_settings(path)


def test_recent_reference_paths_are_mru_and_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for index in range(12):
        remember_recent_reference_path(f"/tmp/ref_{index}.wav")

    recent = load_recent_reference_paths()

    assert len(recent) == 10
    assert recent[0] == "/tmp/ref_11.wav"


def test_legacy_gui_settings_gain_default_output_location_fields(tmp_path: Path) -> None:
    path = tmp_path / "legacy_settings.json"
    payload = _payload()
    payload["project"].pop("output_dir")
    payload["project"].pop("output_filename")
    save_gui_settings(path, payload)

    loaded = load_gui_settings(path)

    assert loaded["project"]["output_dir"] == ""
    assert loaded["project"]["output_filename"] == ""
