from pathlib import Path

import pytest

from the_oracle.gui_settings import (
    GUISettingsError,
    list_templates,
    load_app_settings,
    load_gui_settings,
    load_recent_reference_paths,
    load_template,
    remember_recent_reference_path,
    save_app_settings,
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
            "correction_mode": "moderate",
            "loudness_preset": "light",
            "pause_between_turns_ms": 180,
            "crossfade_ms": 20,
            "target_wpm": 150.0,
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
    assert loaded["project"]["target_wpm"] == pytest.approx(150.0)
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
    assert Path(recent[0]).as_posix() == "/tmp/ref_11.wav"


def test_legacy_gui_settings_gain_default_output_location_fields(tmp_path: Path) -> None:
    path = tmp_path / "legacy_settings.json"
    payload = _payload()
    payload["project"].pop("output_dir")
    payload["project"].pop("output_filename")
    save_gui_settings(path, payload)

    loaded = load_gui_settings(path)

    assert loaded["project"]["output_dir"] == ""
    assert loaded["project"]["output_filename"] == ""


def test_gui_settings_default_inference_backend_to_pytorch(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    save_gui_settings(path, _payload())

    loaded = load_gui_settings(path)

    assert loaded["project"]["inference_backend"] == "pytorch"


def test_gui_settings_preserve_vulkan_inference_backend(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    payload = _payload()
    payload["project"]["inference_backend"] = "vulkan"
    save_gui_settings(path, payload)

    loaded = load_gui_settings(path)

    assert loaded["project"]["inference_backend"] == "vulkan"


def test_app_settings_defaults_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    settings = load_app_settings()

    assert settings["remember_backend"] is True
    assert settings["inference_backend"] == "pytorch"
    assert settings["audio_cpp_device"] is None
    assert settings["audio_cpp_cli"] == ""
    assert settings["audio_cpp_model"] == ""


def test_app_settings_round_trip_persists_backend_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    payload = {
        "remember_backend": True,
        "inference_backend": "vulkan",
        "audio_cpp_device": 0,
        "audio_cpp_threads": 6,
        "audio_cpp_timeout": 120,
        "audio_cpp_max_batch": 16,
        "audio_cpp_cli": "/opt/audiocpp/bin/audiocpp_cli",
        "audio_cpp_model": "/models/chatterbox_q8_0.gguf",
    }

    save_app_settings(payload)
    loaded = load_app_settings()

    assert loaded["remember_backend"] is True
    assert loaded["inference_backend"] == "vulkan"
    assert loaded["audio_cpp_device"] == 0
    assert loaded["audio_cpp_threads"] == 6
    assert loaded["audio_cpp_timeout"] == 120
    assert loaded["audio_cpp_max_batch"] == 16
    assert loaded["audio_cpp_cli"] == "/opt/audiocpp/bin/audiocpp_cli"
    assert loaded["audio_cpp_model"] == "/models/chatterbox_q8_0.gguf"


def test_app_settings_robust_to_corrupt_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from the_oracle.gui_settings import app_settings_path

    app_settings_path().write_text("{not json", encoding="utf-8")

    settings = load_app_settings()

    assert settings["inference_backend"] == "pytorch"
    assert settings["remember_backend"] is True


def test_app_settings_normalizes_invalid_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_app_settings({
        "remember_backend": False,
        "inference_backend": "cuda",  # unsupported -> pytorch
        "audio_cpp_device": -3,  # negative -> None
        "audio_cpp_threads": 0,  # non-positive -> None
        "audio_cpp_timeout": "abc",  # non-numeric -> None
        "audio_cpp_max_batch": 8,
        "audio_cpp_cli": None,
    })

    loaded = load_app_settings()

    assert loaded["remember_backend"] is False
    assert loaded["inference_backend"] == "pytorch"
    assert loaded["audio_cpp_device"] is None
    assert loaded["audio_cpp_threads"] is None
    assert loaded["audio_cpp_timeout"] is None
    assert loaded["audio_cpp_max_batch"] == 8
    assert loaded["audio_cpp_cli"] == ""


def test_gui_settings_coerce_invalid_inference_backend(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    payload = _payload()
    payload["project"]["inference_backend"] = "cuda"
    save_gui_settings(path, payload)

    loaded = load_gui_settings(path)

    assert loaded["project"]["inference_backend"] == "pytorch"


def test_gui_settings_default_audio_cpp_knobs_to_none(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    save_gui_settings(path, _payload())

    loaded = load_gui_settings(path)

    assert loaded["project"]["audio_cpp_device"] is None
    assert loaded["project"]["audio_cpp_threads"] is None
    assert loaded["project"]["audio_cpp_timeout"] is None
    assert loaded["project"]["audio_cpp_max_batch"] is None


def test_gui_settings_preserve_audio_cpp_knobs(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    payload = _payload()
    payload["project"]["audio_cpp_device"] = 2
    payload["project"]["audio_cpp_threads"] = 6
    payload["project"]["audio_cpp_timeout"] = 120
    payload["project"]["audio_cpp_max_batch"] = 16
    save_gui_settings(path, payload)

    loaded = load_gui_settings(path)

    assert loaded["project"]["audio_cpp_device"] == 2
    assert loaded["project"]["audio_cpp_threads"] == 6
    assert loaded["project"]["audio_cpp_timeout"] == 120
    assert loaded["project"]["audio_cpp_max_batch"] == 16


def test_gui_settings_coerce_invalid_audio_cpp_knobs(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    payload = _payload()
    payload["project"]["audio_cpp_device"] = -3
    payload["project"]["audio_cpp_threads"] = 0
    payload["project"]["audio_cpp_timeout"] = 0
    payload["project"]["audio_cpp_max_batch"] = 0
    save_gui_settings(path, payload)

    loaded = load_gui_settings(path)

    assert loaded["project"]["audio_cpp_device"] is None
    assert loaded["project"]["audio_cpp_threads"] is None
    assert loaded["project"]["audio_cpp_timeout"] is None
    assert loaded["project"]["audio_cpp_max_batch"] is None

    # Non-numeric junk also coerces to None instead of crashing the load.
    path2 = tmp_path / "settings2.json"
    payload2 = _payload()
    payload2["project"]["audio_cpp_device"] = "gpu-zero"
    payload2["project"]["audio_cpp_threads"] = "many"
    payload2["project"]["audio_cpp_timeout"] = "whenever"
    payload2["project"]["audio_cpp_max_batch"] = "lots"
    save_gui_settings(path2, payload2)

    loaded2 = load_gui_settings(path2)

    assert loaded2["project"]["audio_cpp_device"] is None
    assert loaded2["project"]["audio_cpp_threads"] is None
    assert loaded2["project"]["audio_cpp_timeout"] is None
    assert loaded2["project"]["audio_cpp_max_batch"] is None
