from pathlib import Path

import pytest

from tests.helpers import isolate_user_config

from the_oracle.gui_settings import (
    GUISettingsError,
    PayloadDefaults,
    WidgetSnapshot,
    cast_request_from_payload,
    current_gui_settings_payload,
    default_gui_settings_payload,
    drop_next_format_backup,
    input_file_is_trusted,
    list_templates,
    load_app_settings,
    load_gui_settings,
    load_recent_reference_paths,
    load_template,
    next_format_backup,
    remember_format_backup,
    remember_recent_reference_path,
    remember_trusted_input_file,
    save_app_settings,
    save_gui_settings,
    save_template,
    speaker_config_from_payload,
)
from the_oracle.gui_utils import normalize_cast_keys


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
    isolate_user_config(monkeypatch, tmp_path)
    save_template("Oracle Template", _payload())

    assert list_templates() == ["Oracle_Template"]
    assert load_template("Oracle Template")["name"] == "Oracle Template"


def test_incomplete_gui_settings_fail_clearly(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"version": 1, "project": {}}', encoding="utf-8")

    with pytest.raises(GUISettingsError, match="missing required fields"):
        load_gui_settings(path)


def test_recent_reference_paths_are_mru_and_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_user_config(monkeypatch, tmp_path)
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
    isolate_user_config(monkeypatch, tmp_path)

    settings = load_app_settings()

    assert settings["remember_backend"] is True
    assert "default_input_dir" not in settings
    assert "default_output_dir" not in settings
    assert "output_filename_warning" not in settings
    assert settings["inference_backend"] == "pytorch"
    assert settings["audio_cpp_device"] is None
    assert settings["device_mode"] == "cpu"
    assert settings["cuda_device"] is None
    assert settings["audio_cpp_cli"] == ""
    assert settings["audio_cpp_model"] == ""


def test_app_settings_round_trip_persists_backend_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_user_config(monkeypatch, tmp_path)
    payload = {
        "remember_backend": True,
        "inference_backend": "pytorch",
        "device_mode": "cuda",
        "cuda_device": 1,
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
    assert loaded["inference_backend"] == "pytorch"
    assert loaded["device_mode"] == "cuda"
    assert loaded["cuda_device"] == 1
    assert loaded["audio_cpp_device"] == 0
    assert loaded["audio_cpp_threads"] == 6
    assert loaded["audio_cpp_timeout"] == 120
    assert loaded["audio_cpp_max_batch"] == 16
    assert loaded["audio_cpp_cli"] == "/opt/audiocpp/bin/audiocpp_cli"
    assert loaded["audio_cpp_model"] == "/models/chatterbox_q8_0.gguf"


def test_app_settings_round_trip_persists_workspace_defaults_and_output_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_user_config(monkeypatch, tmp_path)
    save_app_settings({
        "default_input_dir": " /custom/Input ",
        "default_output_dir": "/custom/Output",
        "output_filename_warning": False,
    })

    loaded = load_app_settings()

    assert loaded["default_input_dir"] == "/custom/Input"
    assert loaded["default_output_dir"] == "/custom/Output"
    assert loaded["output_filename_warning"] is False


def test_app_settings_drops_malformed_workspace_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_user_config(monkeypatch, tmp_path)
    save_app_settings({
        "default_input_dir": 123,
        "default_output_dir": None,
        "output_filename_warning": "no",
    })

    loaded = load_app_settings()

    assert "default_input_dir" not in loaded
    assert "default_output_dir" not in loaded
    assert "output_filename_warning" not in loaded


def test_app_settings_round_trip_persists_recording_preferences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_user_config(monkeypatch, tmp_path)
    save_app_settings({
        "recording_wizard_completed": True,
        "recording_wizard_dismissed": False,
        "recording_settings": {
            "microphone_index": "2",
            "samplerate": "48000",
            "input_file": "/custom/Input/script.txt",
            "output_dir": "/custom/Seashells",
            "output_filename": "Cody_warm.wav",
            "generic_name_warning": False,
            "remember_input_default": True,
            "remember_output_default": False,
        },
    })

    loaded = load_app_settings()

    assert loaded["recording_wizard_completed"] is True
    assert loaded["recording_settings"] == {
        "microphone_index": 2,
        "samplerate": 48000,
        "input_file": "/custom/Input/script.txt",
        "output_dir": "/custom/Seashells",
        "output_filename": "Cody_warm.wav",
        "generic_name_warning": False,
        "remember_input_default": True,
        "remember_output_default": False,
    }


def test_app_settings_drops_malformed_recording_preferences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_user_config(monkeypatch, tmp_path)
    save_app_settings({
        "recording_settings": {
            "microphone_index": "not-an-index",
            "samplerate": None,
            "input_file": 123,
            "output_dir": None,
            "output_filename": 456,
            "generic_name_warning": "no",
        },
    })

    loaded = load_app_settings()

    assert loaded["recording_settings"] == {}


def test_app_settings_round_trip_persists_inference_wizard_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_user_config(monkeypatch, tmp_path)

    save_app_settings({
        "inference_wizard_completed": True,
        "inference_wizard_dismissed": False,
    })

    loaded = load_app_settings()

    assert loaded["inference_wizard_completed"] is True
    assert loaded["inference_wizard_dismissed"] is False


def test_app_settings_drops_malformed_inference_wizard_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_user_config(monkeypatch, tmp_path)

    save_app_settings({
        "inference_wizard_completed": "yes",
        "inference_wizard_dismissed": 1,
    })

    loaded = load_app_settings()

    assert "inference_wizard_completed" not in loaded
    assert "inference_wizard_dismissed" not in loaded


def test_app_settings_robust_to_corrupt_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_user_config(monkeypatch, tmp_path)
    from the_oracle.gui_settings import app_settings_path

    app_settings_path().write_text("{not json", encoding="utf-8")

    settings = load_app_settings()

    assert settings["inference_backend"] == "pytorch"
    assert settings["remember_backend"] is True


def test_app_settings_normalizes_invalid_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_user_config(monkeypatch, tmp_path)
    save_app_settings({
        "remember_backend": False,
        "inference_backend": "cuda",  # unsupported -> pytorch
        "device_mode": "cuda",
        "cuda_device": -3,  # negative -> None
        "audio_cpp_device": -3,  # negative -> None
        "audio_cpp_threads": 0,  # non-positive -> None
        "audio_cpp_timeout": "abc",  # non-numeric -> None
        "audio_cpp_max_batch": 8,
        "audio_cpp_cli": None,
    })

    loaded = load_app_settings()

    assert loaded["remember_backend"] is False
    assert loaded["inference_backend"] == "pytorch"
    assert loaded["device_mode"] == "cuda"
    assert loaded["cuda_device"] is None
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


# ------------- load_gui_settings raises GUISettingsError on IO/parse errors -------------


def test_load_gui_settings_missing_file_raises_gui_settings_error(tmp_path: Path) -> None:
    """A vanished file must surface as GUISettingsError, not raw FileNotFoundError.

    Regression: the template-menu click handler catches GUISettingsError;
    a raw OSError escaped it and crashed the click with an unhandled
    traceback instead of showing the error dialog.
    """
    with pytest.raises(GUISettingsError, match="could not be read"):
        load_gui_settings(tmp_path / "missing.json")


def test_load_gui_settings_corrupt_json_raises_gui_settings_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(GUISettingsError, match="not valid JSON"):
        load_gui_settings(path)


def test_load_template_missing_raises_gui_settings_error() -> None:
    with pytest.raises(GUISettingsError):
        load_template("no_such_template_anywhere")


# ----------------------------------------------------------------------------
# Input format-health bookkeeping (trusted files + backup records)
# ----------------------------------------------------------------------------


def test_remember_format_backup_dedupes_per_file_and_keeps_newest_first() -> None:
    settings: dict = {}
    remember_format_backup(settings, "/tmp/a.txt", "/tmp/a.txt.bak-1", stamp="t1")
    remember_format_backup(settings, "/tmp/b.txt", "/tmp/b.txt.bak-1", stamp="t2")
    remember_format_backup(settings, "/tmp/a.txt", "/tmp/a.txt.bak-2", stamp="t3")

    records = settings["recent_format_backups"]
    assert [r["file"] for r in records] == ["/tmp/a.txt", "/tmp/b.txt"]
    assert records[0]["backup"] == "/tmp/a.txt.bak-2"
    # A no-backup call records nothing.
    remember_format_backup(settings, "/tmp/c.txt", None)
    assert len(settings["recent_format_backups"]) == 2


def test_remember_format_backup_caps_the_history_at_twenty() -> None:
    settings: dict = {}
    for index in range(25):
        remember_format_backup(
            settings, f"/tmp/f{index}.txt", f"/tmp/f{index}.txt.bak", stamp=f"t{index}"
        )
    records = settings["recent_format_backups"]
    assert len(records) == 20
    assert records[0]["file"] == "/tmp/f24.txt"  # newest survives
    assert records[-1]["file"] == "/tmp/f5.txt"  # oldest five dropped


def test_remember_format_backup_stamps_with_the_persisted_format() -> None:
    from datetime import datetime

    settings: dict = {}
    remember_format_backup(settings, "/tmp/a.txt", "/tmp/a.txt.bak")
    stamp = settings["recent_format_backups"][0]["stamp"]
    # Round-trips through the format the schema is documented with.
    assert datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")


def test_trusted_file_check_resolves_paths_before_comparing() -> None:
    settings: dict = {}
    remember_trusted_input_file(settings, "/tmp/repo/input/messy.txt")
    assert input_file_is_trusted(settings, "/tmp/repo/input/messy.txt")
    # The same file via a different spelling of the path is still trusted.
    assert input_file_is_trusted(settings, "/tmp/repo/./input/messy.txt")
    assert not input_file_is_trusted(settings, "/tmp/repo/input/other.txt")
    # An empty settings payload is never trusted.
    assert not input_file_is_trusted({}, "/tmp/repo/input/messy.txt")


def test_next_and_drop_format_backup_skip_malformed_records() -> None:
    settings = {
        "recent_format_backups": [
            "not-a-dict",
            {"file": "/tmp/x.txt"},  # no backup key
            {"file": "/tmp/a.txt", "backup": "/tmp/a.txt.bak", "stamp": "t1"},
            {"backup": "/tmp/b.txt.bak"},  # no file key
            {"file": "/tmp/b.txt", "backup": "/tmp/b.txt.bak", "stamp": "t0"},
        ]
    }
    assert next_format_backup(settings) == {
        "file": "/tmp/a.txt", "backup": "/tmp/a.txt.bak", "stamp": "t1"
    }
    # next does not consume; drop consumes exactly the first usable record.
    assert next_format_backup(settings) is not None
    drop_next_format_backup(settings)
    assert next_format_backup(settings)["file"] == "/tmp/b.txt"
    drop_next_format_backup(settings)
    assert next_format_backup(settings) is None
    # Dropping past the end leaves an empty list, not a crash.
    drop_next_format_backup(settings)
    assert settings["recent_format_backups"] == []


# --- settings-payload policy (the pure builders fed by a WidgetSnapshot) ----


def _defaults() -> "PayloadDefaults":
    from the_oracle.models.project import VoiceSettings
    from the_oracle.models.settings import RenderSettings

    render = RenderSettings()
    return PayloadDefaults(
        model_variant=render.model_variant,
        correction_mode=render.correction_mode,
        loudness_preset=render.loudness_preset,
        crossfade_ms=render.crossfade_ms,
        inference_backend=render.inference_backend,
        device_mode=render.device_mode,
        cuda_device=render.cuda_device,
        audio_cpp_device=render.audio_cpp_device,
        audio_cpp_threads=render.audio_cpp_threads,
        audio_cpp_timeout=render.audio_cpp_timeout,
        audio_cpp_max_batch=render.audio_cpp_max_batch,
        default_voice_dict=VoiceSettings(variant=render.model_variant).to_dict(),
    )


def _snapshot(**overrides) -> "WidgetSnapshot":
    from the_oracle.gui_settings import WidgetSnapshot
    from the_oracle.models.settings import SpeakerSettings
    from the_oracle.models.project import VoiceSettings

    base = dict(
        cast_keys=["A", "B"],
        speaker_names={"A": "Ada"},
        model_variant="standard",
        correction_mode="moderate",
        loudness_preset="light",
        crossfade_ms=20,
        inference_backend="pytorch",
        device_mode="cpu",
        cuda_device=None,
        output_dir="/tmp/out",
        output_filename="my render",
        export_srt=False,
        monologue=False,
        delete_confirm_enabled=True,
        output_filename_warning_enabled=True,
        audio_cpp_values={"audio_cpp_device": 3, "audio_cpp_threads": "4", "audio_cpp_timeout": None, "audio_cpp_max_batch": None},
        speaker_settings={
            "A": SpeakerSettings(reference_path="/ref/a.wav", voice_settings=VoiceSettings()),
            "B": SpeakerSettings(reference_path="", voice_settings=VoiceSettings()),
        },
    )
    base.update(overrides)
    return WidgetSnapshot(**base)


def test_default_payload_schema_threads_the_engine_defaults(tmp_path: Path) -> None:
    payload = default_gui_settings_payload(str(tmp_path / "out"), _defaults())

    assert payload["version"] == 1
    assert payload["cast"] == ["A", "B"]
    assert payload["speaker_names"] == {}
    assert payload["project"]["output_dir"] == str(tmp_path / "out")
    assert payload["project"]["model_variant"] == "standard"
    assert payload["project"]["inference_backend"] == "pytorch"
    assert set(payload["speakers"]) == {"A", "B"}
    for entry in payload["speakers"].values():
        assert entry["reference_path"] == ""
        assert entry["emotion_reference_paths"] == {}
        assert entry["voice_settings"]


def test_audio_cpp_knobs_persist_only_under_vulkan() -> None:
    """The disabled-widget rule: knobs left over from an earlier backend must
    not be saved alongside inference_backend: pytorch."""
    vulkan = current_gui_settings_payload(
        _snapshot(inference_backend="vulkan", device_mode="cpu")
    )["project"]
    assert vulkan["audio_cpp_device"] == 3
    assert vulkan["audio_cpp_threads"] == "4"

    pytorch = current_gui_settings_payload(_snapshot())["project"]
    assert pytorch["inference_backend"] == "pytorch"
    assert pytorch["audio_cpp_device"] is None
    assert pytorch["audio_cpp_threads"] is None
    assert pytorch["audio_cpp_timeout"] is None
    assert pytorch["audio_cpp_max_batch"] is None


def test_current_payload_carries_cast_names_and_speaker_entries() -> None:
    payload = current_gui_settings_payload(_snapshot())

    assert payload["cast"] == ["A", "B"]
    assert payload["speakers"]["A"]["name"] == "Ada"
    assert payload["speakers"]["B"]["name"] == ""
    assert payload["speakers"]["A"]["reference_path"] == "/ref/a.wav"
    assert payload["project"]["output_filename"] == "my render.flac"
    assert payload["project"]["correction_mode"] == "moderate"


def test_cast_request_fallback_chain() -> None:
    # Explicit cast wins, normalized (uppercased, deduped, capped).
    assert cast_request_from_payload({"cast": ["b", "A", "C"]}) == ["B", "A", "C"]
    # Missing or invalid cast: normalize_cast_keys floors at the single
    # narrator ["A"] — it never returns empty, which is exactly what makes
    # the unreachable saved-speaker-keys/A/B fallbacks safe to have collapsed.
    assert cast_request_from_payload({"speakers": {"C": {}, "A": {}}}) == ["A"]
    assert cast_request_from_payload({}) == ["A"]
    assert cast_request_from_payload({"speakers": "garbage"}) == ["A"]
    assert cast_request_from_payload({"cast": "garbage"}) == ["A"]


def test_cast_keys_floor_property_holds() -> None:
    """The property the collapsed helper leans on: the normalizer's floor."""
    for garbage in (None, [], "", {}, ["??"], ["   "]):
        assert normalize_cast_keys(garbage) == ["A"]


def test_speaker_config_decode_validates_blend_fields() -> None:
    from the_oracle.models.project import VoiceSettings

    decoded = speaker_config_from_payload(
        {
            "reference_path": "/ref/x.wav",
            "voice_settings": VoiceSettings().to_dict(),
            "blend_references": ["  ", "/second.wav", ""],
            "blend_weight": 2.0,
            "blend_mode": "ROTATION",
            "emotion_reference_paths": "not-a-dict",
        }
    )
    assert decoded["reference_path"] == "/ref/x.wav"
    assert decoded["blend_references"] == ["/second.wav"]  # blanks dropped
    assert decoded["blend_weight"] == 1.0  # clamped into [0, 1]
    assert decoded["blend_mode"] == "mix"  # invalid -> default
    assert decoded["emotion_reference_paths"] == {}  # non-dict guarded

    garbage = speaker_config_from_payload({"blend_weight": "not-a-number"})
    assert garbage["blend_weight"] == 0.5


def test_current_payload_decodes_back_to_equal_speaker_settings() -> None:
    """A current-payload speakers entry round-trips through the decoder."""
    from the_oracle.models.settings import SpeakerSettings

    snapshot = _snapshot()
    payload = current_gui_settings_payload(snapshot)
    rebuilt = SpeakerSettings(**speaker_config_from_payload(payload["speakers"]["A"]))
    assert rebuilt == snapshot.speaker_settings["A"]
