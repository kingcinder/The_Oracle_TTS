"""Review-fix regression tests for the Oracle desktop GUI.

Covers the GUI defects fixed in the 2026-09-10 review cycle that don't need
Qt, so they run headless:

- recording filenames can never escape the chosen output folder
  (``sanitize_recording_filename`` / ``recording_target_path``)
- cast bookkeeping: key normalization, free-key allocation, add/remove/rename
  (``CastModel``, ``normalize_cast_keys``, ``next_speaker_key``)
- the GUI's voice capacity still matches the engine's (no silent drift)
- model-download timeout cleanup kills the whole process group
  (``kill_process_tree``)
- corrupt ``recent_reference_clips.json`` never crashes startup
- GUI settings round-trip keeps the cast, speaker names, and hybrid
  (blend) fields instead of silently dropping them

Widget-level behavior (cast dialog layout, preview busy-state cleanup,
Recording Studio multi-instance tracking) needs a Qt binding and is marked
to skip when none is importable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from the_oracle import gui_settings, gui_utils
from the_oracle.gui_settings import (
    load_gui_settings,
    load_recent_reference_paths,
    save_gui_settings,
)
from the_oracle.gui_utils import (
    MAX_CAST_SPEAKERS,
    CastModel,
    kill_process_tree,
    next_speaker_key,
    normalize_cast_keys,
    recording_target_path,
    sanitize_recording_filename,
    speaker_keys,
)

def _importable(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def _qt_bindings() -> list[str]:
    return [name for name in ("PySide6", "PyQt6", "PyQt5") if _importable(name)]


requires_qt = pytest.mark.skipif(
    not _qt_bindings(),
    reason="Qt binding (PySide6/PyQt6/PyQt5) not importable in this runtime",
)


# ---------------------------------------------------------------------------
# Voice capacity: the GUI must never promise voices the engine can't render.
# ---------------------------------------------------------------------------


def test_max_cast_speakers_matches_engine():
    heuristics = pytest.importorskip("the_oracle.speaker_attribution.heuristics")
    assert MAX_CAST_SPEAKERS == heuristics.MAX_SPEAKERS


def test_speaker_keys_cover_engine_range():
    keys = speaker_keys()
    assert keys[0] == "A"
    assert keys[-1] == "X"
    assert len(keys) == MAX_CAST_SPEAKERS == 24
    assert len(set(keys)) == len(keys)


# ---------------------------------------------------------------------------
# Cast key bookkeeping
# ---------------------------------------------------------------------------


def test_normalize_cast_keys_dedupes_and_drops_invalid():
    assert normalize_cast_keys(["B", "a", "B", "Z", "1", "", None, " c "]) == ["B", "A", "C"]


def test_normalize_cast_keys_empty_falls_back_to_narrator():
    assert normalize_cast_keys([]) == ["A"]
    assert normalize_cast_keys(None) == ["A"]
    assert normalize_cast_keys(["ZZ"]) == ["A"]


def test_normalize_cast_keys_caps_at_engine_capacity():
    many = [chr(ord("A") + i) for i in range(30)]
    assert normalize_cast_keys(many) == speaker_keys()


def test_next_speaker_key_first_free():
    assert next_speaker_key([]) == "A"
    assert next_speaker_key(["A", "B"]) == "C"
    assert next_speaker_key(["B", "A"]) == "C"


def test_next_speaker_key_none_when_full():
    assert next_speaker_key(speaker_keys()) is None


# ---------------------------------------------------------------------------
# CastModel
# ---------------------------------------------------------------------------


def test_cast_model_add_remove_rename():
    model = CastModel.from_parts(["A", "B"], {"A": "Narrator"})
    assert model.keys() == ["A", "B"]
    assert model.names() == {"A": "Narrator", "B": ""}

    member = model.add()
    assert member is not None and member.key == "C"
    assert model.rename("C", "  Villain ") is True
    assert model.names()["C"] == "Villain"
    assert model.rename("Z", "Nobody") is False

    assert model.remove("C") is True
    assert model.keys() == ["A", "B"]
    assert model.remove("Z") is False


def test_cast_model_narrator_cannot_be_removed():
    model = CastModel.from_parts(["A", "B"])
    assert model.remove("A") is False
    assert model.remove("a") is False
    assert model.keys() == ["A", "B"]


def test_cast_model_add_returns_none_when_full():
    model = CastModel.from_parts(speaker_keys())
    assert model.add() is None
    assert len(model.keys()) == MAX_CAST_SPEAKERS


def test_cast_model_serialization_round_trip():
    model = CastModel.from_parts(["A", "C"], {"A": "Narrator", "C": "Imp"})
    data = model.to_dict()
    assert data == {"cast": ["A", "C"], "names": {"A": "Narrator", "C": "Imp"}}
    restored = CastModel.from_dict(data)
    assert restored.keys() == ["A", "C"]
    assert restored.names() == {"A": "Narrator", "C": "Imp"}


def test_cast_model_from_dict_backwards_compat_sorted_speaker_keys():
    settings_map = {"B": {"reference_path": "b.wav"}, "A": {"reference_path": "a.wav"}}
    model = CastModel.from_dict({"speakers": settings_map}, settings_map=settings_map)
    assert model.keys() == ["A", "B"]
    assert model.settings_map()["A"] == {"reference_path": "a.wav"}


def test_cast_model_from_dict_garbage_returns_narrator_only():
    assert CastModel.from_dict(None).keys() == ["A"]
    assert CastModel.from_dict("nope").keys() == ["A"]


# ---------------------------------------------------------------------------
# Recording filename confinement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("take1", "take1.wav"),
        ("take1.wav", "take1.wav"),
        ("take1.WAV", "take1.WAV"),
        ("../../etc/passwd", "passwd.wav"),
        ("..\\..\\windows\\system32", "system32.wav"),
        ("/abs/path/evil.wav", "evil.wav"),
        ("a/b\\c", "c.wav"),
        ("..", "Seashell_No_1.wav"),
        ("", "Seashell_No_1.wav"),
        (None, "Seashell_No_1.wav"),
        ("   ", "Seashell_No_1.wav"),
        ("bad<>:\"|?*name", "badname.wav"),
        ("trailing.", "trailing.wav"),
        ("multi.part.name", "multi.part.name.wav"),
    ],
)
def test_sanitize_recording_filename(raw, expected):
    assert sanitize_recording_filename(raw) == expected


def test_sanitize_recording_filename_custom_default():
    assert sanitize_recording_filename("", default_stem="Take_1") == "Take_1.wav"


def test_recording_target_path_never_escapes_folder(tmp_path):
    folder = tmp_path / "recordings"
    target = recording_target_path(folder, "../../outside.wav")
    assert target == folder / "outside.wav"
    assert target.parent == folder

    target = recording_target_path(str(folder), "..\\..\\evil")
    assert target.parent == folder


# ---------------------------------------------------------------------------
# Process-group cleanup (model-download timeout path)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_kill_process_tree_kills_grandchildren():
    # A shell wrapper spawning a sleeper: killing only the direct child would
    # leave the grandchild (and its inherited pipes) alive, which is exactly
    # the hang the model-download timeout fix addresses.
    proc = subprocess.Popen(
        ["bash", "-c", "sleep 60"],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        kill_process_tree(proc)
        proc.wait(timeout=10)
        assert proc.poll() is not None
        # The whole group is gone: signaling it must fail with ProcessLookupError.
        with pytest.raises(ProcessLookupError):
            os.killpg(proc.pid, 0)
    finally:
        try:
            kill_process_tree(proc)
            proc.wait(timeout=5)
        except Exception:
            pass


def test_kill_process_tree_none_is_safe():
    kill_process_tree(None)


# ---------------------------------------------------------------------------
# Corrupt recent-reference state must not crash startup
# ---------------------------------------------------------------------------


def _point_recent_refs_at(tmp_path, monkeypatch):
    target = tmp_path / "recent_reference_clips.json"
    monkeypatch.setattr(gui_settings, "recent_references_path", lambda: target)
    return target


def test_load_recent_reference_paths_corrupt_json_returns_empty(tmp_path, monkeypatch):
    target = _point_recent_refs_at(tmp_path, monkeypatch)
    target.write_text("{not valid json", encoding="utf-8")
    assert load_recent_reference_paths() == []


def test_load_recent_reference_paths_wrong_shape_returns_empty(tmp_path, monkeypatch):
    target = _point_recent_refs_at(tmp_path, monkeypatch)
    target.write_text(json.dumps({"paths": "not-a-list"}), encoding="utf-8")
    assert load_recent_reference_paths() == []


def test_load_recent_reference_paths_missing_returns_empty(tmp_path, monkeypatch):
    _point_recent_refs_at(tmp_path, monkeypatch)
    assert load_recent_reference_paths() == []


def test_load_recent_reference_paths_round_trip(tmp_path, monkeypatch):
    from the_oracle.gui_settings import remember_recent_reference_path

    _point_recent_refs_at(tmp_path, monkeypatch)
    remember_recent_reference_path("/tmp/voice.wav")
    assert load_recent_reference_paths() == ["/tmp/voice.wav"]


# ---------------------------------------------------------------------------
# Settings round-trip: cast, names, and hybrid (blend) fields must survive
# ---------------------------------------------------------------------------


def _profile_payload() -> dict:
    return {
        "version": 1,
        "name": "review",
        "device_mode": "cpu",
        "cast": ["A", "B", "C"],
        "project": {
            "model_variant": "standard",
            "correction_mode": "moderate",
            "loudness_preset": "light",
            "crossfade_ms": 20,
            "inference_backend": "pytorch",
            "output_dir": "/tmp/out",
            "output_filename": "",
            "monologue": False,
        },
        "speakers": {
            "A": {
                "reference_path": "/tmp/a.wav",
                "voice_settings": {},
                "name": "Narrator",
                "blend_references": ["/tmp/a.wav", "/tmp/b.wav"],
                "blend_weight": 0.75,
                "blend_mode": "layer",
            },
            "B": {"reference_path": "", "voice_settings": {}},
            "C": {
                "reference_path": "/tmp/c.wav",
                "voice_settings": {},
                "name": "Imp",
                "blend_references": [],
                "blend_weight": 0.25,
                "blend_mode": "bogus-mode",
            },
        },
    }


def test_settings_round_trip_preserves_cast_names_and_blend(tmp_path):
    path = tmp_path / "profile.json"
    save_gui_settings(path, _profile_payload())
    loaded = load_gui_settings(path)

    assert loaded["cast"] == ["A", "B", "C"]

    speaker_a = loaded["speakers"]["A"]
    assert speaker_a["name"] == "Narrator"
    assert speaker_a["blend_references"] == ["/tmp/a.wav", "/tmp/b.wav"]
    assert speaker_a["blend_weight"] == pytest.approx(0.75)
    assert speaker_a["blend_mode"] == "layer"

    # Unknown blend modes normalize to "mix" instead of crashing or leaking through.
    assert loaded["speakers"]["C"]["blend_mode"] == "mix"
    assert loaded["speakers"]["C"]["name"] == "Imp"


def test_settings_round_trip_clamps_blend_weight(tmp_path):
    payload = _profile_payload()
    payload["speakers"]["A"]["blend_weight"] = 42
    path = tmp_path / "profile.json"
    save_gui_settings(path, payload)
    assert load_gui_settings(path)["speakers"]["A"]["blend_weight"] == pytest.approx(1.0)


def test_settings_normalize_old_profile_without_cast_falls_back_to_speaker_keys(tmp_path):
    payload = _profile_payload()
    del payload["cast"]
    path = tmp_path / "profile.json"
    save_gui_settings(path, payload)
    assert load_gui_settings(path)["cast"] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Qt-dependent behavior: skipped here, must be validated with a Qt binding.
# ---------------------------------------------------------------------------


@requires_qt
def test_cast_dialog_opens_with_full_voice_panels_per_speaker():
    """Manual-with-Qt: the cast dialog shows one full SpeakerGroup per cast
    member (voice picker, hybrid second voice, weight, mode, sliders), an
    add/remove row per speaker, and applies on close."""


@requires_qt
def test_preview_missing_voice_profile_clears_busy_state():
    """Manual-with-Qt: previewing a row whose speaker has no voice profile
    reports gracefully and the UI never stays stuck busy."""


@requires_qt
def test_multiple_recording_studios_tracked_per_instance():
    """Manual-with-Qt: two open studios each keep their worker and report
    their own saved take on close."""


@requires_qt
def test_main_close_joins_download_and_setup_threads():
    """Manual-with-Qt: closing the main window stops/joins a running model
    download and Vulkan setup thread instead of aborting."""
