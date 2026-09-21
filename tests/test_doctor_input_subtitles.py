"""Tests for the doctor's Input/ subtitle-encoding check (scripts/doctor.py).

The subtitle decode chain (``the_oracle.srt_ingest``) reads UTF-8 with BOM
first and falls back to CP1252. This doctor check surfaces, before a render,
which Input/ subtitle files would take that fallback — and, critically,
which would be BLOCKED by it: ``convert_srt_file`` re-decodes strictly, so a
file carrying bytes CP1252 does not define (0x81 et al) passes the lossy
pre-check and then fails conversion outright. The tests pin the
classification, the read-only contract (missing Input/ is reported, never
created), the human-report rendering, and the next-steps wiring.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

DOCTOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "doctor.py"


def _load_doctor():
    spec = importlib.util.spec_from_file_location("oracle_doctor", DOCTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- classification ----------------------------------------------------------


def test_classifies_utf8_fallback_and_blocked_files(tmp_path: Path) -> None:
    doctor = _load_doctor()
    input_dir = tmp_path / "Input"
    (input_dir / "sub").mkdir(parents=True)

    (input_dir / "clean.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nWinston: café.\n", encoding="utf-8")
    # A UTF-8 BOM must classify as UTF-8, not as a fallback file.
    (input_dir / "bom.vtt").write_bytes(
        b"\xef\xbb\xbf" + "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Julia>voilà.\n".encode("utf-8")
    )
    # Cleanly legacy-encoded: converts via the CP1252 fallback (also nested,
    # proving the scan is recursive).
    (input_dir / "sub" / "legacy.srt").write_bytes(
        "1\n00:00:01,000 --> 00:00:02,000\nWinston: café.\n".encode("cp1252")
    )
    # 0x81 is not defined in CP1252: the lossy pre-check passes it, the
    # strict conversion re-decode would reject it.
    (input_dir / "broken.srt").write_bytes(
        "1\n00:00:01,000 --> 00:00:02,000\nWinston: ".encode("cp1252") + b"\x81" + "\n".encode("ascii")
    )

    status = doctor._input_subtitles_status(tmp_path)

    assert status["scanned"] == 4
    assert status["utf8_count"] == 2
    assert [entry["path"] for entry in status["fallback"]] == ["Input/sub/legacy.srt"]
    # 0x81 has no CP1252 mapping: blocked, with the actionable error.
    assert [entry["path"] for entry in status["blocked"]] == ["Input/broken.srt"]
    assert status["ok"] is False


def test_flags_blocked_file_with_cp1252_undefined_bytes(tmp_path: Path) -> None:
    doctor = _load_doctor()
    input_dir = tmp_path / "Input"
    input_dir.mkdir()
    (input_dir / "weird.vtt").write_bytes(b"WEBVTT\n\n\x81\n")

    status = doctor._input_subtitles_status(tmp_path)

    assert status["ok"] is False
    assert len(status["blocked"]) == 1
    entry = status["blocked"][0]
    assert entry["path"] == "Input/weird.vtt"
    assert "CP1252 cannot decode" in entry["error"]
    assert "Re-save the file as UTF-8" in entry["error"]
    assert "Input/weird.vtt" in status["error"]


def test_missing_input_dir_reported_and_not_created(tmp_path: Path) -> None:
    """Read-only contract: no Input/ is reported, never mkdir'd."""
    doctor = _load_doctor()

    status = doctor._input_subtitles_status(tmp_path)

    assert status["exists"] is False
    assert status["scanned"] == 0
    assert status["ok"] is True
    assert not (tmp_path / "Input").exists(), "the check must not create Input/"


def test_non_subtitle_files_are_ignored(tmp_path: Path) -> None:
    doctor = _load_doctor()
    input_dir = tmp_path / "Input"
    input_dir.mkdir()
    (input_dir / "script.txt").write_bytes(b"\xff\xfe not a subtitle")
    (input_dir / "readme.md").write_text("prose", encoding="utf-8")

    status = doctor._input_subtitles_status(tmp_path)

    assert status["scanned"] == 0
    assert status["ok"] is True


# --- human report ------------------------------------------------------------


def _full_report(input_subtitles: dict) -> dict:
    """A report fixture with every key _print_human_report reads."""
    return {
        "repo_root": "/repo",
        "platform": "test-platform",
        "ci_mode": False,
        "python": {"ok": True, "executable": "python3", "version": "3.12.0"},
        "ffmpeg": {"ok": True, "path": "/usr/bin/ffmpeg"},
        "entrypoint": {"ok": True, "fresh_shell_path": "/bin/the-oracle", "path_entrypoint": "", "venv_entrypoint": ""},
        "chatterbox_import": {"ok": True, "target": "from chatterbox.tts import ChatterboxTTS", "error": ""},
        "chatterbox_init": {"ok": True, "seconds": 1.0, "skipped": False, "error": ""},
        "perth": {"ok": True, "watermarker_symbol": "PerthImplicitWatermarker", "error": ""},
        "turbo": {"ok": True, "checkpoint_dir": "/cache", "error": ""},
        "cuda_backend": {"runtime_available": False, "reason": "none", "devices": []},
        "qt": {
            "ok": True,
            "plugin_path": "/plugins",
            "qt_platform": "offscreen",
            "error": "",
            "missing_libraries": [],
            "suggested_packages": [],
        },
        "voice_sources": {
            "ok": True,
            "primary_source": "seashells",
            "default_voice_assessment": "ok",
            "seashell_clip_count": 3,
            "fallback_clip_count": 0,
            "better_local_assets_detail": "",
            "voice_mixing_detail": "",
        },
        "deterministic_smoke": {"ok": True, "output_path": "/repo/build/out.flac", "error": ""},
        "real_engine_smoke": {"ok": True, "expected_paths": {"output": "/repo/build/smoke.flac"}, "output_exists": True},
        "vulkan_backend": {
            "ok": False,
            "binary_built": False,
            "model_override_set": False,
            "model_file_exists": False,
            "model_path": "",
            "vulkan_device": False,
            "device_name": "",
            "rdna1_device": False,
            "device_index_env": "",
            "threads_env": "",
            "batch_env": "",
            "vendored_patch_applied": None,
            "caveat": "",
            "error": "",
            "audio_cpp_devices": [],
        },
        "dependency_pins": {"ok": True, "error": "", "checked_count": 5},
        "input_subtitles": input_subtitles,
        "next_steps": [],
    }


def test_human_report_lists_fallback_files_and_blocked_detail(capsys) -> None:
    doctor = _load_doctor()
    report = _full_report(
        {
            "ok": False,
            "input_dir": "/repo/Input",
            "exists": True,
            "scanned": 2,
            "utf8_count": 1,
            "fallback": [{"path": "Input/legacy.srt"}],
            "blocked": [{"path": "Input/weird.vtt", "error": "contains byte(s) CP1252 cannot decode"}],
            "unreadable": [],
            "error": "Input/weird.vtt: contains byte(s) CP1252 cannot decode",
        }
    )

    doctor._print_human_report(report)

    out = capsys.readouterr().out
    assert "Input subtitle encoding: 2 subtitle file(s) scanned; 1 UTF-8; CP1252 fallback: Input/legacy.srt" in out
    assert "Input/weird.vtt: contains byte(s) CP1252 cannot decode" in out


def test_human_report_skips_line_when_input_dir_missing(capsys) -> None:
    doctor = _load_doctor()
    report = _full_report({"ok": True, "input_dir": "/repo/Input", "exists": False, "scanned": 0, "utf8_count": 0, "fallback": [], "blocked": [], "unreadable": []})

    doctor._print_human_report(report)

    out = capsys.readouterr().out
    assert "SKIP Input subtitle encoding: Input/ not present" in out


def test_human_report_handles_older_reports_without_the_key(capsys) -> None:
    """Backwards compatibility: a report dict without input_subtitles must not
    crash the renderer (same contract as dependency_pins)."""
    doctor = _load_doctor()
    report = _full_report(None)
    del report["input_subtitles"]

    doctor._print_human_report(report)

    out = capsys.readouterr().out
    assert "Input subtitle encoding" not in out


# --- next steps wiring --------------------------------------------------------


def _minimal_report_for_next_steps(input_subtitles: dict) -> dict:
    return {
        "python": {"ok": True},
        "ffmpeg": {"ok": True},
        "qt": {"suggested_packages": []},
        "entrypoint": {"ok": True, "path_has_local_bin": True},
        "chatterbox_import": {"ok": True},
        "chatterbox_init": {"ok": True, "skipped": False},
        "perth": {"ok": True},
        "deterministic_smoke": {"ok": True},
        "real_engine_smoke": {"ok": True},
        "turbo": {"ok": True},
        "voice_sources": {"primary_source": "seashells"},
        "vulkan_backend": {
            "binary_built": False,
            "model_override_set": False,
            "model_file_exists": False,
            "vendored_patch_applied": None,
            "device_index_env": "",
            "audio_cpp_devices": [],
        },
        "input_subtitles": input_subtitles,
    }


def test_next_steps_flags_blocked_subtitle_file() -> None:
    doctor = _load_doctor()
    report = _minimal_report_for_next_steps(
        {"blocked": [{"path": "Input/weird.vtt", "error": "contains byte(s) CP1252 cannot decode; subtitle conversion would fail after the fallback pre-check. Re-save the file as UTF-8."}], "unreadable": []}
    )

    steps = doctor._build_next_steps(report, ci_mode=True)

    assert any("Input/weird.vtt" in step and "Re-save the file as UTF-8" in step for step in steps)


def test_next_steps_quiet_when_no_subtitle_problems() -> None:
    doctor = _load_doctor()
    report = _minimal_report_for_next_steps({"blocked": [], "unreadable": []})

    steps = doctor._build_next_steps(report, ci_mode=True)

    assert not any("Input/" in step for step in steps)


# --- wiring guard -------------------------------------------------------------


def test_check_is_wired_into_the_report() -> None:
    """Vacuity guard: the check must actually run in run()'s report."""
    doctor_src = DOCTOR_PATH.read_text(encoding="utf-8")
    assert '"input_subtitles": _input_subtitles_status(repo_root),' in doctor_src
    assert doctor_src.count("_input_subtitles_status(") >= 2  # definition + call site
