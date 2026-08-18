"""Tests for the doctor's opt-in Vulkan backend check (scripts/doctor.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from the_oracle.tts_engines import vulkan_backend

DOCTOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "doctor.py"


def _load_doctor():
    spec = importlib.util.spec_from_file_location("oracle_doctor", DOCTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vulkan_backend_status_reports_missing_pieces(monkeypatch, tmp_path: Path) -> None:
    doctor = _load_doctor()
    monkeypatch.delenv("ORACLE_AUDIOCPP_CLI", raising=False)
    monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
    # The doctor lazy-imports find_audiocpp_binary; patch it at its source so
    # the test is independent of what is actually on disk in this workspace
    # (e.g. a previously built audio.cpp/ binary under the repo root).
    monkeypatch.setattr(vulkan_backend, "find_audiocpp_binary", lambda: None)
    monkeypatch.setattr(doctor, "_vulkaninfo_summary", lambda: (False, ""))
    monkeypatch.setattr(doctor, "_vulkan_patches_applied", lambda repo_root: None)

    status = doctor._vulkan_backend_status(tmp_path)

    assert status["ok"] is False
    assert status["binary_built"] is False
    assert status["model_override_set"] is False
    assert status["model_file_exists"] is False
    assert status["model_path"] == ""
    assert status["vulkan_device"] is False
    assert status["rdna1_device"] is False
    assert status["vendored_patch_applied"] is None
    assert status["caveat"] == ""


def test_vulkan_backend_status_ready_with_rdna1_caveat(monkeypatch, tmp_path: Path) -> None:
    doctor = _load_doctor()
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    model = tmp_path / "chatterbox-q8_0.gguf"
    model.write_bytes(b"stub-model")
    monkeypatch.setenv("ORACLE_AUDIOCPP_CLI", str(binary))
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model))
    monkeypatch.setattr(
        doctor,
        "_vulkaninfo_summary",
        lambda: (True, "deviceName = Radeon RX 5700 XT (RADV NAVI10)"),
    )
    monkeypatch.setattr(doctor, "_vulkan_patches_applied", lambda repo_root: True)

    status = doctor._vulkan_backend_status(tmp_path)

    assert status["ok"] is True
    assert status["binary_built"] is True
    assert status["model_override_set"] is True
    assert status["model_path"] == str(model)
    assert status["model_file_exists"] is True
    assert status["vulkan_device"] is True
    assert status["device_name"] == "Radeon RX 5700 XT (RADV NAVI10)"
    assert status["rdna1_device"] is True
    assert status["vendored_patch_applied"] is True
    assert "VK_ERROR_DEVICE_LOST" in status["caveat"]


def test_vulkan_backend_status_flags_missing_vendored_patch(monkeypatch, tmp_path: Path) -> None:
    doctor = _load_doctor()
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    model = tmp_path / "chatterbox-q8_0.gguf"
    model.write_bytes(b"stub-model")
    monkeypatch.setenv("ORACLE_AUDIOCPP_CLI", str(binary))
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model))
    monkeypatch.setattr(doctor, "_vulkaninfo_summary", lambda: (True, "deviceName = some gpu"))
    monkeypatch.setattr(doctor, "_vulkan_patches_applied", lambda repo_root: False)

    status = doctor._vulkan_backend_status(tmp_path)

    assert status["ok"] is True
    assert status["vendored_patch_applied"] is False
    assert "patch_audio_cpp_ggml.sh" in status["caveat"]


def test_vulkan_backend_status_flags_missing_model_file(monkeypatch, tmp_path: Path) -> None:
    """A set-but-dangling ORACLE_AUDIOCPP_MODEL must not count as ready."""
    doctor = _load_doctor()
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    missing = tmp_path / "no-such-model.gguf"
    monkeypatch.setenv("ORACLE_AUDIOCPP_CLI", str(binary))
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(missing))
    monkeypatch.setattr(doctor, "_vulkaninfo_summary", lambda: (True, "deviceName = some gpu"))
    monkeypatch.setattr(doctor, "_vulkan_patches_applied", lambda repo_root: True)

    status = doctor._vulkan_backend_status(tmp_path)

    assert status["ok"] is False
    assert status["binary_built"] is True
    assert status["model_override_set"] is True
    assert status["model_path"] == str(missing)
    assert status["model_file_exists"] is False
    assert status["vulkan_device"] is True


_LIST_DEVICES_SAMPLE = """\
ggml_vulkan: Found 2 Vulkan devices:\n\
Vulkan:0 "AMD Radeon RX 5700 XT (RADV NAVI10)" [GPU]\n\
Vulkan:1 "AMD Radeon RX 6900 XT (RADV NAVI21)" [GPU]\n\
CPU:0 "AMD Ryzen 7 3700X 8-Core Processor" [CPU]\n\
select with: --backend <cuda|hip|vulkan|metal|cpu> --device <index>\n\
"""


def test_audiocpp_devices_parses_vulkan_indexes(monkeypatch, tmp_path: Path) -> None:
    doctor = _load_doctor()
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setattr(
        doctor,
        "_run_command",
        lambda args, **kwargs: {"ok": True, "stdout": _LIST_DEVICES_SAMPLE, "stderr": "", "returncode": 0, "timed_out": False},
    )

    devices = doctor._audiocpp_devices(binary)

    assert devices == [
        {"index": 0, "name": "AMD Radeon RX 5700 XT (RADV NAVI10)"},
        {"index": 1, "name": "AMD Radeon RX 6900 XT (RADV NAVI21)"},
    ]


def test_audiocpp_devices_returns_empty_when_binary_missing() -> None:
    doctor = _load_doctor()
    assert doctor._audiocpp_devices(Path("/does/not/exist")) == []
    assert doctor._audiocpp_devices(None) == []


def test_audiocpp_devices_returns_empty_when_command_fails(monkeypatch, tmp_path: Path) -> None:
    doctor = _load_doctor()
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 1\n")
    binary.chmod(0o755)
    monkeypatch.setattr(
        doctor,
        "_run_command",
        lambda args, **kwargs: {"ok": False, "stdout": "", "stderr": "boom", "returncode": 1, "timed_out": False},
    )

    assert doctor._audiocpp_devices(binary) == []


def test_audiocpp_devices_parses_stderr_when_stdout_empty(monkeypatch, tmp_path: Path) -> None:
    doctor = _load_doctor()
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setattr(
        doctor,
        "_run_command",
        lambda args, **kwargs: {
            "ok": True,
            "stdout": "",
            "stderr": 'ggml_vulkan: device 0\nVulkan:0 "AMD Radeon RX 5700 XT (RADV NAVI10)" [GPU]\n',
            "returncode": 0,
            "timed_out": False,
        },
    )

    devices = doctor._audiocpp_devices(binary)

    assert devices == [{"index": 0, "name": "AMD Radeon RX 5700 XT (RADV NAVI10)"}]


def test_run_command_tolerates_non_executable_binary(monkeypatch, tmp_path: Path) -> None:
    doctor = _load_doctor()
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n")  # deliberately not chmod +x
    monkeypatch.setattr(doctor, "_audiocpp_devices", lambda _binary: [])
    monkeypatch.setenv("ORACLE_AUDIOCPP_CLI", str(binary))
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", "/models/chatterbox")
    monkeypatch.setattr(doctor, "_vulkaninfo_summary", lambda: (True, "deviceName = some gpu"))
    monkeypatch.setattr(doctor, "_vulkan_patches_applied", lambda repo_root: True)

    # The real _run_command now catches PermissionError as a structured
    # failure (OSError branch), so the doctor must not crash.
    result = doctor._run_command([str(binary), "--backend", "vulkan", "--list-devices"], timeout=5)

    assert result["ok"] is False
    assert result["returncode"] == 127
    assert "error" in result and result["error"]

    # status["ok"] is computed from env/device checks, not executability,
    # so assert the real invariant: the status call survives and reports no
    # device list (the binary probe failed) without raising.
    status = doctor._vulkan_backend_status(tmp_path)
    assert status["audio_cpp_devices"] == []
    assert status["error"] == ""


def test_vulkan_backend_status_includes_audio_cpp_devices(monkeypatch, tmp_path: Path) -> None:
    doctor = _load_doctor()
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("ORACLE_AUDIOCPP_CLI", str(binary))
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", "/models/chatterbox")
    monkeypatch.setattr(doctor, "_vulkaninfo_summary", lambda: (True, "deviceName = some gpu"))
    monkeypatch.setattr(doctor, "_vulkan_patches_applied", lambda repo_root: True)
    monkeypatch.setattr(doctor, "_audiocpp_devices", lambda _binary: [{"index": 0, "name": "GPU A"}])

    status = doctor._vulkan_backend_status(tmp_path)

    assert status["audio_cpp_devices"] == [{"index": 0, "name": "GPU A"}]


def test_next_steps_hints_device_pick_on_multi_gpu(monkeypatch) -> None:
    doctor = _load_doctor()
    report = {
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
            "binary_built": True,
            "model_override_set": True,
            "model_file_exists": True,
            "vendored_patch_applied": True,
            "device_index_env": "",
            "audio_cpp_devices": [
                {"index": 0, "name": "GPU A"},
                {"index": 1, "name": "GPU B"},
            ],
        },
    }

    steps = doctor._build_next_steps(report, ci_mode=True)

    assert any("ORACLE_AUDIOCPP_DEVICE" in step for step in steps)


def test_next_steps_suggests_download_script_when_model_unset(monkeypatch) -> None:
    """Binary built but ORACLE_AUDIOCPP_MODEL unset: point at the download
    script (and the one-shot build), not just at the variable."""
    doctor = _load_doctor()
    report = {
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
            "binary_built": True,
            "model_override_set": False,
            "model_file_exists": False,
            "vendored_patch_applied": True,
            "device_index_env": "",
            "audio_cpp_devices": [],
        },
    }

    steps = doctor._build_next_steps(report, ci_mode=True)

    assert any("download_audio_cpp_model.sh" in step for step in steps)
    assert any("--with-model" in step for step in steps)


def test_next_steps_flags_missing_model_file(monkeypatch) -> None:
    """ORACLE_AUDIOCPP_MODEL set but dangling: the step names the broken path
    and points at the download script to re-fetch."""
    doctor = _load_doctor()
    report = {
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
            "binary_built": True,
            "model_override_set": True,
            "model_file_exists": False,
            "model_path": "/broken/models/chatterbox.gguf",
            "vendored_patch_applied": True,
            "device_index_env": "",
            "audio_cpp_devices": [],
        },
    }

    steps = doctor._build_next_steps(report, ci_mode=True)

    model_steps = [step for step in steps if "ORACLE_AUDIOCPP_MODEL" in step]
    assert model_steps, "expected a step flagging the dangling model path"
    assert "/broken/models/chatterbox.gguf" in model_steps[0]
    assert "missing model file" in model_steps[0]


def test_next_steps_omit_device_hint_when_env_already_set(monkeypatch) -> None:
    doctor = _load_doctor()
    report = {
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
            "binary_built": True,
            "model_override_set": True,
            "model_file_exists": True,
            "vendored_patch_applied": True,
            "device_index_env": "1",
            "audio_cpp_devices": [{"index": 0, "name": "GPU A"}, {"index": 1, "name": "GPU B"}],
        },
    }

    steps = doctor._build_next_steps(report, ci_mode=True)

    assert not any("ORACLE_AUDIOCPP_DEVICE" in step for step in steps)


def test_human_report_lists_device_indexes(monkeypatch, capsys) -> None:
    doctor = _load_doctor()
    report = {
        "ci_mode": True,
        "repo_root": "/repo",
        "platform": "linux",
        "python": {"ok": True, "executable": "python3", "version": "3.12"},
        "ffmpeg": {"ok": True, "path": "ffmpeg"},
        "entrypoint": {"ok": True, "fresh_shell_path": "the-oracle", "path_entrypoint": "", "venv_entrypoint": "", "fresh_shell_error": "", "help_error": ""},
        "chatterbox_import": {"ok": True, "target": "x", "error": ""},
        "chatterbox_init": {"ok": True, "seconds": 1.0, "skipped": False, "error": ""},
        "perth": {"ok": True, "watermarker_symbol": "w", "error": ""},
        "turbo": {"ok": True, "checkpoint_dir": "", "error": ""},
        "voice_sources": {
            "ok": True,
            "default_voice_assessment": "ok",
            "seashell_clip_count": 2,
            "fallback_clip_count": 0,
            "better_local_assets_detail": "",
            "voice_mixing_detail": "",
        },
        "qt": {"ok": True, "plugin_path": "", "qt_platform": "offscreen", "error": "", "missing_libraries": [], "suggested_packages": [], "offscreen_error": "", "ldd_error": ""},
        "deterministic_smoke": {"ok": True, "output_path": "/o", "error": ""},
        "real_engine_smoke": {"ok": True, "expected_paths": {"output": "/r"}, "error": ""},
        "vulkan_backend": {
            "ok": True,
            "binary_built": True,
            "model_override_set": True,
            "model_file_exists": True,
            "model_path": "/models/chatterbox.gguf",
            "vulkan_device": True,
            "device_name": "GPU A",
            "rdna1_device": False,
            "vendored_patch_applied": True,
            "device_index_env": "",
            "threads_env": "",
            "audio_cpp_devices": [
                {"index": 0, "name": "GPU A"},
                {"index": 1, "name": "GPU B"},
            ],
            "caveat": "",
            "error": "",
        },
        "next_steps": [],
    }

    doctor._print_human_report(report)

    captured = capsys.readouterr().out
    assert "audio.cpp Vulkan devices" in captured
    assert "0=GPU A" in captured
    assert "1=GPU B" in captured
    assert "ORACLE_AUDIOCPP_DEVICE" in captured


def test_vulkan_backend_status_reports_device_and_threads_env(monkeypatch, tmp_path: Path) -> None:
    doctor = _load_doctor()
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("ORACLE_AUDIOCPP_CLI", str(binary))
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", "/models/chatterbox")
    monkeypatch.setenv("ORACLE_AUDIOCPP_DEVICE", "1")
    monkeypatch.setenv("ORACLE_AUDIOCPP_THREADS", "8")
    monkeypatch.setattr(doctor, "_vulkaninfo_summary", lambda: (True, "deviceName = some gpu"))
    monkeypatch.setattr(doctor, "_vulkan_patches_applied", lambda repo_root: True)

    status = doctor._vulkan_backend_status(tmp_path)

    assert status["device_index_env"] == "1"
    assert status["threads_env"] == "8"


def test_vulkan_backend_status_reports_effective_batch_cap(monkeypatch, tmp_path: Path) -> None:
    """The doctor reports the effective per-subprocess batch cap: the env var
    value when set (the engine's single source of truth), else the default 32."""
    doctor = _load_doctor()
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("ORACLE_AUDIOCPP_CLI", str(binary))
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", "/models/chatterbox")
    monkeypatch.setattr(doctor, "_vulkaninfo_summary", lambda: (True, "deviceName = some gpu"))
    monkeypatch.setattr(doctor, "_vulkan_patches_applied", lambda repo_root: True)

    monkeypatch.delenv("ORACLE_AUDIOCPP_MAX_BATCH", raising=False)
    status = doctor._vulkan_backend_status(tmp_path)
    assert status["batch_env"] == ""
    assert status["effective_batch_cap"] == 32

    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "16")
    status = doctor._vulkan_backend_status(tmp_path)
    assert status["batch_env"] == "16"
    assert status["effective_batch_cap"] == 16

    # Out-of-range values are clamped exactly like the engine (>= 1).
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "0")
    status = doctor._vulkan_backend_status(tmp_path)
    assert status["effective_batch_cap"] == 1


def test_next_steps_warns_when_batch_cap_out_of_range(monkeypatch) -> None:
    """A batch cap outside the sensible 1-128 range must be surfaced as a
    next step (a tiny cap destroys batching; a huge one risks an oversized
    requests.json) -- and a sane cap produces no warning."""
    doctor = _load_doctor()

    def _report_with_cap(batch_env: str, effective: int) -> dict:
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
                "binary_built": True,
                "model_override_set": True,
                "model_file_exists": True,
                "vendored_patch_applied": True,
                "device_index_env": "",
                "audio_cpp_devices": [],
                "batch_env": batch_env,
                "effective_batch_cap": effective,
            },
        }

    steps = doctor._build_next_steps(_report_with_cap("512", 512), ci_mode=True)
    batch_steps = [step for step in steps if "batch cap" in step]
    assert batch_steps, "expected a warning for the 512 cap"
    assert "512" in batch_steps[0]
    assert "1-128" in batch_steps[0]

    # A sub-1 raw value silently clamps to 1 (the engine's effective cap) but
    # must still warn: the raw env value is the user's actual intent.
    steps = doctor._build_next_steps(_report_with_cap("0", 1), ci_mode=True)
    batch_steps = [step for step in steps if "batch cap" in step]
    assert batch_steps, "expected a warning for the raw cap of 0"
    assert "0" in batch_steps[0]

    steps = doctor._build_next_steps(_report_with_cap("-5", 1), ci_mode=True)
    batch_steps = [step for step in steps if "batch cap" in step]
    assert batch_steps, "expected a warning for the raw cap of -5"

    steps = doctor._build_next_steps(_report_with_cap("32", 32), ci_mode=True)
    assert not any("batch cap" in step for step in steps)

    steps = doctor._build_next_steps(_report_with_cap("", 32), ci_mode=True)
    assert not any("batch cap" in step for step in steps)

    # Non-numeric env values are ignored by the engine (falls back to 32) and
    # must not warn.
    steps = doctor._build_next_steps(_report_with_cap("bogus", 32), ci_mode=True)
    assert not any("batch cap" in step for step in steps)


def test_human_report_lists_batch_cap(monkeypatch, capsys) -> None:
    """The human report shows the effective batch cap when the env var is set."""
    doctor = _load_doctor()
    report = {
        "ci_mode": True,
        "repo_root": "/repo",
        "platform": "linux",
        "python": {"ok": True, "executable": "python3", "version": "3.12"},
        "ffmpeg": {"ok": True, "path": "ffmpeg"},
        "entrypoint": {"ok": True, "fresh_shell_path": "the-oracle", "path_entrypoint": "", "venv_entrypoint": "", "fresh_shell_error": "", "help_error": ""},
        "chatterbox_import": {"ok": True, "target": "x", "error": ""},
        "chatterbox_init": {"ok": True, "seconds": 1.0, "skipped": False, "error": ""},
        "perth": {"ok": True, "watermarker_symbol": "w", "error": ""},
        "turbo": {"ok": True, "checkpoint_dir": "", "error": ""},
        "voice_sources": {"ok": True, "default_voice_assessment": "ok", "seashell_clip_count": 2, "fallback_clip_count": 0, "better_local_assets_detail": "", "voice_mixing_detail": ""},
        "qt": {"ok": True, "plugin_path": "", "qt_platform": "offscreen", "error": "", "missing_libraries": [], "suggested_packages": [], "offscreen_error": "", "ldd_error": ""},
        "deterministic_smoke": {"ok": True, "output_path": "/o", "error": ""},
        "real_engine_smoke": {"ok": True, "expected_paths": {"output": "/r"}, "error": ""},
        "vulkan_backend": {
            "ok": True,
            "binary_built": True,
            "model_override_set": True,
            "model_file_exists": True,
            "model_path": "/models/chatterbox.gguf",
            "vulkan_device": True,
            "device_name": "GPU A",
            "rdna1_device": False,
            "vendored_patch_applied": True,
            "device_index_env": "",
            "threads_env": "",
            "batch_env": "64",
            "effective_batch_cap": 64,
            "audio_cpp_devices": [{"index": 0, "name": "GPU A"}],
            "caveat": "",
            "error": "",
        },
        "next_steps": [],
    }

    doctor._print_human_report(report)

    captured = capsys.readouterr().out
    assert "batch cap=64 (env)" in captured


def test_vulkan_backend_status_not_ready_without_model_env(monkeypatch, tmp_path: Path) -> None:
    """A built binary with no ORACLE_AUDIOCPP_MODEL must not count as ready."""
    doctor = _load_doctor()
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("ORACLE_AUDIOCPP_CLI", str(binary))
    monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
    monkeypatch.setattr(doctor, "_vulkaninfo_summary", lambda: (True, "deviceName = some gpu"))
    monkeypatch.setattr(doctor, "_vulkan_patches_applied", lambda repo_root: True)

    status = doctor._vulkan_backend_status(tmp_path)

    assert status["ok"] is False
    assert status["binary_built"] is True
    assert status["model_override_set"] is False
    assert status["vulkan_device"] is True
