"""Tests for the doctor's release-metadata check (scripts/doctor.py).

The doctor surfaces release-metadata drift — the single-source version
invariant that ``scripts/release.py --check`` enforces (pyproject, the
README/STATE banners, and the CHANGELOG section for the current version,
dated the release day) — so a release attempt fails on prepared ground
instead of mid-flight. The probes are reused, never reimplemented:
``release.py`` is loaded from beside ``doctor.py`` and its ``check()``
runs read-only. These tests pin the probe, the read-only contract, the
human-report rendering (including older reports missing the key), the
next-steps wiring, and the verdict policy: the check deliberately does
NOT join ``required_checks``, because a changelog not dated today is the
release-day gate working as designed — not a broken install.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

DOCTOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "doctor.py"
RELEASE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_doctor():
    spec = importlib.util.spec_from_file_location("oracle_doctor", DOCTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_release():
    spec = importlib.util.spec_from_file_location("oracle_release_tool_probe", RELEASE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- fake repo ----------------------------------------------------------------


def _write_consistent_repo(
    tmp_path: Path,
    *,
    version: str = "9.9.9",
    readme_version: str | None = None,
    changelog_date: str | None = None,
) -> Path:
    """A minimal tree that satisfies the full release invariant (dated today)."""
    (tmp_path / "src" / "the_oracle").mkdir(parents=True)
    (tmp_path / "src" / "the_oracle" / "__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "the-oracle"\ndynamic = ["version"]\n'
        '[tool.setuptools.dynamic]\nversion = { attr = "the_oracle.__version__" }\n',
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        f"# The Oracle\n\n**Release V{readme_version or version}** — notes.\n",
        encoding="utf-8",
    )
    (tmp_path / "STATE.md").write_text(
        f"# State\n\n**Current release: V{readme_version or version} (notes)**\n",
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n### Changed\n\n- (nothing yet)\n\n"
        f"## [{version}] — {changelog_date or date.today().isoformat()}\n\n### Added\n\n- Something.\n",
        encoding="utf-8",
    )
    return tmp_path


# --- the probe ----------------------------------------------------------------


def test_probe_passes_on_a_consistent_repo(tmp_path: Path) -> None:
    doctor = _load_doctor()
    repo = _write_consistent_repo(tmp_path)

    status = doctor._release_metadata_status(repo)

    assert status["ok"] is True
    assert status["problems"] == []
    assert status["version"] == "9.9.9"
    assert status["error"] == ""


def test_probe_surfaces_changelog_drift(tmp_path: Path) -> None:
    doctor = _load_doctor()
    yesterday = date.fromordinal(date.today().toordinal() - 1).isoformat()
    repo = _write_consistent_repo(tmp_path, changelog_date=yesterday)

    status = doctor._release_metadata_status(repo)

    assert status["ok"] is False
    assert len(status["problems"]) == 1
    assert "CHANGELOG.md" in status["error"]
    assert yesterday in status["error"]


def test_probe_surfaces_banner_drift_too(tmp_path: Path) -> None:
    """The probe is the whole --check invariant, not just the changelog leg."""
    doctor = _load_doctor()
    repo = _write_consistent_repo(tmp_path, readme_version="8.8.8")

    status = doctor._release_metadata_status(repo)

    assert status["ok"] is False
    assert any("README.md" in problem for problem in status["problems"])


def test_probe_agrees_with_release_check_on_the_real_repo() -> None:
    """Cross-module pin: the doctor's verdict is byte-identical to running
    release.check() directly against this repo."""
    doctor = _load_doctor()
    release = _load_release()

    status = doctor._release_metadata_status(REPO_ROOT)

    assert status["version"] == release.read_version(REPO_ROOT)
    assert status["problems"] == release.check(REPO_ROOT)


def test_probe_survives_a_broken_release_module(tmp_path: Path, monkeypatch) -> None:
    """A broken probe must degrade to a failed check, never crash the doctor."""
    doctor = _load_doctor()
    repo = _write_consistent_repo(tmp_path)
    monkeypatch.setattr("importlib.util.spec_from_file_location", lambda *args, **kwargs: None)

    status = doctor._release_metadata_status(repo)

    assert status["ok"] is False
    assert "could not be loaded" in status["error"]


def test_probe_is_read_only(tmp_path: Path) -> None:
    """No file may be created or modified by the probe (same contract the
    whole-tree doctor gate enforces for the rest of the report)."""
    doctor = _load_doctor()
    repo = _write_consistent_repo(tmp_path)

    def snapshot(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    before = snapshot(repo)
    doctor._release_metadata_status(repo)
    assert snapshot(repo) == before


# --- human report ---------------------------------------------------------------


def _full_report(release_metadata: dict) -> dict:
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
        "input_subtitles": {"ok": True, "exists": True, "scanned": 0, "utf8_count": 0, "fallback": [], "blocked": [], "unreadable": [], "input_dir": "/repo/Input"},
        "release_metadata": release_metadata,
        "next_steps": [],
    }


def test_human_report_pass_line(capsys) -> None:
    doctor = _load_doctor()
    report = _full_report({"ok": True, "version": "9.9.9", "problems": [], "error": ""})

    doctor._print_human_report(report)
    out = capsys.readouterr().out

    assert "PASS Release metadata: version 9.9.9" in out
    assert "CHANGELOG section dated today" in out


def test_human_report_renders_drift_detail(capsys) -> None:
    doctor = _load_doctor()
    report = _full_report(
        {
            "ok": False,
            "version": "9.9.9",
            "problems": ["CHANGELOG.md:9: the 9.9.9 section is dated 2026-01-01 but today is 2026-09-21"],
            "error": "",
        }
    )

    doctor._print_human_report(report)
    out = capsys.readouterr().out

    assert "WARN Release metadata drift:" in out
    assert "      CHANGELOG.md:9: the 9.9.9 section is dated 2026-01-01 but today is 2026-09-21" in out


def test_human_report_handles_older_reports_without_the_key(capsys) -> None:
    """A report from before this check existed must render without a
    misleading line, exactly like the input_subtitles contract."""
    doctor = _load_doctor()
    report = _full_report({"ok": True, "version": "9.9.9", "problems": [], "error": ""})
    del report["release_metadata"]

    doctor._print_human_report(report)
    out = capsys.readouterr().out

    assert "Release metadata" not in out


# --- next steps wiring -----------------------------------------------------------


def _minimal_report_for_next_steps(release_metadata: dict) -> dict:
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
        "input_subtitles": {"blocked": [], "unreadable": []},
        "release_metadata": release_metadata,
    }


def test_next_steps_flag_drift() -> None:
    doctor = _load_doctor()
    report = _minimal_report_for_next_steps(
        {"ok": False, "problems": ["CHANGELOG.md: no section for version 1.2.1"], "error": "", "version": "1.2.1"}
    )

    steps = doctor._build_next_steps(report, ci_mode=True)

    assert any("Release metadata: CHANGELOG.md: no section for version 1.2.1" in step for step in steps)


def test_next_steps_quiet_when_consistent() -> None:
    doctor = _load_doctor()
    report = _minimal_report_for_next_steps({"ok": True, "problems": [], "error": "", "version": "9.9.9"})

    steps = doctor._build_next_steps(report, ci_mode=True)

    assert not any("Release metadata" in step for step in steps)


# --- wiring + verdict-policy guards ------------------------------------------------


def test_check_is_wired_into_the_report() -> None:
    """Vacuity guard: the probe must actually run in run()'s report."""
    doctor_src = DOCTOR_PATH.read_text(encoding="utf-8")
    assert '"release_metadata": _release_metadata_status(repo_root),' in doctor_src
    assert doctor_src.count("_release_metadata_status(") >= 2  # definition + call site


def test_check_does_not_join_required_checks() -> None:
    """Verdict-policy pin: drift must never gate overall_ready — a changelog
    not dated today is expected between releases, not a broken install."""
    doctor_src = DOCTOR_PATH.read_text(encoding="utf-8")
    start = doctor_src.index("required_checks = [")
    end = doctor_src.index("]", start)
    assert "release_metadata" not in doctor_src[start:end]
