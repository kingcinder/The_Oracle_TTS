"""Diagnostics: useful, JSON-safe, and free of identifying paths.

The collector is the debug-data half of the support layer. Its doctor input is
injected here so the suite never shells out; the privacy pins are the point —
no home directory, no environment values, no log contents, no recent-file
paths, and never a heavyweight import.
"""

from __future__ import annotations

import json
from pathlib import Path

from the_oracle.support import collect_diagnostics, compact_diagnostics


def _root(tmp_path: Path) -> Path:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "oracle.log").write_text(
        "started\nlicense ORACLE1.aaa.bbb\n", encoding="utf-8"
    )
    settings = tmp_path / "user_settings"
    settings.mkdir()
    (settings / "app_settings.json").write_text(
        json.dumps(
            {
                "theme": "Night Deck",
                "backend": "pytorch",
                "last_input_file": f"{Path.home()}/private/story.txt",
                "license_token": "ORACLE1.ccc.ddd",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_snapshot_has_the_sections_a_triager_needs(tmp_path: Path) -> None:
    diagnostics = collect_diagnostics(_root(tmp_path), run_doctor=False)
    for key in (
        "app_version",
        "system",
        "memory",
        "disk",
        "qt",
        "engines",
        "gpu",
        "models",
        "ffmpeg",
        "settings",
        "logs",
        "crash",
        "environment",
        "doctor",
        "redaction",
    ):
        assert key in diagnostics, key
    assert diagnostics["system"]["python"]
    assert diagnostics["install"]["root_kind"] == "install"
    json.dumps(diagnostics)  # the snapshot must always serialize


def test_no_home_paths_or_environment_values_leak(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ORACLE_LICENSE_TOKEN", "ORACLE1.secret.value")
    diagnostics = collect_diagnostics(_root(tmp_path), run_doctor=False)
    text = json.dumps(diagnostics)
    assert str(Path.home()) not in text
    assert "private/story.txt" not in text
    assert "ORACLE1." not in text
    assert "ORACLE_LICENSE_TOKEN" in diagnostics["environment"]["variables_present"]
    assert "secret.value" not in text
    assert diagnostics["environment"]["values_included"] is False


def test_settings_summary_is_a_whitelist(tmp_path: Path) -> None:
    settings = collect_diagnostics(_root(tmp_path), run_doctor=False)["settings"]
    assert settings["theme"] == "Night Deck"
    assert settings["backend"] == "pytorch"
    assert "last_input_file" not in settings
    assert "license_token" not in settings


def test_log_summary_has_names_and_sizes_only(tmp_path: Path) -> None:
    logs = collect_diagnostics(_root(tmp_path), run_doctor=False)["logs"]
    assert logs["files"] == [{"name": "oracle.log", "bytes": len("started\nlicense ORACLE1.aaa.bbb\n")}]
    assert "content" not in json.dumps(logs)


def test_injected_doctor_report_is_summarised(tmp_path: Path) -> None:
    doctor = {"ok": False, "ci_mode": True, "venv": {"ok": False}, "audio": {"ok": True}}
    diagnostics = collect_diagnostics(_root(tmp_path), doctor_report=doctor, run_doctor=False)
    summary = diagnostics["doctor"]
    assert summary["ran"] is True
    assert summary["ok"] is False
    assert summary["failing_checks"] == ["venv"]


def test_doctor_can_be_recorded_as_excluded(tmp_path: Path) -> None:
    diagnostics = collect_diagnostics(_root(tmp_path), run_doctor=False)
    assert diagnostics["doctor"]["ran"] is False
    assert "excluded" in diagnostics["doctor"]["detail"]


def test_compact_view_keeps_the_environment_and_drops_the_noise(tmp_path: Path) -> None:
    compact = compact_diagnostics(collect_diagnostics(_root(tmp_path), run_doctor=False))
    assert "system" in compact and "doctor" in compact
    assert "settings" not in compact and "environment" not in compact
