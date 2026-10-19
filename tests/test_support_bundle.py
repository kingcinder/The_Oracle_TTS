"""Support bundle: redaction, consent gating, doctor handling, and CLI wiring."""

from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from the_oracle import support_bundle


def _make_root(tmp_path: Path) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir()
    home = str(Path.home())
    (logs / "app.log").write_text(
        f"started in {home}/private/project\n"
        "license ORACLE1.aaa.bbb\n"
        "contact writer@example.com\n"
        "key sk-abcdefghijklmnopqrstuvwxyz\n",
        encoding="utf-8",
    )
    settings = tmp_path / "user_settings"
    settings.mkdir()
    (settings / "app_settings.json").write_text(
        json.dumps(
            {
                "theme": "Night Deck",
                "last_input_file": f"{home}/private/story.txt",
                "license_token": "ORACLE1.ccc.ddd",
                "nested": {"api_key": "secret", "backend": "pytorch"},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _read_zip(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_bundle_is_a_valid_zip_with_manifest_and_redaction(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    out = tmp_path / "bundle.zip"
    result = support_bundle.build_support_bundle(
        root,
        out,
        run_doctor=False,
        now=datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc),
    )
    assert result.path == out
    assert out.is_file()
    entries = _read_zip(out)
    assert "manifest.json" in entries
    assert "environment.json" in entries
    assert "license.json" in entries
    assert "settings.json" in entries
    manifest = json.loads(entries["manifest.json"])
    assert manifest["app_version"]
    assert manifest["bundle_version"] == support_bundle.BUNDLE_VERSION
    assert "logs/01-app.log" in manifest["included"]

    combined = b"\n".join(entries.values()).decode("utf-8", errors="replace")
    assert str(Path.home()) not in combined
    assert "ORACLE1." not in combined
    assert "writer@example.com" not in combined
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in combined
    assert "private/story.txt" not in combined
    assert str(Path(sys.executable)) not in combined


def test_license_payload_never_contains_the_token(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    out = tmp_path / "bundle.zip"
    support_bundle.build_support_bundle(root, out, run_doctor=False)
    entries = _read_zip(out)
    license_payload = json.loads(entries["license.json"])
    assert license_payload["token_included"] is False
    assert license_payload["licensee_included"] is False
    assert "ORACLE1." not in json.dumps(license_payload)
    assert "licensee" not in license_payload


def test_crash_reports_are_excluded_without_the_flag(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    crash_dir = root / "crash_reports"
    crash_dir.mkdir()
    (crash_dir / "crash-1.json").write_text('{"message": "boom"}', encoding="utf-8")
    (root / "crash_consent.json").write_text('{"crash_reports_enabled": true}', encoding="utf-8")
    out = tmp_path / "bundle.zip"
    result = support_bundle.build_support_bundle(root, out, run_doctor=False)
    entries = _read_zip(out)
    assert not any(name.startswith("crash_reports/") for name in entries)
    assert any("crash reports excluded" in note for note in result.skipped)


def test_crash_reports_are_excluded_when_consent_is_off(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    crash_dir = root / "crash_reports"
    crash_dir.mkdir()
    (crash_dir / "crash-1.json").write_text('{"message": "boom"}', encoding="utf-8")
    out = tmp_path / "bundle.zip"
    result = support_bundle.build_support_bundle(
        root, out, include_crash_reports=True, run_doctor=False
    )
    entries = _read_zip(out)
    assert not any(name.startswith("crash_reports/") for name in entries)
    assert any("consent is off" in note for note in result.skipped)


def test_crash_reports_are_included_with_consent_and_flag(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    crash_dir = root / "crash_reports"
    crash_dir.mkdir()
    (crash_dir / "crash-1.json").write_text(
        json.dumps({"message": "boom", "path": str(Path.home()) + "/private/x"}), encoding="utf-8"
    )
    (root / "crash_consent.json").write_text('{"crash_reports_enabled": true}', encoding="utf-8")
    out = tmp_path / "bundle.zip"
    result = support_bundle.build_support_bundle(
        root, out, include_crash_reports=True, run_doctor=False
    )
    entries = _read_zip(out)
    crash_entries = [name for name in entries if name.startswith("crash_reports/")]
    assert crash_entries
    assert str(Path.home()) not in b"\n".join(entries.values()).decode("utf-8", errors="replace")


def test_supplied_doctor_report_is_redacted_and_included(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    out = tmp_path / "bundle.zip"
    result = support_bundle.build_support_bundle(
        root,
        out,
        doctor_report={"verdict": "ok", "home": str(Path.home()), "token": "ORACLE1.x.y"},
        run_doctor=False,
    )
    entries = _read_zip(out)
    assert "doctor.json" in entries
    doctor = entries["doctor.json"].decode("utf-8")
    assert str(Path.home()) not in doctor
    assert "ORACLE1." not in doctor
    assert "doctor.json" in result.included


def test_no_doctor_is_recorded_as_skipped(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    out = tmp_path / "bundle.zip"
    result = support_bundle.build_support_bundle(root, out, run_doctor=False)
    assert any("doctor" in note for note in result.skipped)


def test_default_output_lands_under_output(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    result = support_bundle.build_support_bundle(root, run_doctor=False)
    assert result.path.parent == root / "Output"
    assert result.path.is_file()


def test_missing_output_parent_is_created(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    out = tmp_path / "nested" / "deeper" / "bundle.zip"
    result = support_bundle.build_support_bundle(root, out, run_doctor=False)
    assert result.path.is_file()


def test_cli_handler_prints_the_manifest(monkeypatch, tmp_path: Path, capsys) -> None:
    import the_oracle.cli as cli
    from the_oracle.support_bundle import SupportBundleResult

    calls: dict[str, object] = {}

    def fake_build(root, out, **kwargs):
        calls.update({"root": Path(root), "out": out, **kwargs})
        return SupportBundleResult(
            path=tmp_path / "bundle.zip",
            manifest={"app_version": "9.9.9", "included": [], "skipped": []},
            included=(),
            skipped=(),
        )

    monkeypatch.setattr("the_oracle.support_bundle.build_support_bundle", fake_build)
    monkeypatch.setattr("the_oracle.offline.repo_root", lambda: tmp_path)
    code = cli.main(["support-bundle", "--no-doctor", "--out", str(tmp_path / "bundle.zip"), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["app_version"] == "9.9.9"
    assert calls["run_doctor"] is False
    assert calls["include_crash_reports"] is False
