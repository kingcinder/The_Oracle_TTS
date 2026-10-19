"""Report packaging: the report rides inside the redacted bundle.

The bundle's consent gate must survive composition: asking for a report
package with crash reports on a consent-off install still yields zero crash
records. Attachments are archive members, so names are validated before they
are written, and the manifest lists them as attachments.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from the_oracle import crash, support
from the_oracle import support_bundle


def _root(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "oracle.log").write_text("hello\n", encoding="utf-8")
    return tmp_path


def _report(root: Path):
    return support.new_report(
        str(root),
        title="Export fails with an empty file",
        summary="The FLAC is zero bytes.",
        steps=["Render", "Export"],
        area="audio",
        severity="high",
        run_doctor=False,
    )


def _entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_package_contains_report_diagnostics_and_bundle(tmp_path: Path) -> None:
    root = _root(tmp_path / "install")
    root.mkdir(parents=True, exist_ok=True)
    report = _report(root)
    result = support.build_report_package(root, report, run_doctor=False)
    assert result.path.is_file()
    entries = _entries(result.path)
    assert {"report.md", "report.json", "diagnostics.json", "manifest.json"} <= set(entries)
    manifest = json.loads(entries["manifest.json"])
    assert manifest["bundle_version"] == support_bundle.BUNDLE_VERSION
    assert manifest["attachments"] == ["diagnostics.json", "report.json", "report.md"]
    assert "report.md" in manifest["included"]
    assert report.report_id in entries["report.md"].decode("utf-8")
    assert json.loads(entries["report.json"])["report_id"] == report.report_id


def test_package_persists_the_report_with_its_bundle_name(tmp_path: Path) -> None:
    root = _root(tmp_path / "install")
    root.mkdir(parents=True, exist_ok=True)
    report = _report(root)
    result = support.build_report_package(root, report, run_doctor=False)
    stored = support.load_report(root, report.report_id)
    assert stored.bundle_path == result.path.name
    assert stored.updated_utc >= stored.created_utc


def test_crash_records_need_consent_even_when_requested(tmp_path: Path) -> None:
    root = _root(tmp_path / "install")
    root.mkdir(parents=True, exist_ok=True)
    crash.write_record(
        root,
        crash.build_record(exception_type="RuntimeError", exception_message="boom"),
    )
    report = _report(root)
    result = support.build_report_package(
        root, report, include_crash_reports=True, run_doctor=False
    )
    names = _entries(result.path)
    assert not any(name.startswith("crash_reports/") for name in names)
    assert result.crash_reports_included is False


def test_crash_records_are_included_with_consent(tmp_path: Path) -> None:
    root = _root(tmp_path / "install")
    root.mkdir(parents=True, exist_ok=True)
    crash.write_consent(root, True)
    crash.write_record(
        root, crash.build_record(exception_type="RuntimeError", exception_message="boom")
    )
    report = _report(root)
    result = support.build_report_package(
        root, report, include_crash_reports=True, run_doctor=False
    )
    assert result.crash_reports_included is True
    assert any(name.startswith("crash_reports/") for name in _entries(result.path))


def test_no_diagnostics_means_no_refresh(tmp_path: Path) -> None:
    root = _root(tmp_path / "install")
    root.mkdir(parents=True, exist_ok=True)
    report = support.new_report(
        str(root),
        title="No diagnostics please",
        include_diagnostics=False,
        run_doctor=False,
    )
    result = support.build_report_package(
        root, report, refresh_diagnostics=False, run_doctor=False
    )
    assert report.diagnostics == {}
    assert json.loads(_entries(result.path)["report.json"])["diagnostics"] == {}


def test_attachment_names_cannot_rewrite_the_manifest(tmp_path: Path) -> None:
    root = _root(tmp_path / "install")
    root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="reserved"):
        support_bundle.build_support_bundle(
            root, tmp_path / "x.zip", run_doctor=False, extra_entries={"manifest.json": b"{}"}
        )
    with pytest.raises(ValueError, match="safe relative path"):
        support_bundle.build_support_bundle(
            root, tmp_path / "y.zip", run_doctor=False, extra_entries={"../escape.txt": b"x"}
        )
    with pytest.raises(TypeError, match="must be bytes"):
        support_bundle.build_support_bundle(
            root, tmp_path / "z.zip", run_doctor=False, extra_entries={"note.txt": "text"}
        )


def test_extra_entries_are_listed_and_redaction_still_applies(tmp_path: Path) -> None:
    root = _root(tmp_path / "install")
    root.mkdir(parents=True, exist_ok=True)
    result = support_bundle.build_support_bundle(
        root,
        tmp_path / "bundle.zip",
        run_doctor=False,
        extra_entries={"notes/readme.txt": b"safe"},
    )
    assert "notes/readme.txt" in result.included
    assert result.manifest["attachments"] == ["notes/readme.txt"]
