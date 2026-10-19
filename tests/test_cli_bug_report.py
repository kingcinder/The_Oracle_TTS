"""The ``bug-report`` CLI family: local drafts, packages, and links.

Every test points ``repo_root`` at a sandbox, so nothing touches the real
install. The pins are the promises made in --help: drafts stay local, delete
needs an explicit --yes, a package contains the report, and the issue command
prints links instead of opening anything.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import the_oracle.cli as cli
import the_oracle.offline as offline
from the_oracle import crash, support


@pytest.fixture()
def root(tmp_path: Path, monkeypatch) -> Path:
    install = tmp_path / "install"
    install.mkdir()
    monkeypatch.setattr(offline, "repo_root", lambda: install)
    return install


def _new(root: Path, *extra: str) -> dict:
    code = cli.main(
        [
            "bug-report",
            "new",
            "--title",
            "Preview closes the window",
            "--summary",
            "Clicking Preview exits the app.",
            "--step",
            "Open a project",
            "--step",
            "Click Preview",
            "--area",
            "gui",
            "--no-doctor",
            "--json",
            *extra,
        ]
    )
    assert code == 0
    return support.list_reports(root)[0].to_dict()


def test_new_writes_a_draft_and_prints_links(root: Path, capsys) -> None:
    _new(root)
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"]["title"] == "Preview closes the window"
    assert payload["issue_url"].startswith("https://github.com/")
    assert payload["mailto_url"].startswith("mailto:")
    assert Path(payload["path"]).is_file()


def test_short_title_exits_two_without_writing(root: Path, capsys) -> None:
    code = cli.main(["bug-report", "new", "--title", "bad", "--no-doctor"])
    assert code == 2
    assert "at least" in capsys.readouterr().err
    assert support.list_reports(root) == []


def test_list_and_show(root: Path, capsys) -> None:
    report = _new(root)
    capsys.readouterr()
    assert cli.main(["bug-report", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [item["report_id"] for item in listed] == [report["report_id"]]

    assert cli.main(["bug-report", "show", report["report_id"]]) == 0
    shown = capsys.readouterr().out
    assert f"# [The Oracle" in shown
    assert "## Steps to reproduce" in shown
    assert "Open a project" in shown


def test_show_unknown_id_is_an_error(root: Path, capsys) -> None:
    assert cli.main(["bug-report", "show", "br-20260101-000000-abcdef"]) == 1
    assert "no bug report" in capsys.readouterr().err


def test_package_includes_the_report_and_manifest(root: Path, capsys) -> None:
    report = _new(root)
    capsys.readouterr()
    assert cli.main(["bug-report", "package", report["report_id"], "--no-doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    path = Path(payload["path"])
    assert path.is_file()
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    assert {"report.md", "report.json", "diagnostics.json", "manifest.json"} <= names
    assert payload["crash_reports_included"] is False


def test_issue_prints_both_links_without_opening_anything(root: Path, capsys) -> None:
    report = _new(root)
    capsys.readouterr()
    assert cli.main(["bug-report", "issue", report["report_id"], "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["issue_url"].startswith("https://github.com/kingcinder/The_Oracle_TTS/")
    assert payload["mailto_url"].startswith("mailto:")
    assert "github.com" in payload["issue_url"]


def test_delete_requires_yes(root: Path, capsys) -> None:
    report = _new(root)
    capsys.readouterr()
    assert cli.main(["bug-report", "delete", report["report_id"]]) == 2
    assert "--yes" in capsys.readouterr().err
    assert support.list_reports(root)
    assert cli.main(["bug-report", "delete", report["report_id"], "--yes"]) == 0
    assert support.list_reports(root) == []


def test_delete_unknown_id_exits_one(root: Path) -> None:
    assert cli.main(["bug-report", "delete", "br-20260101-000000-abcdef", "--yes"]) == 1


def test_diagnose_is_redacted_and_serializable(root: Path, capsys) -> None:
    assert cli.main(["bug-report", "diagnose", "--no-doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    text = json.dumps(payload)
    assert str(Path.home()) not in text
    assert payload["app_version"]
    assert payload["environment"]["values_included"] is False


def test_new_can_reference_the_newest_crash_record(root: Path, capsys) -> None:
    crash.write_consent(root, True)
    crash.write_record(
        root, crash.build_record(exception_type="RuntimeError", exception_message="boom")
    )
    _new(root, "--crash")
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"]["crash_record"]


def test_crash_reference_to_unknown_record_exits_two(root: Path, capsys) -> None:
    crash.write_consent(root, True)
    crash.write_record(
        root, crash.build_record(exception_type="RuntimeError", exception_message="boom")
    )
    code = cli.main(
        [
            "bug-report",
            "new",
            "--title",
            "Preview closes the window",
            "--crash",
            "not-a-record.json",
            "--no-doctor",
        ]
    )
    assert code == 2
    assert "no crash record" in capsys.readouterr().err
    assert support.list_reports(root) == []


def test_no_diagnostics_keeps_the_snapshot_out(root: Path, capsys) -> None:
    report = _new(root, "--no-diagnostics")
    assert report["diagnostics"] == {}
    capsys.readouterr()
    assert cli.main(["bug-report", "show", report["report_id"]]) == 0
    assert "## Diagnostics" not in capsys.readouterr().out
