"""The bug-report model: validation, redaction, rendering, and share links.

The report is the customer-facing half of the support layer, so the pins are
privacy-shaped: user text must not smuggle the home directory or a license
token into a shared Markdown file, and a report without a usable title must
never be saved. Links are strings only — building one contacts nothing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pytest

from the_oracle import support
from the_oracle.support import channels, report as report_module


def _report(tmp_path: Path, **overrides):
    values = {
        "title": "Render stops near the last line",
        "summary": "The progress bar freezes at 92 percent.",
        "steps": ["Open a two-speaker script", "Click Render", "Wait for the last line"],
        "expected": "A finished FLAC",
        "actual": "The render never finishes",
        "area": "render",
        "severity": "high",
        "run_doctor": False,
    }
    values.update(overrides)
    return support.new_report(str(tmp_path), **values)


def test_report_carries_diagnostics_and_fields(tmp_path: Path) -> None:
    report = _report(tmp_path)
    assert report.report_id.startswith("br-")
    assert report.app_version
    assert report.diagnostics["diagnostics_version"] == 1
    assert report.diagnostics["system"]["python"]
    assert report.area == "render" and report.severity == "high"
    assert report.steps == [
        "Open a two-speaker script",
        "Click Render",
        "Wait for the last line",
    ]


def test_short_title_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least"):
        _report(tmp_path, title="oops")


def test_unknown_area_and_severity_fall_back_safely(tmp_path: Path) -> None:
    report = _report(tmp_path, area="not-a-real-area", severity="apocalyptic")
    assert report.area == "other"
    assert report.severity == "normal"
    assert support.validate_report(report) == []


def test_user_text_is_redacted_not_obeyed(tmp_path: Path) -> None:
    home = str(Path.home())
    report = _report(
        tmp_path,
        summary=f"Crash while reading {home}/private/story.txt",
        actual="The log printed ORACLE1.aaa.bbb and sk-abcdefghijklmnopqrstuvwxyz",
    )
    markdown = support.report_markdown(report)
    assert home not in markdown
    assert "ORACLE1." not in markdown
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in markdown
    assert "~/private/story.txt" in markdown


def test_contact_is_kept_for_reply_but_not_in_the_public_body(tmp_path: Path) -> None:
    report = _report(tmp_path, contact="writer@example.com")
    assert report.contact == "writer@example.com"
    body = support.report_markdown(report)
    assert "writer@example.com" not in body


def test_markdown_names_every_section(tmp_path: Path) -> None:
    report = _report(tmp_path)
    markdown = support.report_markdown(report)
    for heading in (
        "## Summary",
        "## Steps to reproduce",
        "## Expected behaviour",
        "## Actual behaviour",
        "## Diagnostics (redacted)",
    ):
        assert heading in markdown
    assert report.report_id in markdown
    assert "Nothing was uploaded" in markdown


def test_diagnostics_can_be_left_out_of_the_markdown(tmp_path: Path) -> None:
    report = _report(tmp_path)
    markdown = support.report_markdown(report, include_diagnostics=False)
    assert "## Diagnostics" not in markdown


def test_json_round_trip(tmp_path: Path) -> None:
    report = _report(tmp_path)
    restored = report_module.BugReport.from_dict(json.loads(support.report_json(report)))
    assert restored == report


def test_invalid_report_id_is_refused_not_used_as_a_path(tmp_path: Path) -> None:
    payload = _report(tmp_path).to_dict()
    payload["report_id"] = "../../etc/passwd"
    with pytest.raises(ValueError, match="invalid report id"):
        report_module.BugReport.from_dict(payload)


def test_issue_url_targets_the_tracked_repo_and_fits_url_limits(tmp_path: Path) -> None:
    report = _report(tmp_path)
    url = channels.issue_url(report)
    assert url.startswith(f"https://github.com/{channels.ISSUE_REPO}/issues/new?")
    assert "title=" in url and "labels=bug" in url
    assert len(url) <= channels.MAX_ISSUE_URL_CHARS


def test_pathological_report_still_yields_a_bounded_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long report falls back from the full body to the compact body, and
    then to a truncated head — always bounded, always carrying the id. The
    cap is lowered here so the fallback branch is exercised on a normal
    report instead of needing a pathological one."""
    report = _report(tmp_path, summary="x" * 3900)
    monkeypatch.setattr(channels, "MAX_ISSUE_URL_CHARS", 3000)
    url = channels.issue_url(report)
    assert len(url) <= 3000
    assert quote(report.report_id, safe="") in url
    assert "truncated" in url


def test_mailto_has_subject_report_id_and_support_address(tmp_path: Path) -> None:
    report = _report(tmp_path)
    url = channels.mailto_url(report)
    assert url.startswith(f"mailto:{channels.SUPPORT_EMAIL}?")
    assert "subject=" in url
    assert "Report%20ID" in url
    assert len(url) <= channels.MAX_MAILTO_URL_CHARS


def test_share_note_warns_that_nothing_was_sent(tmp_path: Path) -> None:
    note = channels.share_note(_report(tmp_path))
    assert "Nothing was sent" in note
    assert "bug-report package" in note


def test_new_report_uses_injected_clock(tmp_path: Path) -> None:
    moment = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    report = _report(tmp_path, now=moment)
    assert report.created_utc == "2026-10-19T12:00:00+00:00"
    assert report.report_id.startswith("br-20261019-120000-")
