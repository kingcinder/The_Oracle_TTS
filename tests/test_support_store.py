"""Draft storage: atomic, capped, and incapable of escaping its directory.

A report id is the only thing that names a file, so an id that is not one of
ours must fail before any path is built. Retention is explicit: the oldest
drafts are dropped at the cap and the ids come back to the caller.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from the_oracle import support


def _report(tmp_path: Path, *, offset_minutes: int = 0, title: str = "Render stops at the last line"):
    moment = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=offset_minutes)
    return support.new_report(
        str(tmp_path), title=title, summary="Something went wrong.", run_doctor=False, now=moment
    )


def test_save_load_and_list_newest_first(tmp_path: Path) -> None:
    first = support.save_report(tmp_path, _report(tmp_path, offset_minutes=0, title="First failure here"))
    second = support.save_report(tmp_path, _report(tmp_path, offset_minutes=5, title="Second failure here"))
    assert first.is_file() and second.is_file()
    assert first.parent == support.store_dir(tmp_path)
    reports = support.list_reports(tmp_path)
    assert [report.title for report in reports] == ["Second failure here", "First failure here"]
    assert support.load_report(tmp_path, reports[0].report_id) == reports[0]


def test_empty_store_lists_nothing(tmp_path: Path) -> None:
    assert support.list_reports(tmp_path) == []


def test_delete_reports_what_it_did(tmp_path: Path) -> None:
    report = _report(tmp_path)
    support.save_report(tmp_path, report)
    assert support.delete_report(tmp_path, report.report_id) is True
    assert support.delete_report(tmp_path, report.report_id) is False
    assert support.list_reports(tmp_path) == []


def test_corrupt_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    good = _report(tmp_path)
    support.save_report(tmp_path, good)
    (support.store_dir(tmp_path) / "br-20260101-000000-dead.json").write_text("{not json", encoding="utf-8")
    reports = support.list_reports(tmp_path)
    assert [report.report_id for report in reports] == [good.report_id]


def test_id_with_path_segments_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid report id"):
        support.load_report(tmp_path, "../../crash_reports/x")
    with pytest.raises(ValueError, match="invalid report id"):
        support.delete_report(tmp_path, "br-20260101-000000-abcdef/../x")


def test_retention_cap_drops_the_oldest_and_reports_them(tmp_path: Path) -> None:
    for index in range(support.MAX_REPORTS + 2):
        support.save_report(
            tmp_path,
            _report(tmp_path, offset_minutes=index, title=f"Failure number {index:02d} here"),
        )
    reports = support.list_reports(tmp_path)
    assert len(reports) == support.MAX_REPORTS
    assert support.prune_reports(tmp_path) == []
    support.save_report(
        tmp_path,
        _report(tmp_path, offset_minutes=999, title="Newest failure in the store"),
    )
    assert len(support.list_reports(tmp_path)) == support.MAX_REPORTS


def test_saved_json_is_human_readable_and_complete(tmp_path: Path) -> None:
    report = _report(tmp_path)
    path = support.save_report(tmp_path, report)
    text = path.read_text(encoding="utf-8")
    assert '"report_version": 1' in text
    assert '"title": "Render stops at the last line"' in text
    assert "diagnostics" in text
    assert not path.with_name(path.name + ".tmp").exists()
