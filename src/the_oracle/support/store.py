"""Local draft store for bug reports.

Reports live in ``support_reports/`` at the install root, one JSON file each,
alongside the other repo-local runtime artifacts (``logs/``,
``crash_reports/``, ``user_settings/``). The store is deliberately boring:
atomic writes, a hard cap, and no index file that can disagree with the disk.
The cap is enforced on save, oldest first, and the prune result is returned so
the CLI can tell the user instead of deleting silently.

Every id is validated before it touches a path: a report id is an opaque
filename, never a path a caller supplies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from the_oracle.support.report import BugReport, _ID_RE

STORE_DIRNAME = "support_reports"
MAX_REPORTS = 25
_SUFFIX = ".json"


def store_dir(root: str | Path) -> Path:
    return Path(root) / STORE_DIRNAME


def _report_path(root: str | Path, report_id: str) -> Path:
    if not _ID_RE.match(report_id):
        raise ValueError(f"invalid report id: {report_id!r}")
    return store_dir(root) / f"{report_id}{_SUFFIX}"


def save_report(root: str | Path, report: BugReport) -> Path:
    """Write one report atomically, then enforce the retention cap."""
    path = _report_path(root, report.report_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    prune_reports(root)
    return path


def load_report(root: str | Path, report_id: str) -> BugReport:
    path = _report_path(root, report_id)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"no bug report with id {report_id!r}") from None
    except OSError as error:
        raise ValueError(f"bug report {report_id!r} could not be read: {error}") from error
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"bug report {report_id!r} is not valid JSON: {error}") from error
    return BugReport.from_dict(data)


def list_reports(root: str | Path) -> list[BugReport]:
    """Every readable report, newest first. Unreadable files are skipped: one
    corrupt draft must not make the whole list unavailable."""
    directory = store_dir(root)
    if not directory.is_dir():
        return []
    reports: list[BugReport] = []
    for path in directory.glob(f"*{_SUFFIX}"):
        try:
            reports.append(BugReport.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, TypeError):
            continue
    reports.sort(key=lambda report: (report.created_utc, report.report_id), reverse=True)
    return reports


def delete_report(root: str | Path, report_id: str) -> bool:
    path = _report_path(root, report_id)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def prune_reports(root: str | Path, *, keep: int = MAX_REPORTS) -> list[str]:
    """Delete the oldest drafts beyond ``keep`` and return their ids."""
    reports = list_reports(root)
    if len(reports) <= keep:
        return []
    removed: list[str] = []
    for report in reports[keep:]:
        if delete_report(root, report.report_id):
            removed.append(report.report_id)
    return removed
