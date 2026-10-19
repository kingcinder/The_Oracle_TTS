"""One zip that a customer can attach: report, diagnostics, and bundle.

This composes rather than duplicates: ``build_support_bundle`` already
collects the redacted environment, settings, logs, doctor report, and
consent-gated crash records. Here the report itself rides along as three
extra entries (Markdown, JSON, and the full diagnostics snapshot), and the
bundle manifest records them as attachments so the archive describes itself
truthfully.

Crash records keep the bundle's consent gate. A report may name one crash
record for context; that is a pointer, not a bypass — without consent the
package contains no crash data at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from the_oracle.support.channels import issue_url, mailto_url
from the_oracle.support.diagnostics import collect_diagnostics
from the_oracle.support.report import BugReport, report_json, report_markdown
from the_oracle.support.store import save_report
from the_oracle.support_bundle import (
    SupportBundleResult,
    build_support_bundle,
    run_doctor_report,
)


@dataclass(frozen=True, slots=True)
class ReportPackageResult:
    path: Path
    report: BugReport
    bundle: SupportBundleResult
    issue_url: str
    mailto_url: str
    crash_reports_included: bool


def build_report_package(
    root: str | Path,
    report: BugReport,
    out_path: str | Path | None = None,
    *,
    include_crash_reports: bool = False,
    refresh_diagnostics: bool = True,
    doctor_report: dict[str, Any] | None = None,
    run_doctor: bool = True,
    now: datetime | None = None,
) -> ReportPackageResult:
    """Build the report zip, persist the report, and return the share links."""
    root = Path(root).resolve()
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    doctor_attempted = False
    if run_doctor and doctor_report is None:
        doctor_report, _problem = run_doctor_report(root)
        doctor_attempted = True
    if refresh_diagnostics:
        report.diagnostics = collect_diagnostics(
            root,
            doctor_report=doctor_report,
            run_doctor=run_doctor and not doctor_attempted,
            now=moment,
        )
    report.updated_utc = moment.astimezone(timezone.utc).isoformat()

    if out_path is None:
        output = root / "Output" / f"oracle-report-{report.report_id}.zip"
    else:
        output = Path(out_path).resolve()
    report.bundle_path = output.name

    extra_entries = {
        "report.md": report_markdown(report).encode("utf-8"),
        "report.json": report_json(report).encode("utf-8"),
        "diagnostics.json": json.dumps(report.diagnostics, indent=2, ensure_ascii=False).encode(
            "utf-8"
        ),
    }
    bundle = build_support_bundle(
        root,
        output,
        include_crash_reports=include_crash_reports,
        doctor_report=doctor_report,
        run_doctor=run_doctor and not doctor_attempted,
        extra_entries=extra_entries,
        now=moment,
    )
    save_report(root, report)
    return ReportPackageResult(
        path=bundle.path,
        report=report,
        bundle=bundle,
        issue_url=issue_url(report),
        mailto_url=mailto_url(report),
        crash_reports_included=any(
            name.startswith("crash_reports/") for name in bundle.included
        ),
    )
