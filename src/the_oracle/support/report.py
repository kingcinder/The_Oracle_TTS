"""The bug-report model: what a customer composes, previews, and shares.

A report is deliberately a plain JSON document plus a Markdown rendering. It
is written to the install's ``support_reports/`` folder, never uploaded, and
only leaves the machine when the user copies it, attaches it, or opens a
prefilled issue/email they chose to open.

User-authored free text passes through the same home/token/credential
redaction as the support bundle. That is a safety net, not a guarantee: the
Markdown preview is shown before sharing and the user is told to read it.
The contact field is the one exception — the user typed it so support can
reply, so it is stored verbatim and never published in the Markdown body
unless the user asked to include it.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from the_oracle import __version__
from the_oracle.support.diagnostics import collect_diagnostics, compact_diagnostics
from the_oracle.support_bundle import redact_text

REPORT_VERSION = 1
MIN_TITLE_CHARS = 8
MAX_TITLE_CHARS = 160
MAX_SUMMARY_CHARS = 4000
MAX_STEP_CHARS = 500
MAX_STEPS = 20
MAX_DETAIL_CHARS = 2000
MAX_CONTACT_CHARS = 200

AREAS = (
    "render",
    "gui",
    "ingest",
    "voice",
    "audio",
    "license",
    "install",
    "performance",
    "other",
)
SEVERITIES = ("low", "normal", "high", "blocking")

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{5,63}$")


def _clean(value: str, *, limit: int) -> str:
    return redact_text(str(value or "")).strip()[:limit]


def new_report_id(*, now: datetime | None = None) -> str:
    """A sortable, non-identifying id: UTC timestamp + random suffix."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return f"br-{moment.astimezone(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


@dataclass(slots=True)
class BugReport:
    report_id: str
    created_utc: str
    updated_utc: str
    title: str
    summary: str = ""
    steps: list[str] = field(default_factory=list)
    expected: str = ""
    actual: str = ""
    area: str = "other"
    severity: str = "normal"
    contact: str = ""
    app_version: str = __version__
    diagnostics: dict[str, Any] = field(default_factory=dict)
    crash_record: str | None = None
    bundle_path: str | None = None
    report_version: int = REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "report_id": self.report_id,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "app_version": self.app_version,
            "title": self.title,
            "summary": self.summary,
            "steps": list(self.steps),
            "expected": self.expected,
            "actual": self.actual,
            "area": self.area,
            "severity": self.severity,
            "contact": self.contact,
            "diagnostics": self.diagnostics,
            "crash_record": self.crash_record,
            "bundle_path": self.bundle_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BugReport:
        if not isinstance(data, dict):
            raise ValueError("a bug report must be a JSON object")
        report_id = str(data.get("report_id") or "")
        if not _ID_RE.match(report_id):
            raise ValueError(f"invalid report id: {report_id!r}")
        steps = data.get("steps") or []
        if not isinstance(steps, list):
            raise ValueError("a bug report's steps must be a list")
        try:
            report_version = int(data.get("report_version") or REPORT_VERSION)
        except (TypeError, ValueError):
            report_version = REPORT_VERSION
        return cls(
            report_id=report_id,
            created_utc=str(data.get("created_utc") or ""),
            updated_utc=str(data.get("updated_utc") or ""),
            title=str(data.get("title") or ""),
            summary=str(data.get("summary") or ""),
            steps=[str(step) for step in steps],
            expected=str(data.get("expected") or ""),
            actual=str(data.get("actual") or ""),
            area=str(data.get("area") or "other"),
            severity=str(data.get("severity") or "normal"),
            contact=str(data.get("contact") or ""),
            app_version=str(data.get("app_version") or __version__),
            diagnostics=data.get("diagnostics") if isinstance(data.get("diagnostics"), dict) else {},
            crash_record=data.get("crash_record") if isinstance(data.get("crash_record"), str) else None,
            bundle_path=data.get("bundle_path") if isinstance(data.get("bundle_path"), str) else None,
            report_version=report_version,
        )


def new_report(
    root: str,
    *,
    title: str,
    summary: str = "",
    steps: list[str] | None = None,
    expected: str = "",
    actual: str = "",
    area: str = "other",
    severity: str = "normal",
    contact: str = "",
    crash_record: str | None = None,
    include_diagnostics: bool = True,
    doctor_report: dict[str, Any] | None = None,
    run_doctor: bool = True,
    now: datetime | None = None,
) -> BugReport:
    """Assemble one draft. Raises ValueError when a field is unusable, so no
    invalid draft is ever written."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    report = BugReport(
        report_id=new_report_id(now=moment),
        created_utc=moment.astimezone(timezone.utc).isoformat(),
        updated_utc=moment.astimezone(timezone.utc).isoformat(),
        title=_clean(title, limit=MAX_TITLE_CHARS),
        summary=_clean(summary, limit=MAX_SUMMARY_CHARS),
        steps=[_clean(step, limit=MAX_STEP_CHARS) for step in (steps or []) if str(step).strip()][:MAX_STEPS],
        expected=_clean(expected, limit=MAX_DETAIL_CHARS),
        actual=_clean(actual, limit=MAX_DETAIL_CHARS),
        area=area if area in AREAS else "other",
        severity=severity if severity in SEVERITIES else "normal",
        contact=str(contact or "").strip()[:MAX_CONTACT_CHARS],
        crash_record=crash_record,
        diagnostics=(
            collect_diagnostics(
                root, doctor_report=doctor_report, run_doctor=run_doctor, now=moment
            )
            if include_diagnostics
            else {}
        ),
    )
    problems = validate_report(report)
    if problems:
        raise ValueError("; ".join(problems))
    return report


def validate_report(report: BugReport) -> list[str]:
    """Field-level problems that would make a report hard to act on."""
    problems: list[str] = []
    if not _ID_RE.match(report.report_id):
        problems.append(f"invalid report id {report.report_id!r}")
    if len(report.title.strip()) < MIN_TITLE_CHARS:
        problems.append(f"title must be at least {MIN_TITLE_CHARS} characters")
    if len(report.title) > MAX_TITLE_CHARS:
        problems.append(f"title must be at most {MAX_TITLE_CHARS} characters")
    if len(report.summary) > MAX_SUMMARY_CHARS:
        problems.append(f"summary must be at most {MAX_SUMMARY_CHARS} characters")
    if len(report.steps) > MAX_STEPS:
        problems.append(f"at most {MAX_STEPS} reproduction steps")
    if any(len(step) > MAX_STEP_CHARS for step in report.steps):
        problems.append(f"each step must be at most {MAX_STEP_CHARS} characters")
    if len(report.expected) > MAX_DETAIL_CHARS:
        problems.append(f"expected behaviour must be at most {MAX_DETAIL_CHARS} characters")
    if len(report.actual) > MAX_DETAIL_CHARS:
        problems.append(f"actual behaviour must be at most {MAX_DETAIL_CHARS} characters")
    if report.area not in AREAS:
        problems.append(f"area must be one of: {', '.join(AREAS)}")
    if report.severity not in SEVERITIES:
        problems.append(f"severity must be one of: {', '.join(SEVERITIES)}")
    if len(report.contact) > MAX_CONTACT_CHARS:
        problems.append(f"contact must be at most {MAX_CONTACT_CHARS} characters")
    return problems


def report_markdown(report: BugReport, *, include_diagnostics: bool = True) -> str:
    """Render the shareable Markdown. Deterministic for a given report."""
    lines: list[str] = [
        f"# [The Oracle {report.app_version}] {report.title}",
        "",
        f"- **Report ID:** `{report.report_id}`",
        f"- **Created (UTC):** {report.created_utc}",
        f"- **Area:** {report.area}",
        f"- **Severity:** {report.severity}",
    ]
    if report.crash_record:
        lines.append(f"- **Crash record:** `{report.crash_record}`")
    if report.bundle_path:
        lines.append(f"- **Support bundle:** `{report.bundle_path}`")
    lines.extend(["", "## Summary", "", report.summary or "_(not provided)_", ""])
    lines.extend(["## Steps to reproduce", ""])
    if report.steps:
        lines.extend(f"{index}. {step}" for index, step in enumerate(report.steps, start=1))
    else:
        lines.append("_(not provided)_")
    lines.extend(["", "## Expected behaviour", "", report.expected or "_(not provided)_", ""])
    lines.extend(["## Actual behaviour", "", report.actual or "_(not provided)_", ""])
    if include_diagnostics and report.diagnostics:
        compact = compact_diagnostics(report.diagnostics)
        lines.extend(
            [
                "## Diagnostics (redacted)",
                "",
                "```json",
                json.dumps(compact, indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "---",
            f"Generated by `the-oracle bug-report` from The Oracle {report.app_version}. "
            "Nothing was uploaded: this file is shared only when you choose to share it.",
            "",
        ]
    )
    return "\n".join(lines)


def report_json(report: BugReport) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
