"""Skip accounting for the test suite.

A skip is a claim that a test cannot run *here*. That claim is legitimate for
platform differences and for capabilities a developer machine may not have — but
in a green run it is indistinguishable from a test that quietly stopped covering
anything. That is exactly how the Vulkan smoke went untested in CI: ``audio.cpp``
was never built there, so the test skipped and the suite stayed green for months.

So skips are always reported, and a job that provisions its own requirements can
demand there be none: set ``ORACLE_FAIL_ON_SKIP=1`` (the CI job that builds the
Vulkan backend does) and any skip fails the session.

The reporting and the verdict live here rather than in ``conftest.py`` so they can
be exercised directly, without a subprocess run.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

#: Set this to make a skipped test fail the run. Truthy values are "1", "true",
#: "yes" and "on" (case-insensitive); anything else, or unset, leaves skips as
#: skips and merely reports them.
FAIL_ON_SKIP_ENV = "ORACLE_FAIL_ON_SKIP"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Used when pytest reports a skip with no usable explanation, so the audit never
#: shows a bare test id with no reason next to it.
NO_REASON = "(no reason given)"


@dataclass(frozen=True)
class SkipRecord:
    """One test that did not run, and why."""

    nodeid: str
    reason: str = NO_REASON


def fail_on_skip_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether this run demands that nothing be skipped."""
    source = os.environ if env is None else env
    return str(source.get(FAIL_ON_SKIP_ENV, "")).strip().lower() in _TRUTHY


def skip_reason(longrepr: object) -> str:
    """The human reason out of a pytest skip report's ``longrepr``.

    Pytest hands a skip's explanation over as ``(file, line, reason)``, but a
    collection-level skip or a custom reporter can hand over a string instead.
    """
    if longrepr is None:
        return NO_REASON
    if isinstance(longrepr, (tuple, list)) and len(longrepr) == 3:
        reason = str(longrepr[2]).strip()
        return reason or NO_REASON
    text = str(longrepr).strip()
    return text or NO_REASON


def dedupe(records: Iterable[SkipRecord]) -> list[SkipRecord]:
    """One record per test, keeping the first reason seen.

    A test can be reported more than once (setup and teardown both can skip), and
    a whole module skipped at collection time reports once for the module.
    """
    seen: dict[str, SkipRecord] = {}
    for record in records:
        seen.setdefault(record.nodeid, record)
    return sorted(seen.values(), key=lambda record: record.nodeid)


def _listing(records: Iterable[SkipRecord]) -> list[str]:
    """One display line per part of each record: id, then its reason indented."""
    lines: list[str] = []
    for record in records:
        lines.append(f"  {record.nodeid}")
        lines.append(f"      {record.reason}")
    return lines


def format_audit(records: Iterable[SkipRecord]) -> list[str]:
    """The lines that make skips visible in every run, empty when there are none."""
    unique = dedupe(records)
    if not unique:
        return []
    return [f"{len(unique)} test(s) did not run here:", *_listing(unique)]


def format_failure(records: Iterable[SkipRecord]) -> list[str]:
    """The lines explaining why a strict run failed.

    Addressed to whoever has to fix it, so it names the skipped tests, their
    reasons, and the two legitimate responses: provision the capability, or stop
    demanding it.
    """
    unique = dedupe(records)
    lines = [
        f"FAIL: {len(unique)} test(s) were skipped while {FAIL_ON_SKIP_ENV} is set.",
        f"{FAIL_ON_SKIP_ENV} means this run provides everything its tests need, so a skip",
        "here is a missing capability rather than a test that does not apply:",
    ]
    lines.extend(_listing(unique))
    lines.append(
        f"Either provision what the test needs, or drop {FAIL_ON_SKIP_ENV} from this job's"
        " environment if the skip is genuinely expected here."
    )
    return lines
