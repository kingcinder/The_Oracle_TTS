"""The crash-record schema — CRASH_TELEMETRY_DESIGN §4's table, enforced.

Every field is built here, and every free-text field passes through
``sanitize`` on its way in. The table's "never contains" column is what the
tests mutate against: no absolute paths, no usernames, no transcript
content, no variable values, no raw machine identifiers.
"""

from __future__ import annotations

import datetime as dt
import sys

from the_oracle.crash.sanitize import sanitize_frame, sanitize_log_tail, sanitize_text

RECORD_VERSION = 1


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_record(
    *,
    exception_type: str,
    exception_message: str,
    traceback_frames: list[tuple[str, str, int]] | None = None,
    log_tail: list[str] | None = None,
    gpu_backend: str | None = None,
    gpu_device: str | None = None,
    edition: str | None = None,
    qt_version: str | None = None,
    thread_name: str | None = None,
) -> dict[str, object]:
    """Assemble one crash record. Pure-ish: reads only sys/platform constants,
    the package version, and its arguments — never the user's documents."""
    from the_oracle import __version__

    record: dict[str, object] = {
        "record_version": RECORD_VERSION,
        "timestamp": _utc_timestamp(),
        "app_version": __version__,
        "platform": {
            "os": sys.platform,
            "python": sys.version.split()[0],
        },
        "exception": {
            "type": sanitize_text(str(exception_type)[:120]),
            "message": sanitize_text(exception_message),
        },
        "traceback": [
            sanitize_frame(filename, function, lineno)
            for filename, function, lineno in (traceback_frames or [])
        ],
        "log_tail": sanitize_log_tail(log_tail or []),
    }
    if qt_version:
        record["platform"]["qt"] = qt_version
    if thread_name:
        record["thread"] = sanitize_text(thread_name)[:80]
    if gpu_backend or gpu_device:
        record["gpu_backend"] = {
            "name": sanitize_text(gpu_backend or "")[:80],
            "device": sanitize_text(gpu_device or "")[:120],
        }
    if edition:
        record["edition"] = sanitize_text(edition)[:40]
    return record
