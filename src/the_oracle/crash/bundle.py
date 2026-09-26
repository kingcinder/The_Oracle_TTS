"""Crash-record persistence: one JSON file per event, directory capped.

Atomic write (temp + os.replace) so a crash mid-write cannot corrupt the
previous record. The directory holds at most MAX_RECORDS records: the oldest
matching file is removed only when the cap is exceeded, and only files this
unit named are ever candidates — never anything else the user keeps in the
folder.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from the_oracle.crash.sanitize import MAX_FIELD_CHARS

MAX_RECORDS = 20
#: A record that outgrows the byte cap has failed sanitization somewhere;
#: it is truncated at write time rather than stored oversized.
MAX_RECORD_BYTES = 32 * 1024
_RECORD_GLOB = "crash-*.json"


def crash_dir(root: str | Path) -> Path:
    return Path(root) / "crash_reports"


def write_record(root: str | Path, record: dict[str, object]) -> Path:
    """Persist one record atomically and enforce the directory cap."""
    directory = crash_dir(root)
    directory.mkdir(parents=True, exist_ok=True)

    # The filename itself carries no content — a UTC stamp for ordering and a
    # random suffix for uniqueness within the same second.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = directory / f"crash-{stamp}-{uuid.uuid4().hex[:8]}.json"

    body = json.dumps(record, indent=2, ensure_ascii=False)
    if len(body.encode("utf-8")) > MAX_RECORD_BYTES:
        # Keep the record truthful: truncate the largest text block rather
        # than silently dropping the whole event.
        record = dict(record)
        tail = record.get("log_tail")
        if isinstance(tail, list) and tail:
            record["log_tail"] = tail[: max(1, len(tail) // 2)]
        body = json.dumps(record, indent=2, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())

    _enforce_cap(directory)
    return path


def _enforce_cap(directory: Path) -> None:
    records = sorted(directory.glob(_RECORD_GLOB))
    for stale in records[:-MAX_RECORDS] if len(records) > MAX_RECORDS else []:
        try:
            stale.unlink()
        except OSError:
            pass


def list_records(root: str | Path) -> list[Path]:
    """Newest first."""
    return sorted(crash_dir(root).glob(_RECORD_GLOB), reverse=True)


def clear_records(root: str | Path) -> int:
    """Revocation story: delete every record this unit wrote. Returns count."""
    removed = 0
    for record in list_records(root):
        try:
            record.unlink()
            removed += 1
        except OSError:
            continue
    return removed
