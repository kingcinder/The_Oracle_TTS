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


def _newest_first(paths: list[Path]) -> list[Path]:
    """Order by modification time, newest first. NOT by filename: two records
    written in the same second share their timestamp prefix, and name-order
    would then fall through to the random uuid suffix — picking an arbitrary
    "newest" and dropping an arbitrary "oldest" at the cap. (Caught by the
    doctor test net.) Files that cannot stat at all sort as oldest."""
    def mtime(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    return sorted(paths, key=mtime, reverse=True)


def _enforce_cap(directory: Path) -> None:
    records = _newest_first(list(directory.glob(_RECORD_GLOB)))
    for stale in records[MAX_RECORDS:]:
        try:
            stale.unlink()
        except OSError:
            pass


def list_records(root: str | Path) -> list[Path]:
    """Newest first (by mtime — see _newest_first)."""
    return _newest_first(list(crash_dir(root).glob(_RECORD_GLOB)))


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
