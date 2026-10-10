"""Crash-record persistence: one JSON file per event, directory capped.

Atomic write (temp + os.replace) so a crash mid-write cannot corrupt the
previous record. The directory holds at most MAX_RECORDS records: the oldest
matching file is removed only when the cap is exceeded, and only files this
unit named are ever candidates — never anything else the user keeps in the
folder.

Also home to the park/restore pair that unattended offscreen runs (the
standalone certifier scripts) use to keep the D8 startup modal out of their
way: the startup branch opens a MODAL review whenever records exist, which
nothing in a headless sweep can answer.
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


# --- parking: keeping the D8 startup modal out of unattended runs -------------

#: A parked store is renamed to this prefix + the parking process's pid, in
#: the repo root next to the store it came from. In the root (not a temp
#: dir) on purpose: a sweep that dies skips every ``finally``, and a park a
#: later run can find is a park it can bring home — evidence is at worst
#: mislaid until the next parked run, never lost.
_PARK_PREFIX = "crash_reports.certifier-park-"


def _merge_dir(source: Path, destination: Path) -> None:
    """Move every child of ``source`` into ``destination``; lose nothing.

    A name already present in ``destination`` is byte-compared: an identical
    duplicate is dropped (the same record arriving by both paths), a
    differing one is moved beside it under ``<name>.parked`` so both copies
    survive. Records carry unique timestamp+uuid names, so a collision is
    not a normal event and neither half of it may be silently discarded.
    """
    if not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        target = destination / child.name
        if not target.exists():
            child.rename(target)
            continue
        if child.is_file() and target.is_file():
            try:
                if child.read_bytes() == target.read_bytes():
                    child.unlink()
                    continue
            except OSError:
                pass
        survivor = destination / f"{child.name}.parked"
        counter = 1
        while survivor.exists():
            survivor = destination / f"{child.name}.parked.{counter}"
            counter += 1
        child.rename(survivor)
    try:
        source.rmdir()
    except OSError:
        # Something the merge could not place stays put; the next run's
        # recovery sweep gets another chance.
        pass


def park_records(root: str | Path) -> Path | None:
    """Hide the records store for the duration of an unattended run.

    ``gui_crash.maybe_run_startup_flow`` (startup branch D8) opens a modal
    review whenever ``list_records`` is non-empty — a dialog no headless
    sweep can answer, so the run blocks forever. Parking renames the whole
    directory out of the D8 reader's sight; ``list_records`` inside the
    parked window sees an empty store. Returns the park path, or None when
    there was nothing to park. Restore with :func:`restore_parked_records`
    in the caller's cleanup path (the certifiers' ``finally`` blocks).

    First, any park left behind by a crashed predecessor is merged home —
    a sweep that died (SIGSEGV skips every ``finally``) leaves its park in
    the root, and the next sweep recovers it before hiding the store again.
    """
    root = Path(root)
    store = crash_dir(root)

    for leftover in sorted(root.glob(_PARK_PREFIX + "*")):
        _merge_dir(leftover, store)

    if not store.is_dir():
        return None
    parked = root / f"{_PARK_PREFIX}{os.getpid()}"
    store.rename(parked)
    return parked


def restore_parked_records(root: str | Path, parked: Path | None) -> None:
    """Bring a park home (no-op for ``parked=None`` or an already-empty
    park). Merge, not rename, so records written DURING the parked window —
    a genuine crash in the sweep itself, the Qt fatal handler — survive
    alongside the parked originals."""
    if parked is None:
        return
    parked = Path(parked)
    if parked.exists():
        _merge_dir(parked, crash_dir(root))
