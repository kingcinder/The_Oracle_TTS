"""EULA acceptance record (first-run click-through).

The record is one small JSON file under the install root. It stores the
SHA-256 of the exact EULA text the user accepted, so a changed agreement
asks again. Missing, unreadable, or malformed record = NOT accepted.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ACCEPTANCE_FILENAME = "eula_acceptance.json"
EULA_FILENAME = "EULA-DRAFT.md"


def eula_path(root: str | Path) -> Path:
    """The EULA the user must accept.

    The packaged copy (next to this module) is authoritative, so an installed
    wheel always has it. ``root`` is the fallback for a bare checkout layout.
    """
    packaged = Path(__file__).resolve().parent / EULA_FILENAME
    if packaged.is_file():
        return packaged
    return Path(root) / EULA_FILENAME


def acceptance_path(root: str | Path) -> Path:
    return Path(root) / ACCEPTANCE_FILENAME


def eula_text_and_digest(root: str | Path) -> tuple[str, str] | None:
    """The EULA text and its digest, or None when the file is missing."""
    try:
        text = eula_path(root).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_accepted(root: str | Path) -> bool:
    """True only when the stored digest matches the current EULA text."""
    current = eula_text_and_digest(root)
    if current is None:
        return False
    try:
        data = json.loads(acceptance_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    return data.get("eula_sha256") == current[1]


def record_acceptance(root: str | Path) -> Path:
    """Persist acceptance of the current EULA text, atomically."""
    current = eula_text_and_digest(root)
    if current is None:
        raise FileNotFoundError(eula_path(root))
    path = acceptance_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "eula_sha256": current[1],
        "accepted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    return path
