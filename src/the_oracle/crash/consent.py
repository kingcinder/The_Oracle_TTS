"""Consent store — fail-closed, by construction and by pin.

One small JSON file next to the crash reports. The fail direction is the
privacy-critical property of the whole unit (CRASH_TELEMETRY_DESIGN §3):

    **unreadable, missing, or malformed consent file = OPTED OUT.**

No default-on path exists anywhere. Deliberately NOT app_settings.json:
consent must be decidable before settings load (a segfault during startup
still needs the answer), and settings corruption must never be able to flip
consent on. Handlers re-read consent at fire time, so mid-session revocation
takes effect on the next event, not at next launch.

**Mutation contract (M-CONSENT):** flipping any failure branch of
read_consent() to ``return True`` must fail tests/test_crash_consent.py.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

CONSENT_FILENAME = "crash_consent.json"


def consent_path(root: str | Path) -> Path:
    return Path(root) / CONSENT_FILENAME


def read_consent(root: str | Path) -> bool:
    """Whether the user consented to local crash capture. False unless the
    file exists, parses, has the right shape, and says exactly ``True``.

    Read-only by design: a failed read never writes anything (a consent
    probe that "helpfully" recreated the file would be an unconsented write).
    """
    path = consent_path(root)
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return False
    try:
        data = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(data, dict):
        return False
    enabled = data.get("crash_reports_enabled")
    if enabled is not True:
        return False
    return True


def write_consent(root: str | Path, enabled: bool) -> Path:
    """Persist consent atomically. The opt-IN is the only write this unit's
    consent machinery ever performs, and only the user's explicit action
    (CLI command / GUI dialog) calls it."""
    path = consent_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "crash_reports_enabled": bool(enabled),
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    return path
