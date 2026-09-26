"""Machine fingerprint — SHA-256 of the OS machine id, hash only, never raw.

The raw identifier (a stable per-install OS value) is *never* stored, logged,
or displayed; only its SHA-256 hex prefix exists anywhere. That is a deliberate
privacy decision (docs/LICENSING_DESIGN.md §3) and what lets PRIVACY.md say the
fingerprint "is stored only as a hash and never leaves the machine". V1 tokens
are not machine-locked; this module ships so opting in later is a vendor-side
token field, zero client releases (the design doc's exact rationale).

Any failure to read an identifier returns "" — a token carrying a machine_hash
then simply cannot mismatch (tokens.verify_token skips binding when either
side is empty), which is the fail-open direction for the *user*: a broken
fingerprint must never lock a legitimate customer out. The privacy direction
is unaffected: nothing raw is kept regardless.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys


def _raw_machine_id() -> str:
    """Best-effort stable per-install identifier, by platform. Never persisted."""
    if sys.platform == "linux":
        for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                with open(candidate, encoding="ascii") as handle:
                    value = handle.read().strip()
                    if value:
                        return value
            except OSError:
                continue
        return ""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in result.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split("=")[-1].strip().strip('"')
        except (OSError, subprocess.SubprocessError):
            pass
        return ""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "MachineGuid")
                return str(value)
        except OSError:
            return ""
    return ""


def machine_hash() -> str:
    """SHA-256 hex (first 32 chars) of the raw machine id — the only form the
    suite ever keeps, compares, or could put in a token. Empty string when no
    identifier is available (see module docstring for the fail direction)."""
    raw = _raw_machine_id()
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:32]


def fingerprint_context_note() -> str:
    """One honest line for support output about what the fingerprint is."""
    return (
        "machine fingerprint: SHA-256 of the OS machine id, stored only as a "
        f"hash; raw value never read back ({platform.system()}/{os.name})"
    )
