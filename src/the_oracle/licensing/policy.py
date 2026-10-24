"""Single licensed/unlicensed product gate."""

from __future__ import annotations

from the_oracle.licensing.tokens import LicenseStatus


class LicenseRequired(Exception):
    """Raised when a requested feature is outside this install's edition."""


def is_licensed(status: LicenseStatus) -> bool:
    """True only for a verified, unexpired, machine-matching product key."""
    return bool(status.ok and status.state == "valid")


def require(status: LicenseStatus, entitlement: str) -> None:
    """Require a valid serial number for every functional feature."""
    if is_licensed(status):
        return
    raise LicenseRequired(
        f"Enter a serial number to use {entitlement}. "
        f"{status.detail + ' ' if status.detail else ''}"
        "Activation is offline and takes one paste."
    )
