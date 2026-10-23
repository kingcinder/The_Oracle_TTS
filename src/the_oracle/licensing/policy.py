"""Community defaults and feature-level Studio entitlement checks."""

from __future__ import annotations

from the_oracle.licensing.tokens import LicenseStatus


class LicenseRequired(Exception):
    """Raised when a requested feature is outside this install's edition."""


def is_licensed(status: LicenseStatus) -> bool:
    """True only for a verified, unexpired, machine-matching product key."""
    return bool(status.ok and status.state == "valid")


EDITIONS: dict[str, frozenset[str]] = {
    "community": frozenset({"render", "conversation", "recording_studio", "voice_craft"}),
    "studio": frozenset({"render", "conversation", "recording_studio", "voice_craft", "studio_features"}),
    "trial": frozenset({"render", "conversation", "recording_studio", "voice_craft", "studio_features"}),
}


def current_edition(status: LicenseStatus) -> str:
    """A verified current key enables Studio; every failure degrades to Community."""
    return "studio" if is_licensed(status) else "community"


def require(status: LicenseStatus, entitlement: str) -> None:
    """Raise only when the requested feature is not included in the edition."""
    if entitlement in EDITIONS[current_edition(status)]:
        return
    raise LicenseRequired(
        f"'{entitlement}' requires a Studio license. "
        f"{status.detail + ' ' if status.detail else ''}"
        "Contact support to upgrade — activation is offline and takes one paste."
    )
