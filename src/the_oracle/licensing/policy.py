"""Editions → entitlements — the single source the CLI, GUI, and doctor read.

The critical rollout rule (docs/LICENSING_DESIGN.md §4): **community is
everything that ships today.** No existing workflow is gated by this unit;
entitlement checks exist so the first gated feature is a future one. An
expired or absent license degrades to community entitlements — renders in
flight finish, and nothing is ever locked away.
"""

from __future__ import annotations

from the_oracle.licensing.tokens import LicenseStatus

#: Entitlement sets by edition. Trial behaves as studio while valid.
EDITIONS: dict[str, frozenset[str]] = {
    "community": frozenset({"render", "conversation", "recording_studio", "voice_craft"}),
    "studio": frozenset({"render", "conversation", "recording_studio", "voice_craft", "studio_features"}),
    "trial": frozenset({"render", "conversation", "recording_studio", "voice_craft", "studio_features"}),
}

#: The default edition of an unlicensed (or degraded) install.
DEFAULT_EDITION = "community"


class LicenseRequired(Exception):
    """Raised by require() when an entitlement is not granted.

    The CLI and GUI catch this and render ``str(exc)`` — a clean purchase-path
    message, never a traceback. There is no enforcement deeper than feature
    boundaries, by design (docs/LICENSING_DESIGN.md §1).
    """


def edition_entitlements(edition: str | None) -> frozenset[str]:
    """Entitlements for an edition string; unknown editions get community."""
    if not edition:
        return EDITIONS[DEFAULT_EDITION]
    return EDITIONS.get(edition, EDITIONS[DEFAULT_EDITION])


def current_edition(status: LicenseStatus) -> str:
    """The edition an install actually runs as, per its license status."""
    if status.ok and status.edition:
        return status.edition
    return DEFAULT_EDITION


def require(status: LicenseStatus, entitlement: str) -> None:
    """Raise LicenseRequired unless the install's edition grants ``entitlement``."""
    if entitlement in edition_entitlements(current_edition(status)):
        return
    raise LicenseRequired(
        f"'{entitlement}' requires a Studio license. "
        "Contact support to upgrade — activation is offline and takes one paste."
    )
