"""Product-key gate: one license unlocks the whole program.

There are no editions. A valid, unexpired, machine-matching product key
unlocks every command. Without one, the CLI and GUI block each command with
the purchase-path message, except the commands that exist to activate and
inspect a license (see ``UNLOCKED_COMMANDS`` in ``the_oracle.cli``).
"""

from __future__ import annotations

from the_oracle.licensing.tokens import LicenseStatus


class LicenseRequired(Exception):
    """Raised when the program is used without a valid product key.

    The CLI and GUI catch this and render ``str(exc)``: a clean purchase-path
    message, never a traceback.
    """


def is_licensed(status: LicenseStatus) -> bool:
    """True only for a verified, unexpired, machine-matching product key."""
    return bool(status.ok and status.state == "valid")


def require(status: LicenseStatus) -> None:
    """Raise LicenseRequired unless a valid product key is present."""
    if is_licensed(status):
        return
    raise LicenseRequired(
        f"{status.detail or 'The Oracle needs a product key.'} "
        "Activate with: the-oracle activate <key>. "
        "Purchase and support: see the license page."
    )
