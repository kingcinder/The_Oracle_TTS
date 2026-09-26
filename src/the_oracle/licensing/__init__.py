"""The Oracle licensing: offline token verification (docs/LICENSING_DESIGN.md).

Public surface. Import discipline: this package imports stdlib plus the
vendored ``the_oracle._ed25519`` only — never huggingface_hub, torch, or any
ML dependency (pinned by tests/test_offline_guarantee.py), and it runs nothing
at import time. ``current_license()`` is called lazily at feature gates so an
unlicensed install never blocks startup.

Everything is offline by construction: verification is a pure function of the
token bytes, the embedded public keys, and the wall clock. Activation never
contacts a server (pins in tests/test_licensing_offline.py).
"""

from __future__ import annotations

from pathlib import Path

from the_oracle.licensing.policy import (
    DEFAULT_EDITION,
    EDITIONS,
    LicenseRequired,
    current_edition,
    edition_entitlements,
    require,
)
from the_oracle.licensing.store import clear_token, load_token, save_token, store_path
from the_oracle.licensing.tokens import (
    LicenseStatus,
    TokenPayload,
    canonical_payload_bytes,
    mint,
    parse,
    verify_token,
)

__all__ = [
    "DEFAULT_EDITION",
    "EDITIONS",
    "LicenseRequired",
    "LicenseStatus",
    "TokenPayload",
    "canonical_payload_bytes",
    "clear_token",
    "current_edition",
    "current_license",
    "edition_entitlements",
    "load_token",
    "mint",
    "parse",
    "require",
    "save_token",
    "store_path",
    "verify_token",
]


def current_license(repo_root: str | Path | None = None) -> LicenseStatus:
    """The install's license status — the one call every consumer makes.

    ``repo_root=None`` resolves the running install's root (the same policy
    the offline module owns). Any store-level failure is a typed
    ``store_error`` status with the remedy in ``detail``; it never raises.
    """
    from the_oracle.licensing import machine
    from the_oracle.offline import repo_root as resolve_repo_root

    root = repo_root if repo_root is not None else resolve_repo_root()
    result = load_token(root)
    if result.state == "no_token":
        return LicenseStatus(
            ok=True,
            state="no_token",
            edition="community",
            detail="Not activated — community edition. Everything included ships in community.",
        )
    if result.state != "loaded" or result.stored is None:
        return LicenseStatus(ok=False, state="store_error", detail=result.detail)

    return verify_token(result.stored.token, machine_hash=machine.machine_hash())
