"""License token envelope: ORACLE1.<base64url(payload)>.<base64url(signature)>.

Canonical JSON (sorted keys, compact separators, UTF-8) makes the signed
payload byte-stable across platforms and key-order shuffles — the signature
covers bytes, so the bytes must be deterministic. Verification order per
docs/LICENSING_DESIGN.md §3: envelope shape, key_id known, signature, expiry,
machine binding, edition policy. Every failure is a typed status, never an
exception at the call site.
"""

from __future__ import annotations

import base64
import binascii
import json
import time
from dataclasses import dataclass, field

from the_oracle.licensing import keys

TOKEN_PREFIX = "ORACLE1"

_KNOWN_PAYLOAD_KEYS = frozenset(
    {"v", "key_id", "lic_id", "edition", "licensee", "machine_hash", "iat", "exp"}
)


@dataclass(frozen=True)
class TokenPayload:
    key_id: str
    lic_id: str
    edition: str
    licensee: str
    machine_hash: str
    iat: int
    exp: int | None  # None = perpetual

    def to_dict(self) -> dict[str, object]:
        return {
            "v": 1,
            "key_id": self.key_id,
            "lic_id": self.lic_id,
            "edition": self.edition,
            "licensee": self.licensee,
            "machine_hash": self.machine_hash,
            "iat": self.iat,
            "exp": self.exp,
        }


@dataclass(frozen=True)
class LicenseStatus:
    """The single typed outcome of verification. ``detail`` is user-renderable."""

    ok: bool
    state: str  # valid | no_token | malformed | unknown_key | bad_signature | expired | machine_mismatch | store_error
    edition: str | None = None
    licensee: str | None = None
    lic_id: str | None = None
    key_id: str | None = None
    exp: int | None = None
    machine_locked: bool = False
    detail: str = ""
    payload: TokenPayload | None = field(default=None, repr=False, compare=False)


def canonical_payload_bytes(payload: dict[str, object]) -> bytes:
    """The exact bytes a signature covers — frozen by test against key order."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.b64decode(text + padding, altchars=b"-_")


def mint(payload: dict[str, object], signing_seed: bytes) -> str:
    """Produce a signed token. Vendor-side (scripts/license_sign.py) — the
    client package can mint only in tests, against test seeds."""
    from the_oracle import _ed25519

    body = canonical_payload_bytes(payload)
    signature = _ed25519.sign(signing_seed, body)
    return f"{TOKEN_PREFIX}.{_b64url_encode(body)}.{_b64url_encode(signature)}"


def parse(token: str) -> tuple[dict[str, object], bytes] | None:
    """Split a token into (payload dict, signature bytes); None if malformed."""
    if not isinstance(token, str):
        return None
    parts = token.strip().split(".")
    if len(parts) != 3 or parts[0] != TOKEN_PREFIX:
        return None
    try:
        body = _b64url_decode(parts[1])
        signature = _b64url_decode(parts[2])
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(signature, bytes):
        return None
    if payload.get("v") != 1 or set(payload) != _KNOWN_PAYLOAD_KEYS:
        return None
    return payload, signature


def verify_token(
    token: str,
    *,
    now: float | None = None,
    machine_hash: str | None = None,
) -> LicenseStatus:
    """Full verification in the documented order. Pure function of its inputs
    and the embedded key registry — the doctor's history-independence pin
    depends on exactly that."""
    parsed = parse(token)
    if parsed is None:
        return LicenseStatus(ok=False, state="malformed", detail="The activation token is malformed. Re-enter it exactly as issued.")
    payload_dict, signature = parsed

    key_id = payload_dict.get("key_id")
    if not isinstance(key_id, str) or keys.key_for(key_id) is None:
        return LicenseStatus(ok=False, state="unknown_key", detail="The token was issued under a key this install does not know. Upgrade the suite, or contact support with the token.")

    body = canonical_payload_bytes(payload_dict)
    if not keys.verify_signature(key_id, body, signature):
        return LicenseStatus(ok=False, state="bad_signature", detail="The token is not a genuine Oracle license. Check for truncation; if it matches what you received, contact support.")

    try:
        payload = TokenPayload(
            key_id=key_id,
            lic_id=str(payload_dict["lic_id"]),
            edition=str(payload_dict["edition"]),
            licensee=str(payload_dict.get("licensee") or ""),
            machine_hash=str(payload_dict.get("machine_hash") or ""),
            iat=int(payload_dict["iat"]),
            exp=None if payload_dict.get("exp") is None else int(payload_dict["exp"]),
        )
    except (KeyError, TypeError, ValueError):
        return LicenseStatus(ok=False, state="malformed", detail="The token payload is malformed. Re-enter it exactly as issued.")

    current_time = time.time() if now is None else now
    if payload.exp is not None and current_time >= payload.exp:
        return LicenseStatus(
            ok=False,
            state="expired",
            edition=payload.edition,
            licensee=payload.licensee,
            lic_id=payload.lic_id,
            key_id=key_id,
            exp=payload.exp,
            machine_locked=bool(payload.machine_hash),
            detail="This license has expired. Contact support to renew — verification never requires a network connection.",
        )

    if payload.machine_hash and machine_hash and payload.machine_hash != machine_hash:
        return LicenseStatus(
            ok=False,
            state="machine_mismatch",
            edition=payload.edition,
            licensee=payload.licensee,
            lic_id=payload.lic_id,
            key_id=key_id,
            exp=payload.exp,
            machine_locked=True,
            detail="This license is locked to a different machine. Contact support to have it re-issued.",
        )

    return LicenseStatus(
        ok=True,
        state="valid",
        edition=payload.edition,
        licensee=payload.licensee,
        lic_id=payload.lic_id,
        key_id=key_id,
        exp=payload.exp,
        machine_locked=bool(payload.machine_hash),
        payload=payload,
    )
