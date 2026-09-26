"""License verify-key registry — the ONLY place key material lives in the client.

Anti-piracy model (docs/LICENSING_DESIGN.md §3): the client embeds *public*
verify keys; signatures can only be minted with the private seed, which never
ships (the vendor mints with scripts/license_sign.py). Extracting these keys
buys an attacker nothing — that is the whole reason the scheme is asymmetric.

Key rotation: add the new ``(key_id, verify_key)`` to KEYS, keep the old entry
for one release cycle so outstanding tokens keep verifying, then drop it. A
token whose key_id matches no entry is "unknown key", never a crash.
"""

from __future__ import annotations

from dataclasses import dataclass

from the_oracle import _ed25519


@dataclass(frozen=True)
class VerifyKey:
    key_id: str
    public_key: bytes


#: Ordered registry, oldest first. k1 is the commissioning key: its hex value
#: is pinned by tests/test_licensing_keys.py together with the RFC 8032 test
#: vector in tests/test_licensing_ed25519.py, so a corruption of this table
#: cannot silently pass verification of nothing.
KEYS: tuple[VerifyKey, ...] = (
    VerifyKey(
        key_id="k1",
        public_key=bytes.fromhex(
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
        ),
    ),
)

DEFAULT_KEY_ID = KEYS[-1].key_id


def key_for(key_id: str) -> VerifyKey | None:
    """The registered verify key for ``key_id``, or None if unknown."""
    for entry in KEYS:
        if entry.key_id == key_id:
            return entry
    return None


def verify_signature(key_id: str, message: bytes, signature: bytes) -> bool:
    """Verify ``signature`` over ``message`` under the named registered key.

    Unknown key_id -> False (typed "unknown key" at the token layer, not an
    exception). Any malformed key/signature length is False, never a raise.
    """
    entry = key_for(key_id)
    if entry is None:
        return False
    return _ed25519.verify(entry.public_key, message, signature)
