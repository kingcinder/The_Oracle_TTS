"""License verify-key registry — the ONLY place key material lives in the client.

Anti-piracy model (docs/LICENSING_DESIGN.md §3): the client embeds *public*
verify keys; signatures can only be minted with the private seed, which never
ships (the vendor mints with scripts/license_sign.py). Extracting these keys
buys an attacker nothing — that is the whole reason the scheme is asymmetric.

Production key custody
----------------------
The registry must hold no key whose signing seed is public. The former ``k1``
entry was the RFC 8032 §7.1 TEST 1 public key: its seed is published in the
spec's test vectors and was hard-coded throughout this repository's tests, so
anyone who read the source could mint a token that this build would call a
genuine licence. It has been removed.

The production key **``k2`` was registered on 2026-10-19**. Its signing seed
was generated on the owner's machine and lives only in the owner's secret
manager (the provisioning run wrote it to
``~/.config/the-oracle/license-signing-seed``; move it into the secret manager
and delete the local copy). This project never writes the seed to disk, never
logs it, and never passes it to a test; losing it means rotating keys, not
losing the product.

An empty registry still fails closed — every token reports ``unknown_key``, so an
unactivated install runs as the community edition and no forged token is ever
accepted. Two nets keep the compromised key from coming back: the production
registry is scanned for RFC 8032 test-vector keys
(tests/test_licensing_registry_hygiene.py), and tests inject their own
test-only registry (tests/licensing_fixtures.py) instead of relying on this
table holding a test key.

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


#: Ordered registry, oldest first. ``k2`` is the production verify key
#: registered on 2026-10-19; its signing seed lives only in the owner's
#: secret manager (never in this repo). Tests do not read this table: they
#: inject a test-only registry via the ``test_license_registry`` fixture, so a
#: test can never make the shipped install trust a test key.
KEYS: tuple[VerifyKey, ...] = (
    VerifyKey(
        key_id="k2",
        public_key=bytes.fromhex(
            "cfb15a8029ad74a99ae3f52eaddd92a011f38786ba6de19f38de0b88eede0a88"
        ),
    ),
)


def key_for(key_id: str) -> VerifyKey | None:
    """The registered verify key for ``key_id``, or None if unknown."""
    for entry in KEYS:
        if entry.key_id == key_id:
            return entry
    return None


def registry_is_configured() -> bool:
    """True once at least one production verify key is registered.

    False means the install fails closed (community edition for everyone) and
    that no license can be minted *for* it yet — a distribution blocker, not a
    user-visible defect. scripts/release.py and the doctor surface it.
    """
    return bool(KEYS)


def verify_signature(key_id: str, message: bytes, signature: bytes) -> bool:
    """Verify ``signature`` over ``message`` under the named registered key.

    Unknown key_id -> False (typed "unknown key" at the token layer, not an
    exception). Any malformed key/signature length is False, never a raise.
    """
    entry = key_for(key_id)
    if entry is None:
        return False
    return _ed25519.verify(entry.public_key, message, signature)
