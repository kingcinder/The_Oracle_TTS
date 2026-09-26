"""The vendored Ed25519 (the_oracle/_ed25519.py) against RFC 8032 itself.

The licensing scheme's trust root is this module, so it is pinned to the
spec's own test vectors (RFC 8032 §7.1 TEST 1 and TEST 2) rather than to
self-consistent roundtrips — a wrong-but-deterministic implementation would
pass roundtrips and fail here.

**Mutation contract:** deleting the non-canonical-S check, the on-curve
check, or any length guard in _ed25519.verify must make a test in this file
fail. If one of these tests passes with a check removed, the net is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import _ed25519 as ed  # noqa: E402

# RFC 8032 §7.1 TEST 1
SEED_1 = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
PUB_1 = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
SIG_1 = bytes.fromhex(
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
    "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
)

# RFC 8032 §7.1 TEST 2
SEED_2 = bytes.fromhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb")
PUB_2 = bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")
SIG_2 = bytes.fromhex(
    "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
    "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
)
MSG_2 = bytes.fromhex("72")


def test_rfc8032_test1_public_key_and_signature() -> None:
    assert ed.public_key(SEED_1) == PUB_1
    assert ed.sign(SEED_1, b"") == SIG_1


def test_rfc8032_test2_public_key_and_signature() -> None:
    assert ed.public_key(SEED_2) == PUB_2
    assert ed.sign(SEED_2, MSG_2) == SIG_2


def test_rfc8032_vectors_verify() -> None:
    assert ed.verify(PUB_1, b"", SIG_1) is True
    assert ed.verify(PUB_2, MSG_2, SIG_2) is True


def test_tampered_message_rejected() -> None:
    assert ed.verify(PUB_2, bytes.fromhex("73"), SIG_2) is False


def test_tampered_signature_rejected() -> None:
    flipped = bytes([SIG_2[0] ^ 0x01]) + SIG_2[1:]
    assert ed.verify(PUB_2, MSG_2, flipped) is False


def test_non_canonical_s_rejected() -> None:
    """S >= L is a malleable forgery of a valid signature; must be refused."""
    s_forged = int.from_bytes(SIG_2[32:], "little") + ed._L
    malleable = SIG_2[:32] + s_forged.to_bytes(32, "little")
    assert ed.verify(PUB_2, MSG_2, malleable) is False


def test_public_key_not_on_curve_rejected() -> None:
    off_curve = bytes(31) + b"\x02"
    assert ed.verify(off_curve, MSG_2, SIG_2) is False


def test_wrong_lengths_rejected() -> None:
    assert ed.verify(PUB_2, MSG_2, SIG_2[:63]) is False
    assert ed.verify(PUB_2, MSG_2, SIG_2 + b"\x00") is False
    assert ed.verify(PUB_2[:31], MSG_2, SIG_2) is False
    assert ed.verify(PUB_2 + b"\x00", MSG_2, SIG_2) is False


def test_signing_seed_length_guard() -> None:
    import pytest

    with pytest.raises(ValueError):
        ed.public_key(SEED_1[:31])
    with pytest.raises(ValueError):
        ed.sign(SEED_1 + b"\x00", b"")


def test_registry_key_k1_is_the_rfc_vector_key() -> None:
    """The commissioning verify key IS the RFC 8032 TEST 1 public key, so the
    registry inherits the vector pinning: any corruption of keys.py fails
    test_rfc8032_test1_public_key_and_signature's assertion against the spec."""
    from the_oracle.licensing import keys

    entry = keys.key_for("k1")
    assert entry is not None
    assert entry.public_key == PUB_1
