"""The vendored Ed25519 (the_oracle/_ed25519.py) against RFC 8032 itself.

The licensing scheme's trust root is this module, so it is pinned to the
spec's own test vectors rather than to self-consistent roundtrips — a
wrong-but-deterministic implementation would pass roundtrips and fail here.
It carries the **complete** RFC 8032 §7.1 vector set (TEST 1, TEST 2, TEST 3,
TEST 1024, TEST SHA(abc)) plus the malformed-input classes Wycheproof names
(non-canonical S, points not on the curve, small-order public keys, wrong
lengths), because "verifies the vectors" and "refuses everything else" are two
different claims and a licence trust root needs both.

**Registry:** no assertion here may depend on the production key registry. The
registry is scanned for these very vectors by
``tests/test_licensing_registry_hygiene.py``; this file pins the *primitive*.

**Mutation contract:** deleting the non-canonical-S check, the on-curve check,
the length guards, or a small-order-public-key check in ``_ed25519.verify``
must make a test in this file fail. If one of these tests passes with a check
removed, the net is wrong.

**Known open defect (found by these cases, fix belongs in ``_ed25519.py``):**
``verify`` accepts a small-order public key. With the identity element
(``01`` followed by 31 zero bytes) as A, ``[h]A`` is the neutral element for
every h, so the pair (R = [s]B, S = s) verifies for ANY message — a universal
forgery from a 32-byte public key alone.
``test_identity_public_key_must_not_verify_a_forged_signature`` fails until
``verify`` rejects small-order A (a cofactor/order check per RFC 8032 §5.1.7's
"decode and validate the public key" requirement).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import _ed25519 as ed  # noqa: E402
from the_oracle.licensing import keys as license_keys  # noqa: E402

pytestmark = pytest.mark.usefixtures("test_license_registry")

#: The PRODUCTION registry, captured at import time — before any fixture can
#: patch ``license_keys.KEYS``. This module's other tests run with the test-only
#: registry installed, so a live read of the attribute would inspect the fixture
#: instead of the shipped table; the import-time capture is what keeps the
#: registry claim honest here.
PRODUCTION_KEYS = license_keys.KEYS

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

# RFC 8032 §7.1 TEST 3 (two-byte message)
SEED_3 = bytes.fromhex("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7")
PUB_3 = bytes.fromhex("fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025")
MSG_3 = bytes.fromhex("af82")
SIG_3 = bytes.fromhex(
    "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
    "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"
)

# RFC 8032 §7.1 TEST 1024. The message is the spec's own 1023-byte octet
# string (transcribed verbatim from the RFC; the sign-equality assertion below
# is what proves the transcription, since a single wrong nibble changes SIG).
SEED_1024 = bytes.fromhex("f5e5767cf153319517630f226876b86c8160cc583bc013744c6bf255f5cc0ee5")
PUB_1024 = bytes.fromhex("278117fc144c72340f67d0f2316e8386ceffbf2b2428c9c51fef7c597f1d426e")
MSG_1024 = bytes.fromhex(
    "08b8b2b733424243760fe426a4b54908632110a66c2f6591eabd3345e3e4eb98"
    "fa6e264bf09efe12ee50f8f54e9f77b1e355f6c50544e23fb1433ddf73be84d8"
    "79de7c0046dc4996d9e773f4bc9efe5738829adb26c81b37c93a1b270b20329d"
    "658675fc6ea534e0810a4432826bf58c941efb65d57a338bbd2e26640f89ffbc"
    "1a858efcb8550ee3a5e1998bd177e93a7363c344fe6b199ee5d02e82d522c4fe"
    "ba15452f80288a821a579116ec6dad2b3b310da903401aa62100ab5d1a36553e"
    "06203b33890cc9b832f79ef80560ccb9a39ce767967ed628c6ad573cb116dbef"
    "efd75499da96bd68a8a97b928a8bbc103b6621fcde2beca1231d206be6cd9ec7"
    "aff6f6c94fcd7204ed3455c68c83f4a41da4af2b74ef5c53f1d8ac70bdcb7ed1"
    "85ce81bd84359d44254d95629e9855a94a7c1958d1f8ada5d0532ed8a5aa3fb2"
    "d17ba70eb6248e594e1a2297acbbb39d502f1a8c6eb6f1ce22b3de1a1f40cc24"
    "554119a831a9aad6079cad88425de6bde1a9187ebb6092cf67bf2b13fd65f270"
    "88d78b7e883c8759d2c4f5c65adb7553878ad575f9fad878e80a0c9ba63bcbcc"
    "2732e69485bbc9c90bfbd62481d9089beccf80cfe2df16a2cf65bd92dd597b07"
    "07e0917af48bbb75fed413d238f5555a7a569d80c3414a8d0859dc65a46128ba"
    "b27af87a71314f318c782b23ebfe808b82b0ce26401d2e22f04d83d1255dc51a"
    "ddd3b75a2b1ae0784504df543af8969be3ea7082ff7fc9888c144da2af58429e"
    "c96031dbcad3dad9af0dcbaaaf268cb8fcffead94f3c7ca495e056a9b47acdb7"
    "51fb73e666c6c655ade8297297d07ad1ba5e43f1bca32301651339e22904cc8c"
    "42f58c30c04aafdb038dda0847dd988dcda6f3bfd15c4b4c4525004aa06eeff8"
    "ca61783aacec57fb3d1f92b0fe2fd1a85f6724517b65e614ad6808d6f6ee34df"
    "f7310fdc82aebfd904b01e1dc54b2927094b2db68d6f903b68401adebf5a7e08"
    "d78ff4ef5d63653a65040cf9bfd4aca7984a74d37145986780fc0b16ac451649"
    "de6188a7dbdf191f64b5fc5e2ab47b57f7f7276cd419c17a3ca8e1b939ae49e4"
    "88acba6b965610b5480109c8b17b80e1b7b750dfc7598d5d5011fd2dcc5600a3"
    "2ef5b52a1ecc820e308aa342721aac0943bf6686b64b2579376504ccc493d97e"
    "6aed3fb0f9cd71a43dd497f01f17c0e2cb3797aa2a2f256656168e6c496afc5f"
    "b93246f6b1116398a346f1a641f3b041e989f7914f90cc2c7fff357876e506b5"
    "0d334ba77c225bc307ba537152f3f1610e4eafe595f6d9d90d11faa933a15ef1"
    "369546868a7f3a45a96768d40fd9d03412c091c6315cf4fde7cb68606937380d"
    "b2eaaa707b4c4185c32eddcdd306705e4dc1ffc872eeee475a64dfac86aba41c"
    "0618983f8741c5ef68d3a101e8a3b8cac60c905c15fc910840b94c00a0b9d0"
)
SIG_1024 = bytes.fromhex(
    "0aab4c900501b3e24d7cdf4663326a3a87df5e4843b2cbdb67cbf6e460fec350"
    "aa5371b1508f9f4528ecea23c436d94b5e8fcd4f681e30a6ac00a9704a188a03"
)

# RFC 8032 §7.1 TEST SHA(abc): the 64-byte message is SHA-512("abc")
SEED_SHA = bytes.fromhex("833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42")
PUB_SHA = bytes.fromhex("ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf")
MSG_SHA = hashlib.sha512(b"abc").digest()
SIG_SHA = bytes.fromhex(
    "dc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b589"
    "09351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704"
)

#: Every valid (seed, public key, message, signature) quadruple from §7.1.
VALID_VECTORS: tuple[tuple[bytes, bytes, bytes, bytes], ...] = (
    (SEED_1, PUB_1, b"", SIG_1),
    (SEED_2, PUB_2, MSG_2, SIG_2),
    (SEED_3, PUB_3, MSG_3, SIG_3),
    (SEED_1024, PUB_1024, MSG_1024, SIG_1024),
    (SEED_SHA, PUB_SHA, MSG_SHA, SIG_SHA),
)

#: Small-order curve points as 32-byte encodings: the identity, the order-2
#: point (0, -1), and the order-4 point y = 0 with both sign bits. A verifier
#: that does not validate A against them can be fooled (see the module
#: docstring): with A = identity, [h]A is neutral for every h.
SMALL_ORDER_PUBLIC_KEYS: tuple[bytes, ...] = (
    b"\x01" + b"\x00" * 31,
    bytes.fromhex("ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"),
    b"\x00" * 32,
    b"\x00" * 31 + b"\x80",
)


def _flip(blob: bytes, index: int) -> bytes:
    return blob[:index] + bytes([blob[index] ^ 0x01]) + blob[index + 1 :]


def test_rfc8032_test1_public_key_and_signature() -> None:
    assert ed.public_key(SEED_1) == PUB_1
    assert ed.sign(SEED_1, b"") == SIG_1


def test_rfc8032_test2_public_key_and_signature() -> None:
    assert ed.public_key(SEED_2) == PUB_2
    assert ed.sign(SEED_2, MSG_2) == SIG_2


def test_rfc8032_test3_public_key_and_signature() -> None:
    assert ed.public_key(SEED_3) == PUB_3
    assert ed.sign(SEED_3, MSG_3) == SIG_3


def test_rfc8032_test1024_public_key_and_signature() -> None:
    # The RFC labels this message "length 1023 bytes"; the length assertion is
    # what catches a transcription slip before the signature comparison does.
    assert len(MSG_1024) == 1023
    assert ed.public_key(SEED_1024) == PUB_1024
    assert ed.sign(SEED_1024, MSG_1024) == SIG_1024


def test_rfc8032_test_sha_abc_public_key_and_signature() -> None:
    """The spec's fifth vector, whose message is the SHA-512 digest of "abc"."""
    assert MSG_SHA == hashlib.sha512(b"abc").digest()
    assert len(MSG_SHA) == 64
    assert ed.public_key(SEED_SHA) == PUB_SHA
    assert ed.sign(SEED_SHA, MSG_SHA) == SIG_SHA


def test_rfc8032_vectors_verify() -> None:
    assert ed.verify(PUB_1, b"", SIG_1) is True
    assert ed.verify(PUB_2, MSG_2, SIG_2) is True


def test_every_rfc8032_vector_verifies() -> None:
    for _seed, public, message, signature in VALID_VECTORS:
        assert ed.verify(public, message, signature) is True, public.hex()


def test_one_bit_mutations_of_every_vector_are_rejected() -> None:
    """A bit flipped anywhere in R, in S, in the message or in the key must fail.

    Looped over all five vectors so a check that only happens to hold for the
    empty message (TEST 1) cannot pass for the set.
    """
    for _seed, public, message, signature in VALID_VECTORS:
        if message:
            mutated_message = message[:-1] + bytes([message[-1] ^ 0x01])
            assert ed.verify(public, mutated_message, signature) is False
        assert ed.verify(public, message, _flip(signature, 0)) is False  # R
        assert ed.verify(public, message, _flip(signature, 63)) is False  # S
        assert ed.verify(_flip(public, 0), message, signature) is False  # A


def test_tampered_message_rejected() -> None:
    assert ed.verify(PUB_2, bytes.fromhex("73"), SIG_2) is False


def test_tampered_signature_rejected() -> None:
    flipped = bytes([SIG_2[0] ^ 0x01]) + SIG_2[1:]
    assert ed.verify(PUB_2, MSG_2, flipped) is False


def test_non_canonical_s_rejected() -> None:
    """S >= L is a malleable forgery of a valid signature; must be refused.

    Looped over every vector so the rejection cannot be an artefact of TEST 2's
    particular S value.
    """
    for _seed, public, message, signature in VALID_VECTORS:
        s_forged = int.from_bytes(signature[32:], "little") + ed._L
        malleable = signature[:32] + s_forged.to_bytes(32, "little")
        assert ed.verify(public, message, malleable) is False, public.hex()


def test_public_key_not_on_curve_rejected() -> None:
    off_curve = bytes(31) + b"\x02"
    assert ed.verify(off_curve, MSG_2, SIG_2) is False


def test_r_not_on_curve_rejected() -> None:
    off_curve_r = bytes(31) + b"\x02" + SIG_2[32:]
    assert ed.verify(PUB_2, MSG_2, off_curve_r) is False


def test_identity_public_key_must_not_verify_a_forged_signature() -> None:
    """**The forgery this net was extended for.**

    With A = the identity element, ``[h]A`` is neutral for every h, so
    (R = [s]B, S = s) satisfies the verification equation for ANY message: the
    private key plays no part. A trust root that accepts this lets anyone mint
    licences from a published 32-byte public key.

    The signature below is *constructed* rather than random, so the assertion
    is not vacuous: it is exactly the pair that the equation accepts. The
    negative control shows the construct is a forgery only through the
    small-order key — under a real public key the same (R, S) is rejected.
    """
    s = 7
    forged_r = ed._encode_point(ed._scalarmult(ed._B, s))
    forged = forged_r + s.to_bytes(32, "little")
    identity = b"\x01" + b"\x00" * 31
    message = b"any message at all"

    # Negative control first: the same (R, S) is not a signature by PUB_2.
    assert ed.verify(PUB_2, message, forged) is False

    assert ed.verify(identity, message, forged) is False, (
        "the identity public key verified a forged signature: _ed25519.verify "
        "does not reject small-order public keys"
    )


def test_small_order_public_keys_are_rejected() -> None:
    """No small-order point may be accepted as a signing key."""
    for public in SMALL_ORDER_PUBLIC_KEYS:
        assert ed.verify(public, MSG_2, SIG_2) is False, public.hex()
        assert ed.verify(public, b"", SIG_1) is False, public.hex()


def test_wrong_lengths_rejected() -> None:
    assert ed.verify(PUB_2, MSG_2, SIG_2[:63]) is False
    assert ed.verify(PUB_2, MSG_2, SIG_2 + b"\x00") is False
    assert ed.verify(PUB_2[:31], MSG_2, SIG_2) is False
    assert ed.verify(PUB_2 + b"\x00", MSG_2, SIG_2) is False
    assert ed.verify(PUB_2, MSG_2, b"") is False
    assert ed.verify(b"", MSG_2, SIG_2) is False


def test_verify_never_raises_on_malformed_input() -> None:
    """Fuzz-shaped: verification is a total function, False instead of raising.

    The client calls this on bytes a user can paste, so an exception would be a
    crash at the activation boundary rather than a rejected token. Totality is
    asserted over the whole cross product; the *rejection* half is asserted only
    for the shapes whose verdict is unambiguous (bad lengths), because some
    pairs of well-formed-but-hostile 32/64-byte strings can legitimately satisfy
    the verification equation — that is the small-order-key defect below.
    """
    junk = (
        b"",
        b"\x00",
        b"\x00" * 31,
        b"\x00" * 32,
        b"\xff" * 32,
        b"\xff" * 64,
        b"\x00" * 64,
        bytes(range(32)),
        bytes(range(64)),
        bytes([0xFF]) * 32 + bytes([0x01]) * 32,
    )
    for public in junk:
        for signature in junk:
            result = ed.verify(public, b"message", signature)
            assert isinstance(result, bool), (public.hex(), signature.hex())

    wrong_lengths = (b"", b"\x00", b"\x00" * 31, b"\xff" * 33, b"\x00" * 63, b"\xff" * 65)
    for blob in wrong_lengths:
        assert ed.verify(blob, b"message", b"\x00" * 64) is False
        assert ed.verify(b"\x00" * 32, b"message", blob) is False


def test_signing_seed_length_guard() -> None:
    with pytest.raises(ValueError):
        ed.public_key(SEED_1[:31])
    with pytest.raises(ValueError):
        ed.sign(SEED_1 + b"\x00", b"")


def test_production_registry_must_not_hold_an_rfc8032_vector_key() -> None:
    """The registry pin's crypto half (its source-level half lives in
    tests/test_licensing_registry_hygiene.py).

    Previously this test asserted the reverse — that the production registry's
    only entry WAS the RFC 8032 TEST 1 key — which made a published seed the
    thing every shipped install trusted. The direction is now inverted on
    purpose: a green run means a shipped install does not trust the vectors.
    """
    forbidden = {public for _seed, public, _message, _signature in VALID_VECTORS}
    assert forbidden, "vacuity: the vector set must not be empty"
    offenders = [
        entry.key_id for entry in PRODUCTION_KEYS if entry.public_key in forbidden
    ]
    assert offenders == [], offenders
