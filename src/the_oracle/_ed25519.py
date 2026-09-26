"""Pure-Python Ed25519 (RFC 8032) — the licensing crypto, vendored.

Chosen deliberately (docs/LICENSING_DESIGN.md §3, decision recorded 2026-09-26):
no new dependency, no offline-bundle change, fully auditable, and the suite's
offline guarantee stays pure stdlib. Verification happens once per activation
and once per doctor run, so the pure-Python speed cost is irrelevant. Swapping
in pynacl later means replacing this module and `keys.py`'s imports — nothing
else in the package touches curve arithmetic.

Reference-shaped implementation of RFC 8032 (edwards25519sha512batch / the
Ed25519 scheme exactly as specified): SHA-512, the standard base point,
little-endian field arithmetic mod 2**255-19. Verify rejects non-canonical S
(malleability) and points not on the curve. The signing seed never ships: the
client embeds only 32-byte public keys (`keys.py`).
"""

from __future__ import annotations

import hashlib

#: RFC 8032 group order and field prime.
_Q = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493


def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def _inv(x: int) -> int:
    return pow(x, _Q - 2, _Q)


_D = -121665 * _inv(121666) % _Q
_I = pow(2, (_Q - 1) // 4, _Q)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * _inv(_D * y * y + 1)
    x = pow(xx, (_Q + 3) // 8, _Q)
    if (x * x - xx) % _Q != 0:
        x = (x * _I) % _Q
    if x % 2 != 0:
        x = _Q - x
    return x


def _is_on_curve(point: tuple[int, int]) -> bool:
    x, y = point
    return (-x * x + y * y - 1 - _D * x * x * y * y) % _Q == 0


_By = 4 * _inv(5) % _Q
_B = (_xrecover(_By), _By)


def _edwards_add(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = p
    x2, y2 = q
    x3 = (x1 * y2 + x2 * y1) * _inv(1 + _D * x1 * x2 * y1 * y2)
    y3 = (y1 * y2 + x1 * x2) * _inv(1 - _D * x1 * x2 * y1 * y2)
    return (x3 % _Q, y3 % _Q)


def _scalarmult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    """Double-and-add on the Edwards curve. Iterative: no recursion-depth risk."""
    result = (0, 1)  # the neutral element
    addend = point
    while scalar > 0:
        if scalar & 1:
            result = _edwards_add(result, addend)
        addend = _edwards_add(addend, addend)
        scalar >>= 1
    return result


def _encode_point(point: tuple[int, int]) -> bytes:
    x, y = point
    bits = [(y >> i) & 1 for i in range(255)] + [x & 1]
    return bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(32))


def _decode_point(data: bytes) -> tuple[int, int]:
    if len(data) != 32:
        raise ValueError("point encoding must be 32 bytes")
    y = int.from_bytes(data, "little") & ((1 << 255) - 1)
    x_sign = data[31] >> 7
    x = _xrecover(y)
    if x & 1 != x_sign:
        x = _Q - x
    point = (x, y)
    if not _is_on_curve(point):
        raise ValueError("point is not on the curve")
    return point


def _clamp(secret_seed: bytes) -> int:
    """RFC 8032 §5.1.2 step 2: the scalar derived from the 32-byte seed."""
    digest = _sha512(secret_seed)
    scalar = bytearray(digest[:32])
    scalar[0] &= 0b11111000
    scalar[31] &= 0b01111111
    scalar[31] |= 0b01000000
    return int.from_bytes(scalar, "little")


def public_key(secret_seed: bytes) -> bytes:
    """Derive the 32-byte public key for a 32-byte seed."""
    if len(secret_seed) != 32:
        raise ValueError("the signing seed must be exactly 32 bytes")
    return _encode_point(_scalarmult(_B, _clamp(secret_seed)))


def sign(secret_seed: bytes, message: bytes) -> bytes:
    """RFC 8032 §5.1.6 — Ed25519 signature (64 bytes: R || S)."""
    if len(secret_seed) != 32:
        raise ValueError("the signing seed must be exactly 32 bytes")
    digest = _sha512(secret_seed)
    scalar = _clamp(secret_seed)
    prefix = digest[32:]
    encoded_public = _encode_point(_scalarmult(_B, scalar))
    r = int.from_bytes(_sha512(prefix + message), "little") % _L
    encoded_r = _encode_point(_scalarmult(_B, r))
    h = int.from_bytes(_sha512(encoded_r + encoded_public + message), "little") % _L
    s = (r + h * scalar) % _L
    return encoded_r + s.to_bytes(32, "little")


def verify(public_key_bytes: bytes, message: bytes, signature: bytes) -> bool:
    """RFC 8032 §5.1.7 verification. Returns False on any malformed input.

    Rejects signatures whose S half is >= the group order (non-canonical,
    which would make signatures malleable) and public keys or R points that
    do not decode to points on the curve.
    """
    if len(public_key_bytes) != 32 or len(signature) != 64:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    try:
        a_point = _decode_point(public_key_bytes)
        r_point = _decode_point(signature[:32])
    except ValueError:
        return False
    h = int.from_bytes(_sha512(signature[:32] + public_key_bytes + message), "little") % _L
    lhs = _scalarmult(_B, s)
    rhs = _edwards_add(r_point, _scalarmult(a_point, h))
    return lhs == rhs
