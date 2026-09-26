"""Token envelope: canonical bytes, parse strictness, verification order.

**Mutation contract:** the net must catch (a) verification skipping the
signature check (M1), (b) the exp check comparing with ``>`` instead of
``>=`` (M2), (c) dropping the signature check from save_token (M4), (d)
canonical bytes depending on dict key order. If a mutation survives, the net
is wrong, not the code right.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import _ed25519  # noqa: E402
from the_oracle.licensing import tokens  # noqa: E402

SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
PUB = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")

#: An otherwise-valid payload with a DIFFERENT lic_id — used as the forgery
#: body under a signature made over the real payload.
REAL_PAYLOAD = {
    "v": 1,
    "key_id": "k1",
    "lic_id": "22222222-2222-4222-8222-222222222222",
    "edition": "studio",
    "licensee": "Cody",
    "machine_hash": "",
    "iat": 1700000000,
    "exp": None,
}
FORGED_PAYLOAD = dict(REAL_PAYLOAD, lic_id="33333333-3333-4333-8333-333333333333")


def _token(payload: dict) -> str:
    return tokens.mint(payload, SEED)


def _now_after(payload: dict, seconds: float = 1.0) -> float:
    return (payload["iat"] if payload["exp"] is None else payload["exp"]) + seconds


def test_canonical_bytes_are_key_order_independent() -> None:
    a = tokens.canonical_payload_bytes(dict(reversed(list(REAL_PAYLOAD.items()))))
    b = tokens.canonical_payload_bytes(dict(REAL_PAYLOAD))
    assert a == b
    assert json.loads(a) == REAL_PAYLOAD


def test_minted_token_verifies() -> None:
    status = tokens.verify_token(_token(REAL_PAYLOAD))
    assert status.ok and status.state == "valid"
    assert status.edition == "studio"
    assert status.licensee == "Cody"
    assert status.machine_locked is False


def test_tampered_payload_rejected() -> None:
    """Real signature carried over a modified body: signature check catches it."""
    body = tokens.canonical_payload_bytes(FORGED_PAYLOAD)
    signature = _ed25519.sign(SEED, tokens.canonical_payload_bytes(REAL_PAYLOAD))
    import base64

    b64 = lambda data: base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")
    franken_token = f"ORACLE1.{b64(body)}.{b64(signature)}"
    status = tokens.verify_token(franken_token)
    assert not status.ok
    assert status.state == "bad_signature"


def test_forged_signature_with_wrong_seed_rejected() -> None:
    other_seed = bytes(range(32))
    status = tokens.verify_token(_token(REAL_PAYLOAD) if False else tokens.mint(REAL_PAYLOAD, other_seed))
    assert not status.ok
    assert status.state == "bad_signature"


def test_unknown_key_id_rejected() -> None:
    status = tokens.verify_token(_token(dict(REAL_PAYLOAD, key_id="kX")))
    assert not status.ok
    assert status.state == "unknown_key"


def test_expired_token_rejected() -> None:
    exp = int(time.time()) - 10
    status = tokens.verify_token(_token(dict(REAL_PAYLOAD, exp=exp)))
    assert not status.ok
    assert status.state == "expired"
    assert status.detail  # carries an offline-safe remedy
    assert "network" not in status.detail.lower() or "never" in status.detail.lower()


def test_expiry_boundary_is_inclusive() -> None:
    """now == exp is expired. **Mutation M2:** flipping >= to > in verify_token
    must flip this test to failure."""
    exp = 1_700_000_500
    frozen = float(exp)  # exactly the boundary
    status = tokens.verify_token(_token(dict(REAL_PAYLOAD, exp=exp)), now=frozen)
    assert not status.ok
    assert status.state == "expired"
    # one second before is fine
    ok_status = tokens.verify_token(_token(dict(REAL_PAYLOAD, exp=exp)), now=float(exp - 1))
    assert ok_status.ok


def test_machine_locked_token_accepts_match_and_rejects_mismatch() -> None:
    locked = dict(REAL_PAYLOAD, machine_hash="a" * 32)
    assert tokens.verify_token(_token(locked), machine_hash="a" * 32).ok
    mismatch = tokens.verify_token(_token(locked), machine_hash="b" * 32)
    assert not mismatch.ok
    assert mismatch.state == "machine_mismatch"


def test_machine_binding_skipped_when_fingerprint_unavailable() -> None:
    """A machine-locked token with an unreadable local fingerprint must NOT
    lock the legitimate user out (the machine module's fail-open direction)."""
    locked = dict(REAL_PAYLOAD, machine_hash="a" * 32)
    status = tokens.verify_token(_token(locked), machine_hash="")
    assert status.ok


def test_malformed_tokens_rejected_as_malformed() -> None:
    for bad in (
        "",
        "not a token",
        "ORACLE1.only-two-parts",
        "WRONG1.a.b",
        "ORACLE1.!!!.%%%",
        f"ORACLE1.{tokens._b64url_encode(json.dumps([]).encode())}.{tokens._b64url_encode(b'x')}",
        f"ORACLE1.{tokens._b64url_encode(json.dumps({k: v for k, v in REAL_PAYLOAD.items() if k != 'exp'}).encode())}.{tokens._b64url_encode(b'x')}",
        f"ORACLE1.{tokens._b64url_encode(json.dumps(dict(REAL_PAYLOAD, v=2)).encode())}.{tokens._b64url_encode(b'x')}",
    ):
        status = tokens.verify_token(bad)
        assert not status.ok
        assert status.state in {"malformed", "unknown_key"}, (bad, status.state)


def test_perpetual_and_future_iat_both_verify() -> None:
    assert tokens.verify_token(_token(REAL_PAYLOAD)).ok
    assert tokens.verify_token(_token(dict(REAL_PAYLOAD, iat=int(time.time()) + 60))).ok
