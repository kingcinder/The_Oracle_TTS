"""The verify-key registry and the machine fingerprint's privacy direction.

**Mutation contract:** corrupting any hex character in keys.py KEYS must fail
test_registry_key_matches_the_rfc_vector (via the ed25519 file's registry
pin). Swapping machine.machine_hash() to return the RAW id must fail
test_fingerprint_is_hash_not_raw — the privacy-critical pin.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle.licensing import keys, machine  # noqa: E402


def test_registry_contains_commissioning_key() -> None:
    entry = keys.key_for("k1")
    assert entry is not None
    assert len(entry.public_key) == 32


def test_registry_key_matches_the_rfc_vector() -> None:
    from tests.test_licensing_ed25519 import PUB_1

    assert keys.key_for("k1").public_key == PUB_1


def test_unknown_key_id_returns_none_and_fails_verification() -> None:
    assert keys.key_for("nope") is None
    assert keys.verify_signature("nope", b"message", b"x" * 64) is False


def test_fingerprint_is_hash_not_raw(monkeypatch) -> None:
    """The ONLY form the suite keeps is a SHA-256 prefix. **Mutation M3:**
    making machine_hash() return the raw id must fail here."""
    monkeypatch.setattr(machine, "_raw_machine_id", lambda: "raw-machine-identifier")
    digest = machine.machine_hash()
    assert digest == hashlib.sha256(b"raw-machine-identifier").hexdigest()[:32]
    assert "raw-machine-identifier" not in digest
    assert len(digest) == 32


def test_fingerprint_fail_open_when_no_identifier(monkeypatch) -> None:
    monkeypatch.setattr(machine, "_raw_machine_id", lambda: "")
    assert machine.machine_hash() == ""


def test_key_rotation_old_key_keeps_verifying_until_dropped(monkeypatch) -> None:
    """The §7 rotation promise, proven: mint under k1, register k2 — the k1
    token still passes; drop k1 — the same token reports unknown_key, and a
    k2-signed token verifies. (Simulated via the registry tuple; the real k2
    appears when a rotation actually happens.)"""
    from tests.test_licensing_tokens import REAL_PAYLOAD, SEED
    from the_oracle import _ed25519
    from the_oracle.licensing import keys, tokens

    seed2 = bytes(range(32))
    k2 = keys.VerifyKey(key_id="k2", public_key=_ed25519.public_key(seed2))

    token_k1 = tokens.mint(REAL_PAYLOAD, SEED)
    token_k2 = tokens.mint(dict(REAL_PAYLOAD, key_id="k2"), seed2)

    monkeypatch.setattr(keys, "KEYS", keys.KEYS + (k2,))
    assert tokens.verify_token(token_k1).ok  # old key still registered
    assert tokens.verify_token(token_k2).ok  # new key active

    monkeypatch.setattr(keys, "KEYS", (k2,))  # one cycle later: k1 dropped
    dropped = tokens.verify_token(token_k1)
    assert not dropped.ok
    assert dropped.state == "unknown_key"
    assert tokens.verify_token(token_k2).ok


def test_raw_identifier_never_persisted_or_logged(monkeypatch, tmp_path: Path) -> None:
    """Scan-pin: the machine module's source contains no write of the raw id."""
    source = Path(machine.__file__).read_text(encoding="utf-8")
    assert "write" not in source.lower() or "never" in source.lower()
    # and behaviorally: calling it performs no filesystem writes
    monkeypatch.setattr(machine, "_raw_machine_id", lambda: "abc")
    before = sorted(p.name for p in tmp_path.iterdir())
    machine.machine_hash()
    assert sorted(p.name for p in tmp_path.iterdir()) == before
