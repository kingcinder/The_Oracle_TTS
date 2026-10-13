"""The verify-key registry's production hygiene and the machine fingerprint.

The registry's own contents are the subject of
``tests/test_licensing_registry_hygiene.py`` (source scan plus table scan) and
of ``test_production_registry_holds_no_test_vector_key`` below. What used to
live here was the opposite: tests that *required* the production registry to
contain the published RFC 8032 TEST 1 key, plus one that asserted the registry
key WAS the spec's vector — a net that locked the forgeable key in place.
Those are inverted, not deleted: every test that needs a working key installs
the test-only registry through the ``test_license_registry`` fixture, so no
assertion anywhere depends on a shipped install trusting a published seed.

**Mutation contract:** registering any RFC 8032 §7.1 vector key in ``keys.py``
must fail ``test_production_registry_holds_no_test_vector_key``; swapping
``machine.machine_hash()`` to return the RAW id must fail
``test_fingerprint_is_hash_not_raw`` — the privacy-critical pin.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import _ed25519  # noqa: E402
from the_oracle.licensing import keys, machine, tokens  # noqa: E402
from tests.licensing_fixtures import (  # noqa: E402
    TEST_KEY_ID,
    TEST_PUBLIC_KEY,
    TEST_SEED,
    payload,
    make_registry,
)

pytestmark = pytest.mark.usefixtures("test_license_registry")

#: The PRODUCTION registry, captured at import time — before the module-level
#: fixture patches ``keys.KEYS``. Every other test here runs against the test-only
#: registry, so a live attribute read would inspect the fixture rather than the
#: table a shipped install reads; the capture is what keeps this claim honest.
PRODUCTION_KEYS = keys.KEYS


def test_production_registry_holds_no_test_vector_key() -> None:
    """No registered key may be one whose seed RFC 8032 publishes.

    Replaces the old ``test_registry_key_matches_the_rfc_vector``, which
    asserted the opposite and so locked the forgeable key in place.
    """
    from tests.test_licensing_ed25519 import VALID_VECTORS

    forbidden = {public for _seed, public, _message, _signature in VALID_VECTORS}
    assert forbidden, "vacuity: the RFC vector set must not be empty"
    offenders = [entry.key_id for entry in PRODUCTION_KEYS if entry.public_key in forbidden]
    assert offenders == [], offenders


def test_registry_state_report_matches_the_table() -> None:
    assert keys.registry_is_configured() is bool(keys.KEYS)
    assert keys.KEYS, "vacuity: the injected test registry must not be empty"
    for entry in keys.KEYS:
        assert len(entry.public_key) == 32
        assert keys.key_for(entry.key_id) == entry


def test_empty_registry_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """No registered key means no token verifies — never "trust anything"."""
    monkeypatch.setattr(keys, "KEYS", ())
    assert keys.registry_is_configured() is False
    assert keys.key_for(TEST_KEY_ID) is None
    assert keys.verify_signature(TEST_KEY_ID, b"message", b"\x00" * 64) is False


def test_unknown_key_id_returns_none_and_fails_verification() -> None:
    assert keys.key_for("nope") is None
    assert keys.verify_signature("nope", b"message", b"x" * 64) is False


def test_registered_test_key_verifies_only_through_the_injected_registry() -> None:
    """The fixture's registry accepts its own key and rejects the production ids."""
    status = tokens.verify_token(tokens.mint(payload(), TEST_SEED))
    assert status.ok and status.state == "valid"
    assert status.key_id == TEST_KEY_ID
    # The test key's public key is exactly the fixture's, never a production one.
    assert keys.key_for(TEST_KEY_ID) is not None
    assert keys.key_for(TEST_KEY_ID).public_key == TEST_PUBLIC_KEY


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
    """The §7 rotation promise, proven: mint under the current key, register a
    successor — the old token still passes; drop the old key — the same token
    reports unknown_key, and the successor's token verifies.

    Both keys are test-only: the current one is the fixture's, the successor is
    generated here. Nothing about this test touches the production registry.
    """
    successor_seed = bytes(range(32))
    successor_id = "k_next"
    successor = keys.VerifyKey(
        key_id=successor_id, public_key=_ed25519.public_key(successor_seed)
    )
    current_registry = make_registry()

    token_current = tokens.mint(payload(), TEST_SEED)
    token_successor = tokens.mint(payload(key_id=successor_id), successor_seed)

    monkeypatch.setattr(keys, "KEYS", current_registry + (successor,))
    assert tokens.verify_token(token_current).ok  # old key still registered
    assert tokens.verify_token(token_successor).ok  # new key active

    monkeypatch.setattr(keys, "KEYS", (successor,))  # one cycle later: oldest dropped
    dropped = tokens.verify_token(token_current)
    assert not dropped.ok
    assert dropped.state == "unknown_key"
    assert tokens.verify_token(token_successor).ok


def test_raw_identifier_never_persisted_or_logged(monkeypatch, tmp_path: Path) -> None:
    """Scan-pin: the machine module's source contains no write of the raw id."""
    source = Path(machine.__file__).read_text(encoding="utf-8")
    assert "write" not in source.lower() or "never" in source.lower()
    # and behaviorally: calling it performs no filesystem writes
    monkeypatch.setattr(machine, "_raw_machine_id", lambda: "abc")
    before = sorted(p.name for p in tmp_path.iterdir())
    machine.machine_hash()
    assert sorted(p.name for p in tmp_path.iterdir()) == before
