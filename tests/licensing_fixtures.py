"""Test-only license registry — no test may depend on a production key.

The suite's signing fixture is RFC 8032 §7.1 TEST 1: a published seed, so any
key derived from it is forgeable by anyone who copies the spec. The production
registry must therefore contain no such key (tests/test_licensing_registry_hygiene.py
enforces that), and the licensing tests must not need it to:

    pytestmark = pytest.mark.usefixtures("test_license_registry")

registers this table for the module, and every payload that the tests mint uses
``TEST_KEY_ID``. The key id is not secret; what matters is that it exists only
in this file, never in ``the_oracle.licensing.keys``.

Deliberately NOT autouse: a suite-wide patch of the production registry would
hide a real regression in it (the doctor and the shipped install read that
table), and the hygiene test that reads it unpatched would silently certify the
fixture instead of the product.
"""

from __future__ import annotations

import pytest

from the_oracle.licensing import keys
from the_oracle.licensing.keys import VerifyKey

#: The key id every test-minted payload carries. Not a production id.
TEST_KEY_ID = "ktest"

#: RFC 8032 §7.1 TEST 1 seed/public key — a *published* test vector, used here
#: purely as convenient crypto material. Never register it in production.
TEST_SEED = bytes.fromhex(
    "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
)
TEST_PUBLIC_KEY = bytes.fromhex(
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
)


def make_registry() -> tuple[VerifyKey, ...]:
    """A one-entry registry holding only the test key.

    Deliberately not named ``test_*``: a test module that imports this helper
    would otherwise have pytest collect it as a test (and it returns a value,
    which pytest warns about).
    """
    return (VerifyKey(key_id=TEST_KEY_ID, public_key=TEST_PUBLIC_KEY),)


#: The PRODUCTION registry as it is on disk, captured at import time — before
#: any fixture can patch it. Tests that assert what a *shipped* install trusts
#: must read this, because the injected table would otherwise shadow the very
#: thing under test.
PRODUCTION_KEYS: tuple[VerifyKey, ...] = keys.KEYS
PRODUCTION_PUBLIC_KEYS: frozenset[bytes] = frozenset(
    entry.public_key for entry in PRODUCTION_KEYS
)


def payload(**overrides: object) -> dict[str, object]:
    """A valid studio payload keyed to the test registry, with overrides."""
    base: dict[str, object] = {
        "v": 2,
        "key_id": TEST_KEY_ID,
        "lic_id": "22222222-2222-4222-8222-222222222222",
        "licensee": "Tester",
        "machine_hash": "",
        "iat": 1700000000,
        "exp": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def test_license_registry(monkeypatch: pytest.MonkeyPatch) -> str:
    """Install the test-only registry for one test and return its key id."""
    from the_oracle.licensing import keys

    monkeypatch.setattr(keys, "KEYS", make_registry())
    return TEST_KEY_ID
