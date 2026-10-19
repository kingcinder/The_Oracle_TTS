"""The production verify-key registry must hold no forgeable key.

The release-blocking defect this file exists for: ``keys.py`` shipped the RFC
8032 §7.1 TEST 1 public key as its only entry. That key's seed is published in
the spec and was hard-coded in this repository's own tests, so any reader could
mint a token that every shipped install would accept as a genuine licence.

These tests read the production registry **unpatched** — no
``test_license_registry`` fixture — because the subject is the table the shipped
install reads, and a test that certified an injected fixture instead would
certify nothing. The vacuity guard at the bottom proves the scan can actually
see a planted test key, so a green run is evidence and not an empty loop.

**Mutation contract:** re-adding the RFC 8032 vector public key (or its seed) to
keys.py must fail ``test_production_registry_holds_no_rfc8032_test_vector_key``
or ``test_keys_source_carries_no_public_test_vector_material``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import _ed25519  # noqa: E402
from the_oracle.licensing import keys  # noqa: E402

#: RFC 8032 §7.1 published test vectors: (seed, public key). Their seeds are
#: public, so any registry entry derived from one is forgeable.
RFC8032_TEST_VECTORS: tuple[tuple[bytes, bytes], ...] = (
    (
        bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"),
        bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"),
    ),
    (
        bytes.fromhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb"),
        bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"),
    ),
)

KEY_SOURCE = (REPO_ROOT / "src" / "the_oracle" / "licensing" / "keys.py").read_text(
    encoding="utf-8"
)


def test_production_registry_holds_no_rfc8032_test_vector_key() -> None:
    """No entry may be a key whose seed is published in the spec."""
    forbidden = {public for _seed, public in RFC8032_TEST_VECTORS}
    offenders = [entry.key_id for entry in keys.KEYS if entry.public_key in forbidden]
    assert offenders == [], (
        "the production verify-key registry holds a published RFC 8032 test "
        f"vector key ({offenders}); anyone with the spec can mint licences"
    )


def test_keys_source_carries_no_public_test_vector_material() -> None:
    """Neither the vector public key nor its seed may appear in keys.py.

    Source-level on purpose: a seed pasted into the module (even unused) leaks
    signing authority into a shipped file, and a hex literal is exactly what a
    re-introduction looks like.
    """
    haystack = KEY_SOURCE.lower()
    for seed, public in RFC8032_TEST_VECTORS:
        assert seed.hex() not in haystack, "an RFC 8032 test-vector SEED is present in keys.py"
        assert public.hex() not in haystack, "an RFC 8032 test-vector public key is present in keys.py"


def test_every_registered_key_is_a_well_formed_ed25519_public_key() -> None:
    """Length and encoding, so a typo cannot register a key that never verifies."""
    for entry in keys.KEYS:
        assert isinstance(entry.public_key, bytes), entry.key_id
        assert len(entry.public_key) == 32, entry.key_id
        assert entry.key_id, "a registered key must have a non-empty id"


def test_registry_is_configured_reports_the_actual_state() -> None:
    assert keys.registry_is_configured() is bool(keys.KEYS)


def test_an_empty_registry_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented safe default: no key means community, never "trust all"."""
    monkeypatch.setattr(keys, "KEYS", ())
    assert keys.key_for("ktest") is None
    assert keys.verify_signature("ktest", b"message", b"\x00" * 64) is False
    assert keys.registry_is_configured() is False


def test_release_refuses_to_ship_without_a_production_verify_key() -> None:
    """scripts/release.py must treat an empty registry as a blocker.

    Source-pin rather than an invocation: the release script's gate runs against
    a tree being packaged, and the property here is that the gate exists, reads
    the tree's own registry file, and knows the published vectors by value (the
    repo's established wiring-pin convention).
    """
    import importlib.util

    release_path = REPO_ROOT / "scripts" / "release.py"
    release_source = release_path.read_text(encoding="utf-8")
    assert "licensing_key_problems" in release_source
    assert "licensing" in release_source and "keys.py" in release_source
    # And behaviourally: the real gate passes a configured registry and blocks
    # an empty or forgeable one. The production k2 key was registered on
    # 2026-10-19, so this tree exercises the passing branch.
    spec = importlib.util.spec_from_file_location("oracle_release_hygiene", release_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module's dataclasses resolve their own module
    # through sys.modules, so an unregistered exec raises from dataclasses.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    problems = module.licensing_key_problems(REPO_ROOT)
    if not keys.registry_is_configured():
        assert problems, "an unconfigured registry must block a release"
        assert "no production verify key" in problems[0]
    else:
        assert problems == [], problems


def test_the_scan_would_catch_a_planted_test_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vacuity guard: the same check must fail on the historical defect.

    Re-registers k1 exactly as the compromised revision had it and runs the
    production-registry scan again. A green run above is only evidence if this
    one goes red.
    """
    from tests.licensing_fixtures import TEST_PUBLIC_KEY

    monkeypatch.setattr(
        keys, "KEYS", (keys.VerifyKey(key_id="k1", public_key=TEST_PUBLIC_KEY),)
    )
    forbidden = {public for _seed, public in RFC8032_TEST_VECTORS}
    assert [entry.key_id for entry in keys.KEYS if entry.public_key in forbidden] == ["k1"], (
        "the vacuity probe failed: the scan cannot see the historical defect"
    )
    # And the key really is the test vector the fixtures mint with.
    assert _ed25519.verify(TEST_PUBLIC_KEY, b"", bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    ))
