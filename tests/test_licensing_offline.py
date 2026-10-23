"""The offline guarantee, licensing edition (docs/LICENSING_DESIGN.md §6).

1. Activation and verification perform ZERO network I/O — proven with
   outbound connections forbidden.
2. Importing ``the_oracle.licensing`` never imports a heavyweight/ML
   dependency — the import-discipline pin, proven by module-delta.
3. The package source imports no network module at all — the architectural
   pin that makes "no phone-home" a structural fact, not a behavior.

Registry: every token here verifies against the **injected** test-only registry
(tests/licensing_fixtures.py). The shipped table holds no test key, and
test_production_install_does_not_trust_the_test_key pins that from the other
direction, so a green run here cannot be evidence that a shipped install would
accept the suite's tokens.

**Mutation contract:** adding ``import socket`` (or any network import) to a
licensing module must fail test_package_imports_no_network_modules; making
verification lazily fetch anything (e.g. calling huggingface_hub) must fail
test_activation_and_verification_perform_no_network_io or the import-delta pin.
"""

from __future__ import annotations

import importlib
import socket
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle.licensing import keys, store, tokens  # noqa: E402
from tests.licensing_fixtures import TEST_KEY_ID, TEST_PUBLIC_KEY, TEST_SEED  # noqa: E402

#: Every test here verifies through the INJECTED test-only registry — the
#: production table ships no test key, so these tests must not depend on it.
pytestmark = pytest.mark.usefixtures("test_license_registry")

#: Test-only crypto material (RFC 8032 §7.1 TEST 1, a published vector).
SEED = TEST_SEED

PAYLOAD = {
    "v": 2,
    "key_id": TEST_KEY_ID,
    "lic_id": "66666666-6666-4666-8666-666666666666",
    "licensee": "Cody",
    "machine_hash": "",
    "iat": 1700000000,
    "exp": None,
}


@pytest.fixture()
def no_network(monkeypatch):
    """Same pattern as tests/test_offline_guarantee.py: any connect attempt
    fails loudly and is recorded."""
    attempts: list = []

    def blocked(self, address):  # noqa: ANN001
        attempts.append(address)
        raise AssertionError(f"outbound connection attempted to {address}")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    return attempts


def _verifier():
    return lambda token: tokens.verify_token(token)


def test_production_install_does_not_trust_the_test_key(monkeypatch) -> None:
    """Inverts the removed premise that the shipped registry holds k1.

    The old suite verified its tokens against the PRODUCTION table because that
    table contained the published RFC 8032 test key, which meant a shipped
    install trusted a forgeable key. The table must now hold no such key, so an
    installed build that never sees this fixture rejects the suite's tokens:
    the fixture is what makes the tests pass, not the product.

    The table is read from the import-time capture in the fixtures module, not
    from ``keys.KEYS``: this module's ``test_license_registry`` fixture has
    already swapped that attribute by the time the body runs.
    """
    from tests.licensing_fixtures import PRODUCTION_KEYS, PRODUCTION_PUBLIC_KEYS

    assert TEST_PUBLIC_KEY not in PRODUCTION_PUBLIC_KEYS
    # Put the shipped table back for this one assertion: the claim is about an
    # install that never sees the fixture, so verifying against the injected
    # registry would test the fixture instead of the product.
    monkeypatch.setattr(keys, "KEYS", PRODUCTION_KEYS)
    status = tokens.verify_token(tokens.mint(PAYLOAD, SEED))
    assert status.ok is False
    assert status.state == "unknown_key"


def test_activation_and_verification_perform_no_network_io(tmp_path: Path, no_network) -> None:
    """Mint -> save -> load -> verify, sockets forbidden, zero attempts."""
    token = tokens.mint(PAYLOAD, SEED)
    assert store.save_token(tmp_path, token, verify_before_save=_verifier()).state == "saved"

    from the_oracle.licensing import current_license

    status = current_license(tmp_path)
    assert status.ok and status.state == "valid"
    assert no_network == []


def test_verification_of_a_bad_token_also_needs_no_network(tmp_path: Path, no_network) -> None:
    from the_oracle.licensing import current_license

    store.store_path(tmp_path).write_text("{broken", encoding="utf-8")
    status = current_license(tmp_path)
    assert status.state == "store_error"
    assert no_network == []


def test_importing_licensing_never_imports_ml_dependencies(monkeypatch) -> None:
    """Module-delta pin: importing the package must not ADD any heavyweight
    module to sys.modules (whatever earlier tests already imported is not
    this package's doing).

    The re-import is undone before returning, and that is load-bearing rather
    than tidy: re-executing the package installs **fresh** module objects in
    ``sys.modules``, while every module that imported ``the_oracle.licensing``
    earlier keeps a reference to the old ones. The registry injection this
    suite relies on patches the module it looks up at test time, so leaving the
    fresh copies in place would leave that patch attached to a module the
    already-imported ``tokens``/``store`` no longer use — verification would
    fall back to the real (empty, fail-closed) registry and later modules would
    fail for a reason that has nothing to do with imports.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("the_oracle.licensing")
    }
    for name in saved:
        del sys.modules[name]
    try:
        before = set(sys.modules)
        import the_oracle.licensing  # noqa: F401

        added = set(sys.modules) - before
        offenders = sorted(
            name
            for name in added
            if any(marker in name for marker in ("huggingface", "torch", "transformers", "chatterbox", "sklearn", "scipy"))
        )
        assert offenders == [], f"licensing import pulled in: {offenders}"
    finally:
        # Drop the fresh copies and put the original objects back, so one
        # canonical module object per name survives for the rest of the session.
        for name in [n for n in sys.modules if n.startswith("the_oracle.licensing")]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_package_imports_no_network_modules() -> None:
    """Source-scan: the licensing package may not import socket/urllib/http/
    requests at all — 'no phone-home' is structural."""
    package_dir = Path(store.__file__).parent
    forbidden = ("import socket", "import urllib", "from urllib", "import http", "from http", "import requests", "import ssl")
    for source_file in package_dir.glob("*.py"):
        text = source_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            assert not any(stripped.startswith(marker) for marker in forbidden), (
                f"{source_file.name}: {stripped}"
            )


def test_expired_remedy_never_suggests_going_online(tmp_path: Path) -> None:
    from the_oracle.licensing import current_license

    # Written directly: the save gate refuses expired tokens, but an install
    # whose perpetual-looking token aged past its exp holds exactly this.
    expired = tokens.mint(dict(PAYLOAD, exp=int(time.time()) - 5), SEED)
    store.store_path(tmp_path).write_text(
        __import__("json").dumps({"token": expired}), encoding="utf-8"
    )
    status = current_license(tmp_path)
    assert status.state == "expired"
    assert "never requires a network connection" in status.detail
