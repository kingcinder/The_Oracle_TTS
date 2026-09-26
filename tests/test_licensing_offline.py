"""The offline guarantee, licensing edition (docs/LICENSING_DESIGN.md §6).

1. Activation and verification perform ZERO network I/O — proven with
   outbound connections forbidden.
2. Importing ``the_oracle.licensing`` never imports a heavyweight/ML
   dependency — the import-discipline pin, proven by module-delta.
3. The package source imports no network module at all — the architectural
   pin that makes "no phone-home" a structural fact, not a behavior.

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

from the_oracle.licensing import store, tokens  # noqa: E402

SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")

PAYLOAD = {
    "v": 1,
    "key_id": "k1",
    "lic_id": "66666666-6666-4666-8666-666666666666",
    "edition": "studio",
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
    this package's doing)."""
    for name in [n for n in sys.modules if n.startswith("the_oracle.licensing")]:
        del sys.modules[name]

    before = set(sys.modules)
    import the_oracle.licensing  # noqa: F401

    added = set(sys.modules) - before
    offenders = sorted(
        name
        for name in added
        if any(marker in name for marker in ("huggingface", "torch", "transformers", "chatterbox", "sklearn", "scipy"))
    )
    assert offenders == [], f"licensing import pulled in: {offenders}"


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
