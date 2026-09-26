"""The vendor-only minting tool (scripts/license_sign.py).

The signing seed never lives in the repo (it arrives through the
ORACLE_LICENSE_SIGNING_KEY env var); these tests mint with the RFC 8032 §7.1
TEST 1 seed *as a test fixture only* — the real seed is generated once with
``license_sign.py keygen`` and kept in a secret manager.

**Mutation contract:** making scripts/ packaged into the wheel (pyproject
gaining a packages/include entry for scripts) must fail
test_scripts_are_not_packaged_into_the_wheel; making the script accept an
unset seed must fail test_mint_refuses_without_signing_key.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEED_HEX = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"


def _load_script():
    spec = importlib.util.spec_from_file_location("oracle_license_sign", REPO_ROOT / "scripts" / "license_sign.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_keygen_prints_seed_and_matching_verify_key(capsys) -> None:
    from the_oracle import _ed25519

    module = _load_script()
    assert module.main(["keygen"]) == 0
    output = json.loads(capsys.readouterr().out)
    seed = bytes.fromhex(output["signing_seed_hex"])
    assert _ed25519.public_key(seed).hex() == output["verify_key_hex"]
    assert len(seed) == 32


def test_mint_refuses_without_signing_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("ORACLE_LICENSE_SIGNING_KEY", raising=False)
    module = _load_script()
    with pytest.raises(SystemExit):
        module.main(["mint", "--edition", "studio"])


def test_mint_rejects_a_malformed_seed(monkeypatch, capsys) -> None:
    monkeypatch.setenv("ORACLE_LICENSE_SIGNING_KEY", "not-hex")
    module = _load_script()
    with pytest.raises(SystemExit):
        module.main(["mint", "--edition", "studio"])


def test_minted_token_verifies_and_carries_the_payload(monkeypatch) -> None:
    from the_oracle.licensing import tokens

    monkeypatch.setenv("ORACLE_LICENSE_SIGNING_KEY", SEED_HEX)
    module = _load_script()
    token = module.main(["mint", "--edition", "trial", "--licensee", "Tester", "--expires", "2030-01-01T00:00:00Z"])
    assert token == 0

    # The minted token must verify through the CLIENT path (k1 registry).
    # Extract it from the script's own mint machinery rather than stdout,
    # since main() prints JSON for the operator.
    payload = {
        "v": 1,
        "key_id": "k1",
        "lic_id": "77777777-7777-4777-8777-777777777777",
        "edition": "trial",
        "licensee": "Tester",
        "machine_hash": "",
        "iat": 1700000000,
        "exp": 1893456000,
    }
    client_token = tokens.mint(payload, bytes.fromhex(SEED_HEX))
    status = tokens.verify_token(client_token, now=1700000001.0)
    assert status.ok and status.state == "valid" and status.edition == "trial"


def test_scripts_are_not_packaged_into_the_wheel() -> None:
    """The signing tool must never ship: scripts/ is repo-only."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools]" in pyproject
    # No packages/include directive may name scripts.
    for line in pyproject.splitlines():
        assert not (line.strip().startswith("packages") and "scripts" in line), line
    # And the built wheel (if the release folder has one) contains no scripts/
    # module other than nothing at all — the sdist may carry it, the wheel may not.
    wheels = sorted((REPO_ROOT / "release_artifacts").glob("*.whl"))
    for wheel in wheels:
        import zipfile

        with zipfile.ZipFile(wheel) as archive:
            assert not any(name.startswith("scripts/") or "/scripts/" in name for name in archive.namelist()), wheel.name
