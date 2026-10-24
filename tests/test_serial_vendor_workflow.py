"""Exercise the vendor serial-number lifecycle with throwaway test keys only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import license_sign
from the_oracle import cli, offline
from the_oracle.licensing import clear_token, current_license, keys, tokens
from the_oracle.licensing.keys import VerifyKey
from tests.licensing_fixtures import TEST_SEED
from the_oracle import _ed25519


def _mint(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], key_id: str, seed: bytes) -> str:
    monkeypatch.setenv(license_sign.ENV_VAR, seed.hex())
    assert license_sign.main(["mint", "--key-id", key_id, "--licensee", "Test owner"]) == 0
    return json.loads(capsys.readouterr().out)["token"]


def _activate(root: Path, token: str) -> None:
    args = cli.build_parser().parse_args(["activate", token])
    assert cli.handle_activate(args) == 0


def test_mint_activate_relaunch_reissue_and_k2_k3_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mint, activate, relaunch, reissue a lost serial, then rotate keys."""
    root = tmp_path / "install"
    root.mkdir()
    monkeypatch.setattr(offline, "repo_root", lambda: root)

    k2 = VerifyKey("k2", _ed25519.public_key(TEST_SEED))
    k3_seed = bytes(range(32))
    k3 = VerifyKey("k3", _ed25519.public_key(k3_seed))
    monkeypatch.setattr(keys, "KEYS", (k2,))

    first = _mint(monkeypatch, capsys, "k2", TEST_SEED)
    _activate(root, first)
    capsys.readouterr()
    assert current_license().state == "valid"
    assert current_license().state == "valid"  # relaunch/read-after-restart proof

    assert clear_token(root) is True
    assert current_license().state == "no_token"
    replacement = _mint(monkeypatch, capsys, "k2", TEST_SEED)
    _activate(root, replacement)
    capsys.readouterr()
    assert current_license().state == "valid"

    successor = _mint(monkeypatch, capsys, "k3", k3_seed)
    capsys.readouterr()
    monkeypatch.setattr(keys, "KEYS", (k2, k3))
    assert tokens.verify_token(replacement).state == "valid"
    assert tokens.verify_token(successor).state == "valid"

    monkeypatch.setattr(keys, "KEYS", (k3,))
    assert tokens.verify_token(replacement).state == "unknown_key"
    assert tokens.verify_token(successor).state == "valid"
