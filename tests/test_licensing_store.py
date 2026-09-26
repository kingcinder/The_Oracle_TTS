"""The license store: atomic writes, typed failures, save-then-verify order.

**Mutation contract:** removing the verify_before_save gate from
store.save_token must fail test_refuses_to_store_an_invalid_token (M4);
breaking the atomic write (direct write to the final path) is pinned by
test_atomic_write_leaves_no_temp_file.
"""

from __future__ import annotations

import json
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
    "lic_id": "44444444-4444-4444-8444-444444444444",
    "edition": "studio",
    "licensee": "Cody",
    "machine_hash": "",
    "iat": 1700000000,
    "exp": None,
}


def _good_token() -> str:
    return tokens.mint(PAYLOAD, SEED)


def _verifier():
    return lambda token: tokens.verify_token(token)


def test_missing_store_is_no_token(tmp_path: Path) -> None:
    assert store.load_token(tmp_path).state == "no_token"


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    token = _good_token()
    result = store.save_token(tmp_path, token, verify_before_save=_verifier())
    assert result.state == "saved"
    loaded = store.load_token(tmp_path)
    assert loaded.state == "loaded"
    assert loaded.stored.token == token


def test_refuses_to_store_an_invalid_token(tmp_path: Path) -> None:
    """**Mutation M4:** deleting the verify gate in save_token flips this."""
    result = store.save_token(tmp_path, "ORACLE1.garbage.garbage", verify_before_save=_verifier())
    assert result.state == "refused"
    assert not store.store_path(tmp_path).exists()


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    store.save_token(tmp_path, _good_token(), verify_before_save=_verifier())
    names = [p.name for p in tmp_path.iterdir()]
    assert store.STORE_FILENAME in names
    assert not any(name.endswith(".tmp") for name in names)


def test_corrupted_json_is_typed_not_raised(tmp_path: Path) -> None:
    store.store_path(tmp_path).write_text("{not json", encoding="utf-8")
    result = store.load_token(tmp_path)
    assert result.state == "corrupted_content"
    assert result.detail  # remedy present


def test_wrong_shape_is_corrupted_content(tmp_path: Path) -> None:
    store.store_path(tmp_path).write_text('{"something": 1}', encoding="utf-8")
    assert store.load_token(tmp_path).state == "corrupted_content"


def test_unreadable_store_is_typed(tmp_path: Path) -> None:
    path = store.store_path(tmp_path)
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o000)
    try:
        result = store.load_token(tmp_path)
        assert result.state == "unreadable"
    finally:
        path.chmod(0o600)


def test_clear_token_removes_and_reports(tmp_path: Path) -> None:
    store.save_token(tmp_path, _good_token(), verify_before_save=_verifier())
    assert store.clear_token(tmp_path) is True
    assert store.load_token(tmp_path).state == "no_token"
    assert store.clear_token(tmp_path) is False


def test_store_never_holds_a_token_the_verifier_rejects(tmp_path: Path) -> None:
    """End-to-end of the save gate: a token that expired between mint and
    save is refused, so the store cannot contain a future store_error."""
    expired = tokens.mint(dict(PAYLOAD, exp=int(time.time()) - 5), SEED)
    result = store.save_token(tmp_path, expired, verify_before_save=_verifier())
    assert result.state == "refused"
    from the_oracle.licensing import current_license

    assert current_license(tmp_path).state == "no_token"
