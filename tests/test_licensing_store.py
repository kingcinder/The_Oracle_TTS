"""The license store: atomic writes, typed failures, save-then-verify order.

Registry: tokens are minted for the **injected** test-only registry
(tests/licensing_fixtures.py); the shipped table holds no test key, so nothing
here can pass by trusting a published one.

**Mutation contract:** removing the verify_before_save gate from
store.save_token must fail test_refuses_to_store_an_invalid_token (M4);
breaking the atomic write (direct write to the final path) is pinned by
test_atomic_write_leaves_no_temp_file.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle.licensing import store, tokens  # noqa: E402
from tests.licensing_fixtures import TEST_KEY_ID, TEST_SEED  # noqa: E402

#: Tokens are minted for the INJECTED test-only registry: the production
#: registry ships no test key (see tests/test_licensing_registry_hygiene.py).
pytestmark = pytest.mark.usefixtures("test_license_registry")

#: Test-only crypto material (RFC 8032 §7.1 TEST 1, a published vector).
SEED = TEST_SEED

PAYLOAD = {
    "v": 2,
    "key_id": TEST_KEY_ID,
    "lic_id": "44444444-4444-4444-8444-444444444444",
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
    """An unreadable store is a typed state, never a traceback.

    The mechanism has to be one the running platform actually honours. This
    test used to ``chmod(0o000)`` the token file, which NTFS does not enforce
    against the *owner*: on Windows the file stayed readable, a token of ``{}``
    loaded as corrupted_content, and the assertion failed in CI for a reason
    that had nothing to do with the store. A directory sitting where the token
    file belongs raises OSError on every platform (POSIX ``IsADirectoryError``,
    Windows ``PermissionError``) and is a real shape on disk — a stray
    directory, a half-synced backup — so the portable half of the claim is
    pinned everywhere and the permission-specific half below where it can be.
    """
    path = store.store_path(tmp_path)
    path.mkdir()
    result = store.load_token(tmp_path)
    assert result.state == "unreadable"
    assert "could not read" in result.detail


@pytest.mark.skipif(
    os.name == "nt",
    reason="NTFS does not deny the owner read access for chmod(0o000)",
)
def test_permission_denied_store_is_typed(tmp_path: Path) -> None:
    """The same state from a permission failure, where permissions mean that."""
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
