"""Direct enforcement pins for the licensing policy surface.

Everything here is the production truth the CLI, GUI, and doctor all read:

- a valid, unexpired, machine-matching product key unlocks the program;
- without one, require() raises the typed LicenseRequired with the
  purchase-path message;
- the render path is never gated (nothing is ever locked away);
- `the-oracle license-status` reports the install's state as JSON.

The rules are pinned on hand-built statuses (require() in isolation) and again
on statuses that came out of the real verifier. Verified statuses use the
**injected** test-only registry (tests/licensing_fixtures.py); the shipped table
holds no test key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import cli  # noqa: E402
from the_oracle.licensing import store, tokens  # noqa: E402
from the_oracle.licensing.policy import (  # noqa: E402
    LicenseRequired,
    is_licensed,
    require,
)
from the_oracle.licensing.tokens import LicenseStatus  # noqa: E402
from tests.licensing_fixtures import TEST_KEY_ID, TEST_SEED  # noqa: E402

#: Verified statuses come from the INJECTED test-only registry: the production
#: registry ships no test key (tests/test_licensing_registry_hygiene.py).
pytestmark = pytest.mark.usefixtures("test_license_registry")

#: Test-only crypto material (RFC 8032 §7.1 TEST 1, a published vector).
SEED = TEST_SEED


def _status(
    *,
    ok: bool = True,
    state: str = "valid",
    exp: int | None = None,
) -> LicenseStatus:
    return LicenseStatus(
        ok=ok,
        state=state,
        licensee="Cody",
        lic_id="test-lic",
        key_id=TEST_KEY_ID,
        exp=exp,
        machine_locked=False,
        detail="",
    )


# --- require(): the single enforcement gate ---------------------------------


def test_a_valid_key_is_licensed_and_passes_require() -> None:
    status = _status()
    assert is_licensed(status) is True
    require(status)  # must not raise


@pytest.mark.parametrize(
    ("ok", "state"),
    [(False, "no_token"), (False, "expired"), (False, "bad_signature"), (True, "store_error")],
)
def test_require_raises_typed_license_required_without_a_valid_key(ok: bool, state: str) -> None:
    status = _status(ok=ok, state=state)
    assert is_licensed(status) is False
    with pytest.raises(LicenseRequired) as excinfo:
        require(status)
    assert "the-oracle activate" in str(excinfo.value)  # the purchase-path message


# --- verified statuses: the real verifier decides ---------------------------


def test_a_real_minted_key_unlocks_the_program() -> None:
    """A genuinely minted+verified token passes the gate.

    Every other test here reads a hand-built LicenseStatus; this one goes
    through the real verifier (the "verify" half of the guarantee).
    """
    valid = tokens.verify_token(_minted_token(), now=1700000001.0)
    assert valid.ok is True and valid.state == "valid"
    assert is_licensed(valid) is True
    require(valid)  # must not raise


def test_a_real_expired_key_is_blocked_but_loses_nothing_on_disk() -> None:
    """An expired token blocks the program with the typed error.

    The render path is not behind this gate at all (see
    tests/test_license_state_render.py), so an expired key never takes an
    existing render away from the user.
    """
    exp = 1_700_000_500
    expired = tokens.verify_token(_minted_token(exp=exp), now=float(exp + 60))
    assert expired.ok is False
    assert expired.state == "expired"
    with pytest.raises(LicenseRequired):
        require(expired)


# --- the CLI surface: license-status -----------------------------------------


@pytest.fixture()
def sandbox_root(tmp_path: Path, monkeypatch) -> Path:
    from the_oracle import offline

    monkeypatch.setattr(offline, "repo_root", lambda: tmp_path)
    return tmp_path


def _minted_token(exp: int | None = None) -> str:
    payload = {
        "v": 2,
        "key_id": TEST_KEY_ID,
        "lic_id": "77777777-7777-4777-7777-777777777777",
        "licensee": "Cody",
        "machine_hash": "",
        "iat": 1700000000,
        "exp": exp,
    }
    return tokens.mint(payload, SEED)


def test_license_status_reports_no_token_when_nothing_is_activated(sandbox_root: Path, capsys) -> None:
    assert cli.handle_license_status() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["state"] == "no_token"
    assert "edition" not in payload


def test_license_status_reports_the_valid_stored_license(sandbox_root: Path, capsys) -> None:
    result = store.save_token(
        sandbox_root, _minted_token(), verify_before_save=tokens.verify_token
    )
    assert result.state == "saved"  # save_token's success state (load_token says "loaded")
    assert cli.handle_license_status() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["state"] == "valid"
    assert payload["licensee"] == "Cody"


def test_license_status_fails_closed_on_a_corrupted_store(sandbox_root: Path, capsys) -> None:
    store.store_path(sandbox_root).write_text("{not json", encoding="utf-8")
    assert cli.handle_license_status() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["state"] != "valid"
    assert payload["detail"], "a corrupted store must explain itself"
