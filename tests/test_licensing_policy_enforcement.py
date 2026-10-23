"""Pin the Community default and the one gated Studio entitlement."""

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


def test_a_valid_key_is_licensed_and_passes_studio_entitlement() -> None:
    status = _status()
    assert is_licensed(status) is True
    require(status, "studio_features")


@pytest.mark.parametrize(
    ("ok", "state"),
    [(False, "no_token"), (False, "expired"), (False, "bad_signature"), (True, "store_error")],
)
def test_community_keeps_existing_entitlements_but_not_studio_features(ok: bool, state: str) -> None:
    status = _status(ok=ok, state=state)
    assert is_licensed(status) is False
    require(status, "render")
    with pytest.raises(LicenseRequired) as excinfo:
        require(status, "studio_features")
    assert "Studio" in str(excinfo.value)


# --- verified statuses: the real verifier decides ---------------------------


def test_a_real_minted_key_unlocks_the_program() -> None:
    """A genuinely minted+verified token passes the gate.

    Every other test here reads a hand-built LicenseStatus; this one goes
    through the real verifier (the "verify" half of the guarantee).
    """
    valid = tokens.verify_token(_minted_token(), now=1700000001.0)
    assert valid.ok is True and valid.state == "valid"
    assert is_licensed(valid) is True
    require(valid, "studio_features")


def test_a_real_expired_key_degrades_to_community() -> None:
    """An expired token loses Studio-only entitlements while Community works."""
    exp = 1_700_000_500
    expired = tokens.verify_token(_minted_token(exp=exp), now=float(exp + 60))
    assert expired.ok is False
    assert expired.state == "expired"
    require(expired, "render")
    with pytest.raises(LicenseRequired):
        require(expired, "studio_features")


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
    assert payload["edition"] == "community"


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


def test_community_can_use_existing_cli_commands_without_activation(monkeypatch) -> None:
    from the_oracle import licensing

    calls: list[str] = []
    monkeypatch.setattr(
        licensing,
        "current_license",
        lambda: LicenseStatus(ok=False, state="no_token", detail="No key."),
    )
    monkeypatch.setattr(cli, "handle_voices", lambda _args: calls.append("voices") or 0)

    assert cli.main(["voices"]) == 0
    assert calls == ["voices"]
