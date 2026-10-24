from __future__ import annotations

import json
from pathlib import Path

import pytest

from the_oracle import cli
from the_oracle.licensing import policy, store, tokens
from the_oracle.licensing.tokens import LicenseStatus
from tests.licensing_fixtures import TEST_KEY_ID, TEST_SEED

pytestmark = pytest.mark.usefixtures("test_license_registry")


def _unlicensed() -> LicenseStatus:
    return LicenseStatus(ok=False, state="no_token", detail="No serial number is activated.")


def _token() -> str:
    return tokens.mint(
        {
            "v": 2,
            "key_id": TEST_KEY_ID,
            "lic_id": "77777777-7777-4777-7777-777777777777",
            "licensee": "Cody",
            "machine_hash": "",
            "iat": 1700000000,
            "exp": None,
        },
        TEST_SEED,
    )


def test_unlicensed_status_rejects_every_functional_feature() -> None:
    with pytest.raises(policy.LicenseRequired, match="serial number"):
        policy.require(_unlicensed(), "render")


def test_license_status_reports_licensed_boolean_without_edition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from the_oracle import offline

    monkeypatch.setattr(offline, "repo_root", lambda: tmp_path)
    assert cli.handle_license_status() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["licensed"] is False
    assert "edition" not in payload
    store.save_token(tmp_path, _token(), verify_before_save=tokens.verify_token)
    assert cli.handle_license_status() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["licensed"] is True
    assert "edition" not in payload


def test_cli_functional_command_requires_serial_number(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("the_oracle.licensing.current_license", _unlicensed)
    assert cli.main(["voices"]) == 1
    assert "serial number" in capsys.readouterr().err
