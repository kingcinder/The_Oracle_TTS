"""Direct enforcement pins for the licensing policy surface.

The 2026-09-27 campaign reconciliation found these symbols had no test
references at all — the enforcement machinery existed but nothing pinned it.
Everything here is the production truth the CLI, GUI, and doctor all read
(docs/LICENSING_DESIGN.md §4):

- community = today's full feature set, so nothing existing is gated;
- trial behaves as studio while valid;
- an expired/absent/degraded license runs as community (degrade, never lock);
- require() raises the typed LicenseRequired ONLY outside the edition's
  entitlements, with the purchase-path message;
- `the-oracle license-status` reports the install's state as JSON.

The degrade-never-lock guarantee is pinned twice: on hand-built statuses (the
edition map and require() in isolation) and again on statuses that came out of
the real verifier, so the edition the product computes is the one under test.
Verified statuses use the **injected** test-only registry
(tests/licensing_fixtures.py); the shipped table holds no test key.

A regression in any of these silently changes what customers can run —
which is exactly the class of drift the safety nets exist to catch.
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
    DEFAULT_EDITION,
    EDITIONS,
    LicenseRequired,
    current_edition,
    edition_entitlements,
    require,
)
from the_oracle.licensing.tokens import LicenseStatus  # noqa: E402
from tests.licensing_fixtures import TEST_KEY_ID, TEST_SEED  # noqa: E402

#: Verified statuses come from the INJECTED test-only registry: the production
#: registry ships no test key (tests/test_licensing_registry_hygiene.py).
pytestmark = pytest.mark.usefixtures("test_license_registry")

#: Test-only crypto material (RFC 8032 §7.1 TEST 1, a published vector).
SEED = TEST_SEED

#: The entitlements every edition must grant — "community is everything that
#: ships today" is the rollout rule that makes the licensing unit safe.
CORE_ENTITLEMENTS = frozenset({"render", "conversation", "recording_studio", "voice_craft"})


def _status(
    *,
    ok: bool = True,
    state: str = "valid",
    edition: str | None = "studio",
    exp: int | None = None,
) -> LicenseStatus:
    return LicenseStatus(
        ok=ok,
        state=state,
        edition=edition,
        licensee="Cody",
        lic_id="test-lic",
        key_id=TEST_KEY_ID,
        exp=exp,
        machine_locked=False,
        detail="",
    )


# --- the edition map ---------------------------------------------------------


def test_community_grants_everything_that_ships_today() -> None:
    assert CORE_ENTITLEMENTS <= edition_entitlements("community")


def test_studio_and_trial_add_only_future_entitlements() -> None:
    assert EDITIONS["community"] < EDITIONS["studio"]
    assert EDITIONS["studio"] == EDITIONS["trial"]


def test_unknown_or_missing_edition_degrades_to_community() -> None:
    assert edition_entitlements("") == EDITIONS[DEFAULT_EDITION]
    assert edition_entitlements(None) == EDITIONS[DEFAULT_EDITION]
    assert edition_entitlements("future_edition") == EDITIONS[DEFAULT_EDITION]


# --- current_edition: the degrade-never-lock rule ---------------------------


def test_current_edition_degrades_absent_and_failed_licenses() -> None:
    assert current_edition(_status(ok=False, state="no_token", edition=None)) == "community"
    assert current_edition(_status(ok=False, state="expired", edition="studio")) == "community"
    assert current_edition(_status(ok=True, state="valid", edition="trial")) == "trial"
    assert current_edition(_status(ok=True, state="valid", edition="community")) == "community"


# --- require(): the single enforcement gate ---------------------------------


def test_require_passes_for_core_entitlements_under_any_edition() -> None:
    for edition in ("community", "studio", "trial", None):
        status = _status(edition=edition)
        for entitlement in CORE_ENTITLEMENTS:
            require(status, entitlement)  # must not raise


def test_require_raises_typed_license_required_for_studio_features() -> None:
    status = _status(edition="community")
    with pytest.raises(LicenseRequired) as excinfo:
        require(status, "studio_features")
    assert "offline" in str(excinfo.value)  # the purchase-path message


def test_require_raises_for_expired_studio_license() -> None:
    with pytest.raises(LicenseRequired):
        require(_status(ok=False, state="expired", edition="studio"), "studio_features")


# --- degrade-never-lock, proven on REAL verified statuses --------------------


def test_real_studio_license_grants_studio_features() -> None:
    """A genuinely minted+verified studio token reaches studio entitlements.

    Every other test in this file reads a hand-built LicenseStatus; this one
    goes through the real verifier so the edition the product actually computes
    is the one under test (the "verify" half of the guarantee).
    """
    valid = tokens.verify_token(_minted_token(), now=1700000001.0)
    assert valid.ok is True and valid.state == "valid"
    assert current_edition(valid) == "studio"
    require(valid, "studio_features")  # must not raise
    for entitlement in CORE_ENTITLEMENTS:
        require(valid, entitlement)


def test_real_expired_studio_license_degrades_to_community_and_loses_nothing() -> None:
    """The release guarantee, end to end through verification.

    An expired studio token must (a) not be ok, (b) run as community, (c) refuse
    a studio-only entitlement with the typed purchase-path error, and (d) still
    grant every entitlement that ships today — renders in flight finish and no
    existing workflow is locked away. "Expired" is a billing state, never a
    damaged install.
    """
    exp = 1_700_000_500
    expired = tokens.verify_token(_minted_token(exp=exp), now=float(exp + 60))
    assert expired.ok is False
    assert expired.state == "expired"
    assert current_edition(expired) == "community"

    with pytest.raises(LicenseRequired) as excinfo:
        require(expired, "studio_features")
    assert "Studio" in str(excinfo.value)

    # Degrade, never lock: the whole shipped feature set keeps working.
    for entitlement in CORE_ENTITLEMENTS:
        require(expired, entitlement)
    assert CORE_ENTITLEMENTS <= edition_entitlements(current_edition(expired))


# --- the CLI surface: license-status -----------------------------------------


@pytest.fixture()
def sandbox_root(tmp_path: Path, monkeypatch) -> Path:
    from the_oracle import offline

    monkeypatch.setattr(offline, "repo_root", lambda: tmp_path)
    return tmp_path


def _minted_token(exp: int | None = None) -> str:
    payload = {
        "v": 1,
        "key_id": TEST_KEY_ID,
        "lic_id": "77777777-7777-4777-7777-777777777777",
        "edition": "studio",
        "licensee": "Cody",
        "machine_hash": "",
        "iat": 1700000000,
        "exp": exp,
    }
    return tokens.mint(payload, SEED)


def test_license_status_reports_community_when_no_token(sandbox_root: Path, capsys) -> None:
    args = cli.build_parser().parse_args(["license-status"])
    assert cli.handle_license_status() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["state"] == "no_token"
    assert payload["edition"] == "community"


def test_license_status_reports_the_valid_stored_license(sandbox_root: Path, capsys) -> None:
    result = store.save_token(
        sandbox_root, _minted_token(), verify_before_save=tokens.verify_token
    )
    assert result.state == "saved"  # save_token's success state (load_token says "loaded")
    args = cli.build_parser().parse_args(["license-status"])
    assert cli.handle_license_status() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["state"] == "valid"
    assert payload["edition"] == "studio"
    assert payload["licensee"] == "Cody"


def test_license_status_fails_closed_on_a_corrupted_store(sandbox_root: Path, capsys) -> None:
    store.store_path(sandbox_root).write_text("{not json", encoding="utf-8")
    args = cli.build_parser().parse_args(["license-status"])
    assert cli.handle_license_status() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["state"] != "valid"
    assert payload["detail"], "a corrupted store must explain itself"
