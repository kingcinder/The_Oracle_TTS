"""The activation flow end-to-end: `the-oracle activate` → current_license.

The handler must refuse a bad token BEFORE any bytes are written (the store
cannot come to hold a token the verifier rejects), and a good token activates
offline against a sandboxed repo root (offline.repo_root monkeypatched — the
handler must never write into the real repo during tests).

**Mutation contract:** making handle_activate write the token before
verification must fail test_bad_token_writes_nothing (that ordering is the
save gate's whole point, so the e2e test guards it independently of M4).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import cli  # noqa: E402
from the_oracle.licensing import store, tokens  # noqa: E402

SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")

PAYLOAD = {
    "v": 1,
    "key_id": "k1",
    "lic_id": "88888888-8888-4888-8888-888888888888",
    "edition": "studio",
    "licensee": "Cody",
    "machine_hash": "",
    "iat": 1700000000,
    "exp": None,
}


@pytest.fixture()
def sandbox_root(tmp_path: Path, monkeypatch) -> Path:
    from the_oracle import offline

    monkeypatch.setattr(offline, "repo_root", lambda: tmp_path)
    return tmp_path


def _good_token() -> str:
    return tokens.mint(PAYLOAD, SEED)


def test_good_token_activates_offline(sandbox_root: Path, capsys) -> None:
    args = cli.build_parser().parse_args(["activate", _good_token()])
    assert cli.handle_activate(args) == 0
    output = capsys.readouterr().out
    assert "Activated" in output and "no server was contacted" in output

    from the_oracle.licensing import current_license
    from the_oracle import offline

    status = current_license(offline.repo_root())
    assert status.ok and status.state == "valid" and status.edition == "studio"


def test_bad_token_writes_nothing(sandbox_root: Path, capsys) -> None:
    args = cli.build_parser().parse_args(["activate", "ORACLE1.forged.forged"])
    assert cli.handle_activate(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert not store.store_path(sandbox_root).exists()


def test_no_token_at_all_is_a_usage_error(sandbox_root: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", type("Stdin", (), {"read": lambda self: ""})())
    args = cli.build_parser().parse_args(["activate"])
    assert cli.handle_activate(args) == 2
    assert "no token" in capsys.readouterr().err


def test_activation_via_stdin_pipe(sandbox_root: Path, monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", type("Stdin", (), {"read": lambda self: _good_token() + "\n"})())
    args = cli.build_parser().parse_args(["activate"])
    assert cli.handle_activate(args) == 0
    assert store.load_token(sandbox_root).state == "loaded"


def test_reactivation_replaces_the_stored_token(sandbox_root: Path) -> None:
    first = cli.build_parser().parse_args(["activate", _good_token()])
    assert cli.handle_activate(first) == 0
    second_payload = dict(PAYLOAD, licensee="Other Name", lic_id="99999999-9999-4999-8999-999999999999")
    second = cli.build_parser().parse_args(["activate", tokens.mint(second_payload, SEED)])
    assert cli.handle_activate(second) == 0
    status = tokens.verify_token(store.load_token(sandbox_root).stored.token)
    assert status.licensee == "Other Name"


def test_machine_id_prints_hash_only(sandbox_root: Path, capsys) -> None:
    args = cli.build_parser().parse_args(["machine-id"])
    assert cli.handle_machine_id() == 0
    captured = capsys.readouterr()
    printed = captured.out.strip()
    assert len(printed) == 32 and all(c in "0123456789abcdef" for c in printed)
    assert "hash" in captured.err  # the privacy note
