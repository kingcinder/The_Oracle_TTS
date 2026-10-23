"""EULA acceptance: the record is keyed to the exact text, fails closed, and a
declined dialog records nothing."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import eula_acceptance  # noqa: E402


@pytest.fixture()
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Point the packaged copy at a location with no EULA so these tests drive
    # the root-fallback text they write themselves.
    monkeypatch.setattr(eula_acceptance, "__file__", str(tmp_path / "absent" / "eula_acceptance.py"))
    (tmp_path / eula_acceptance.EULA_FILENAME).write_text("EULA v1\n", encoding="utf-8")
    return tmp_path


def test_not_accepted_without_record(root: Path) -> None:
    assert eula_acceptance.is_accepted(root) is False


def test_record_then_accepted(root: Path) -> None:
    eula_acceptance.record_acceptance(root)
    assert eula_acceptance.is_accepted(root) is True


def test_changed_eula_asks_again(root: Path) -> None:
    eula_acceptance.record_acceptance(root)
    (root / eula_acceptance.EULA_FILENAME).write_text("EULA v2\n", encoding="utf-8")
    assert eula_acceptance.is_accepted(root) is False


def test_malformed_record_fails_closed(root: Path) -> None:
    eula_acceptance.acceptance_path(root).write_text("{not json", encoding="utf-8")
    assert eula_acceptance.is_accepted(root) is False


def test_wrong_shape_fails_closed(root: Path) -> None:
    eula_acceptance.acceptance_path(root).write_text(json.dumps(["x"]), encoding="utf-8")
    assert eula_acceptance.is_accepted(root) is False


def test_missing_eula_is_never_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eula_acceptance, "__file__", str(tmp_path / "absent" / "eula_acceptance.py"))
    assert eula_acceptance.is_accepted(tmp_path) is False
    with pytest.raises(FileNotFoundError):
        eula_acceptance.record_acceptance(tmp_path)


def test_packaged_eula_matches_root_copy() -> None:
    """The copy shipped inside the package must equal the repo-root EULA."""
    packaged = eula_acceptance.eula_path(REPO_ROOT / "nonexistent")
    assert packaged.parent == REPO_ROOT / "src" / "the_oracle"
    assert packaged.read_bytes() == (REPO_ROOT / "EULA-DRAFT.md").read_bytes()


def test_record_stores_digest_of_current_text(root: Path) -> None:
    path = eula_acceptance.record_acceptance(root)
    stored = json.loads(path.read_text(encoding="utf-8"))
    _text, digest = eula_acceptance.eula_text_and_digest(root)
    assert stored["eula_sha256"] == digest
    assert os.path.exists(path)
