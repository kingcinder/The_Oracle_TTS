"""The consent store — the privacy-critical fail direction, pinned.

**Mutation contract (M-CONSENT):** flipping ANY failure branch of
consent.read_consent() to ``return True`` must fail one of these tests.
Three branches exist (unreadable OSError, bad JSON, wrong shape/value);
each has its own test below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle.crash import consent  # noqa: E402


def test_missing_file_means_opted_out(tmp_path: Path) -> None:
    assert consent.read_consent(tmp_path) is False


def test_unreadable_file_means_opted_out(tmp_path: Path) -> None:
    path = consent.consent_path(tmp_path)
    path.write_text('{"crash_reports_enabled": true}', encoding="utf-8")
    path.chmod(0o000)
    try:
        assert consent.read_consent(tmp_path) is False
    finally:
        path.chmod(0o600)


def test_malformed_json_means_opted_out(tmp_path: Path) -> None:
    consent.consent_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert consent.read_consent(tmp_path) is False


def test_wrong_shape_means_opted_out(tmp_path: Path) -> None:
    for bad in ('"just a string"', '[1, 2]', '{"other_key": true}', '{"crash_reports_enabled": "yes"}',
                '{"crash_reports_enabled": 1}'):
        consent.consent_path(tmp_path).write_text(bad, encoding="utf-8")
        assert consent.read_consent(tmp_path) is False, bad


def test_exact_true_means_opted_in(tmp_path: Path) -> None:
    consent.write_consent(tmp_path, True)
    assert consent.read_consent(tmp_path) is True
    data = json.loads(consent.consent_path(tmp_path).read_text(encoding="utf-8"))
    assert data["crash_reports_enabled"] is True
    assert "updated_at" in data


def test_write_consent_false_persists_opt_out(tmp_path: Path) -> None:
    consent.write_consent(tmp_path, False)
    assert consent.read_consent(tmp_path) is False
    # The file records the explicit opt-out (auditability), but the read
    # direction stays fail-closed either way.
    data = json.loads(consent.consent_path(tmp_path).read_text(encoding="utf-8"))
    assert data["crash_reports_enabled"] is False


def test_write_is_atomic_leaving_no_temp_file(tmp_path: Path) -> None:
    consent.write_consent(tmp_path, True)
    names = [p.name for p in tmp_path.iterdir()]
    assert consent.CONSENT_FILENAME in names
    assert not any(name.endswith(".tmp") for name in names)


def test_explicit_false_is_not_the_same_as_failure(tmp_path: Path) -> None:
    """The CLI shows different text for 'you opted out' vs 'consent unknown';
    read_consent cannot distinguish them (both False), so the CLI must read
    the raw file when it needs the distinction — this test pins that the
    raw file DOES carry the distinction."""
    consent.write_consent(tmp_path, False)
    data = json.loads(consent.consent_path(tmp_path).read_text(encoding="utf-8"))
    assert data["crash_reports_enabled"] is False  # explicit, not "unknown"
    missing = tmp_path / "nothing-here"
    assert not missing.exists()
