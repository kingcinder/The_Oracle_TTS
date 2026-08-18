"""Regression guard for the global ORACLE_* env isolation.

The autouse ``_restore_oracle_env`` fixture in ``tests/conftest.py`` must
guarantee that no test can leak ``ORACLE_*`` environment variables into a
later test — even when the code under test writes ``os.environ`` directly,
bypassing ``monkeypatch``. These two tests prove the mechanism: the first
deliberately mutates the environment the worst possible way (direct writes,
including a brand-new key), and the second asserts the environment is exactly
back to the module baseline. If the fixture is ever weakened or removed, the
second test fails loudly instead of silently hiding the pollution.
"""

from __future__ import annotations

import os

_ORACLE_PREFIX = "ORACLE_"

# Captured at module import (before any test in this module runs). Because the
# autouse fixture restores ORACLE_* after every test — including tests in
# earlier files — this is the clean baseline the mutating test must not disturb.
_BASELINE = {
    key: value
    for key, value in os.environ.items()
    if key.startswith(_ORACLE_PREFIX)
}


def _oracle_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_ORACLE_PREFIX)
    }


def test_oracle_env_mutated_directly() -> None:
    """Deliberately pollute the session env the worst possible way: direct
    ``os.environ`` writes and deletions that ``monkeypatch`` cannot track — a
    changed value, a brand-new key, and every pre-existing ORACLE_* key
    removed. The autouse fixture must put all of it back afterwards."""
    os.environ["ORACLE_AUDIOCPP_MODEL"] = "/tmp/leaked-model.gguf"
    os.environ["ORACLE_BRAND_NEW_FLAG"] = "1"
    for key in [key for key in os.environ if key.startswith(_ORACLE_PREFIX)]:
        os.environ.pop(key, None)


def test_oracle_env_restored_after_mutating_test() -> None:
    """Runs after the mutator; the autouse restore fixture must have put the
    environment back to the module baseline — changed values, created keys,
    and deleted keys all restored."""
    assert _oracle_env() == _BASELINE
