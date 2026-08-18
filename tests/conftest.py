"""Shared pytest fixtures for The Oracle's test suite.

The star here is :func:`_restore_oracle_env`: an autouse fixture that makes
``ORACLE_*`` environment variables a per-test sandbox. Without it, a test
that (or whose code under test) writes ``ORACLE_AUDIOCPP_MODEL`` /
``ORACLE_AUDIOCPP_CLI`` straight into ``os.environ`` silently changes the
state every later test sees — the classic cross-file pollution that made
``test_vulkan_backend.py`` fail only when suites ran together.
"""

from __future__ import annotations

import os

import pytest

_ORACLE_ENV_PREFIX = "ORACLE_"


@pytest.fixture(autouse=True)
def _restore_oracle_env():
    """Snapshot and restore every ``ORACLE_*`` environment variable per test.

    ``monkeypatch`` alone cannot guarantee a clean session env: it only
    restores the keys a test *explicitly* patched, whereas the code under test
    (``vulkan_setup.run_vulkan_setup``, the GUI's Vulkan handlers) writes
    ``ORACLE_AUDIOCPP_MODEL`` / ``ORACLE_AUDIOCPP_CLI`` directly into
    ``os.environ`` — and tests exercise that code. This fixture therefore
    snapshots the whole ``ORACLE_*`` set before every test and restores it in
    teardown: values changed are put back, keys deleted are re-added, and keys
    created during the test are removed. Tests can never leak session state
    (e.g. a downloaded model path) into a later test or suite.
    """
    saved = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_ORACLE_ENV_PREFIX)
    }
    yield
    # Walk the union of saved and current keys: a key created by the test is
    # popped, a changed key is put back, and a pre-existing key that the test
    # *deleted* (absent from the current env, so a walk of os.environ alone
    # would never see it) is re-added.
    for key in set(saved) | {key for key in os.environ if key.startswith(_ORACLE_ENV_PREFIX)}:
        if key in saved:
            if os.environ.get(key) != saved[key]:
                os.environ[key] = saved[key]
        else:
            os.environ.pop(key, None)
