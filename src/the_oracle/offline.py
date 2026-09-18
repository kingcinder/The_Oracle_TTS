"""Offline-install enforcement: resolve models from the seeded cache, never the network.

An offline install writes ``.oracle_offline`` in the repo root and seeds the
pinned models into the local Hugging Face cache. The marker is what makes the
offline promise true, and it has to be honoured by *every* process that resolves
a model — not only by the generated launchers.

Why this module exists rather than an inline check in each entry point:
``huggingface_hub`` reads ``HF_HUB_OFFLINE`` **once, at import time** (it sets
``constants.HF_HUB_OFFLINE`` from the environment during its own import), so a
process that imports it before the variable is set silently keeps network
resolution on. Measured on this machine, resolving an *unseeded* model with a
seeded cache present and the network unreachable:

======================  ===========  ===========
when the env is set     constant     time to fail
======================  ===========  ===========
before the HF import    True         0.0s
after the HF import     False        23.0s (5 retries, 8s backoff)
after + forced constant True         0.0s
======================  ===========  ===========

So :func:`apply_offline_environment` does both: it exports the variables for
anything imported or spawned later, and — when ``huggingface_hub`` is already
imported — forces the import-time constant. Entry points call it before their
first model work; the shell launchers keep exporting the same variables so a
process started from them is already correct at import.

Deleting the marker (documented in ``manage_install``) re-enables network model
fetches; nothing here ever turns offline mode back off within a process.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: Marker file written into the install's repo root by an offline install.
#: ``scripts/manage_install.py`` imports this name rather than defining its own,
#: so the launcher generation and the runtime check can never disagree.
OFFLINE_MARKER_FILENAME = ".oracle_offline"

#: Variables the model libraries read. ``huggingface_hub`` honours
#: ``HF_HUB_OFFLINE`` (and ``TRANSFORMERS_OFFLINE``); ``transformers`` reads
#: both dynamically through ``is_offline_mode()``.
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def repo_root() -> Path:
    """The checkout/install root this module belongs to (``<root>/src/the_oracle``)."""
    return Path(__file__).resolve().parents[2]


def marker_path(root: str | Path | None = None) -> Path:
    """Path of the offline marker for ``root`` (defaults to this install)."""
    return Path(root) / OFFLINE_MARKER_FILENAME if root is not None else repo_root() / OFFLINE_MARKER_FILENAME


def is_offline_install(root: str | Path | None = None) -> bool:
    """Whether the offline marker exists for ``root``."""
    return marker_path(root).is_file()


def apply_offline_environment(root: str | Path | None = None) -> bool:
    """Force offline model resolution when the marker exists.

    Returns whether offline mode is in force (the marker was found). Safe to
    call repeatedly from nested entry points. Never unsets an inherited
    ``HF_HUB_OFFLINE`` when there is no marker: the generated launchers and
    ``manage_install run`` set it, and child processes must not undo that.
    """
    if not is_offline_install(root):
        return False
    for name, value in OFFLINE_ENV.items():
        if os.environ.get(name) != value:
            os.environ[name] = value
    _force_import_time_constant()
    logger.info(
        "Offline install: model resolution is local-only (%s present)",
        OFFLINE_MARKER_FILENAME,
    )
    return True


def _force_import_time_constant() -> None:
    """Force ``huggingface_hub``'s import-time offline constant, if imported.

    Importing the package here would defeat the purpose (a heavy import entry
    points do not otherwise need before their own lazy imports), so this only
    corrects a process where it is already in play. The ``constants`` submodule
    is looked up in ``sys.modules`` rather than through the package: the package
    attribute is lazy and raises for parts of the package that have not been
    imported yet, which would silently leave the constant alone.
    """
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is None or getattr(constants, "HF_HUB_OFFLINE", False):
        return
    constants.HF_HUB_OFFLINE = True
