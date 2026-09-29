"""U4.2 hardening: the shiboken import-hook invariant, swept across the suite.

A module whose FIRST import happens inside a QThread races the GUI's main
thread through shiboken6's signature import hook (``inspect.getsource`` runs
during import) and the pair segfaults natively — reproduced and
faulthandler-verified for mutagen (``tests/test_gui_render_import_safety.py``
holds the narrative and the original pins; see also
``.serpent-circle/04-debug/root-causes.md``).

This file makes the invariant SUITE-WIDE and reusable: the guard lives in
``tests/helpers.py`` (``worker_module_import_violations``) so any future
worker file or worker-reachable module is checked automatically, and the
one-hop target list is explicit so a new worker call-graph edge must be
added here deliberately.

**Mutation contract:** deleting the mutagen preload in
``src/the_oracle/audio/export_flac.py`` and reverting its lazy-import hoist
must fail ``test_no_worker_thread_first_imports_suite_wide``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests import helpers  # noqa: E402

SRC = REPO_ROOT / "src" / "the_oracle"

#: Modules a worker's run() call graph can reach one hop out. Every module
#: here must be fully module-scoped. When a worker gains a new call edge,
#: add the target here — deliberately, with the preload it needs.
ONE_HOP_TARGETS = (
    "audio/export_flac.py",
    "audio/export_srt.py",
    "audio/recorder.py",
    "pipeline.py",
    "subtitle_targets.py",
    "vulkan_setup.py",
    "tts_engines/vulkan_backend.py",
)


def test_no_worker_thread_first_imports_suite_wide() -> None:
    """Every QThread file and every one-hop target must be module-scoped."""
    violations = helpers.worker_module_import_violations(SRC, one_hop_targets=ONE_HOP_TARGETS)
    assert not violations, (
        "worker-thread first-import hazards (shiboken import hook): "
        + "; ".join(violations)
    )


def test_qthread_inventory_is_pinned() -> None:
    """The sweep must see exactly the QThread subclasses that exist.

    If this fails, a new QThread subclass appeared (or one moved) and the
    suite-wide rule above silently started covering it — or stopped covering
    a removed one. Update this tuple deliberately in the same commit.
    """
    inventory = tuple(sorted(f"{rel}::{name}" for rel, name in helpers.iter_qthread_classes(SRC)))
    assert inventory == (
        "app_gui.py::PrewarmThread",
        "gui_recording.py::RecordStudioWorker",
        "gui_render.py::PreviewWorker",
        "gui_render.py::RenderWorker",
        "gui_vulkan.py::ModelDownloadThread",
        "gui_vulkan.py::VulkanDeviceProbeThread",
        "gui_vulkan.py::VulkanPreflightThread",
        "gui_vulkan.py::VulkanSetupThread",
    )
