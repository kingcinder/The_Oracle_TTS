"""U4.2 regression net: worker threads must never perform first-imports.

A module whose FIRST import happens inside a QThread races the GUI's main
thread through shiboken6's signature import hook (``inspect.getsource`` runs
during import), and the pair segfaults natively. This is the 2026-09-28
repro: ``mutagen.flac`` first imported in the render worker's thread while
the main thread ran Qt — caught by the U1 faulthandler net
(``crash_reports/native-crash.txt``), same class as the 2026-09-01 crash
recorded in .serpent-circle/04-debug/root-causes.md.

The fix is structural: everything the workers' ``run()`` call graph needs is
imported in the main thread before a worker can start. These tests pin that
shape from three angles so a regression cannot slip in quietly.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Modules a worker's run() call graph lazily pulled before the fix. The
#: first import of any of these must happen on the main thread, before
#: gui_render is even usable — never inside a QThread.
WORKER_REACHABLE_LAZY_IMPORTS = (
    "mutagen",
    "mutagen.flac",
    "tempfile",
    "the_oracle.audio.export_srt",
    "the_oracle.subtitle_targets",
)


def _function_level_imports(path: Path) -> list[str]:
    """Return ``module`` names imported inside function bodies in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []

    def visit(node, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_in_function = in_function or isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            if child_in_function and isinstance(child, ast.Import):
                found.extend(alias.name for alias in child.names)
            elif child_in_function and isinstance(child, ast.ImportFrom):
                if child.module:
                    found.append(child.module)
            visit(child, child_in_function)

    visit(tree, False)
    return found


def test_importing_gui_render_preloads_worker_reachable_modules() -> None:
    """Importing the workers' home module must itself load the lazy set.

    The preload runs at import time on whatever thread imports gui_render —
    which, before any window exists, is the main thread. If this assertion
    fails, a worker can be the first importer again.
    """
    import the_oracle.gui_render  # noqa: F401 - the import IS the assertion

    missing = [name for name in WORKER_REACHABLE_LAZY_IMPORTS if name not in sys.modules]
    assert not missing, f"not preloaded by the_oracle.gui_render import: {missing}"


def test_background_thread_performs_no_first_imports() -> None:
    """A worker-shaped background thread must add nothing to sys.modules.

    Mirrors the crash geometry: a QThread run() body executing while the main
    thread is live. After the fix there is nothing left to import lazily, so
    the module set must not grow.
    """
    import threading

    import the_oracle.gui_render  # noqa: F401 - ensure preload already ran

    before = frozenset(sys.modules)
    added: list[str] = []
    done = threading.Event()

    def worker_body() -> None:
        # The exact call-graph shape RenderWorker.run() exercises after the
        # fix: srt export path, subtitle target naming, flac tagging.
        from the_oracle.audio.export_srt import write_srt  # noqa: F401
        from the_oracle.subtitle_targets import subtitle_sidecar_target  # noqa: F401

        from mutagen.flac import FLAC  # noqa: F401

        done.set()

    thread = threading.Thread(target=worker_body, daemon=True)
    thread.start()
    assert done.wait(timeout=30.0), "worker body did not finish"
    thread.join(timeout=10.0)

    added = [name for name in sys.modules if name not in before]
    assert not added, f"background thread performed first-imports: {added}"


def test_no_function_level_imports_in_worker_call_graph_modules() -> None:
    """gui_render and pipeline must import everything at module level.

    Static pin: any function-level ``import`` in the workers' home module or
    the pipeline they call can become a worker-thread first-import the moment
    a code path reaches it first. The two modules were cleaned for U4.2; this
    keeps them clean.
    """
    offenders: dict[str, list[str]] = {}
    for rel in ("src/the_oracle/gui_render.py", "src/the_oracle/pipeline.py"):
        lazy = _function_level_imports(REPO_ROOT / rel)
        if lazy:
            offenders[rel] = lazy
    assert not offenders, f"function-level imports reintroduced: {offenders}"
