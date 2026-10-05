"""Import-direction guard: ``gui_render`` never looks up.

The MainWindow extraction campaign splits ``app_gui`` into ``gui_*`` owner
modules (gui_ingest, gui_settings, gui_render, gui_vulkan, gui_chrome, ...).
The dependency direction is one-way::

    app_gui  ->  gui_*  ->  pipeline / models / utils

``app_gui`` imports FROM the gui layer (that direction is legal and expected);
the gui layer must never import ``app_gui`` back, and ``gui_render`` — the
bottom of the gui stack, already depended on by ``gui_chrome`` — must never
import a ``gui_*`` sibling. A sibling import would put a cycle under the next
extraction slice's feet, and the failure would surface far from its cause.

Contracts:

1. ONE-WAY — ``gui_render`` imports nothing rooted at ``app_gui`` or ``gui_``.
   The pattern covers every current and future ``gui_*`` module (the campaign
   grows) and ``gui_render`` itself: a self-import is never needed. If a slice
   ever legitimately needs shared gui-layer code from ``gui_render``, that code
   belongs below the gui layer (``utils`` and friends) or the rule needs an
   explicit, reviewed exception — this guard failing is the forcing function.
2. FORMS — the rule holds for every import spelling: ``import x``,
   ``from x import y``, ``from the_oracle import x``, relative
   ``from . import x`` / ``from .x import y``, alias and submodule forms.
   A scan that only understands one spelling is a scan that goes blind the day
   someone writes another.
3. NO SNEAKY LOADER — ``importlib.import_module(...)`` / ``__import__(...)``
   of a forbidden module is the same violation, and a dynamic import whose
   target AST cannot resolve (non-literal argument, star import) fails loudly
   as unreviewed rather than passing unseen.
4. VACUITY — the scanner must demonstrably see ``gui_render``'s real imports
   and the real ``gui_*`` sibling set, and must demonstrably flag each
   forbidden form (synthetic one-form-per-case proofs run in every CI pass).
   A green scan over an empty parse is the failure mode this file exists to
   prevent.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "src" / "the_oracle"
GUI_RENDER = PACKAGE_DIR / "gui_render.py"

# Known gui-layer modules the rule must see; a future slice adding another
# gui_* file is covered by the name pattern without touching this file.
KNOWN_SIBLINGS = ("gui_cast", "gui_chrome", "gui_recording", "gui_settings", "gui_vulkan")


def _gui_siblings() -> tuple[str, ...]:
    """Every ``gui_*`` module in the package except ``gui_render`` itself."""
    return tuple(
        sorted(
            path.stem
            for path in PACKAGE_DIR.glob("gui_*.py")
            if path.stem != "gui_render"
        )
    )


def _module_root(module: str) -> str | None:
    """Root name of a module path, relative to the ``the_oracle`` package.

    ``the_oracle.app_gui`` -> ``app_gui``; ``the_oracle.gui_settings.x`` ->
    ``gui_settings``; ``app_gui`` -> ``app_gui``; ``the_oracle`` alone -> None.
    """
    if module == "the_oracle" or module.startswith("the_oracle."):
        rest = module[len("the_oracle") :].lstrip(".")
    else:
        rest = module
    return rest.split(".", 1)[0] if rest else None


def _is_forbidden(module: str) -> bool:
    root = _module_root(module)
    return root is not None and (root == "app_gui" or root.startswith("gui_"))


@dataclass(frozen=True)
class ImportScan:
    """Everything one module's source imports, and what the rule flags."""

    modules: tuple[str, ...]  # imported-from module paths, normalized
    violations: tuple[str, ...]  # forbidden imports: app_gui / gui_*
    unresolved: tuple[str, ...]  # imports AST cannot verify — fail loudly


def _candidates(node: ast.AST) -> list[str]:
    """Normalize one import statement to full dotted module candidates.

    ``from the_oracle import app_gui`` yields ``the_oracle.app_gui`` alongside
    its base; ``from .gui_settings import X`` resolves relative to
    ``the_oracle``; ``from . import x`` promotes the imported name. Alias and
    submodule spellings land on the same candidate as their plain form, so one
    forbidden check covers them all.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level == 0:
            base = node.module or ""
        elif node.level == 1:
            # gui_render lives at the_oracle/gui_render.py: '.' is the package.
            base = "the_oracle" + (f".{node.module}" if node.module else "")
        else:
            # '..' and beyond leave the package — treat as top-level names.
            base = node.module or ""
        candidates = [base] if base else []
        for alias in node.names:
            candidates.append(f"{base}.{alias.name}" if base else alias.name)
        return candidates
    return []


def _dynamic_import_target(call: ast.Call) -> tuple[str, bool] | None:
    """Match ``importlib.import_module(...)`` / ``__import__(...)`` calls.

    Returns ``(target_module, resolved)`` when the call is a dynamic import;
    ``resolved`` is False when the target is not a literal string and the
    checker cannot see what will be imported.
    """
    func = call.func
    if isinstance(func, ast.Name):
        if func.id not in ("__import__", "import_module"):
            return None
    elif isinstance(func, ast.Attribute):
        if func.attr not in ("__import__", "import_module"):
            return None
    else:
        return None
    if not call.args:
        return None
    head = call.args[0]
    if isinstance(head, ast.Constant) and isinstance(head.value, str):
        return (head.value, True)
    return ("<non-literal>", False)


def _scan_imports(
    source: str, *, forbidden: Callable[[str], bool] | None = None
) -> ImportScan:
    """Scan one module's source for imports; ``forbidden`` overrides the
    default app_gui/gui_* rule for reuse by other layer pins (e.g.
    gui_vulkan, where gui_utils is a legal sibling but app_gui is not).
    The predicate receives the normalized module path — compare roots with
    :func:`_module_root`.
    """
    _is_forbidden_module = forbidden or _is_forbidden
    tree = ast.parse(source)
    modules: list[str] = []
    violations: list[str] = []
    unresolved: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            segment = ast.get_source_segment(source, node) or "<import>"
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "*" for alias in node.names
            ):
                unresolved.append(
                    f"line {node.lineno}: star import `{segment}` — the "
                    "checker cannot see what it pulls in"
                )
                continue
            for candidate in _candidates(node):
                modules.append(candidate)
                if _is_forbidden_module(candidate):
                    violations.append(
                        f"line {node.lineno}: `{segment}` imports {candidate}"
                    )
        elif isinstance(node, ast.Call):
            target = _dynamic_import_target(node)
            if target is None:
                continue
            module, resolved = target
            segment = ast.get_source_segment(source, node) or "<call>"
            if not resolved:
                unresolved.append(
                    f"line {node.lineno}: dynamic import `{segment}` — the "
                    "checker cannot see what it pulls in"
                )
            else:
                modules.append(module)
                if _is_forbidden_module(module):
                    violations.append(
                        f"line {node.lineno}: `{segment}` imports {module}"
                    )

    return ImportScan(
        modules=tuple(sorted(set(modules))),
        violations=tuple(violations),
        unresolved=tuple(unresolved),
    )


def _render_scan() -> ImportScan:
    return _scan_imports(GUI_RENDER.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The guard itself.
# ---------------------------------------------------------------------------


def test_gui_render_imports_nothing_from_app_gui_or_a_gui_sibling() -> None:
    """gui_render looks up: no app_gui, no gui_* sibling, no unseen loader."""
    scan = _render_scan()
    assert scan.violations == () and scan.unresolved == (), (
        "gui_render must stay at the bottom of the gui stack "
        "(app_gui -> gui_* -> pipeline/models); forbidden import(s):\n  "
        + "\n  ".join(scan.violations + scan.unresolved)
        + "\n  (gui_* modules covered today: "
        + ", ".join(("gui_render",) + _gui_siblings())
        + ")"
    )


# ---------------------------------------------------------------------------
# Vacuity guards: a scan that cannot see is worse than no scan.
# ---------------------------------------------------------------------------


def test_scan_sees_gui_renders_real_imports() -> None:
    """The scanner must see the imports gui_render actually makes today.

    If parsing goes stale (wrong file, empty read, AST drift) ``modules``
    shrinks or loses the known-good bottom-layer imports — fail loudly rather
    than pass green over an empty parse.
    """
    scan = _render_scan()
    assert {"the_oracle.pipeline", "the_oracle.models.project"} <= set(
        scan.modules
    ), f"scan lost track of gui_render's real imports: {scan.modules}"
    assert len(scan.modules) >= 10, (
        f"scan found only {len(scan.modules)} imported modules in gui_render; "
        f"the real file imports far more — the scan went blind: {scan.modules}"
    )


def test_gui_sibling_set_is_visible() -> None:
    """The gui_* names the rule protects must be the real, live modules.

    The forbidden pattern is name-based so future slices are covered without
    edits here — but if the package's gui_* files ever vanish or move, this
    names the loss instead of letting the guard silently protect nothing.
    """
    siblings = _gui_siblings()
    assert set(KNOWN_SIBLINGS) <= set(siblings), (
        f"the gui_* sibling set no longer contains {KNOWN_SIBLINGS}: "
        f"{siblings} — the extraction campaign's layout changed; update "
        "this guard deliberately."
    )
    assert len(siblings) >= 10, (
        f"only {len(siblings)} gui_* siblings found: {siblings} — the known "
        "set is 10. The listing went blind — fix the listing, not this assertion."
    )


# ---------------------------------------------------------------------------
# Form-by-form proofs: each forbidden spelling must be flagged, each allowed
# spelling must pass. These run in every CI pass, so the scanner can never
# quietly lose a form.
# ---------------------------------------------------------------------------

# (source, the forbidden module the violation must name). Spelling each form
# and its classified target out loud is what keeps the scanner honest: a flag
# that fires for the wrong reason still passes a bare `violations != ()`.
_FORBIDDEN_FORMS = (
    ("import the_oracle.app_gui", "the_oracle.app_gui"),
    ("import the_oracle.gui_chrome", "the_oracle.gui_chrome"),
    ("import app_gui", "app_gui"),
    ("from the_oracle.app_gui import MainWindow", "the_oracle.app_gui"),
    ("from the_oracle import app_gui", "the_oracle.app_gui"),
    ("from the_oracle import gui_vulkan as gv", "the_oracle.gui_vulkan"),
    ("from the_oracle.gui_settings import SettingsDialog", "the_oracle.gui_settings"),
    ("from . import gui_settings", "the_oracle.gui_settings"),
    ("from .gui_ingest import IngestPanel", "the_oracle.gui_ingest"),
    ("from .gui_sections import build_section", "the_oracle.gui_sections"),
    ("from .gui_recording import RecordStudioWorker", "the_oracle.gui_recording"),
    ("from app_gui import MainWindow", "app_gui"),
    ("from gui_widgets import Floater", "gui_widgets"),
    ("importlib.import_module('the_oracle.app_gui')", "the_oracle.app_gui"),
    ("importlib.import_module('the_oracle.gui_themes')", "the_oracle.gui_themes"),
    ("__import__('the_oracle.gui_tooltips')", "the_oracle.gui_tooltips"),
)

_ALLOWED_FORMS = (
    "from the_oracle.pipeline import OraclePipeline, RenderProgress",
    "from the_oracle.models.project import RenderPlan, Utterance, VoiceProfile",
    "from the_oracle.models.settings import RenderSettings",
    "from the_oracle.audio.export_srt import write_srt",
    "from the_oracle.subtitle_targets import subtitle_sidecar_target",
    "from the_oracle.utils.audio import record_synthesis_retry",
    "from the_oracle import pipeline",
    "from the_oracle import models",
    "from copy import deepcopy",
    "from dataclasses import asdict",
    "import json, os, signal, subprocess, sys, tempfile",
    "from PySide6.QtCore import QThread, Signal",
    "import PySide6.QtWidgets",
    "import the_oracle.pipeline",
)


def test_every_forbidden_import_form_is_flagged() -> None:
    """Each spelling of 'gui_render imports upward' must trip the guard."""
    for source, expected in _FORBIDDEN_FORMS:
        scan = _scan_imports(source)
        assert scan.violations, f"form not flagged: `{source}`"
        assert any(f"imports {expected}" in v for v in scan.violations), (
            f"form `{source}` flagged, but not as {expected}: {scan.violations}"
        )
        assert scan.unresolved == (), (
            f"form `{source}` was misclassified as unresolved, not forbidden: "
            f"{scan.unresolved}"
        )


def test_every_allowed_import_form_passes() -> None:
    """Legitimate bottom-layer and stdlib imports must not trip the guard."""
    for source in _ALLOWED_FORMS:
        scan = _scan_imports(source)
        assert scan.violations == () and scan.unresolved == (), (
            f"allowed form `{source}` was flagged: "
            f"{scan.violations + scan.unresolved}"
        )


def test_unresolvable_imports_fail_loudly() -> None:
    """A dynamic import or star import the checker cannot see must not pass."""
    blind_forms = (
        "name = 'the_oracle.app_gui'\nimportlib.import_module(name)",
        "mod = __import__('the_oracle.' + name)",
        "from the_oracle import *",
        "from . import *",
    )
    for source in blind_forms:
        scan = _scan_imports(source)
        assert scan.unresolved, f"blind form passed unseen: `{source}`"
        assert scan.violations == (), (
            f"blind form `{source}` was misclassified as forbidden, not "
            f"unresolved: {scan.violations}"
        )
