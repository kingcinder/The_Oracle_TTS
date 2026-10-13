"""Ownership pin for widget-level stylesheet restyles (the construction-storm
audit, 2026-10-10).

``QWidget.setStyleSheet`` re-polishes its subtree synchronously, streaming
PaletteChange/FontChange/StyleChange events that every app-wide event filter
is invoked for — the same storm class ``gui_themes.bulk_restyle`` gates for
``QApplication.setStyleSheet``. The audit measured the widget-construction
sites on a themed app with the real Ctrl+hover filter installed: one change
event and ~0.06–0.17 ms per call (``gui_sections``'s collapse toggle,
``gui_cast``'s three panel dividers and the cast row's key label), so the
gate is an invariant rather than a speedup — no app-wide filter is ever
re-entered by a widget-level restyle, and a future heavier per-widget
stylesheet inherits the gate instead of silently taxing every filter.

Two nets keep it:

1. an AST scan (the ``test_qt_timer_ownership`` shape): every
   ``setStyleSheet`` call in the four widget modules that own construction-
   time stylesheet sites — ``gui_sections.py`` / ``gui_cast.py`` (gated
   2026-10-10) and ``gui_chrome.py`` / ``gui_widgets.py`` (added 2026-10-12)
   — must sit inside a ``with bulk_restyle():`` block, with a vacuity floor
   so the scan cannot go blind, and a synthetic self-check so a broken rule
   fails loudly;
2. a runtime pin that constructs a real eager ``SpeakerGroup`` + ``_CastRow``
   (the sections/cast sites and — through the panel's seven captioned
   ``PerceptualSlider``s — the gui_widgets site) and a ``LivePanel`` (the
   gui_chrome header), and observes, from inside a traced
   ``setStyleSheet``, that every call found the gate OPEN.

The runtime half also runs per theme from the chrome aspect of
``scripts/certify_gui.py`` (its own construction-time trace over six real
themed windows, unit-pinned by tests/test_certify_gui_gate_trace.py), so the
invariant is re-proven on the windows a certification actually builds — not
only in this net's isolated constructions.

The two late additions were the standing "noticed, not actioned" note on the
2026-10-10 entry: ``gui_chrome``'s Live header and ``gui_widgets``'
PerceptualSlider caption — one call per slider, seven per speaker panel, the
largest construction-time share — carried the same ungated pattern.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "the_oracle"

#: Modules whose widget-level restyles the gate covers.
GATED_MODULES = ("gui_sections.py", "gui_cast.py", "gui_chrome.py", "gui_widgets.py")


def _styleset_calls(root: Path) -> list[tuple[str, int, ast.Call]]:
    """Every ``*.setStyleSheet(...)`` call in the gated modules."""
    found: list[tuple[str, int, ast.Call]] = []
    for name in GATED_MODULES:
        path = root / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "setStyleSheet":
                found.append((name, node.lineno, node))
    return found


def _is_gated(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> bool:
    """True when the call sits inside a ``with bulk_restyle():`` block.

    The walk stops at a nested def/lambda/class boundary: a ``with`` outside
    a function does not cover that function's body at runtime, and crediting
    it would let a factory defined inside a gated builder ship an ungated
    call.
    """
    node: ast.AST | None = call
    while node is not None:
        if isinstance(node, ast.With):
            for item in node.items:
                expr = item.context_expr
                if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
                    if expr.func.id == "bulk_restyle":
                        return True
        if isinstance(node, (ast.FunctionDef, ast.Lambda, ast.ClassDef)) and node is not call:
            return False
        node = parents.get(node)
    return False


def _offenders(root: Path) -> list[str]:
    bad: list[str] = []
    for name in GATED_MODULES:
        path = root / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "setStyleSheet"):
                continue
            if not _is_gated(node, parents):
                bad.append(f"{name}:{node.lineno}: setStyleSheet outside bulk_restyle()")
    return bad


def test_every_widget_level_styleset_in_the_gated_modules_is_gated() -> None:
    offenders = _offenders(REPO_SRC)
    assert not offenders, (
        "widget-level setStyleSheet calls outside a bulk_restyle() scope — "
        "wrap the call in `with bulk_restyle():` (gui_themes) so app-wide "
        "event filters are never re-entered by a widget-level restyle:\n"
        + "\n".join(offenders)
    )


def test_the_scan_sees_the_real_sites() -> None:
    """Vacuity: the net must actually see the five known call sites, so a
    rename or a moved file cannot silently blind it."""
    calls = _styleset_calls(REPO_SRC)
    assert len(calls) >= 5, (
        f"only {len(calls)} setStyleSheet call(s) found in {GATED_MODULES}; "
        "the known floor is 5 (the collapse toggle, the panel-divider "
        "factory, the cast row key label, the Live header, the slider "
        "caption). The scan went blind — fix the scan."
    )


def test_the_scanner_catches_an_ungated_call() -> None:
    """Self-check of the rule (the timer-ownership net's shape): a bare call
    fails, a gated one passes, and a ``with`` OUTSIDE a nested def does not
    cover its body."""
    import tempfile

    src = (
        "from the_oracle.gui_themes import bulk_restyle\n"
        "def ungated(w):\n"
        "    w.setStyleSheet('color: red')\n"
        "def gated(w):\n"
        "    with bulk_restyle():\n"
        "        w.setStyleSheet('color: red')\n"
        "def false_cover(w):\n"
        "    def inner():\n"
        "        w.setStyleSheet('color: red')\n"
        "    with bulk_restyle():\n"
        "        pass\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "gui_sections.py").write_text(src, encoding="utf-8")
        for sibling in ("gui_cast.py", "gui_chrome.py", "gui_widgets.py"):
            (root / sibling).write_text("", encoding="utf-8")
        bad = _offenders(root)
    assert len(bad) == 2, bad
    assert any(":3:" in entry for entry in bad), bad
    # false_cover's inner call is at line 9: the `with` on line 10 comes
    # AFTER the def, so it must not be credited with covering inner()'s body.
    assert any(":9:" in entry for entry in bad), bad


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_the_construction_calls_find_the_gate_open_at_runtime(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Runtime half: construct a real eager SpeakerGroup (which fires the
    collapse toggle, the three panel dividers, the cast row's key label and
    the panel's seven captioned PerceptualSlider captions) plus a LivePanel
    (the gui_chrome header) while tracing QWidget.setStyleSheet, and observe
    the real gate from inside every call."""
    from the_oracle import gui_themes
    from the_oracle.gui_cast import SpeakerGroup, _CastRow
    from the_oracle.gui_chrome import LivePanel
    from PySide6.QtWidgets import QWidget

    import inspect

    original = QWidget.setStyleSheet
    # (gate open, caller module) per traced call; non-gated modules map to None.
    seen: list[tuple[bool, str | None]] = []

    def traced(self, style: str) -> None:
        caller = inspect.currentframe().f_back.f_code.co_filename
        name = Path(caller).name
        seen.append(
            (gui_themes.in_bulk_restyle(), name if name in GATED_MODULES else None)
        )
        original(self, style)

    QWidget.setStyleSheet = traced  # type: ignore[method-assign]
    try:
        group = SpeakerGroup("A", tmp_path)
        row = _CastRow("A", group)
        panel = LivePanel()
    finally:
        QWidget.setStyleSheet = original  # type: ignore[method-assign]
    try:
        per_module: dict[str, int] = {}
        closed: list[str] = []
        for gate_open, name in seen:
            if name is None:
                continue
            per_module[name] = per_module.get(name, 0) + 1
            if not gate_open:
                closed.append(name)
        # Non-vacuity, per module: every gated construction site must have
        # fired, so a rename or a lazily-skipped build cannot pass silently.
        # Known floors (measured 2026-10-12): sections = the collapse toggle;
        # cast = 3 panel dividers + the row key label; widgets = the speaker
        # panel's 7 captioned sliders; chrome = the Live header.
        floors = {
            "gui_sections.py": 1,
            "gui_cast.py": 4,
            "gui_widgets.py": 7,
            "gui_chrome.py": 1,
        }
        for module, floor in floors.items():
            fired = per_module.get(module, 0)
            assert fired >= floor, (
                f"only {fired} in-scope setStyleSheet call(s) traced from "
                f"{module} during an eager SpeakerGroup + _CastRow + "
                f"LivePanel construction; the known floor is {floor}. The "
                "trace went blind — fix the trace."
            )
        assert not closed, (
            "a construction-time setStyleSheet in the gated modules ran with "
            "the bulk-restyle gate CLOSED (" + ", ".join(sorted(set(closed))) +
            ") — an app-wide filter would be re-entered for its change "
            "events; wrap the call in `with bulk_restyle():`"
        )
    finally:
        row.container.deleteLater()
        group.deleteLater()
        panel.deleteLater()
        qt_app.processEvents()
