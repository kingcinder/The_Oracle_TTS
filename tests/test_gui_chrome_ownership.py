"""Ownership tests for the gui_chrome slice — the Live sidebar chrome.

``LivePanel`` and the Live column's section chrome moved from ``app_gui`` to
``the_oracle.gui_chrome`` (2026-09-21 extraction slice). The safety nets each
slice relies on (``test_app_gui_patch_surface``, ``test_payload_policy_ownership``)
ran green as this slice's pre-flight gates; these tests pin the slice itself.

Three contracts keep the move honest:

1. ONE OWNER — ``gui_chrome`` defines the Live chrome and no other module
   defines a competing ``LivePanel``.
2. RE-EXPORT — ``app_gui.LivePanel`` IS ``gui_chrome.LivePanel``, so
   MainWindow's construction, the two progress handlers that drive it, and the
   existing ``from the_oracle.app_gui import LivePanel`` test import keep
   resolving exactly as before.
3. ASSEMBLY — MainWindow's Live column comes from ``gui_chrome``'s
   ``build_live_section`` (the collapsible/resizable section chrome), while the
   splitter and ``_register_section`` persistence wiring stay in ``app_gui``.
   That is the split-readership the patch-surface net registers as
   ``PARTIAL_OWNED``.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
APP_GUI = SRC / "the_oracle" / "app_gui.py"
GUI_CHROME = SRC / "the_oracle" / "gui_chrome.py"

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def _calls(node: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _callee_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_gui_chrome_is_the_single_owner_of_the_live_panel() -> None:
    """gui_chrome defines the Live chrome; no other module competes."""
    tree = ast.parse(GUI_CHROME.read_text(encoding="utf-8"))
    live_panel = _class(tree, "LivePanel")
    assert live_panel is not None, "gui_chrome must define LivePanel"

    builders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_live_section"
    ]
    assert len(builders) == 1, "gui_chrome must define exactly one build_live_section"

    # No second definition anywhere in the package (a leftover copy in app_gui
    # would mean two live definitions, only one of which the window uses).
    others = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if path != GUI_CHROME
        and _class(ast.parse(path.read_text(encoding="utf-8")), "LivePanel") is not None
    ]
    assert others == [], f"LivePanel defined outside gui_chrome: {others}"


def test_app_gui_no_longer_defines_the_live_panel() -> None:
    """The moved class is gone from app_gui — it must be re-exported, not copied."""
    tree = ast.parse(APP_GUI.read_text(encoding="utf-8"))
    assert _class(tree, "LivePanel") is None, (
        "app_gui still defines LivePanel; the slice must leave exactly one "
        "definition (in gui_chrome) and re-export it."
    )


def test_app_gui_reexports_the_gui_chrome_names() -> None:
    """The re-export contract the patch surface and tests depend on.

    ``tests/test_app_gui_progress.py`` imports ``LivePanel`` from app_gui, and
    MainWindow constructs it from app_gui globals — both require the identical
    object, not a lookalike.
    """
    import the_oracle.app_gui as app_gui
    import the_oracle.gui_chrome as gui_chrome

    assert app_gui.LivePanel is gui_chrome.LivePanel
    assert app_gui.build_live_section is gui_chrome.build_live_section


def test_main_window_builds_the_live_column_from_gui_chrome() -> None:
    """Assembly: the Live section comes from gui_chrome; the splitter wiring
    and section registration stay in app_gui (where the layout state lives)."""
    tree = ast.parse(APP_GUI.read_text(encoding="utf-8"))
    main_window = _class(tree, "MainWindow")
    assert main_window is not None
    build_ui = _method(main_window, "_build_ui")
    assert build_ui is not None

    builder_calls = [c for c in _calls(build_ui) if _callee_name(c) == "build_live_section"]
    assert len(builder_calls) == 1, "MainWindow._build_ui must call build_live_section once"
    # ...and hand it the panel it owns.
    passed = builder_calls[0].args
    assert len(passed) == 1
    assert isinstance(passed[0], ast.Attribute) and passed[0].attr == "live_panel", (
        "build_live_section must receive MainWindow's self.live_panel"
    )

    # The Live section is no longer constructed in app_gui from app_gui's
    # QHSectionGroup — that is exactly the PARTIAL_OWNED split.
    live_sections_here = [
        c
        for c in _calls(build_ui)
        if _callee_name(c) == "QHSectionGroup"
        and c.args
        and isinstance(c.args[0], ast.Constant)
        and c.args[0].value == "Live"
    ]
    assert live_sections_here == [], (
        "app_gui still builds the Live QHSectionGroup; it belongs to "
        "gui_chrome.build_live_section now."
    )

    # Wiring app_gui still owns: the section is registered for layout
    # persistence and added to the main splitter.
    registered = [
        c
        for c in _calls(build_ui)
        if _callee_name(c) == "addWidget"
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "addWidget"
        and isinstance(c.func.value, ast.Attribute)
        and c.func.value.attr == "_main_splitter"
    ]
    assert len(registered) == 2, "the main splitter must still receive both arms"
    register_calls = [
        c
        for c in _calls(build_ui)
        if _callee_name(c) == "_register_section"
        and c.args
        and isinstance(c.args[0], ast.Constant)
        and c.args[0].value == "live"
    ]
    assert len(register_calls) == 1, "the 'live' section must stay registered for persistence"


@pytest.mark.slow
def test_build_live_section_wraps_the_panel_in_section_chrome() -> None:
    """The chrome itself: a collapsible, resizable 'Live' section holding the
    panel, with the Live column's narrow floor."""
    from PySide6.QtWidgets import QApplication

    from the_oracle.gui_chrome import LivePanel, build_live_section
    from the_oracle.gui_sections import QHSectionGroup

    QApplication.instance() or QApplication([])

    panel = LivePanel()
    section = build_live_section(panel)

    assert isinstance(section, QHSectionGroup)
    assert section.title() == "Live"
    assert section.has_size_slider(), "the section must be resizable"
    assert not section.is_collapsed()
    assert section.minimumWidth() == 220
    assert panel.parent() is section, "the panel must be reparented into the section"

    wider = build_live_section(LivePanel(), minimum_width=300)
    assert wider.minimumWidth() == 300


@pytest.mark.slow
def test_live_column_updates_through_the_wrapped_panel() -> None:
    """Driving the panel through its section chrome behaves identically: the
    progress mirror still updates and still resets."""
    from PySide6.QtWidgets import QApplication

    from the_oracle.gui_chrome import LivePanel, build_live_section
    from the_oracle.pipeline import RenderProgress

    QApplication.instance() or QApplication([])

    panel = LivePanel()
    # The section owns the panel (addWidget reparents), so the reference must
    # outlive the test body — dropping it would let Qt delete the whole
    # subtree, panel and progress bar included. MainWindow keeps the same
    # ownership via its splitter.
    section = build_live_section(panel)
    assert section is not None

    panel.update_from_progress(
        RenderProgress(
            stage="Synthesizing",
            detail="utterance 3/10",
            current_step=3,
            total_steps=10,
            current_segment=3,
            total_segments=10,
            elapsed_seconds=12.5,
            eta_seconds=8.0,
            fraction=0.3,
            backend="vulkan",
            device_label="AMD RX 5700 XT",
            synth_seconds_total=9.2,
            synth_seconds_latest=1.5,
        )
    )
    assert panel.progress_bar.value() == 30
    assert "Vulkan" in panel.backend_label.text()
    assert "AMD RX 5700 XT" in panel.backend_label.text()
    assert "Synthesizing" in panel.stage_label.text()
    assert "3/10" in panel.segment_label.text()

    panel.set_idle()
    assert panel.progress_bar.value() == 0
    assert panel.backend_label.text() == "Backend: idle"
    assert panel.stage_label.text() == ""
