"""Section chrome for the previously bare areas (Phase 3 of the legibility
pass): Script & Cast (path rows + cast bar), Review (the table), and Extra
Voices must be QHSectionGroup sections with registry entries and working
persistence, per the approved design.
"""

import os
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from the_oracle.gui_sections import QHSectionGroup, _COLLAPSED_HEADER_HEIGHT
from tests.test_app_gui_profiles import _build_window


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_review_table_is_collapsible_resizable_section(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert "table" in window._section_registry
        section, splitter, index = window._section_registry["table"]
        assert splitter is window._lower_splitter
        assert index == 0
        assert isinstance(section, QHSectionGroup)
        # The table lives inside the section.
        assert window.table.parentWidget() is section
        assert section.is_collapsed() is False
        section.set_collapsed(True)
        assert section.is_collapsed() is True
        # Collapse clips the section to its header (mechanism proven by the
        # Status/Errors section).
        assert section.maximumHeight() == _COLLAPSED_HEADER_HEIGHT
        section.set_collapsed(False)
        assert section.is_collapsed() is False
        assert section.maximumHeight() > _COLLAPSED_HEADER_HEIGHT
        # Status/Errors keeps its slot and index.
        assert "status" in window._section_registry
        _status, status_splitter, status_index = window._section_registry["status"]
        assert status_splitter is window._lower_splitter
        assert status_index == 1
    finally:
        window.close()


def test_paths_section_wraps_path_rows_and_cast_bar(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert "paths" in window._section_registry
        section, splitter, index = window._section_registry["paths"]
        assert splitter is window._paths_splitter
        assert index == 0
        assert isinstance(section, QHSectionGroup)
        assert "Script & Cast" in section.title()
        # Path fields and the cast bar live inside the section.
        assert window.input_path.parentWidget() is section
        assert window.outdir_path.parentWidget() is section
        assert window.output_name.parentWidget() is section
        assert window.manage_cast_button.parentWidget() is section
        # The Analyze/Render row stays bare: it is outside the section.
        assert window.analyze_button.parentWidget() is not section
        assert section.is_collapsed() is False
        section.set_collapsed(True)
        assert section.maximumHeight() == _COLLAPSED_HEADER_HEIGHT
        section.set_collapsed(False)
    finally:
        window.close()


def test_extra_voices_section_registered_and_hidden_for_two_speakers(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert "extra_voices" in window._section_registry
        section, splitter, index = window._section_registry["extra_voices"]
        assert splitter is window._sections_splitter
        assert index == 3
        assert isinstance(section, QHSectionGroup)
        assert window.extra_speaker_scroll.parentWidget() is section
        # A fresh window has only speakers A and B: the column is hidden.
        assert section.isHidden() is True
    finally:
        window.close()


def test_hidden_extra_voices_pane_does_not_break_size_slider(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        shared, _splitter, _index = window._section_registry["shared"]
        assert window._section_registry["extra_voices"][0].isHidden() is True
        # Redistribution must skip the hidden pane rather than misalign or
        # raise (Qt setSizes only honors visible-pane lists).
        shared.set_size_share(70)
        assert shared.size_share() == 70
        sizes = window._sections_splitter.sizes()
        assert len(sizes) == 4
        assert sum(sizes) > 0
        assert sizes[0] > 0
    finally:
        window.close()


def test_section_title_elides_before_header_chrome_at_floor_width(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Regression pin for the title-under-chrome overlap: at the 260 px
    section floor the long "Shared Render Settings" title must be elided
    before it reaches the size slider, and the original text kept for when
    the pane widens again."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window.resize(1160, 760)
        window.show()
        qt_app.processEvents()
        window._sections_splitter.setSizes([260, 260, 389])
        qt_app.processEvents()
        section, _splitter, _index = window._section_registry["shared"]
        assert section.width() <= 270, "expected the pane at its floor"
        assert section._full_title == "Shared Render Settings"
        assert section.title() != section._full_title, (
            "long title was not elided at the section floor — it paints "
            "under the header chrome"
        )
        assert section.title().endswith("…")
        # Widening restores the full text.
        window._sections_splitter.setSizes([460, 360, 360])
        qt_app.processEvents()
        assert section.title() == "Shared Render Settings"
    finally:
        window.close()


def test_paths_collapse_and_splitter_survive_persist_round_trip(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # _persist_workspace_layout no-ops until _on_gui_shown has run (the
    # restore must never be overwritten by widget defaults), so show each
    # window first, exactly like the real app.
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window.show()
        qt_app.processEvents()
        section, _splitter, _index = window._section_registry["paths"]
        assert section.is_collapsed() is False
        window._paths_splitter.setSizes([150, 650])
        window._persist_workspace_layout()
    finally:
        window.close()

    # Fresh window over the same isolated settings: the splitter share is
    # restored while expanded (a later collapse clips the pane to its
    # 30 px header, which is a different state and would overwrite the
    # share, so sizes are asserted before any collapsing).
    window2, _paths2 = _build_window(monkeypatch, tmp_path)
    try:
        window2.show()
        qt_app.processEvents()
        section2, splitter2, index2 = window2._section_registry["paths"]
        assert splitter2 is window2._paths_splitter
        assert index2 == 0
        assert section2.is_collapsed() is False
        sizes = window2._paths_splitter.sizes()
        assert abs(sizes[0] - 150) <= 30, f"paths pane not restored: {sizes}"
        # Registry-driven restore covers the new keys too: the Review and
        # Extra Voices entries flow through _apply_workspace_layout.
        assert "table" in window2._section_registry
        assert "extra_voices" in window2._section_registry

        # Second hop: collapse now persists and restores as collapsed.
        section2.set_collapsed(True)
        window2._persist_workspace_layout()
    finally:
        window2.close()

    window3, _paths3 = _build_window(monkeypatch, tmp_path)
    try:
        window3.show()
        qt_app.processEvents()
        section3, _splitter3, _index3 = window3._section_registry["paths"]
        assert section3.is_collapsed() is True
    finally:
        window3.close()


# --- CI: the six-theme sweep is a named step on every push --------------------


def test_workflow_runs_the_section_chrome_sweep_on_both_oses() -> None:
    """The unit pins above drive a faked window; the sweep certifies the
    same chrome on the REAL MainWindow per theme. Gated only by operating
    system, so it runs on every push and pull_request -- the `on:` block
    needs no per-step opt-in."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = {step.get("name"): step for step in workflow["jobs"]["test"]["steps"]}
    linux = steps["Section Chrome Six-Theme Sweep (Linux)"]
    windows = steps["Section Chrome Six-Theme Sweep (Windows)"]
    assert ".venv/bin/python" in linux["run"]
    assert "scripts/certify_section_chrome.py" in linux["run"]
    assert ".venv\\Scripts\\python.exe" in windows["run"]
    assert "scripts/certify_section_chrome.py" in windows["run"]
    assert linux["if"] == "runner.os == 'Linux'"
    assert windows["if"] == "runner.os == 'Windows'"
