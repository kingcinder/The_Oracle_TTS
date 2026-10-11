"""Lazy collapsed sections in MainWindow (2026-10-10, lazy-collapsed-sections).

The workspace restore remembers which sections were collapsed. The approved
design: a section restored collapsed never constructs its content widgets —
startup builds a far smaller tree (~85% of MainWindow's widgets live in the
eight sections) and every later theme switch re-polishes less. Content is
built on first demand (expand click, or any code touching one of its
widgets, which routes through ensure-on-access), and the saved workspace
state is applied to it at that moment, exactly as the eager path applied it
at startup.

Pinned here:
* a fresh window (no saved collapses) builds every section eagerly — the
  default path is byte-for-byte today's behavior;
* saved-collapsed sections construct no content at startup, and the whole
  window is correspondingly smaller;
* the real startup tail (``_on_gui_shown``: remembered backend, workspace
  restore, prewarm, theme) does not force-build deferred sections;
* expanding a deferred section builds it exactly once and replays its saved
  settings (shared options, speaker panel values);
* touching a deferred section's widget (``window.table``,
  ``window.error_panel``, ``window._render_settings()``) builds just that
  section on demand;
* persisting the workspace keeps deferred sections unbuilt and preserves
  their last saved payload slice instead of forcing them alive;
* collapsing a built section keeps its content and object identities (the
  approved keep-content semantics);
* the AST completeness pin: the section builders assign exactly the
  attributes the ensure-on-access map owns (plus an explicit plain-data
  allowlist), so a widget added to a builder cannot silently miss the map.
"""

import ast
from pathlib import Path

import pytest

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from tests.test_app_gui_profiles import _build_window

APP_GUI = Path(__file__).resolve().parents[1] / "src" / "the_oracle" / "app_gui.py"

#: Every registered section key, in registration order.
ALL_SECTION_KEYS = ("shared", "speaker_a", "speaker_b", "extra_voices", "table", "status", "paths", "live")


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _section_built(window, key: str) -> bool:
    """Content-builtness of one section (the live column keeps its content in
    the panel, so it reports through the panel)."""
    if key == "live":
        return window.live_panel.is_content_built()
    return window._section_registry[key][0].is_content_built()


def _relaunch_with_collapses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    collapse_keys,
    mutate=None,
):
    """Simulate a relaunch: a first window configures the workspace, its
    payload is saved with the requested sections collapsed, then a fresh
    window is built on top of that settings file."""
    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        if mutate is not None:
            mutate(window)
        payload = window._current_gui_settings_payload()
    finally:
        window.close()
    settings = {
        "gui": payload,
        "sections": {key: {"collapsed": True, "size_share": 50} for key in collapse_keys},
    }
    window2, paths2 = _build_window(monkeypatch, tmp_path, preseed_app_settings=settings)
    return window2, paths2


def test_fresh_window_builds_every_section_eagerly(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Baseline regression: without saved collapses nothing is deferred."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        for key in ALL_SECTION_KEYS:
            assert _section_built(window, key) is True, f"section {key} must be eager on a fresh window"
        assert window._section_registry["shared"][0].is_collapsed() is False
    finally:
        window.close()


def test_saved_collapsed_sections_are_not_built_at_startup(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The core win: a relaunch with everything collapsed constructs no
    content widgets, and the window is correspondingly smaller."""
    baseline, _ = _build_window(monkeypatch, tmp_path)
    try:
        baseline_widgets = len(baseline.findChildren(QWidget))
    finally:
        baseline.close()

    window, _paths = _relaunch_with_collapses(monkeypatch, tmp_path, ALL_SECTION_KEYS)
    try:
        for key in ALL_SECTION_KEYS:
            assert _section_built(window, key) is False, f"section {key} must stay deferred at startup"
            assert window._section_registry[key][0].is_collapsed() is True
        lazy_widgets = len(window.findChildren(QWidget))
        assert lazy_widgets < baseline_widgets - 200, (
            f"deferred startup must drop most section widgets: "
            f"{baseline_widgets} eager vs {lazy_widgets} deferred"
        )
    finally:
        window.close()


def test_expanding_a_deferred_section_builds_it_exactly_once(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The user-facing trigger: the expand click builds the content once."""
    window, _paths = _relaunch_with_collapses(monkeypatch, tmp_path, ["shared"])
    try:
        section = window._section_registry["shared"][0]
        assert section.is_content_built() is False
        section.set_collapsed(False)
        assert section.is_content_built() is True
        combo_count = window.findChildren(type(window.variant_combo))
        section.set_collapsed(True)
        section.set_collapsed(False)
        assert window.findChildren(type(window.variant_combo)) == combo_count, (
            "re-expanding must never rebuild the section"
        )
    finally:
        window.close()


def test_deferred_sections_replay_their_saved_settings_on_first_build(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Saved state must reach the lazily built widgets, exactly as it reached
    the eagerly built ones: shared options and speaker-panel values alike."""
    def mutate(window) -> None:
        window.crossfade_spin.setValue(55)
        window.loudness_combo.setCurrentText("medium")
        window.speaker_b.cfg_weight.setValue(1.25)

    window, _paths = _relaunch_with_collapses(
        monkeypatch, tmp_path, ["shared", "speaker_b"], mutate=mutate
    )
    try:
        assert window._section_content_built("shared") is False
        assert window.speaker_b.is_content_built() is False

        window._section_registry["shared"][0].set_collapsed(False)
        assert window.crossfade_spin.value() == 55
        assert window.loudness_combo.currentText() == "medium"

        window._section_registry["speaker_b"][0].set_collapsed(False)
        assert window.speaker_b.cfg_weight.value() == pytest.approx(1.25)
    finally:
        window.close()


def test_touching_a_deferred_widget_builds_only_that_section_on_demand(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ensure-on-access safety net: real code paths touch widgets
    directly, so a missing widget builds its section and nothing else."""
    def mutate(window) -> None:
        window.crossfade_spin.setValue(55)

    window, _paths = _relaunch_with_collapses(
        monkeypatch, tmp_path, ["table", "status", "shared", "speaker_a"], mutate=mutate
    )
    try:
        assert window.table.rowCount() == 0, "reading a deferred widget builds it"
        assert window._section_content_built("table") is True
        assert window._section_content_built("status") is False, "only the touched section builds"

        window.error_panel.append("hello")
        assert window._section_content_built("status") is True
        assert window.speaker_a.is_content_built() is False

        settings = window._render_settings()
        assert settings.crossfade_ms == 55, "the deferred shared section must supply saved values"
        assert window._section_content_built("shared") is True
    finally:
        window.close()


def test_persist_keeps_deferred_sections_unbuilt_and_preserves_their_slice(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Persisting the workspace must not force-build deferred sections, and
    their last saved payload slice must survive untouched (a later expand
    still sees the previous session's values)."""
    def mutate(window) -> None:
        window.crossfade_spin.setValue(55)

    window, _paths = _relaunch_with_collapses(
        monkeypatch, tmp_path, ["shared", "paths", "status", "table"], mutate=mutate
    )
    try:
        window._persist_workspace_layout()
        for key in ("shared", "paths", "status", "table"):
            assert window._section_content_built(key) is False, f"persist forced {key} alive"

        from the_oracle.gui_settings import load_app_settings

        saved = load_app_settings()
        assert saved["gui"]["project"]["crossfade_ms"] == 55, (
            "the deferred shared slice must be preserved, not reset to defaults"
        )
        for key in ("shared", "paths", "status", "table"):
            assert saved["sections"][key]["collapsed"] is True
    finally:
        window.close()


def test_startup_tail_does_not_force_build_deferred_sections(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real production startup (``_on_gui_shown``: remembered backend,
    workspace restore, prewarm, theme) must leave deferred sections alone."""
    window, _paths = _relaunch_with_collapses(monkeypatch, tmp_path, ALL_SECTION_KEYS)
    try:
        qt_app.processEvents()  # fires the single-shot _gui_shown_timer
        for key in ("shared", "paths", "table", "status"):
            assert window._section_content_built(key) is False, f"_on_gui_shown forced {key} alive"
        assert window.speaker_a.is_content_built() is False
        assert window.speaker_b.is_content_built() is False
        assert window.live_panel.is_content_built() is False
    finally:
        window.close()


def test_collapsing_a_built_section_keeps_its_content_and_identities(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Approved semantics: laziness applies at first build only. Collapsing
    after that clips to the header and keeps every widget (slider positions
    and typed values survive)."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        section = window._section_registry["speaker_a"][0]
        reference = window.speaker_a.reference_path
        widget_count = len(window.speaker_a.findChildren(QWidget))

        section.set_collapsed(True)
        assert window.speaker_a.is_content_built() is True
        assert len(window.speaker_a.findChildren(QWidget)) == widget_count, (
            "collapsing must clip, not destroy"
        )
        section.set_collapsed(False)
        assert window.speaker_a.reference_path is reference, "no rebuild, same widgets"
    finally:
        window.close()


# --- AST completeness pin ---------------------------------------------------


#: Builder method -> section key, and the plain-data assignments each builder
#: may make OUTSIDE the ensure-on-access map (eager-initialized containers and
#: flags that exist regardless of laziness). A new widget in a builder must be
#: added to the map; a new plain-data field must be listed here — either way
#: the change cannot land silently.
_BUILDERS = {
    "_build_shared_content": "shared",
    "_build_paths_content": "paths",
    "_build_review_content": "table",
    "_build_status_content": "status",
    "_build_extra_voices_content": "extra_voices",
}
_UNMAPPED_ALLOWED = {
    "_build_paths_content": {"_output_name_edited"},
}


def _self_assignments(method: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                names.add(target.attr)
    return names


def test_section_builders_assign_exactly_the_mapped_attributes() -> None:
    """Completeness of the ensure-on-access map, both directions.

    Every attribute a section builder assigns must be owned (mapped) by that
    same section — a widget added to a builder without a map entry would
    AttributeError on the lazy path only, i.e. exactly when a user's saved
    workspace has that section collapsed. Unmapped assignments are allowed
    only for the explicit plain-data allowlist above.
    """
    import the_oracle.app_gui as app_gui

    owned = dict(app_gui._SECTION_OWNED_ATTRS)
    tree = ast.parse(APP_GUI.read_text(encoding="utf-8"))
    main_window = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    methods = {
        item.name: item
        for item in main_window.body
        if isinstance(item, ast.FunctionDef)
    }

    for method_name, section_key in _BUILDERS.items():
        method = methods.get(method_name)
        assert method is not None, f"missing section builder {method_name}"
        assigned = _self_assignments(method)
        allowed_unmapped = _UNMAPPED_ALLOWED.get(method_name, set())
        mapped_to_key = {name for name, key in owned.items() if key == section_key}

        unmapped = assigned - set(owned)
        assert unmapped <= allowed_unmapped, (
            f"{method_name} assigns {sorted(unmapped)} that the ensure-on-access "
            f"map does not own; map them to '{section_key}' or allowlist them "
            f"as plain data"
        )
        wrong_section = {name for name in assigned & set(owned) if owned[name] != section_key}
        assert not wrong_section, (
            f"{method_name} assigns {sorted(wrong_section)} owned by another section"
        )
        missing = mapped_to_key - assigned
        assert not missing, (
            f"map entries {sorted(missing)} claim section '{section_key}' but "
            f"{method_name} never assigns them"
        )
