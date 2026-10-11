"""Lazy SpeakerGroup panels (2026-10-10, lazy-collapsed-sections slice).

A Speaker *section* is the SpeakerGroup itself (its chrome and its ~129
content widgets are one object), so the section-level content factory is
registered by the group, not by MainWindow. ``lazy=True`` moves the whole
panel construction out of ``__init__``: a restored-collapsed Speaker A/B
never builds its reference pickers and perceptual sliders until something
needs them (first expand, or any code touching a panel attribute, which
routes through ``__getattr__`` → ``ensure_content``).

Pinned here:
* the default (eager) construction is unchanged — the cast dialog's panels
  and every existing test keep today's semantics;
* ``lazy=True`` constructs no content widgets up front;
* touching any panel attribute builds exactly once, then reads normally;
* expanding the collapsed group builds its content (same trigger as every
  other section);
* the MainWindow-driven population calls (language/reference/blend
  choices) work on a lazily built panel.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QWidget

from the_oracle.gui_cast import SpeakerGroup


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_eager_speaker_group_builds_content_immediately(qt_app, tmp_path: Path) -> None:
    """Baseline: without lazy=True nothing changes."""
    group = SpeakerGroup("A", tmp_path)
    assert group.is_content_built() is True
    assert isinstance(group.reference_path, QLineEdit)
    assert isinstance(group.reference_picker, QComboBox)
    assert group.blend_target_path() == ""
    group.deleteLater()


def test_lazy_speaker_group_defers_its_panel(qt_app, tmp_path: Path) -> None:
    """Construction cost is the whole point: no widgets until first demand."""
    group = SpeakerGroup("B", tmp_path, lazy=True)
    try:
        assert group.is_content_built() is False
        # Only the section chrome exists (the collapse toggle and the Section
        # Size slider from QHSectionGroup.__init__ — the header must render
        # even while the panel is deferred). None of the panel's own widgets.
        children = group.findChildren(QWidget)
        assert len(children) == 2, f"only chrome controls may exist, found {children}"
        assert group.findChildren(QLineEdit) == []
        assert "reference_path" not in group.__dict__
        assert "cfg_weight" not in group.__dict__
    finally:
        group.deleteLater()


def test_touching_a_panel_attribute_builds_it_exactly_once(qt_app, tmp_path: Path) -> None:
    """The ensure-on-access safety net: MainWindow code reads
    ``group.reference_path`` etc. directly, so a missing attribute triggers
    the one build and subsequent reads are plain attribute lookups."""
    group = SpeakerGroup("B", tmp_path, lazy=True)
    try:
        reference = group.reference_path
        assert isinstance(reference, QLineEdit)
        assert group.is_content_built() is True
        widget_count = len(group.findChildren(QWidget))
        assert widget_count > 50, "the full panel must exist after the build"

        # Second wave of reads: no rebuild, same widgets.
        assert group.reference_path is reference
        assert group.language_combo is group.language_combo
        assert len(group.findChildren(QWidget)) == widget_count
    finally:
        group.deleteLater()


def test_expanding_a_collapsed_lazy_group_builds_its_content(qt_app, tmp_path: Path) -> None:
    """Same trigger as every other section: the collapse toggle's expand."""
    group = SpeakerGroup("B", tmp_path, lazy=True)
    try:
        group.set_collapsed(True)
        assert group.is_content_built() is False
        group.set_collapsed(False)
        assert group.is_content_built() is True
        assert isinstance(group.cfg_weight, QWidget)
    finally:
        group.deleteLater()


def test_population_calls_work_on_a_lazily_built_panel(qt_app, tmp_path: Path) -> None:
    """MainWindow restores a deferred panel through exactly these calls
    (language options, reference/blend choices), so they must be safe on a
    group that starts cold."""
    group = SpeakerGroup("A", tmp_path, lazy=True)
    try:
        group.set_language_options({"en": "English"}, True)
        group.set_reference_choices([], [], "")
        group.set_blend_choices([], [], "")
        assert group.is_content_built() is True
        assert group.language_combo.currentData() == "en"
        assert group.blend_target_path() == ""
        assert group.reference_path.text() == ""
    finally:
        group.deleteLater()


def test_missing_attribute_on_a_built_group_stays_a_plain_attribute_error(
    qt_app, tmp_path: Path
) -> None:
    """The ensure-on-access hook must not mask real typos: after the build
    (and for underscore internals) a missing name raises as usual."""
    group = SpeakerGroup("A", tmp_path, lazy=True)
    try:
        with pytest.raises(AttributeError):
            group.no_such_widget
        with pytest.raises(AttributeError):
            group._no_such_internal
    finally:
        group.deleteLater()
