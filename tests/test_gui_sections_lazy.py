"""Lazy section content: a QHSectionGroup may defer building its content.

Contract (2026-10-10, lazy-collapsed-sections slice): a section whose saved
workspace state is collapsed must not construct its content widgets at all —
fewer widgets means a faster window build and a smaller tree for Qt to
re-polish on every theme switch (apply_theme's documented ~0.45 s stall on a
full MainWindow). The section therefore accepts a *content factory*: a
callable that populates it on first demand instead of at construction.

Rules pinned here:
* no factory -> the section behaves exactly as before (content is the
  caller's business, already "built");
* with a factory, the content does not exist until ``ensure_content()`` or
  the first expand;
* the factory runs exactly once per section, no matter how often the section
  is expanded, collapsed, or ensured;
* collapsing never destroys built content (state preservation was the
  approved semantics) — it clips to the header, as before;
* ``content_built`` fires once, after the factory ran.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QWidget

from the_oracle.gui_sections import QHSectionGroup, _COLLAPSED_HEADER_HEIGHT


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_section_without_factory_reports_content_built(qt_app) -> None:
    """Factory-less sections (every wizard/collapsible_section caller) are
    unchanged: there is nothing to defer, so content counts as built and
    ensure_content is a harmless no-op."""
    section = QHSectionGroup("Plain")
    assert section.is_content_built() is True
    section.ensure_content()  # must not raise
    assert section.is_content_built() is True
    section.deleteLater()


def test_factory_runs_only_on_demand_and_exactly_once(qt_app) -> None:
    """The factory is the section's one build: first ensure builds, later
    ensures/collapse/expand cycles never rebuild."""
    calls: list[int] = []

    section = QHSectionGroup("Deferred")

    def build() -> None:
        calls.append(1)
        QLabel("content", section)

    section.set_content_factory(build)
    assert section.is_content_built() is False, "registering a factory must not build"
    assert calls == [], "content must not be constructed while deferred"

    section.ensure_content()
    assert calls == [1]
    assert section.is_content_built() is True

    # Idempotence across every path that could re-trigger the build.
    section.ensure_content()
    section.set_collapsed(True)
    section.set_collapsed(False)  # expand -> _on_toggled -> ensure_content
    section.ensure_content()
    assert calls == [1], "the factory must run exactly once per section"
    section.deleteLater()


def test_expanding_a_deferred_section_builds_its_content(qt_app) -> None:
    """The user-facing trigger: clicking the toggle on a never-built section
    constructs the content before the section unhides."""
    section = QHSectionGroup("Deferred")
    built: list[QWidget] = []

    def build() -> None:
        widget = QLabel("content", section)
        built.append(widget)

    section.set_content_factory(build)
    section.set_collapsed(True)
    assert section.is_content_built() is False

    section.set_collapsed(False)  # the expand click
    assert section.is_content_built() is True
    assert built and built[0].parent() is section
    section.deleteLater()


def test_collapsing_never_destroys_built_content(qt_app) -> None:
    """Approved semantics: laziness applies at first build only. A collapse
    after the content exists clips to the header and keeps every widget (the
    user's slider positions and typed values must survive)."""
    calls: list[int] = []
    section = QHSectionGroup("Deferred")

    def build() -> None:
        calls.append(1)
        QLabel("content", section)

    section.set_content_factory(build)
    section.ensure_content()
    child_count = len(section.findChildren(QWidget))

    section.set_collapsed(True)
    assert section.maximumHeight() == _COLLAPSED_HEADER_HEIGHT
    assert len(section.findChildren(QWidget)) == child_count, (
        "collapsing must clip, not destroy"
    )
    section.set_collapsed(False)
    assert len(section.findChildren(QWidget)) == child_count
    assert calls == [1]
    section.deleteLater()


def test_content_built_signal_fires_once_after_the_factory(qt_app) -> None:
    """MainWindow's per-section restore hooks ride this signal, so it must
    fire exactly once and strictly after the factory populated the section."""
    events: list[str] = []
    section = QHSectionGroup("Deferred")
    section.set_content_factory(lambda: events.append("factory"))
    section.content_built.connect(lambda: events.append("signal"))

    section.ensure_content()
    assert events == ["factory", "signal"]
    section.ensure_content()
    assert events == ["factory", "signal"], "restore must not run twice"
    section.deleteLater()
