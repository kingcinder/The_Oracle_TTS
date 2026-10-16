"""Section chrome for The Oracle desktop GUI.

``QHSectionGroup`` is a QGroupBox with two affordances in its title row:

* a collapse/expand toggle (the collapsed state clips the section to its
  header, never destroys content);
* an optional ``Section Size`` slider that redistributes this section's share
  of its QSplitter against the sibling sections — the "resizing slider" for
  the GUI's sections. Slider positions and collapsed states are emitted via
  :attr:`size_share_changed` / :attr:`collapsed_changed` so the main window
  can persist them across restarts.

Lazy content: a section may register a *content factory* with
:meth:`QHSectionGroup.set_content_factory` instead of being populated by its
constructor. The factory runs on first demand — an explicit
:meth:`ensure_content` or the first expand — so a section that starts
collapsed (restored workspace state) never constructs its widgets: a smaller
tree at startup and one less subtree for Qt to re-polish on a theme switch.
Collapsing never destroys built content; :attr:`content_built` fires once,
after the factory ran, for callers that must restore saved state into the
freshly built widgets.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QGroupBox, QSlider, QToolButton, QWidget

from the_oracle.gui_themes import bulk_restyle

COLLAPSE_TOOLTIP = (
    "Collapse / Expand: hides this section's controls without discarding them; "
    "click again to bring them back. The collapsed state is remembered across restarts."
)
SECTION_SIZE_TOOLTIP = (
    "Section Size: drag to grow or shrink this section. The slider "
    "redistributes this section's share of its row or column against the "
    "neighboring sections, exactly like dragging the divider between them. "
    "The position is remembered across restarts."
)

_COLLAPSED_HEADER_HEIGHT = 30


class QHSectionGroup(QGroupBox):
    """A QGroupBox with a collapse toggle and an optional section-size slider."""

    size_share_changed = Signal(int)
    collapsed_changed = Signal(bool)
    #: Fired once, right after the registered content factory populated the
    #: section. Deferred sections use it to restore saved state on first
    #: expand (the widgets only exist from that point on).
    content_built = Signal()

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        collapsible: bool = True,
        resizable: bool = True,
    ) -> None:
        super().__init__(title, parent)
        self._full_title = title
        self._splitter = None
        self._splitter_index = -1
        self._collapsed = False
        # Lazy content: no factory registered means the caller populated the
        # section eagerly, i.e. content is "built" by definition.
        self._content_factory: Callable[[], None] | None = None
        self._content_is_built = True

        self._toggle = QToolButton(self)
        self._toggle.setText("\u2013")  # minus: expanded
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setToolTip(COLLAPSE_TOOLTIP)
        self._toggle.setAccessibleName(f"Collapse or expand the {title} section")
        # Widget-level restyles run inside the bulk-restyle scope: measured
        # (2026-10-10 construction-storm audit) at 1 change event / 0.06 ms
        # per call, so the gate costs nothing here and guarantees no app-wide
        # event filter is ever re-entered by a widget-level stylesheet — a
        # future heavier per-widget style gets the gate for free.
        with bulk_restyle():
            self._toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._toggle.setFixedSize(22, 18)
        self._toggle.toggled.connect(self._on_toggled)

        self._size_slider: QSlider | None = None
        if resizable:
            self._size_slider = QSlider(Qt.Orientation.Horizontal, self)
            self._size_slider.setRange(10, 100)
            self._size_slider.setValue(50)
            self._size_slider.setFixedWidth(90)
            self._size_slider.setToolTip(SECTION_SIZE_TOOLTIP)
            self._size_slider.setAccessibleName(f"{title} section size")
            self._size_slider.valueChanged.connect(self._on_size_slider)

        self._relayout_header()

    # -- public API ----------------------------------------------------------

    def attach_splitter(self, splitter, index: int) -> None:
        """Register the QSplitter this section lives in (for size sharing)."""
        self._splitter = splitter
        self._splitter_index = index
        splitter.splitterMoved.connect(self._sync_slider_from_splitter)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        self._toggle.setChecked(not collapsed)

    def size_share(self) -> int:
        if self._size_slider is None:
            return 50
        return self._size_slider.value()

    def set_size_share(self, percent: int) -> None:
        if self._size_slider is None:
            return
        self._size_slider.blockSignals(True)
        self._size_slider.setValue(max(10, min(100, int(percent))))
        self._size_slider.blockSignals(False)

    def has_size_slider(self) -> bool:
        return self._size_slider is not None

    def set_content_factory(self, factory: Callable[[], None]) -> None:
        """Register the callable that populates this section on first demand.

        Call before the first :meth:`ensure_content` (typically right after
        construction): registering flips the section to "not built" and the
        factory replaces any eager population. Sections without a factory
        keep the eager, always-built behavior.
        """
        self._content_factory = factory
        self._content_is_built = False

    def release_content_factory(self) -> None:
        """Drop the pending factory and the one-shot ``content_built`` hooks.

        The factory and the connected receivers are Python callables —
        closures over the main window — held by this wrapper, and PySide
        keeps such non-QObject callables alive in C-level connection
        structures that CPython's garbage collector cannot trace. A window
        destroyed with them still registered therefore keeps its Python
        wrapper pinned alive forever (the leak
        ``tests/test_gui_window_lifetime.py`` pins). Called from the accepted
        close path while the window is still fully alive; afterwards the
        section behaves as eagerly built. Collapsing/expanding still works —
        content, once built, is never discarded.
        """
        self._content_factory = None
        # disconnect() with no receivers is a printed RuntimeWarning in
        # PySide6 (not an exception), which the close path must not emit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.content_built.disconnect()

    def is_content_built(self) -> bool:
        """True once the registered factory has run (or none is registered)."""
        return self._content_is_built

    def ensure_content(self) -> None:
        """Run the registered content factory once; later calls are no-ops.

        Marked built *before* the factory runs, so a factory (or a restore
        hook on :attr:`content_built`) that touches something which circles
        back here cannot recurse forever — it sees "built" and proceeds.
        """
        if self._content_is_built:
            return
        self._content_is_built = True
        factory = self._content_factory
        if factory is not None:
            factory()
        self.content_built.emit()

    # -- internals -----------------------------------------------------------

    def _on_toggled(self, expanded: bool) -> None:
        self._collapsed = not expanded
        self._toggle.setText("\u2013" if expanded else "+")
        if expanded:
            # First expand of a deferred section: build its content before it
            # becomes visible (the factory is a no-op once built).
            self.ensure_content()
            self.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
        else:
            self.setMaximumHeight(_COLLAPSED_HEADER_HEIGHT)
        self.collapsed_changed.emit(self._collapsed)

    def _on_size_slider(self, percent: int) -> None:
        self._redistribute(percent)
        self.size_share_changed.emit(percent)

    def _redistribute(self, percent: int) -> None:
        splitter = self._splitter
        if splitter is None or self._splitter_index < 0:
            return
        sizes = splitter.sizes()
        total = sum(sizes)
        if total <= 0 or len(sizes) <= 1:
            return
        # Qt's setSizes only honors a list that matches the VISIBLE pane
        # count (a hidden pane makes it redistribute proportionally
        # instead), so build the request over visible panes only, in order.
        visible_indexes = [
            index
            for index in range(splitter.count())
            if splitter.widget(index) is not None and not splitter.widget(index).isHidden()
        ]
        if self._splitter_index not in visible_indexes or len(visible_indexes) <= 1:
            return
        mine = max(20, int(round(total * percent / 100.0)))
        others_total = max(0, total - mine)
        rest_sizes = [sizes[index] for index in visible_indexes if index != self._splitter_index]
        rest_sum = sum(rest_sizes)
        others_count = max(1, len(visible_indexes) - 1)
        even_share = max(20, others_total // others_count)
        new_sizes: list[int] = []
        for index in range(splitter.count()):
            if index not in visible_indexes:
                continue
            if index == self._splitter_index:
                new_sizes.append(mine)
            elif rest_sum <= 0:
                new_sizes.append(even_share)
            else:
                new_sizes.append(max(20, int(round(others_total * sizes[index] / rest_sum))))
        splitter.setSizes(new_sizes)

    def _sync_slider_from_splitter(self, *args) -> None:
        splitter = self._splitter
        if splitter is None or self._splitter_index < 0 or self._size_slider is None:
            return
        sizes = splitter.sizes()
        total = sum(sizes)
        if total <= 0 or self._splitter_index >= len(sizes):
            return
        percent = int(round(100.0 * sizes[self._splitter_index] / total))
        self._size_slider.blockSignals(True)
        self._size_slider.setValue(max(10, min(100, percent)))
        self._size_slider.blockSignals(False)

    # Explicit floor: QSplitter honors minimumWidth over the layout's
    # minimumSizeHint (which a wide form can inflate past the window's own
    # width, pinning every pane and deadlocking the resize sliders).
    _MIN_SECTION_WIDTH = 260

    def _relayout_header(self) -> None:
        # The header controls live above the content area; reserving their
        # width for every row needlessly narrows forms and sliders.
        self.setContentsMargins(8, 22, 8, 8)
        self.setMinimumWidth(self._MIN_SECTION_WIDTH)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt casing)
        super().resizeEvent(event)
        width = self.width()
        toggle_x = max(4, width - self._toggle.width() - 6)
        self._toggle.move(toggle_x, 2)
        if self._size_slider is not None:
            self._size_slider.move(max(4, toggle_x - self._size_slider.width() - 8), 3)
        self._elide_title()

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt casing)
        super().changeEvent(event)
        # A style/theme change can alter the title's font metrics without a
        # resize; re-elide so the title still clears the chrome.
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.FontChange):
            self._elide_title()

    def _elide_title(self) -> None:
        """Elide the title so it never paints underneath the header chrome.

        The stylesheet paints the title at the top-left while the toggle and
        size slider are pinned to the right edge; at narrow pane widths a
        long title (wide display fonts especially) would run under those
        controls. The original title is kept and re-elided on every resize
        and style change, so widening the section restores the full text.
        """
        if self.width() <= 0:
            return
        chrome_x = (
            self._size_slider.geometry().left()
            if self._size_slider is not None and self._size_slider.geometry().left() > 0
            else self._toggle.geometry().left()
        )
        # 24 px for the stylesheet's title offset, 6 px clearance gap.
        available = chrome_x - 30
        if available <= 0:
            return
        fm = QFontMetrics(self.font())
        # Calibration: the painted title tracks the widget font's advance
        # (the stylesheet's ::title letter-spacing does not reach the paint).
        if fm.horizontalAdvance(self._full_title) <= available:
            elided = self._full_title
        else:
            elided = fm.elidedText(self._full_title, Qt.ElideRight, available)
        if self.title() != elided:
            self.setTitle(elided)


def collapsible_section(
    title: str,
    parent: QWidget | None = None,
    *,
    collapsible: bool = True,
    resizable: bool = True,
) -> QHSectionGroup:
    """Convenience constructor for :class:`QHSectionGroup`."""
    return QHSectionGroup(title, parent, collapsible=collapsible, resizable=resizable)
