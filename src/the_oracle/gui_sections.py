"""Section chrome for The Oracle desktop GUI.

``QHSectionGroup`` is a QGroupBox with two affordances in its title row:

* a collapse/expand toggle (the collapsed state clips the section to its
  header, never destroys content);
* an optional ``Section Size`` slider that redistributes this section's share
  of its QSplitter against the sibling sections — the "resizing slider" for
  the GUI's sections. Slider positions and collapsed states are emitted via
  :attr:`size_share_changed` / :attr:`collapsed_changed` so the main window
  can persist them across restarts.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGroupBox, QSlider, QToolButton, QWidget

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

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        collapsible: bool = True,
        resizable: bool = True,
    ) -> None:
        super().__init__(title, parent)
        self._splitter = None
        self._splitter_index = -1
        self._collapsed = False

        self._toggle = QToolButton(self)
        self._toggle.setText("\u2013")  # minus: expanded
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setToolTip(COLLAPSE_TOOLTIP)
        self._toggle.setAccessibleName(f"Collapse or expand the {title} section")
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

    # -- internals -----------------------------------------------------------

    def _on_toggled(self, expanded: bool) -> None:
        self._collapsed = not expanded
        self._toggle.setText("\u2013" if expanded else "+")
        if expanded:
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
        # Reserve top-right space for the toggle (+ slider) so content never
        # underlaps the header controls.
        reserved = 30 if self._size_slider is None else 130
        self.setContentsMargins(8, 22, reserved, 8)
        self.setMinimumWidth(self._MIN_SECTION_WIDTH)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt casing)
        super().resizeEvent(event)
        width = self.width()
        toggle_x = max(4, width - self._toggle.width() - 6)
        self._toggle.move(toggle_x, 2)
        if self._size_slider is not None:
            self._size_slider.move(max(4, toggle_x - self._size_slider.width() - 8), 3)


def collapsible_section(
    title: str,
    parent: QWidget | None = None,
    *,
    collapsible: bool = True,
    resizable: bool = True,
) -> QHSectionGroup:
    """Convenience constructor for :class:`QHSectionGroup`."""
    return QHSectionGroup(title, parent, collapsible=collapsible, resizable=resizable)
