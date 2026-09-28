"""Window chrome for the Oracle desktop GUI: the Live sidebar column.

Owns the persistent right-side progress sidebar (2026-09-21 extraction slice).
:class:`LivePanel` mirrors the render progress dialog so the user always sees
the last backend/render state without a modal, and :func:`build_live_section`
wraps it in the collapsible/resizable section chrome the main splitter holds.

``app_gui`` re-imports both names, so MainWindow's construction
(``self.live_panel = LivePanel()``), the two progress handlers that drive it,
and the existing ``from the_oracle.app_gui import LivePanel`` test import all
keep resolving unchanged.

Patch-surface rule (see tests/test_app_gui_patch_surface.PARTIAL_OWNED): the
Live column's ``QHSectionGroup`` is built HERE, from this module's globals,
while the shared-settings and status sections are still built from
``app_gui``'s. A test that patches ``app_gui.QHSectionGroup`` therefore covers
those sections but *not* the Live column — patch the site you mean.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from the_oracle.gui_render import RenderProgressDialog
from the_oracle.gui_sections import QHSectionGroup
from the_oracle.pipeline import RenderProgress


class LivePanel(QWidget):
    """Persistent right-side sidebar that mirrors the render progress dialog
    so the user always sees the last backend/render state without a modal.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Resizable, not fixed: the Live column is now one arm of the main
        # splitter, so users can widen it (or collapse it via its section
        # header) without losing the rest of the layout.
        self.setMinimumWidth(240)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Live")
        header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px 0;")
        outer.addWidget(header)

        self.backend_label = QLabel("Backend: idle")
        self.backend_label.setWordWrap(True)
        self.synth_label = QLabel("")
        self.synth_label.setWordWrap(True)
        # Hiccup-retry tally: every RenderProgress carrying a retry_note is
        # one healed synthesis. The count is CUMULATIVE for the render (and
        # the preview that follows in the same session) and deliberately
        # survives set_idle — a self-heal is part of the session's record,
        # not a transient frame. reset_retry_tally clears it when the next
        # render or preview starts.
        self._retry_count = 0
        self.retry_label = QLabel("")
        self.retry_label.setWordWrap(True)
        self.stage_label = QLabel("")
        self.stage_label.setWordWrap(True)
        self.segment_label = QLabel("")
        self.eta_label = QLabel("")
        self.eta_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        outer.addWidget(self.backend_label)
        outer.addWidget(self.synth_label)
        outer.addWidget(self.retry_label)
        outer.addWidget(self.stage_label)
        outer.addWidget(self.segment_label)
        outer.addWidget(self.eta_label)
        outer.addWidget(self.progress_bar)
        outer.addStretch(1)

    @property
    def retry_count(self) -> int:
        """Self-heals counted so far this session (test/observer surface)."""
        return self._retry_count

    def update_from_progress(self, progress: RenderProgress) -> None:
        """Update the sidebar with live render data (same logic as
        RenderProgressDialog.update_progress but writes to our own labels)."""
        if progress.fraction is not None:
            percent = int(round(progress.fraction * 100))
        else:
            percent = 0 if progress.total_steps <= 0 else int(round((progress.current_step / progress.total_steps) * 100))
        self.progress_bar.setValue(max(0, min(100, percent)))

        # Backend line
        if progress.backend:
            label = "Vulkan (audio.cpp)" if progress.backend == "vulkan" else "PyTorch"
            if progress.device_label:
                label += f" — {progress.device_label}"
            self.backend_label.setText(f"Backend: {label}")

        # Hiccup-retry tally. retry_note rides exactly one event per healed
        # synthesis (staged by the render loop, cleared on delivery), so
        # counting note-bearing events counts self-heals. Both the render and
        # the preview progress handlers mirror through here.
        if progress.retry_note:
            self._retry_count += 1
            plural = "" if self._retry_count == 1 else "s"
            self.retry_label.setText(f"Self-healed: {self._retry_count} hiccup{plural}")

        # Render time
        if progress.synth_seconds_total is not None:
            text = f"Render time: {RenderProgressDialog._format_seconds(progress.synth_seconds_total)} total"
            if progress.synth_seconds_latest is not None:
                text += f"\nlast {RenderProgressDialog._format_seconds(progress.synth_seconds_latest)}"
            self.synth_label.setText(text)

        self.stage_label.setText(f"{progress.stage}: {progress.detail}")

        if progress.total_segments > 0:
            self.segment_label.setText(f"Segments: {progress.current_segment}/{progress.total_segments}")
        elif progress.total_steps > 0:
            self.segment_label.setText(f"Steps: {progress.current_step}/{progress.total_steps}")
        else:
            self.segment_label.setText("Segments: preparing...")

        if progress.eta_seconds is None:
            self.eta_label.setText(f"Elapsed: {RenderProgressDialog._format_seconds(progress.elapsed_seconds)}\nETA: calculating...")
        else:
            self.eta_label.setText(
                f"Elapsed: {RenderProgressDialog._format_seconds(progress.elapsed_seconds)}\n"
                f"ETA: {RenderProgressDialog._format_seconds(progress.eta_seconds)}"
            )

    def set_idle(self) -> None:
        """Reset to the idle state after a render finishes or fails.

        The hiccup-retry tally deliberately survives: a self-heal is part of
        what this session's renders produced, so the user can still see it
        after the render ends — that is the point of putting the count on the
        persistent mirror instead of only the modal dialog. The next render
        or preview calls reset_retry_tally to start a fresh count.
        """
        self.backend_label.setText("Backend: idle")
        self.synth_label.setText("")
        self.stage_label.setText("")
        self.segment_label.setText("")
        self.eta_label.setText("")
        self.progress_bar.setValue(0)

    def reset_retry_tally(self) -> None:
        """Clear the hiccup-retry tally for a new render or preview.

        MainWindow calls this when a new activity starts, so the count always
        reflects the current session's self-heals rather than an unbounded
        lifetime total.
        """
        self._retry_count = 0
        self.retry_label.setText("")


def build_live_section(panel: LivePanel, *, minimum_width: int = 220) -> QHSectionGroup:
    """Wrap ``panel`` in the Live column's collapsible section chrome.

    Mirrors ``gui_sections.collapsible_section`` for this one column: the
    section is collapsible and resizable, so the Live arm of the main splitter
    takes part in the workspace-layout persistence MainWindow wires through
    ``_register_section``.
    """
    section = QHSectionGroup("Live", collapsible=True, resizable=True)
    section.setMinimumWidth(minimum_width)
    layout = QVBoxLayout(section)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(panel)
    return section
