"""Shared GUI widgets for The Oracle desktop app.

``PerceptualSlider`` presents an engine knob (CFG weight, temperature, pause, ...)
as a slider whose scale and copy say what the change *sounds* like: an internal
0-1000 tick domain maps linearly onto the engine's numeric range, a live readout
shows the real value + unit, and a caption states the audible effect. It exposes
``value()`` / ``setValue()`` / ``setRange()`` so existing code, profiles, and tests
that drive the old spin boxes keep working unchanged.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

# 10k internal ticks: ms/% domains round-trip exactly and 2-decimal float
# knobs (e.g. temperature 0.8 on 0.1-1.5) land on exact readbacks, so saving a
# profile and reloading it never drifts the stored values.
_TICKS = 10000


class PerceptualSlider(QWidget):
    """A slider with an audible-effect caption and a live numeric readout.

    Parameters mirror what the callers need: a domain range, a starting value, an
    optional unit suffix, an always-visible caption, and whether the domain is
    integer (pause ms, blend %) or float (0-1.5 sampling knobs).
    """

    def __init__(
        self,
        minimum: float = 0.0,
        maximum: float = 1.0,
        value: float | None = None,
        suffix: str = "",
        caption: str = "",
        int_mode: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._lo = float(minimum)
        self._hi = float(maximum)
        self._suffix = suffix
        self._int_mode = int_mode

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, _TICKS)
        self._slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._slider.valueChanged.connect(self._sync_readout)

        self._readout = QLabel(self)
        self._readout.setMinimumWidth(64)
        self._readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self._slider, 1)
        row.addWidget(self._readout, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(row)
        if caption:
            caption_label = QLabel(caption, self)
            caption_label.setWordWrap(True)
            caption_label.setStyleSheet("color: palette(mid); font-size: 9pt;")
            layout.addWidget(caption_label)

        start = float(value) if value is not None else (self._lo + self._hi) / 2.0
        self.setValue(start)

    # -- domain / value API (float or int semantics per int_mode) -----------

    def setRange(self, minimum: float, maximum: float) -> None:
        self._lo = float(minimum)
        self._hi = float(maximum)
        if self._hi < self._lo:
            self._hi = self._lo
        self.setValue(self.value())  # re-clamp and re-map the current value

    def setValue(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(_ticks_for(self._lo, self._hi, float(value)))
        self._slider.blockSignals(False)
        self._sync_readout()

    def value(self) -> float | int:
        domain = _domain_for(self._lo, self._hi, self._slider.value())
        if self._int_mode:
            return int(round(domain))
        # Snap float domains to the 2-decimal grid the old spin boxes used, so
        # a stored 0.5 reads back exactly as 0.5 (never 0.49995 from tick
        # quantization) and profile/project round-trips never drift.
        return round(domain, 2)

    def slider(self) -> QSlider:
        """The underlying QSlider (tooltip/help wiring and tests)."""
        return self._slider

    def suffix(self) -> str:
        return self._suffix

    # -- Qt overrides --------------------------------------------------------

    def setToolTip(self, text: str) -> None:
        super().setToolTip(text)
        self._slider.setToolTip(text)

    # -- internals -----------------------------------------------------------

    def _sync_readout(self) -> None:
        current = self.value()
        display = str(current) if self._int_mode else f"{current:.2f}"
        self._readout.setText(f"{display}{self._suffix}")


def _ticks_for(lo: float, hi: float, value: float) -> int:
    span = hi - lo
    if span <= 0.0:
        return 0
    ratio = (value - lo) / span
    return int(round(max(0.0, min(1.0, ratio)) * _TICKS))


def _domain_for(lo: float, hi: float, ticks: int) -> float:
    span = hi - lo
    return lo + (span * ticks / _TICKS)
