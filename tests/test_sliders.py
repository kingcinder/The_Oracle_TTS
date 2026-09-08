from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from the_oracle.gui_widgets import PerceptualSlider

_APP = QApplication.instance() or QApplication([])


def _make(**kwargs) -> PerceptualSlider:
    defaults = {"minimum": 0.0, "maximum": 1.5, "value": 0.5, "caption": "How it sounds"}
    defaults.update(kwargs)
    return PerceptualSlider(**defaults)


def test_float_value_round_trips():
    slider = _make()
    assert slider.value() == pytest.approx(0.5, abs=0.002)
    slider.setValue(1.2)
    assert slider.value() == pytest.approx(1.2, abs=0.002)
    assert slider.slider().value() > 0


def test_value_clamped_to_domain():
    slider = _make()
    slider.setValue(99.0)
    assert slider.value() == pytest.approx(1.5, abs=0.002)
    slider.setValue(-5.0)
    assert slider.value() == pytest.approx(0.0, abs=0.002)


def test_set_range_remaps():
    slider = _make(value=1.0)
    slider.setRange(0.0, 2.0)
    assert slider.value() == pytest.approx(1.33, abs=0.002)


def test_int_mode_returns_ints():
    slider = _make(minimum=0, maximum=2000, value=180, int_mode=True, suffix=" ms")
    assert slider.value() == 180
    slider.setValue(429)
    assert slider.value() == 429  # exact: 10k-tick domain resolves 1 ms
    assert isinstance(slider.value(), int)


def test_tooltip_forwards_to_inner_slider():
    slider = _make()
    slider.setToolTip("hello")
    assert slider.toolTip() == "hello"
    assert slider.slider().toolTip() == "hello"
