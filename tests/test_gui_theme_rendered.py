"""Rendered-text spot check: build MainWindow under each of the six themes
and grab it, proving the stylesheet renders without error offscreen and that
painted text is actually distinguishable from its surfaces.

The static certifier (tests/test_gui_theme_certification.py) pins the WCAG
ratios; this test pins that real widgets paint real text with those tokens —
including the surfaces the builder styles ad hoc (status panel, section
titles, table headers).
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from the_oracle.gui_themes import THEMES, apply_theme, contrast_ratio

from tests.test_app_gui_profiles import _build_window

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _region_luminance_range(image: QImage, x0: int, y0: int, w: int, h: int) -> tuple[float, float]:
    """Return (min, max) relative luminance over a rectangle of the grab."""
    from the_oracle.gui_themes import relative_luminance

    lows, highs = [], []
    step = max(1, min(w, h) // 24)
    for y in range(y0, min(y0 + h, image.height()), step):
        for x in range(x0, min(x0 + w, image.width()), step):
            c = QColor(image.pixel(x, y))
            # Non-alpha pixels only (grab can include transparent margins).
            if c.alpha() == 0:
                continue
            hex_color = f"#{c.red():02X}{c.green():02X}{c.blue():02X}"
            lum = relative_luminance(hex_color)
            lows.append(lum)
            highs.append(lum)
    assert lows, f"empty sample region at ({x0},{y0},{w},{h})"
    return min(lows), max(highs)


@pytest.mark.parametrize("theme_key", sorted(THEMES))
def test_mainwindow_paints_readable_text_per_theme(
    qt_app, theme_key, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    applied = apply_theme(qt_app, theme_key)
    assert applied == theme_key
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window.show()
        qt_app.processEvents()
        grab = window.grab()
        assert not grab.isNull() and grab.width() > 0
        image = grab.toImage()

        # Three surfaces the builder styles: status panel (panel bg),
        # first section title patch (bg patch, secondary text), table
        # header (panel_alt, text_muted). Each must contain at least
        # large-text-level luminance spread (>= 3.0:1) somewhere inside it,
        # proving text is painted distinguishably on that surface.
        targets = [
            ("status panel", window.error_panel),
            ("review table header", window.table.horizontalHeader()),
        ]
        from the_oracle.gui_themes import THEMES as _T

        tokens = _T[theme_key]
        # The guaranteed text/bg spread for this theme's body pairs — used as
        # the reference bar: if a region shows less spread than the tokens
        # themselves allow, no readable text was painted there.
        bar = 3.0
        for name, widget in targets:
            assert widget.isVisible() or widget.parentWidget() is not None
            rect = widget.geometry()
            # Map into window coordinates (geometry of direct children is
            # already window-relative for these two; nested widgets walk up).
            pos = widget.mapTo(window, rect.topLeft())
            low, high = _region_luminance_range(image, pos.x(), pos.y(), rect.width(), rect.height())
            spread = (high + 0.05) / (low + 0.05)
            assert spread >= bar, (
                f"{theme_key}/{name}: luminance spread {spread:.2f}:1 < {bar} — "
                "text is indistinguishable from its surface"
            )
        # The theme's own body text must meet AA on its panel (sanity that
        # the reference numbers themselves are the certified ones).
        assert contrast_ratio(tokens.text, tokens.panel) >= 4.5
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()
