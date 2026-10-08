"""Overlap detection matrix (Phase 4 of the legibility pass).

Reproduces MainWindow offscreen across all six themes x window geometries x
section states and asserts no painted text sits under another widget's
chrome or a sibling's rect.

Method (per the approved design): section titles are located by scanning the
grab for the title's certified text color — pixel truth rather than guessed
font metrics — and compared against the header chrome rects (toggle, size
slider). Sibling text surfaces (cast bar, path rows) are checked by mapped
rect intersection.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QColor, QFontMetrics, QImage
from PySide6.QtWidgets import QApplication, QWidget

from the_oracle.gui_sections import QHSectionGroup
from the_oracle.gui_themes import THEMES, apply_theme
from tests.test_app_gui_profiles import _build_window

# minimum (setMinimumSize), default (resize in __init__), large
GEOMETRIES = [(1160, 760), (1320, 1060), (1920, 1080)]

# Pixel tolerance when matching the certified title color in the grab.
_COLOR_TOL = 70
# A painted title produces many matching pixels; the toggle glyph is a short
# dash in the palette color, so a modest count separates them.
_MIN_TEXT_PIXELS = 15


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _rect_in_window(widget: QWidget, window: QWidget) -> QRect:
    return QRect(widget.mapTo(window, QPoint(0, 0)), widget.size())


def _color_near(c: QColor, ref: QColor) -> bool:
    return (
        abs(c.red() - ref.red()) <= _COLOR_TOL
        and abs(c.green() - ref.green()) <= _COLOR_TOL
        and abs(c.blue() - ref.blue()) <= _COLOR_TOL
        and c.alpha() > 128
    )


def _title_painted_left(
    image: QImage, sec_rect: QRect, title_color: QColor, chrome_left: int
) -> int | None:
    """Leftmost x (section coords) of title-colored pixels in the header band,
    scanning ONLY left of the chrome so the toggle/slider glyphs — which can
    share the title color — are never mistaken for title text."""
    x0 = sec_rect.x() + 4
    x1 = sec_rect.x() + max(5, chrome_left - 4)
    y0 = sec_rect.y() + 4
    y1 = min(sec_rect.y() + sec_rect.height() - 4, sec_rect.y() + 22)
    left = None
    count = 0
    for y in range(y0, max(y0 + 1, y1)):
        for x in range(x0, max(x0 + 1, x1)):
            if x >= image.width() or y >= image.height():
                continue
            if _color_near(QColor(image.pixel(x, y)), title_color):
                count += 1
                if left is None or x - sec_rect.x() < left:
                    left = x - sec_rect.x()
    if left is None or count < _MIN_TEXT_PIXELS:
        return None
    return left


def collect_overlaps(window: QWidget, image: QImage, theme_key: str) -> list[str]:
    findings: list[str] = []
    tokens = THEMES[theme_key]
    title_color = QColor(tokens.secondary)

    # 1. Section titles vs header chrome (toggle / size slider).
    #    The painted title is measured from pixel truth for its start (the
    #    QSS title offset differs per theme), and its width from the widget
    #    font plus a 1 px/char letter-spacing safety — calibration against
    #    grabs shows the painted run tracks these metrics. The end of the
    #    title must stay clear of the chrome's left edge.
    for sec in window.findChildren(QHSectionGroup):
        if not sec.isVisible():
            continue
        sec_rect = _rect_in_window(sec, window)
        chrome: list[tuple[str, QRect]] = [("toggle", QRect(sec._toggle.geometry()))]
        if sec._size_slider is not None:
            chrome.append(("size slider", QRect(sec._size_slider.geometry())))
        chrome_left = min(rect.left() for _name, rect in chrome)
        painted_left = _title_painted_left(image, sec_rect, title_color, chrome_left)
        if painted_left is None:
            continue
        fm = QFontMetrics(sec.font())
        # Calibration against grabs: the painted title run tracks the widget
        # font's advance exactly (the stylesheet's letter-spacing/font-size
        # on ::title does not reach the paint), so +4 px covers AA fringe.
        title_w = fm.horizontalAdvance(sec.title()) + 4
        title_end = painted_left + title_w
        for name, chrome_rect in chrome:
            if title_end >= chrome_rect.left() - 2:
                findings.append(
                    f"theme={theme_key} section={sec.title()!r} width={sec.width()}: "
                    f"title spans x={painted_left}..{title_end}, reaching the "
                    f"{name} at x={chrome_rect.left()} (rect {chrome_rect.getRect()})"
                )

    # 2. Cast summary label vs Manage-cast button.
    for label, button in ((window.cast_summary_label, window.manage_cast_button),):
        if label.isVisible() and button.isVisible():
            r1 = _rect_in_window(label, window)
            r2 = _rect_in_window(button, window)
            if r1.intersects(r2):
                findings.append(
                    f"theme={theme_key}: cast summary {r1.getRect()} overlaps "
                    f"Manage cast button {r2.getRect()}"
                )

    # 3. Path rows: labels vs fields vs browse buttons.
    path_widgets = [
        w
        for w in (
            *window._path_row_labels,
            window.input_path,
            window.outdir_path,
            window.output_name,
            window.output_name_label,
            *window._path_row_buttons,
        )
        if isinstance(w, QWidget) and w.isVisible()
    ]
    for i, a in enumerate(path_widgets):
        ra = _rect_in_window(a, window)
        for b in path_widgets[i + 1 :]:
            rb = _rect_in_window(b, window)
            if ra.intersects(rb):
                findings.append(
                    f"theme={theme_key}: {type(a).__name__} {ra.getRect()} "
                    f"overlaps {type(b).__name__} {rb.getRect()}"
                )
    return findings


def _matrix_cases():
    cases = [
        (key, w, h, "expanded")
        for key in sorted(THEMES)
        for (w, h) in GEOMETRIES
    ]
    # Mixed collapse state at the minimum geometry (widest titles + chrome).
    cases += [(key, 1160, 760, "mixed") for key in sorted(THEMES)]
    # Squeezed state: settings panes dragged to the section floor (260 px),
    # the narrowest state a user can put a section in.
    cases += [(key, 1160, 760, "squeezed") for key in sorted(THEMES)]
    return cases


@pytest.mark.parametrize("theme_key,w,h,state", _matrix_cases())
def test_no_text_overlap(qt_app, theme_key, w, h, state, monkeypatch, tmp_path):
    apply_theme(qt_app, theme_key)
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window.resize(w, h)
        window.show()
        qt_app.processEvents()
        if state == "mixed":
            for key in ("status", "paths"):
                section, _splitter, _index = window._section_registry[key]
                section.set_collapsed(True)
            qt_app.processEvents()
        elif state == "squeezed":
            # Visible panes only (the Extra Voices pane is hidden): two
            # settings sections pushed to their 260 px floor, the narrowest
            # state a user can put a section in.
            window._sections_splitter.setSizes([260, 260, 389])
            qt_app.processEvents()
        grab = window.grab()
        assert not grab.isNull()
        image = grab.toImage()
        findings = collect_overlaps(window, image, theme_key)
        assert not findings, "\n".join(findings)
    finally:
        window.close()
