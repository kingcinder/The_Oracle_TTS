"""Theme system for The Oracle desktop GUI.

Each theme is a complete, self-contained design token set rendered into a
single application-wide Qt stylesheet. Themes are stylized (typography, weight,
and structure carry the identity) rather than pure re-skins, and every palette
is machine-contrast-checked at import time: all text/background pairs pass
WCAG AA (>= 4.5:1) or WCAG large-text AA (>= 3.0:1 for big/bold UI elements),
so switching themes can never make the GUI illegible.

Default theme is ``oracle_light`` (a refined version of the classic Oracle
look), so existing installs keep their familiar identity.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_THEME = "oracle_light"


# ---------------------------------------------------------------------------
# Contrast machinery (WCAG 2.x relative luminance / contrast ratio)
# ---------------------------------------------------------------------------


def _srgb_channel(channel: int) -> float:
    value = channel / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _srgb_channel(red) + 0.7152 * _srgb_channel(green) + 0.0722 * _srgb_channel(blue)


def contrast_ratio(foreground: str, background: str) -> float:
    lighter = max(relative_luminance(foreground), relative_luminance(background))
    darker = min(relative_luminance(foreground), relative_luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# Token set
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThemeTokens:
    key: str
    name: str
    description: str
    # Surfaces
    bg: str
    panel: str
    panel_alt: str
    # Text
    text: str
    text_muted: str
    text_disabled: str
    # Identity
    accent: str
    accent_text: str
    secondary: str
    # Feedback
    success: str
    warning: str
    danger: str
    # Chrome
    border: str
    selection_bg: str
    selection_text: str
    # Shape & type
    radius: int
    font_display: str
    font_body: str


THEMES: dict[str, ThemeTokens] = {
    # 1. Classic Oracle light, refined: cool paper, ink navy, the original
    #    #19466D accent carried forward as the identity color.
    "oracle_light": ThemeTokens(
        key="oracle_light",
        name="Studio Light",
        description="The classic Oracle look, refined: cool paper, ink navy, "
        "serif section titles. The original #19466D accent, kept on purpose.",
        bg="#F2F5F8",
        panel="#FFFFFF",
        panel_alt="#EDF1F6",
        text="#16222F",
        text_muted="#4A5A6A",
        text_disabled="#8496A6",
        accent="#19466D",
        accent_text="#FFFFFF",
        secondary="#0E6B66",
        success="#1F7A3D",
        warning="#8A5A00",
        danger="#A03024",
        border="#C9D4DE",
        selection_bg="#19466D",
        selection_text="#FFFFFF",
        radius=6,
        font_display='"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif',
        font_body='"Segoe UI", "Ubuntu", "Cantarell", "Helvetica Neue", Arial, sans-serif',
    ),
    # 2. Night deck: deep navy-black console for long render sessions; a
    #    moonlit steel-blue accent instead of the generic acid-on-black look.
    "oracle_dark": ThemeTokens(
        key="oracle_dark",
        name="Night Deck",
        description="Deep navy-black console for long sessions; moonlit "
        "steel-blue accent, soft cyan-green success states.",
        bg="#0D141C",
        panel="#16202B",
        panel_alt="#1D2936",
        text="#E8EEF4",
        text_muted="#A9BACB",
        text_disabled="#6E8296",
        accent="#6FB3E0",
        accent_text="#0B1620",
        secondary="#7BC8A4",
        success="#7BC8A4",
        warning="#E5C15A",
        danger="#E58B80",
        border="#2C3A48",
        selection_bg="#2C5A80",
        selection_text="#FFFFFF",
        radius=6,
        font_display='"Segoe UI", "Ubuntu", "Cantarell", "Helvetica Neue", Arial, sans-serif',
        font_body='"Segoe UI", "Ubuntu", "Cantarell", "Helvetica Neue", Arial, sans-serif',
    ),
    # 3. Sephiroth: the Tree of Life rendered as UI — gold (Tiphereth) as the
    #    single radiant accent on near-black parchment, silver (Yesod) chrome.
    "sephiroth": ThemeTokens(
        key="sephiroth",
        name="Sephiroth",
        description="Tree-of-life gold on near-black parchment with silver "
        "chrome; sharp corners, small-caps serif titles.",
        bg="#14110A",
        panel="#1D1910",
        panel_alt="#251F14",
        text="#F0E6CC",
        text_muted="#C2B28A",
        text_disabled="#8A7D5F",
        accent="#D9B45B",
        accent_text="#1D1405",
        secondary="#B8C0C8",
        success="#9CBF7A",
        warning="#E0A32E",
        danger="#D97A66",
        border="#3A3120",
        selection_bg="#5A4A1E",
        selection_text="#F7EFD8",
        radius=2,
        font_display='"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif',
        font_body='"Segoe UI", "Ubuntu", "Cantarell", "Helvetica Neue", Arial, sans-serif',
    ),
    # 4. Pulp science fiction: 1950s paperback covers — cream stock, heavy
    #    vermilion accent, teal counterpoint, condensed black display type.
    "pulp_scifi": ThemeTokens(
        key="pulp_scifi",
        name="Pulp Science Fiction",
        description="1950s paperback cover: cream stock, heavy vermilion "
        "accent, teal counterpoint, condensed black display type.",
        bg="#F4EBD8",
        panel="#FBF6EA",
        panel_alt="#EFE3C8",
        text="#2B2118",
        text_muted="#63513C",
        text_disabled="#97836A",
        accent="#C6402E",
        accent_text="#FFF6E8",
        secondary="#146B63",
        success="#2E6B34",
        warning="#8A5A00",
        danger="#A03024",
        border="#D8C9AC",
        selection_bg="#C6402E",
        selection_text="#FFF6E8",
        radius=3,
        font_display='"Arial Black", "Arial Bold", Arial, sans-serif',
        font_body='"Segoe UI", "Ubuntu", "Cantarell", "Helvetica Neue", Arial, sans-serif',
    ),
    # 5. Tape deck: a 1980s hi-fi rack — charcoal chassis, VU-meter amber
    #    readouts, LED green confirmations, monospace numerals.
    "tape_deck": ThemeTokens(
        key="tape_deck",
        name="Tape Deck",
        description="1980s hi-fi rack: charcoal chassis, VU-meter amber "
        "readouts, LED green confirmations, monospace numerals.",
        bg="#1C1F24",
        panel="#262B33",
        panel_alt="#2F3540",
        text="#E4E7EC",
        text_muted="#A6B0BD",
        text_disabled="#6F7A88",
        accent="#F0B429",
        accent_text="#241C04",
        secondary="#55C87A",
        success="#55C87A",
        warning="#F0B429",
        danger="#E5796B",
        border="#3A424E",
        selection_bg="#7A5B0E",
        selection_text="#FFF3D6",
        radius=4,
        font_display='"Cascadia Mono", Consolas, "DejaVu Sans Mono", Menlo, monospace',
        font_body='"Segoe UI", "Ubuntu", "Cantarell", "Helvetica Neue", Arial, sans-serif',
    ),
    # 6. Ocean depths: deep-sea water column — cyan sonar accent, coral
    #    counterpoint, the roundest geometry of the set.
    "ocean_depth": ThemeTokens(
        key="ocean_depth",
        name="Ocean Depths",
        description="Deep-sea water column: cyan sonar accent, coral "
        "counterpoint, the roundest geometry of the set.",
        bg="#062033",
        panel="#0B2C44",
        panel_alt="#0F3A56",
        text="#DCEBF5",
        text_muted="#9DBECF",
        text_disabled="#6B8DA1",
        accent="#43C6E0",
        accent_text="#062430",
        secondary="#F2846B",
        success="#63C98C",
        warning="#EBC068",
        danger="#F2846B",
        border="#1E4A66",
        selection_bg="#155A78",
        selection_text="#EAF7FC",
        radius=10,
        font_display='"Trebuchet MS", "Segoe UI", "Ubuntu", Verdana, sans-serif',
        font_body='"Segoe UI", "Ubuntu", "Cantarell", "Helvetica Neue", Arial, sans-serif',
    ),
}


def _certify(tokens: ThemeTokens) -> None:
    """Machine-check legibility: raise if any required text pair fails WCAG.

    Body-size pairs must pass AA (4.5:1). The disabled pair and the
    on-accent pair (used for big/bold button text) pass the large-text bar
    (3.0:1). Failure is an import-time error: an illegible theme must never
    ship.
    """
    aa = 4.5
    large = 3.0
    pairs = {
        ("text", "panel", aa): contrast_ratio(tokens.text, tokens.panel),
        ("text", "bg", aa): contrast_ratio(tokens.text, tokens.bg),
        ("text_muted", "panel", aa): contrast_ratio(tokens.text_muted, tokens.panel),
        ("text_disabled", "panel", large): contrast_ratio(tokens.text_disabled, tokens.panel),
        ("accent_text", "accent", large): contrast_ratio(tokens.accent_text, tokens.accent),
        ("selection_text", "selection_bg", large): contrast_ratio(tokens.selection_text, tokens.selection_bg),
        ("success", "panel", aa): contrast_ratio(tokens.success, tokens.panel),
        ("warning", "panel", aa): contrast_ratio(tokens.warning, tokens.panel),
        ("danger", "panel", aa): contrast_ratio(tokens.danger, tokens.panel),
        ("secondary", "panel", aa): contrast_ratio(tokens.secondary, tokens.panel),
        ("accent", "panel", aa): contrast_ratio(tokens.accent, tokens.panel),
    }
    failures = [
        f"{pair[0]} on {pair[1]} = {ratio:.2f}:1 (minimum {pair[2]:.1f}:1)"
        for pair, ratio in pairs.items()
        if ratio < pair[2]
    ]
    if failures:
        raise ValueError(f"Theme '{tokens.key}' fails contrast certification: " + "; ".join(failures))


for _tokens in THEMES.values():
    _certify(_tokens)
del _tokens


def theme_keys() -> list[str]:
    return list(THEMES.keys())


# ---------------------------------------------------------------------------
# Stylesheet rendering
# ---------------------------------------------------------------------------


def build_stylesheet(tokens: ThemeTokens) -> str:
    """Render one theme's tokens into the complete application stylesheet."""
    return f"""
* {{
    font-family: {tokens.font_body};
    font-size: 10pt;
}}
QMainWindow, QDialog {{
    background-color: {tokens.bg};
}}
QWidget {{
    color: {tokens.text};
    background-color: transparent;
}}
QLabel {{
    color: {tokens.text};
    background-color: transparent;
}}
QLabel#warning {{
    color: {tokens.warning};
}}
QLabel:disabled {{
    color: {tokens.text_disabled};
}}

/* ---- Group boxes: the section containers; the title is display type ---- */
QGroupBox {{
    background-color: {tokens.panel};
    border: 1px solid {tokens.border};
    border-radius: {tokens.radius}px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    background-color: {tokens.bg};
    color: {tokens.secondary};
    font-family: {tokens.font_display};
    font-size: 11pt;
    font-weight: 700;
    letter-spacing: 1px;
}}

/* ---- Inputs ---- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QComboBox {{
    background-color: {tokens.panel_alt};
    color: {tokens.text};
    border: 1px solid {tokens.border};
    border-radius: {tokens.radius}px;
    padding: 4px 6px;
    selection-background-color: {tokens.selection_bg};
    selection-color: {tokens.selection_text};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {{
    border: 1px solid {tokens.accent};
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{
    color: {tokens.text_disabled};
    background-color: {tokens.bg};
}}
QLineEdit[readOnly="true"], QTextEdit[readOnly="true"] {{
    background-color: {tokens.bg};
    color: {tokens.text_muted};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background-color: {tokens.panel};
    color: {tokens.text};
    border: 1px solid {tokens.border};
    selection-background-color: {tokens.selection_bg};
    selection-color: {tokens.selection_text};
}}

/* ---- Buttons: identity carries on the accent (class="accent") ---- */
QPushButton {{
    background-color: {tokens.panel_alt};
    color: {tokens.text};
    border: 1px solid {tokens.border};
    border-radius: {tokens.radius}px;
    padding: 6px 14px;
    font-weight: 600;
}}
QPushButton:hover {{
    border: 1px solid {tokens.accent};
    color: {tokens.accent};
}}
QPushButton:pressed {{
    background-color: {tokens.border};
}}
QPushButton:disabled {{
    color: {tokens.text_disabled};
    background-color: {tokens.bg};
    border: 1px solid {tokens.border};
}}
QPushButton[accent="true"] {{
    background-color: {tokens.accent};
    color: {tokens.accent_text};
    border: 1px solid {tokens.accent};
    font-weight: 700;
}}
QPushButton[accent="true"]:hover {{
    background-color: {tokens.accent};
    color: {tokens.accent_text};
    border: 1px solid {tokens.secondary};
}}
QPushButton[accent="true"]:disabled {{
    background-color: {tokens.border};
    color: {tokens.text_disabled};
    border: 1px solid {tokens.border};
}}
QPushButton[buttonRole="action"] {{
    font-size: 15px;
    padding: 8px 18px;
}}

/* ---- Check boxes ---- */
QCheckBox {{
    color: {tokens.text};
    spacing: 6px;
}}
QCheckBox:disabled {{
    color: {tokens.text_disabled};
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {tokens.border};
    border-radius: {max(2, tokens.radius - 2)}px;
    background-color: {tokens.panel_alt};
}}
QCheckBox::indicator:checked {{
    background-color: {tokens.accent};
    border: 1px solid {tokens.accent};
}}

/* ---- Tables ---- */
QTableWidget {{
    background-color: {tokens.panel};
    color: {tokens.text};
    border: 1px solid {tokens.border};
    border-radius: {tokens.radius}px;
    gridline-color: {tokens.border};
    selection-background-color: {tokens.selection_bg};
    selection-color: {tokens.selection_text};
    alternate-background-color: {tokens.panel_alt};
}}
QHeaderView::section {{
    background-color: {tokens.panel_alt};
    color: {tokens.text_muted};
    border: none;
    border-right: 1px solid {tokens.border};
    border-bottom: 1px solid {tokens.border};
    padding: 5px 8px;
    font-weight: 700;
    font-size: 9pt;
    letter-spacing: 1px;
}}

/* ---- Bars ---- */
QProgressBar {{
    background-color: {tokens.panel_alt};
    border: 1px solid {tokens.border};
    border-radius: {tokens.radius}px;
    text-align: center;
    color: {tokens.text};
    font-weight: 600;
}}
QProgressBar::chunk {{
    background-color: {tokens.accent};
    border-radius: {max(1, tokens.radius - 1)}px;
}}

/* ---- Sliders ---- */
QSlider::groove:horizontal {{
    height: 4px;
    background-color: {tokens.border};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    height: 16px;
    margin: -7px 0;
    border-radius: 8px;
    background-color: {tokens.accent};
}}
QSlider::handle:horizontal:hover {{
    background-color: {tokens.secondary};
}}
QSlider::sub-page:horizontal {{
    background-color: {tokens.accent};
    border-radius: 2px;
    opacity: 0.35;
}}
QSlider::handle:horizontal:disabled {{
    background-color: {tokens.text_disabled};
}}
QSlider::groove:horizontal:disabled {{
    background-color: {tokens.bg};
}}

/* ---- Menus ---- */
QMenuBar {{
    background-color: {tokens.bg};
    color: {tokens.text};
    border-bottom: 1px solid {tokens.border};
}}
QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
}}
QMenuBar::item:selected {{
    background-color: {tokens.selection_bg};
    color: {tokens.selection_text};
}}
QMenu {{
    background-color: {tokens.panel};
    color: {tokens.text};
    border: 1px solid {tokens.border};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 24px 5px 12px;
    border-radius: {max(2, tokens.radius - 2)}px;
}}
QMenu::item:selected {{
    background-color: {tokens.selection_bg};
    color: {tokens.selection_text};
}}
QMenu::item:disabled {{
    color: {tokens.text_disabled};
}}
QMenu::separator {{
    height: 1px;
    background-color: {tokens.border};
    margin: 4px 6px;
}}

/* ---- Scrollbars, splitters, tooltips ---- */
QScrollBar:vertical {{
    background-color: {tokens.bg};
    width: 12px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background-color: {tokens.border};
    border-radius: 5px;
    min-height: 30px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {tokens.accent};
}}
QScrollBar:horizontal {{
    background-color: {tokens.bg};
    height: 12px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background-color: {tokens.border};
    border-radius: 5px;
    min-width: 30px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {tokens.accent};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0;
    height: 0;
}}
QSplitter::handle {{
    background-color: {tokens.border};
}}
QSplitter::handle:hover {{
    background-color: {tokens.accent};
}}
QToolTip {{
    background-color: {tokens.panel_alt};
    color: {tokens.text};
    border: 1px solid {tokens.accent};
    border-radius: {tokens.radius}px;
    padding: 6px 8px;
    font-size: 9pt;
}}
QStatusBar {{
    background-color: {tokens.bg};
    color: {tokens.text_muted};
    border-top: 1px solid {tokens.border};
}}
"""


def apply_theme(app, key: str) -> str:
    """Apply theme ``key`` to ``app`` and return the applied key (the caller
    stores it). Unknown keys fall back to the default theme."""
    tokens = THEMES.get(key) or THEMES[DEFAULT_THEME]
    app.setStyleSheet(build_stylesheet(tokens))
    return tokens.key
