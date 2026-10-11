"""The slice-12 THEMES seam, promoted from an ephemeral probe (2026-10-10).

Extraction slice 12 moved ``MainWindow._build_menu`` into
``the_oracle.gui_menu``. The one non-widget dependency of that moved body is
the theme mapping, so ``build_menu`` takes it as a required keyword-only
parameter and app_gui's delegate resolves it from app_gui's own globals
(``build_menu(self, themes=THEMES)``). That is the slice's deliberate seam: a
module-global ``THEMES`` in gui_menu would have turned every
``app_gui.THEMES`` rebind into a silent no-op for the merged menu body — the
harm class the patch-couple net exists to catch.

Slice 12 proved the seam with an EPHEMERAL probe (rebind, build, assert,
revert, never committed). This file is that probe made permanent, which is
also why ``THEMES`` changed sides in the patch-couple record: the census is
DERIVED from the suite, so a test that rebinds ``app_gui.THEMES`` makes
THEMES patch-coupled by construction. The pair ``gui_themes.py:THEMES`` is
sanctioned in ``patch_couple_policy.sanctioned_exemptions`` and the latent
table's THEMES entry was pruned in the same edit.

Two legs, the same shape the other seam nets use for their form proofs:

* the REBOUND leg is the claim — a sentinel look that exists only in the
  rebound mapping must appear in the window's Theme menu;
* the BASELINE leg is the blindness guard — with nothing rebound, the same
  menu carries exactly the real mapping, so the first leg cannot pass on a
  menu that would contain the probe regardless.
"""

import os
from dataclasses import replace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMenu

import the_oracle.app_gui as app_gui
from the_oracle.gui_themes import THEMES
from tests.test_app_gui_profiles import _build_window

# A key no real theme can carry, plus the display name its tokens put on the
# menu action. Both are asserted literally, so this probe cannot pass by
# matching something the production mapping happens to contain.
SENTINEL_KEY = "sentinel_probe_theme"
SENTINEL_NAME = "Sentinel Probe Theme"


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _theme_menu_labels(window) -> list[str]:
    """The Theme menu's action texts, read from the menu bar itself.

    Reading the real menu rather than only ``window._theme_actions`` is the
    point of the claim: the sentinel must reach the MENU the user opens, so
    the assertion looks at what the menu bar actually carries.
    """
    theme_menu = next(
        menu for menu in window.menuBar().findChildren(QMenu) if menu.title() == "Theme"
    )
    return [action.text() for action in theme_menu.actions()]


def _sentinel_mapping() -> dict:
    """The real mapping plus one synthetic theme.

    A complete ``ThemeTokens`` (built by ``replace``), not a stub object:
    window construction reads the mapping for the menu, for the restored
    current theme, and for the layout-persistence path, so a stub would break
    the build for reasons unrelated to the seam and hide the claim under test.
    """
    mapping = dict(THEMES)
    mapping[SENTINEL_KEY] = replace(
        THEMES["oracle_dark"],
        key=SENTINEL_KEY,
        name=SENTINEL_NAME,
        description="synthetic theme carried by the THEMES seam test",
    )
    return mapping


def test_theme_menu_reflects_an_app_gui_level_themes_rebind(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Rebind app_gui.THEMES, build the real window, find the sentinel in the menu.

    The delegate resolves THEMES from app_gui's globals at call time and hands
    it to ``gui_menu.build_menu`` as the ``themes`` parameter, so this rebind
    MUST reach the menu. If someone later binds THEMES as a gui_menu module
    global, or hard-codes the built-in looks inside the moved body, the
    sentinel stops reaching the menu and this fails.
    """
    assert SENTINEL_KEY not in THEMES, "a real theme may not collide with the probe key"
    sentinel = _sentinel_mapping()
    monkeypatch.setattr(app_gui, "THEMES", sentinel)

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert set(window._theme_actions) == set(sentinel), (
            "the Theme menu must mirror the rebound mapping exactly, not a "
            "fixed list of built-in looks"
        )
        labels = _theme_menu_labels(window)
        assert SENTINEL_NAME in labels, (
            f"the rebound sentinel theme never reached the Theme menu: {labels}"
        )
        assert window._theme_actions[SENTINEL_KEY].text() == SENTINEL_NAME
    finally:
        window.close()


def test_theme_menu_is_the_real_mapping_when_nothing_is_rebound(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Blindness guard: unrebound, the menu carries exactly the real themes.

    Without this leg the rebound leg would also pass on a menu that carried
    the probe for some unrelated reason.
    """
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert set(window._theme_actions) == set(THEMES)
        labels = _theme_menu_labels(window)
        assert SENTINEL_NAME not in labels
        assert len(labels) == len(THEMES)
    finally:
        window.close()
