"""Every color pair the stylesheet builder actually emits must be certified.

`build_stylesheet()` paints text on surfaces beyond the original 10-pair
check (inputs, tooltips, headers, disabled states, hover text...). USED_PAIRS
is the single inventory `_certify` checks at import; EMITTED mirrors the
builder's real emissions so a new unchecked pair fails this test instead of
silently shipping an illegible theme.
"""

import pytest

from the_oracle.gui_themes import (
    DEFAULT_THEME,
    THEMES,
    USED_PAIRS,
    apply_theme,
    contrast_ratio,
    in_bulk_restyle,
)

# (fg, bg, min) pairs build_stylesheet() actually paints. Keep in sync with
# gui_themes.build_stylesheet — this list is the drift pin.
EMITTED = [
    ("text", "panel", 4.5),
    ("text", "bg", 4.5),
    ("text", "panel_alt", 4.5),  # inputs, tooltips, progress bars, buttons
    ("text_muted", "panel", 4.5),
    ("text_muted", "panel_alt", 4.5),  # table headers, wizard notes
    ("text_muted", "bg", 4.5),  # status bar, read-only fields
    ("text_disabled", "panel", 3.0),
    ("text_disabled", "panel_alt", 3.0),
    ("text_disabled", "bg", 3.0),
    ("text_disabled", "border", 3.0),  # disabled accent buttons
    ("accent_text", "accent", 3.0),
    ("selection_text", "selection_bg", 3.0),
    ("success", "panel", 4.5),
    ("warning", "panel", 4.5),
    ("warning", "bg", 4.5),
    ("danger", "panel", 4.5),
    ("secondary", "panel", 4.5),
    ("secondary", "bg", 4.5),  # QGroupBox::title sits on a bg-colored patch
    ("accent", "panel", 4.5),
    ("accent", "panel_alt", 4.5),  # button hover text
]


def test_certifier_covers_every_emitted_pair() -> None:
    missing = set(EMITTED) - set(USED_PAIRS)
    stale = set(USED_PAIRS) - set(EMITTED)
    assert not missing, f"build_stylesheet emits pairs the certifier does not check: {sorted(missing)}"
    assert not stale, f"certifier checks pairs the builder never paints: {sorted(stale)}"


def test_all_six_themes_certify_with_full_pair_set() -> None:
    assert len(THEMES) == 6
    failures = []
    for tokens in THEMES.values():
        for fg, bg, bar in USED_PAIRS:
            ratio = contrast_ratio(getattr(tokens, fg), getattr(tokens, bg))
            if ratio < bar:
                failures.append(f"{tokens.key}: {fg} on {bg} = {ratio:.2f} < {bar}")
    assert not failures, "\n".join(failures)


def test_apply_theme_skips_a_redundant_restyle() -> None:
    """Re-applying the theme already shown must not call setStyleSheet.

    Qt re-polishes the whole widget tree on every ``setStyleSheet`` call:
    measured ~0.45 s synchronous on a fully built MainWindow (a first set on
    an unstyled app is ~0.08 s). The stall lands inside whatever event-loop
    pass triggered the call and starves sub-second timers around it —
    reproduced as the recording guide's 150 ms launch timer missing its
    deadline whenever a later window re-applied the default theme over an
    earlier window's, which every multi-window test run does.
    """

    class _AppSpy:
        """Duck-typed QApplication stand-in: records stylesheet writes and
        whether the bulk-restyle scope was open inside each one."""

        def __init__(self) -> None:
            self.style_sheet = ""
            self.set_calls = 0
            self.scope_during_set: list[bool] = []

        def styleSheet(self) -> str:  # noqa: N802 (Qt naming)
            return self.style_sheet

        def setStyleSheet(self, css: str) -> None:  # noqa: N802 (Qt naming)
            self.set_calls += 1
            self.scope_during_set.append(in_bulk_restyle())
            self.style_sheet = css

    app = _AppSpy()
    assert apply_theme(app, DEFAULT_THEME) == DEFAULT_THEME
    assert app.set_calls == 1, "the first application must set the stylesheet"

    assert apply_theme(app, DEFAULT_THEME) == DEFAULT_THEME
    assert app.set_calls == 1, (
        "re-applying the current theme re-polished the whole tree "
        "(setStyleSheet called again)"
    )

    other = next(key for key in sorted(THEMES) if key != DEFAULT_THEME)
    assert apply_theme(app, other) == other
    assert app.set_calls == 2, "a genuine theme switch must still apply"

    # Unknown keys fall back to the default theme — still a real change.
    assert apply_theme(app, "no-such-theme") == DEFAULT_THEME
    assert app.set_calls == 3
    # ...and asking for that same default again stays a no-op.
    assert apply_theme(app, "no-such-theme") == DEFAULT_THEME
    assert app.set_calls == 3

    # Every stylesheet write ran inside the bulk-restyle scope (so app-wide
    # event filters can skip the ~20k-event storm), and the scope is closed
    # again by the time apply_theme returns.
    assert app.scope_during_set == [True, True, True]
    assert in_bulk_restyle() is False


def test_bulk_restyle_scope_closes_when_setstylesheet_raises() -> None:
    """A raising ``setStyleSheet`` must not leave app-wide event filters
    gated off forever: the scope unwinds in ``finally``."""

    class _RaisingApp:
        def styleSheet(self) -> str:  # noqa: N802 (Qt naming)
            return ""

        def setStyleSheet(self, css: str) -> None:  # noqa: N802 (Qt naming)
            raise RuntimeError("style rejected")

    with pytest.raises(RuntimeError, match="style rejected"):
        apply_theme(_RaisingApp(), DEFAULT_THEME)
    assert in_bulk_restyle() is False
