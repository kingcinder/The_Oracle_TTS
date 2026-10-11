"""Every color pair the stylesheet builder actually emits must be certified.

`build_stylesheet()` paints text on surfaces beyond the original 10-pair
check (inputs, tooltips, headers, disabled states, hover text...). USED_PAIRS
is the single inventory `_certify` checks at import; EMITTED mirrors the
builder's real emissions so a new unchecked pair fails this test instead of
silently shipping an illegible theme.

The token inventory above is the fast half. The other half — the six-theme
certification over a REAL MainWindow (contrast on the rendered window, chrome
truncation, section sliders, collapse, the workspace persist round-trip) —
lives in ``scripts/certify_gui.py``, the merged harness whose ``contrast``
aspect this file's slow test runs (``--aspects contrast``; the ``chrome``
aspect is driven from tests/test_gui_sections_chrome.py). Before 2026-10-10
none of it ran in the suite at all, and the certification was executed by
nothing but a human's shell. It is now a local gate, so a theme regression
fails a developer's ``pytest`` instead of the gate.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from the_oracle.gui_themes import (
    DEFAULT_THEME,
    THEMES,
    USED_PAIRS,
    apply_theme,
    build_stylesheet,
    contrast_ratio,
    in_bulk_restyle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CERTIFIER = REPO_ROOT / "scripts" / "certify_gui.py"


def _run_certifier_in_scratch(
    scratch: Path, *args: str
) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run the certifier with ``TMPDIR`` redirected; report what it left there.

    ``TMPDIR`` is the whole interface this needs: the harness's throwaway
    config root comes from ``tempfile.mkdtemp``, which honors it, so every root
    the run creates lands under ``scratch``. That is what makes "nothing left
    behind" assertable at all — a run graded against the shared ``/tmp`` would
    be graded against other runs' litter as much as its own.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(CERTIFIER), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "TMPDIR": str(scratch)},
    )
    return proc, sorted(path.name for path in scratch.iterdir())


def _assert_no_throwaway_root(leftover: list[str], detail: str) -> None:
    assert not leftover, (
        "the run left its throwaway config root behind in TMPDIR: "
        f"{leftover} — it has to be reaped on every exit path, not only on the "
        f"one the sweep's finally covers\n{detail}"
    )

#: One theme's summary line, exactly as the certifier prints it:
#: ``  Ocean Depths (ocean_depth): OK - contrast body 11.83:1, on-accent 7.98:1``
_THEME_LINE = re.compile(
    r"^\s+(?P<name>.+?) \((?P<key>[a-z_]+)\): OK - contrast body "
    r"(?P<body>\d+\.\d+):1, on-accent (?P<accent>\d+\.\d+):1$",
    re.MULTILINE,
)


#: One theme's dialog line, as the studio/wizard aspects print it:
#: ``  ocean_depth: OK - studio 97% themed, wizard 94% themed``
_DIALOG_LINE = re.compile(
    r"^\s+(?P<key>[a-z_]+): OK - (?P<body>.+)$", re.MULTILINE
)
_THEMED_SHARE = re.compile(r"(?P<aspect>studio|wizard) (?P<share>\d+)% themed")


def _parse_dialog_report(stdout: str) -> dict[str, dict[str, int]]:
    """``{theme key: {aspect: share}}`` from the dialog aspects' own report.

    Same shape as ``_parse_theme_report`` and the same rule against passing
    vacuously: a line that is missing, or that carries no aspect reading at
    all, contributes nothing, so the caller comparing the key set against
    ``THEMES`` fails rather than reading an empty dict as agreement.
    """
    report: dict[str, dict[str, int]] = {}
    for match in _DIALOG_LINE.finditer(stdout):
        readings = {
            share["aspect"]: int(share["share"])
            for share in _THEMED_SHARE.finditer(match["body"])
        }
        if readings:
            report[match["key"]] = readings
    return report


def _parse_theme_report(stdout: str) -> dict[str, tuple[float, float]]:
    """``{theme key: (body, on-accent)}`` from the certifier's own summary.

    Missing or malformed lines yield a SHORTER dict rather than an exception:
    the caller compares the key set against ``THEMES``, so a truncated report
    can never pass by being empty (the vacuous-pass shape a crash or an early
    return would produce).
    """
    report: dict[str, tuple[float, float]] = {}
    for match in _THEME_LINE.finditer(stdout):
        report[match["key"]] = (float(match["body"]), float(match["accent"]))
    return report

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


# --- the certification itself, run locally -----------------------------------


def test_the_report_parser_cannot_pass_on_a_truncated_report() -> None:
    """The gate below leans on this parse, so pin it: a report cut short — the
    shape a crash, a skipped theme, or a mangled print line produces — must
    come back with FEWER keys, never the full six. This is the assertion that
    stops the gate from passing vacuously on an empty stdout."""
    two = (
        "  Sephiroth (sephiroth): OK - contrast body 14.09:1, on-accent 9.21:1\n"
        "  Ocean Depths (ocean_depth): OK - contrast body 11.83:1, on-accent 7.98:1\n"
    )
    parsed = _parse_theme_report(two)
    assert set(parsed) == {"sephiroth", "ocean_depth"}
    assert parsed["sephiroth"] == (14.09, 9.21)
    # Truncation and noise both read as "fewer themes", and the caller's
    # `== set(THEMES)` is what turns that into a failure.
    assert _parse_theme_report("") == {}
    assert set(_parse_theme_report("All good across all 6 themes\n")) == set()


def test_design_token_certification_runs_locally_and_passes(tmp_path: Path) -> None:
    """Run the six-theme design-token certification the way CI's step does —
    as a real subprocess on the real MainWindow — so the developer's own
    ``pytest`` catches a theme regression instead of the gate.

    Two claims, one ~8 s run (13 real windows across the six themes; the
    contrast aspect reads them, the expensive chrome checks belong to the
    sibling gate in tests/test_gui_sections_chrome.py):

    1. **The verdict.** Exit 0 is not enough on its own, because a certifier
       that printed nothing and returned 0 would look identical to a clean
       sweep. Every theme must report its own certified pair with the ratios
       the certification promises (body >= 4.5 AA, on-accent >= 3.0), and the
       summary line must be present — the parser pin above keeps that check
       from being vacuous.
    2. **Hermeticity**, which is what makes the previous claim safe to run on
       a developer's machine at all. Measured before the script's isolation
       layers landed: this same run rewrote the real app settings file once
       per theme (backed up and restored, so a SIGSEGV mid-run lost it) and
       wrote ``Output/logs/gui_startup_timing.json`` and
       ``gui_prewarm_timing.json`` into the repo. The before/after snapshots
       below — repo tree and real config root, git-ignored space included —
       are the contract that says neither can happen again.

    Deliberately NOT ``slow``-marked: this is the local half of a named CI
    step, and a ``-m "not slow"`` loop that dropped it would report green
    without certifying a single theme. tests/test_fast_subset.py asks pytest's
    own collector whether the fast subset still keeps this gate.
    """
    from the_oracle.platform_support import user_config_root

    from tests.helpers import repo_tree_snapshot, snapshot_differences

    config_root = user_config_root() / "the_oracle"
    before_repo = repo_tree_snapshot(REPO_ROOT)
    before_config = repo_tree_snapshot(config_root) if config_root.exists() else {}

    proc, leftover = _run_certifier_in_scratch(
        tmp_path / "scratch", "--aspects", "contrast"
    )
    detail = (
        f"rc={proc.returncode}\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )
    assert proc.returncode == 0, f"the design-token certification failed\n{detail}"
    _assert_no_throwaway_root(leftover, detail=detail)

    report = _parse_theme_report(proc.stdout)
    assert set(report) == set(THEMES), (
        f"the run did not report every theme (got {sorted(report)})\n{detail}"
    )
    for key, (body, on_accent) in sorted(report.items()):
        assert body >= 4.5, f"{key}: reported body contrast {body}:1 is below AA\n{detail}"
        assert on_accent >= 3.0, (
            f"{key}: reported on-accent contrast {on_accent}:1 is below 3.0\n{detail}"
        )
    assert "All good across all 6 themes" in proc.stdout, (
        f"the certification did not print its summary line\n{detail}"
    )

    for label, before, after in (
        ("real repo tree", before_repo, repo_tree_snapshot(REPO_ROOT)),
        (
            "real app-config root",
            before_config,
            repo_tree_snapshot(config_root) if config_root.exists() else {},
        ),
    ):
        created, changed, removed = snapshot_differences(before, after)
        assert not (created or changed or removed), (
            f"the certification wrote to the {label}: "
            f"created={created} changed={changed} removed={removed}\n{detail}"
        )


def test_the_harness_refuses_an_unknown_aspect_without_certifying(tmp_path: Path) -> None:
    """The gate above selects its aspect by name, so the name's plumbing is
    worth pinning: an unknown aspect exits 2 having certified NOTHING (no
    summary line, no window built), and ``--help`` lists the aspects that do
    exist. A typo that silently certified nothing is the failure this stops —
    the same class as passing on an empty report.

    Both calls are also the harness's two EXIT-EARLY paths, which is why each
    is checked for the root it made on the way: the throwaway config root is
    created at import, these exits never reach the sweep's ``finally``, and
    they used to hand the directory to ``/tmp`` for good.
    """
    rejected, rejected_left = _run_certifier_in_scratch(
        tmp_path / "rejected", "--aspects", "legibility"
    )
    assert rejected.returncode == 2, (
        f"an unknown aspect must be a usage error, got rc={rejected.returncode}"
    )
    assert "unknown aspect" in rejected.stderr
    # Every aspect the harness knows is named back to the user on a typo and
    # advertised in --help: an aspect that exists but cannot be selected (or
    # selected but never announced) is a check nobody runs.
    for aspect in ("contrast", "chrome", "studio", "wizard"):
        assert aspect in rejected.stderr, f"{aspect} missing from the usage error"
    assert "All good" not in rejected.stdout
    _assert_no_throwaway_root(rejected_left, detail=f"stderr:\n{rejected.stderr[-800:]}")

    helped, helped_left = _run_certifier_in_scratch(tmp_path / "helped", "--help")
    assert helped.returncode == 0
    for aspect in ("contrast", "chrome", "studio", "wizard"):
        assert aspect in helped.stdout, f"{aspect} missing from --help"
    _assert_no_throwaway_root(helped_left, detail=f"stdout:\n{helped.stdout[-800:]}")


def test_the_dialog_aspects_certify_the_studio_and_the_wizard(tmp_path: Path) -> None:
    """The two dialog aspects, on the real dialogs, for every theme.

    The suite pinned the main window's themes and chrome; nothing pinned that
    a theme reached the recording studio or the inference tour at all, which
    is what these aspects exist for. Exit 0 alone would be satisfied by an
    aspect that certified nothing, so every theme must report BOTH readings
    with a share at or above the certificate's documented floor
    (``THEMED_PIXEL_SHARE_BAR`` = 70% — measured: studio 97%, wizard 94%),
    and the parser pin below keeps that check from being vacuous.

    Deliberately NOT ``slow``-marked, for the same reason as its ``contrast``
    sibling above: a subset run must not be able to retire the only proof that
    a theme reaches the dialogs.
    """
    proc, leftover = _run_certifier_in_scratch(
        tmp_path / "scratch", "--aspects", "studio,wizard"
    )
    detail = (
        f"rc={proc.returncode}\n"
        f"stdout tail:\n{proc.stdout[-2500:]}\n"
        f"stderr tail:\n{proc.stderr[-2500:]}"
    )
    assert proc.returncode == 0, f"a dialog aspect failed\n{detail}"
    _assert_no_throwaway_root(leftover, detail=detail)

    report = _parse_dialog_report(proc.stdout)
    assert set(report) == set(THEMES), (
        f"the run did not report both dialog aspects for every theme "
        f"(got {sorted(report)})\n{detail}"
    )
    for key, readings in sorted(report.items()):
        assert set(readings) == {"studio", "wizard"}, (key, readings, detail)
        for aspect, share in sorted(readings.items()):
            assert share >= 70, (
                f"{key}: the {aspect} aspect reported {share}% of the dialog's "
                f"painted pixels as theme colors, below the certificate's "
                f"70% floor\n{detail}"
            )

    for phrase in (
        "the recording studio renders painted by the theme",
        "every inference-tour stage renders",
    ):
        assert phrase in proc.stdout, f"no summary line for {phrase!r}\n{detail}"


def test_the_dialog_report_parser_cannot_pass_on_a_truncated_report() -> None:
    """Form proof for the parser above: silence must not read as agreement."""
    assert _parse_dialog_report("") == {}
    assert _parse_dialog_report("  ocean_depth: OK - studio 97% themed\n") == {
        "ocean_depth": {"studio": 97}
    }
    # A theme line with no reading at all contributes no key, so the caller's
    # `set(report) == set(THEMES)` comparison fails instead of passing empty.
    assert _parse_dialog_report("  ocean_depth: OK - sections render\n") == {}


#: Rule-count budget for the app stylesheet. The 2026-10-10 selector diet
#: merged every rule whose declaration block was byte-identical to another
#: one's (59 -> 51 blocks) without changing a single painted pixel — proven
#: per-theme by a pixel-comparison probe over a widget zoo covering every
#: merged state (disabled buttons/inputs/labels/checkboxes, a pressed
#: button, a popped-up menu with a disabled item): all six themes rendered
#: IDENTICAL. New styling must share a block with an existing rule or raise
#: this budget consciously. What the diet actually bought is recorded in
#: STATE.md: rule matching is a real term (duplicating the stylesheet adds a
#: measured ~79 ms per 59-rule copy), but REMOVING 8 rules sat below the
#: noise floor of an interleaved A/B on one window (~±35 ms), so this
#: budget's job is to stop regrowth, not to promise a speedup.
RULE_BUDGET = 51


def test_stylesheet_stays_within_its_rule_budget() -> None:
    """Pin the selector diet: the app stylesheet may not silently regrow.

    Every rule is matched against every widget on every polish, so rule
    count is a budget like the certifier's rule floor — one direction
    guards against gutting (MIN_REAL_RULES), this one against creep.
    """
    for key, tokens in THEMES.items():
        rules = build_stylesheet(tokens).count("{")
        assert rules <= RULE_BUDGET, (
            f"theme '{key}' builds {rules} rules (budget {RULE_BUDGET}). "
            "Merge the new rule into an existing block with an identical "
            "declaration body, or raise RULE_BUDGET consciously with the "
            "measurement that justifies it."
        )


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
