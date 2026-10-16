#!/usr/bin/env python3
"""Six-theme GUI certification: legibility (contrast) and chrome, one sweep.

This is the merged successor to ``certify_gui_themes.py`` and
``certify_section_chrome.py`` (both deleted 2026-10-10). They built their own
windows and repeated the same six-theme loop; this builds one window per theme
and runs every family of checks against it, so the main-window half of a full
certification sweeps 13 window builds instead of 25. The studio and wizard
aspects (2026-10-10) add one dialog build per theme each — the dialogs are
windows a user reads text in, and before them nothing certified that a theme
reached anything except MainWindow. One place where a check can be added.

Aspects (``--aspects contrast,chrome,studio,wizard``, the default; the suite's
local gates and CI both select them by name — CI runs one aspect per named
step, so a failing job names the aspect rather than the harness):

* **contrast** — for each theme: the token ratios it promises are met
  (body text on its panel at the certification's floor, accent text on the
  accent), AND the window being measured really is rendering that theme
  (``_current_theme``), so the numbers certify what is on screen rather than
  the token table twice.
* **chrome** — for each theme: every registered section is one the
  certification covers and renders, no visible widget is zero-sized or
  clipped, the section size sliders move splitter space, collapse clips to
  the header and restores, the review table renders rows inside its chrome,
  and the workspace round-trips (theme, sliders, splitters, section states,
  the input file) through the app settings file and a fresh window. Each
  theme's window is also built under a ``QWidget.setStyleSheet`` trace: every
  widget-level stylesheet from the four gated modules (gui_sections,
  gui_cast, gui_chrome, gui_widgets) must find the bulk-restyle gate OPEN
  during construction, and every one of the four must fire at least once —
  the runtime invariant ``tests/test_widget_restyle_gate.py`` pins in
  isolation, proven here on six real themed windows. After the
  loop: the File menu carries the profile actions and the Theme menu carries
* **studio** — for each theme: the RECORDING STUDIO dialog built on that
  theme is painted by it (the dialog's own rendered pixels are the theme's
  tokens and it carries no stylesheet of its own), its teleprompter lists the
  text scripts it was given and never the audio decoy beside them, the Font
  Size spin reaches the prompt area, and no visible widget is zero-sized or
  clipped.
* **wizard** — for each theme: the INFERENCE TOUR walked stage by stage on
  the real main window — every stage renders a heading and an explanation,
  every stage that names a highlight target RESOLVES to a real control on
  that window (a renamed control would leave the tour pointing at nothing),
  the last stage offers Finish, the dialog stays non-modal so the user can
  keep working, and the stages carrying the most chrome are painted by the
  theme. These two aspects exist because the sweep used to certify only the
  main window: the dialogs are windows a user reads text in, and nothing
  checked that a theme reaches them at all.
  all six themes.

**Certification entries.** ``THEME_CERTIFICATIONS`` holds one explicit entry
per theme, declaring the contrast bars that theme must meet and the sections
it must render. A theme in ``gui_themes.THEMES`` with no entry fails the sweep
by name before any window is built, and an entry for a theme that no longer
exists fails it too: adding a theme is a deliberate act that must state its
budget, not something that gets swept implicitly. The same drift is covered
for SECTIONS — a key registered on the window that no entry mentions fails the
sweep (the old script carried that expectation as a comment, which is exactly
how ``paths``/``table``/``extra_voices`` came to be uncertified for a while).

Exit codes: 0 every selected aspect certified across all six themes; 1 a
certification failure (or a certification-table defect: a theme with no entry,
a stale entry, an uncertified section); 2 usage error (unknown aspect name).

Machine state is isolated, not backed up: three layers, all pointing at the
one throwaway directory (``tempfile.mkdtemp`` under /tmp) — never the
developer's real state. This is what makes the sweep runnable on every
developer run (the suite executes it; see ``--aspects`` above), and it is
measured rather than intended: before the layers existed, a run rewrote the
real settings file once per theme (a backup copy and restore in ``finally``,
so a SIGSEGV mid-run lost the user's theme, layout and last input had it not
been) and wrote ``Output/logs/gui_startup_timing.json`` and
``gui_prewarm_timing.json`` into the repo.

That one directory is also removed again on EVERY exit path — the sweep's
``finally`` plus the ``atexit`` hook that ``scripts/certifier_isolation.py``
registers when it creates the root (``reap_isolated_root``) — so ``--help``, an
aspect typo, or a failure before the window loop cannot plant a root in /tmp for
good. Measured: the exits that skip the ``finally`` are exactly where this
harness's /tmp litter came from, one root per run.

1. **Config root** — ``XDG_CONFIG_HOME`` / ``%APPDATA%`` (both, per the
   isolate_user_config contract) are redirected before any ``the_oracle``
   import, so the per-theme settings rewrites and the persistence
   round-trip never touch the real config — even if this process dies
   mid-run, which is why the old backup-and-restore was not enough and is
   gone.
2. **Window default paths** — ``app_gui.ensure_repo_default_paths`` is
   repointed at the throwaway, so ``Input/ Output/ Profiles/ Seashells``
   and, decisively, the timing logs MainWindow writes under ``Output/logs/``
   land there.
3. **Crash roots** — ``handlers.install(root=<throwaway>)`` before the
   launch-path arm, so the native dump plus exception/Qt-fatal records
   resolve into the throwaway, and the user's EXISTING consent decision is
   mirrored byte-for-byte (never created from nothing) so a consented
   machine still arms while a fresh checkout stays fail-closed.

Startup dialogs are neutralized in two layers. The records store is parked
for the whole run (``bundle.park_records``: ``crash_reports/`` is renamed to
``crash_reports.certifier-park-<pid>`` in the repo root and merged back in the
``finally``; a park left by a crashed predecessor is recovered by the next
run), so the D8 review branch — which opens a modal whenever records exist —
has nothing to read even if the stubs below were removed. On top of that, the
per-window stubs: the D8 crash flow (next-session review / first-run consent)
is modal and reads the machine's real consent state, and the first-run
inference wizard launches hardware discovery that has nothing to do with
chrome. Both timers connect inside ``_on_gui_shown``, which looks the methods
up on the instance at connect time, so plain instance attributes set right
after construction are what the timers fire — the idiom
tests/test_app_gui_profiles._build_window uses.

Windows are destroyed on the suite's own teardown discipline
(tests/conftest.destroy_leftover_main_windows): stop the pending startup
timers, close (honoring closeEvent's refusal), deleteLater, flush
DeferredDelete. A window merely dropped stays pinned alive through C-level
signal connections the GC cannot trace, and enough pinned windows replay
every deferred startup callback at once.

Cross-platform note: the checks are OS-independent, but this file has only
ever been executed on Linux. CI runs each aspect as its own named step on both
operating systems (".github/workflows/ci.yml", "GUI Six-Theme Certification -
<aspect>"), so a red job names the certification that broke.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# The throwaway config root, the environment redirect, and the reap that covers
# every exit path are the certifier family's shared preamble — the policy, its
# retries and the measurements behind them live in scripts/certifier_isolation.py.
# Two rules stay here at the call site: this must sit ABOVE any the_oracle import
# (the config root is read from the environment at call time, and the order is
# pinned by tests/test_certifier_hermeticity.py), and the sweep must also reap at
# its own teardown point, because the helper's atexit hook cannot run before Qt's
# audio stack has settled (see the sweep's ``finally``).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certifier_isolation import isolated_config_root, reap_isolated_root  # noqa: E402

_CONFIG_ROOT = isolated_config_root("oracle_gui_cert_cfg_")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QScrollArea, QSlider, QTableWidgetItem, QWidget  # noqa: E402

from the_oracle import app_gui as _app_gui  # noqa: E402
from the_oracle import gui_settings  # noqa: E402
from the_oracle.app_gui import MainWindow  # noqa: E402
from the_oracle.app_paths import ensure_repo_default_paths  # noqa: E402
from the_oracle.gui_recording import RecordingStudioDialog  # noqa: E402
from the_oracle.inference_wizard import InferenceSetupWizard  # noqa: E402
from the_oracle.gui_settings import app_settings_path  # noqa: E402
from the_oracle.gui_themes import THEMES, contrast_ratio, in_bulk_restyle  # noqa: E402


def _hermetic_paths(_repo_root):
    """The real builder, rooted in the throwaway: MainWindow's ``self.paths``
    (and every ``Output/logs`` write hanging off it) resolve under the
    config root for this whole run. Same seam the suite patches
    (tests/test_app_gui_profiles._build_window)."""
    return ensure_repo_default_paths(_CONFIG_ROOT)


_app_gui.ensure_repo_default_paths = _hermetic_paths

#: The check families this harness can run, in report order.
ASPECTS = ("contrast", "chrome", "studio", "wizard")

#: Every QHSectionGroup key the window registers today. A key the window
#: registers but this tuple misses is a section no certification entry
#: mentions — the drift ``_check_registry_is_covered`` fails on. It was a
#: comment in the old script, and the reason paths/table/extra_voices went
#: uncertified for a while after the 2026-10-08 legibility pass registered
#: them.
ALL_SECTIONS = (
    "shared",
    "speaker_a",
    "speaker_b",
    "extra_voices",
    "paths",
    "table",
    "status",
    "live",
)

#: Section keys the 2026-10-08 legibility pass gave chrome to, and the header
#: title each one must render.
CHROME_SECTIONS = ("paths", "table", "extra_voices")
TITLES = {"paths": "Script & Cast", "table": "Review", "extra_voices": "Extra Voices"}

#: Section keys whose panes are hidden by design at launch (no cast beyond
#: A/B yet). The sweep renders them explicitly before checking geometry, since
#: the size redistributer deliberately skips hidden panes.
HIDDEN_BY_DESIGN = ("extra_voices",)


@dataclass(frozen=True)
class ThemeCertification:
    """What one theme is certified for.

    ``body_bar``: floor for the theme's body text contrast against its panel.
    ``on_accent_bar``: floor for accent text against the accent fill.
    ``sections``: the registry keys this theme must render.

    The defaults are the values every theme currently holds; an entry states
    them anyway, because a theme added later may legitimately need a different
    budget (a high-contrast variant, a deliberately dimmed retro palette) and
    that decision should be written down where the sweep reads it.
    """

    body_bar: float = 4.5
    on_accent_bar: float = 3.0
    sections: tuple[str, ...] = ALL_SECTIONS


#: One explicit entry per theme — literals, never a comprehension over THEMES:
#: this is the pin. A theme with no entry here fails the sweep before a window
#: is built, and so does an entry with no theme.
THEME_CERTIFICATIONS: dict[str, ThemeCertification] = {
    "oracle_light": ThemeCertification(),
    "oracle_dark": ThemeCertification(),
    "sephiroth": ThemeCertification(),
    "pulp_scifi": ThemeCertification(),
    "tape_deck": ThemeCertification(),
    "ocean_depth": ThemeCertification(),
}

#: The four widget modules whose construction-time ``setStyleSheet`` calls
#: the bulk-restyle gate must cover (the ``tests/test_widget_restyle_gate.py``
#: net's own list, copied here because scripts cannot import tests; a test
#: pins the two lists equal, so the copy cannot drift silently).
GATED_WIDGET_MODULES = ("gui_sections.py", "gui_cast.py", "gui_chrome.py", "gui_widgets.py")

#: Construction-time floors for the gate trace, per gated module. Measured on
#: a real offscreen MainWindow with fresh settings (2026-10-12): gui_sections
#: 8 (one collapse toggle per registered section), gui_widgets 14 (two
#: SpeakerGroups x 7 captioned sliders), gui_cast 6 (2 x 3 panel dividers),
#: gui_chrome 1 (the Live header) — and no other caller at all. The floor is
#: one per module: the certificate claims each gated module's site ran on a
#: real themed window, so each must have fired at least once, or the trace
#: went blind for that module and certifies nothing about it.
GATE_TRACE_FLOORS: dict[str, int] = {
    "gui_sections.py": 1,
    "gui_cast.py": 1,
    "gui_chrome.py": 1,
    "gui_widgets.py": 1,
}

failures: list[str] = []
notes: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


# --- window plumbing ---------------------------------------------------------


def _neutralize_startup_dialogs(window: MainWindow) -> None:
    """Stub the modal startup flows before the first event pump.

    The queued timers connect inside ``_on_gui_shown``, which looks the
    methods up on the instance at connect time — instance attributes set
    right after construction are what the timers fire (the
    tests/test_app_gui_profiles idiom), so no modal can open on any machine
    state. The dialogs themselves stay covered by test_gui_crash.py and
    test_inference_wizard.py.
    """
    window._maybe_run_crash_startup_flow = lambda: None
    window._start_inference_wizard = lambda: None
    # Theme certification is independent of model availability. A cached
    # model must not start native warmup work during repeated GUI teardown.
    window._start_prewarm = lambda: None


def _build_window(app: QApplication) -> MainWindow:
    """A real window, startup dialogs stubbed, pumped like a user session."""
    window = MainWindow()
    _neutralize_startup_dialogs(window)
    window.show()
    window.resize(1500, 1500)
    for _ in range(8):
        app.processEvents()
    # Hide/show exercises the re-layout path a real session hits (minimize,
    # second monitor).
    window.hide()
    window.show()
    for _ in range(8):
        app.processEvents()
    return window


def _destroy_window(app: QApplication, window: MainWindow) -> None:
    """Tear a window down on the suite's discipline (see the module docstring).

    close() can be REFUSED (closeEvent keeps the window open when a GUI-owned
    thread cannot stop inside its bounded wait); honor the refusal —
    deleteLater() on a refused window is a hard Qt abort.
    """
    for attr in ("_gui_shown_timer", "_crash_flow_timer", "_wizard_launch_timer"):
        timer = getattr(window, attr, None)
        if timer is not None:
            timer.stop()
    if not window.close():
        raise RuntimeError(
            "certifier window refused to close; refusing to build another window"
        )
    helper = getattr(window, "_ctrl_help", None)
    if helper is not None and helper.window is not None:
        raise RuntimeError("Ctrl-hover helper remained attached after window close")
    window.deleteLater()
    # A bare processEvents() does not deliver DeferredDelete events posted
    # while outside an event loop — the explicit typed flush is the cut that
    # actually destroys the window (conftest's docstring).
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def _find_header(section) -> str | None:
    """Resolve the full title, including a readable right-elided prefix."""
    title = section.title()
    for full in TITLES.values():
        if title and (title in full or full in title):
            return full
        visible_prefix = title.removesuffix("\u2026").removesuffix("...").removesuffix("\ufffd").rstrip()
        if len(visible_prefix) >= 5 and full.startswith(visible_prefix):
            return full
    return None


def _collect_problems(widget, prefix: str = "") -> list[str]:
    """Every visible widget with a zero or clipped geometry inside its parent."""
    problems: list[str] = []
    if not widget.isVisible():
        return problems
    geo = widget.geometry()
    if widget.parentWidget() is not None and not widget.isWindow():
        parent_geo = widget.parentWidget().rect()
        if geo.width() <= 0 or geo.height() <= 0:
            problems.append(
                f"{prefix}{widget.objectName() or type(widget).__name__}: "
                f"zero size {geo.size().toTuple()}"
            )
        elif (
            not (
                isinstance(widget.parentWidget(), QScrollArea)
                or isinstance(widget.parentWidget().parentWidget(), QScrollArea)
            )
            and (geo.width() > parent_geo.width() + 2 or geo.height() > parent_geo.height() + 2)
        ):
            problems.append(
                f"{prefix}{widget.objectName() or type(widget).__name__}: "
                f"clipped child {geo.size().toTuple()} in parent {parent_geo.size().toTuple()}"
            )
    for child in widget.children():
        if isinstance(child, QWidget):
            problems.extend(_collect_problems(child, prefix))
    return problems


# --- the widget-restyle gate trace -------------------------------------------

class _StyleSetTrace:
    """Context manager recording ``(gate open, caller module)`` for every
    ``QWidget.setStyleSheet`` call made while it is open.

    The runtime half of ``tests/test_widget_restyle_gate.py``, run per theme
    on the REAL window the certification builds: the unit net proves the four
    gated modules' construction sites find the bulk-restyle gate open in
    isolation; this proves the same on a themed, shown, pumped window — six
    times over, once per theme. Only the initial construction is wrapped (the
    persistence round-trip's reopen builds untraced): the construction storm
    is the subject, and the trace costs one Python call per stylesheet.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[bool, str]] = []
        self._original = QWidget.setStyleSheet

    def __enter__(self) -> "_StyleSetTrace":
        original = self._original

        def traced(widget, style: str) -> None:
            caller = inspect.currentframe().f_back.f_code.co_filename
            self.calls.append((in_bulk_restyle(), Path(caller).name))
            original(widget, style)

        QWidget.setStyleSheet = traced  # type: ignore[method-assign]
        return self

    def __exit__(self, *_exc) -> None:
        QWidget.setStyleSheet = self._original  # type: ignore[method-assign]


def _certify_restyle_gate(key: str, calls: list[tuple[bool, str]]) -> None:
    """Every construction-time widget-level stylesheet found the gate OPEN.

    Two ways to fail, both naming the module: a call from a gated module that
    ran with the bulk-restyle gate closed (an app-wide filter is re-entered
    for each of its change events), and a gated module that fired no call at
    all during construction (the trace went blind for it, so this theme
    certifies nothing about it). The per-module counts ride the report as
    evidence rather than only being asserted.
    """
    per_module: dict[str, int] = {}
    closed: list[str] = []
    for gate_open, module in calls:
        if module not in GATED_WIDGET_MODULES:
            continue
        per_module[module] = per_module.get(module, 0) + 1
        if not gate_open:
            closed.append(module)
    check(
        not closed,
        f"{key}: widget-level setStyleSheet ran with the bulk-restyle gate "
        f"CLOSED in {sorted(set(closed))} during window construction — each "
        "call re-enters every app-wide event filter for its change events; "
        "wrap the site in `with bulk_restyle():` (the "
        "tests/test_widget_restyle_gate.py net names the site)",
    )
    for module, floor in GATE_TRACE_FLOORS.items():
        fired = per_module.get(module, 0)
        check(
            fired >= floor,
            f"{key}: the construction-time gate trace saw {fired} "
            f"setStyleSheet call(s) from {module} (floor {floor}) — the trace "
            "went blind for that module, so this theme certifies nothing "
            "about it",
        )
    # Evidence only on the passing side: a failing theme gets its failure
    # line, never an "all open" note beside it — the two readings of one
    # trace must not both be printed.
    if per_module and not closed:
        notes.append(
            f"{key}: OK - gate trace: "
            + ", ".join(f"{module} {per_module[module]}" for module in sorted(per_module))
            + " - all open"
        )


# --- the certification table -------------------------------------------------


def _audit_certification_table() -> list[str]:
    """Problems with the table itself, reported before any window is built.

    Both directions: a theme that exists and carries no entry (the sweep
    cannot certify what it has not declared), and an entry for a theme that
    no longer exists (a stale budget that would hide a rename).
    """
    problems: list[str] = []
    uncertified = sorted(set(THEMES) - set(THEME_CERTIFICATIONS))
    if uncertified:
        problems.append(
            f"theme(s) {uncertified} are in gui_themes.THEMES with no certification "
            "entry: add one to THEME_CERTIFICATIONS in scripts/certify_gui.py "
            "declaring its contrast bars and sections"
        )
    stale = sorted(set(THEME_CERTIFICATIONS) - set(THEMES))
    if stale:
        problems.append(
            f"THEME_CERTIFICATIONS names theme(s) {stale} that gui_themes.THEMES "
            "no longer defines: remove or rename the entries"
        )
    return problems


def _check_registry_is_covered(window: MainWindow) -> None:
    """No section may be registered without a certification entry covering it.

    Run once per sweep (the registry is theme-independent): a section that
    exists in the window but not in any entry is a section this harness looks
    at and says nothing about.
    """
    uncovered = sorted(set(window._section_registry) - set(ALL_SECTIONS))
    check(
        not uncovered,
        f"section(s) {uncovered} are registered on the main window but no "
        "certification entry covers them: add them to ALL_SECTIONS and to each "
        "theme's `sections` in scripts/certify_gui.py",
    )


# --- aspect: contrast --------------------------------------------------------


def _certify_contrast(key: str, cert: ThemeCertification, window: MainWindow) -> None:
    """The theme's promised ratios, on the theme the window is showing."""
    tokens = THEMES[key]
    body = contrast_ratio(tokens.text, tokens.panel)
    on_accent = contrast_ratio(tokens.accent_text, tokens.accent)
    check(
        body >= cert.body_bar,
        f"{key}: body text contrast {body:.2f} < {cert.body_bar}",
    )
    check(
        on_accent >= cert.on_accent_bar,
        f"{key}: on-accent contrast {on_accent:.2f} < {cert.on_accent_bar}",
    )
    # Without this, a stylesheet that silently fell back (or a settings file
    # the window refused) would still print two green ratio lines for tokens
    # that are not on screen.
    check(
        window._current_theme == key,
        f"{key}: the window is rendering {window._current_theme!r}, so the ratios "
        "above certify a theme that is not the one on screen",
    )
    notes.append(
        f"{tokens.name} ({key}): OK - contrast body {body:.2f}:1, "
        f"on-accent {on_accent:.2f}:1"
    )


# --- aspect: chrome ----------------------------------------------------------


def _certify_section_registry(
    app: QApplication, key: str, cert: ThemeCertification, window: MainWindow
) -> None:
    """Registry, visibility, geometry, titles, and the size sliders.

    Every slider is driven from launch geometry, BEFORE any collapse runs: a
    collapse clips its pane to 30 px and the splitter keeps that size after
    expand, so checks interleaved with collapses measure leftover state
    instead of the launch layout (reproduced: the Review pane fully
    floor-locked at [174, 104] by an earlier section's collapse, where a
    slider at its floor with zero slack cannot be moved by any value).
    """
    for section_key in cert.sections:
        entry = window._section_registry.get(section_key)
        check(entry is not None, f"{key}: section {section_key!r} missing from registry")
        if entry is None:
            continue
        section, splitter, index = entry
        check(
            section.is_collapsed() is False,
            f"{key}: {section_key} unexpectedly collapsed at launch",
        )
        hidden_by_design = section_key in HIDDEN_BY_DESIGN
        if hidden_by_design:
            check(
                not section.isVisible(),
                f"{key}: {section_key} unexpectedly visible without extra cast",
            )
            # Render it anyway: the geometry/slider/header checks below are
            # about the pane as built, and the redistributer skips hidden
            # panes by design.
            section.show()
            app.processEvents()
        check(section.isVisible(), f"{key}: {section_key} not visible when shown")
        geo = section.geometry()
        check(
            geo.width() > 20 and geo.height() > 20,
            f"{key}: {section_key} degenerate geometry {geo.size().toTuple()}",
        )
        if section_key in TITLES:
            check(
                _find_header(section) == TITLES[section_key],
                f"{key}: {section_key} header title not rendered "
                f"(found {_find_header(section)!r})",
            )
        slider = section.findChild(QSlider)
        check(slider is not None, f"{key}: {section_key} has no size slider")
        if slider is not None and not section.isHidden():
            sizes_before = splitter.sizes()
            before = sizes_before[index]
            # Drive the slider like a user (unblock signals) so the
            # redistribute path itself is what gets certified, and drive BOTH
            # directions: a single request has a no-op regime in each branch —
            # 75 matches the launch geometry when the pane already sits near
            # there (the slider only re-syncs from splitterMoved), and 30
            # clamps silently to the pane's content floor (the Review table
            # pins 140 px + 34 px chrome = 174). A wired slider moves the
            # splitter in at least one direction; a dead one moves it in
            # neither.
            for target in (30, 75):
                slider.blockSignals(False)
                slider.setValue(target)
                app.processEvents()
                if splitter.sizes()[index] != before:
                    break
            after = splitter.sizes()[index]
            check(
                after != before,
                f"{key}: {section_key} size slider did not move splitter space "
                f"(started {before}; setValue 30 and 75 were both no-ops)",
            )
            # Put the launch geometry back so this section's drive cannot
            # squeeze the next section's check (and the persistence step
            # below measures the launch layout).
            splitter.setSizes(sizes_before)
            app.processEvents()
        if hidden_by_design:
            section.hide()
            app.processEvents()


def _certify_collapse(app: QApplication, key: str, cert: ThemeCertification, window: MainWindow) -> None:
    """Collapse clips each section to its header; restore brings it back.

    Size-independent (a maxHeight clip), so it may run after every
    geometry/slider check is done. Restore is asserted because the
    persistence step runs next and must snapshot an uncollapsed launch
    layout.
    """
    for section_key in cert.sections:
        entry = window._section_registry.get(section_key)
        if entry is None:
            continue
        section, _splitter, _index = entry
        section.set_collapsed(True)
        app.processEvents()
        check(
            section.maximumHeight() < 80,
            f"{key}: {section_key} did not clip when collapsed",
        )
        section.set_collapsed(False)
        app.processEvents()
        check(
            not section.is_collapsed(),
            f"{key}: {section_key} did not restore after collapse",
        )


def _certify_truncation(app: QApplication, key: str, window: MainWindow) -> None:
    """No visible widget is zero-sized or clipped, with the table populated.

    The review table is populated first: an EMPTY table legitimately renders a
    zero-width vertical header offscreen (no rows to number), which would read
    as a false truncation.
    """
    window.table.setRowCount(3)
    for row in range(3):
        window.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        window.table.setItem(row, 2, QTableWidgetItem("sample narration line"))
    app.processEvents()
    problems = _collect_problems(window)
    if problems:
        table = window.table
        notes.append(
            f"{key} DIAGNOSTIC: table {table.geometry().width()}x{table.geometry().height()}, "
            f"header {table.horizontalHeader().geometry().width()}x"
            f"{table.horizontalHeader().geometry().height()}, "
            f"viewport {table.viewport().geometry().width()}x{table.viewport().geometry().height()}, "
            f"lower sizes {window._lower_splitter.sizes()}, "
            f"main sizes {window._main_splitter.sizes()}, "
            f"window {window.width()}x{window.height()}"
        )
    check(not problems, f"{key}: geometry problems: {problems[:8]}")
    # The review table renders rows inside its chrome.
    vp = window.table.viewport().geometry()
    check(
        vp.width() > 100 and vp.height() > 40,
        f"{key}: review table viewport degenerate {vp.size().toTuple()}",
    )
    window.table.setRowCount(0)
    app.processEvents()


def _certify_persistence(
    app: QApplication, key: str, cert: ThemeCertification, window: MainWindow, settings_file: Path
) -> None:
    """Mutate layout + settings, persist, reopen, and require it all back."""
    window.speaker_a.cfg_weight.setValue(1.11)
    window.speaker_a.pause_spin.setValue(432)
    window._sections_splitter.setSizes([500, 300, 260, 140])
    window._lower_splitter.setSizes([500, 220])
    window._paths_splitter.setSizes([420, 900])
    window.input_path.setText(str(REPO_ROOT / "Input" / "What is, reality.txt"))
    window._persist_workspace_layout()
    saved = json.loads(settings_file.read_text(encoding="utf-8"))
    check(saved.get("theme") == key, f"{key}: theme not persisted ({saved.get('theme')})")
    check(
        Path(saved.get("last_input_file", "")).name == "What is, reality.txt",
        f"{key}: input file not persisted ({saved.get('last_input_file')})",
    )
    check(
        len(saved.get("splitters", {}).get("sections", [])) == 4,
        f"{key}: sections splitter not persisted",
    )
    check(
        len(saved.get("splitters", {}).get("paths", [])) == 2,
        f"{key}: paths splitter sizes not persisted "
        f"({saved.get('splitters', {}).get('paths')})",
    )
    sections = saved.get("sections", {})
    for section_key in cert.sections:
        check(
            section_key in sections,
            f"{key}: section {section_key} missing from persistence",
        )
    _destroy_window(app, window)

    reopened = _build_window(app)
    check(
        abs(float(reopened.speaker_a.cfg_weight.value()) - 1.11) < 0.011,
        f"{key}: slider value did not survive restart "
        f"({reopened.speaker_a.cfg_weight.value()})",
    )
    check(
        reopened.speaker_a.pause_spin.value() == 432,
        f"{key}: pause value did not survive restart "
        f"({reopened.speaker_a.pause_spin.value()})",
    )
    check(
        reopened.input_path.text().endswith("What is, reality.txt"),
        f"{key}: input file did not survive restart ({reopened.input_path.text()})",
    )
    check(
        reopened._current_theme == key,
        f"{key}: theme did not survive restart ({reopened._current_theme})",
    )
    check(
        len(reopened._paths_splitter.sizes()) == 2
        and sum(reopened._paths_splitter.sizes()) > 500,
        f"{key}: paths splitter did not restore ({reopened._paths_splitter.sizes()})",
    )
    sections_after = reopened._app_settings.get("sections", {})
    check(bool(sections_after), f"{key}: section persistence empty after restart")
    _destroy_window(app, reopened)


def _certify_menus(app: QApplication, key: str, window: MainWindow) -> None:
    """The File menu carries the profile actions; the Theme menu all six."""
    labels = [
        action.text()
        for menu in window.menuBar().actions()
        if menu.menu()
        for action in menu.menu().actions()
    ]
    check("Save Profile…" in labels, f"File menu missing Save Profile (got {labels})")
    check("Load Profile…" in labels, f"File menu missing Load Profile (got {labels})")
    theme_labels = [action.text() for action in window._theme_actions.values()]
    check(
        len(theme_labels) == len(THEMES),
        f"Theme menu should list {len(THEMES)} themes, got {theme_labels}",
    )
    _destroy_window(app, window)


# --- aspects: studio and wizard ----------------------------------------------
#
# The dialogs are the other windows a user reads text in, and until these two
# aspects existed NOTHING certified that a theme reached them: the sweep built
# MainWindow and stopped there. Both share one certificate — the window's own
# pixels — plus the facts a real window supplies for that particular dialog.

#: Share of a dialog's OPAQUE pixels that must be one of the active theme's
#: tokens for its rendering to count as themed, the stride of the sampled
#: census, and the theme attributes that count as tokens. Measured on both
#: dialogs under all six themes with THIS stride: the recording studio paints
#: ~97% theme colors and the inference wizard ~94%, both led by panel_alt over
#: panel, border and bg (the studio paints all three surfaces; the wizard
#: legitimately paints one, which is why the floor below is about tokens and
#: not surfaces). A full pixel scan reads ~86% for the same dialogs — the grid
#: steps over most glyph antialiasing, which is blended by definition and
#: never a flat token — so the sampled number is the one the floor is set
#: against, with generous headroom. The floor catches a dialog that STOPPED
#: being painted by the theme (a local stylesheet, a hardcoded palette, a
#: fallback look), which is the failure this certificate exists for, rather
#: than pinning a pixel count.
THEMED_PIXEL_SHARE_BAR = 0.70
#: Distinct theme colors a themed dialog must paint. Measured: 6 for the
#: studio, 4 for the wizard. A window painted in a single flat color — the
#: shape a hardcoded background produces — fails here even if that one color
#: happens to be a token.
MIN_PAINTED_TOKENS = 2
CENSUS_STRIDE = 3
SURFACE_TOKENS = ("bg", "panel", "panel_alt")
#: The remaining token colors a themed surface may legitimately paint.
SURFACE_ACCENT_TOKENS = ("border", "accent", "accent_text", "text", "selection_bg")


def _pump(app: QApplication, rounds: int = 6) -> None:
    """Let Qt lay out and paint before anything is measured."""
    for _ in range(rounds):
        app.processEvents()


def _destroy_dialog(app: QApplication, dialog) -> None:
    """Close a dialog and let Qt actually destroy it.

    The main window's teardown discipline minus its startup timers, which a
    dialog does not have: close, deleteLater, then the typed DeferredDelete
    flush that a bare ``processEvents()`` does not deliver. A dialog merely
    dropped stays pinned alive through its C-level signal connections.
    """
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def _opaque_color_census(widget) -> tuple[dict[str, int], int]:
    """Sampled color census of a rendered widget: ``(colors, opaque_total)``.

    Alpha-0 pixels are skipped rather than counted: ``QWidget.grab()`` leaves
    regions Qt never fully painted transparent, and read as black they would
    look like an unthemed surface (measured: ~13% of the studio dialog's
    pixels are that artefact, and every one of them is alpha 0).
    """
    image = widget.grab().toImage()
    colors: dict[str, int] = {}
    opaque = 0
    for x in range(0, image.width(), CENSUS_STRIDE):
        for y in range(0, image.height(), CENSUS_STRIDE):
            pixel = image.pixelColor(x, y)
            if pixel.alpha() < 255:
                continue
            opaque += 1
            name = pixel.name().upper()
            colors[name] = colors.get(name, 0) + 1
    return colors, opaque


def _certify_themed_surface(app: QApplication, key: str, label: str, dialog) -> float:
    """The dialog is painted BY the theme; returns the measured token share.

    Three facts, in the order they can fail: the dialog carries no stylesheet
    of its own (a local sheet is how a dialog stops following the theme), the
    app-wide sheet is really installed, and the pixels Qt painted are the
    theme's — the dominant opaque color is one of the theme's surfaces and its
    tokens cover at least ``THEMED_PIXEL_SHARE_BAR`` of the opaque pixels.
    ``contrast`` certifies the token ratios; this certifies that the tokens
    reach the dialog.
    """
    check(
        dialog.styleSheet() == "",
        f"{key}: {label} carries a stylesheet of its own, so it paints itself "
        "instead of following the theme",
    )
    check(
        bool(app.styleSheet()),
        f"{key}: nothing installed an application stylesheet, so no theme is "
        f"on screen for {label}",
    )
    tokens = THEMES[key]
    surfaces = {getattr(tokens, attr).upper() for attr in SURFACE_TOKENS}
    token_hexes = surfaces | {
        getattr(tokens, attr).upper() for attr in SURFACE_ACCENT_TOKENS
    }
    colors, opaque = _opaque_color_census(dialog)
    if not opaque or not colors:
        check(False, f"{key}: {label} painted nothing, so nothing is certified")
        return 0.0

    dominant = max(colors, key=lambda name: colors[name])
    share = sum(count for name, count in colors.items() if name in token_hexes) / opaque
    top = sorted(colors.items(), key=lambda item: -item[1])[:4]
    check(
        dominant in surfaces,
        f"{key}: {label}'s dominant painted color {dominant} is not one of the "
        f"theme's surfaces {sorted(surfaces)}: the theme is not what is on "
        f"screen (top colors: {top})",
    )
    check(
        share >= THEMED_PIXEL_SHARE_BAR,
        f"{key}: only {share:.0%} of {label}'s painted pixels are the theme's "
        f"colors (floor {THEMED_PIXEL_SHARE_BAR:.0%}; top colors: {top})",
    )
    painted_tokens = sorted(name for name in colors if name in token_hexes)
    check(
        len(painted_tokens) >= MIN_PAINTED_TOKENS,
        f"{key}: {label} painted only {painted_tokens} of the theme's colors, "
        "so its chrome is not following the theme",
    )
    return share


def _studio_files() -> tuple[Path, Path]:
    """The throwaway Input/Seashells the studio reads, with content to certify.

    Inside the throwaway config root, like everything else this run writes: one
    real text script for the teleprompter to list, and one decoy audio file it
    must not list. No real folder is read or written.
    """
    root = Path(_CONFIG_ROOT)
    input_dir = root / "Input"
    voice_dir = root / "Seashells"
    input_dir.mkdir(parents=True, exist_ok=True)
    voice_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "What is, reality.txt").write_text("Hello there.\n", encoding="utf-8")
    (input_dir / "decoy.wav").write_bytes(b"\x00")
    return voice_dir, input_dir


def _certify_studio(app: QApplication, key: str) -> float:
    """The recording studio dialog, rendered under ``key``.

    Beyond the themed-surface certificate: the teleprompter lists the text
    scripts it was given and never the audio decoy next to them (the dialog's
    own filter, on a real folder), the Font Size spin reaches the prompt area,
    and no visible widget is zero-sized or clipped.
    """
    voice_dir, input_dir = _studio_files()
    dialog = RecordingStudioDialog(_CONFIG_ROOT, voice_dir, input_dir)
    dialog.show()
    _pump(app)
    share = _certify_themed_surface(app, key, "the recording studio", dialog)

    listed = [dialog.script_combo.itemText(i) for i in range(dialog.script_combo.count())]
    check(
        any("What is, reality.txt" in name for name in listed),
        f"{key}: the teleprompter does not list the text script it was given "
        f"({listed})",
    )
    check(
        not any(name.endswith(".wav") for name in listed),
        f"{key}: the teleprompter offers audio files as scripts ({listed})",
    )
    dialog.font_spin.setValue(24)
    _pump(app, 3)
    check(
        dialog.prompt_area.font().pointSize() == 24,
        f"{key}: Font Size 24 did not reach the prompt area "
        f"({dialog.prompt_area.font().pointSize()})",
    )
    problems = _collect_problems(dialog)
    check(not problems, f"{key}: recording studio geometry problems: {problems[:8]}")
    _destroy_dialog(app, dialog)
    return share


def _certify_wizard(app: QApplication, key: str, main_window: MainWindow) -> float:
    """The inference tour, walked stage by stage on the real main window.

    The tour exists to point at real controls, so the facts worth certifying
    are the ones only a real window can supply: every stage renders a heading
    and an explanation, every stage that names a highlight target resolves to
    a widget on THIS window (a renamed control would leave the tour silently
    pointing at nothing), the last stage offers Finish, the dialog stays
    non-modal so the user can keep working while it is up, and the stages with
    the most chrome are painted by the theme.
    """
    wizard = InferenceSetupWizard(main_window, mode="full")
    wizard.start()
    _pump(app)
    check(
        not wizard.isModal(),
        f"{key}: the inference wizard opened modal; it has to stay out of the "
        "way while it highlights real controls",
    )

    shares: list[float] = []
    targets = 0
    last = len(wizard._stages) - 1
    for index, stage in enumerate(wizard._stages):
        check(
            bool(wizard.heading.text().strip()),
            f"{key}: tutorial stage {stage.key!r} rendered no heading",
        )
        check(
            bool(wizard.explanation.toPlainText().strip()),
            f"{key}: tutorial stage {stage.key!r} rendered no explanation",
        )
        if stage.target_name:
            targets += 1
            check(
                wizard._target(stage.target_name) is not None,
                f"{key}: tutorial stage {stage.key!r} highlights "
                f"{stage.target_name!r}, which does not exist on the main "
                "window: the tour would point at nothing",
            )
        if index in (0, last):
            shares.append(
                _certify_themed_surface(
                    app, key, f"the inference wizard (stage {index + 1})", wizard
                )
            )
        problems = _collect_problems(wizard)
        check(
            not problems,
            f"{key}: inference wizard geometry problems on stage "
            f"{stage.key!r}: {problems[:8]}",
        )
        if index < last:
            wizard._continue()
            _pump(app, 3)

    check(
        wizard.continue_button.text() == "Finish",
        f"{key}: the last tutorial stage does not offer Finish "
        f"({wizard.continue_button.text()!r})",
    )
    check(targets > 0, f"{key}: no tutorial stage declared a highlight target at all")
    _destroy_dialog(app, wizard)
    return min(shares) if shares else 0.0


# --- the sweep ---------------------------------------------------------------


def _parse_aspects(raw: str) -> tuple[str, ...]:
    """Canonical aspect tuple from a comma-separated list; ValueError if bad.

    Unknown names and an empty selection are refused rather than ignored: a
    typo that silently certified nothing is the failure mode this guards.
    """
    requested = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not requested:
        raise ValueError("no aspects selected")
    unknown = [aspect for aspect in requested if aspect not in ASPECTS]
    if unknown:
        raise ValueError(
            f"unknown aspect(s) {unknown}; valid: {', '.join(ASPECTS)}"
        )
    return tuple(aspect for aspect in ASPECTS if aspect in requested)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="certify_gui.py",
        description="Certify the six themes' contrast and chrome offscreen.",
    )
    parser.add_argument(
        "--aspects",
        default=",".join(ASPECTS),
        metavar="LIST",
        help=(
            "comma-separated subset of "
            f"{','.join(ASPECTS)} (default: all of them)"
        ),
    )
    args = parser.parse_args(argv)
    try:
        aspects = _parse_aspects(args.aspects)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # The table is audited BEFORE the crash machinery and before any window:
    # an uncertified theme is a defect in the certification itself, and the
    # cheap fail-fast is what makes adding a theme require a deliberate entry.
    table_problems = _audit_certification_table()
    if table_problems:
        print("CERTIFICATION FAILED:", file=sys.stderr)
        for problem in table_problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    # Launch-path arming (the entry-point audit, 2026-10-08): this process
    # builds the REAL MainWindow offscreen — native-crash territory — and a
    # script entry never passes through cli.main's install+arm. Idempotent;
    # fails closed to unarmed when consent is off.
    #
    # The root override makes every crash artifact this run can produce — the
    # native dump, exception records, Qt-fatal records — resolve into the
    # throwaway config root instead of the repo's crash_reports/ (hermeticity
    # has to hold for the crash channel too; the bare arm below resolves
    # _root(), which the override has already repointed).
    from the_oracle.crash import consent as crash_consent
    from the_oracle.crash import handlers as crash_handlers

    crash_handlers.install(root=_CONFIG_ROOT)
    if crash_consent.read_consent(REPO_ROOT):
        # Mirror the user's EXISTING consent decision, byte for byte, into the
        # isolated root: read_consent is fail-closed, so without this a
        # consented machine would run the sweep silently unarmed (the
        # entry-point pin is source-level and passes either way). No decision
        # is ever created here — absent/false consent stays absent/false and
        # the arm stays fail-closed.
        shutil.copyfile(
            crash_consent.consent_path(REPO_ROOT),
            crash_consent.consent_path(_CONFIG_ROOT),
        )
    crash_handlers.arm_native_capture()
    from the_oracle.crash import bundle as crash_bundle

    app = QApplication.instance() or QApplication([])
    # Resolved AFTER the env redirect above: this is the throwaway root's
    # file, never the developer's real settings.
    settings_file = app_settings_path()
    parked = None

    try:
        # Park the records store (repo root, the same root MainWindow's D8
        # flow reads): a leftover park from a crashed predecessor is recovered
        # here, and the D8 review modal has no records to react to for the
        # whole run.
        parked = crash_bundle.park_records(REPO_ROOT)
        registry_checked = False
        for key in THEMES:
            cert = THEME_CERTIFICATIONS[key]
            # Fresh settings for each theme so the launch path runs its
            # defaults; a real window (not a shim) is required here.
            if settings_file.exists():
                settings_file.unlink()
            gui_settings.save_app_settings({"theme": key})

            gate_trace = _StyleSetTrace()
            with gate_trace:
                window = _build_window(app)

            if "chrome" in aspects:
                _certify_restyle_gate(key, gate_trace.calls)

            if "contrast" in aspects:
                _certify_contrast(key, cert, window)

            if "studio" in aspects or "wizard" in aspects:
                # Both run while this theme's window is alive: the tour needs a
                # real main window to resolve its highlight targets against.
                dialog_shares: list[str] = []
                if "studio" in aspects:
                    dialog_shares.append(
                        f"studio {_certify_studio(app, key):.0%} themed"
                    )
                if "wizard" in aspects:
                    dialog_shares.append(
                        f"wizard {_certify_wizard(app, key, window):.0%} themed"
                    )
                notes.append(f"{key}: OK - " + ", ".join(dialog_shares))

            if "chrome" in aspects:
                if not registry_checked:
                    # One window is enough: the registry is theme-independent.
                    _check_registry_is_covered(window)
                    registry_checked = True
                _certify_truncation(app, key, window)
                _certify_section_registry(app, key, cert, window)
                _certify_collapse(app, key, cert, window)

            if "chrome" in aspects:
                _certify_persistence(app, key, cert, window, settings_file)
                notes.append(
                    f"{key}: OK - sections render, sliders move, collapse clips, "
                    "layout round-trips"
                )
            else:
                _destroy_window(app, window)

        if "chrome" in aspects:
            # The menus are theme-independent: certify them on the last theme
            # the loop left in settings (a real window either way).
            if settings_file.exists():
                settings_file.unlink()
            gui_settings.save_app_settings({"theme": next(iter(THEMES))})
            _certify_menus(app, next(iter(THEMES)), _build_window(app))

        print("\n".join(f"  {note}" for note in notes))
        if failures:
            print("CERTIFICATION FAILED:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1
        count = len(THEMES)
        if "contrast" in aspects:
            print(
                f"All good across all {count} themes: contrast certified on the "
                "rendered window (body text and on-accent pairs)."
            )
        if "chrome" in aspects:
            print(
                f"All good across all {count} themes: sections render, sliders move, "
                "collapse clips, layout round-trips, no truncation, and every "
                "widget-level restyle found the gate open."
            )
        if "studio" in aspects:
            print(
                f"All good across all {count} themes: the recording studio renders "
                "painted by the theme, its teleprompter reads text scripts, its "
                "font control reaches the prompt area, and no widget is clipped."
            )
        if "wizard" in aspects:
            print(
                f"All good across all {count} themes: every inference-tour stage "
                "renders, every highlight target resolves on the real main window, "
                "the tour finishes, and it is painted by the theme."
            )
        return 0
    finally:
        # Bring the records store home first: merge (not rename), so a record
        # written during the parked window survives beside the parked
        # originals.
        crash_bundle.restore_parked_records(REPO_ROOT, parked)
        # The config root is throwaway, and the sweep is where the audio stack
        # has settled: every window sits destroyed and its pending deletions
        # were drained. So this is the removal that runs on the right side of
        # Qt's teardown; ``reap_isolated_root`` retries internally, and the
        # helper's ``atexit`` hook covers the exits that never get here.
        reap_isolated_root(_CONFIG_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
