"""Offscreen section-chrome certification: Script & Cast / Review / Extra Voices.

Builds the real MainWindow offscreen for every one of the six themes and
certifies the three areas the 2026-10-08 legibility pass gave chrome to:
  * the section registry carries ``paths``/``table``/``extra_voices``, each a
    visible QHSectionGroup with non-degenerate geometry;
  * the chrome shows the right header title ("Script & Cast", "Review",
    "Extra Voices") — read from the section's own ``title()``, which may be
    elided, and matched against the full expected strings;
  * each section's size slider actually moves splitter space (the hidden
    extra-voices pane is exempt by design: the redistributer skips hidden
    panes, pinned by test_hidden_extra_voices_pane_does_not_break_size_slider);
  * collapse clips the section to its header and restore brings it back;
  * the review table renders rows inside its chrome (populated viewport);
  * the new paths splitter persists to the app settings file and restores on
    a fresh window.

Checks run in two passes — every geometry/slider check first, from fresh
launch geometry, then the collapse checks — because a collapse clips its
pane to 30 px and the splitter keeps that size on expand: interleaved, the
slider checks measured leftover state instead of the launch layout
(reproduced: the Review pane fully floor-locked [174, 104] on one tree,
where a splitter whose panes already sit at their minimums cannot be moved
by ANY slider value).

This is the six-theme sweep behind tests/test_gui_sections_chrome.py's unit
pins: the suite drives the sections through a faked window; this script
drives the REAL window on every theme, so a theme-stylesheet regression that
only shows up in rendered chrome fails here. Standalone like
certify_gui_themes.py — CI runs it as a named step on both operating systems.

Machine state is isolated, not backed up: the whole run points the platform
config root (``XDG_CONFIG_HOME`` / ``%APPDATA%`` — both, per the
isolate_user_config contract) at a throwaway directory, so the per-theme
settings rewrites and the persistence round-trip never touch the developer's
real config — even if this process dies mid-run (a SIGSEGV skips every
``finally``; a backup file the crash leaves behind is a recovery chore, a
throwaway root is simply litter in /tmp).

Startup dialogs are neutralized per window, not by moving machine state
around: the D8 crash flow (next-session review / first-run consent) is modal
and reads the machine's real crash_reports/consent state — a fresh CI
checkout has no recorded decision, so it would open a modal and block the
sweep — and the first-run inference wizard launches hardware discovery that
has nothing to do with chrome. Both timers connect inside ``_on_gui_shown``
(the first queued startup timer), which looks the methods up on the instance
at connect time, so plain instance attributes set right after construction
are what the timers fire — the same stub idiom
tests/test_app_gui_profiles._build_window uses. The dialogs themselves stay
covered by test_gui_crash.py and test_inference_wizard.py.

Windows are destroyed on the suite's own teardown discipline
(tests/conftest.destroy_leftover_main_windows): stop the pending startup
timers, close (honoring closeEvent's refusal), deleteLater, flush
DeferredDelete. A window merely dropped stays pinned alive through C-level
signal connections the GC cannot trace, and enough pinned windows replay
every deferred startup callback at once — the flaky teardown-segfault class
this sweep hit before adopting the discipline.

Exit code 0 means every theme renders the three sections correctly offscreen.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Throwaway config root for the whole run (must precede any the_oracle
# import: user_config_root() reads the env at call time, but keep the order
# honest anyway). APPDATA covers the Windows leg — XDG alone would still
# read and write the developer's real %APPDATA%\the_oracle.
_CONFIG_ROOT = tempfile.mkdtemp(prefix="oracle_chrome_cfg_")
os.environ["XDG_CONFIG_HOME"] = _CONFIG_ROOT
os.environ["APPDATA"] = _CONFIG_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QSlider, QTableWidgetItem  # noqa: E402

from the_oracle import gui_settings  # noqa: E402
from the_oracle.app_gui import MainWindow  # noqa: E402
from the_oracle.gui_settings import app_settings_path  # noqa: E402
from the_oracle.gui_themes import THEMES  # noqa: E402

failures: list[str] = []
notes: list[str] = []

NEW_SECTIONS = ("paths", "table", "extra_voices")
TITLES = {"paths": "Script & Cast", "table": "Review", "extra_voices": "Extra Voices"}


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def find_header(section) -> str | None:
    """QHSectionGroup is a QGroupBox: the title is its own, possibly elided."""
    title = section.title()
    for full in TITLES.values():
        if title in full or full in title:
            return full
    return None


def _build_window(app: QApplication) -> MainWindow:
    """A real window, startup dialogs stubbed, pumped like a user session."""
    window = MainWindow()
    window._maybe_run_crash_startup_flow = lambda: None
    window._start_inference_wizard = lambda: None
    window.show()
    window.resize(1500, 1500)
    for _ in range(8):
        app.processEvents()
    # Hide/show exercises the re-layout path a real session hits (minimize,
    # second monitor) — the same cycle certify_gui_themes certifies.
    window.hide()
    window.show()
    for _ in range(8):
        app.processEvents()
    return window


def _destroy_window(app: QApplication, window: MainWindow) -> None:
    """Tear a window down on the suite's discipline (conftest's helper).

    close() can be REFUSED (closeEvent keeps the window open when a
    GUI-owned thread cannot stop inside its bounded wait); honor the refusal
    — deleteLater() on a refused window is a hard Qt abort.
    """
    for attr in ("_gui_shown_timer", "_crash_flow_timer", "_wizard_launch_timer"):
        timer = getattr(window, attr, None)
        if timer is not None:
            timer.stop()
    if window.close():
        window.deleteLater()
    # A bare processEvents() does not deliver DeferredDelete events posted
    # while outside an event loop — the explicit typed flush is the cut
    # that actually destroys the window (conftest's docstring).
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def main() -> int:
    # Launch-path arming (the entry-point audit, 2026-10-08): this process
    # builds the REAL MainWindow offscreen — native-crash territory — and a
    # script entry never passes through cli.main's install+arm. Idempotent;
    # fails closed to unarmed when consent is off.
    from the_oracle.crash import handlers as crash_handlers

    crash_handlers.arm_native_capture()
    app = QApplication.instance() or QApplication([])
    # Resolved AFTER the env redirect above: this is the throwaway root's
    # file, never the developer's real settings.
    settings_file = app_settings_path()

    try:
        for key in THEMES:
            failures_before = len(failures)
            # Fresh settings for each theme so the launch path runs its
            # defaults; a real window (not a shim) is required here.
            if settings_file.exists():
                settings_file.unlink()
            gui_settings.save_app_settings({"theme": key})

            window = _build_window(app)

            # Pass 1: registry, visibility, geometry, titles, and the size
            # sliders — all BEFORE any collapse runs, so every slider is
            # driven from launch geometry. (A collapse clips its pane to30
            # px and the splitter keeps that size after expand, so checks
            # interleaved with collapses measure leftover state instead of
            # the launch layout — reproduced: the Review pane fully
            # floor-locked [174,104] by an earlier section's collapse on
            # one tree while another had slack, and a slider at its floor
            # with zero slack cannot be moved by ANY value.)
            for section_key in NEW_SECTIONS:
                entry = window._section_registry.get(section_key)
                check(entry is not None, f"{key}: section {section_key!r} missing from registry")
                if entry is None:
                    continue
                section, splitter, index = entry
                # Extra Voices is hidden by design until a cast beyond A/B
                # exists; render it anyway for the geometry/slider checks.
                hidden_by_design = section_key == "extra_voices"
                if hidden_by_design:
                    check(
                        not section.isVisible(),
                        f"{key}: extra_voices unexpectedly visible without extra cast",
                    )
                    section.show()
                    app.processEvents()
                check(section.isVisible(), f"{key}: {section_key} not visible when shown")
                geo = section.geometry()
                check(
                    geo.width() > 20 and geo.height() > 20,
                    f"{key}: {section_key} degenerate geometry {geo.size().toTuple()}",
                )
                check(
                    find_header(section) == TITLES[section_key],
                    f"{key}: {section_key} header title not rendered (found {find_header(section)!r})",
                )
                # Slider moves splitter space — for VISIBLE panes only; the
                # redistributer deliberately skips hidden panes (design pin
                # named in the module docstring). Drive BOTH directions: a
                # single request has a no-op regime in each branch — 75%
                # matches the geometry when the pane already sits at ~75%
                # (the slider only re-syncs from splitterMoved, i.e. user
                # drags), and 30% clamps silently to the pane's content
                # floor (the Review table pins 140px + chrome = 174). A
                # wired slider moves the splitter in at least one
                # direction; a dead one (unwired signal, unattached
                # splitter) moves it in neither.
                slider = section.findChild(QSlider)
                check(slider is not None, f"{key}: {section_key} has no size slider")
                if slider is not None and not section.isHidden():
                    sizes_before = splitter.sizes()
                    before = sizes_before[index]
                    for target in (30, 75):
                        slider.setValue(target)
                        app.processEvents()
                        if splitter.sizes()[index] != before:
                            break
                    after = splitter.sizes()[index]
                    check(
                        after != before,
                        f"{key}: {section_key} slider did not move splitter space "
                        f"(started {before}; setValue 30 and 75 were both no-ops)",
                    )
                    # Put the launch geometry back so this section's drive
                    # cannot squeeze the next section's check (and the
                    # persistence check below measures the launch layout).
                    splitter.setSizes(sizes_before)
                    app.processEvents()
                if hidden_by_design:
                    section.hide()
                    app.processEvents()

            # Pass 2: collapse clips each section to its header and restore
            # brings it back — size-independent (a maxHeight clip), so it
            # may run after the geometry checks are done.
            for section_key in NEW_SECTIONS:
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

            # The review table renders rows inside its new chrome. Populated
            # first: an EMPTY table legitimately renders a zero-width vertical
            # header offscreen (no rows to number).
            window.table.setRowCount(3)
            for row in range(3):
                window.table.setItem(row, 2, QTableWidgetItem("sample narration line"))
            app.processEvents()
            vp = window.table.viewport().geometry()
            check(
                vp.width() > 100 and vp.height() > 40,
                f"{key}: review table viewport degenerate {vp.size().toTuple()}",
            )
            window.table.setRowCount(0)
            app.processEvents()

            # The new paths splitter persists and restores on a fresh window.
            window._paths_splitter.setSizes([420, 900])
            window._persist_workspace_layout()
            saved = json.loads(settings_file.read_text(encoding="utf-8"))
            check(
                len(saved.get("splitters", {}).get("paths", [])) == 2,
                f"{key}: paths splitter sizes not persisted ({saved.get('splitters', {}).get('paths')})",
            )
            _destroy_window(app, window)

            reopened = _build_window(app)
            check(
                len(reopened._paths_splitter.sizes()) == 2 and sum(reopened._paths_splitter.sizes()) > 500,
                f"{key}: paths splitter did not restore ({reopened._paths_splitter.sizes()})",
            )
            _destroy_window(app, reopened)
            if len(failures) == failures_before:
                notes.append(f"{key}: OK - sections render, sliders move, collapse clips, layout round-trips")
            else:
                notes.append(f"{key}: FAILED - see the list below")

        print("\n".join(f"  {note}" for note in notes))
    finally:
        # The config root is throwaway; cleaning it up is courtesy (a crash
        # that skips this leaves only an inert directory in /tmp). No real
        # settings exist to restore — that is the point of the redirect.
        shutil.rmtree(_CONFIG_ROOT, ignore_errors=True)

    if failures:
        print("CERTIFICATION FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"All good across all {len(THEMES)} themes: the three chrome'd sections render correctly offscreen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
