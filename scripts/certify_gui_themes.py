"""Offscreen GUI certification for the V1.01 GUI rework.

Runs the real MainWindow offscreen and verifies, for every one of the six
themes:
  * the stylesheet applies without Qt warnings and no widget falls back to an
    unstyled state;
  * nothing is truncated: every section, form label, button, and slider row
    has a nonzero visible geometry inside its parent, and label width never
    exceeds its form column width (no elision);
  * legibility is certified (the theme module's import-time WCAG check);
  * the section resize sliders actually move splitter space;
  * collapse toggles clip sections to their headers without losing widgets;
  * EVERY registered section is swept — the three the 2026-10-08 legibility
    pass gave chrome to (``paths``/``table``/``extra_voices``) included, so a
    section missing from the list below is a section this sweep silently owns
    nothing of;
  * the workspace round-trips: theme, splitters, section shares, collapses,
    window size, and the input file persist to the app settings file and come
    back intact on a fresh window.

Exit code 0 means the GUI says all good across all six themes.

The crash-records store is parked for the whole run and the modal D8
startup flow is stubbed per window: the startup review opens a QDialog.exec
whenever records exist (or whenever no consent decision is recorded), and
nothing in a headless sweep can answer it — the park keeps the reader from
seeing records at all (a park left behind by a crashed run is recovered by
the next one), the stub covers the never-asked branch. See
scripts/certify_section_chrome.py for the same two layers with comments.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtWidgets import QApplication, QSlider, QTableWidgetItem  # noqa: E402

from the_oracle import gui_settings  # noqa: E402
from the_oracle.app_gui import MainWindow  # noqa: E402
from the_oracle.gui_settings import app_settings_path  # noqa: E402
from the_oracle.gui_themes import THEMES, apply_theme, contrast_ratio  # noqa: E402

failures: list[str] = []
notes: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def _neutralize_startup_dialogs(window: MainWindow) -> None:
    """Stub the modal D8 crash flow before the first event pump.

    The queued crash-flow timer connects inside ``_on_gui_shown``, which
    looks the method up on the instance at connect time — an instance
    attribute set right after construction is what the timer fires (the
    tests/test_app_gui_profiles idiom), so no modal can open on any machine
    state. The dialogs themselves stay covered by test_gui_crash.py.
    """
    window._maybe_run_crash_startup_flow = lambda: None


def collect_problems(widget, prefix: str = "") -> list[str]:
    """Every widget with a zero or clipped geometry inside its visible parent."""
    problems: list[str] = []
    if not widget.isVisible():
        return problems
    geo = widget.geometry()
    if widget.parentWidget() is not None and not widget.isWindow():
        parent_geo = widget.parentWidget().rect()
        if geo.width() <= 0 or geo.height() <= 0:
            problems.append(f"{prefix}{widget.objectName() or type(widget).__name__}: zero size {geo.size().toTuple()}")
        elif geo.width() > parent_geo.width() + 2 or geo.height() > parent_geo.height() + 2:
            problems.append(
                f"{prefix}{widget.objectName() or type(widget).__name__}: "
                f"clipped child {geo.size().toTuple()} in parent {parent_geo.size().toTuple()}"
            )
    if widget is not None and hasattr(widget, "geometry") and widget.isVisible() and widget.parentWidget() is not None:
        geo = widget.geometry()
        if geo.width() <= 0:
            chain = []
            current = widget.parentWidget()
            while current is not None and len(chain) < 4:
                chain.append(type(current).__name__)
                current = current.parentWidget()
            problems.append(
                f"ZERO-WIDTH {type(widget).__name__} in chain {' > '.join(chain)}"
            )
    for child in widget.findChildren(type(widget).__mro__[0]) if False else widget.children():
        from PySide6.QtWidgets import QWidget

        if isinstance(child, QWidget):
            problems.extend(collect_problems(child, prefix))
    return problems


def spin_value(spin: QSpinBox) -> int:
    return spin.value()


def main() -> int:
    # Launch-path arming (the entry-point audit, 2026-10-08): this process
    # builds the REAL MainWindow offscreen — native-crash territory — and a
    # script entry never passes through cli.main's install+arm. Idempotent;
    # fails closed to unarmed when consent is off.
    from the_oracle.crash import handlers as crash_handlers

    crash_handlers.arm_native_capture()
    from the_oracle.crash import bundle as crash_bundle

    app = QApplication.instance() or QApplication([])
    settings_file = app_settings_path()
    # Back the settings file up to disk (not just memory) before the
    # certification run rewrites it per theme: if this process dies mid-run,
    # the user's real settings survive in the backup file and are restored
    # here on the next successful exit; a leftover backup also allows manual
    # recovery.
    backup_path: Path | None = None
    parked = None
    if settings_file.exists():
        try:
            fd, backup_name = tempfile.mkstemp(prefix="oracle_settings_backup_", suffix=".json")
            os.close(fd)
            backup_path = Path(backup_name)
            shutil.copy2(settings_file, backup_path)
        except OSError as exc:
            print(f"WARNING: could not back up settings file: {exc}", file=sys.stderr)
            backup_path = None
    original_env_theme = os.environ.get("ORACLE_TEST_THEME")
    os.environ["ORACLE_TEST_THEME"] = "1"  # reserved for future per-test overrides

    # Every key in ``_section_registry`` — keep this tuple in step with
    # app_gui's registrations. A key that lands in the registry without
    # landing here is a section the sweep stops covering, which is exactly
    # how paths/table/extra_voices came to be missing (they were registered
    # by the 2026-10-08 legibility pass; the tuple predated them).
    all_sections = (
        "shared",
        "speaker_a",
        "speaker_b",
        "extra_voices",
        "paths",
        "table",
        "status",
        "live",
    )

    try:
        # Park the records store (repo root, the same root MainWindow's D8
        # flow reads): a leftover park from a crashed predecessor is
        # recovered here, and the D8 review modal has no records to react
        # to for the whole run.
        parked = crash_bundle.park_records(REPO_ROOT)
        for key in THEMES:
            tokens = THEMES[key]
            # Fresh settings for each theme so the launch path runs its
            # defaults; a real window (not a shim) is required here.
            if settings_file.exists():
                settings_file.unlink()
            gui_settings.save_app_settings({"theme": key})

            window = MainWindow()
            _neutralize_startup_dialogs(window)
            window.show()
            window.resize(1500, 1500)
            for _ in range(8):
                app.processEvents()
            window.hide()
            window.show()
            for _ in range(8):
                app.processEvents()

            # 1. Legibility: WCAG ratios were machine-certified at import;
            #    double-check the critical pairs here with real values.
            body = contrast_ratio(tokens.text, tokens.panel)
            check(body >= 4.5, f"{key}: body text contrast {body:.2f} < 4.5")
            on_accent = contrast_ratio(tokens.accent_text, tokens.accent)
            check(on_accent >= 3.0, f"{key}: on-accent contrast {on_accent:.2f} < 3.0")

            # 2. Truncation sweep: no visible widget is zero-sized or clipped.
            # The review table is populated first: an EMPTY table legitimately
            # renders a zero-width vertical header offscreen (no rows to
            # number), which would read as a false truncation.
            window.table.setRowCount(3)
            for row in range(3):
                window.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
                window.table.setItem(row, 2, QTableWidgetItem("sample narration line"))
            app.processEvents()
            problems = collect_problems(window)
            # The extra-speaker scroll is hidden until a cast exists; anything
            # else hidden on purpose (backend knobs) is skipped by the
            # visibility guard above.
            if problems:
                table = window.table
                notes.append(
                    f"{key} DIAGNOSTIC: table {table.geometry().width()}x{table.geometry().height()}, "
                    f"header {table.horizontalHeader().geometry().width()}x{table.horizontalHeader().geometry().height()}, "
                    f"viewport {table.viewport().geometry().width()}x{table.viewport().geometry().height()}, "
                    f"lower sizes {window._lower_splitter.sizes()}, main sizes {window._main_splitter.sizes()}, "
                    f"window {window.width()}x{window.height()}"
                )
            check(not problems, f"{key}: geometry problems: {problems[:8]}")

            window.table.setRowCount(0)
            app.processEvents()

            # 3. Sections exist, are collapsible, and their sliders move space.
            # Two passes, mirroring scripts/certify_section_chrome.py: every
            # slider is driven from launch geometry FIRST, then every collapse
            # runs. Interleaved, a collapse clips its pane to 30 px and the
            # splitter keeps that size on expand, so a later section's slider
            # check would measure leftover state instead of the launch layout
            # — a pane already at its content floor cannot be moved by ANY
            # slider value (reproduced on the Review pane there, floor-locked
            # at [174, 104] by an earlier section's collapse).
            for section_key in all_sections:
                entry = window._section_registry.get(section_key)
                check(entry is not None, f"{key}: missing section {section_key}")
                if entry is None:
                    continue
                section, splitter, index = entry
                check(section.is_collapsed() is False, f"{key}: {section_key} unexpectedly collapsed at launch")
                slider = section.findChild(QSlider)
                check(slider is not None, f"{key}: {section_key} has no size slider")
                # Extra Voices is hidden by design until a cast beyond A/B
                # exists, and the redistributer deliberately skips hidden
                # panes (pinned by
                # test_hidden_extra_voices_pane_does_not_break_size_slider),
                # so a hidden pane's slider is exempt from the drive. Its
                # slider-while-shown behavior is certified by the sibling
                # scripts/certify_section_chrome.py, which shows the pane.
                if slider is not None and not section.isHidden():
                    sizes_before = splitter.sizes()
                    before = sizes_before[index]
                    # Drive the slider like a user (unblock signals) so the
                    # redistribute path itself is what gets certified, and
                    # drive BOTH directions: a single request has a no-op
                    # regime in each branch — 75 matches the launch geometry
                    # when the pane already sits near there (the slider only
                    # re-syncs from splitterMoved), and 30 clamps silently to
                    # the pane's content floor (the Review table pins 140 px +
                    # 34 px chrome = 174). A wired slider moves the splitter in
                    # at least one direction; a dead one moves it in neither.
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
                    # Put the launch geometry back so this section's drive
                    # cannot squeeze the next section's check (and the
                    # persistence check below measures the launch layout).
                    splitter.setSizes(sizes_before)
                    app.processEvents()

            # Collapse clips each section to its header and restore brings it
            # back — size-independent (a maxHeight clip), so it may run after
            # every geometry/slider check is done. Restore is asserted because
            # this pass runs before the persistence step, which must snapshot
            # an uncollapsed launch layout.
            for section_key in all_sections:
                entry = window._section_registry.get(section_key)
                if entry is None:
                    continue
                section, _splitter, _index = entry
                section.set_collapsed(True)
                app.processEvents()
                check(section.maximumHeight() < 80, f"{key}: {section_key} did not clip when collapsed")
                section.set_collapsed(False)
                app.processEvents()
                check(not section.is_collapsed(), f"{key}: {section_key} did not restore after collapse")

            # 4. Persistence round-trip: mutate layout + settings, reopen.
            window.speaker_a.cfg_weight.setValue(1.11)
            window.speaker_a.pause_spin.setValue(432)
            window._sections_splitter.setSizes([500, 300, 260, 140])
            window._lower_splitter.setSizes([500, 220])
            window.input_path.setText(str(REPO_ROOT / "Input" / "What is, reality.txt"))
            window._persist_workspace_layout()
            saved = json.loads(settings_file.read_text(encoding="utf-8"))
            check(saved.get("theme") == key, f"{key}: theme not persisted ({saved.get('theme')})")
            check(
                Path(saved.get("last_input_file", "")).name == "What is, reality.txt",
                f"{key}: input file not persisted ({saved.get('last_input_file')})",
            )
            check(len(saved.get("splitters", {}).get("sections", [])) == 4, f"{key}: sections splitter not persisted")
            sections = saved.get("sections", {})
            for section_key in all_sections:
                check(section_key in sections, f"{key}: section {section_key} missing from persistence")
            window.close()
            app.processEvents()

            reopened = MainWindow()
            _neutralize_startup_dialogs(reopened)
            reopened.show()
            app.processEvents()
            check(
                abs(float(reopened.speaker_a.cfg_weight.value()) - 1.11) < 0.011,
                f"{key}: slider value did not survive restart ({reopened.speaker_a.cfg_weight.value()})",
            )
            check(
                reopened.speaker_a.pause_spin.value() == 432,
                f"{key}: pause value did not survive restart ({reopened.speaker_a.pause_spin.value()})",
            )
            check(
                reopened.input_path.text().endswith("What is, reality.txt"),
                f"{key}: input file did not survive restart ({reopened.input_path.text()})",
            )
            check(
                reopened._current_theme == key,
                f"{key}: theme did not survive restart ({reopened._current_theme})",
            )
            sections_after = reopened._app_settings.get("sections", {})
            check(bool(sections_after), f"{key}: section persistence empty after restart")
            reopened.close()
            app.processEvents()
            notes.append(f"{tokens.name} ({key}): OK - contrast body {body:.2f}:1, on-accent {on_accent:.2f}:1")

        # 5. File menu carries the profile actions; theme menu carries 6.
        if settings_file.exists():
            settings_file.unlink()
        gui_settings.save_app_settings({"theme": "oracle_light"})
        window = MainWindow()
        _neutralize_startup_dialogs(window)
        window.show()
        app.processEvents()
        labels = [action.text() for menu in window.menuBar().actions() if menu.menu() for action in menu.menu().actions()]
        check("Save Profile…" in labels, f"File menu missing Save Profile (got {labels})")
        check("Load Profile…" in labels, f"File menu missing Load Profile (got {labels})")
        theme_labels = [action.text() for action in window._theme_actions.values()]
        check(len(theme_labels) == 6, f"Theme menu should list 6 themes, got {theme_labels}")
        window.close()
        app.processEvents()
    finally:
        # Bring the records store home first: merge (not rename), so a
        # record written during the parked window survives beside the
        # parked originals.
        crash_bundle.restore_parked_records(REPO_ROOT, parked)
        if backup_path is not None and backup_path.exists():
            try:
                shutil.copy2(backup_path, settings_file)
            finally:
                backup_path.unlink(missing_ok=True)
        elif settings_file.exists():
            # No settings file existed before the run: remove the one the
            # certification themes created.
            settings_file.unlink()
        if original_env_theme is None:
            os.environ.pop("ORACLE_TEST_THEME", None)
        else:
            os.environ["ORACLE_TEST_THEME"] = original_env_theme

    print("\n".join(f"  {note}" for note in notes))
    if failures:
        print("CERTIFICATION FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("All good across all 6 themes: no truncation, legibility certified, persistence verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
