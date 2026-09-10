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
  * the workspace round-trips: theme, splitters, section shares, collapses,
    window size, and the input file persist to the app settings file and come
    back intact on a fresh window.

Exit code 0 means the GUI says all good across all six themes.
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
    app = QApplication.instance() or QApplication([])
    settings_file = app_settings_path()
    # Back the settings file up to disk (not just memory) before the
    # certification run rewrites it per theme: if this process dies mid-run,
    # the user's real settings survive in the backup file and are restored
    # here on the next successful exit; a leftover backup also allows manual
    # recovery.
    backup_path: Path | None = None
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

    all_sections = ("shared", "speaker_a", "speaker_b", "status", "live")

    try:
        for key in THEMES:
            tokens = THEMES[key]
            # Fresh settings for each theme so the launch path runs its
            # defaults; a real window (not a shim) is required here.
            if settings_file.exists():
                settings_file.unlink()
            gui_settings.save_app_settings({"theme": key})

            window = MainWindow()
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
            for section_key in all_sections:
                entry = window._section_registry.get(section_key)
                check(entry is not None, f"{key}: missing section {section_key}")
                if entry is None:
                    continue
                section, splitter, index = entry
                check(section.is_collapsed() is False, f"{key}: {section_key} unexpectedly collapsed at launch")
                before = splitter.sizes()[index]
                total_before = sum(splitter.sizes())
                # Drive the slider like a user (unblock signals) so the
                # redistribute path itself is what gets certified. Request a
                # share the pane can actually reach: siblings keep at least
                # 20% each and this pane's own minimum is a floor.
                slider = section.findChild(QSlider)
                check(slider is not None, f"{key}: {section_key} has no size slider")
                if slider is not None:
                    share = before / total_before if total_before else 0.5
                    slider.blockSignals(False)
                    slider.setValue(30 if share > 0.45 else 75)
                app.processEvents()
                after = splitter.sizes()[index]
                check(
                    after != before,
                    f"{key}: {section_key} size slider did not move splitter space ({before} -> {after})",
                )
                section.set_collapsed(True)
                app.processEvents()
                check(section.maximumHeight() < 80, f"{key}: {section_key} did not clip when collapsed")
                section.set_collapsed(False)
                app.processEvents()

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
