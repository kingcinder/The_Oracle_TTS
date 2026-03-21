"""
Lightweight GUI smoke harness for headless/manual runs.

Usage:
    QT_QPA_PLATFORM=offscreen PYTHONPATH=src python scripts/gui_smoke_prewarm.py

What it does:
- launches the GUI (offscreen)
- lets background prewarm start and sit idle briefly
- pokes a few widgets while the app is live
- exits after a short delay

This is a smoke check to ensure startup/prewarm does not freeze or crash the process.
"""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

# Ensure headless/offscreen operation by default for CI or manual smoke runs
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _pick_first_selectable(combo):
    for idx in range(combo.count()):
        item = combo.model().item(idx)
        if item is None or not item.isEnabled():
            continue
        if combo.itemData(idx):
            combo.setCurrentIndex(idx)
            return


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    from the_oracle.app_gui import MainWindow

    window = MainWindow()
    window.show()

    def interact_and_exit():
        _pick_first_selectable(window.speaker_a.reference_picker)
        _pick_first_selectable(window.speaker_b.reference_picker)
        window.variant_combo.setCurrentText("standard")
        window.input_path.setText("")
        QTimer.singleShot(1000, app.quit)

    QTimer.singleShot(750, interact_and_exit)
    app.exec()


if __name__ == "__main__":
    main()
