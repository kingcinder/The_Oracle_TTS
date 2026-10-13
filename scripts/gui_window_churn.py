"""Window-churn repro for the certifier's second-window native fault.

The 1.4.0 CI certifier logs (2026-10-13) show the fault landing in
``the_oracle.gui_tooltips.CtrlHoverHelp.eventFilter`` -> ``event.type()``
while a *later* MainWindow is being built, after an earlier one was torn
down and deleted. This driver reproduces that shape deterministically:

    build a real MainWindow -> show/pump -> close -> deleteLater -> flush
    DeferredDelete -> build the next one

and repeats it ``--passes`` times (default 12). ``faulthandler`` is armed
first, so a native fault prints the Python stack of the frame that died
instead of exiting with a bare ``Segmentation fault``.

Usage::

    .venv/bin/python scripts/gui_window_churn.py --passes 25

Exit status 0 means no fault in this run — it is *evidence for this machine*,
not proof of portability (the fault was reported on Windows/Qt 6.9 CI).
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Fully isolated user state: never touch the developer's real settings.
os.environ.setdefault("XDG_CONFIG_HOME", str(REPO / "build/gui-churn/config"))
os.environ.setdefault("APPDATA", str(REPO / "build/gui-churn/config"))

faulthandler.enable()

from PySide6.QtCore import QCoreApplication, QEvent, QObject  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


class CountingApp(QApplication):
    """A QApplication that records every Python-level event-filter install."""

    def __init__(self, argv: list[str]) -> None:
        super().__init__(argv)
        self.installs: list[QObject] = []

    def installEventFilter(self, obj: QObject) -> None:  # type: ignore[override]
        self.installs.append(obj)
        super().installEventFilter(obj)


def _neutralize(window) -> None:
    window._maybe_run_crash_startup_flow = lambda: None
    window._start_inference_wizard = lambda: None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--passes", type=int, default=12)
    parser.add_argument(
        "--park-object",
        action="store_true",
        help="leak one extra parentless QObject per pass (parentless-object churn)",
    )
    args = parser.parse_args(argv)

    from the_oracle.app_gui import MainWindow

    app = CountingApp([])
    parked: list[QObject] = []
    baseline_installs = len(app.installs)
    worst = 0.0
    for index in range(1, args.passes + 1):
        if args.park_object:
            parked.append(QObject())
        started = time.perf_counter()
        window = MainWindow()
        _neutralize(window)
        window.show()
        for _ in range(8):
            app.processEvents()
        # Second-half of pass 1 in the certifier: hide/show re-layout.
        window.hide()
        window.show()
        for _ in range(8):
            app.processEvents()
        elapsed = time.perf_counter() - started
        worst = max(worst, elapsed)
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
        print(
            f"pass {index}/{args.passes} ok ({elapsed * 1000:.0f} ms, "
            f"app-level filters {len(app.installs) - baseline_installs})",
            flush=True,
        )
    print(
        f"no fault in {args.passes} passes; worst build {worst * 1000:.0f} ms; "
        f"app-level event filters installed: {len(app.installs) - baseline_installs}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
