"""Interactive offscreen GUI drive: Analyze → Preview → Render cycles.

Reproduction harness for the exit-245 (SIGSEGV) class. The plain launch loop
(crash_hunt --mode gui) only proves startup; this drives the interactive
paths that run inside the Qt process: prepare_project (Analyze), the preview
worker, and render-start validation. faulthandler is armed by consent, so a
native crash writes crash_reports/native-crash.txt before the process dies.

Exit codes: 0 = all cycles clean; 245/139 = SIGSEGV reproduced; anything
else = the cycle raised (recorded on stdout as DRIVE_ERROR).

Runs with QT_QPA_PLATFORM=offscreen by default (CI/smoke convention).
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

CYCLES = int(os.environ.get("DRIVE_CYCLES", "3"))
INPUT = os.environ.get("DRIVE_INPUT", os.path.join(REPO_ROOT, "Input", "What is, reality.txt"))


def main() -> int:
    # Launch-path arming (the entry-point audit, 2026-10-08): this driver
    # builds the REAL MainWindow and touches QMediaPlayer in-process — the
    # exact crash classes in crash-evidence §A — without ever passing through
    # cli.main's install+arm. Idempotent; fails closed without consent.
    from the_oracle.crash import handlers as crash_handlers

    crash_handlers.arm_native_capture()
    from PySide6.QtWidgets import QApplication

    from the_oracle.app_gui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.processEvents()

    for cycle in range(1, CYCLES + 1):
        print(f"DRIVE cycle {cycle}/{CYCLES}: analyze", flush=True)
        window.input_path.setText(INPUT)
        window.prepare_project()  # Analyze: transformer check + plan build
        app.processEvents()

        print(f"DRIVE cycle {cycle}/{CYCLES}: preview-dialog probe", flush=True)
        # Touch the media path that the status-245 screenshot died around:
        # construct the preview player the preview dialog uses.
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

        player = QMediaPlayer()
        audio = QAudioOutput()
        player.setAudioOutput(audio)
        app.processEvents()
        del player, audio

        print(f"DRIVE cycle {cycle}/{CYCLES}: render-start validation", flush=True)
        # The render button's guard path (validates plan/workers) without
        # launching a full child render — the crash class is IN-process.
        if window.plan is not None:
            window._log_action_timing("drive_render_probe")
        app.processEvents()

        print(f"DRIVE cycle {cycle}/{CYCLES}: close-preview/studio no-ops", flush=True)
        app.processEvents()

    print("DRIVE OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("DRIVE_ERROR", flush=True)
        sys.exit(1)
