"""U4.2 candidate-state repro driver (debug session artifact).

Drives the REAL MainWindow choreography in-process with the deterministic
smoke engines patched at the pipeline module level, the same substitution
run_deterministic_smoke_render uses. Each pass is a fresh window+loop;
candidate states from root-causes.md are driven in order of suspicion:

  pass 1  render -> render                (two renders, 86 s pattern compressed)
  pass 2  render -> preview-playback active -> render   (13:49:25 signature)
  pass 3  preview twice (player reuse) -> render
  pass 4  render -> user-stop -> render   (progress-dialog close path)
"""
from __future__ import annotations

import faulthandler
import os
import shutil
import sys
import traceback
from pathlib import Path

REPO = Path("/home/cody/Documents/the oracle tts")
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Fully isolated user state; also exercises crash-report capture end to end.
os.environ["XDG_CONFIG_HOME"] = str(REPO / "build/u42/config")
os.environ["APPDATA"] = str(REPO / "build/u42/config")

PASS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
REPO2 = Path(sys.argv[2]) if len(sys.argv) > 2 else None

from unittest.mock import patch  # noqa: E402

from the_oracle import pipeline as pipeline_mod  # noqa: E402
from the_oracle.smoke import (  # noqa: E402
    _DeterministicChatterboxEngine,
    _SmokeEmotionClassifier,
    _write_reference,
)
from the_oracle.app_paths import ensure_repo_default_paths  # noqa: E402
from the_oracle.crash import handlers as crash_handlers  # noqa: E402


def wait_until(pred, timeout_s: float, what: str) -> None:
    from PySide6.QtCore import QDeadlineTimer, QEventLoop

    loop = QEventLoop()
    deadline = QDeadlineTimer(int(timeout_s * 1000))
    deadline.setRemainingTime(int(timeout_s * 1000))
    import time as _t
    end = _t.monotonic() + timeout_s

    def poll() -> None:
        if pred():
            loop.quit()
        elif _t.monotonic() > end:
            loop.quit()

    timer = loop_start_timer(50, poll)
    loop.exec()
    timer.stop()
    if not pred():
        raise TimeoutError(f"timed out waiting for: {what}")


def loop_start_timer(ms, cb):
    from PySide6.QtCore import QTimer

    t = QTimer()
    t.setInterval(ms)
    t.timeout.connect(cb)
    t.start()
    return t


def spin(ms: float) -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(int(ms), loop.quit)
    loop.exec()


def write_input(path: Path) -> None:
    path.write_text(
        "A: The tide keeps its own ledger, and tonight it is calling in every debt.\n"
        "B: Then let it call. I have paid worse collectors for less wisdom.\n"
        "A: You say that every season, and every season the shore erodes a little more.\n"
        "B: Erosion is only the coast remembering where it started.\n",
        encoding="utf-8",
    )


def build_window(repo: Path):
    import the_oracle.app_gui as app_gui

    # Pre-seed crash-report consent at the repo root (the U1 contract: consent
    # is the only gate on the faulthandler net) so the candidate-state passes
    # run with native-crash capture LIVE, exactly like the fatal real session.
    from the_oracle.crash import consent as crash_consent

    crash_consent.write_consent(repo, True)
    from the_oracle.crash import handlers as crash_handlers

    # Mirror what a consented launch is supposed to do (design §3): arm the
    # native net for this scripted session.
    crash_handlers.enable_faulthandler_catch(repo)
    paths = ensure_repo_default_paths(repo)
    # XDG_CONFIG_HOME points at an empty dir, so app settings are absent:
    # defaults apply and the backend combo lands on PyTorch (index 0).
    window = app_gui.MainWindow()
    print(
        f"[setup] faulthandler net armed={crash_handlers._STATE.get('native_dump_path') is not None}",
        flush=True,
    )
    window.resize(1280, 800)
    window.show()
    return window, paths


def drive_pass(pass_no: int, repo: Path) -> str:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window, paths = build_window(repo)

    # A real user waits for the startup prewarm to settle (the Render button
    # is disabled until then); the guard inside render_project enforces it.
    wait_until(lambda: window._prewarm_state in {"ready", "failed"}, 60.0, "prewarm settle")

    # --- Give every speaker a reference voice, like a real user would pick
    # one from the Seashells library before rendering. ---
    for speaker, group in window._all_speaker_groups().items():
        ref = repo / "Seashells" / f"u42_ref_{speaker}.wav"
        _write_reference(ref, 220.0 if speaker == "A" else 330.0)
        group.reference_path.setText(str(ref))

    # --- Analyze (the GUI's own slot, with its own pre-action guards) ---
    window.input_path.setText(str(repo / "Input" / "u42_dialogue.txt"))
    write_input(window.input_path.text() and Path(window.input_path.text()))
    window.prepare_project()
    wait_until(lambda: window.plan is not None, 120.0, "analyze plan ready")
    rows = window.plan.utterances
    assert rows, "analyze produced no utterances"
    print(f"[pass {pass_no}] analyzed: {len(rows)} rows", flush=True)

    # Race-free completion tracking: wrap the GUI's own finish/fail slots so
    # a fast render (hot cache) can complete between two of our polls and
    # still be counted. Also keep every artifact inside the sandbox repo.
    window._u42_done = 0
    window._u42_failed = 0
    window._u42_prev_done = 0
    window._u42_prev_failed = 0
    orig_finish = window._finish_render
    orig_fail = window._fail_render

    def _finish(plan, path):
        window._u42_done += 1
        orig_finish(plan, path)

    def _fail(payload, msg=None):
        window._u42_failed += 1
        orig_fail(payload, msg)

    window._finish_render = _finish
    window._fail_render = _fail
    window.outdir_path.setText(str(repo / "Output"))

    def render_once(tag: str) -> None:
        window._u42_prev_done, window._u42_prev_failed = window._u42_done, window._u42_failed
        print(
            f"[pass {pass_no}] pre-render: prewarm={window._prewarm_state} "
            f"plan={'yes' if window.plan else 'no'} rows={len(window.plan.utterances) if window.plan else 0} "
            f"preview_worker={window.preview_worker is not None} render_worker={window.render_worker is not None}",
            flush=True,
        )
        window.output_name.setText(f"u42_{tag}.flac")
        window.render_project()
        wait_until(
            lambda: (window._u42_done + window._u42_failed) > (window._u42_prev_done + window._u42_prev_failed),
            180.0,
            f"{tag}: render completed (or failed)",
        )
        if window._u42_failed > window._u42_prev_failed:
            raise AssertionError(f"{tag}: render FAILED (see [modal] line)")
        print(f"[pass {pass_no}] render {tag} complete", flush=True)

    orig_pfinish = window._finish_preview
    orig_pfail = window._fail_preview

    def _pfinish(row, path):
        window._u42_done += 1
        orig_pfinish(row, path)

    def _pfail(msg):
        window._u42_failed += 1
        orig_pfail(msg)

    window._finish_preview = _pfinish
    window._fail_preview = _pfail

    def preview_once(tag: str) -> None:
        window._u42_prev_done, window._u42_prev_failed = window._u42_done, window._u42_failed
        window.preview_utterance(0)
        wait_until(
            lambda: (window._u42_done + window._u42_failed) > (window._u42_prev_done + window._u42_prev_failed),
            120.0,
            f"{tag}: preview completed (or failed)",
        )
        if window._u42_failed > window._u42_prev_failed:
            raise AssertionError(f"{tag}: preview FAILED (see [modal] line)")
        state = getattr(window.player, "playbackState", lambda: None)()
        print(f"[pass {pass_no}] preview {tag} done; player state={state}", flush=True)

    if pass_no == 1:
        render_once("r1")
        spin(400)
        render_once("r2")
    elif pass_no == 2:
        render_once("r1")
        preview_once("p1")
        spin(400)  # playback ACTIVE at render click — the fatal pattern
        render_once("r2")
    elif pass_no == 3:
        preview_once("p1")
        preview_once("p2")
        render_once("r1")
    elif pass_no == 4:
        render_once("r1")
        render_once("r2")
        window.close()
        wait_until(lambda: not window.isVisible(), 10.0, "window closed")
        print(f"[pass {pass_no}] closed with workers drained", flush=True)
    else:
        raise SystemExit(f"unknown pass {pass_no}")

    # Drain queued cleanup slots before the window object dies.
    app.processEvents()
    window.deleteLater()
    app.processEvents()
    return str(repo)


def force_in_process_workers() -> None:
    """Force Render/Preview workers to run in-process.

    The real GUI spawns isolated render children because loading
    Chatterbox/PyTorch inside the Qt process segfaults with the REAL engine.
    The child would import an unpatched pipeline and try to load/download the
    real model, which is exactly what the smoke path exists to avoid
    (deterministic, offline). Forcing run_in_subprocess=False keeps every
    other piece of the choreography real: the QThread workers, their signal
    wiring, and the GUI-side slots.
    """
    import the_oracle.gui_render as gui_render

    orig_render_init = gui_render.RenderWorker.__init__

    def render_init(self, plan, settings, **kwargs):
        kwargs["run_in_subprocess"] = False
        orig_render_init(self, plan, settings, **kwargs)

    gui_render.RenderWorker.__init__ = render_init

    orig_preview_init = gui_render.PreviewWorker.__init__

    def preview_init(self, utterance, profile, model_variant, device_mode, **kwargs):
        kwargs["run_in_subprocess"] = False
        orig_preview_init(self, utterance, profile, model_variant, device_mode, **kwargs)

    gui_render.PreviewWorker.__init__ = preview_init


def auto_dismiss_modals() -> None:
    """Auto-answer every QMessageBox so a failure modal can't deadlock the
    scripted session (a modal exec spins its own loop and never returns)."""
    import the_oracle.app_gui as app_gui
    from PySide6.QtWidgets import QMessageBox

    def answer(name, ret):
        def f(*a, **k):
            args = [x for x in a if isinstance(x, str)]
            print(f"[modal] {name}: {' | '.join(args[:3])}", flush=True)
            return ret

        return f

    ok = QMessageBox.StandardButton.Ok
    app_gui.QMessageBox.critical = staticmethod(answer("critical", ok))
    app_gui.QMessageBox.information = staticmethod(answer("information", ok))
    app_gui.QMessageBox.warning = staticmethod(answer("warning", ok))
    QMessageBox.exec = lambda self, *a, **k: (print("[modal] exec auto-dismissed", flush=True), ok)[1]


def main() -> int:
    repo = REPO2 or (REPO / "build/u42/repo")
    if REPO2 is None:
        if repo.exists():
            shutil.rmtree(repo)
        ensure_repo_default_paths(repo)

    with (
        patch.object(pipeline_mod, "ChatterboxEngine", _DeterministicChatterboxEngine),
        patch.object(pipeline_mod, "GoEmotionsClassifier", _SmokeEmotionClassifier),
    ):
        force_in_process_workers()
        auto_dismiss_modals()
        try:
            where = drive_pass(PASS, repo)
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            print(f"PASS {PASS} FAILED (python-level, no native crash)", flush=True)
            return 1
    print(f"PASS {PASS} OK repo={where}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
