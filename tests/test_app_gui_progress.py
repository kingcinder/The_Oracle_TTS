import os

import pytest

from PySide6.QtWidgets import QApplication

from the_oracle.app_gui import RenderProgressDialog
from the_oracle.pipeline import RenderProgress

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_progress_dialog_updates_and_resets(qt_app) -> None:
    dialog = RenderProgressDialog(title="Test")

    # initial state is reset
    assert dialog.progress_bar.value() == 0
    assert "Starting" in dialog.stage_label.text()

    progress = RenderProgress(
        stage="Loading model",
        detail="step",
        current_step=2,
        total_steps=4,
        current_segment=1,
        total_segments=4,
        elapsed_seconds=0.5,
    )
    dialog.update_progress(progress)
    assert dialog.progress_bar.value() == 50
    assert "Loading model" in dialog.stage_label.text()

    done = RenderProgress(
        stage="Complete",
        detail="done",
        current_step=4,
        total_steps=4,
        current_segment=4,
        total_segments=4,
        elapsed_seconds=1.0,
    )
    dialog.update_progress(done)
    assert dialog.progress_bar.value() == 100

    # a fresh dialog starts clean (no stale progress)
    dialog2 = RenderProgressDialog(title="Test2")
    assert dialog2.progress_bar.value() == 0
    assert "Starting" in dialog2.stage_label.text()
