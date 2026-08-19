import os

import pytest

from PySide6.QtWidgets import QApplication

from the_oracle.app_gui import RenderProgressDialog
from the_oracle.pipeline import RenderProgress

pytestmark = pytest.mark.slow

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


def test_progress_dialog_shows_live_backend_panel(qt_app) -> None:
    """The live panel shows the active backend + GPU and cumulative synthesis
    time, and reset() clears it."""
    dialog = RenderProgressDialog(title="Render")

    dialog.update_progress(
        RenderProgress(
            stage="Rendering segment",
            detail="segment",
            current_step=1,
            total_steps=3,
            current_segment=1,
            total_segments=3,
            elapsed_seconds=3.0,
            backend="vulkan",
            device_label="AMD Radeon RX 5700 XT (RADV NAVI10)",
            synth_seconds_total=4.8,
            synth_seconds_latest=2.4,
        )
    )
    assert "Vulkan" in dialog.backend_label.text()
    assert "5700" in dialog.backend_label.text()
    assert "5s total" in dialog.synth_label.text()
    assert "2s" in dialog.synth_label.text()

    # PyTorch renders show the CPU device instead.
    dialog.update_progress(
        RenderProgress(
            stage="Rendering segment",
            detail="segment",
            current_step=2,
            total_steps=3,
            current_segment=2,
            total_segments=3,
            elapsed_seconds=4.0,
            backend="pytorch",
            device_label="CPU",
            synth_seconds_total=30.0,
            synth_seconds_latest=25.0,
        )
    )
    assert "PyTorch" in dialog.backend_label.text()
    assert "CPU" in dialog.backend_label.text()
    assert "30s total" in dialog.synth_label.text()

    dialog.reset()
    assert dialog.backend_label.text() == "Backend: ..."
    assert dialog.synth_label.text() == ""
