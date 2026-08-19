import os

import pytest

from PySide6.QtWidgets import QApplication

from the_oracle.app_gui import LivePanel, RenderProgressDialog
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


def test_progress_dialog_uses_time_weighted_fraction(qt_app) -> None:
    """When the pipeline supplies a time-weighted fraction, the bar follows it
    (even when the step math would disagree, e.g. one step of five that eats
    most of the wall time during model load). Without a fraction, the bar
    falls back to step math."""
    dialog = RenderProgressDialog(title="Render")

    # Model load: step 1 of 5, but 60% of the wall time — the fraction says 60%.
    dialog.update_progress(
        RenderProgress(
            stage="Loading model",
            detail="step",
            current_step=1,
            total_steps=5,
            current_segment=1,
            total_segments=5,
            elapsed_seconds=30.0,
            fraction=0.6,
        )
    )
    assert dialog.progress_bar.value() == 60

    # Same event without a fraction → step math (20%).
    dialog.update_progress(
        RenderProgress(
            stage="Loading model",
            detail="step",
            current_step=1,
            total_steps=5,
            current_segment=1,
            total_segments=5,
            elapsed_seconds=30.0,
        )
    )
    assert dialog.progress_bar.value() == 20

    # Completion always pins the bar to 100.
    dialog.update_progress(
        RenderProgress(
            stage="Complete",
            detail="done",
            current_step=5,
            total_steps=5,
            current_segment=5,
            total_segments=5,
            elapsed_seconds=60.0,
            fraction=1.0,
        )
    )
    assert dialog.progress_bar.value() == 100


def test_time_weighted_progress_advances_and_pins_completion() -> None:
    """Unit test for the pipeline's `_time_weighted_progress`: the fraction
    rises smoothly with elapsed time (through a 20s model load that is only
    one of many steps), never exceeds 99% before completion, and Complete pins
    to 1.0 with a zero ETA. The ETA appears only once the EWMA is measured
    and enough segments have landed."""
    from the_oracle.pipeline import _time_weighted_progress

    state = {"backend": "vulkan", "segments_total": 5, "segments_done": 0, "segment_avg": None}

    # During model load (30s elapsed, 0 of 5 segments done): the bar should
    # already be partway up instead of stuck at step 1 of N at ~12%.
    fraction, eta = _time_weighted_progress(state, elapsed=30.0, stage="Loading model")
    assert 0.2 < fraction < 0.9
    assert eta is None  # no measurement yet → no ETA

    # After the first segment lands, the fraction keeps climbing.
    state["segments_done"] = 1
    fraction2, _ = _time_weighted_progress(state, elapsed=40.0, stage="Rendering segment")
    assert fraction2 > fraction

    # With a measured average and enough completed segments, an ETA appears.
    state["segment_avg"] = 3.0
    state["segments_done"] = 3
    _fraction3, eta3 = _time_weighted_progress(state, elapsed=50.0, stage="Rendering segment")
    assert eta3 is not None and eta3 > 0

    # Completion pins to 1.0 / 0 ETA regardless of state.
    fraction4, eta4 = _time_weighted_progress(state, elapsed=50.0, stage="Complete")
    assert fraction4 == 1.0 and eta4 == 0.0


# ---------------------------------------------------------------------------
# LivePanel (persistent sidebar)
# ---------------------------------------------------------------------------


def test_live_panel_starts_idle(qt_app) -> None:
    panel = LivePanel()
    assert panel.progress_bar.value() == 0
    assert "idle" in panel.backend_label.text().lower()
    assert panel.stage_label.text() == ""
    assert panel.segment_label.text() == ""
    assert panel.eta_label.text() == ""


def test_live_panel_updates_from_progress(qt_app) -> None:
    panel = LivePanel()
    progress = RenderProgress(
        stage="Synthesizing",
        detail="utterance 3/10",
        current_step=3,
        total_steps=10,
        current_segment=3,
        total_segments=10,
        elapsed_seconds=12.5,
        eta_seconds=8.0,
        fraction=0.3,
        backend="vulkan",
        device_label="AMD RX 5700 XT",
        synth_seconds_total=9.2,
        synth_seconds_latest=1.5,
    )
    panel.update_from_progress(progress)
    assert panel.progress_bar.value() == 30
    assert "Vulkan" in panel.backend_label.text()
    assert "AMD RX 5700 XT" in panel.backend_label.text()
    assert "Synthesizing" in panel.stage_label.text()
    assert "3/10" in panel.segment_label.text()
    assert "12s" in panel.eta_label.text() or "0m 12s" in panel.eta_label.text()


def test_live_panel_reset_to_idle(qt_app) -> None:
    panel = LivePanel()
    progress = RenderProgress(
        stage="Synthesizing",
        detail="utterance 1/5",
        current_step=1,
        total_steps=5,
        current_segment=1,
        total_segments=5,
        elapsed_seconds=2.0,
    )
    panel.update_from_progress(progress)
    assert panel.progress_bar.value() == 20
    panel.set_idle()
    assert panel.progress_bar.value() == 0
    assert "idle" in panel.backend_label.text().lower()
    assert panel.stage_label.text() == ""


def test_live_panel_fractionless_fallback(qt_app) -> None:
    panel = LivePanel()
    progress = RenderProgress(
        stage="Loading",
        detail="model",
        current_step=5,
        total_steps=20,
        current_segment=0,
        total_segments=0,
        elapsed_seconds=1.0,
        # no fraction, no backend
    )
    panel.update_from_progress(progress)
    assert panel.progress_bar.value() == 25  # 5/20 * 100
    assert panel.backend_label.text() == "Backend: idle"  # unchanged
