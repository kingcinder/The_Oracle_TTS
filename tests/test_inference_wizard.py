from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QWidget

from the_oracle.device_support import CUDADeviceInfo
from the_oracle.inference_wizard import InferenceSetupWizard, STAGES, hardware_summary


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _device(index: int, name: str, *, suitable: bool, torch_available: bool = True, gib: int = 8):
    return CUDADeviceInfo(
        index=index,
        name=name,
        vram_bytes=gib * 1024**3,
        torch_available=torch_available,
        suitable=suitable,
        reason="ready" if suitable else "below the Chatterbox memory floor",
    )


def test_tutorial_stages_are_dependency_ordered():
    keys = [stage.key for stage in STAGES]
    assert keys == ["discovery", "backend", "input", "model", "voices", "timing", "hybrid", "review"]
    assert STAGES[0].discovery is True
    assert all(not stage.discovery for stage in STAGES[1:])
    assert keys.index("voices") < keys.index("timing") < keys.index("hybrid") < keys.index("review")


def test_hardware_summary_explains_usable_and_rejected_devices():
    text = hardware_summary([
        _device(0, "Legacy NVIDIA", suitable=False, gib=1),
        _device(1, "Workstation NVIDIA", suitable=True),
    ])

    assert "Legacy NVIDIA" in text
    assert "not selectable" in text
    assert "below the Chatterbox memory floor" in text
    assert "Workstation NVIDIA" in text
    assert "available" in text
    assert "CPU / system DRAM" in text


def test_discovery_wizard_offers_cpu_and_disables_unsuitable_cuda(qt_app):
    main_window = QWidget()
    main_window.pytorch_device_combo = QComboBox(main_window)
    main_window.pytorch_device_combo.addItem("CPU / system DRAM", "cpu")
    wizard = InferenceSetupWizard(
        main_window,
        devices=[_device(0, "Old NVIDIA", suitable=False, gib=1)],
        mode="discovery",
    )
    try:
        assert wizard._stages[0].key == "discovery"
        assert wizard.device_picker.findData("cpu") >= 0
        cuda_index = wizard.device_picker.findData("cuda:0")
        assert cuda_index >= 0
        assert not wizard.device_picker.model().item(cuda_index).isEnabled()
        assert not wizard.hardware_summary.isHidden()
        assert not wizard.selection_box.isHidden()
    finally:
        wizard.close()
        main_window.deleteLater()


def test_discovery_wizard_exposes_default_folder_preferences(qt_app):
    main_window = QWidget()
    main_window.paths = type("Paths", (), {"input_dir": "/repo/Input", "output_dir": "/repo/Output"})()
    wizard = InferenceSetupWizard(main_window, devices=[], mode="discovery")
    try:
        assert wizard.input_folder_edit.text() == "/repo/Input"
        assert wizard.output_folder_edit.text() == "/repo/Output"
        assert wizard.remember_input_folder.isChecked() is True
        assert wizard.remember_output_folder.isChecked() is True
    finally:
        wizard.close()
        main_window.deleteLater()


def test_continue_reports_selected_cuda_device(qt_app):
    main_window = QWidget()
    main_window.pytorch_device_combo = QComboBox(main_window)
    seen: list[tuple[str, int | None]] = []
    wizard = InferenceSetupWizard(
        main_window,
        devices=[_device(1, "Good NVIDIA", suitable=True)],
        mode="discovery",
        on_selection=lambda mode, index: seen.append((mode, index)),
    )
    try:
        wizard.device_picker.setCurrentIndex(wizard.device_picker.findData("cuda:1"))
        wizard._continue()
        assert seen == [("cuda", 1)]
        assert wizard._finished_once is True
    finally:
        wizard.close()
        main_window.deleteLater()


def test_replay_modes_select_expected_stage_ranges(qt_app):
    main_window = QWidget()
    try:
        full = InferenceSetupWizard(main_window, mode="full")
        discovery = InferenceSetupWizard(main_window, mode="discovery")
        main = InferenceSetupWizard(main_window, mode="main")
        assert [stage.key for stage in full._stages] == [stage.key for stage in STAGES]
        assert [stage.key for stage in discovery._stages] == ["discovery"]
        assert not discovery.selection_box.isHidden()
        assert [stage.key for stage in main._stages] == [stage.key for stage in STAGES[1:]]
        assert main.selection_box.isHidden()
        full.close()
        discovery.close()
        main.close()
    finally:
        main_window.deleteLater()
