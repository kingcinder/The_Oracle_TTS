import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from the_oracle.app_paths import OraclePaths, default_output_filename, ensure_repo_default_paths
from the_oracle.models.project import RenderPlan, Utterance, VoiceProfile, VoiceSettings
from the_oracle.pipeline import RenderSettings, SpeakerSettings
from the_oracle.gui_settings import load_app_settings, save_app_settings
from the_oracle.project_manifest import build_saved_project

pytestmark = pytest.mark.slow


class _FakeAudioOutput:
    def __init__(self, *_args, **_kwargs) -> None:
        pass


class _FakeMediaPlayer:
    def __init__(self, *_args, **_kwargs) -> None:
        self.audio_output = None
        self.source = None

    def setAudioOutput(self, output) -> None:
        self.audio_output = output

    def setSource(self, source) -> None:
        self.source = source

    def play(self) -> None:
        return None


class _FakePipeline:
    def __init__(self, **_kwargs) -> None:
        # Accept the GUI-safe dependency switches used by MainWindow._pipeline.
        # The fake only exercises widget wiring, not pipeline construction.
        pass

    def available_model_variants(self) -> list[str]:
        return ["standard", "multilingual", "turbo"]

    def supported_languages(self, model_variant: str = "standard") -> dict[str, str]:
        if model_variant == "multilingual":
            return {"en": "English", "es": "Spanish"}
        return {"en": "English"}


class _FakeChatterboxEngine:
    def __init__(self, variant: str = "standard", device: str | None = None) -> None:
        self.variant = variant
        self.device = device

    def supported_languages(self) -> dict[str, str]:
        if self.variant == "multilingual":
            return {"en": "English", "es": "Spanish"}
        return {"en": "English"}


class _FakeSignal:
    def connect(self, *_args, **_kwargs) -> None:
        return None

    def disconnect(self, *_args, **_kwargs) -> None:
        return None


class _FakeVulkanProbeThread:
    """Stand-in for VulkanDeviceProbeThread so GUI tests never spawn a real
    QThread or probe the actual audio.cpp binary."""

    def __init__(self, _parent=None) -> None:
        self.devices = _FakeSignal()
        self.failed = _FakeSignal()
        self.finished = _FakeSignal()
        self.started = False

    def start(self) -> None:
        self.started = True

    def wait(self, _timeout: int = 0) -> bool:
        return True

    def deleteLater(self) -> None:
        return None


class _FakeVulkanPreflightThread:
    """Stand-in for VulkanPreflightThread so GUI tests never spawn a real
    QThread or probe the actual audio.cpp binary."""

    def __init__(self, _parent=None, *, device_index=None) -> None:
        self.completed = _FakeSignal()
        self.failed = _FakeSignal()
        self.finished = _FakeSignal()
        self.device_index = device_index
        self.started = False

    def start(self) -> None:
        self.started = True

    def wait(self, _timeout: int = 0) -> bool:
        return True

    def deleteLater(self) -> None:
        return None


class _FakeVulkanSetupThread:
    """Stand-in for VulkanSetupThread so GUI tests never spawn a real QThread
    or run the build/download scripts. Mirrors the real thread's signal surface
    and records whether setup was requested/cancelled."""

    def __init__(self, _repo_root=None, _parent=None) -> None:
        self.progress = _FakeSignal()
        self.completed = _FakeSignal()
        self.failed = _FakeSignal()
        self.finished = _FakeSignal()
        self.repo_root = _repo_root
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def request_cancel(self) -> None:
        self.cancelled = True

    def wait(self, _timeout: int = 0) -> bool:
        return True

    def deleteLater(self) -> None:
        return None


class _FakeModelDownloadThread:
    """Stand-in for ModelDownloadThread so GUI tests never spawn a real
    QThread or shell out to the download script."""

    def __init__(self, _script=None, _parent=None) -> None:
        self.completed = _FakeSignal()
        self.failed = _FakeSignal()
        self.finished = _FakeSignal()
        self.script = _script
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def request_cancel(self) -> None:
        self.cancelled = True

    def wait(self, _timeout: int = 0) -> bool:
        return True

    def deleteLater(self) -> None:
        return None


class _FakeRenderWorker:
    """Stand-in for RenderWorker so GUI tests never spawn a real QThread.
    Records whether the worker was started (the render/preview path always
    calls start() on a fresh worker)."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.progress = _FakeSignal()
        self.completed = _FakeSignal()
        self.failed = _FakeSignal()
        self.finished = _FakeSignal()
        self.started = False

    def start(self) -> None:
        self.started = True


class _FakePreviewWorker:
    """Stand-in for PreviewWorker so GUI tests never spawn a real QThread.
    Records whether the worker was started (preview always calls start())."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.progress = _FakeSignal()
        self.completed = _FakeSignal()
        self.failed = _FakeSignal()
        self.finished = _FakeSignal()
        self.started = False

    def start(self) -> None:
        self.started = True


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _minimal_plan(paths: OraclePaths | None) -> RenderPlan:
    profile_a = VoiceProfile(name="Speaker A", speaker="A", reference_audio=[], engine_params=VoiceSettings())
    profile_b = VoiceProfile(name="Speaker B", speaker="B", reference_audio=[], engine_params=VoiceSettings())
    return RenderPlan(
        title="test",
        source_path="",
        output_dir=str(paths.output_dir) if paths else "",
        engine="chatterbox",
        correction_mode="moderate",
        metadata={"model_variant": "standard"},
        voice_profiles={"A": profile_a, "B": profile_b},
    )


def _build_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import the_oracle.app_gui as app_gui

    paths = ensure_repo_default_paths(tmp_path / "repo")
    monkeypatch.setattr(app_gui, "QAudioOutput", _FakeAudioOutput)
    monkeypatch.setattr(app_gui, "QMediaPlayer", _FakeMediaPlayer)
    monkeypatch.setattr(app_gui, "OraclePipeline", _FakePipeline)
    monkeypatch.setattr(app_gui, "ChatterboxEngine", _FakeChatterboxEngine)
    monkeypatch.setattr(app_gui, "VulkanDeviceProbeThread", _FakeVulkanProbeThread)
    monkeypatch.setattr(app_gui, "VulkanPreflightThread", _FakeVulkanPreflightThread)
    monkeypatch.setattr(app_gui, "VulkanSetupThread", _FakeVulkanSetupThread)
    monkeypatch.setattr(app_gui, "ModelDownloadThread", _FakeModelDownloadThread)
    monkeypatch.setattr(app_gui, "ensure_repo_default_paths", lambda _repo_root: paths)
    monkeypatch.setattr(app_gui, "default_voice_choices", lambda _repo_root: [])
    monkeypatch.setattr(app_gui, "load_recent_reference_paths", lambda limit=10: [])
    # Vulkan-backend prerequisites are "configured" by default so tests that
    # select the backend don't trip the selection-time warning (the model env
    # must point at an actually-existing file, mirroring ensure_model_ready);
    # tests that exercise the warning break these explicitly.
    model_file = tmp_path / "chatterbox-model"
    model_file.write_text("model", encoding="utf-8")
    monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: tmp_path / "audiocpp_cli")
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model_file))
    # Isolate the app-level settings file (remembered backend + audio.cpp
    # paths) so tests never read or write the developer's real config.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    window = app_gui.MainWindow()
    return window, paths


def test_gui_pipeline_uses_native_safe_analysis_dependencies(
    qt_app,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GUI analysis must not load transformer/LanguageTool native stacks.

    Those stacks are safe for the synchronous CLI path but can segfault when
    loaded from the Qt process on this Ubuntu desktop; the GUI uses the local
    deterministic fallbacks instead.
    """
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        captured: dict[str, object] = {}

        class _RecordingPipeline:
            def __init__(self, **kwargs) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(app_gui, "OraclePipeline", _RecordingPipeline)
        window.pipeline = None
        window._pipeline()

        assert captured == {
            "use_transformers": False,
            "use_language_tool": False,
            "use_punctuation_model": False,
        }
    finally:
        window.close()


def test_analyze_click_builds_review_plan_with_gui_safe_pipeline(
    qt_app,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression coverage for the real Analyze click path.

    Use the production pipeline with every optional native analysis dependency
    disabled, then exercise MainWindow.prepare_project end-to-end through table
    population. This catches the former Ubuntu/Qt crash path without loading a
    TTS model or rendering audio.
    """
    import the_oracle.app_gui as app_gui
    from the_oracle.pipeline import OraclePipeline as ProductionPipeline

    window, _paths = _build_window(monkeypatch, tmp_path)
    input_path = tmp_path / "dialogue.txt"
    input_path.write_text("Alice: Hello there.\nBob: Hi, how are you?\n", encoding="utf-8")
    try:
        window.pipeline = ProductionPipeline(
            use_transformers=False,
            use_language_tool=False,
            use_punctuation_model=False,
        )
        window.input_path.setText(str(input_path))
        window.prepare_project()

        assert window.plan is not None
        # Text ingestion strips explicit speaker prefixes into the utterance
        # body; the speaker attribution remains available on each row.
        assert [item.original_text for item in window.plan.utterances] == [
            "Hello there.",
            "Hi, how are you?",
        ]
        assert [item.speaker for item in window.plan.utterances] == ["A", "B"]
        assert window.table.rowCount() == 2
        assert "Analysis complete." in window.error_panel.toPlainText()
    finally:
        window.close()


def test_loading_profile_payload_replaces_current_settings(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    custom_output = tmp_path / "custom_output"
    custom_output.mkdir()

    try:
        window.variant_combo.setCurrentText("multilingual")
        window.correction_mode_combo.setCurrentText("Aggressive")
        window.loudness_combo.setCurrentText("medium")
        window.crossfade_spin.setValue(75)
        window.outdir_path.setText(str(custom_output))
        window.output_name.setText("chapter_one")
        window.speaker_a.reference_path.setText("/tmp/speaker_a.wav")
        window.speaker_a.cfg_weight.setValue(0.9)
        window.speaker_b.reference_path.setText("/tmp/speaker_b.wav")
        window.speaker_b.pause_spin.setValue(420)
        payload = window._current_gui_settings_payload()

        window.variant_combo.setCurrentText("standard")
        window.correction_mode_combo.setCurrentText("Moderate")
        window.loudness_combo.setCurrentText("off")
        window.crossfade_spin.setValue(20)
        window.outdir_path.setText(str(paths.output_dir))
        window.output_name.clear()
        window.speaker_a.reference_path.clear()
        window.speaker_a.cfg_weight.setValue(0.5)
        window.speaker_b.reference_path.clear()
        window.speaker_b.pause_spin.setValue(180)

        window._apply_gui_settings_payload(payload)

        assert window.variant_combo.currentText() == "multilingual"
        assert window.correction_mode_combo.currentData() == "aggressive"
        assert window.loudness_combo.currentText() == "medium"
        assert window.crossfade_spin.value() == 75
        assert window.outdir_path.text() == str(custom_output)
        assert window.output_name.text() == "chapter_one.flac"
        assert window.speaker_a.reference_path.text() == "/tmp/speaker_a.wav"
        assert window.speaker_a.cfg_weight.value() == pytest.approx(0.9)
        assert window.speaker_b.reference_path.text() == "/tmp/speaker_b.wav"
        assert window.speaker_b.pause_spin.value() == 420
    finally:
        window.close()


def test_new_project_keeps_current_profile_settings(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    custom_output = tmp_path / "custom_output"
    custom_output.mkdir()

    try:
        window.input_path.setText(str(tmp_path / "dialogue.txt"))
        window.outdir_path.setText(str(custom_output))
        window.output_name.setText("keep_me")
        window.loudness_combo.setCurrentText("medium")
        window.speaker_a.reference_path.setText("/tmp/speaker_a.wav")
        window.speaker_b.reference_path.setText("/tmp/speaker_b.wav")

        window.new_project()

        assert window.input_path.text() == ""
        assert window.table.rowCount() == 0
        assert window.outdir_path.text() == str(custom_output)
        assert window.output_name.text() == "keep_me"
        assert window.loudness_combo.currentText() == "medium"
        assert window.speaker_a.reference_path.text() == "/tmp/speaker_a.wav"
        assert window.speaker_b.reference_path.text() == "/tmp/speaker_b.wav"
        assert window.paths.output_dir == paths.output_dir
    finally:
        window.close()


def test_reset_to_defaults_restores_profile_baseline(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    defaults = VoiceSettings()
    render_defaults = RenderSettings()

    try:
        window.variant_combo.setCurrentText("multilingual")
        window.correction_mode_combo.setCurrentText("Aggressive")
        window.loudness_combo.setCurrentText("medium")
        window.crossfade_spin.setValue(60)
        window.outdir_path.setText(str(tmp_path / "custom_output"))
        window.output_name.setText("custom_render")
        window.speaker_a.reference_path.setText("/tmp/speaker_a.wav")
        window.speaker_a.cfg_weight.setValue(1.0)
        window.speaker_b.reference_path.setText("/tmp/speaker_b.wav")
        window.speaker_b.pause_spin.setValue(360)

        window.reset_settings_to_defaults()

        assert window.variant_combo.currentText() == render_defaults.model_variant
        assert window.correction_mode_combo.currentData() == render_defaults.correction_mode
        assert window.loudness_combo.currentText() == render_defaults.loudness_preset
        assert window.crossfade_spin.value() == render_defaults.crossfade_ms
        assert window.outdir_path.text() == str(paths.output_dir)
        assert window.output_name.text() == ""
        assert window.speaker_a.reference_path.text() == ""
        assert window.speaker_b.reference_path.text() == ""
        assert window.speaker_a.cfg_weight.value() == pytest.approx(defaults.cfg_weight)
        assert window.speaker_b.pause_spin.value() == defaults.pause_ms
    finally:
        window.close()


def test_custom_output_folder_autofills_filename(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        input_file = tmp_path / "dialogue.txt"
        input_file.write_text("Line 1\nLine 2\n")
        custom_output = tmp_path / "custom_output"
        custom_output.mkdir()

        window.input_path.setText(str(input_file))
        window.outdir_path.setText(str(custom_output))

        expected = default_output_filename(input_file)
        assert window.output_name.text() == expected
        window.output_name.setText("modified")
        window.outdir_path.setText(str(custom_output))
        assert window.output_name.text() == "modified"
    finally:
        window.close()


def test_custom_output_folder_requires_filename(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        custom_output = tmp_path / "custom_output"
        custom_output.mkdir()
        window.plan = _minimal_plan(paths)
        window.input_path.setText(str(tmp_path / "dialogue.txt"))
        window.outdir_path.setText(str(custom_output))
        window.output_name.clear()

        def fail(*args, **kwargs):
            raise RuntimeError(args[2])

        monkeypatch.setattr(QMessageBox, "critical", fail)

        with pytest.raises(RuntimeError) as excinfo:
            window.render_project()
        assert "Choose an output filename" in str(excinfo.value)
    finally:
        window.close()


def test_input_browse_defaults_to_repo_input_dir_on_fresh_use(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    captured: dict[str, str] = {}

    try:
        def fake_get_open_file_name(_parent, _title, start_dir, _filter):
            captured["start_dir"] = start_dir
            return ("", "")

        monkeypatch.setattr("the_oracle.app_gui.QFileDialog.getOpenFileName", fake_get_open_file_name)
        window.input_path.clear()

        window._pick_input()

        assert captured["start_dir"] == str(paths.input_dir)
    finally:
        window.close()


def test_custom_reference_picker_custom_option_works_on_first_click(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    calls: list[str] = []

    try:
        monkeypatch.setattr(window.speaker_a, "_pick_audio", lambda: calls.append("picked"))
        window.speaker_a.set_reference_choices([], [], "")

        assert window.speaker_a.reference_picker.currentData() == "__custom__"
        window.speaker_a.reference_picker.activated.emit(window.speaker_a.reference_picker.currentIndex())

        assert calls == ["picked"]
    finally:
        window.close()


def test_inference_backend_selector_defaults_to_pytorch(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert window.inference_backend_combo.currentData() == "pytorch"
        assert window._render_settings().inference_backend == "pytorch"
    finally:
        window.close()


def test_inference_backend_selector_wires_into_render_settings(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        vulkan_index = window.inference_backend_combo.findData("vulkan")
        window.inference_backend_combo.setCurrentIndex(vulkan_index)
        assert window._render_settings().inference_backend == "vulkan"
    finally:
        window.close()


def test_inference_backend_round_trips_through_settings_payload(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        vulkan_index = window.inference_backend_combo.findData("vulkan")
        window.inference_backend_combo.setCurrentIndex(vulkan_index)
        payload = window._current_gui_settings_payload()
        assert payload["project"]["inference_backend"] == "vulkan"

        # Reset to pytorch, then re-apply the saved payload and confirm restore.
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("pytorch"))
        window._apply_gui_settings_payload(payload)
        assert window.inference_backend_combo.currentData() == "vulkan"
        assert window._render_settings().inference_backend == "vulkan"
    finally:
        window.close()


def test_turbo_variant_disables_vulkan_backend_option(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        vulkan_index = window.inference_backend_combo.findData("vulkan")
        window.inference_backend_combo.setCurrentIndex(vulkan_index)
        assert window.inference_backend_combo.currentData() == "vulkan"

        window.variant_combo.setCurrentText("turbo")
        # Vulkan option is disabled and selection falls back to pytorch.
        assert not window.inference_backend_combo.model().item(vulkan_index).isEnabled()
        assert window.inference_backend_combo.currentData() == "pytorch"
        assert window._render_settings().inference_backend == "pytorch"

        window.variant_combo.setCurrentText("standard")
        assert window.inference_backend_combo.model().item(vulkan_index).isEnabled()
    finally:
        window.close()


def test_inconsistent_turbo_vulkan_profile_falls_back_to_pytorch(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A hand-edited profile pairing turbo with vulkan must not leave the
    disabled vulkan option selected (which would surface as a confusing engine
    error at render time)."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        payload = window._current_gui_settings_payload()
        payload["project"]["model_variant"] = "turbo"
        payload["project"]["inference_backend"] = "vulkan"
        window._apply_gui_settings_payload(payload)

        assert window.variant_combo.currentText() == "turbo"
        assert window.inference_backend_combo.currentData() == "pytorch"
        assert window._render_settings().inference_backend == "pytorch"
    finally:
        window.close()


def test_selecting_vulkan_starts_device_probe(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert window._vulkan_probe_thread is None
        assert window.audio_cpp_device_label.text() == ""

        vulkan_index = window.inference_backend_combo.findData("vulkan")
        window.inference_backend_combo.setCurrentIndex(vulkan_index)

        assert window._vulkan_probe_thread is not None
        assert window._vulkan_probe_thread.started is True
        assert window.audio_cpp_device_label.text() == "Probing audio.cpp devices..."
    finally:
        window.close()


def test_vulkan_devices_populate_picker_with_real_device_names(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The Vulkan Device picker lists audio.cpp's actual devices by name
    instead of a blind 0-15 index range, and a stale saved value survives the
    rebuild as an explicit "(not detected)" row (never silently changed)."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        # Before the probe only Auto exists.
        assert window.audio_cpp_device_combo.count() == 1
        assert window._audio_cpp_device_value() is None

        window._handle_vulkan_devices(
            [
                {"index": 0, "name": "AMD Radeon RX 5700 XT (RADV NAVI10)"},
                {"index": 1, "name": "AMD Radeon RX 6900 XT (RADV NAVI21)"},
            ]
        )

        texts = [
            window.audio_cpp_device_combo.itemText(i)
            for i in range(window.audio_cpp_device_combo.count())
        ]
        assert "Auto (audio.cpp default)" in texts
        assert "Device 0: AMD Radeon RX 5700 XT (RADV NAVI10)" in texts
        assert "Device 1: AMD Radeon RX 6900 XT (RADV NAVI21)" in texts
        # Selecting a detected device reports its real index.
        window.audio_cpp_device_combo.setCurrentIndex(
            window.audio_cpp_device_combo.findData(1)
        )
        assert window._audio_cpp_device_value() == 1
        assert "5700 XT" in window.audio_cpp_device_combo.toolTip()
    finally:
        window.close()


def test_vulkan_devices_stale_value_preserved_as_not_detected(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A stale saved device index is kept as an explicit custom row and the
    change is surfaced in the label, not silently clamped or dropped."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._set_audio_cpp_device_value(7)
        window._handle_vulkan_devices([{"index": 0, "name": "GPU A"}])

        texts = [
            window.audio_cpp_device_combo.itemText(i)
            for i in range(window.audio_cpp_device_combo.count())
        ]
        assert any("7: (not detected)" in text for text in texts)
        assert window._audio_cpp_device_value() == 7  # preserved, not clamped
        assert "not in the detected list" in window.audio_cpp_device_label.text()
        assert "GPU A" in window.audio_cpp_device_label.text()
    finally:
        window.close()


def test_vulkan_devices_empty_result_shows_note(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._handle_vulkan_devices([])
        assert "No Vulkan devices detected" in window.audio_cpp_device_label.text()
    finally:
        window.close()


def test_vulkan_devices_empty_result_with_stale_selection_notes_it(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A stale device index survives an empty probe result as a visible
    '(not detected)' row plus the note, never silently dropped."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._set_audio_cpp_device_value(7)
        window._handle_vulkan_devices([])

        texts = [
            window.audio_cpp_device_combo.itemText(i)
            for i in range(window.audio_cpp_device_combo.count())
        ]
        assert any("7: (not detected)" in text for text in texts)
        assert window._audio_cpp_device_value() == 7
        assert "No Vulkan devices detected" in window.audio_cpp_device_label.text()
        assert "not in the detected list" in window.audio_cpp_device_label.text()
    finally:
        window.close()


def test_vulkan_device_probe_failure_surfaces_message(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._handle_vulkan_probe_failed("audiocpp_cli is not built; run scripts/build_audio_cpp.sh")
        assert "audio.cpp devices unavailable" in window.audio_cpp_device_label.text()
        assert "build_audio_cpp" in window.audio_cpp_device_label.text()
        assert "audiocpp_cli is not built" in window.audio_cpp_device_combo.toolTip()
    finally:
        window.close()


def test_vulkan_selection_auto_starts_setup_when_prerequisites_missing(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Selecting Vulkan without audiocpp_cli / the model kicks off the
    automatic CPU→GPU setup (instead of only warning) so the switch completes
    by itself rather than failing deep in the render worker."""
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        # Break the (default-configured) Vulkan setup.
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: None)
        monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
        vulkan_index = window.inference_backend_combo.findData("vulkan")
        window.inference_backend_combo.setCurrentIndex(vulkan_index)

        # The auto-setup thread was started, and the status panel explains it.
        assert window._vulkan_setup_thread is not None
        assert window._vulkan_setup_thread.started is True
        assert window._vulkan_setup_attempted is True
        assert not window.vulkan_prerequisite_warning.isHidden()
        assert "setting it up automatically" in window.vulkan_prerequisite_warning.text()

        # Switching back to PyTorch hides the status panel.
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("pytorch"))
        assert window.vulkan_prerequisite_warning.isHidden()
    finally:
        window.close()


def test_vulkan_selection_no_warning_when_configured(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        vulkan_index = window.inference_backend_combo.findData("vulkan")
        window.inference_backend_combo.setCurrentIndex(vulkan_index)
        assert window.vulkan_prerequisite_warning.isHidden()
    finally:
        window.close()


def test_vulkan_selection_auto_starts_setup_when_model_file_missing(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A stale ORACLE_AUDIOCPP_MODEL pointing at a deleted file still triggers
    the automatic setup (the exists() check), even when audiocpp_cli resolves."""
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        # audiocpp_cli still resolves; only the model file is gone.
        monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(tmp_path / "deleted-model"))
        assert not (tmp_path / "deleted-model").exists()
        vulkan_index = window.inference_backend_combo.findData("vulkan")
        window.inference_backend_combo.setCurrentIndex(vulkan_index)

        assert window._vulkan_setup_thread is not None
        assert window._vulkan_setup_thread.started is True
        assert not window.vulkan_prerequisite_warning.isHidden()
        assert "setting it up automatically" in window.vulkan_prerequisite_warning.text()
    finally:
        window.close()


def test_render_project_queues_setup_when_vulkan_unconfigured(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Render with an unconfigured Vulkan backend starts the automatic setup
    and queues the render (no worker yet) instead of blocking with a manual-
    instructions warning; the render fires once setup completes."""
    import the_oracle.app_gui as app_gui

    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        plan = _minimal_plan(paths)
        plan.utterances.append(Utterance(index=0, original_text="Hi."))
        window.plan = plan
        window.input_path.setText(str(tmp_path / "dialogue.txt"))
        window.output_name.setText("chapter")
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: None)
        monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("vulkan"))

        dialogs: list[str] = []
        monkeypatch.setattr(
            app_gui.QMessageBox,
            "information",
            lambda _parent, title, message: dialogs.append(message),
        )

        window.render_project()

        assert dialogs, "expected a friendly setup-in-progress dialog"
        assert "being set up automatically" in dialogs[0]
        assert window.render_worker is None
        assert window._render_queued_after_setup is True
        assert "Render queued" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_vulkan_setup_completion_starts_queued_render(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The core promise: when setup completes, the queued render fires by
    itself (render_project() re-enters and a worker starts) instead of leaving
    the user to click Render again."""
    import the_oracle.app_gui as app_gui

    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        plan = _minimal_plan(paths)
        plan.utterances.append(Utterance(index=0, original_text="Hi."))
        window.plan = plan
        window.input_path.setText(str(tmp_path / "dialogue.txt"))
        window.output_name.setText("chapter")
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: None)
        monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("vulkan"))
        monkeypatch.setattr(
            app_gui.QMessageBox,
            "information",
            lambda _parent, _title, _message: None,
        )
        window.render_project()
        assert window._render_queued_after_setup is True
        assert window.render_worker is None

        monkeypatch.setattr(app_gui, "RenderWorker", _FakeRenderWorker)
        # Setup "completes": env vars are now set and the binary resolves, so
        # the queued render passes the prerequisite check and starts.
        model_file = tmp_path / "chatterbox-model"
        model_file.write_text("model", encoding="utf-8")
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: tmp_path / "audiocpp_cli")
        monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model_file))

        class _SetupResult:
            ok = True
            messages = ["audiocpp_cli ready: /tmp/audiocpp_cli"]

        window._handle_vulkan_setup_completed(_SetupResult())

        assert window._render_queued_after_setup is False
        assert window.render_worker is not None
        assert window.render_worker.started is True
        # The auto-fired render must show its progress dialog (the user asked
        # for the render to start visibly, not just silently queue).
        assert window.progress_dialog is not None
        assert "Vulkan backend setup complete" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_vulkan_setup_failure_clears_queue_and_surfaces_error(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed setup clears the queued render/preview and surfaces the error
    visibly, and re-allows a later attempt (no silent queue-forever)."""
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._render_queued_after_setup = True
        window._preview_queued_after_setup = True
        window._preflight_queued_after_setup = True
        window._vulkan_setup_attempted = True

        window._handle_vulkan_setup_failed("cmake not found")

        assert window._render_queued_after_setup is False
        assert window._preview_queued_after_setup is False
        assert window._preflight_queued_after_setup is False
        # A later explicit click may retry (the attempt guard was reset).
        assert window._vulkan_setup_attempted is False
        assert "Vulkan backend setup failed" in window.error_panel.toPlainText()
        assert not window.vulkan_prerequisite_warning.isHidden()
        assert "cmake not found" in window.vulkan_prerequisite_warning.text()
    finally:
        window.close()


def test_vulkan_preflight_queues_behind_running_setup(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clicking Test Vulkan Backend while the automatic setup is running
    (prerequisites still missing) queues the preflight instead of failing with
    a manual-scripts dialog: no preflight thread spawns yet, a friendly
    setup-in-progress dialog is shown, and the queued flag is set so the test
    fires on its own once setup completes."""
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        # Break the (default-configured) Vulkan setup, then select Vulkan so
        # the selection-time auto-setup thread starts and is still running.
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: None)
        monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("vulkan"))
        assert window._vulkan_setup_thread is not None

        infos: list[str] = []
        warnings: list[str] = []
        monkeypatch.setattr(app_gui.QMessageBox, "information", lambda _parent, _title, message: infos.append(message))
        monkeypatch.setattr(app_gui.QMessageBox, "warning", lambda _parent, _title, message: warnings.append(message))

        window.test_vulkan_button.click()

        assert window._vulkan_preflight_thread is None, "preflight must wait for setup"
        assert window._preflight_queued_after_setup is True
        assert infos and "being set up automatically" in infos[0]
        assert warnings == [], "no manual-steps warning when setup is running"
        assert "test queued" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_vulkan_setup_completion_runs_queued_preflight(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When a Test Vulkan Backend click was queued behind the automatic setup,
    setup completion fires the preflight on its own (a preflight thread starts
    and the queue flag clears) — the promise that picking GPU needs no extra
    steps or extra clicks."""
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._preflight_queued_after_setup = True
        # Setup "completes": env vars are now set and the binary resolves, so
        # the queued preflight passes the prerequisite check and starts.
        model_file = tmp_path / "chatterbox-model"
        model_file.write_text("model", encoding="utf-8")
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: tmp_path / "audiocpp_cli")
        monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model_file))

        class _SetupResult:
            ok = True
            messages = ["audiocpp_cli ready: /tmp/audiocpp_cli"]

        window._handle_vulkan_setup_completed(_SetupResult())

        assert window._preflight_queued_after_setup is False
        assert window._vulkan_preflight_thread is not None
        assert window._vulkan_preflight_thread.started is True
        assert "Vulkan backend setup complete" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_vulkan_preflight_runs_immediately_when_configured(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When the Vulkan backend is already configured, clicking Test Vulkan
    Backend still runs the preflight right away (no setup queue involved)."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert window._vulkan_preflight_thread is None
        window.test_vulkan_button.click()
        assert window._vulkan_preflight_thread is not None
        assert window._vulkan_preflight_thread.started is True
        assert window._preflight_queued_after_setup is False
        assert "Testing Vulkan backend" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_vulkan_preflight_double_click_queues_once(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Double-clicking Test Vulkan Backend while setup is running queues the
    preflight exactly once: the second click is a log line, not a second
    dialog or a second queue entry."""
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: None)
        monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("vulkan"))
        assert window._vulkan_setup_thread is not None

        infos: list[str] = []
        monkeypatch.setattr(app_gui.QMessageBox, "information", lambda _parent, _title, message: infos.append(message))

        window.test_vulkan_button.click()
        window.test_vulkan_button.click()

        assert window._preflight_queued_after_setup is True
        assert len(infos) == 1, "double-click must not show two dialogs"
        assert "already queued" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_vulkan_preflight_from_pytorch_backend_starts_setup(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clicking Test Vulkan Backend from the PyTorch backend (prerequisites
    missing) still starts the automatic setup and queues the preflight — the
    button is not gated on Vulkan being selected, so validating the GPU path
    never demands manual wiring."""
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert window.inference_backend_combo.currentData() == "pytorch"
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: None)
        monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)

        infos: list[str] = []
        monkeypatch.setattr(app_gui.QMessageBox, "information", lambda _parent, _title, message: infos.append(message))

        window.test_vulkan_button.click()

        assert window._vulkan_setup_thread is not None
        assert window._vulkan_setup_thread.started is True
        assert window._preflight_queued_after_setup is True
        assert infos and "being set up automatically" in infos[0]
    finally:
        window.close()


def test_render_project_proceeds_when_vulkan_configured(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import the_oracle.app_gui as app_gui

    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        plan = _minimal_plan(paths)
        plan.utterances.append(Utterance(index=0, original_text="Hi."))
        window.plan = plan
        window.input_path.setText(str(tmp_path / "dialogue.txt"))
        window.output_name.setText("chapter")
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("vulkan"))

        dialogs: list[str] = []
        monkeypatch.setattr(
            app_gui.QMessageBox,
            "warning",
            lambda _parent, title, message: dialogs.append(message),
        )

        # Replace RenderWorker with a recording fake so no real QThread spawns
        # (the _FakePipeline has no render method; a real worker would crash).
        monkeypatch.setattr(app_gui, "RenderWorker", _FakeRenderWorker)

        window.render_project()

        assert dialogs == [], "no warning expected when Vulkan is configured"
        # Render proceeds (a worker object is created and started).
        assert window.render_worker is not None
        assert window.render_worker.started is True
    finally:
        window.close()


def test_audio_cpp_knobs_default_to_auto_and_disabled_on_pytorch(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        # Defaults: Auto/Default (None) and disabled while the PyTorch backend
        # is selected (the knobs are Vulkan-only).
        assert window._audio_cpp_device_value() is None
        assert window._audio_cpp_threads_value() is None
        assert window._audio_cpp_timeout_value() is None
        assert window._audio_cpp_max_batch_value() is None
        assert not window.audio_cpp_device_combo.isEnabled()
        assert not window.audio_cpp_threads_spin.isEnabled()
        assert not window.audio_cpp_timeout_spin.isEnabled()
        assert not window.audio_cpp_max_batch_spin.isEnabled()
        settings = window._render_settings()
        assert settings.audio_cpp_device is None
        assert settings.audio_cpp_threads is None
        assert settings.audio_cpp_timeout is None
        assert settings.audio_cpp_max_batch is None
    finally:
        window.close()


def test_audio_cpp_knobs_wire_into_render_settings_and_enable_on_vulkan(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        vulkan_index = window.inference_backend_combo.findData("vulkan")
        window.inference_backend_combo.setCurrentIndex(vulkan_index)
        assert window.audio_cpp_device_combo.isEnabled()
        assert window.audio_cpp_threads_spin.isEnabled()
        assert window.audio_cpp_timeout_spin.isEnabled()
        assert window.audio_cpp_max_batch_spin.isEnabled()

        window._set_audio_cpp_device_value(2)
        window._set_audio_cpp_threads_value(6)
        window._set_audio_cpp_timeout_value(120)
        window._set_audio_cpp_max_batch_value(16)
        settings = window._render_settings()
        assert settings.audio_cpp_device == 2
        assert settings.audio_cpp_threads == 6
        assert settings.audio_cpp_timeout == 120
        assert settings.audio_cpp_max_batch == 16
    finally:
        window.close()


def test_audio_cpp_knobs_round_trip_through_settings_payload(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        vulkan_index = window.inference_backend_combo.findData("vulkan")
        window.inference_backend_combo.setCurrentIndex(vulkan_index)
        window._set_audio_cpp_device_value(1)
        window._set_audio_cpp_threads_value(8)
        window._set_audio_cpp_timeout_value(300)
        window._set_audio_cpp_max_batch_value(24)
        payload = window._current_gui_settings_payload()
        assert payload["project"]["audio_cpp_device"] == 1
        assert payload["project"]["audio_cpp_threads"] == 8
        assert payload["project"]["audio_cpp_timeout"] == 300
        assert payload["project"]["audio_cpp_max_batch"] == 24

        # Reset to defaults, then restore from the saved payload.
        window._set_audio_cpp_device_value(None)
        window._set_audio_cpp_threads_value(None)
        window._set_audio_cpp_timeout_value(None)
        window._set_audio_cpp_max_batch_value(None)
        window._apply_gui_settings_payload(payload)
        assert window._audio_cpp_device_value() == 1
        assert window._audio_cpp_threads_value() == 8
        assert window._audio_cpp_timeout_value() == 300
        assert window._audio_cpp_max_batch_value() == 24
        settings = window._render_settings()
        assert settings.audio_cpp_device == 1
        assert settings.audio_cpp_threads == 8
        assert settings.audio_cpp_timeout == 300
        assert settings.audio_cpp_max_batch == 24
    finally:
        window.close()


def test_stale_audio_cpp_knobs_not_persisted_on_pytorch_backend(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Setting the knobs on Vulkan then switching back to PyTorch must not
    leak the disabled widget values into render settings or saved profiles."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        vulkan_index = window.inference_backend_combo.findData("vulkan")
        window.inference_backend_combo.setCurrentIndex(vulkan_index)
        window._set_audio_cpp_device_value(2)
        window._set_audio_cpp_threads_value(6)
        window._set_audio_cpp_timeout_value(120)
        window._set_audio_cpp_max_batch_value(32)
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("pytorch"))

        settings = window._render_settings()
        assert settings.inference_backend == "pytorch"
        assert settings.audio_cpp_device is None
        assert settings.audio_cpp_threads is None
        assert settings.audio_cpp_timeout is None
        assert settings.audio_cpp_max_batch is None
        payload = window._current_gui_settings_payload()
        assert payload["project"]["audio_cpp_device"] is None
        assert payload["project"]["audio_cpp_threads"] is None
        assert payload["project"]["audio_cpp_timeout"] is None
        assert payload["project"]["audio_cpp_max_batch"] is None

        # Widget values are retained so switching back to Vulkan restores them.
        window.inference_backend_combo.setCurrentIndex(vulkan_index)
        assert window._audio_cpp_device_value() == 2
        assert window._audio_cpp_threads_value() == 6
        assert window._audio_cpp_timeout_value() == 120
        assert window._audio_cpp_max_batch_value() == 32
    finally:
        window.close()


def test_audio_cpp_knobs_restored_from_project_manifest(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        plan = _minimal_plan(paths)
        settings = RenderSettings(
            model_variant="standard",
            inference_backend="vulkan",
            audio_cpp_device=3,
            audio_cpp_threads=12,
            audio_cpp_timeout=240,
            audio_cpp_max_batch=48,
        )
        speakers = {
            "A": SpeakerSettings(reference_path="/tmp/a.wav", voice_settings=VoiceSettings()),
            "B": SpeakerSettings(reference_path="/tmp/b.wav", voice_settings=VoiceSettings()),
        }
        saved = build_saved_project(plan, settings, speakers)

        window._load_project_into_ui(saved)

        assert window.inference_backend_combo.currentData() == "vulkan"
        assert window._audio_cpp_device_value() == 3
        assert window._audio_cpp_threads_value() == 12
        assert window._audio_cpp_timeout_value() == 240
        assert window._audio_cpp_max_batch_value() == 48
        render_settings = window._render_settings()
        assert render_settings.audio_cpp_device == 3
        assert render_settings.audio_cpp_threads == 12
        assert render_settings.audio_cpp_timeout == 240
        assert render_settings.audio_cpp_max_batch == 48
    finally:
        window.close()


def test_preview_guards_stale_knobs_on_pytorch_backend(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The preview path must mirror _render_settings: stale knob values set
    while Vulkan was selected are forwarded as None once PyTorch is active."""
    import the_oracle.app_gui as app_gui

    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        vulkan_index = window.inference_backend_combo.findData("vulkan")
        window.inference_backend_combo.setCurrentIndex(vulkan_index)
        window._set_audio_cpp_device_value(2)
        window._set_audio_cpp_threads_value(6)
        window._set_audio_cpp_timeout_value(120)
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("pytorch"))

        plan = _minimal_plan(paths)
        plan.utterances.append(Utterance(index=0, original_text="Hi."))
        window.plan = plan
        window._populate_table(plan)

        captured: dict = {}

        class _FakeSignal:
            def connect(self, *_args, **_kwargs) -> None:
                return None

        class _FakePreviewWorker:
            def __init__(self, *_args, **kwargs) -> None:
                captured["kwargs"] = kwargs
                self.progress = _FakeSignal()
                self.completed = _FakeSignal()
                self.failed = _FakeSignal()
                self.finished = _FakeSignal()

            def start(self) -> None:
                return None

        monkeypatch.setattr(app_gui, "PreviewWorker", _FakePreviewWorker)

        window.preview_utterance(0)

        kwargs = captured["kwargs"]
        assert kwargs["inference_backend"] == "pytorch"
        assert kwargs["audio_cpp_device"] is None
        assert kwargs["audio_cpp_threads"] is None
        assert kwargs["audio_cpp_timeout"] is None
    finally:
        window.close()


def test_preview_worker_forwards_audio_cpp_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    import the_oracle.app_gui as app_gui

    class _RecordingPipeline:
        def __init__(self) -> None:
            self.kwargs = None

        def render_preview(self, utterance, profile, model_variant, **kwargs):
            self.kwargs = kwargs
            return "/tmp/preview.wav"

    pipeline = _RecordingPipeline()
    utterance = Utterance(index=0, original_text="Hi.")
    profile = VoiceProfile(name="Speaker A", speaker="A", reference_audio=[])
    worker = app_gui.PreviewWorker(
        utterance,
        profile,
        "standard",
        "cpu",
        pipeline=pipeline,
        inference_backend="vulkan",
        audio_cpp_device=4,
        audio_cpp_threads=10,
        audio_cpp_timeout=150,
    )

    worker.run()

    assert pipeline.kwargs["inference_backend"] == "vulkan"
    assert pipeline.kwargs["audio_cpp_device"] == 4
    assert pipeline.kwargs["audio_cpp_threads"] == 10
    assert pipeline.kwargs["audio_cpp_timeout"] == 150


def test_vulkan_preflight_button_runs_background_thread(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The Test Vulkan Backend button starts a background preflight thread
    (never blocking the UI) and disables itself while it runs."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert window.test_vulkan_button is not None
        assert window._vulkan_preflight_thread is None
        # Button is usable once the Vulkan option is selectable (enabled after
        # startup refreshes the knob/button state).
        assert window.test_vulkan_button.isEnabled()

        window.test_vulkan_button.click()

        assert window._vulkan_preflight_thread is not None
        assert window._vulkan_preflight_thread.started is True
        assert not window.test_vulkan_button.isEnabled()
        assert "Testing Vulkan backend" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_vulkan_preflight_thread_forwards_selected_device_index(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The preflight threads carry the currently selected Vulkan device index
    so the report names the GPU a render would actually use."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._set_audio_cpp_device_value(2)
        window.test_vulkan_button.click()

        thread = window._vulkan_preflight_thread
        assert thread is not None
        assert thread.device_index == 2
    finally:
        window.close()


def test_vulkan_preflight_completion_reports_gpu(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        infos: list[str] = []
        monkeypatch.setattr(app_gui.QMessageBox, "information", lambda _parent, _title, message: infos.append(message))

        window._handle_vulkan_preflight_completed("Vulkan backend preflight passed.\nGPU to be used: Vulkan device 0")

        assert infos and "preflight passed" in infos[0]
        assert "GPU to be used: Vulkan device 0" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_vulkan_preflight_failure_surfaces_message(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        warnings: list[str] = []
        monkeypatch.setattr(app_gui.QMessageBox, "warning", lambda _parent, _title, message: warnings.append(message))

        window._handle_vulkan_preflight_failed("audiocpp_cli is not built")

        assert "audiocpp_cli is not built" in window.error_panel.toPlainText()
        assert warnings and "audiocpp_cli is not built" in warnings[0]
    finally:
        window.close()


def test_vulkan_preflight_cleanup_reenables_button(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The button re-enables once the preflight thread finishes."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window.test_vulkan_button.click()
        assert not window.test_vulkan_button.isEnabled()

        window._cleanup_vulkan_preflight_thread()

        assert window._vulkan_preflight_thread is None
        assert window.test_vulkan_button.isEnabled()
    finally:
        window.close()


def test_close_event_waits_for_vulkan_preflight_thread(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Closing while a preflight runs waits on the (fake) thread instead of
    destroying a running QThread (Qt aborts on that)."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    window.test_vulkan_button.click()
    thread = window._vulkan_preflight_thread
    assert thread is not None

    window.close()

    assert window._vulkan_preflight_thread is None


def test_vulkan_preflight_report_builds_gpu_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The report names the selected device and lists what audio.cpp sees."""
    import the_oracle.app_gui as app_gui

    model_file = tmp_path / "chatterbox-model"
    model_file.write_text("model", encoding="utf-8")
    monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: tmp_path / "audiocpp_cli")
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model_file))

    class _FakeEngine:
        def __init__(self, device_index=None) -> None:
            self.device_index = device_index

        def list_devices(self):
            return [
                {"index": 0, "name": "AMD Radeon RX 5700 XT (RADV NAVI10)"},
                {"index": 1, "name": "AMD Radeon RX 6900 XT (RADV NAVI21)"},
            ]

    monkeypatch.setattr(app_gui.AudioCppVulkanEngine, "list_devices", _FakeEngine.list_devices)

    report = app_gui._vulkan_preflight_report(0)

    assert "preflight passed" in report
    assert "GPU to be used: Vulkan device 0" in report
    assert "5700 XT" in report
    assert "(selected)" in report
    assert "Device 1: AMD Radeon RX 6900 XT (RADV NAVI21)" in report


def test_vulkan_preflight_report_auto_device_when_none_selected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import the_oracle.app_gui as app_gui

    model_file = tmp_path / "chatterbox-model"
    model_file.write_text("model", encoding="utf-8")
    monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: tmp_path / "audiocpp_cli")
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model_file))

    class _FakeEngine:
        def __init__(self, device_index=None) -> None:
            self.device_index = device_index

        def list_devices(self):
            return [{"index": 0, "name": "AMD Radeon RX 5700 XT (RADV NAVI10)"}]

    monkeypatch.setattr(app_gui.AudioCppVulkanEngine, "list_devices", _FakeEngine.list_devices)

    report = app_gui._vulkan_preflight_report(None)

    assert "Auto (audio.cpp picks its default device)" in report
    assert "(selected)" not in report


def test_vulkan_preflight_report_raises_when_no_devices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A zero-device --list-devices result fails the preflight instead of
    reporting a misleading success, since rendering would fail."""
    import the_oracle.app_gui as app_gui

    model_file = tmp_path / "chatterbox-model"
    model_file.write_text("model", encoding="utf-8")
    monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: tmp_path / "audiocpp_cli")
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model_file))

    class _NoDevicesEngine:
        def __init__(self, device_index=None) -> None:
            self.device_index = device_index

        def list_devices(self):
            return []

    monkeypatch.setattr(app_gui.AudioCppVulkanEngine, "list_devices", _NoDevicesEngine.list_devices)

    with pytest.raises(app_gui.AudioCppUnavailableError, match="no Vulkan devices"):
        app_gui._vulkan_preflight_report(None)


def test_vulkan_preflight_report_raises_when_prerequisites_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A missing binary/model makes the preflight fail with a clear error
    instead of probing devices."""
    import the_oracle.app_gui as app_gui

    monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: None)
    monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)

    with pytest.raises(app_gui.AudioCppUnavailableError, match="audiocpp_cli is not built"):
        app_gui._vulkan_preflight_report(None)


def test_vulkan_preflight_thread_emits_report(monkeypatch: pytest.MonkeyPatch) -> None:
    import the_oracle.app_gui as app_gui

    captured: dict[str, str] = {}
    monkeypatch.setattr(app_gui, "_vulkan_preflight_report", lambda device_index: f"report for device {device_index}")

    thread = app_gui.VulkanPreflightThread(None, device_index=3)
    thread.completed.connect(lambda report: captured.update({"completed": report}))
    thread.failed.connect(lambda message: captured.update({"failed": message}))
    thread.run()

    assert captured == {"completed": "report for device 3"}


def test_vulkan_preflight_thread_emits_failed_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import the_oracle.app_gui as app_gui

    captured: dict[str, str] = {}

    def boom(_device_index):
        raise app_gui.AudioCppUnavailableError("audiocpp_cli is not built")

    monkeypatch.setattr(app_gui, "_vulkan_preflight_report", boom)

    thread = app_gui.VulkanPreflightThread(None)
    thread.completed.connect(lambda report: captured.update({"completed": report}))
    thread.failed.connect(lambda message: captured.update({"failed": message}))
    thread.run()

    assert captured == {"failed": "audiocpp_cli is not built"}


def test_settings_menu_download_vulkan_model_action_starts_background_download(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Settings -> Download Vulkan Model... starts the background thread and
    disables the action while it runs, without blocking the UI."""
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert window.download_vulkan_model_action is not None
        assert window.download_vulkan_model_action.isEnabled()
        assert window._model_download_thread is None

        window.download_vulkan_model()

        assert window._model_download_thread is not None
        assert window._model_download_thread.started is True
        assert not window.download_vulkan_model_action.isEnabled()
        assert "Downloading the Vulkan Chatterbox model" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_download_vulkan_model_guard_when_script_missing(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        warnings: list[str] = []
        monkeypatch.setattr(app_gui.QMessageBox, "warning", lambda _parent, _title, message: warnings.append(message))
        # Point at a fake repo root that has no scripts/ checkout.
        window.repo_root = tmp_path / "empty-repo"
        (window.repo_root / "scripts").mkdir(parents=True)

        window.download_vulkan_model()

        assert window._model_download_thread is None
        assert warnings, "expected a warning when the download script is missing"
        assert "download_audio_cpp_model.sh" in warnings[0]
        assert "missing" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_vulkan_model_download_completion_sets_env_and_reports_path(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        model_file = tmp_path / "downloaded-model.gguf"
        model_file.write_bytes(b"stub-model")
        infos: list[str] = []
        monkeypatch.setattr(app_gui.QMessageBox, "information", lambda _parent, _title, message: infos.append(message))
        # _handle_vulkan_model_downloaded mutates the process env directly;
        # record the pre-existing value so monkeypatch restores it on teardown
        # and this test cannot pollute later tests (e.g. vulkan_backend's
        # missing-model check) with a now-configured ORACLE_AUDIOCPP_MODEL.
        monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", os.environ.get("ORACLE_AUDIOCPP_MODEL", ""))

        window._handle_vulkan_model_downloaded(str(model_file))

        assert os.environ.get("ORACLE_AUDIOCPP_MODEL") == str(model_file)
        # The error panel carries the unquoted session value; the dialog shows
        # the exact export line the user can persist in their shell profile.
        assert f"ORACLE_AUDIOCPP_MODEL={model_file}" in window.error_panel.toPlainText()
        assert infos and f'export ORACLE_AUDIOCPP_MODEL="{model_file}"' in infos[0]
        # The prerequisite warning clears once the model exists for the session.
        assert window.vulkan_prerequisite_warning.isHidden()
    finally:
        window.close()


def test_vulkan_model_download_failure_surfaces_message(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        warnings: list[str] = []
        monkeypatch.setattr(app_gui.QMessageBox, "warning", lambda _parent, _title, message: warnings.append(message))

        window._handle_vulkan_model_download_failed("network timeout")

        assert "network timeout" in window.error_panel.toPlainText()
        assert warnings and "network timeout" in warnings[0]
    finally:
        window.close()


def test_parse_oracle_model_path_extracts_export_line() -> None:
    import the_oracle.app_gui as app_gui

    sample = (
        "Chatterbox model installed: /tmp/audio.cpp/models/Chatterbox-GGUF/chatterbox-q8_0.gguf\n\n"
        "Point The Oracle at it and render on the Vulkan backend:\n\n"
        '    export ORACLE_AUDIOCPP_MODEL="/tmp/audio.cpp/models/Chatterbox-GGUF/chatterbox-q8_0.gguf"\n'
        '    export ORACLE_AUDIOCPP_CLI="..."\n'
    )
    assert (
        app_gui._parse_oracle_model_path(sample)
        == "/tmp/audio.cpp/models/Chatterbox-GGUF/chatterbox-q8_0.gguf"
    )
    assert app_gui._parse_oracle_model_path("no export line here") == ""


class _FakePopen:
    """Minimal stand-in for subprocess.Popen that returns canned output."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.returncode = 0
        self.stdout_text = ""
        self.stderr_text = ""
        self.terminated = False
        self.killed = False

    def communicate(self, timeout=None):
        return self.stdout_text, self.stderr_text

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _fake_popen(stdout: str = "", stderr: str = "", returncode: int = 0):
    def factory(*_args, **_kwargs):
        proc = _FakePopen()
        proc.stdout_text = stdout
        proc.stderr_text = stderr
        proc.returncode = returncode
        return proc

    return factory


def test_model_download_thread_emits_completed_with_parsed_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import the_oracle.app_gui as app_gui

    script = tmp_path / "download_audio_cpp_model.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        app_gui.subprocess,
        "Popen",
        _fake_popen(stdout='Chatterbox model installed\n\nexport ORACLE_AUDIOCPP_MODEL="/tmp/model.gguf"\n'),
    )

    thread = app_gui.ModelDownloadThread(script)
    thread.completed.connect(lambda path: captured.update({"completed": path}))
    thread.failed.connect(lambda message: captured.update({"failed": message}))
    thread.run()

    assert captured == {"completed": "/tmp/model.gguf"}


def test_model_download_thread_emits_failed_on_script_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import the_oracle.app_gui as app_gui

    script = tmp_path / "download_audio_cpp_model.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        app_gui.subprocess,
        "Popen",
        _fake_popen(stderr="ERROR: model manager not found", returncode=1),
    )

    thread = app_gui.ModelDownloadThread(script)
    thread.completed.connect(lambda path: captured.update({"completed": path}))
    thread.failed.connect(lambda message: captured.update({"failed": message}))
    thread.run()

    assert "failed" in captured
    assert "model manager not found" in captured["failed"]
    assert "completed" not in captured


def test_close_event_waits_for_model_download_thread(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Closing the window while a download is in flight cancels the subprocess
    and waits on the thread instead of destroying a running QThread (Qt aborts
    on that)."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    window.download_vulkan_model()
    thread = window._model_download_thread
    assert thread is not None

    window.close()

    # closeEvent cancelled and waited on the (fake) thread and released the
    # reference. (The action is re-enabled by the normal finished-cleanup path,
    # which closeEvent deliberately disconnects so cancel can't pop a dialog.)
    assert window._model_download_thread is None
    assert thread.cancelled is True


def test_prewarm_thread_skips_heavy_backend_init(monkeypatch: pytest.MonkeyPatch) -> None:
    import the_oracle.app_gui as app_gui

    def fail(*_args, **_kwargs):
        raise AssertionError("heavy backend init should not run during startup prewarm")

    monkeypatch.setattr(app_gui, "OraclePipeline", fail)
    monkeypatch.setattr(app_gui, "ChatterboxEngine", fail)

    thread = app_gui.PrewarmThread(device="cpu")
    ready_payload: dict[str, object] = {}

    thread.ready.connect(
        lambda pipeline, engine, timing: ready_payload.update(
            {"pipeline": pipeline, "engine": engine, "timing": timing}
        ),
        Qt.DirectConnection,
    )
    thread.failed.connect(
        lambda message, _timing: pytest.fail(f"prewarm failed unexpectedly: {message}"),
        Qt.DirectConnection,
    )

    thread.run()

    assert ready_payload["pipeline"] is None
    assert ready_payload["engine"] is None
    timing = ready_payload["timing"]
    assert isinstance(timing, dict)
    assert "prewarm_start" in timing
    assert "prewarm_complete" in timing


# -------------------- Cross-session backend memory --------------------

def _prewrite_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: dict) -> None:
    """Write the app-level settings file (remembered backend + audio.cpp paths)
    before the window is built, simulating a previous session's persisted state."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    save_app_settings(payload)


def test_remembered_vulkan_backend_restored_on_launch(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The core promise: a session that chose Vulkan and resolved the audio.cpp
    paths restores both at next launch — backend selected, env vars set — with
    zero re-setup."""
    cli = tmp_path / "audiocpp_cli"
    cli.write_text("bin", encoding="utf-8")
    model = tmp_path / "model.gguf"
    model.write_text("model", encoding="utf-8")
    _prewrite_app_settings(monkeypatch, tmp_path, {
        "remember_backend": True,
        "inference_backend": "vulkan",
        "audio_cpp_device": 0,
        "audio_cpp_threads": 6,
        "audio_cpp_timeout": 120,
        "audio_cpp_max_batch": 16,
        "audio_cpp_cli": str(cli),
        "audio_cpp_model": str(model),
    })

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        # Register the env keys for restoration: _apply_remembered_backend
        # writes them directly, and monkeypatch must restore the pre-test
        # values on teardown so the session env never leaks into other tests.
        monkeypatch.setenv("ORACLE_AUDIOCPP_CLI", os.environ.get("ORACLE_AUDIOCPP_CLI", ""))
        monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", os.environ.get("ORACLE_AUDIOCPP_MODEL", ""))
        window._apply_remembered_backend()

        assert window.inference_backend_combo.currentData() == "vulkan"
        assert os.environ.get("ORACLE_AUDIOCPP_CLI") == str(cli)
        assert os.environ.get("ORACLE_AUDIOCPP_MODEL") == str(model)
        assert window._audio_cpp_device_value() == 0
        assert window._audio_cpp_threads_value() == 6
        assert window._audio_cpp_timeout_value() == 120
        assert window._audio_cpp_max_batch_value() == 16
        assert "Restored the remembered Vulkan backend" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_remembered_backend_disabled_stays_pytorch(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With 'Remember GPU/CPU choice' disabled, launch always starts on PyTorch
    and never touches the environment."""
    model = tmp_path / "model.gguf"
    model.write_text("model", encoding="utf-8")
    _prewrite_app_settings(monkeypatch, tmp_path, {
        "remember_backend": False,
        "inference_backend": "vulkan",
        "audio_cpp_model": str(model),
    })

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        before_model = os.environ.get("ORACLE_AUDIOCPP_MODEL")
        window._apply_remembered_backend()

        assert window.inference_backend_combo.currentData() == "pytorch"
        assert os.environ.get("ORACLE_AUDIOCPP_MODEL") == before_model
    finally:
        window.close()


def test_remembered_backend_stale_model_path_not_applied(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A remembered path whose file no longer exists is surfaced but never
    applied, so the app falls back to the normal (auto-healing) prerequisite
    flow instead of silently pointing at a dead path."""
    cli = tmp_path / "audiocpp_cli"
    cli.write_text("bin", encoding="utf-8")
    _prewrite_app_settings(monkeypatch, tmp_path, {
        "remember_backend": True,
        "inference_backend": "vulkan",
        "audio_cpp_cli": str(cli),
        "audio_cpp_model": str(tmp_path / "deleted-model.gguf"),  # never created
    })

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        # Guard the CLI env key for restoration (the model key is already
        # monkeypatched by _build_window and restored on teardown).
        monkeypatch.setenv("ORACLE_AUDIOCPP_CLI", os.environ.get("ORACLE_AUDIOCPP_CLI", ""))
        window._apply_remembered_backend()

        # The stale model path must not be applied; the fixture's valid model
        # env (set by _build_window) is left untouched.
        assert os.environ.get("ORACLE_AUDIOCPP_MODEL") == str(tmp_path / "chatterbox-model")
        assert os.environ.get("ORACLE_AUDIOCPP_CLI") == str(cli)
        assert window.inference_backend_combo.currentData() == "vulkan"
        assert "no longer exists" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_backend_change_persists_remembered_choice(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Switching the Inference Backend dropdown records the choice so the next
    launch restores it."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._app_settings_ready = True
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("vulkan"))

        payload = load_app_settings()
        assert payload["inference_backend"] == "vulkan"
        assert payload["remember_backend"] is True
    finally:
        window.close()


def test_vulkan_knob_change_persists_remembered_choice(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The Vulkan device/threads/timeout/batch knobs are recorded too, so a
    chosen GPU survives across sessions."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._app_settings_ready = True
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("vulkan"))
        window.audio_cpp_threads_spin.setValue(8)
        window.audio_cpp_timeout_spin.setValue(300)
        window.audio_cpp_max_batch_spin.setValue(24)

        payload = load_app_settings()
        assert payload["inference_backend"] == "vulkan"
        assert payload["audio_cpp_threads"] == 8
        assert payload["audio_cpp_timeout"] == 300
        assert payload["audio_cpp_max_batch"] == 24
    finally:
        window.close()


def test_setup_completion_persists_resolved_paths(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When the automatic setup finishes, the resolved audiocpp_cli and model
    paths are persisted so a later launch can restore them without re-running
    the build/download."""
    import the_oracle.app_gui as app_gui

    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._app_settings_ready = True
        # Setup only ever runs while the Vulkan backend is selected (that is
        # what triggers it), so select it before completing setup — the
        # persisted record must match the backend the user is on.
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("vulkan"))
        monkeypatch.setattr(app_gui.QMessageBox, "information", lambda *_args, **_kwargs: None)
        cli = tmp_path / "built" / "audiocpp_cli"
        model = tmp_path / "downloaded" / "chatterbox.gguf"
        # Precompute strings: a class body cannot read an enclosing-function
        # name that it also assigns (that name becomes class-local), so build
        # the values outside the class to avoid the NameError gotcha.
        binary_value = str(cli)
        model_value = str(model)

        class _SetupResult:
            ok = True
            messages = ["audiocpp_cli ready"]
            binary = binary_value
            model = model_value

        window._handle_vulkan_setup_completed(_SetupResult())

        payload = load_app_settings()
        assert payload["audio_cpp_cli"] == str(cli)
        assert payload["audio_cpp_model"] == str(model)
        assert payload["inference_backend"] == "vulkan"
    finally:
        window.close()


def test_remember_toggle_persists_flag_and_state(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Toggling the Settings-menu option persists the flag immediately; turning
    it off records the disabled state so the next launch starts on PyTorch."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._app_settings_ready = True
        # setChecked(False) drives the real toggled(bool) signal path, so the
        # handler is invoked exactly as a user unchecking the menu action would.
        window.remember_backend_action.setChecked(False)

        payload = load_app_settings()
        assert payload["remember_backend"] is False
    finally:
        window.close()


def test_disabled_remember_skips_persisting_changes(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With 'Remember GPU/CPU choice' unchecked, changing the backend must not
    keep writing the settings file — the disabled flag stays, nothing else is
    recorded, and the next launch still starts on PyTorch."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window._app_settings_ready = True
        # Unchecking the action fires toggled(False), which persists the
        # disabled flag; from then on the backend must not rewrite the file.
        window.remember_backend_action.setChecked(False)
        before = load_app_settings()
        assert before["remember_backend"] is False

        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("vulkan"))
        after = load_app_settings()

        assert after == before
        assert after["remember_backend"] is False
        assert after["inference_backend"] == "pytorch"
    finally:
        window.close()


def test_vulkan_setup_completion_starts_queued_preview(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The preview promise: a preview clicked while the Vulkan setup runs is
    remembered with its row and fires by itself when setup completes — the
    worker starts and the preview progress dialog appears, no second click."""
    import the_oracle.app_gui as app_gui

    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        plan = _minimal_plan(paths)
        plan.utterances.append(Utterance(index=0, original_text="Hi."))
        window.plan = plan
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("vulkan"))
        # Break the prerequisites so previewing queues instead of starting.
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: None)
        monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
        monkeypatch.setattr(app_gui.QMessageBox, "information", lambda *_args, **_kwargs: None)

        window.preview_utterance(0)
        assert window._preview_queued_after_setup is True
        assert window._preview_row_queued_after_setup == 0
        assert window.preview_worker is None

        # Setup "completes": prerequisites are now met and the queued preview
        # must fire on its own.
        model_file = tmp_path / "chatterbox-model"
        model_file.write_text("model", encoding="utf-8")
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: tmp_path / "audiocpp_cli")
        monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model_file))
        monkeypatch.setattr(app_gui, "PreviewWorker", _FakePreviewWorker)

        class _SetupResult:
            ok = True
            messages = []

        window._handle_vulkan_setup_completed(_SetupResult())

        assert window._preview_queued_after_setup is False
        assert window._preview_row_queued_after_setup is None
        assert window.preview_worker is not None
        assert window.preview_worker.started is True
        assert window.preview_dialog is not None
        assert "starting the queued preview" in window.error_panel.toPlainText()
    finally:
        window.close()


def test_vulkan_setup_completion_drops_queued_preview_when_render_queued(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If both a preview and a full render were queued behind the setup, the
    render wins (they cannot run concurrently) and the preview queue is
    dropped cleanly — no stale flag or stray preview worker."""
    import the_oracle.app_gui as app_gui

    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        plan = _minimal_plan(paths)
        plan.utterances.append(Utterance(index=0, original_text="Hi."))
        window.plan = plan
        window.input_path.setText(str(tmp_path / "dialogue.txt"))
        window.output_name.setText("chapter")
        window.inference_backend_combo.setCurrentIndex(window.inference_backend_combo.findData("vulkan"))
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: None)
        monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
        monkeypatch.setattr(app_gui.QMessageBox, "information", lambda *_args, **_kwargs: None)

        window.preview_utterance(0)
        window.render_project()
        assert window._render_queued_after_setup is True
        assert window._preview_queued_after_setup is True

        # Setup completes: the render fires and the preview queue is dropped.
        model_file = tmp_path / "chatterbox-model"
        model_file.write_text("model", encoding="utf-8")
        monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: tmp_path / "audiocpp_cli")
        monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model_file))
        monkeypatch.setattr(app_gui, "RenderWorker", _FakeRenderWorker)
        monkeypatch.setattr(app_gui, "PreviewWorker", _FakePreviewWorker)

        class _SetupResult:
            ok = True
            messages = []

        window._handle_vulkan_setup_completed(_SetupResult())

        assert window.render_worker is not None
        assert window.render_worker.started is True
        assert window._preview_queued_after_setup is False
        assert window._preview_row_queued_after_setup is None
        assert window.preview_worker is None
    finally:
        window.close()
