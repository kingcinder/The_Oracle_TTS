import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from the_oracle.models.project import RenderPlan, Utterance, VoiceProfile, VoiceSettings
from the_oracle.pipeline import RenderSettings, SpeakerSettings
from the_oracle.project_manifest import build_saved_project

from tests.test_app_gui_profiles import _build_window

pytestmark = pytest.mark.slow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeRenderer:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.render_kwargs: dict[str, object] = {}

    def render(self, _plan, _settings, **kwargs) -> Path:
        self.render_kwargs = kwargs
        return self.output_path


def _plan_with_utterance(paths: Path) -> RenderPlan:
    profile_a = VoiceProfile(name="Speaker A", speaker="A", reference_audio=[], engine_params=VoiceSettings())
    profile_b = VoiceProfile(name="Speaker B", speaker="B", reference_audio=[], engine_params=VoiceSettings())
    utterances = [
        Utterance(
            index=0,
            original_text="Hello there.",
            repaired_text="Hello there.",
            speaker="A",
            duration_seconds=1.5,
            pause_after_ms=180,
        )
    ]
    return RenderPlan(
        title="srt test",
        source_path="",
        output_dir=str(paths),
        engine="chatterbox",
        correction_mode="moderate",
        metadata={"model_variant": "standard"},
        utterances=utterances,
        voice_profiles={"A": profile_a, "B": profile_b},
    )


def test_render_settings_metadata_carries_export_srt_flag(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window.export_srt_check.setChecked(False)
        assert window._render_settings().metadata.get("export_srt") == ""
        window.export_srt_check.setChecked(True)
        assert window._render_settings().metadata.get("export_srt") == "1"
    finally:
        window.close()


def test_export_srt_flag_round_trips_through_gui_settings(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window.export_srt_check.setChecked(True)
        payload = window._current_gui_settings_payload()
        window.export_srt_check.setChecked(False)
        window._apply_gui_settings_payload(payload)
        assert window.export_srt_check.isChecked() is True

        window.export_srt_check.setChecked(False)
        payload = window._current_gui_settings_payload()
        window._apply_gui_settings_payload(payload)
        assert window.export_srt_check.isChecked() is False
    finally:
        window.close()


def test_render_worker_writes_srt_next_to_flac_when_enabled(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from the_oracle.app_gui import RenderWorker

    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        flac_path = tmp_path / "render_out.flac"
        settings = RenderSettings(
            model_variant="standard",
            metadata={"output_filename": "render_out.flac", "export_srt": "1"},
        )
        worker = RenderWorker(
            _plan_with_utterance(paths.output_dir),
            settings,
            pipeline=_FakeRenderer(flac_path),
        )
        completed: dict[str, object] = {}
        worker.completed.connect(lambda payload, path: completed.update(payload=payload, path=path))
        worker.failed.connect(lambda _payload, message: pytest.fail(f"render failed unexpectedly: {message}"))

        worker.run()

        srt_path = tmp_path / "render_out.srt"
        assert srt_path.exists()
        content = srt_path.read_text(encoding="utf-8")
        assert "A: Hello there." in content
        assert "00:00:00,000 --> 00:00:01,500" in content
        assert completed["path"] == str(flac_path)
        assert completed["payload"]["metadata"].get("srt_path") == str(srt_path)
    finally:
        window.close()


def test_render_worker_forces_sequential_execution_for_gui_render(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RenderWorker must not spawn native PyTorch workers from the Qt process."""
    from the_oracle.app_gui import RenderWorker

    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        renderer = _FakeRenderer(tmp_path / "render_out.flac")
        worker = RenderWorker(
            _plan_with_utterance(paths.output_dir),
            RenderSettings(model_variant="standard"),
            pipeline=renderer,
        )
        worker.run()
        assert renderer.render_kwargs["force_sequential"] is True
    finally:
        window.close()


def test_render_worker_fallback_uses_gui_safe_pipeline(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A directly constructed worker must keep native imports out of Qt threads.

    This is the regression for the Render-button crash: before the GUI injected
    its safe pipeline, the worker's fallback constructed the feature-rich
    pipeline in the QThread and could crash while importing native backends.
    """
    import the_oracle.app_gui as app_gui
    from the_oracle.app_gui import RenderWorker

    renderer = _FakeRenderer(tmp_path / "render_out.flac")
    pipeline_kwargs: dict[str, object] = {}

    def make_pipeline(**kwargs):
        pipeline_kwargs.update(kwargs)
        return renderer

    monkeypatch.setattr(app_gui, "OraclePipeline", make_pipeline)
    worker = RenderWorker(
        _plan_with_utterance(tmp_path),
        RenderSettings(model_variant="standard"),
    )
    completed: list[str] = []
    failed: list[str] = []
    worker.completed.connect(lambda _payload, path: completed.append(path))
    worker.failed.connect(lambda _payload, message: failed.append(message))

    worker.run()

    assert failed == []
    assert completed == [str(tmp_path / "render_out.flac")]
    assert pipeline_kwargs == {
        "use_transformers": False,
        "use_language_tool": False,
        "use_punctuation_model": False,
    }
    assert renderer.render_kwargs["force_sequential"] is True


def test_failed_render_preserves_partial_row_state(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed worker must not discard statuses recorded before the failure."""
    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        plan = _plan_with_utterance(paths.output_dir)
        plan.utterances[0].status = "success"
        plan.utterances[0].duration_seconds = 1.25
        plan.metadata["failed_rows"] = "2"
        window.plan = RenderPlan.from_dict(plan.to_dict())
        window.render_worker = SimpleNamespace(plan=plan)
        monkeypatch.setattr(
            "the_oracle.app_gui.QMessageBox.critical",
            lambda *_args, **_kwargs: None,
        )

        window._fail_render(plan.to_dict(), "Partial render: one or more synthesis chunks failed.")

        assert window.plan is not None
        assert window.plan.utterances[0].status == "success"
        assert window.plan.utterances[0].duration_seconds == 1.25
        assert window.plan.metadata["failed_rows"] == "2"
    finally:
        window.render_worker = None
        window.close()


def test_render_child_environment_uses_the_managed_runtime(qt_app, tmp_path: Path) -> None:
    """The isolated child must inherit the venv/runtime used by the GUI."""
    import subprocess
    import sys

    from the_oracle.app_gui import _render_child_environment

    env = _render_child_environment(tmp_path)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import chatterbox, symspellpy, torch, perth; print('imports-ok')",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr
    assert "imports-ok" in probe.stdout
    assert env["PYTHONNOUSERSITE"] == "1"
    assert str(tmp_path / "src") in env["PYTHONPATH"].split(os.pathsep)


def test_render_worker_uses_a_new_process_session_for_native_children(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cancelling a render must stop the child process group, not just bash."""
    import io
    import the_oracle.app_gui as app_gui
    from the_oracle.app_gui import RenderWorker

    class _RunningProcess:
        stdout = io.StringIO("")
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self):
            return self.returncode

    process = _RunningProcess()
    popen_kwargs: dict[str, object] = {}

    def fake_popen(*_args, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(app_gui.subprocess, "Popen", fake_popen)
    worker = RenderWorker(
        _plan_with_utterance(tmp_path),
        RenderSettings(model_variant="standard"),
        subprocess_job=(tmp_path / "job.json", tmp_path / "result.json"),
        python_executable="python",
        repo_root=tmp_path,
    )
    worker.run()

    assert popen_kwargs["start_new_session"] is True


def test_render_worker_windows_cancel_kills_the_child_process_tree(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Windows cancellation must terminate descendants, not only the wrapper."""
    import the_oracle.app_gui as app_gui
    from the_oracle.app_gui import RenderWorker

    class _RunningProcess:
        pid = 4321

        def poll(self):
            return None

        def wait(self, timeout=None):
            return -15

        def terminate(self):
            raise AssertionError("Windows cancellation should use taskkill first")

        def kill(self):
            return None

    calls: list[list[str]] = []
    monkeypatch.setattr(app_gui.os, "name", "nt")
    monkeypatch.setattr(
        app_gui.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )
    worker = RenderWorker(
        _plan_with_utterance(tmp_path),
        RenderSettings(model_variant="standard"),
        repo_root=tmp_path,
    )
    worker._process = _RunningProcess()

    worker.request_cancel()

    assert calls == [["taskkill", "/PID", "4321", "/T", "/F"]]


def test_render_worker_surfaces_native_child_crash_without_crashing_qt(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A native Chatterbox crash must become a GUI error, not kill the Qt process.

    Real Chatterbox/Perth initialization segfaults when run in the Qt process
    after QMediaPlayer is initialized. The GUI worker therefore owns a child
    render process; this regression verifies that an exit-by-SIGSEGV is turned
    into the worker's normal failure signal with diagnostic output.
    """
    import io
    import the_oracle.app_gui as app_gui
    from the_oracle.app_gui import RenderWorker

    class _CrashedProcess:
        returncode = -11
        stdout = io.StringIO("before model load\\nperth warning\\n")

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    process = _CrashedProcess()
    monkeypatch.setattr(app_gui.subprocess, "Popen", lambda *args, **kwargs: process)
    worker = RenderWorker(
        _plan_with_utterance(tmp_path),
        RenderSettings(model_variant="standard"),
        subprocess_job=(tmp_path / "job.json", tmp_path / "result.json"),
        python_executable="python",
    )
    failures: list[str] = []
    worker.failed.connect(lambda _payload, message: failures.append(message))

    worker.run()

    assert failures
    assert "SIGSEGV" in failures[0]
    assert "perth warning" in failures[0]


def test_render_worker_skips_srt_when_disabled(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from the_oracle.app_gui import RenderWorker

    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        flac_path = tmp_path / "render_out.flac"
        settings = RenderSettings(
            model_variant="standard",
            metadata={"output_filename": "render_out.flac", "export_srt": ""},
        )
        worker = RenderWorker(
            _plan_with_utterance(paths.output_dir),
            settings,
            pipeline=_FakeRenderer(flac_path),
        )
        completed: dict[str, object] = {}
        worker.completed.connect(lambda payload, path: completed.update(payload=payload, path=path))
        worker.failed.connect(lambda _payload, message: pytest.fail(f"render failed unexpectedly: {message}"))

        worker.run()

        assert not (tmp_path / "render_out.srt").exists()
        assert completed["path"] == str(flac_path)
    finally:
        window.close()


def test_loading_saved_project_restores_export_srt_checkbox(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        plan = _plan_with_utterance(paths.output_dir)
        settings = RenderSettings(
            model_variant="standard",
            metadata={"output_filename": "render_out.flac", "export_srt": "1"},
        )
        voice = VoiceSettings()
        speakers = {
            "A": SpeakerSettings(reference_path="", voice_settings=voice),
            "B": SpeakerSettings(reference_path="", voice_settings=voice),
        }
        saved = build_saved_project(plan, settings, speakers)

        window.export_srt_check.setChecked(False)
        window._load_project_into_ui(saved)

        assert window.export_srt_check.isChecked() is True
    finally:
        window.close()


def test_preview_worker_runs_in_subprocess_and_never_inits_model_inline(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preview must isolate model init in a child process like render does.

    Loading Chatterbox/PyTorch inside the Qt Multimedia process segfaults on
    Ubuntu (SIGSEGV, surfaced as exit 245/139), so the GUI preview worker
    delegates synthesis to a clean interpreter instead of calling the
    pipeline's ``render_preview`` in the QThread.
    """
    import io
    import json
    import the_oracle.app_gui as app_gui
    from the_oracle.app_gui import PreviewWorker

    class _NoInlinePipeline:
        def render_preview(self, *_args, **_kwargs):
            raise AssertionError("preview must run in the isolated child, not the Qt process")

    popen_kwargs: dict[str, object] = {}

    class _DoneProcess:
        stdout = io.StringIO("")
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    def fake_popen(command, **kwargs):
        popen_kwargs.update(kwargs)
        popen_kwargs["command"] = command
        result_index = command.index("--result")
        Path(command[result_index + 1]).write_text(
            json.dumps({"ok": True, "preview_path": str(tmp_path / "preview.wav")}),
            encoding="utf-8",
        )
        return _DoneProcess()

    monkeypatch.setattr(app_gui.subprocess, "Popen", fake_popen)
    utterance = Utterance(index=0, original_text="Hi.", repaired_text="Hi.", speaker="A")
    profile = VoiceProfile(name="Speaker A", speaker="A", reference_audio=[])
    worker = PreviewWorker(
        utterance,
        profile,
        "standard",
        "cpu",
        pipeline=_NoInlinePipeline(),
        subprocess_job=(tmp_path / "job.json", tmp_path / "result.json"),
        python_executable="python",
        repo_root=tmp_path,
    )
    completed: list[str] = []
    worker.completed.connect(completed.append)
    worker.failed.connect(lambda message: pytest.fail(f"preview failed unexpectedly: {message}"))

    worker.run()

    assert completed == [str(tmp_path / "preview.wav")]
    assert "--preview" in popen_kwargs["command"]
    assert popen_kwargs["start_new_session"] is True
    job_payload = json.loads((tmp_path / "job.json").read_text(encoding="utf-8"))
    assert job_payload["utterance"]["repaired_text"] == "Hi."
    assert job_payload["profile"]["speaker"] == "A"
    assert job_payload["model_variant"] == "standard"


def test_preview_worker_surfaces_native_child_crash_without_crashing_qt(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A native Chatterbox crash during preview becomes a normal failure."""
    import io
    import the_oracle.app_gui as app_gui
    from the_oracle.app_gui import PreviewWorker

    class _CrashedProcess:
        returncode = -11
        stdout = io.StringIO("before model load\nperth warning\n")

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(app_gui.subprocess, "Popen", lambda *args, **kwargs: _CrashedProcess())
    utterance = Utterance(index=0, original_text="Hi.", repaired_text="Hi.", speaker="A")
    profile = VoiceProfile(name="Speaker A", speaker="A", reference_audio=[])
    worker = PreviewWorker(
        utterance,
        profile,
        "standard",
        "cpu",
        subprocess_job=(tmp_path / "job.json", tmp_path / "result.json"),
        python_executable="python",
    )
    failures: list[str] = []
    worker.failed.connect(failures.append)

    worker.run()

    assert failures
    assert "SIGSEGV" in failures[0]
    assert "perth warning" in failures[0]


def test_gui_preview_opts_into_isolated_child_process(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The GUI's Preview button must request subprocess isolation."""
    import the_oracle.app_gui as app_gui

    window, paths = _build_window(monkeypatch, tmp_path)
    try:
        plan = _plan_with_utterance(paths.output_dir)
        window.plan = plan
        window._populate_table(plan)

        captured: dict[str, object] = {}

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
        assert kwargs["run_in_subprocess"] is True
        assert kwargs["repo_root"] is not None
        assert kwargs["python_executable"]
    finally:
        window.close()
