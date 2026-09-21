"""GUI render/preview execution: workers, child-process plumbing, progress dialog.

Owns the code the Render/Preview buttons put in motion (2026-09-20 extraction
slice): :class:`RenderWorker` and :class:`PreviewWorker` run synthesis in an
isolated child process that has never initialized Qt (native torch/Chatterbox
segfaults inside the Qt Multimedia process), :func:`_render_child_environment`
builds that child's runtime environment, and :class:`RenderProgressDialog`
shows live render progress.

``app_gui`` re-exports all four names, so MainWindow call-sites, imports, and
test doubles keep resolving them from there. One rule the patch-surface test
(test_app_gui_patch_surface.MOVED_OWNERS) now enforces: tests that patch
``OraclePipeline`` for the workers' direct (non-subprocess) path must patch it
HERE — the worker bodies resolve module globals from this module, so an
``app_gui``-level patch is a silent no-op for them.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import time
import json
import os
import signal
import subprocess
import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout, QWidget

from the_oracle.models.project import RenderPlan, Utterance, VoiceProfile
from the_oracle.models.settings import RenderSettings
from the_oracle.pipeline import OraclePipeline, RenderProgress


def _render_child_environment(repo_root: Path) -> dict[str, str]:
    """Build the exact runtime environment for an isolated render child.

    The GUI may be launched through the managed ``the-oracle`` entry point,
    while the render child is started from a Qt worker thread.  Do not rely on
    either process's current directory or ambient ``PYTHONPATH``: explicitly
    expose this checkout's source tree and keep the managed venv's site-packages
    visible.  ``PYTHONNOUSERSITE`` matches the managed launcher so a user-site
    package cannot shadow the installed runtime.
    """
    env = os.environ.copy()
    src_path = str(Path(repo_root).resolve() / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


class RenderWorker(QThread):
    progress = Signal(object)
    completed = Signal(object, str)
    # Include the worker's defensive plan copy so partial row outcomes survive
    # even if Qt delivers the finished/cleanup slot before the failure slot.
    failed = Signal(object, str)

    def __init__(
        self,
        plan: RenderPlan,
        settings: RenderSettings,
        *,
        pipeline: OraclePipeline | None = None,
        prewarmed_engine=None,
        render_click_wall: float | None = None,
        run_in_subprocess: bool = False,
        subprocess_job: tuple[Path, Path] | None = None,
        python_executable: str | None = None,
        repo_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.plan = RenderPlan.from_dict(plan.to_dict())
        self.settings = deepcopy(settings)
        self._pipeline = pipeline
        self._prewarmed_engine = prewarmed_engine
        self._render_click_wall = render_click_wall
        self._run_in_subprocess = run_in_subprocess or subprocess_job is not None
        self._subprocess_job = subprocess_job
        self._python_executable = python_executable or sys.executable
        self._repo_root = repo_root or Path(__file__).resolve().parents[2]
        self._process: subprocess.Popen[str] | None = None
        self._temporary_dir = None
        self._cancel_requested = False
        self._child_output: list[str] = []

    @staticmethod
    def _terminate_child_process(process) -> None:
        """Terminate a render child and wait for it without touching Qt."""
        if process is None:
            return
        try:
            if process.poll() is None:
                if os.name == "posix":
                    # start_new_session=True makes the child PID its process
                    # group ID, so native torch descendants cannot outlive it.
                    os.killpg(process.pid, signal.SIGTERM)
                else:  # pragma: no cover - exercised on Windows installations
                    # CREATE_NEW_PROCESS_GROUP makes the child independently
                    # addressable, but terminate() alone does not recursively
                    # stop torch/native descendants on Windows. taskkill /T
                    # targets this process tree without affecting unrelated
                    # processes.
                    result = subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    if getattr(result, "returncode", 0) != 0:
                        process.terminate()
        except (OSError, AttributeError):
            try:
                process.terminate()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError, AttributeError, TypeError):
            try:
                process.kill()
            except (OSError, AttributeError):
                pass
            try:
                process.wait()
            except (OSError, AttributeError, TypeError):
                pass

    def request_cancel(self) -> None:
        """Stop the isolated render process and all of its native descendants."""
        self._cancel_requested = True
        self._terminate_child_process(self._process)

    @staticmethod
    def _signal_description(returncode: int) -> str:
        if returncode < 0:
            try:
                return signal.Signals(-returncode).name
            except ValueError:
                return f"signal {-returncode}"
        if returncode >= 128:
            try:
                return signal.Signals(returncode - 128).name
            except ValueError:
                return f"signal {returncode - 128}"
        return ""

    def _run_subprocess_render(self) -> tuple[dict, str]:
        """Run native synthesis in a process that has never initialized Qt.

        PyTorch/Perth reproducibly segfaults when model initialization happens
        in a QThread after Qt Multimedia has been created. Keeping the complete
        pipeline in this child process isolates that native failure; the GUI
        process can then turn any signal exit into a normal error message.
        """
        import tempfile

        output_dir = Path(self.plan.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_dir: tempfile.TemporaryDirectory[str] | None = None
        if self._subprocess_job is None:
            temporary_dir = tempfile.TemporaryDirectory(prefix=".oracle-render-", dir=str(output_dir))
            self._temporary_dir = temporary_dir
            job_path = Path(temporary_dir.name) / "job.json"
            result_path = Path(temporary_dir.name) / "result.json"
        else:
            job_path, result_path = self._subprocess_job
            job_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.parent.mkdir(parents=True, exist_ok=True)

        settings_payload = asdict(self.settings)
        settings_payload.pop("anchors", None)
        job_path.write_text(
            json.dumps(
                {
                    "plan": self.plan.to_dict(),
                    "settings": settings_payload,
                    "render_click_wall": self._render_click_wall or time(),
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        env = _render_child_environment(self._repo_root)
        command = [
            self._python_executable,
            "-m",
            "the_oracle.render_subprocess",
            "--job",
            str(job_path),
            "--result",
            str(result_path),
        ]
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self._repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                # A dedicated session lets request_cancel() terminate the
                # complete child process group on POSIX, not only the Python
                # wrapper while torch/native descendants keep running.
                start_new_session=(os.name == "posix"),
                **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}),
            )
            self._process = process
            stdout = process.stdout
            if stdout is not None:
                for line in stdout:
                    line = line.rstrip("\n")
                    if line.startswith("ORACLE_RENDER_PROGRESS "):
                        try:
                            self.progress.emit(RenderProgress(**json.loads(line.split(" ", 1)[1])))
                        except (TypeError, ValueError, json.JSONDecodeError) as exc:
                            self._child_output.append(f"Malformed progress event: {exc}: {line}")
                    elif line:
                        self._child_output.append(line)
            returncode = process.wait()
        except BaseException:
            # If stdout parsing, a signal handler, or a Qt callback interrupts
            # this worker, never leave a native child running in the background.
            self._terminate_child_process(process)
            raise
        finally:
            self._process = None

        child_result: dict = {}
        if result_path.exists():
            try:
                child_result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self._child_output.append(f"Could not read child render result: {exc}")
        payload = child_result.get("plan") if isinstance(child_result.get("plan"), dict) else self.plan.to_dict()
        output_path = str(child_result.get("output_path") or "")
        if returncode != 0 or child_result.get("ok") is not True or not output_path:
            signal_name = self._signal_description(returncode)
            if self._cancel_requested:
                reason = "Render cancelled."
            elif signal_name:
                reason = f"Render subprocess terminated by {signal_name} (native crash isolated from the GUI)."
            else:
                reason = f"Render subprocess failed with exit code {returncode}."
            details = "\n".join(self._child_output[-20:])
            if child_result.get("error"):
                reason += f"\n{child_result['error']}"
            if details:
                reason += f"\nChild output:\n{details}"
            raise RuntimeError(reason)
        return payload, output_path

    def run(self) -> None:
        try:
            if self._run_in_subprocess:
                plan_payload, output_path = self._run_subprocess_render()
                self.plan = RenderPlan.from_dict(plan_payload)
            else:
                # Direct workers remain useful for tests and injected renderer
                # doubles. The real GUI always uses the isolated child process.
                pipeline = self._pipeline or OraclePipeline(
                    use_transformers=False,
                    use_language_tool=False,
                    use_punctuation_model=False,
                )
                output_path = pipeline.render(
                    self.plan,
                    self.settings,
                    progress_callback=self.progress.emit,
                    prewarmed_engine=self._prewarmed_engine,
                    render_click_wall=self._render_click_wall or time(),
                    force_sequential=True,
                )
            if self.settings.metadata.get("export_srt"):
                from the_oracle.audio.export_srt import write_srt
                from the_oracle.subtitle_targets import subtitle_sidecar_target

                srt_path = write_srt(subtitle_sidecar_target(output_path), self.plan.utterances)
                self.plan.metadata["srt_path"] = str(srt_path)
        except Exception as exc:
            self.failed.emit(self.plan.to_dict(), str(exc))
            return
        finally:
            temporary_dir = self._temporary_dir
            self._temporary_dir = None
            if temporary_dir is not None:
                temporary_dir.cleanup()
        self.completed.emit(self.plan.to_dict(), str(output_path))


class PreviewWorker(QThread):
    progress = Signal(object)
    completed = Signal(str)  # preview_path only
    failed = Signal(str)

    def __init__(
        self,
        utterance: Utterance,
        profile: VoiceProfile,
        model_variant: str,
        device_mode: str,
        *,
        pipeline: OraclePipeline | None = None,
        inference_backend: str = "pytorch",
        cuda_device: int | None = None,
        audio_cpp_device: int | None = None,
        audio_cpp_threads: int | None = None,
        audio_cpp_timeout: int | None = None,
        audio_cpp_max_batch: int | None = None,
        run_in_subprocess: bool = False,
        subprocess_job: tuple[Path, Path] | None = None,
        python_executable: str | None = None,
        repo_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.utterance = Utterance.from_dict(utterance.to_dict())
        self.profile = VoiceProfile.from_dict(profile.to_dict())
        self.model_variant = model_variant
        self.device_mode = device_mode
        self.inference_backend = inference_backend
        self.cuda_device = cuda_device
        self.audio_cpp_device = audio_cpp_device
        self.audio_cpp_threads = audio_cpp_threads
        self.audio_cpp_timeout = audio_cpp_timeout
        self.audio_cpp_max_batch = audio_cpp_max_batch
        self._pipeline = pipeline
        self._run_in_subprocess = run_in_subprocess or subprocess_job is not None
        self._subprocess_job = subprocess_job
        self._python_executable = python_executable or sys.executable
        self._repo_root = repo_root or Path(__file__).resolve().parents[2]
        self._process: subprocess.Popen[str] | None = None
        self._temporary_dir = None
        self._cancel_requested = False
        self._child_output: list[str] = []

    def request_cancel(self) -> None:
        """Stop the isolated preview process and all of its native descendants."""
        self._cancel_requested = True
        RenderWorker._terminate_child_process(self._process)

    def _run_subprocess_preview(self) -> str:
        """Run the preview in a process that has never initialized Qt.

        Preview inherits the render child's safety model: initializing
        Chatterbox/PyTorch inside the Qt Multimedia process segfaults on
        Ubuntu (SIGSEGV, surfaced as exit 245/139), so the model load and
        synthesis happen in a clean interpreter that only touches native
        torch/audio code, never Qt.
        """
        import tempfile

        temporary_dir: tempfile.TemporaryDirectory[str] | None = None
        if self._subprocess_job is None:
            temporary_dir = tempfile.TemporaryDirectory(prefix=".oracle-preview-")
            self._temporary_dir = temporary_dir
            job_path = Path(temporary_dir.name) / "job.json"
            result_path = Path(temporary_dir.name) / "result.json"
        else:
            job_path, result_path = self._subprocess_job
            job_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.parent.mkdir(parents=True, exist_ok=True)

        job_path.write_text(
            json.dumps(
                {
                    "utterance": self.utterance.to_dict(),
                    "profile": self.profile.to_dict(),
                    "model_variant": self.model_variant,
                    "device_mode": self.device_mode,
                    "inference_backend": self.inference_backend,
                    "cuda_device": self.cuda_device,
                    "audio_cpp_device": self.audio_cpp_device,
                    "audio_cpp_threads": self.audio_cpp_threads,
                    "audio_cpp_timeout": self.audio_cpp_timeout,
                    "audio_cpp_max_batch": self.audio_cpp_max_batch,
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        env = _render_child_environment(self._repo_root)
        command = [
            self._python_executable,
            "-m",
            "the_oracle.render_subprocess",
            "--preview",
            "--job",
            str(job_path),
            "--result",
            str(result_path),
        ]
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self._repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                # A dedicated session lets request_cancel() terminate the
                # complete child process group on POSIX, not only the Python
                # wrapper while torch/native descendants keep running.
                start_new_session=(os.name == "posix"),
                **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}),
            )
            self._process = process
            stdout = process.stdout
            if stdout is not None:
                for line in stdout:
                    line = line.rstrip("\n")
                    if line.startswith("ORACLE_RENDER_PROGRESS "):
                        try:
                            self.progress.emit(RenderProgress(**json.loads(line.split(" ", 1)[1])))
                        except (TypeError, ValueError, json.JSONDecodeError) as exc:
                            self._child_output.append(f"Malformed progress event: {exc}: {line}")
                    elif line:
                        self._child_output.append(line)
            returncode = process.wait()
        except BaseException:
            # Never leave a native child running behind a live Qt thread.
            RenderWorker._terminate_child_process(process)
            raise
        finally:
            self._process = None

        child_result: dict = {}
        if result_path.exists():
            try:
                child_result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self._child_output.append(f"Could not read child preview result: {exc}")
        preview_path = str(child_result.get("preview_path") or "")
        if returncode != 0 or child_result.get("ok") is not True or not preview_path:
            signal_name = RenderWorker._signal_description(returncode)
            if self._cancel_requested:
                reason = "Preview cancelled."
            elif signal_name:
                reason = f"Preview subprocess terminated by {signal_name} (native crash isolated from the GUI)."
            else:
                reason = f"Preview subprocess failed with exit code {returncode}."
            details = "\n".join(self._child_output[-20:])
            if child_result.get("error"):
                reason += f"\n{child_result['error']}"
            if details:
                reason += f"\nChild output:\n{details}"
            raise RuntimeError(reason)
        return preview_path

    def run(self) -> None:
        try:
            if self._run_in_subprocess:
                preview_path = self._run_subprocess_preview()
            else:
                # Direct workers remain useful for tests and injected renderer
                # doubles. The real GUI always uses the isolated child process.
                # Do not fall back to feature-rich OraclePipeline here:
                # PreviewWorker can be constructed directly by a caller, and the
                # GUI-safe fallback must remain safe even when pipeline
                # injection is omitted.
                pipeline = self._pipeline or OraclePipeline(
                    use_transformers=False,
                    use_language_tool=False,
                    use_punctuation_model=False,
                )
                preview_path = pipeline.render_preview(
                    self.utterance,
                    self.profile,
                    self.model_variant,
                    device_mode=self.device_mode,
                    inference_backend=self.inference_backend,
                    cuda_device=self.cuda_device,
                    audio_cpp_device=self.audio_cpp_device,
                    audio_cpp_threads=self.audio_cpp_threads,
                    audio_cpp_timeout=self.audio_cpp_timeout,
                    audio_cpp_max_batch=self.audio_cpp_max_batch,
                    progress_callback=self.progress.emit,
                )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        finally:
            temporary_dir = self._temporary_dir
            self._temporary_dir = None
            if temporary_dir is not None:
                temporary_dir.cleanup()
        # Emit only the preview path - preview does not mutate row-level render state
        self.completed.emit(str(preview_path))


class RenderProgressDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, *, title: str = "Rendering") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        self.backend_label = QLabel("Backend: ...")
        self.synth_label = QLabel("")
        self.stage_label = QLabel("Starting render...")
        self.segment_label = QLabel("Segments: 0/0")
        self.eta_label = QLabel("ETA: calculating...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.backend_label)
        layout.addWidget(self.synth_label)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.segment_label)
        layout.addWidget(self.eta_label)
        layout.addWidget(self.progress_bar)
        self.reset()

    def reset(self) -> None:
        self.progress_bar.setValue(0)
        self.backend_label.setText("Backend: ...")
        self.synth_label.setText("")
        self.stage_label.setText("Starting render...")
        self.segment_label.setText("Segments: 0/0")
        self.eta_label.setText("ETA: calculating...")

    def _backend_panel_text(self, progress: RenderProgress) -> str:
        """Live backend/device line: the active inference backend plus the
        device it renders on (the GPU name for Vulkan, CPU for PyTorch)."""
        if not progress.backend:
            return "Backend: ..."
        label = "Vulkan (audio.cpp)" if progress.backend == "vulkan" else "PyTorch"
        if progress.device_label:
            label += f" — {progress.device_label}"
        return f"Backend: {label}"

    def update_progress(self, progress: RenderProgress) -> None:
        # Time-weighted fraction (when the pipeline supplies one) drives the
        # bar smoothly through model load and synthesis; fall back to the
        # old step-count math for progress payloads without a fraction.
        if progress.fraction is not None:
            percent = int(round(progress.fraction * 100))
        else:
            percent = 0 if progress.total_steps <= 0 else int(round((progress.current_step / progress.total_steps) * 100))
        self.progress_bar.setValue(max(0, min(100, percent)))
        self.backend_label.setText(self._backend_panel_text(progress))
        if progress.synth_seconds_total is not None:
            text = f"Render time: {self._format_seconds(progress.synth_seconds_total)} total"
            if progress.synth_seconds_latest is not None:
                text += f" · last {self._format_seconds(progress.synth_seconds_latest)}"
            self.synth_label.setText(text)
        self.stage_label.setText(f"{progress.stage}: {progress.detail}")
        if progress.total_segments > 0:
            self.segment_label.setText(f"Segments: {progress.current_segment}/{progress.total_segments}")
        elif progress.total_steps > 0:
            self.segment_label.setText(f"Steps: {progress.current_step}/{progress.total_steps}")
        else:
            self.segment_label.setText("Segments: preparing...")
        if progress.eta_seconds is None:
            self.eta_label.setText(f"Elapsed: {self._format_seconds(progress.elapsed_seconds)} | ETA: calculating...")
        else:
            self.eta_label.setText(
                f"Elapsed: {self._format_seconds(progress.elapsed_seconds)} | ETA: {self._format_seconds(progress.eta_seconds)}"
            )

    @staticmethod
    def _format_seconds(value: float) -> str:
        seconds = max(0, int(round(value)))
        minutes, seconds = divmod(seconds, 60)
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"


