"""PySide6 desktop GUI for the Chatterbox-only The Oracle app."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter, time
import json
import os
import sys
import signal
import subprocess
import threading

from PySide6.QtCore import QThread, Qt, QUrl, Signal, QTimer
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from the_oracle.app_paths import (
    OraclePaths,
    ensure_repo_default_paths,
    normalize_output_filename,
    default_output_filename,
    resolve_output_filename,
)
from the_oracle.correction_modes import CORRECTION_MODE_OPTIONS, correction_mode_label, normalize_correction_mode
from the_oracle.emotion.goemotions import SUPPORTED_EMOTIONS
from the_oracle.gui_settings import (
    GUISettingsError,
    list_templates,
    load_app_settings,
    load_gui_settings,
    load_recent_reference_paths,
    load_template,
    remember_recent_reference_path,
    save_app_settings,
    save_gui_settings,
    save_template,
)
from the_oracle.gui_tooltips import install_ctrl_hover_help
from the_oracle.models.project import RenderPlan, VoiceProfile, VoiceSettings, Utterance
from the_oracle.pipeline import OraclePipeline, RenderProgress, RenderSettings, SpeakerSettings
from the_oracle.project_manifest import build_saved_project, load_project_manifest, save_project_manifest
from the_oracle.voice_catalog import VoiceChoice, default_voice_choices
from the_oracle.tts_engines.chatterbox_engine import SUPPORTED_VARIANTS, ChatterboxEngine
from the_oracle.tts_engines.vulkan_backend import AudioCppUnavailableError, AudioCppVulkanEngine, find_audiocpp_binary
from the_oracle.vulkan_setup import parse_model_export, run_vulkan_setup, vulkan_setup_needed

# CPU is the only verified Chatterbox execution path in this project.
# Preview and render always use "cpu"; the constant is defined here so
# it can be updated in one place if a verified GPU path is added later.
_DEVICE_MODE: str = "cpu"


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


def _device_row_text(index: int, name: str) -> str:
    """Human label for one Vulkan device, shared by the dropdown items and
    the picker's summary label so the two can never drift apart."""
    return f"Device {index}: {name}"


def _vulkan_prerequisite_missing() -> list[str]:
    """Return human-readable reasons the Vulkan backend is not ready, else [].

    Delegates to :func:`the_oracle.vulkan_setup.vulkan_setup_needed` (the
    single source of truth shared with the CLI), passing this module's own
    ``find_audiocpp_binary`` reference so tests that monkeypatch it keep
    working. Used to auto-start the setup at backend-selection time instead of
    failing deep inside the render worker.
    """
    return vulkan_setup_needed(find_binary=find_audiocpp_binary)


def _vulkan_preflight_report(device_index: int | None) -> str:
    """Run the audio.cpp preflight and return a human-readable report.

    Checks the binary and model (same checks as ``_vulkan_prerequisite_missing``),
    then lists the Vulkan devices audio.cpp sees and states which GPU a render
    would use (the selected ``device_index``, or audio.cpp's default when None).
    Raises :class:`AudioCppUnavailableError` when the setup is not ready.
    """
    missing = _vulkan_prerequisite_missing()
    if missing:
        raise AudioCppUnavailableError(
            "Vulkan backend preflight failed. Missing: " + "; ".join(missing) + "."
        )
    engine = AudioCppVulkanEngine(device_index=device_index)
    devices = engine.list_devices()
    if not devices:
        # The binary and model are present, but no GPU is visible to audio.cpp.
        # Rendering would fail, so this must be a failure, not a "passed"
        # report — the button exists to validate setup before rendering.
        raise AudioCppUnavailableError(
            "audio.cpp --list-devices reported no Vulkan devices. A Vulkan driver/"
            "device must be visible before rendering on the Vulkan backend."
        )
    lines = ["Vulkan backend preflight passed."]
    if device_index is None:
        lines.append("GPU to be used: Auto (audio.cpp picks its default device)")
    else:
        lines.append(f"GPU to be used: Vulkan device {device_index}")
    lines.append("Devices audio.cpp sees:")
    for item in devices:
        marker = " (selected)" if device_index == item["index"] else ""
        lines.append(f"  Device {item['index']}: {item['name']}{marker}")
    return "\n".join(lines)


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

                srt_path = write_srt(Path(output_path).with_suffix(".srt"), self.plan.utterances)
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


class VulkanDeviceProbeThread(QThread):
    """Probe audio.cpp's Vulkan devices off the GUI thread.

    Emits ``devices`` with ``[{"index", "name"}, ...]`` on success and
    ``failed`` with the error message when the binary cannot be probed (e.g.
    audiocpp_cli is not built). The result populates the Vulkan Device picker
    so users can choose the right ``ORACLE_AUDIOCPP_DEVICE`` index.
    """

    devices = Signal(object)
    failed = Signal(str)

    def run(self) -> None:
        try:
            result = AudioCppVulkanEngine().list_devices()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.devices.emit(result)


class VulkanPreflightThread(QThread):
    """Run a quick audio.cpp preflight (binary + model + --list-devices) off the
    GUI thread so the 'Test Vulkan backend' button never blocks the UI.

    Emits ``completed`` with the human-readable report on success and
    ``failed`` with the error message when the setup is not ready.
    """

    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QWidget | None = None, *, device_index: int | None = None) -> None:
        super().__init__(parent)
        self._device_index = device_index

    def run(self) -> None:
        try:
            report = _vulkan_preflight_report(self._device_index)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(report)


_MODEL_DOWNLOAD_TIMEOUT = 1800.0  # large GGUF downloads can take a while


def _parse_oracle_model_path(output: str) -> str:
    """Extract the installed model path from the download script's output.

    Delegates to :func:`the_oracle.vulkan_setup.parse_model_export` (the
    single source of truth shared with the auto-setup orchestrator).
    """
    return parse_model_export(output)


class ModelDownloadThread(QThread):
    """Run scripts/download_audio_cpp_model.sh off the GUI thread.

    Model downloads are large and slow, so the UI must never block on them.
    Emits ``completed`` with the installed model path (the
    ``ORACLE_AUDIOCPP_MODEL`` value the script prints) on success and
    ``failed`` with the captured output otherwise. Uses ``Popen`` so the
    running subprocess can be terminated on app close instead of destroying a
    still-running QThread (which Qt aborts on).
    """

    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, script: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._script = script
        self._proc: subprocess.Popen[str] | None = None

    def request_cancel(self) -> None:
        """Terminate the download subprocess tree if it is still running.

        Called from the GUI thread (e.g. closeEvent) while ``run`` blocks in
        ``communicate``. The script spawns a python child (the model manager)
        that inherits our stdout/stderr pipes, so killing only the direct bash
        child would leave python holding the pipes open and ``communicate``
        blocked; SIGTERM the whole process group instead so ``wait`` returns.
        """
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass

    def run(self) -> None:
        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(
                ["bash", str(self._script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # POSIX-only (start_new_session / os.killpg in request_cancel);
                # the Vulkan audio.cpp path is Linux-only, so this is fine, and
                # on Windows Popen raises here and run() surfaces a clear error.
                start_new_session=True,
            )
            self._proc = proc
            try:
                stdout, stderr = proc.communicate(timeout=_MODEL_DOWNLOAD_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                self.failed.emit("Model download timed out (the download process was stopped).")
                return
        except Exception as exc:
            self.failed.emit(f"Failed to run {self._script}: {exc}")
            return
        finally:
            self._proc = None
        assert proc is not None  # every failure path returned above
        output = f"{stdout}\n{stderr}"
        if proc.returncode != 0:
            self.failed.emit(output.strip() or f"download script exited with {proc.returncode}")
            return
        model_path = _parse_oracle_model_path(output)
        if not model_path:
            self.failed.emit("Download finished but no ORACLE_AUDIOCPP_MODEL export line was printed.")
            return
        self.completed.emit(model_path)


class VulkanSetupThread(QThread):
    """Run the automatic CPU→GPU (Vulkan backend) setup off the GUI thread.

    When the Vulkan backend is selected but its prerequisites are missing
    (audiocpp_cli not built, and/or the Chatterbox model not downloaded), the
    GUI kicks this thread off instead of just warning: it builds the CLI if
    needed, downloads the model if needed, and sets ORACLE_AUDIOCPP_CLI /
    ORACLE_AUDIOCPP_MODEL for the session (see
    :func:`the_oracle.vulkan_setup.run_vulkan_setup`).

    Emits ``progress`` for each script output line, ``completed`` with the
    :class:`VulkanSetupResult` on success, and ``failed`` with the error
    message otherwise. Uses the same Popen + process-group cancel pattern as
    :class:`ModelDownloadThread` so a running build/download can be terminated
    on app close instead of destroying a running QThread (which Qt aborts on).
    """

    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, repo_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo_root = repo_root
        self._cancel = threading.Event()

    def request_cancel(self) -> None:
        """Signal the running setup script(s) to stop (SIGTERM the group)."""
        self._cancel.set()

    def run(self) -> None:
        try:
            result = run_vulkan_setup(
                progress=self.progress.emit,
                cancel=self._cancel,
                repo_root=self._repo_root,
            )
        except Exception as exc:
            self.failed.emit(f"Vulkan backend setup crashed: {exc}")
            return
        if result.ok:
            self.completed.emit(result)
        else:
            self.failed.emit(result.error or "Vulkan backend setup failed.")


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
        label = "Vulkan (audio.cpp)" if progress.backend == "vulkan" else "PyTorch (CPU)"
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


class SpeakerGroup(QGroupBox):
    def __init__(self, speaker: str, custom_reference_dir: Path) -> None:
        super().__init__(f"Speaker {speaker}")
        self.custom_reference_dir = custom_reference_dir
        self.reference_path = QLineEdit()
        self.reference_picker = QComboBox()
        self.reference_picker.currentIndexChanged.connect(self._handle_reference_selection)
        # Use activated so selecting the already-current custom option still fires.
        self.reference_picker.activated.connect(self._handle_reference_selection)
        self._available_reference_paths: set[str] = set()

        self.language_combo = QComboBox()
        self.cfg_weight = self._double_box(0.0, 1.5, 0.5, 0.05)
        self.exaggeration = self._double_box(0.0, 1.5, 0.5, 0.05)
        self.temperature = self._double_box(0.1, 1.5, 0.8, 0.05)
        self.emotion_intensity = self._double_box(0.0, 2.0, 1.0, 0.1)
        self.naturalness = self._double_box(0.0, 1.0, 0.0, 0.05)
        self.pause_spin = QSpinBox()
        self.pause_spin.setRange(0, 2000)
        self.pause_spin.setValue(180)

        form = QFormLayout(self)
        self.form = form  # kept so Ctrl+hover help can register row labels
        form.addRow("Custom Voice Reference Audio", self.reference_picker)
        form.addRow("Language", self.language_combo)
        form.addRow("CFG Weight", self.cfg_weight)
        form.addRow("Exaggeration", self.exaggeration)
        form.addRow("Temperature", self.temperature)
        form.addRow("Emotion Intensity", self.emotion_intensity)
        form.addRow("Naturalness (Heuristic)", self.naturalness)
        form.addRow("Pause After Speaker Turn (ms)", self.pause_spin)

    def _double_box(self, minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(2)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _pick_audio(self) -> None:
        current_reference = Path(self.reference_path.text()).expanduser()
        start_dir = current_reference.parent if current_reference.exists() else self.custom_reference_dir
        path, _ = QFileDialog.getOpenFileName(self, "Choose Reference Audio", str(start_dir), "Audio Files (*.wav *.flac *.mp3)")
        if path:
            self.reference_path.setText(path)

    def set_language_options(self, languages: dict[str, str], enabled: bool) -> None:
        selected = self.language_combo.currentData() or "en"
        self.language_combo.clear()
        for code, name in languages.items():
            self.language_combo.addItem(f"{code} - {name}", code)
        index = self.language_combo.findData(selected if enabled else "en")
        if index < 0:
            index = self.language_combo.findData("en")
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        self.language_combo.setEnabled(enabled)

    def set_reference_choices(self, defaults: list[VoiceChoice], recents: list[str], selected_path: str = "") -> None:
        current_path = selected_path or self.reference_path.text()
        self.reference_picker.blockSignals(True)
        self.reference_picker.clear()
        self._available_reference_paths = set()
        if defaults:
            header_index = self.reference_picker.count()
            self.reference_picker.addItem("Default Voices")
            header_item = self.reference_picker.model().item(header_index)
            if header_item is not None:
                header_item.setEnabled(False)
            for voice in defaults[:10]:
                self.reference_picker.addItem(f"  {voice.label}", voice.path)
                self._available_reference_paths.add(voice.path)
        if recents:
            header_index = self.reference_picker.count()
            self.reference_picker.addItem("Recent Custom Clips")
            header_item = self.reference_picker.model().item(header_index)
            if header_item is not None:
                header_item.setEnabled(False)
            for path in recents[:10]:
                resolved = str(Path(path).expanduser())
                self.reference_picker.addItem(f"  {Path(resolved).name}", resolved)
                self._available_reference_paths.add(resolved)
        self.reference_picker.addItem("Custom Voice Reference Audio...", "__custom__")
        target_index = self.reference_picker.findData(current_path)
        if target_index < 0:
            target_index = self.reference_picker.findData("__custom__")
        self.reference_picker.setCurrentIndex(target_index)
        self.reference_picker.blockSignals(False)
        if current_path in self._available_reference_paths:
            self.reference_path.setText(current_path)

    def _handle_reference_selection(self, _index=None) -> None:
        data = self.reference_picker.currentData()
        if data == "__custom__":
            self._pick_audio()
            return
        if isinstance(data, str) and data:
            self.reference_path.setText(data)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._startup_t0 = perf_counter()
        # App-level settings (remembered inference backend + resolved audio.cpp
        # paths) are loaded before the UI so the menu action can reflect them;
        # they are applied to the widgets in _on_gui_shown. Persistence of
        # changes is gated until then so the initial default apply never writes
        # over the remembered choice.
        self._app_settings = load_app_settings()
        self._app_settings_ready = False
        self._startup_marks: list[tuple[str, float]] = []
        self._gui_shown_wall: float | None = None
        self.repo_root = Path(__file__).resolve().parents[2]
        self.paths: OraclePaths = ensure_repo_default_paths(self.repo_root)
        self.pipeline: OraclePipeline | None = None
        self._prewarmed_pipeline: OraclePipeline | None = None
        self._prewarmed_engine = None
        self._prewarm_state = "not_started"  # not_started, warming, ready, failed
        self._prewarm_thread: PrewarmThread | None = None
        self._prewarm_lock = threading.Lock()
        self._prewarm_timing: dict[str, float] | None = None
        self.plan: RenderPlan | None = None
        self.current_project_path: Path | None = None
        self.render_worker: RenderWorker | None = None
        self.preview_worker: PreviewWorker | None = None
        self._vulkan_probe_thread: VulkanDeviceProbeThread | None = None
        self._vulkan_devices_probed = False
        self._vulkan_preflight_thread: VulkanPreflightThread | None = None
        self._model_download_thread: ModelDownloadThread | None = None
        # Automatic CPU→GPU setup: when the Vulkan backend is selected but its
        # prerequisites are missing, the GUI builds audiocpp_cli / downloads the
        # model in the background (VulkanSetupThread) and queues any render or
        # preview that arrived in the meantime, so the switch completes by itself.
        self._vulkan_setup_thread: VulkanSetupThread | None = None
        self._vulkan_setup_attempted = False
        self._render_queued_after_setup = False
        self._preview_queued_after_setup = False
        # The row whose preview was queued behind the setup, so the completion
        # handler can re-fire that exact preview without the user re-clicking.
        self._preview_row_queued_after_setup: int | None = None
        self._preflight_queued_after_setup = False
        self.progress_dialog: RenderProgressDialog | None = None
        self.preview_dialog: RenderProgressDialog | None = None
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.setWindowTitle("The Oracle")
        self.resize(1320, 900)
        self._mark_startup("mainwindow_init_begin")
        self._ctrl_help = install_ctrl_hover_help(QApplication.instance())
        self._build_ui()
        self._build_menu()
        self._register_ctrl_help_descriptions()
        self.delete_confirm_enabled = True
        self._apply_gui_settings_payload(self._default_gui_settings_payload())
        self._mark_startup("mainwindow_init_end")
        self._write_startup_timeline()
        QTimer.singleShot(0, self._on_gui_shown)

    def _mark_startup(self, label: str) -> None:
        self._startup_marks.append((label, perf_counter() - self._startup_t0))

    def _write_startup_timeline(self) -> None:
        try:
            log_dir = self.paths.output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            payload = {"events": self._startup_marks}
            (log_dir / "gui_startup_timing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_gui_shown(self) -> None:
        self._gui_shown_wall = time()
        # Restore the remembered inference backend (and audio.cpp paths) before
        # anything else runs, so a previous Vulkan session picks up right where
        # it left off. If the persisted paths are stale, the normal
        # selection-time prerequisite check auto-starts the setup again.
        self._apply_remembered_backend()
        # Persistence of backend/knob changes is only enabled after the
        # restore above, so the restore itself can never overwrite the stored
        # choice with widget defaults.
        self._app_settings_ready = True
        self._start_prewarm()

    def _log_action_timing(self, label: str, wall: float | None = None, extra: dict | None = None) -> None:
        try:
            log_dir = self.paths.output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            payload_path = log_dir / "gui_action_timing.json"
            existing = json.loads(payload_path.read_text()) if payload_path.exists() else []
            entry = {"label": label, "wall": wall or time()}
            if extra:
                entry.update(extra)
            existing.append(entry)
            payload_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)

        controls = QGridLayout()
        self.input_path = QLineEdit()
        self.outdir_path = QLineEdit()
        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("Auto-derived from the input file when using the default Output folder")
        self.input_path.textChanged.connect(self._handle_outdir_changed)
        self.outdir_path.textChanged.connect(self._handle_outdir_changed)
        self._path_row_labels: list[QLabel | None] = [None, None]
        self._path_row_buttons: list[QPushButton | None] = [None, None]
        self._add_path_row(controls, 0, "Input", self.input_path, self._pick_input)
        self._add_path_row(controls, 1, "Output Folder", self.outdir_path, self._pick_outdir)
        self.output_name_label = QLabel("Output Filename")
        controls.addWidget(self.output_name_label, 2, 0)
        controls.addWidget(self.output_name, 2, 1, 1, 2)
        layout.addLayout(controls)

        settings_row = QHBoxLayout()
        settings_row.addWidget(self._build_project_settings())
        self.speaker_a = SpeakerGroup("A", self.paths.voice_dir)
        self.speaker_b = SpeakerGroup("B", self.paths.voice_dir)
        # Extra character voices (C..X) for audiobook casts. They are created
        # lazily when a plan detects more than two speakers; the layout holds
        # them in a scrollable column so a 24-voice cast stays usable.
        self.extra_speaker_groups: dict[str, SpeakerGroup] = {}
        self.extra_speaker_scroll = QScrollArea()
        self.extra_speaker_scroll.setWidgetResizable(True)
        self.extra_speaker_scroll.setFixedWidth(420)
        self.extra_speaker_container = QWidget()
        self.extra_speaker_layout = QVBoxLayout(self.extra_speaker_container)
        self.extra_speaker_layout.setContentsMargins(0, 0, 0, 0)
        self.extra_speaker_scroll.setWidget(self.extra_speaker_container)
        self.extra_speaker_scroll.hide()
        settings_row.addWidget(self.speaker_a)
        settings_row.addWidget(self.speaker_b)
        settings_row.addWidget(self.extra_speaker_scroll)
        layout.addLayout(settings_row)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        actions.addStretch(1)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.prepare_project)
        self._style_action_button(self.analyze_button)
        self.render_button = QPushButton("Render FLAC")
        self.render_button.clicked.connect(self.render_project)
        self._style_action_button(self.render_button, accent=True)
        actions.addWidget(self.analyze_button)
        actions.addWidget(self.render_button)
        layout.addLayout(actions)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "Index",
            "Speaker",
            "Original Text",
            "Repaired Text",
            "Emotion",
            "Duration",
            "Status",
            "Preview",
            "+/-",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, stretch=1)

        self.error_panel = QTextEdit()
        self.error_panel.setReadOnly(True)
        self.error_panel.setPlaceholderText("Status and model errors appear here.")
        self.status_label = QLabel("Status / Errors")
        layout.addWidget(self.status_label)
        layout.addWidget(self.error_panel)

        self.setCentralWidget(root)
        self.outdir_path.setText(str(self.paths.output_dir))
        self._refresh_language_options()
        self._refresh_reference_pickers()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        new_action = QAction("New Project", self)
        new_action.triggered.connect(self.new_project)
        open_action = QAction("Open Project", self)
        open_action.triggered.connect(self.open_project)
        save_action = QAction("Save Project", self)
        save_action.triggered.connect(self.save_project)
        save_as_action = QAction("Save Project As", self)
        save_as_action.triggered.connect(self.save_project_as)
        for action in (new_action, open_action, save_action, save_as_action):
            file_menu.addAction(action)

        settings_menu = self.menuBar().addMenu("Settings")
        reset_defaults_action = QAction("Reset to Defaults", self)
        reset_defaults_action.triggered.connect(self.reset_settings_to_defaults)
        save_settings_action = QAction("Save Settings...", self)
        save_settings_action.triggered.connect(self.save_settings_profile)
        load_settings_action = QAction("Load Settings...", self)
        load_settings_action.triggered.connect(self.load_settings_profile)
        save_template_action = QAction("Save Current as Template...", self)
        save_template_action.triggered.connect(self.save_template_profile)
        settings_menu.addAction(reset_defaults_action)
        settings_menu.addSeparator()
        settings_menu.addAction(save_settings_action)
        settings_menu.addAction(load_settings_action)
        settings_menu.addSeparator()
        settings_menu.addAction(save_template_action)

        self.templates_menu = settings_menu.addMenu("Load Template")
        self.templates_menu.aboutToShow.connect(self._rebuild_templates_menu)
        self.confirmation_action = QAction("Re-enable delete confirmations", self)
        self.confirmation_action.triggered.connect(self._enable_delete_confirmation)
        settings_menu.addSeparator()
        settings_menu.addAction(self.confirmation_action)
        settings_menu.addSeparator()
        self.remember_backend_action = QAction("Remember GPU/CPU choice", self)
        self.remember_backend_action.setCheckable(True)
        self.remember_backend_action.setChecked(bool(self._app_settings.get("remember_backend", True)))
        self.remember_backend_action.setToolTip(
            "Remember which inference backend was chosen (PyTorch CPU or Vulkan "
            "GPU) plus the audiocpp_cli and Chatterbox model paths, and restore "
            "them automatically at next launch so the GPU needs no setup again. "
            "Uncheck to always start on PyTorch (CPU)."
        )
        # toggled (not triggered): PySide6's triggered binding here takes no
        # checked argument, so a real menu click would not deliver the bool to
        # the handler. toggled(bool) always does, and nothing programmatically
        # toggles the action after the setChecked above (which precedes the
        # connect), so no spurious early emission is possible.
        self.remember_backend_action.toggled.connect(self._on_remember_backend_toggled)
        settings_menu.addAction(self.remember_backend_action)
        self.download_vulkan_model_action = QAction("Download Vulkan Model...", self)
        self.download_vulkan_model_action.setToolTip(
            "Run scripts/download_audio_cpp_model.sh in the background and report "
            "the resulting ORACLE_AUDIOCPP_MODEL path to use for the Vulkan backend."
        )
        self.download_vulkan_model_action.triggered.connect(self.download_vulkan_model)
        settings_menu.addAction(self.download_vulkan_model_action)

        # Keep references for Ctrl+hover help registration (see
        # _register_ctrl_help_descriptions).
        self.new_action = new_action
        self.open_action = open_action
        self.save_action = save_action
        self.save_as_action = save_as_action
        self.reset_defaults_action = reset_defaults_action
        self.save_settings_action = save_settings_action
        self.load_settings_action = load_settings_action
        self.save_template_action = save_template_action

    def _register_ctrl_help_descriptions(self) -> None:
        """Register Ctrl+hover help text for the main controls.

        Holding the left Control key while hovering a control pops up a short
        description of what it does and how to use it (see
        the_oracle.gui_tooltips). Buttons, fields, spin boxes, backend knobs,
        speaker controls, and menu actions are covered; unregistered widgets
        fall back to their regular Qt tooltip if one is set.
        """
        ctrl_help = self._ctrl_help
        if ctrl_help is None:
            return
        ctrl_help.register_many([
            (
                self.input_path,
                "Input script or transcript file to convert into dialogue audio. "
                "Click Browse or type a path; Analyze then reads, repairs, and "
                "attributes the lines.",
            ),
            (
                self.outdir_path,
                "Output folder for the rendered FLAC (plus stems and optional "
                "SRT). Defaults to the repo's Output/ folder; typing a new "
                "path creates it.",
            ),
            (
                self.output_name,
                "Filename for the final render, without the extension. Leave "
                "blank to auto-derive from the input file name.",
            ),
            (
                self.analyze_button,
                "Analyze the input: repair text, attribute lines to speakers "
                "A and B, detect emotions, and fill the review table. No "
                "audio is rendered yet.",
            ),
            (
                self.render_button,
                "Render the full dialogue to FLAC: synthesize every utterance "
                "with the selected inference backend, assemble the two "
                "speakers, and write the output file.",
            ),
            (
                self.table,
                "Review table, one row per utterance: index, speaker, "
                "original/repaired text, emotion (editable), duration, "
                "status, preview, and stem on/off.",
            ),
            (
                self.error_panel,
                "Status and error log. Render progress, Vulkan preflight "
                "reports, model downloads, and failures are appended here.",
            ),
        ])
        ctrl_help.register_many([
            (
                self.variant_combo,
                "Chatterbox model variant: standard (default), multilingual "
                "(per-speaker language), or turbo (faster; PyTorch-only, so "
                "it disables the Vulkan backend).",
            ),
            (
                self.correction_mode_combo,
                "How aggressively the text-repair pass fixes grammar, "
                "spelling, punctuation, and directives before synthesis.",
            ),
            (
                self.loudness_combo,
                "Loudness normalization applied to the final render: off, "
                "light, or medium.",
            ),
            (
                self.crossfade_spin,
                "Crossfade in milliseconds between consecutive utterances so "
                "speaker turns blend smoothly.",
            ),
            (
                self.export_srt_check,
                "Also write a .srt subtitle file next to the rendered FLAC, "
                "one cue per utterance.",
            ),
            (
                self.inference_backend_combo,
                "Which engine synthesizes audio: PyTorch (CPU, in-process "
                "Chatterbox) or Vulkan (opt-in; shells out to audio.cpp, "
                "needs audiocpp_cli + model).",
            ),
            (
                self.test_vulkan_button,
                "Run a quick audio.cpp preflight (binary + model + device "
                "list) and report which GPU the Vulkan backend would use, "
                "before rendering.",
            ),
            (
                self.vulkan_prerequisite_warning,
                "Inline warning shown when the Vulkan backend is selected but "
                "audiocpp_cli or the Chatterbox model is not configured yet.",
            ),
            (
                self.audio_cpp_device_combo,
                "Vulkan device passed to audio.cpp (--device). Auto lets "
                "audio.cpp pick; the dropdown lists detected devices. "
                "Equivalent to ORACLE_AUDIOCPP_DEVICE.",
            ),
            (
                self.audio_cpp_device_label,
                "Read-only summary of the Vulkan devices audio.cpp detected; "
                "the dropdown above is populated from this list.",
            ),
            (
                self.audio_cpp_threads_spin,
                "Thread count passed to audio.cpp (--threads). Default lets "
                "audio.cpp decide. Equivalent to ORACLE_AUDIOCPP_THREADS.",
            ),
            (
                self.audio_cpp_timeout_spin,
                "Per-synthesis timeout in seconds for audio.cpp. Default (0) "
                "uses audio.cpp's 600 s. Equivalent to ORACLE_AUDIOCPP_TIMEOUT.",
            ),
            (
                self.audio_cpp_max_batch_spin,
                "Maximum cache-missing stems per audio.cpp --request-sequence "
                "subprocess. Default (0) uses the engine's 32-request cap. "
                "Equivalent to ORACLE_AUDIOCPP_MAX_BATCH.",
            ),
        ])
        for group in self._all_speaker_groups().values():
            ctrl_help.register_many([
                (
                    group.reference_picker,
                    "Voice reference audio this speaker clones: a default "
                    "voice, a recent custom clip, or a new file you choose.",
                ),
                (
                    group.language_combo,
                    "Spoken language for this speaker. Only selectable on the "
                    "multilingual variant; otherwise fixed to English.",
                ),
                (
                    group.cfg_weight,
                    "CFG guidance weight for the voice clone (0.0-1.5). "
                    "Higher values follow the reference voice more strictly.",
                ),
                (
                    group.exaggeration,
                    "How much the speaker's emotion and intonation are "
                    "exaggerated (0.0-1.5).",
                ),
                (
                    group.temperature,
                    "Sampling temperature (0.1-1.5). Higher is more varied "
                    "delivery; lower is steadier and closer to the reference.",
                ),
                (
                    group.emotion_intensity,
                    "Strength applied to the detected emotion for this "
                    "speaker (0.0-2.0).",
                ),
                (
                    group.naturalness,
                    "Heuristic naturalness boost applied after synthesis "
                    "(0.0-1.0).",
                ),
                (
                    group.pause_spin,
                    "Pause in milliseconds inserted after this speaker's "
                    "turns.",
                ),
            ])
        ctrl_help.register_many([
            (self.new_action, "Start a new project: clears the review table and plan while keeping the current shared and speaker settings."),
            (self.open_action, "Load a saved project manifest, restoring the plan, review table, and settings."),
            (self.save_action, "Save the current project to its manifest file."),
            (self.save_as_action, "Save the current project to a new manifest file."),
            (self.reset_defaults_action, "Restore every setting to its built-in default."),
            (self.save_settings_action, "Export the current GUI settings (shared + speakers) to a profile file."),
            (self.load_settings_action, "Apply a previously saved GUI settings profile."),
            (self.save_template_action, "Save the current settings as a named template for the Load Template menu."),
            (self.confirmation_action, "Re-enable the delete-confirmation prompt for removing review-table rows."),
            (self.remember_backend_action, "Remember which inference backend was chosen (PyTorch CPU or Vulkan GPU) plus the audiocpp_cli and Chatterbox model paths, and restore them automatically at next launch so the GPU needs no setup again. Uncheck to always start on PyTorch (CPU)."),
            (self.download_vulkan_model_action, "Fetch the audio.cpp Chatterbox model in the background and report the ORACLE_AUDIOCPP_MODEL path to use."),
        ])
        # The row labels and Browse buttons share their field's description,
        # so hovering the *name* of a control works too (the user asked for
        # "names, selections, buttons, toggles, or other areas of interest").
        ctrl_help.register(self._path_row_labels[0], ctrl_help.description_for(self.input_path))
        ctrl_help.register(self._path_row_buttons[0], "Open a file picker to choose the input script or transcript.")
        ctrl_help.register(self._path_row_labels[1], ctrl_help.description_for(self.outdir_path))
        ctrl_help.register(self._path_row_buttons[1], "Open a folder picker to choose where the rendered FLAC is written.")
        ctrl_help.register(self.output_name_label, ctrl_help.description_for(self.output_name))
        ctrl_help.register(self.status_label, ctrl_help.description_for(self.error_panel))
        # Form row labels (e.g. "Model Variant", "CFG Weight") get the same
        # help as their field.
        ctrl_help.register_form_labels(self._project_settings_form, [
            self.variant_combo,
            self.correction_mode_combo,
            self.loudness_combo,
            self.crossfade_spin,
            self.inference_backend_combo,
            self.audio_cpp_device_combo,
            self.audio_cpp_threads_spin,
            self.audio_cpp_timeout_spin,
            self.audio_cpp_max_batch_spin,
        ])
        # The Inference Backend row's field is a layout, so its label is not
        # covered by register_form_labels; register it explicitly.
        backend_label = self._project_settings_form.labelForField(self._inference_backend_row)
        if backend_label is not None:
            ctrl_help.register(backend_label, ctrl_help.description_for(self.inference_backend_combo))
        for group in self._all_speaker_groups().values():
            ctrl_help.register_form_labels(group.form, [
                group.reference_picker,
                group.language_combo,
                group.cfg_weight,
                group.exaggeration,
                group.temperature,
                group.emotion_intensity,
                group.naturalness,
                group.pause_spin,
            ])
        menubar_actions = self.menuBar().actions()
        if len(menubar_actions) >= 1:
            ctrl_help.register_action(menubar_actions[0], "Project management: new, open, save, save-as.")
        if len(menubar_actions) >= 2:
            ctrl_help.register_action(menubar_actions[1], "GUI settings: profiles, templates, and Vulkan backend setup.")

    def _build_project_settings(self) -> QGroupBox:
        box = QGroupBox("Shared Render Settings")
        form = QFormLayout(box)
        self._project_settings_form = form  # kept so Ctrl+hover help can register row labels
        self.variant_combo = QComboBox()
        self.variant_combo.addItems(list(SUPPORTED_VARIANTS))
        self.variant_combo.currentTextChanged.connect(self._refresh_language_options)
        self.correction_mode_combo = QComboBox()
        for label, value in CORRECTION_MODE_OPTIONS:
            self.correction_mode_combo.addItem(label, value)
        self._set_correction_mode(RenderSettings().correction_mode)
        self.loudness_combo = QComboBox()
        self.loudness_combo.addItems(["off", "light", "medium"])
        self.loudness_combo.setCurrentText(RenderSettings().loudness_preset)
        self.crossfade_spin = QSpinBox()
        self.crossfade_spin.setRange(0, 500)
        self.crossfade_spin.setValue(RenderSettings().crossfade_ms)
        self.export_srt_check = QCheckBox("Export SRT subtitles")
        self.export_srt_check.setToolTip("Write a .srt subtitle file next to the rendered FLAC, one cue per utterance.")
        self.monologue_check = QCheckBox("Monologue (single narrator voice)")
        self.monologue_check.setToolTip(
            "Render the entire input in one narrator voice (Speaker A), ignoring "
            "per-line attribution. Use this to read a book aloud as a single "
            "narrator instead of a cast of characters."
        )
        self.inference_backend_combo = QComboBox()
        self.inference_backend_combo.addItem("PyTorch (CPU)", "pytorch")
        self.inference_backend_combo.addItem("Vulkan (audio.cpp)", "vulkan")
        self.inference_backend_combo.setCurrentIndex(0)
        self.inference_backend_combo.setToolTip(
            "Inference backend for render and preview. PyTorch (CPU) is the default "
            "in-process Chatterbox path. Vulkan (audio.cpp) is the GPU path: selecting "
            "it automatically builds audiocpp_cli and downloads the Chatterbox model "
            "if missing, then sets the env vars for the session (see README "
            "'Vulkan Backend')."
        )
        self.test_vulkan_button = QPushButton("Test Vulkan Backend")
        self.test_vulkan_button.setToolTip(
            "Run a quick audio.cpp preflight (binary + model + --list-devices) and "
            "report which GPU the Vulkan backend would use, before rendering."
        )
        self.test_vulkan_button.clicked.connect(self.test_vulkan_backend)
        self.test_vulkan_button.setEnabled(False)
        self.vulkan_prerequisite_warning = QLabel("")
        self.vulkan_prerequisite_warning.setWordWrap(True)
        self.vulkan_prerequisite_warning.setStyleSheet("color: #b45309;")  # amber warning
        self.vulkan_prerequisite_warning.hide()
        self.audio_cpp_device_combo = QComboBox()
        self.audio_cpp_device_combo.addItem("Auto (audio.cpp default)", None)
        self.audio_cpp_device_combo.setToolTip(
            "Vulkan device passed to audio.cpp as --device <N> on multi-GPU "
            "machines. Auto lets audio.cpp pick; the dropdown is populated from "
            "'audiocpp_cli --backend vulkan --list-devices' (or the doctor). "
            "The value is equivalent to ORACLE_AUDIOCPP_DEVICE but needs no "
            "environment variable."
        )
        self.audio_cpp_device_label = QLabel("")
        self.audio_cpp_device_label.setWordWrap(True)
        self.audio_cpp_device_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.audio_cpp_device_label.setToolTip(
            "The Vulkan devices audio.cpp detects (from audiocpp_cli --list-devices). "
            "The dropdown lists them by name; picking one is equivalent to "
            "ORACLE_AUDIOCPP_DEVICE."
        )
        self.audio_cpp_threads_spin = QSpinBox()
        self.audio_cpp_threads_spin.setRange(0, 128)
        self.audio_cpp_threads_spin.setValue(0)
        self.audio_cpp_threads_spin.setSpecialValueText("Default (audio.cpp's own)")
        self.audio_cpp_threads_spin.setToolTip(
            "Thread count passed to audio.cpp as --threads <N>. Default lets "
            "audio.cpp decide; the value is equivalent to ORACLE_AUDIOCPP_THREADS "
            "but needs no environment variable."
        )
        self.audio_cpp_timeout_spin = QSpinBox()
        self.audio_cpp_timeout_spin.setRange(0, 3600)
        self.audio_cpp_timeout_spin.setValue(0)
        self.audio_cpp_timeout_spin.setSpecialValueText("Default (600s)")
        self.audio_cpp_timeout_spin.setToolTip(
            "Per-synthesis timeout in seconds passed to audio.cpp. Default (0) uses "
            "audio.cpp's 600s; the value is equivalent to ORACLE_AUDIOCPP_TIMEOUT "
            "but needs no environment variable."
        )
        self.audio_cpp_max_batch_spin = QSpinBox()
        self.audio_cpp_max_batch_spin.setRange(0, 1024)
        self.audio_cpp_max_batch_spin.setValue(0)
        self.audio_cpp_max_batch_spin.setSpecialValueText("Default (32)")
        self.audio_cpp_max_batch_spin.setToolTip(
            "Maximum cache-missing stems per audio.cpp --request-sequence subprocess. "
            "Default (0) uses the engine's 32-request cap; the value is equivalent "
            "to ORACLE_AUDIOCPP_MAX_BATCH but needs no environment variable."
        )
        self.variant_combo.currentTextChanged.connect(self._refresh_inference_backend_options)
        self.inference_backend_combo.currentIndexChanged.connect(self._refresh_audio_cpp_knob_options)
        # Keep the remembered backend choice and Vulkan knobs in sync with the
        # widgets so the next launch restores exactly what the user last had
        # selected. The handlers are gated on _app_settings_ready, so the
        # initial default apply (and the launch-time restore itself) never
        # writes over the stored values.
        self.inference_backend_combo.currentIndexChanged.connect(self._persist_remembered_settings)
        self.audio_cpp_device_combo.currentIndexChanged.connect(self._persist_remembered_settings)
        self.audio_cpp_threads_spin.valueChanged.connect(self._persist_remembered_settings)
        self.audio_cpp_timeout_spin.valueChanged.connect(self._persist_remembered_settings)
        self.audio_cpp_max_batch_spin.valueChanged.connect(self._persist_remembered_settings)
        form.addRow("Model Variant", self.variant_combo)
        backend_row = QHBoxLayout()
        backend_row.addWidget(self.inference_backend_combo, 1)
        backend_row.addWidget(self.test_vulkan_button, 0)
        form.addRow("Inference Backend", backend_row)
        # The row's field is a layout, so labelForField() must be given the
        # layout (not the combo) to find the "Inference Backend" label; kept
        # for Ctrl+hover label registration.
        self._inference_backend_row = backend_row
        form.addRow("", self.vulkan_prerequisite_warning)
        form.addRow("Vulkan Device", self.audio_cpp_device_combo)
        form.addRow("", self.audio_cpp_device_label)
        form.addRow("Vulkan Threads", self.audio_cpp_threads_spin)
        form.addRow("Vulkan Timeout (s)", self.audio_cpp_timeout_spin)
        form.addRow("Vulkan Max Batch", self.audio_cpp_max_batch_spin)
        form.addRow("Correction Mode", self.correction_mode_combo)
        form.addRow("Loudness", self.loudness_combo)
        form.addRow("Crossfade (ms)", self.crossfade_spin)
        form.addRow("", self.export_srt_check)
        form.addRow("", self.monologue_check)
        return box

    def _set_correction_mode(self, value: str) -> None:
        normalized = normalize_correction_mode(value)
        idx = self.correction_mode_combo.findData(normalized)
        if idx < 0:
            idx = self.correction_mode_combo.findData(normalize_correction_mode("moderate"))
        if idx >= 0:
            self.correction_mode_combo.setCurrentIndex(idx)

    def _add_path_row(self, layout: QGridLayout, row: int, label: str, field: QLineEdit, callback) -> None:
        label_widget = QLabel(label)
        button = QPushButton("Browse")
        button.clicked.connect(callback)
        layout.addWidget(label_widget, row, 0)
        layout.addWidget(field, row, 1)
        layout.addWidget(button, row, 2)
        # Keep references so Ctrl+hover help can describe the row label and
        # its Browse button (the user asked for hovering "names" too).
        self._path_row_labels[row] = label_widget
        self._path_row_buttons[row] = button

    def _style_action_button(self, button: QPushButton, accent: bool = False) -> None:
        button.setMinimumHeight(48)
        button.setMinimumWidth(170 if not accent else 210)
        palette = (
            "background-color: #19466d; color: white; border: 1px solid #133652;"
            if accent
            else "background-color: #f4f7fa; color: #12263a; border: 1px solid #9fb0c0;"
        )
        button.setStyleSheet(f"font-size: 15px; font-weight: 600; padding: 8px 18px; border-radius: 6px; {palette}")

    def _pick_input(self) -> None:
        current_text = self.input_path.text().strip()
        if not current_text:
            start_dir = self.paths.input_dir
        else:
            current_input = Path(current_text).expanduser()
            start_dir = current_input.parent if current_input.exists() else self.paths.input_dir
        path, _ = QFileDialog.getOpenFileName(self, "Choose Input", str(start_dir), "Text Files (*.txt *.md)")
        if path:
            self.input_path.setText(path)

    def _handle_outdir_changed(self) -> None:
        folder = Path(self.outdir_path.text() or self.paths.output_dir).expanduser()
        default_folder = Path(self.paths.output_dir)
        if folder == default_folder:
            return
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        if not self.output_name.text().strip():
            default_name = default_output_filename(self.input_path.text() or "")
            self.output_name.setText(default_name)

    def _pick_outdir(self) -> None:
        current_outdir = Path(self.outdir_path.text()).expanduser()
        start_dir = current_outdir if current_outdir.exists() else self.paths.output_dir
        path = QFileDialog.getExistingDirectory(self, "Choose Output Directory", str(start_dir))
        if path:
            self.outdir_path.setText(path)

    def _refresh_language_options(self) -> None:
        variant = self.variant_combo.currentText() if hasattr(self, "variant_combo") else "standard"
        languages = {"en": "English"} if variant != "multilingual" else ChatterboxEngine(variant).supported_languages()
        is_multilingual = variant == "multilingual"
        for group in self._all_speaker_groups().values():
            group.set_language_options(languages, is_multilingual)

    def _refresh_inference_backend_options(self) -> None:
        """Disable the Vulkan backend option when the turbo variant is selected.

        The turbo Chatterbox variant is PyTorch-only; the Vulkan (audio.cpp)
        backend rejects it with a clear error. Blocking it here keeps the GUI
        from offering an unusable combination.
        """
        if not hasattr(self, "inference_backend_combo"):
            return
        is_turbo = self.variant_combo.currentText() == "turbo"
        vulkan_index = self.inference_backend_combo.findData("vulkan")
        if vulkan_index < 0:
            return
        self.inference_backend_combo.model().item(vulkan_index).setEnabled(not is_turbo)
        if is_turbo and self.inference_backend_combo.currentData() == "vulkan":
            self.inference_backend_combo.setCurrentIndex(self.inference_backend_combo.findData("pytorch"))
        self._refresh_audio_cpp_knob_options()
        self._refresh_vulkan_preflight_button()

    def _refresh_audio_cpp_knob_options(self) -> None:
        """Enable the Vulkan device/threads/timeout knobs only when the Vulkan
        backend is selected; they are ignored (and disabled) on the PyTorch path.
        Selecting Vulkan also kicks off the background device probe that
        populates the device list under the picker, and surfaces a warning if
        the backend's prerequisites are not configured yet."""
        if not hasattr(self, "audio_cpp_device_combo"):
            return
        is_vulkan = self.inference_backend_combo.currentData() == "vulkan"
        self.audio_cpp_device_combo.setEnabled(is_vulkan)
        self.audio_cpp_threads_spin.setEnabled(is_vulkan)
        self.audio_cpp_timeout_spin.setEnabled(is_vulkan)
        self.audio_cpp_max_batch_spin.setEnabled(is_vulkan)
        self.audio_cpp_device_label.setEnabled(is_vulkan)
        if is_vulkan:
            self._start_vulkan_device_probe()
            self._refresh_vulkan_prerequisite_warning()
        else:
            self.vulkan_prerequisite_warning.hide()
        self._refresh_vulkan_preflight_button()

    def _refresh_vulkan_preflight_button(self) -> None:
        """Enable the Test Vulkan Backend button whenever the Vulkan option is
        usable (turbo is PyTorch-only) and no preflight is already running.
        The button works from either backend so setup can be validated before
        switching."""
        if not hasattr(self, "test_vulkan_button"):
            return
        vulkan_index = self.inference_backend_combo.findData("vulkan")
        selectable = (
            vulkan_index >= 0
            and self.inference_backend_combo.model().item(vulkan_index).isEnabled()
            and self._vulkan_preflight_thread is None
        )
        self.test_vulkan_button.setEnabled(selectable)

    def _refresh_vulkan_prerequisite_warning(self) -> None:
        """Auto-start the CPU→GPU switch when the Vulkan backend can't run yet.

        Instead of only warning, this kicks off the background setup (build
        audiocpp_cli / download the model) once per selection, showing live
        progress in the warning label. Once setup finishes the warning clears
        and any queued render/preview proceeds on its own. The attempt guard is
        cleared on failure so an explicit Render/Preview click (the retry
        trigger) or a fresh selection starts a new attempt rather than silently
        failing forever."""
        missing = _vulkan_prerequisite_missing()
        if not missing:
            self.vulkan_prerequisite_warning.hide()
            return
        if self._vulkan_setup_thread is None and not self._vulkan_setup_attempted:
            self.vulkan_prerequisite_warning.setText(
                "Vulkan backend selected: setting it up automatically (building "
                "audiocpp_cli and/or downloading the Chatterbox model). "
                "Progress appears here; rendering/preview will continue by itself "
                "once setup finishes."
            )
            self.vulkan_prerequisite_warning.show()
            self._start_vulkan_setup()
            return
        # A setup already ran (or is running); surface the current state rather
        # than a stale "missing" warning.
        if self._vulkan_setup_thread is not None:
            self.vulkan_prerequisite_warning.setText(
                "Vulkan backend setup is still running; rendering/preview will "
                "continue by itself once it finishes."
            )
            self.vulkan_prerequisite_warning.show()

    def _start_vulkan_setup(self) -> None:
        """Launch the background Vulkan backend setup (build + model download)."""
        if self._vulkan_setup_thread is not None:
            return
        self._vulkan_setup_attempted = True
        self.error_panel.append("Starting automatic Vulkan backend setup (build + model download)...")
        # Parented to the window; on close the subprocess is cancelled and the
        # thread waited on so a running QThread is never destroyed.
        thread = VulkanSetupThread(self.repo_root, self)
        thread.progress.connect(self._handle_vulkan_setup_progress)
        thread.completed.connect(self._handle_vulkan_setup_completed)
        thread.failed.connect(self._handle_vulkan_setup_failed)
        thread.finished.connect(self._cleanup_vulkan_setup_thread)
        self._vulkan_setup_thread = thread
        thread.start()

    def _handle_vulkan_setup_progress(self, line: str) -> None:
        """Stream one setup output line into the warning label."""
        if not line.strip():
            return
        self.vulkan_prerequisite_warning.setText(f"Vulkan backend setup: {line.strip()[-140:]}")
        self.vulkan_prerequisite_warning.show()

    def _handle_vulkan_setup_completed(self, result) -> None:
        """Setup finished: env vars are set for the session, so clear the
        warning, re-probe devices (the binary now exists), and continue any
        queued render/preview."""
        self.error_panel.append("Vulkan backend setup complete.")
        for msg in result.messages:
            self.error_panel.append(f"  {msg}")
        # Remember the resolved audio.cpp paths for future sessions so a later
        # launch can restore the GPU stack without re-running setup. getattr
        # tolerates result objects that predate the binary/model fields (tests).
        self._persist_remembered_settings(
            binary=getattr(result, "binary", None),
            model=getattr(result, "model", None),
        )
        # Prerequisites are now met; clear the attempt guard so a later change
        # that makes them stale again (e.g. the model file deleted) re-triggers
        # the automatic setup instead of silently skipping it.
        self._vulkan_setup_attempted = False
        self._refresh_vulkan_prerequisite_warning()
        if self._vulkan_devices_probed:
            # The binary now exists, so the earlier failed probe may be stale.
            self._vulkan_devices_probed = False
            self._start_vulkan_device_probe()
        if self._render_queued_after_setup:
            self._render_queued_after_setup = False
            # A full render subsumes a queued single-row preview (the two
            # cannot run concurrently), so drop the preview queue instead of
            # leaving it to fire later — no stale state lingers.
            self._preview_queued_after_setup = False
            self._preview_row_queued_after_setup = None
            self.render_project()
        elif self._preview_queued_after_setup:
            # Auto-fire the exact preview the user asked for while setup was
            # running — re-entering preview_utterance starts the worker and
            # shows the preview progress dialog, no second click needed.
            row = self._preview_row_queued_after_setup
            self._preview_queued_after_setup = False
            self._preview_row_queued_after_setup = None
            if row is not None and self.plan is not None and 0 <= row < len(self.plan.utterances):
                self.error_panel.append("Vulkan backend ready: starting the queued preview...")
                self.preview_utterance(row)
            else:
                # The plan/row changed while setup ran (e.g. re-analyzed);
                # never crash on a stale row — just ask for a fresh click.
                self.error_panel.append("Vulkan backend ready: select a row and preview again.")
        if self._preflight_queued_after_setup:
            # A Test Vulkan Backend click arrived while setup was running;
            # prerequisites are now met, so fire the queued preflight on its
            # own (the report appears in the status panel + dialog). Note: if
            # a render/preview was queued at the same time, both fire here —
            # the preflight dialog may appear mid-render, which is acceptable
            # (the user asked for both actions). The preflight also runs even
            # if the user switched back to PyTorch meanwhile; it validates the
            # Vulkan setup regardless of the current selection.
            self._preflight_queued_after_setup = False
            self.test_vulkan_backend()

    def _handle_vulkan_setup_failed(self, message: str) -> None:
        """Surface the failure visibly with the manual fallback commands."""
        self._render_queued_after_setup = False
        self._preview_queued_after_setup = False
        self._preview_row_queued_after_setup = None
        self._preflight_queued_after_setup = False
        # Allow a later explicit Render/Preview click to retry setup; a failed
        # attempt is user-visible (not silently looped), and the user's next
        # action is the retry trigger.
        self._vulkan_setup_attempted = False
        self.error_panel.append(f"Vulkan backend setup failed: {message}")
        self.vulkan_prerequisite_warning.setText(
            "Vulkan backend setup failed: " + message.splitlines()[0][:200]
            + " See the error panel and README 'Vulkan Backend (audio.cpp)'; "
            "run scripts/build_audio_cpp.sh and scripts/download_audio_cpp_model.sh "
            "manually, or switch the Inference Backend back to PyTorch."
        )
        self.vulkan_prerequisite_warning.show()

    def _cleanup_vulkan_setup_thread(self) -> None:
        if self._vulkan_setup_thread is not None:
            self._vulkan_setup_thread.deleteLater()
            self._vulkan_setup_thread = None
        self._refresh_vulkan_preflight_button()

    def _start_vulkan_device_probe(self) -> None:
        """Probe audio.cpp's Vulkan devices once, off the UI thread."""
        if self._vulkan_devices_probed or self._vulkan_probe_thread is not None:
            return
        self.audio_cpp_device_label.setText("Probing audio.cpp devices...")
        # Parented to the window so a close while probing can wait() on it
        # instead of destroying a running QThread (which Qt aborts on).
        thread = VulkanDeviceProbeThread(self)
        thread.devices.connect(self._handle_vulkan_devices)
        thread.failed.connect(self._handle_vulkan_probe_failed)
        thread.finished.connect(self._cleanup_vulkan_probe_thread)
        self._vulkan_probe_thread = thread
        thread.start()

    def _repopulate_audio_cpp_device_combo(self, devices: list) -> None:
        """Rebuild the Vulkan Device dropdown from audio.cpp's detected devices.

        Item 0 is always "Auto (audio.cpp default)". A currently selected
        value survives the rebuild: a detected device stays selected, and a
        stale value (e.g. from a saved manifest) is kept as an explicit
        "(not detected)" row rather than silently changed.
        """
        current = self._audio_cpp_device_value()
        combo = self.audio_cpp_device_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Auto (audio.cpp default)", None)
        for item in devices:
            combo.addItem(_device_row_text(int(item["index"]), item["name"]), int(item["index"]))
        if current is not None:
            index = combo.findData(current)
            if index < 0:
                combo.addItem(_device_row_text(current, "(not detected)"), current)
                index = combo.findData(current)
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _handle_vulkan_devices(self, devices: list) -> None:
        """Populate the Vulkan Device picker from audio.cpp's actual device
        list, so users pick a real GPU by name instead of a blind index range.
        A stale selection (e.g. from a saved manifest) is preserved as an
        explicit "(not detected)" row and surfaced in the label, never applied
        silently."""
        self._vulkan_devices_probed = True
        self._repopulate_audio_cpp_device_combo(devices)
        current = self._audio_cpp_device_value()
        detected = {int(item["index"]) for item in devices}
        note = ""
        if current is not None and current not in detected:
            note = (
                f"\nNote: device {current} is not in the detected list and is "
                f"kept as a custom entry; choose a detected device or Auto."
            )
        if not devices:
            self.audio_cpp_device_label.setText("No Vulkan devices detected by audio.cpp." + note)
            return
        labels = [_device_row_text(int(item["index"]), item["name"]) for item in devices]
        self.audio_cpp_device_label.setText("\n".join(labels) + note)
        detail = " | ".join(f"{item['index']}={item['name']}" for item in devices)
        self.audio_cpp_device_combo.setToolTip(
            f"Detected Vulkan devices:\n{detail}\n\n"
            "Pick the device to use (equivalent to ORACLE_AUDIOCPP_DEVICE); "
            "Auto lets audio.cpp choose."
        )

    def _handle_vulkan_probe_failed(self, message: str) -> None:
        self._vulkan_devices_probed = True
        self.audio_cpp_device_label.setText(f"audio.cpp devices unavailable: {message}")
        self.audio_cpp_device_combo.setToolTip(
            f"Could not list audio.cpp devices: {message}\n\n"
            "Auto lets audio.cpp choose; otherwise pick a device index "
            "(equivalent to ORACLE_AUDIOCPP_DEVICE)."
        )

    def _cleanup_vulkan_probe_thread(self) -> None:
        if self._vulkan_probe_thread is not None:
            self._vulkan_probe_thread.deleteLater()
            self._vulkan_probe_thread = None

    def test_vulkan_backend(self) -> None:
        """Run a quick audio.cpp preflight in a background thread and report
        which GPU the Vulkan backend would use.

        When the backend's prerequisites are missing (or a previous setup is
        still running), the automatic CPU→GPU setup runs first and the
        preflight is queued to execute on its own once setup finishes — the
        same no-manual-steps behavior as Render/Preview, so the button never
        demands a hand-run script."""
        if self._vulkan_preflight_thread is not None:
            self.error_panel.append("A Vulkan backend preflight is already running.")
            return
        missing = _vulkan_prerequisite_missing()
        if missing:
            if self._preflight_queued_after_setup:
                self.error_panel.append(
                    "Vulkan backend test already queued; it will run when the setup finishes."
                )
                return
            # Start (or reuse) the automatic setup and queue the preflight
            # behind it. A failed setup is surfaced by
            # _handle_vulkan_setup_failed (which also clears the queue), so
            # the test can never be left dangling.
            self._start_vulkan_setup()
            self._preflight_queued_after_setup = True
            self.error_panel.append(
                "Vulkan backend test queued: the backend is being set up automatically; "
                "the test will run by itself once setup finishes."
            )
            QMessageBox.information(
                self,
                "Vulkan Backend Setup",
                "The Vulkan backend is being set up automatically (building audiocpp_cli "
                "and/or downloading the Chatterbox model). The backend test will run by "
                "itself once setup finishes.",
            )
            return
        self.error_panel.append("Testing Vulkan backend (binary, model, devices)...")
        # Parented to the window so a close while preflighting can wait() on it
        # instead of destroying a running QThread (which Qt aborts on).
        thread = VulkanPreflightThread(self, device_index=self._audio_cpp_device_value())
        thread.completed.connect(self._handle_vulkan_preflight_completed)
        thread.failed.connect(self._handle_vulkan_preflight_failed)
        thread.finished.connect(self._cleanup_vulkan_preflight_thread)
        # Assign before refreshing so the button locks while the preflight runs.
        self._vulkan_preflight_thread = thread
        self._refresh_vulkan_preflight_button()
        thread.start()

    def _handle_vulkan_preflight_completed(self, report: str) -> None:
        self.error_panel.append(report)
        QMessageBox.information(self, "Vulkan Backend Test", report)

    def _handle_vulkan_preflight_failed(self, message: str) -> None:
        self.error_panel.append(f"Vulkan backend test failed: {message}")
        QMessageBox.warning(self, "Vulkan Backend Test", message)

    def _cleanup_vulkan_preflight_thread(self) -> None:
        if self._vulkan_preflight_thread is not None:
            self._vulkan_preflight_thread.deleteLater()
            self._vulkan_preflight_thread = None
            self._refresh_vulkan_preflight_button()

    def download_vulkan_model(self) -> None:
        """Fetch the Chatterbox model in the background.

        Runs scripts/download_audio_cpp_model.sh off the UI thread (model
        downloads are large and slow). On success the reported
        ORACLE_AUDIOCPP_MODEL path is set for this session so the prerequisite
        warning clears and renders are ready; the exact export line is shown
        for users who want to persist it in their shell.
        """
        if self._model_download_thread is not None:
            self.error_panel.append("A Vulkan model download is already in progress.")
            return
        script = self.repo_root / "scripts" / "download_audio_cpp_model.sh"
        if not script.exists():
            # Defensive: the script ships with this repo, so its absence means a
            # broken checkout rather than a missing audio.cpp clone.
            message = "This checkout is missing scripts/download_audio_cpp_model.sh."
            self.error_panel.append(message)
            QMessageBox.warning(self, "Vulkan Model Download Unavailable", message)
            return
        self.download_vulkan_model_action.setEnabled(False)
        self.error_panel.append("Downloading the Vulkan Chatterbox model in the background...")
        # Parented to the window; on close the subprocess is cancelled and the
        # thread waited on so a running QThread is never destroyed.
        thread = ModelDownloadThread(script, self)
        thread.completed.connect(self._handle_vulkan_model_downloaded)
        thread.failed.connect(self._handle_vulkan_model_download_failed)
        thread.finished.connect(self._cleanup_model_download_thread)
        self._model_download_thread = thread
        thread.start()

    def _handle_vulkan_model_downloaded(self, model_path: str) -> None:
        """The model is installed; make it this session's Vulkan model and show
        the exact export line so the user can persist it in their shell."""
        os.environ["ORACLE_AUDIOCPP_MODEL"] = model_path
        # Persist the resolved path so a later launch can restore it (when
        # 'Remember GPU/CPU choice' is enabled) without re-downloading.
        self._persist_remembered_settings(model=model_path)
        self.error_panel.append(f"Vulkan model downloaded: ORACLE_AUDIOCPP_MODEL={model_path}")
        # The inline warning only belongs on the Vulkan backend; the download
        # can be started from the Settings menu while PyTorch is still active.
        if self.inference_backend_combo.currentData() == "vulkan":
            self._refresh_vulkan_prerequisite_warning()
        else:
            self.vulkan_prerequisite_warning.hide()
        QMessageBox.information(
            self,
            "Vulkan Model Downloaded",
            f"The Chatterbox model is installed and ORACLE_AUDIOCPP_MODEL is set for this session:\n\n"
            f'    export ORACLE_AUDIOCPP_MODEL="{model_path}"\n\n'
            f"The Vulkan backend is now ready to render. With Settings → Remember "
            f"GPU/CPU choice enabled (the default), this path is restored "
            f"automatically at the next launch — no re-download. Alternatively, "
            f"add that export line to your shell profile.",
        )

    def _handle_vulkan_model_download_failed(self, message: str) -> None:
        self.error_panel.append(f"Vulkan model download failed: {message}")
        QMessageBox.warning(self, "Vulkan Model Download Failed", message)

    def _cleanup_model_download_thread(self) -> None:
        if self._model_download_thread is not None:
            self._model_download_thread.deleteLater()
            self._model_download_thread = None
            self.download_vulkan_model_action.setEnabled(True)

    def closeEvent(self, event) -> None:
        """Close only after every GUI-owned QThread has stopped.

        Destroying a live QThread is a hard Qt abort, not a normal Python
        exception. This matters most for RenderWorker: the user can close the
        window while rendering, and the old close path only waited for Vulkan
        helpers. If any thread cannot stop within its bounded shutdown window,
        keep the window open rather than trading a close click for a crash.
        """
        def wait_for_thread(thread, label: str, timeout_ms: int) -> bool:
            wait = getattr(thread, "wait", None)
            if not callable(wait):
                # Test doubles and already-detached helpers may not expose the
                # QThread API; there is nothing useful to wait for in that case.
                return True
            try:
                finished = bool(wait(timeout_ms))
            except Exception as exc:
                self.error_panel.append(f"Could not stop {label}: {exc}")
                return False
            if not finished:
                self.error_panel.append(
                    f"{label} is still running. Finish or cancel it before closing The Oracle."
                )
                return False
            return True

        # Render/preview workers and startup prewarm are also QThreads. Waiting
        # here prevents Qt's "QThread: Destroyed while thread is still
        # running" abort when the user closes during one of these operations.
        for thread, label, timeout_ms in (
            (self.render_worker, "Render", 1000),
            (self.preview_worker, "Preview", 1000),
            (self._prewarm_thread, "Startup warmup", 1000),
        ):
            if thread is None:
                continue
            cancel = getattr(thread, "request_cancel", None)
            if callable(cancel):
                cancel()
            if not wait_for_thread(thread, label, timeout_ms):
                event.ignore()
                return

        probe = self._vulkan_probe_thread
        if probe is not None:
            if not wait_for_thread(probe, "Vulkan device probe", 2000):
                event.ignore()
                return
            self._vulkan_probe_thread = None
        preflight = self._vulkan_preflight_thread
        if preflight is not None:
            # Disconnect so a finishing preflight can't pop a modal report
            # dialog during app close; the subprocess probe is quick, so a
            # bounded wait is enough.
            try:
                preflight.completed.disconnect()
                preflight.failed.disconnect()
                preflight.finished.disconnect()
            except RuntimeError:
                pass
            if not wait_for_thread(preflight, "Vulkan preflight", 2000):
                event.ignore()
                return
            self._vulkan_preflight_thread = None
        download = self._model_download_thread
        if download is not None:
            # A model download can take minutes; cancel the subprocess and wait
            # so a running QThread is never destroyed. Signals are disconnected
            # first so the cancel's failure path can't pop a modal dialog
            # during app close.
            try:
                download.completed.disconnect()
                download.failed.disconnect()
                download.finished.disconnect()
            except RuntimeError:
                pass
            download.request_cancel()
            if not wait_for_thread(download, "Vulkan model download", 5000):
                event.ignore()
                return
            self._model_download_thread = None
        setup = self._vulkan_setup_thread
        if setup is not None:
            # A build/download can take minutes; cancel the subprocess group and
            # wait so a running QThread is never destroyed. Signals are
            # disconnected first so the cancel's failure path can't pop a modal
            # dialog during app close.
            try:
                setup.progress.disconnect()
                setup.completed.disconnect()
                setup.failed.disconnect()
                setup.finished.disconnect()
            except RuntimeError:
                pass
            setup.request_cancel()
            if not wait_for_thread(setup, "Vulkan backend setup", 5000):
                event.ignore()
                return
            self._vulkan_setup_thread = None
        super().closeEvent(event)

    def _audio_cpp_device_value(self) -> int | None:
        data = self.audio_cpp_device_combo.currentData()
        return data if isinstance(data, int) else None

    def _audio_cpp_threads_value(self) -> int | None:
        value = self.audio_cpp_threads_spin.value()
        return None if value < 1 else value

    def _audio_cpp_timeout_value(self) -> int | None:
        value = self.audio_cpp_timeout_spin.value()
        return None if value < 1 else value

    def _audio_cpp_max_batch_value(self) -> int | None:
        value = self.audio_cpp_max_batch_spin.value()
        return None if value < 1 else value

    def _set_audio_cpp_device_value(self, value: int | None) -> None:
        combo = self.audio_cpp_device_combo
        if value is None:
            # "Auto (audio.cpp default)" is always item 0.
            combo.setCurrentIndex(0)
            return
        value = max(0, int(value))
        index = combo.findData(value)
        if index < 0:
            combo.addItem(_device_row_text(value, "(not detected)"), value)
            index = combo.findData(value)
        combo.setCurrentIndex(index)

    def _set_audio_cpp_threads_value(self, value: int | None) -> None:
        self.audio_cpp_threads_spin.setValue(0 if value is None else max(1, int(value)))

    def _set_audio_cpp_timeout_value(self, value: int | None) -> None:
        self.audio_cpp_timeout_spin.setValue(0 if value is None else max(1, int(value)))

    def _set_audio_cpp_max_batch_value(self, value: int | None) -> None:
        self.audio_cpp_max_batch_spin.setValue(0 if value is None else max(1, int(value)))

    # --------------------
    # Cross-session backend memory
    # --------------------
    def _remember_backend_enabled(self) -> bool:
        """Whether the 'Remember GPU/CPU choice' option is currently checked."""
        if hasattr(self, "remember_backend_action"):
            return self.remember_backend_action.isChecked()
        return bool(self._app_settings.get("remember_backend", True))

    def _persist_remembered_settings(self, _value=None, *, binary: str | None = None, model: str | None = None) -> None:
        """Write the current backend selection + Vulkan knobs + audio.cpp paths
        to the app-level settings file, so the next launch can restore them.

        Connected to the backend combo and Vulkan knob changes (gated on
        ``_app_settings_ready``) and called explicitly with the resolved paths
        when the automatic setup or a model download finishes. ``binary`` /
        ``model`` (when given) win over the current environment, so a fresh
        setup result is recorded even if the env vars were not pre-set.
        """
        if not self._app_settings_ready:
            return
        enabled = self._remember_backend_enabled()
        if not enabled:
            # Option off = manual control: don't keep writing the settings file
            # on every backend/knob change. The disabled flag itself was already
            # persisted by the toggle handler, so the next launch starts on the
            # default (PyTorch).
            return
        self._app_settings.update({
            "remember_backend": enabled,
            "inference_backend": self.inference_backend_combo.currentData() or "pytorch",
            "audio_cpp_device": self._audio_cpp_device_value(),
            "audio_cpp_threads": self._audio_cpp_threads_value(),
            "audio_cpp_timeout": self._audio_cpp_timeout_value(),
            "audio_cpp_max_batch": self._audio_cpp_max_batch_value(),
            "audio_cpp_cli": binary
            or os.environ.get("ORACLE_AUDIOCPP_CLI")
            or self._app_settings.get("audio_cpp_cli", ""),
            "audio_cpp_model": model
            or os.environ.get("ORACLE_AUDIOCPP_MODEL")
            or self._app_settings.get("audio_cpp_model", ""),
        })
        try:
            save_app_settings(self._app_settings)
        except Exception as exc:
            self.error_panel.append(f"Could not persist GPU settings: {exc}")

    def _on_remember_backend_toggled(self, checked: bool) -> None:
        """React to the Settings menu toggle: record the new state immediately.

        Turning it on persists the current backend + paths right away (so the
        very next launch already restores them); turning it off persists just
        the disabled flag so future launches start on PyTorch (CPU).
        """
        self._app_settings["remember_backend"] = checked
        if checked:
            self._persist_remembered_settings()
        else:
            try:
                save_app_settings(self._app_settings)
            except Exception as exc:
                self.error_panel.append(f"Could not persist GPU settings: {exc}")
        self.error_panel.append(
            "Remember GPU/CPU choice enabled: the backend selection and audio.cpp "
            "paths are restored automatically on the next launch."
            if checked
            else "Remember GPU/CPU choice disabled: the app will start on PyTorch (CPU)."
        )

    def _apply_remembered_backend(self) -> None:
        """Restore the remembered inference backend and audio.cpp paths at launch.

        Runs once, after the window is shown. When 'Remember GPU/CPU choice' is
        enabled it re-applies the persisted ``ORACLE_AUDIOCPP_CLI`` /
        ``ORACLE_AUDIOCPP_MODEL`` paths to the environment (only if the files
        still exist — a stale path is surfaced, never silently applied) and
        selects the remembered backend + Vulkan knobs. Selecting Vulkan then
        runs the normal prerequisite check: if the persisted paths are valid
        there is nothing left to do (zero setup across sessions); if they are
        stale the automatic setup re-runs on its own.
        """
        if not self._app_settings.get("remember_backend", True):
            return
        cli_path = self._app_settings.get("audio_cpp_cli", "")
        model_path = self._app_settings.get("audio_cpp_model", "")
        if cli_path:
            resolved_cli = Path(cli_path).expanduser()
            if resolved_cli.exists():
                os.environ["ORACLE_AUDIOCPP_CLI"] = str(resolved_cli)
            else:
                self.error_panel.append(f"Remembered audiocpp_cli path no longer exists: {cli_path}")
        if model_path:
            resolved_model = Path(model_path).expanduser()
            if resolved_model.exists():
                os.environ["ORACLE_AUDIOCPP_MODEL"] = str(resolved_model)
            else:
                self.error_panel.append(f"Remembered Chatterbox model path no longer exists: {model_path}")
        backend = self._app_settings.get("inference_backend", "pytorch")
        if backend == "vulkan":
            index = self.inference_backend_combo.findData("vulkan")
            if index >= 0:
                self.inference_backend_combo.setCurrentIndex(index)
            self._set_audio_cpp_device_value(self._app_settings.get("audio_cpp_device"))
            self._set_audio_cpp_threads_value(self._app_settings.get("audio_cpp_threads"))
            self._set_audio_cpp_timeout_value(self._app_settings.get("audio_cpp_timeout"))
            self._set_audio_cpp_max_batch_value(self._app_settings.get("audio_cpp_max_batch"))
            self.error_panel.append("Restored the remembered Vulkan backend from the last session.")

    def _refresh_reference_pickers(self) -> None:
        defaults = default_voice_choices(self.repo_root)
        recents = [path for path in load_recent_reference_paths() if Path(path).exists()]
        for group in self._all_speaker_groups().values():
            group.set_reference_choices(defaults, recents, group.reference_path.text())

    # --------------------
    # Prewarm management
    # --------------------
    def _start_prewarm(self) -> None:
        with self._prewarm_lock:
            if self._prewarm_state in {"warming", "ready"}:
                return
            self._prewarm_state = "warming"
        # Keep the UI responsive: disable heavy actions while warmup runs, but do not block the event loop.
        self.analyze_button.setEnabled(False)
        self.render_button.setEnabled(False)
        self._prewarm_thread = PrewarmThread(device=_DEVICE_MODE)
        self._prewarm_thread.ready.connect(self._handle_prewarm_ready)
        self._prewarm_thread.failed.connect(self._handle_prewarm_failed)
        try:
            self._prewarm_thread.start()
        except Exception as exc:
            self._handle_prewarm_failed(str(exc), {"prewarm_failed": time()})

    def _handle_prewarm_ready(self, pipeline: OraclePipeline, engine: object, timing: dict[str, float]) -> None:
        with self._prewarm_lock:
            self._prewarm_state = "ready"
            self._prewarmed_pipeline = pipeline
            self._prewarmed_engine = engine
            self._prewarm_timing = timing
            thread = self._prewarm_thread
            self._prewarm_thread = None
        if thread is not None:
            thread.deleteLater()
        self._write_prewarm_timing(success=True)
        # Enable actions now that warmup finished
        self.analyze_button.setEnabled(True)
        self.render_button.setEnabled(True)

    def _handle_prewarm_failed(self, message: str, timing: dict[str, float]) -> None:
        with self._prewarm_lock:
            self._prewarm_state = "failed"
            self._prewarmed_pipeline = None
            self._prewarmed_engine = None
            self._prewarm_timing = timing | {"error": message}
            thread = self._prewarm_thread
            self._prewarm_thread = None
        if thread is not None:
            thread.deleteLater()
        self.error_panel.append(f"Background prewarm failed: {message}")
        self._write_prewarm_timing(success=False)
        # Allow the user to proceed manually even if warmup failed.
        self.analyze_button.setEnabled(True)
        self.render_button.setEnabled(True)

    def _write_prewarm_timing(self, success: bool) -> None:
        try:
            log_dir = self.paths.output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            timing = dict(self._prewarm_timing or {})
            if self._gui_shown_wall:
                timing["gui_shown"] = self._gui_shown_wall
            timing["prewarm_success"] = success
            (log_dir / "gui_prewarm_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _all_speaker_groups(self) -> dict[str, SpeakerGroup]:
        """Every speaker group: A, B, and any extra character voices (C..X)."""
        return {"A": self.speaker_a, "B": self.speaker_b, **self.extra_speaker_groups}

    def _sync_extra_speaker_groups(self, speakers: list[str]) -> None:
        """Create/refresh SpeakerGroup widgets for characters beyond A and B
        (up to 24 total) so an audiobook cast gets one voice panel each."""
        extras = sorted(speaker for speaker in speakers if speaker not in ("A", "B"))
        for key in extras:
            if key not in self.extra_speaker_groups:
                group = SpeakerGroup(key, self.paths.voice_dir)
                self.extra_speaker_groups[key] = group
                self.extra_speaker_layout.addWidget(group)
        stale = [key for key in self.extra_speaker_groups if key not in extras]
        for key in stale:
            group = self.extra_speaker_groups.pop(key)
            self.extra_speaker_layout.removeWidget(group)
            group.deleteLater()
        self.extra_speaker_scroll.setVisible(bool(extras))
        if extras:
            self.extra_speaker_scroll.setWindowTitle("Character Voices")
        self._refresh_reference_pickers()
        self._refresh_language_options()

    def _speaker_settings(self) -> dict[str, SpeakerSettings]:
        variant = self.variant_combo.currentText()
        return {
            key: SpeakerSettings(
                reference_path=group.reference_path.text(),
                voice_settings=VoiceSettings(
                    variant=variant,
                    language=group.language_combo.currentData() or "en",
                    cfg_weight=group.cfg_weight.value(),
                    exaggeration=group.exaggeration.value(),
                    temperature=group.temperature.value(),
                    emotion_intensity=group.emotion_intensity.value(),
                    naturalness=group.naturalness.value(),
                    pause_ms=group.pause_spin.value(),
                    crossfade_ms=self.crossfade_spin.value(),
                ),
            )
            for key, group in self._all_speaker_groups().items()
        }

    def _render_settings(self) -> RenderSettings:
        variant = self.variant_combo.currentText()
        mode_value = self.correction_mode_combo.currentData() or self.correction_mode_combo.currentText()
        inference_backend = self.inference_backend_combo.currentData() or "pytorch"
        # The device/threads/timeout knobs are Vulkan-only; emit None on the
        # PyTorch path so stale widget values never leak into render metadata
        # or saved profiles as if they were in effect.
        is_vulkan = inference_backend == "vulkan"
        return RenderSettings(
            correction_mode=mode_value,
            model_variant=variant,
            language=self.speaker_a.language_combo.currentData() or "en",
            export_stems=True,
            loudness_preset=self.loudness_combo.currentText(),
            pause_between_turns_ms=self.speaker_a.pause_spin.value(),
            crossfade_ms=self.crossfade_spin.value(),
            device_mode=_DEVICE_MODE,
            inference_backend=inference_backend,
            audio_cpp_device=self._audio_cpp_device_value() if is_vulkan else None,
            audio_cpp_threads=self._audio_cpp_threads_value() if is_vulkan else None,
            audio_cpp_timeout=self._audio_cpp_timeout_value() if is_vulkan else None,
            audio_cpp_max_batch=self._audio_cpp_max_batch_value() if is_vulkan else None,
            monologue=self.monologue_check.isChecked(),
            metadata={
                "output_filename": normalize_output_filename(self.output_name.text()),
                "export_srt": "1" if self.export_srt_check.isChecked() else "",
            },
        )

    def _pipeline(self) -> OraclePipeline:
        if self.pipeline is not None:
            return self.pipeline
        with self._prewarm_lock:
            state = self._prewarm_state
        if state == "ready" and self._prewarmed_pipeline is not None:
            self.pipeline = self._prewarmed_pipeline
            return self.pipeline
        # Keep GUI analysis on the deterministic local fallbacks. The optional
        # transformer/PyTorch and LanguageTool stacks can load native code or
        # spawn helper threads inside the Qt process; on Ubuntu 24.04 that
        # combination has been observed to segfault when Analyze is clicked.
        # The CLI still uses OraclePipeline's feature-rich defaults.
        self.pipeline = OraclePipeline(
            use_transformers=False,
            use_language_tool=False,
            use_punctuation_model=False,
        )
        return self.pipeline

    def _prewarmed_engine_ready(self):
        with self._prewarm_lock:
            state = self._prewarm_state
        if state == "ready":
            # Do not pass live engine across threads; return sentinel to signal no reuse.
            return None
        return None

    def _default_gui_settings_payload(self) -> dict:
        default_render = RenderSettings()
        default_voice = VoiceSettings(variant=default_render.model_variant)
        return {
            "version": 1,
            "name": "",
            "device_mode": default_render.device_mode,
            "project": {
                "model_variant": default_render.model_variant,
                "correction_mode": default_render.correction_mode,
                "loudness_preset": default_render.loudness_preset,
                "crossfade_ms": default_render.crossfade_ms,
                "inference_backend": default_render.inference_backend,
                "audio_cpp_device": default_render.audio_cpp_device,
                "audio_cpp_threads": default_render.audio_cpp_threads,
                "audio_cpp_timeout": default_render.audio_cpp_timeout,
                "audio_cpp_max_batch": default_render.audio_cpp_max_batch,
                "output_dir": str(self.paths.output_dir),
                "output_filename": "",
                "export_srt": False,
                "monologue": False,
            },
            "speakers": {
                speaker: {
                    "reference_path": "",
                    "voice_settings": default_voice.to_dict(),
                    "emotion_reference_paths": {},
                }
                for speaker in ("A", "B")
            },
        }

    def _apply_speaker_group(self, group: SpeakerGroup, settings: SpeakerSettings) -> None:
        voice = VoiceSettings.from_mapping(settings.voice_settings)
        group.reference_path.setText(settings.reference_path)
        language_index = group.language_combo.findData(voice.language)
        if language_index >= 0:
            group.language_combo.setCurrentIndex(language_index)
        group.cfg_weight.setValue(voice.cfg_weight)
        group.exaggeration.setValue(voice.exaggeration)
        group.temperature.setValue(voice.temperature)
        group.emotion_intensity.setValue(voice.emotion_intensity)
        group.naturalness.setValue(voice.naturalness)
        group.pause_spin.setValue(voice.pause_ms)
        self._refresh_reference_pickers()

    def _load_project_into_ui(self, saved_project) -> None:
        self.current_project_path = None
        self.plan = saved_project.plan
        self.input_path.setText(saved_project.input_path)
        self.outdir_path.setText(saved_project.output_path)
        self.output_name.setText(str(saved_project.render_settings.metadata.get("output_filename", "")))
        self.export_srt_check.setChecked(bool(saved_project.render_settings.metadata.get("export_srt")))
        self.monologue_check.setChecked(saved_project.render_settings.monologue)
        self.variant_combo.setCurrentText(saved_project.render_settings.model_variant)
        self._refresh_language_options()
        self._set_correction_mode(saved_project.render_settings.correction_mode)
        self.loudness_combo.setCurrentText(saved_project.render_settings.loudness_preset)
        self.crossfade_spin.setValue(saved_project.render_settings.crossfade_ms)
        backend_index = self.inference_backend_combo.findData(saved_project.render_settings.inference_backend)
        if backend_index >= 0:
            self.inference_backend_combo.setCurrentIndex(backend_index)
        self._set_audio_cpp_device_value(saved_project.render_settings.audio_cpp_device)
        self._set_audio_cpp_threads_value(saved_project.render_settings.audio_cpp_threads)
        self._set_audio_cpp_timeout_value(saved_project.render_settings.audio_cpp_timeout)
        self._set_audio_cpp_max_batch_value(saved_project.render_settings.audio_cpp_max_batch)
        self._refresh_inference_backend_options()
        self._sync_extra_speaker_groups(list(saved_project.speaker_settings))
        for speaker, settings in saved_project.speaker_settings.items():
            self._apply_speaker_group(self._all_speaker_groups()[speaker], settings)
        self._populate_table(self.plan)

    def _current_saved_project(self):
        if not self.plan:
            self.prepare_project()
        if not self.plan:
            raise ValueError("No project is available to save.")
        self._sync_plan_from_table()
        return build_saved_project(self.plan, self._render_settings(), self._speaker_settings())

    def _current_gui_settings_payload(self) -> dict:
        inference_backend = self.inference_backend_combo.currentData() or "pytorch"
        # Only persist the Vulkan knobs when the Vulkan backend is selected;
        # a disabled widget left over from an earlier selection must not be
        # saved alongside inference_backend: pytorch.
        is_vulkan = inference_backend == "vulkan"
        return {
            "version": 1,
            "name": "",
            "device_mode": _DEVICE_MODE,
            "project": {
                "model_variant": self.variant_combo.currentText(),
                "correction_mode": normalize_correction_mode(self.correction_mode_combo.currentData() or self.correction_mode_combo.currentText()),
                "loudness_preset": self.loudness_combo.currentText(),
                "crossfade_ms": self.crossfade_spin.value(),
                "inference_backend": inference_backend,
                "audio_cpp_device": self._audio_cpp_device_value() if is_vulkan else None,
                "audio_cpp_threads": self._audio_cpp_threads_value() if is_vulkan else None,
                "audio_cpp_timeout": self._audio_cpp_timeout_value() if is_vulkan else None,
                "audio_cpp_max_batch": self._audio_cpp_max_batch_value() if is_vulkan else None,
                "output_dir": self.outdir_path.text() or str(self.paths.output_dir),
                "output_filename": normalize_output_filename(self.output_name.text()),
                "export_srt": self.export_srt_check.isChecked(),
                "monologue": self.monologue_check.isChecked(),
                "delete_confirm_enabled": self.delete_confirm_enabled,
            },
            "speakers": {
                speaker: {
                    "reference_path": settings.reference_path,
                    "voice_settings": VoiceSettings.from_mapping(settings.voice_settings).to_dict(),
                    "emotion_reference_paths": dict(settings.emotion_reference_paths),
                }
                for speaker, settings in self._speaker_settings().items()
            },
        }

    def _apply_gui_settings_payload(self, payload: dict) -> None:
        defaults = self._default_gui_settings_payload()
        project = {**defaults["project"], **payload["project"]}
        self.variant_combo.setCurrentText(project.get("model_variant", "standard"))
        self._refresh_language_options()
        self._set_correction_mode(project.get("correction_mode", "moderate"))
        self.loudness_combo.setCurrentText(project.get("loudness_preset", RenderSettings().loudness_preset))
        self.crossfade_spin.setValue(int(project.get("crossfade_ms", 20)))
        self.outdir_path.setText(str(project.get("output_dir", self.paths.output_dir)))
        self.output_name.setText(normalize_output_filename(str(project.get("output_filename", ""))))
        self.export_srt_check.setChecked(bool(project.get("export_srt", False)))
        self.monologue_check.setChecked(bool(project.get("monologue", False)))
        self.delete_confirm_enabled = bool(project.get("delete_confirm_enabled", True))
        backend_index = self.inference_backend_combo.findData(project.get("inference_backend", "pytorch"))
        if backend_index >= 0:
            self.inference_backend_combo.setCurrentIndex(backend_index)
        self._set_audio_cpp_device_value(project.get("audio_cpp_device"))
        self._set_audio_cpp_threads_value(project.get("audio_cpp_threads"))
        self._set_audio_cpp_timeout_value(project.get("audio_cpp_timeout"))
        self._set_audio_cpp_max_batch_value(project.get("audio_cpp_max_batch"))
        # A hand-edited profile could pair turbo with vulkan; the turbo guard
        # falls back to pytorch instead of leaving the invalid combination
        # selected (RenderSettings would otherwise pass it through to a
        # confusing engine error at render time).
        self._refresh_inference_backend_options()
        for speaker, config in payload["speakers"].items():
            default_config = defaults["speakers"].get(speaker, defaults["speakers"]["A"])
            merged = {**default_config, **config}
            group = self._all_speaker_groups().get(speaker)
            if group is None:
                self._sync_extra_speaker_groups(list(payload["speakers"]))
                group = self._all_speaker_groups()[speaker]
            voice = VoiceSettings.from_mapping(merged.get("voice_settings"))
            group.reference_path.setText(merged.get("reference_path", ""))
            language_index = group.language_combo.findData(voice.language)
            if language_index >= 0:
                group.language_combo.setCurrentIndex(language_index)
            group.cfg_weight.setValue(voice.cfg_weight)
            group.exaggeration.setValue(voice.exaggeration)
            group.temperature.setValue(voice.temperature)
            group.emotion_intensity.setValue(voice.emotion_intensity)
            group.naturalness.setValue(voice.naturalness)
            group.pause_spin.setValue(voice.pause_ms)
        self._refresh_reference_pickers()

    def reset_settings_to_defaults(self) -> None:
        self._apply_gui_settings_payload(self._default_gui_settings_payload())
        self.error_panel.append("Settings reset to defaults.")

    def save_settings_profile(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Settings",
            str(self.paths.profile_dir / "oracle_profile.json"),
            "Settings Files (*.json)",
        )
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        payload = self._current_gui_settings_payload()
        try:
            save_gui_settings(destination, payload)
            self.error_panel.append(f"Saved settings: {destination}")
        except Exception as exc:
            self.error_panel.append(f"Save settings failed: {exc}")
            QMessageBox.critical(self, "Save Settings Failed", str(exc))

    def load_settings_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Settings", str(self.paths.profile_dir), "Settings Files (*.json)")
        if not path:
            return
        try:
            self._apply_gui_settings_payload(load_gui_settings(path))
            for speaker in self._speaker_settings().values():
                if speaker.reference_path:
                    remember_recent_reference_path(speaker.reference_path)
            self._refresh_reference_pickers()
            self.error_panel.append(f"Loaded settings: {path}")
        except Exception as exc:
            self.error_panel.append(f"Load settings failed: {exc}")
            QMessageBox.critical(self, "Load Settings Failed", str(exc))

    def save_template_profile(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Template", "Template name")
        if not ok or not name.strip():
            return
        payload = self._current_gui_settings_payload()
        payload["name"] = name.strip()
        try:
            destination = save_template(name.strip(), payload)
            self.error_panel.append(f"Saved template: {destination}")
        except Exception as exc:
            self.error_panel.append(f"Save template failed: {exc}")
            QMessageBox.critical(self, "Save Template Failed", str(exc))

    def _rebuild_templates_menu(self) -> None:
        self.templates_menu.clear()
        names = list_templates()
        if not names:
            empty = QAction("No Templates Saved", self)
            empty.setEnabled(False)
            self.templates_menu.addAction(empty)
            return
        for name in names:
            action = QAction(name, self)
            action.triggered.connect(lambda _checked=False, current=name: self._load_template_by_name(current))
            self.templates_menu.addAction(action)

    def _load_template_by_name(self, name: str) -> None:
        try:
            self._apply_gui_settings_payload(load_template(name))
            for speaker in self._speaker_settings().values():
                if speaker.reference_path:
                    remember_recent_reference_path(speaker.reference_path)
            self._refresh_reference_pickers()
            self.error_panel.append(f"Loaded template: {name}")
        except GUISettingsError as exc:
            self.error_panel.append(f"Load template failed: {exc}")
            QMessageBox.critical(self, "Load Template Failed", str(exc))

    def new_project(self) -> None:
        """Clear the current document (input path, analysis table, plan) while
        intentionally preserving the voice configuration (speaker reference
        clips, voice settings, render settings, output folder).  This mirrors
        the expected workflow: load a new script into an already-configured
        session without having to re-enter reference paths every time."""
        self.current_project_path = None
        self.plan = None
        self.input_path.clear()
        self.error_panel.clear()
        self.table.setRowCount(0)
        self._refresh_reference_pickers()

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Project Files (*.json)")
        if not path:
            return
        try:
            saved_project = load_project_manifest(path)
            self._load_project_into_ui(saved_project)
            self.current_project_path = Path(path)
            for speaker in saved_project.speaker_settings.values():
                if speaker.reference_path:
                    remember_recent_reference_path(speaker.reference_path)
            self._refresh_reference_pickers()
            self.error_panel.append(f"Loaded project: {path}")
        except Exception as exc:
            self.error_panel.append(f"Open failed: {exc}")
            QMessageBox.critical(self, "Open Project Failed", str(exc))

    def save_project(self) -> None:
        if self.current_project_path is None:
            self.save_project_as()
            return
        self._save_project_to_path(self.current_project_path)

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", "Project Files (*.json)")
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        self._save_project_to_path(destination)

    def _save_project_to_path(self, path: Path) -> None:
        try:
            saved_project = self._current_saved_project()
            save_project_manifest(path, saved_project)
            self.current_project_path = path
            self.error_panel.append(f"Saved project: {path}")
        except Exception as exc:
            self.error_panel.append(f"Save failed: {exc}")
            QMessageBox.critical(self, "Save Project Failed", str(exc))

    def prepare_project(self) -> None:
        with self._prewarm_lock:
            if self._prewarm_state == "warming":
                self.error_panel.append("Background warmup still running; please wait a moment before analyzing.")
                return
        analyze_click_wall = time()
        self._log_action_timing("analyze_click", analyze_click_wall)
        try:
            self.plan = self._pipeline().prepare_plan(
                self.input_path.text(),
                self.outdir_path.text(),
                self._speaker_settings(),
                self._render_settings(),
            )
            plan_ready_wall = time()
            self._log_action_timing("plan_ready", plan_ready_wall, {"elapsed": plan_ready_wall - analyze_click_wall})
            for speaker in self._speaker_settings().values():
                if speaker.reference_path:
                    remember_recent_reference_path(speaker.reference_path)
            self._sync_extra_speaker_groups([item.speaker for item in self.plan.utterances])
            self._refresh_reference_pickers()
            self._populate_table(self.plan)
            self.error_panel.append("Analysis complete.")
        except Exception as exc:
            self.error_panel.append(str(exc))
            QMessageBox.critical(self, "Analysis Failed", str(exc))

    def _populate_table(self, plan: RenderPlan) -> None:
        self.table.setRowCount(len(plan.utterances))
        for row, utterance in enumerate(plan.utterances):
            self.table.setItem(row, 0, QTableWidgetItem(str(utterance.index)))
            speaker_combo = QComboBox()
            detected = sorted({item.speaker for item in plan.utterances})
            speaker_combo.addItems(detected or ["A", "B"])
            speaker_combo.setCurrentText(utterance.speaker)
            self.table.setCellWidget(row, 1, speaker_combo)
            self.table.setItem(row, 2, QTableWidgetItem(utterance.original_text))
            repaired = QTableWidgetItem(utterance.repaired_text)
            repaired.setFlags(repaired.flags() | Qt.ItemIsEditable)
            self.table.setItem(row, 3, repaired)
            emotion = self._create_emotion_combo(utterance.emotion)
            self.table.setCellWidget(row, 4, emotion)
            duration = "" if utterance.duration_seconds is None else f"{utterance.duration_seconds:.2f}s"
            self.table.setItem(row, 5, QTableWidgetItem(duration))
            # Show status in a dedicated column
            status = QTableWidgetItem(utterance.status)
            status.setFlags(status.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 6, status)
            preview = QPushButton("Preview")
            preview.clicked.connect(lambda _checked=False, current=row: self.preview_utterance(current))
            self.table.setCellWidget(row, 7, preview)
            control = self._create_row_action(row)
            self.table.setCellWidget(row, 8, control)

    def _create_row_action(self, row: int) -> QComboBox:
        control = QComboBox()
        control.addItems(["+/-", "Extra", "Remove"])
        control.setMaximumWidth(80)
        control.currentIndexChanged.connect(lambda idx, r=row, c=control: self._handle_row_action(idx, r, c))
        return control

    def _create_emotion_combo(self, value: str) -> QComboBox:
        combo = QComboBox()
        for emotion in SUPPORTED_EMOTIONS:
            combo.addItem(emotion, emotion)
        if value and value not in SUPPORTED_EMOTIONS:
            combo.addItem(value, value)
        target = value if value else "neutral"
        idx = combo.findData(target)
        if idx < 0:
            idx = combo.findData("neutral")
        combo.setCurrentIndex(max(0, idx))
        return combo

    def _handle_row_action(self, idx: int, row: int, control: QComboBox) -> None:
        if idx == 0 or not self.plan:
            return
        if idx == 1:
            self.plan.utterances.insert(row + 1, self._blank_utterance())
        elif idx == 2 and 0 <= row < len(self.plan.utterances):
            utterance = self.plan.utterances[row]
            if self._needs_delete_confirmation(utterance):
                if not self._confirm_delete():
                    control.blockSignals(True)
                    control.setCurrentIndex(0)
                    control.blockSignals(False)
                    return
            self.plan.utterances.pop(row)
        control.blockSignals(True)
        control.setCurrentIndex(0)
        control.blockSignals(False)
        self._reindex_utterances()
        self._populate_table(self.plan)

    def _blank_utterance(self) -> Utterance:
        return Utterance(
            index=0,
            original_text="",
            repaired_text="",
            speaker="A",
            emotion="neutral",
            duration_seconds=None,
        )

    def _reindex_utterances(self) -> None:
        if not self.plan:
            return
        for idx, utterance in enumerate(self.plan.utterances):
            utterance.index = idx

    def _needs_delete_confirmation(self, utterance: Utterance) -> bool:
        return self.delete_confirm_enabled and any(
            getattr(utterance, attr) for attr in ("original_text", "repaired_text", "emotion")
        )

    def _confirm_delete(self) -> bool:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Confirm Delete")
        dialog.setText("This row contains text. Look ready to delete?")
        checkbox = QCheckBox("Click here to hide this window into the program settings menu above.", dialog)
        dialog.setCheckBox(checkbox)
        dialog.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        result = dialog.exec()
        if checkbox.isChecked():
            self.delete_confirm_enabled = False
        return result == QMessageBox.Ok

    def _enable_delete_confirmation(self) -> None:
        self.delete_confirm_enabled = True
        self.error_panel.append("Delete confirmations re-enabled.")

    def _sync_plan_from_table(self) -> None:
        if not self.plan:
            return
        for row, utterance in enumerate(self.plan.utterances):
            speaker_widget = self.table.cellWidget(row, 1)
            if isinstance(speaker_widget, QComboBox):
                selected_speaker = speaker_widget.currentText()
                utterance.manual_speaker_override = utterance.manual_speaker_override or selected_speaker != utterance.speaker
                utterance.speaker = selected_speaker
            repaired_item = self.table.item(row, 3)
            if repaired_item:
                repaired_text = repaired_item.text().strip()
                utterance.manual_text_override = utterance.manual_text_override or repaired_text != utterance.repaired_text
                utterance.repaired_text = repaired_text
            emotion_widget = self.table.cellWidget(row, 4)
            if isinstance(emotion_widget, QComboBox):
                emotion_text = emotion_widget.currentData() or emotion_widget.currentText()
                utterance.manual_emotion_override = utterance.manual_emotion_override or emotion_text != utterance.emotion
                utterance.emotion = emotion_text
        speaker_settings = self._speaker_settings()
        self._sync_extra_speaker_groups(list(speaker_settings))
        merged_profiles = dict(self.plan.voice_profiles)
        for speaker, config in speaker_settings.items():
            if speaker in merged_profiles:
                merged_profiles[speaker] = replace(
                    merged_profiles[speaker], engine_params=config.voice_settings
                )
        self.plan.voice_profiles = merged_profiles
        self.plan.source_path = self.input_path.text()
        self.plan.output_dir = self.outdir_path.text()
        self.plan.metadata["model_variant"] = self.variant_combo.currentText()
        speaker_languages = {
            speaker: VoiceSettings.from_mapping(config.voice_settings).language
            for speaker, config in speaker_settings.items()
        }
        self.plan.metadata["language"] = speaker_languages["A"] if len(set(speaker_languages.values())) == 1 else "mixed"
        self.plan.update_hashes()

    def preview_utterance(self, row: int) -> None:
        if not self.plan:
            return
        if self.render_worker is not None:
            self.error_panel.append("Preview is unavailable while a render is in progress.")
            return
        if self.preview_worker is not None:
            self.error_panel.append("Preview is already in progress.")
            return
        try:
            self._sync_plan_from_table()
            utterance = self.plan.utterances[row]
        except Exception as exc:
            self.error_panel.append(f"Preview failed: {exc}")
            QMessageBox.critical(self, "Preview Failed", str(exc))
            return

        self.preview_dialog = RenderProgressDialog(self, title="Generating Preview")
        self.preview_dialog.show()
        self._set_preview_busy(True)
        inference_backend = self.inference_backend_combo.currentData() or "pytorch"
        if inference_backend == "vulkan" and _vulkan_prerequisite_missing():
            # Auto-setup: the preview waits for the CPU→GPU switch to finish
            # instead of failing deep in the worker. The row is remembered so
            # the completion handler can re-fire this exact preview by itself
            # (no second click); explicit clicks always retry.
            self._start_vulkan_setup()
            self._preview_queued_after_setup = True
            self._preview_row_queued_after_setup = row
            self.preview_dialog.close()
            self.preview_dialog = None
            self._set_preview_busy(False)
            self.error_panel.append(
                "Preview queued: the Vulkan backend is being set up automatically; "
                "the preview will start by itself once setup completes."
            )
            QMessageBox.information(
                self,
                "Vulkan Backend Setup",
                "The Vulkan backend is being set up automatically (building audiocpp_cli "
                "and/or downloading the Chatterbox model). The preview will start by "
                "itself once the setup finishes.",
            )
            return
        # Vulkan-only knobs: only forwarded (like _render_settings) when the
        # Vulkan backend is actually selected; render_preview ignores them on
        # the PyTorch path regardless.
        is_vulkan = inference_backend == "vulkan"
        self.preview_worker = PreviewWorker(
            utterance,
            self.plan.voice_profiles[utterance.speaker],
            self.variant_combo.currentText(),
            _DEVICE_MODE,
            # Preview must use the same GUI-safe pipeline as Analyze/render.
            # Omitting this causes PreviewWorker to construct a default
            # feature-rich OraclePipeline in its QThread, reintroducing the
            # native PyTorch/transformers import crash we avoid in the GUI.
            pipeline=self._pipeline(),
            inference_backend=inference_backend,
            audio_cpp_device=self._audio_cpp_device_value() if is_vulkan else None,
            audio_cpp_threads=self._audio_cpp_threads_value() if is_vulkan else None,
            audio_cpp_timeout=self._audio_cpp_timeout_value() if is_vulkan else None,
            audio_cpp_max_batch=self._audio_cpp_max_batch_value() if is_vulkan else None,
            # Preview synthesizes the model in an isolated child process for
            # the same reason render does: loading Chatterbox/PyTorch inside
            # the Qt Multimedia process segfaults on Ubuntu (SIGSEGV / 245).
            run_in_subprocess=True,
            python_executable=sys.executable,
            repo_root=self.repo_root,
        )
        self.preview_worker.progress.connect(self._update_preview_progress)
        self.preview_worker.completed.connect(lambda path: self._finish_preview(row, path))
        self.preview_worker.failed.connect(self._fail_preview)
        self.preview_worker.finished.connect(self._cleanup_preview_worker)
        self.preview_worker.start()

    def render_project(self) -> None:
        with self._prewarm_lock:
            if self._prewarm_state == "warming":
                self.error_panel.append("Background warmup still running; render will be available shortly.")
                QMessageBox.information(self, "Warmup In Progress", "Background warmup is still running. Please try Render again in a moment.")
                return
        if self.preview_worker is not None:
            self.error_panel.append("Wait for the active preview to finish before rendering.")
            return
        if not self.plan:
            message = "Analyze the project before rendering so render work stays off the UI thread."
            self.error_panel.append(message)
            QMessageBox.information(self, "Analyze First", message)
            return
        if self.render_worker is not None:
            self.error_panel.append("Render is already in progress.")
            return
        try:
            self._sync_plan_from_table()
            output_filename = resolve_output_filename(
                self.input_path.text(),
                self.outdir_path.text(),
                self.paths.output_dir,
                self.output_name.text(),
            )
            if not output_filename:
                raise ValueError("Choose an output filename before rendering outside the default Output folder.")
            self.plan.output_dir = self.outdir_path.text() or str(self.paths.output_dir)
        except Exception as exc:
            self.error_panel.append(f"Render failed: {exc}")
            QMessageBox.critical(self, "Render Failed", str(exc))
            return

        render_settings = self._render_settings()
        if render_settings.inference_backend == "vulkan":
            missing = _vulkan_prerequisite_missing()
            if missing:
                # Explicit user action: always (re)start the auto-setup — the
                # guard on _vulkan_setup_thread prevents double-starting, and
                # _start_vulkan_setup resets the failure state so a stale
                # setup can't leave the render queued forever.
                self._start_vulkan_setup()
                self._render_queued_after_setup = True
                message = (
                    "The Vulkan backend is being set up automatically (building "
                    "audiocpp_cli and/or downloading the Chatterbox model). "
                    "The render will start by itself once setup completes."
                )
                self.error_panel.append(f"Render queued: {message}")
                QMessageBox.information(self, "Vulkan Backend Setup", message)
                return

        render_click_wall = time()
        self._log_action_timing("render_click", render_click_wall)
        self.progress_dialog = RenderProgressDialog(self, title="Rendering")
        self.progress_dialog.show()
        self._set_render_busy(True)
        render_settings.metadata["output_filename"] = output_filename
        self.render_worker = RenderWorker(
            self.plan,
            render_settings,
            # Native Chatterbox/Perth synthesis must never initialize in the
            # Qt Multimedia process. The child starts with no QApplication and
            # therefore cannot reproduce the QMediaPlayer + QThread segfault.
            run_in_subprocess=True,
            python_executable=sys.executable,
            repo_root=self.repo_root,
            render_click_wall=render_click_wall,
        )
        self.render_worker.progress.connect(self._update_render_progress)
        self.render_worker.completed.connect(self._finish_render)
        self.render_worker.failed.connect(self._fail_render)
        self.render_worker.finished.connect(self._cleanup_render_worker)
        self.render_worker.start()

    def _update_render_progress(self, progress: RenderProgress) -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.update_progress(progress)

    def _finish_render(self, plan_payload: dict, output_path: str) -> None:
        self.plan = RenderPlan.from_dict(plan_payload)
        self._populate_table(self.plan)
        self.error_panel.append(f"Render complete: {output_path}")
        srt_path = self.plan.metadata.get("srt_path")
        if srt_path:
            self.error_panel.append(f"Subtitles written: {srt_path}")
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None

    def _fail_render(self, plan_payload: object, message: str | None = None) -> None:
        # RenderWorker includes its defensive plan copy in the failure signal.
        # This keeps partial row statuses/durations even if Qt delivers the
        # finished/cleanup slot before this queued failure handler.
        if message is None:
            # Backward-compatible path for direct callers/tests that only have
            # an error string; normal worker failures always use the payload.
            message = str(plan_payload)
            if self.render_worker is not None:
                self.plan = RenderPlan.from_dict(self.render_worker.plan.to_dict())
        elif isinstance(plan_payload, dict):
            self.plan = RenderPlan.from_dict(plan_payload)
        self.error_panel.append(f"Render failed: {message}")
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Refresh the table to show truthful row state after failure:
        # completed rows show their status/duration, failed rows show failed,
        # and rows never reached remain pending.
        if self.plan is not None:
            self._populate_table(self.plan)

        # Show failure summary with row-level information
        failed_rows = self.plan.metadata.get("failed_rows", "") if self.plan is not None else ""
        if failed_rows:
            QMessageBox.critical(self, "Render Failed", 
                               f"Render failed. Failed rows: {failed_rows}\n\nSee error panel for details.")
        else:
            QMessageBox.critical(self, "Render Failed", message)

    def _cleanup_render_worker(self) -> None:
        self._set_render_busy(False)
        if self.render_worker is not None:
            self.render_worker.deleteLater()
            self.render_worker = None

    def _update_preview_progress(self, progress: RenderProgress) -> None:
        if self.preview_dialog is not None:
            self.preview_dialog.update_progress(progress)

    def _finish_preview(self, row: int, preview_path: str) -> None:
        # Do NOT persist preview state to row-level render fields.
        # Preview is a probe operation, not a render. Row-level duration_seconds
        # and status represent full-render truth, not preview-local truth.
        # For chunked rows, preview duration would be first-chunk only (misleading),
        # and preview success is not the same as render success.
        # The GUI Duration/Status columns remain unchanged (showing pending or
        # previous render state) until a full render completes.

        self.player.setSource(QUrl.fromLocalFile(preview_path))
        self.player.play()
        self.error_panel.append(f"Preview ready: {preview_path}")
        if self.preview_dialog is not None:
            self.preview_dialog.close()
            self.preview_dialog = None

    def _fail_preview(self, message: str) -> None:
        self.error_panel.append(f"Preview failed: {message}")
        if self.preview_dialog is not None:
            self.preview_dialog.close()
            self.preview_dialog = None
        QMessageBox.critical(self, "Preview Failed", message)

    def _cleanup_preview_worker(self) -> None:
        self._set_preview_busy(False)
        if self.preview_worker is not None:
            self.preview_worker.deleteLater()
            self.preview_worker = None

    def _set_busy(self, busy: bool) -> None:
        """Disable or re-enable all interactive widgets during a render or preview.

        Centralised here so adding a new interactive widget only requires one
        update rather than mirroring the change in both a render and a preview
        variant.
        """
        self.render_button.setEnabled(not busy)
        self.analyze_button.setEnabled(not busy)
        self.table.setEnabled(not busy)

    # Convenience aliases so call-sites read naturally.
    def _set_render_busy(self, busy: bool) -> None:
        self._set_busy(busy)

    def _set_preview_busy(self, busy: bool) -> None:
        self._set_busy(busy)


def launch_gui() -> None:
    launch_t0 = perf_counter()
    app = QApplication.instance() or QApplication([])
    launch_marks: list[tuple[str, float]] = [("qt_app_created", perf_counter() - launch_t0)]
    window = MainWindow()
    launch_marks.append(("mainwindow_built", perf_counter() - launch_t0))
    window.show()
    try:
        log_dir = Path(__file__).resolve().parents[2] / "Output" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {"events": launch_marks}
        (log_dir / "gui_launch_timing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    app.exec()
class PrewarmThread(QThread):
    ready = Signal(object, object, dict)
    failed = Signal(str, dict)

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device

    def run(self) -> None:
        timeline: dict[str, float] = {}
        try:
            start_wall = time()
            timeline["prewarm_start"] = start_wall
            # Startup prewarm must never risk taking down the GUI process.
            # Keep automatic warmup to a lightweight timing/status pass and
            # leave heavyweight backend construction to explicit user actions.
            ready_wall = time()
            timeline["pipeline_ready"] = ready_wall
            timeline["repair_ready"] = ready_wall
            timeline["emotion_ready"] = ready_wall
            timeline["engine_ready"] = ready_wall
            timeline["prewarm_complete"] = ready_wall
            self.ready.emit(None, None, timeline)
        except Exception as exc:  # pragma: no cover - GUI-only path
            timeline["prewarm_failed"] = time()
            self.failed.emit(str(exc), timeline)
