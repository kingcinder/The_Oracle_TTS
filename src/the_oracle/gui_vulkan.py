"""Vulkan/audio.cpp GUI support surface: device labels, model-path parsing,
the policy functions, and the background thread cluster.

Extracted from app_gui.py in three steps. The 2026-09-20 slice took only the
module-level helpers with NO wholesale patch surface, after the per-cluster
damage assessment (and its execution attempt) confirmed the two policy
functions (``_vulkan_prerequisite_missing``, ``_vulkan_preflight_report``)
could not move as-is: tests patch ``app_gui.find_audiocpp_binary`` by name,
and a function body that resolves that name from its own module globals
breaks under that patching — measured, not assumed (the first attempt failed
exactly those tests).

The 2026-09-27 slice took the four background threads
(``VulkanDeviceProbeThread``, ``VulkanPreflightThread``,
``ModelDownloadThread``, ``VulkanSetupThread``):

* their bodies resolve only move-safe names — ``subprocess``/``os``/``signal``
  are module objects whose attributes tests patch globally (identical from
  any importer), and ``AudioCppVulkanEngine`` patches are class-targets;
* ``VulkanPreflightThread``'s patch-coupled dependency — the report builder —
  arrives as constructor injection (MainWindow passes the bare
  ``_vulkan_preflight_report`` name, which resolves from app_gui's globals at
  call time). This module never imports app_gui (pinned in
  ``tests/test_gui_vulkan_owner.py``);
* app_gui re-imports all four classes (identical objects), so the wholesale
  class patches at app_gui level keep intercepting MainWindow's
  constructions.

The 2026-09-28 slice moved the two policy BODIES here behind the same
injection pattern, generalizing it: both take ``find_binary`` as a required
parameter, and app_gui's delegates pass the bare ``find_audiocpp_binary``
name — resolved from app_gui's globals at call time — so every app_gui-level
patch site stays live while the logic lives with the rest of the Vulkan
surface. The seam is enforced by ``tests/test_gui_vulkan_owner.py``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from the_oracle.gui_utils import kill_process_tree
from the_oracle.tts_engines.vulkan_backend import (
    AudioCppUnavailableError,
    AudioCppVulkanEngine,
)
from the_oracle.vulkan_setup import (
    parse_model_export,
    run_vulkan_setup,
    vulkan_setup_needed,
)


def _parse_oracle_model_path(output: str) -> str:
    """Extract the installed model path from the download script's output.

    Delegates to :func:`the_oracle.vulkan_setup.parse_model_export` (the
    single source of truth shared with the auto-setup orchestrator).
    """
    return parse_model_export(output)


def _device_row_text(index: int, name: str) -> str:
    """Human label for one Vulkan device, shared by the dropdown items and
    the picker's summary label so the two can never drift apart."""
    return f"Device {index}: {name}"


def _vulkan_prerequisite_missing(find_binary: Callable[[], Path | None]) -> list[str]:
    """Return human-readable reasons the Vulkan backend is not ready, else [].

    The policy body lives here (moved from app_gui in the 2026-09-28 slice);
    the binary probe arrives INJECTED: callers pass their own
    ``find_audiocpp_binary`` reference, so the CALLER's module globals decide
    what the probe sees. app_gui's delegates pass the bare
    ``find_audiocpp_binary`` name — it resolves from app_gui's globals at
    call time, keeping the ~12 app_gui-level monkeypatch sites live exactly
    as they were before the move (the 2026-09-20 failed extraction is the
    recorded harm history; this seam is its deliberate resolution).
    Delegates to :func:`the_oracle.vulkan_setup.vulkan_setup_needed`, the
    single source of truth shared with the CLI.
    """
    return vulkan_setup_needed(find_binary=find_binary)


def _vulkan_preflight_report(
    device_index: int | None, *, find_binary: Callable[[], Path | None]
) -> str:
    """Run the audio.cpp preflight and return a human-readable report.

    Checks the binary and model (same checks as
    :func:`_vulkan_prerequisite_missing`), then lists the Vulkan devices
    audio.cpp sees and states which GPU a render would use (the selected
    ``device_index``, or audio.cpp's default when None). Raises
    :class:`AudioCppUnavailableError` when the setup is not ready.

    ``find_binary`` is injected (see the prerequisite function above): the
    caller's patch surface decides what the probe sees, while this body's
    other collaborators (the engine class, the error type) are this module's
    own imports, which class-target patches intercept identically from any
    importer.
    """
    missing = _vulkan_prerequisite_missing(find_binary)
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

    The report builder is INJECTED (``preflight_report``), never resolved from
    this module: the builder must read app_gui's globals at call time (tests
    patch ``app_gui._vulkan_preflight_report`` /
    ``app_gui.find_audiocpp_binary`` by name) and this module must not import
    app_gui. MainWindow passes the bare ``_vulkan_preflight_report`` name, so
    those patches reach the seam exactly as they reached the old in-module
    call.
    """

    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        device_index: int | None = None,
        preflight_report: Callable[[int | None], str],
    ) -> None:
        super().__init__(parent)
        self._device_index = device_index
        self._preflight_report = preflight_report

    def run(self) -> None:
        try:
            report = self._preflight_report(self._device_index)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(report)


_MODEL_DOWNLOAD_TIMEOUT = 1800.0  # large GGUF downloads can take a while


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
                # Kill the whole process group, not just the bash wrapper:
                # the script spawns a python child that inherits our pipes, so
                # killing only the direct child would leave communicate()
                # blocked forever on the inherited descriptors.
                kill_process_tree(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    # A dying grandchild still holds the pipes open. Abandon
                    # the output rather than block this worker thread forever;
                    # the direct child is already dead, so reap it to avoid a
                    # zombie and close our pipe copies.
                    stdout, stderr = "", ""
                    for stream in (proc.stdout, proc.stderr):
                        try:
                            if stream is not None:
                                stream.close()
                        except Exception:
                            pass
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass
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
