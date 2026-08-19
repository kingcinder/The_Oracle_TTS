"""Automatic CPU→GPU (Vulkan backend) setup for The Oracle.

Selecting the Vulkan inference backend (the GPU path) used to be a manual
two-step process: build ``audiocpp_cli`` from audio.cpp
(``scripts/build_audio_cpp.sh``) and download the Chatterbox ggml model
(``scripts/download_audio_cpp_model.sh``), then export ``ORACLE_AUDIOCPP_MODEL``
so the engine could find it. This module turns that whole switch into one call:

- ``vulkan_setup_needed()`` reports, in human terms, what is missing.
- ``run_vulkan_setup()`` builds the CLI if it is missing, downloads the model
  if it is missing, sets ``ORACLE_AUDIOCPP_CLI``/``ORACLE_AUDIOCPP_MODEL`` for
  the current process, and reports what happened (streaming script output to
  an optional progress callback and honouring an optional cancel event).

Both the desktop GUI (selecting the Vulkan backend, or clicking Render while
it is selected) and the render CLI (``--inference-backend vulkan``) call into
this module, so switching from CPU to GPU completes on its own. The manual
scripts remain for advanced/offline use and are exactly what this module
invokes.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from the_oracle.tts_engines.vulkan_backend import find_audiocpp_binary, find_audiocpp_model

# The export line scripts/download_audio_cpp_model.sh prints; parse the
# installed model path out of the captured output.
_EXPORT_MODEL_LINE = re.compile(r'export\s+ORACLE_AUDIOCPP_MODEL="([^"]+)"')


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_model_export(output: str) -> str:
    """Extract the installed model path from the download script's output.

    ``scripts/download_audio_cpp_model.sh`` prints an
    ``export ORACLE_AUDIOCPP_MODEL="<path>"`` line after a successful install;
    that path is what the Vulkan backend needs. Returns "" when the line is
    absent (e.g. dry-run or an error).
    """
    for line in output.splitlines():
        match = _EXPORT_MODEL_LINE.search(line)
        if match:
            return match.group(1)
    return ""


def vulkan_setup_needed(
    find_binary: Callable[[], Path | None] | None = None,
) -> list[str]:
    """Return human-readable reasons the Vulkan backend needs setup, else [].

    Mirrors the checks ``AudioCppVulkanEngine.ensure_model_ready`` performs
    (binary present, model env set, model file existing) so the GUI/CLI can
    decide to auto-setup at backend-selection time instead of failing deep
    inside the render worker. ``find_binary`` is injectable so callers that
    monkeypatch their own binary probe (the GUI tests) keep working.
    """
    probe = find_binary or find_audiocpp_binary
    missing: list[str] = []
    if probe() is None:
        missing.append(
            "audiocpp_cli is not built (run scripts/build_audio_cpp.sh or set ORACLE_AUDIOCPP_CLI)"
        )
    model_env = os.environ.get("ORACLE_AUDIOCPP_MODEL")
    if model_env and not Path(model_env).expanduser().exists():
        missing.append(f"the Chatterbox model path does not exist: {model_env} (fix ORACLE_AUDIOCPP_MODEL)")
    elif find_audiocpp_model() is None:
        missing.append(
            "the Chatterbox model is not downloaded (run scripts/download_audio_cpp_model.sh or set ORACLE_AUDIOCPP_MODEL)"
        )
    return missing


class VulkanSetupCancelled(Exception):
    """Raised internally when the caller cancels a running setup script."""


@dataclass(slots=True)
class VulkanSetupResult:
    """Outcome of :func:`run_vulkan_setup`."""

    ok: bool
    messages: list[str]
    binary: str | None = None
    model: str | None = None
    error: str = ""
    cancelled: bool = False


def _terminate_group(proc: subprocess.Popen) -> None:
    """SIGTERM the script's process group, tolerating a race with exit.

    The script runs in its own session (``start_new_session=True``) so this
    never touches The Oracle itself. ``ProcessLookupError`` means the child
    already exited between the cancel check and the kill, which is fine.
    """
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _run_script_streaming(
    script: Path,
    progress: Callable[[str], None] | None,
    cancel: threading.Event | None,
) -> tuple[int, str]:
    """Run one bash setup script, streaming its output lines to ``progress``.

    Returns ``(returncode, combined_output)``. When ``cancel`` is set while the
    script is running, the script's process group is SIGTERM'd and
    :class:`VulkanSetupCancelled` is raised (the caller converts it into a
    cancelled result). ``start_new_session`` gives the script its own process
    group so the kill never hits The Oracle itself; the audio.cpp path is
    Linux-only, matching the GUI's existing model-download thread.
    """
    if cancel is not None and cancel.is_set():
        raise VulkanSetupCancelled()
    proc = subprocess.Popen(
        ["bash", str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            if progress is not None:
                progress(line)
            if cancel is not None and cancel.is_set():
                _terminate_group(proc)
                raise VulkanSetupCancelled()
    finally:
        # Bounded reap: a script that ignores SIGTERM must not hang the GUI
        # thread forever, so escalate to a group SIGKILL after a short grace
        # period (killpg so grandchildren that ignored SIGTERM die too).
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
    return proc.returncode, "\n".join(lines)


def run_vulkan_setup(
    progress: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
    *,
    repo_root: Path | None = None,
    scripts_dir: Path | None = None,
    find_binary: Callable[[], Path | None] | None = None,
) -> VulkanSetupResult:
    """Automatically complete the CPU→GPU switch and report the outcome.

    Steps, in order:

    1. If ``audiocpp_cli`` is missing, run ``scripts/build_audio_cpp.sh``
       (which clones audio.cpp and compiles with ``ENGINE_ENABLE_VULKAN=ON``),
       then re-probe for the binary.
    2. If the Chatterbox model is missing or its path does not exist, run
       ``scripts/download_audio_cpp_model.sh`` and parse the
       ``ORACLE_AUDIOCPP_MODEL`` export line from its output.
    3. Set ``ORACLE_AUDIOCPP_CLI``/``ORACLE_AUDIOCPP_MODEL`` for the current
       process so the existing Vulkan backend just works without shell exports.

    ``progress`` (when given) receives each script output line; ``cancel``
    (when given) aborts the running script and returns a cancelled result.
    ``scripts_dir`` overrides the repo's ``scripts/`` directory (hermetic
    tests); ``find_binary`` overrides the binary probe.
    """
    root = Path(repo_root) if repo_root is not None else _repo_root()
    scripts = Path(scripts_dir) if scripts_dir is not None else root / "scripts"
    probe = find_binary or find_audiocpp_binary
    messages: list[str] = []

    def emit(line: str) -> None:
        messages.append(line)
        if progress is not None:
            progress(line)

    if not vulkan_setup_needed(find_binary=probe):
        emit("Vulkan backend is already configured; no setup required.")
        binary = probe()
        return VulkanSetupResult(
            ok=True,
            messages=messages,
            binary=str(binary) if binary is not None else None,
            model=os.environ.get("ORACLE_AUDIOCPP_MODEL"),
        )

    binary = probe()
    if binary is None:
        build_script = scripts / "build_audio_cpp.sh"
        if not build_script.exists():
            return VulkanSetupResult(
                ok=False,
                messages=messages,
                error=f"build script not found: {build_script}",
            )
        emit("Building audiocpp_cli (scripts/build_audio_cpp.sh)...")
        try:
            rc, output = _run_script_streaming(build_script, progress, cancel)
        except VulkanSetupCancelled:
            return VulkanSetupResult(ok=False, messages=messages, error="Setup cancelled.", cancelled=True)
        if rc != 0:
            return VulkanSetupResult(
                ok=False,
                messages=messages,
                error=f"audio.cpp build failed (exit {rc}):\n{output[-1000:]}",
            )
        binary = probe()
        if binary is None:
            return VulkanSetupResult(
                ok=False,
                messages=messages,
                error="audio.cpp build finished but audiocpp_cli was still not found.",
            )
        os.environ["ORACLE_AUDIOCPP_CLI"] = str(binary)
        emit(f"audiocpp_cli ready: {binary}")

    model_env = os.environ.get("ORACLE_AUDIOCPP_MODEL")
    model = Path(model_env).expanduser() if model_env else None
    if model is None or not model.exists():
        # A model already installed by the download script is picked up
        # automatically -- no download and no env var needed.
        found = find_audiocpp_model()
        if found is not None:
            model = found
            os.environ.setdefault("ORACLE_AUDIOCPP_MODEL", str(found))
            emit(f"Model ready: {found}")
            return VulkanSetupResult(
                ok=True,
                messages=messages,
                binary=str(binary) if binary is not None else None,
                model=str(model),
            )
        download_script = scripts / "download_audio_cpp_model.sh"
        if not download_script.exists():
            return VulkanSetupResult(
                ok=False,
                messages=messages,
                error=f"download script not found: {download_script}",
            )
        emit("Downloading the Chatterbox model (scripts/download_audio_cpp_model.sh)...")
        try:
            rc, output = _run_script_streaming(download_script, progress, cancel)
        except VulkanSetupCancelled:
            return VulkanSetupResult(ok=False, messages=messages, error="Setup cancelled.", cancelled=True)
        if rc != 0:
            return VulkanSetupResult(
                ok=False,
                messages=messages,
                error=f"model download failed (exit {rc}):\n{output[-1000:]}",
            )
        model_path = parse_model_export(output)
        if not model_path:
            return VulkanSetupResult(
                ok=False,
                messages=messages,
                error="download finished but no ORACLE_AUDIOCPP_MODEL export line was printed.",
            )
        if not Path(model_path).exists():
            return VulkanSetupResult(
                ok=False,
                messages=messages,
                error=f"download reported a model at {model_path} but the file is missing.",
            )
        os.environ["ORACLE_AUDIOCPP_MODEL"] = model_path
        model = Path(model_path)
        emit(f"Model ready: {model_path}")

    return VulkanSetupResult(
        ok=True,
        messages=messages,
        binary=str(binary) if binary is not None else None,
        model=str(model) if model is not None else None,
    )
