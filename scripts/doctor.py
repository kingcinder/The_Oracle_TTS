#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT_DEFAULT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from the_oracle.platform_support import (
    is_linux,
    is_windows,
    managed_launcher_dir,
    managed_launcher_path,
    path_entries,
    repo_bootstrap_display,
    repo_python_display,
    repo_run_display,
    venv_entrypoint_path,
)


JSON_PREFIX = "__ORACLE_TTS_JSON__"
MANAGED_WRAPPER_MARKER = "ORACLE_TTS_WRAPPER"
SUPPORTED_PYTHON_MIN = (3, 11)
SUPPORTED_PYTHON_MAX = (3, 13)
LIBRARY_PACKAGE_CANDIDATES: dict[str, list[str]] = {
    "libasound.so.2": ["libasound2t64", "libasound2"],
    "libdbus-1.so.3": ["libdbus-1-3"],
    "libEGL.so.1": ["libegl1"],
    "libfontconfig.so.1": ["libfontconfig1"],
    "libglib-2.0.so.0": ["libglib2.0-0t64", "libglib2.0-0"],
    "libGL.so.1": ["libgl1"],
    "libgobject-2.0.so.0": ["libglib2.0-0t64", "libglib2.0-0"],
    "libgthread-2.0.so.0": ["libglib2.0-0t64", "libglib2.0-0"],
    "libnss3.so": ["libnss3"],
    "libOpenGL.so.0": ["libopengl0"],
    "libpulse.so.0": ["libpulse0"],
    "libxcb-cursor.so.0": ["libxcb-cursor0"],
    "libxcb-icccm.so.4": ["libxcb-icccm4"],
    "libxcb-image.so.0": ["libxcb-image0"],
    "libxcb-keysyms.so.1": ["libxcb-keysyms1"],
    "libxcb-randr.so.0": ["libxcb-randr0"],
    "libxcb-render-util.so.0": ["libxcb-render-util0"],
    "libxcb-shape.so.0": ["libxcb-shape0"],
    "libxcb-sync.so.1": ["libxcb-sync1"],
    "libxcb-xfixes.so.0": ["libxcb-xfixes0"],
    "libxcb-xinerama.so.0": ["libxcb-xinerama0"],
    "libxkbcommon-x11.so.0": ["libxkbcommon-x11-0"],
}


def _prepend_repo_src(repo_root: Path) -> None:
    src_path = repo_root / "src"
    if src_path.exists():
        src_text = str(src_path)
        if src_text not in sys.path:
            sys.path.insert(0, src_text)


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _tail(text: str, lines: int = 8) -> str:
    if not text:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


def _run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except OSError as exc:
        # FileNotFoundError and PermissionError are both OSError subclasses;
        # a missing *or non-executable* probe binary must never crash the
        # doctor -- it returns a structured failure like any other probe.
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "error": str(exc),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"Timed out after {timeout:.0f}s",
            "timed_out": True,
        }

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": False,
    }


def _probe_environment(repo_root: Path, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    if extra_env:
        env.update(extra_env)
    return env


def _run_python_probe(
    repo_root: Path,
    code: str,
    *,
    timeout: float,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = _run_command(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=_probe_environment(repo_root, extra_env),
        timeout=timeout,
    )
    if result["timed_out"]:
        return {"ok": False, "error": result["error"], "stdout_tail": _tail(result["stdout"]), "stderr_tail": _tail(result["stderr"])}

    payload = None
    for stream in (result["stdout"], result["stderr"]):
        for line in reversed(stream.splitlines()):
            if line.startswith(JSON_PREFIX):
                payload = json.loads(line[len(JSON_PREFIX) :])
                break
        if payload is not None:
            break
    if payload is None:
        payload = {
            "ok": result["ok"],
            "error": result.get("error") or f"Probe returned {result['returncode']}",
        }
    payload["returncode"] = result["returncode"]
    payload["stdout_tail"] = _tail(result["stdout"])
    payload["stderr_tail"] = _tail(result["stderr"])
    return payload


@lru_cache(maxsize=None)
def _package_installed(package_name: str) -> bool:
    if not is_linux():
        return False
    result = _run_command(["dpkg-query", "-W", "-f=${Status}", package_name], timeout=10)
    return result["ok"] and result["stdout"].strip().endswith("installed")


@lru_cache(maxsize=None)
def _package_available(package_name: str) -> bool:
    if not is_linux():
        return False
    result = _run_command(["apt-cache", "show", package_name], timeout=10)
    return result["ok"] and bool(result["stdout"].strip())


def _preferred_package(candidates: list[str]) -> str:
    for candidate in candidates:
        if _package_installed(candidate):
            return candidate
    for candidate in candidates:
        if _package_available(candidate):
            return candidate
    return candidates[0]


def _qt_package_suggestions(missing_libraries: list[str]) -> list[str]:
    suggestions: list[str] = []
    for library in missing_libraries:
        candidates = LIBRARY_PACKAGE_CANDIDATES.get(library)
        if not candidates:
            continue
        suggestions.append(_preferred_package(candidates))
    return sorted(set(suggestions))


def _python_status() -> dict[str, Any]:
    version_tuple = sys.version_info[:3]
    ok = SUPPORTED_PYTHON_MIN <= version_tuple < SUPPORTED_PYTHON_MAX
    return {
        "ok": ok,
        "executable": sys.executable,
        "version": platform.python_version(),
    }


def _ffmpeg_status() -> dict[str, Any]:
    path = shutil.which("ffmpeg")
    return {"ok": path is not None, "path": path or ""}


def _entrypoint_status(repo_root: Path) -> dict[str, Any]:
    venv_entrypoint = venv_entrypoint_path(repo_root, "the-oracle")
    wrapper_path = managed_launcher_path("the-oracle")
    path_entrypoint = shutil.which("the-oracle")
    if is_windows() and not path_entrypoint:
        path_entrypoint = shutil.which("the-oracle.cmd")
    managed_wrapper = False
    if wrapper_path.exists():
        try:
            managed_wrapper = MANAGED_WRAPPER_MARKER in wrapper_path.read_text(encoding="utf-8")
        except Exception:
            managed_wrapper = False

    help_target = None
    if venv_entrypoint.exists():
        help_target = str(venv_entrypoint)
    elif path_entrypoint:
        help_target = path_entrypoint

    help_result = {"ok": False, "returncode": 127, "stdout": "", "stderr": ""}
    if help_target:
        help_result = _run_command([help_target, "--help"], cwd=repo_root, timeout=30)

    if is_windows():
        fresh_shell = _run_command(
            ["cmd", "/d", "/c", "where the-oracle >nul 2>nul && the-oracle --help >nul 2>nul"],
            cwd=repo_root,
            timeout=30,
        )
    elif shutil.which("bash"):
        fresh_shell = _run_command(
            ["bash", "-lc", "command -v the-oracle && the-oracle --help >/dev/null"],
            cwd=repo_root,
            timeout=30,
        )
    else:
        fresh_shell = {"ok": help_result["ok"], "stdout": help_target or "", "stderr": help_result["stderr"]}
    normalized_entries = {entry.lower() if is_windows() else entry for entry in path_entries()}
    launcher_dir = str(managed_launcher_dir())
    launcher_entry = launcher_dir.lower() if is_windows() else launcher_dir
    path_has_local_bin = launcher_entry in normalized_entries

    return {
        "ok": bool(help_target) and help_result["ok"] and fresh_shell["ok"],
        "venv_entrypoint": str(venv_entrypoint),
        "venv_entrypoint_exists": venv_entrypoint.exists(),
        "path_entrypoint": path_entrypoint or "",
        "managed_wrapper_path": str(wrapper_path),
        "managed_wrapper_installed": managed_wrapper,
        "help_ok": help_result["ok"],
        "help_error": help_result["stderr"] or help_result["stdout"],
        "fresh_shell_help_ok": fresh_shell["ok"],
        "fresh_shell_path": fresh_shell["stdout"].strip(),
        "fresh_shell_error": fresh_shell["stderr"].strip() or ("the-oracle is not available in a fresh shell PATH" if not fresh_shell["ok"] else ""),
        "path_has_local_bin": path_has_local_bin,
    }


def _chatterbox_probe(repo_root: Path, timeout: float, skip_model_init: bool) -> dict[str, Any]:
    code = f"""
from __future__ import annotations
import json
import time

payload = {{}}
try:
    import perth
except Exception as exc:
    payload["perth_ok"] = False
    payload["perth_error"] = f"{{type(exc).__name__}}: {{exc}}"
    payload["watermarker_callable"] = False
else:
    watermarker = getattr(perth, "PerthImplicitWatermarker", None)
    payload["perth_ok"] = True
    payload["watermarker_callable"] = callable(watermarker)
    payload["watermarker_symbol"] = str(watermarker)

try:
    from chatterbox.tts import ChatterboxTTS
except Exception as exc:
    payload["import_ok"] = False
    payload["import_error"] = f"{{type(exc).__name__}}: {{exc}}"
    payload["init_ok"] = False
else:
    payload["import_ok"] = True
    payload["import_target"] = "from chatterbox.tts import ChatterboxTTS"
    payload["constructor_symbol"] = str(ChatterboxTTS)
    if {skip_model_init!r}:
        payload["init_ok"] = False
        payload["init_skipped"] = True
    else:
        try:
            started = time.perf_counter()
            model = ChatterboxTTS.from_pretrained(device="cpu")
        except Exception as exc:
            payload["init_ok"] = False
            payload["init_error"] = f"{{type(exc).__name__}}: {{exc}}"
        else:
            payload["init_ok"] = True
            payload["init_seconds"] = round(time.perf_counter() - started, 3)
            payload["sample_rate"] = int(getattr(model, "sr", 0) or 0)

print({JSON_PREFIX!r} + json.dumps(payload))
"""
    probe = _run_python_probe(repo_root, code, timeout=timeout, extra_env={"PYTHONWARNINGS": "ignore"})
    probe["ok"] = bool(probe.get("import_ok")) and (skip_model_init or bool(probe.get("init_ok"))) and bool(probe.get("perth_ok"))
    return probe


def _find_qt_xcb_plugin() -> Path | None:
    if not is_linux():
        return None
    try:
        from PySide6 import __file__ as pyside_file
        from PySide6.QtCore import QLibraryInfo
    except Exception:
        return None

    candidates = []
    try:
        plugins_root = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
        candidates.append(plugins_root / "platforms" / "libqxcb.so")
    except Exception:
        pass

    pyside_root = Path(pyside_file).resolve().parent
    candidates.append(pyside_root / "Qt" / "plugins" / "platforms" / "libqxcb.so")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1] if candidates else None


def _qt_status(repo_root: Path, timeout: float) -> dict[str, Any]:
    try:
        import PySide6  # noqa: F401
    except Exception as exc:
        return {
            "ok": False,
            "import_ok": False,
            "plugin_path": "",
            "plugin_exists": False,
            "missing_libraries": [],
            "suggested_packages": [],
            "offscreen_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    plugin_path = _find_qt_xcb_plugin()
    if is_linux() and plugin_path is None:
        return {
            "ok": False,
            "import_ok": True,
            "plugin_path": "",
            "plugin_exists": False,
            "missing_libraries": [],
            "suggested_packages": [],
            "offscreen_ok": False,
            "error": "Could not locate PySide6 xcb platform plugin.",
        }

    missing_libraries: list[str] = []
    ldd_result = {"ok": True, "stderr": "", "stdout": ""}
    if is_linux() and plugin_path is not None:
        ldd_result = _run_command(["ldd", str(plugin_path)], timeout=30)
        if ldd_result["ok"]:
            for line in ldd_result["stdout"].splitlines():
                if "=> not found" in line:
                    missing_libraries.append(line.split("=>", 1)[0].strip())

    offscreen_code = f"""
from __future__ import annotations
import json
import os

payload = {{}}
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    player = QMediaPlayer()
    audio = QAudioOutput()
    player.setAudioOutput(audio)
except Exception as exc:
    payload["ok"] = False
    payload["error"] = f"{{type(exc).__name__}}: {{exc}}"
else:
    payload["ok"] = True
    payload["qt_platform"] = app.platformName()
    payload["qmedia_player"] = str(type(player).__name__)
    app.quit()

print({JSON_PREFIX!r} + json.dumps(payload))
"""
    offscreen = _run_python_probe(repo_root, offscreen_code, timeout=timeout, extra_env={"QT_QPA_PLATFORM": "offscreen"})
    suggested_packages = _qt_package_suggestions(missing_libraries)
    return {
        "ok": (not is_linux() or bool(plugin_path and plugin_path.exists())) and not missing_libraries and bool(offscreen.get("ok")),
        "import_ok": True,
        "plugin_path": str(plugin_path) if plugin_path is not None else "",
        "plugin_exists": bool(plugin_path and plugin_path.exists()) if is_linux() else True,
        "missing_libraries": missing_libraries,
        "suggested_packages": suggested_packages,
        "offscreen_ok": bool(offscreen.get("ok")),
        "offscreen_error": offscreen.get("error") or offscreen.get("stderr_tail", ""),
        "qt_platform": offscreen.get("qt_platform", ""),
        "ldd_error": "" if ldd_result["ok"] else ldd_result["stderr"] or ldd_result["stdout"],
    }


def _deterministic_smoke_status(repo_root: Path) -> dict[str, Any]:
    _prepend_repo_src(repo_root)
    try:
        from unittest.mock import patch

        from the_oracle.smoke import run_deterministic_smoke_render
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    output_root = repo_root / "build" / "doctor_deterministic_smoke"
    started = time.perf_counter()
    try:
        # Keep the doctor smoke deterministic and lightweight by forcing the
        # text-repair helpers onto their built-in fallback paths.
        with (
            patch("the_oracle.text_repair.grammar.GrammarCorrector._try_load_language_tool", return_value=None),
            patch("the_oracle.text_repair.punctuation.PunctuationRestorer._try_load_punctuator", return_value=None),
        ):
            result = run_deterministic_smoke_render(output_root, source_format="txt")
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": True,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "output_path": str(result.output_path),
        "project_dir": str(result.project_dir),
        "cache_reused_on_second_pass": result.cache_reused_on_second_pass,
    }


def _real_engine_smoke_status(repo_root: Path) -> dict[str, Any]:
    _prepend_repo_src(repo_root)
    try:
        from the_oracle.real_engine_smoke import ensure_real_engine_inputs, real_engine_smoke_prerequisites
    except Exception as exc:
        return {"ok": False, "ready": False, "error": f"{type(exc).__name__}: {exc}"}

    output_root = repo_root / "build" / "real_engine_smoke"
    try:
        ensure_real_engine_inputs(output_root)
        readiness = real_engine_smoke_prerequisites(output_root)
    except Exception as exc:
        return {"ok": False, "ready": False, "error": f"{type(exc).__name__}: {exc}"}

    return {"ok": bool(readiness.get("ready")), **readiness}


_VULKAN_PATCH_MARKER = "ORACLE VENDORED PATCH"


def _vulkaninfo_summary() -> tuple[bool, str]:
    """Return (vulkan_device_visible, vulkaninfo_text).

    ``vulkaninfo --summary`` exits 0 only when at least one device is visible,
    so the exit code is the device-visibility probe.
    """
    if not shutil.which("vulkaninfo"):
        return False, ""
    result = _run_command(["vulkaninfo", "--summary"], timeout=15)
    return result["ok"], f"{result['stdout']}\n{result['stderr']}"


def _first_device_name(vulkaninfo_text: str) -> str:
    for line in vulkaninfo_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("deviceName"):
            return stripped.split("=", 1)[-1].strip()
    return ""


_AUDIOCPP_DEVICE_LINE = re.compile(r'^Vulkan:(\d+)\s+"([^"]+)"')


def _audiocpp_devices(binary: Path | None) -> list[dict[str, Any]]:
    """Return the Vulkan devices audio.cpp reports via ``--list-devices``.

    Each entry is ``{"index": <n>, "name": "..."}`` where ``<n>`` is the value
    to pass as ``ORACLE_AUDIOCPP_DEVICE`` / ``--device <n>``. audio.cpp's own
    indexes are what the backend uses, so this is the authoritative answer for
    multi-GPU machines (vulkaninfo alone cannot tell us which index audio.cpp
    picks). Empty when the binary is missing or reports nothing.
    """
    if binary is None or not Path(binary).exists():
        return []
    result = _run_command([str(binary), "--backend", "vulkan", "--list-devices"], timeout=30)
    if not result["ok"]:
        return []
    devices: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    # ggml builds often print device discovery lines to stderr; parse both
    # streams so a stream change can't silently report zero devices, and
    # dedupe by index in case a build echoes the same device to both.
    for stream in (result["stdout"], result["stderr"]):
        for line in (stream or "").splitlines():
            match = _AUDIOCPP_DEVICE_LINE.match(line.strip())
            if match:
                index = int(match.group(1))
                if index in seen_indexes:
                    continue
                seen_indexes.add(index)
                devices.append({"index": index, "name": match.group(2)})
    return devices


def _vulkan_patches_applied(repo_root: Path) -> bool | None:
    """None when audio.cpp is not cloned; True/False when it is."""
    markers = [
        repo_root / "audio.cpp" / "external" / "ggml" / "src" / "ggml-vulkan" / "ggml-vulkan.cpp",
        repo_root / "audio.cpp" / "external" / "sentencepiece" / "CMakeLists.txt",
    ]
    if not markers[0].exists():
        return None
    try:
        return all(_VULKAN_PATCH_MARKER in path.read_text(encoding="utf-8", errors="ignore") for path in markers)
    except Exception:
        return False


def _vulkan_caveat(*, rdna1_device: bool, patched: bool | None, binary_built: bool) -> str:
    notes: list[str] = []
    if rdna1_device:
        notes.append(
            "RDNA1 (gfx1010/gfx1012) GPU detected: audio.cpp's ggml can hit "
            "VK_ERROR_DEVICE_LOST during buffer init unless the vendored "
            "ORACLE_VENDORED ggml patch is applied (whisper.cpp#3611)."
        )
    if binary_built and patched is False:
        notes.append(
            "The vendored RDNA1 ggml patch is NOT applied to audio.cpp/external/ggml; "
            "re-run scripts/patch_audio_cpp_ggml.sh and rebuild before using "
            "--inference-backend vulkan on RDNA1."
        )
    if notes:
        notes.append("Fall back to --inference-backend pytorch if the device-lost error still fires.")
    return " ".join(notes)


def _vulkan_backend_status(repo_root: Path) -> dict[str, Any]:
    """Informational (opt-in) check: binary, model env, Vulkan device, RDNA1 caveat.

    The Vulkan backend is opt-in (inference_backend: vulkan, default pytorch), so
    this never gates overall_ready -- it reports readiness and surfaces the RDNA1
    device-lost caveat when an RDNA1 GPU or an unpatched clone is detected.
    """
    _prepend_repo_src(repo_root)
    try:
        from the_oracle.tts_engines.vulkan_backend import _vulkan_batch_max_requests, find_audiocpp_binary, find_audiocpp_model
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "binary_built": False,
            "binary": "",
            "cli_override": "",
            "model_override_set": False,
            "model_path": "",
            "model_file_exists": False,
            "model_auto_found": False,
            "vulkan_device": False,
            "device_name": "",
            "rdna1_device": False,
            "vendored_patch_applied": None,
            "device_index_env": "",
            "threads_env": "",
            "batch_env": "",
            "effective_batch_cap": 32,
            "audio_cpp_devices": [],
            "caveat": "",
        }

    binary = find_audiocpp_binary()
    model_env = os.environ.get("ORACLE_AUDIOCPP_MODEL", "")
    model_override = Path(model_env).expanduser() if model_env else None
    # Mirrors AudioCppVulkanEngine.ensure_model_ready: the env override always
    # wins (a dangling value must surface, not silently fall through); without
    # one, the repo-local install the download script writes to is
    # auto-detected -- no export required.
    if model_override is not None:
        model_path = model_override
        auto_model = None
    else:
        auto_model = find_audiocpp_model()
        model_path = auto_model
    model_file_exists = bool(model_path and model_path.exists())
    model_auto_found = auto_model is not None
    device_available, vulkaninfo_text = _vulkaninfo_summary()
    lowered = vulkaninfo_text.lower()
    rdna1_device = any(token in lowered for token in ("rdna1", "navi1", "navi10", "navi14"))
    patched = _vulkan_patches_applied(repo_root)
    # The effective per-subprocess request cap: same single source of truth as
    # the engine (ORACLE_AUDIOCPP_MAX_BATCH, clamped >= 1, default 32).
    batch_env = os.environ.get("ORACLE_AUDIOCPP_MAX_BATCH", "")
    effective_batch_cap = _vulkan_batch_max_requests()
    return {
        "ok": bool(binary) and model_file_exists and device_available,
        "binary_built": binary is not None,
        "binary": str(binary) if binary else "",
        "cli_override": os.environ.get("ORACLE_AUDIOCPP_CLI", ""),
        "model_override_set": bool(model_env),
        "model_path": str(model_path) if model_path else "",
        "model_file_exists": model_file_exists,
        "model_auto_found": model_auto_found,
        "vulkan_device": device_available,
        "device_name": _first_device_name(vulkaninfo_text),
        "rdna1_device": rdna1_device,
        "vendored_patch_applied": patched,
        "device_index_env": os.environ.get("ORACLE_AUDIOCPP_DEVICE", ""),
        "threads_env": os.environ.get("ORACLE_AUDIOCPP_THREADS", ""),
        "batch_env": batch_env,
        "effective_batch_cap": effective_batch_cap,
        "audio_cpp_devices": _audiocpp_devices(binary),
        "caveat": _vulkan_caveat(rdna1_device=rdna1_device, patched=patched, binary_built=binary is not None),
        "error": "",
    }


def _turbo_status(repo_root: Path, timeout: float) -> dict[str, Any]:
    code = f"""
from __future__ import annotations
import json

from the_oracle.tts_engines.chatterbox_engine import turbo_readiness_report

payload = turbo_readiness_report(device="cpu")
print({JSON_PREFIX!r} + json.dumps(payload))
"""
    probe = _run_python_probe(repo_root, code, timeout=timeout, extra_env={"PYTHONWARNINGS": "ignore"})
    return {
        "ok": bool(probe.get("ok")),
        "cached": bool(probe.get("cached")),
        "checkpoint_dir": probe.get("checkpoint_dir", ""),
        "sample_rate": probe.get("sample_rate"),
        "error": probe.get("error") or probe.get("stderr_tail", "") or probe.get("stdout_tail", ""),
    }


def _build_next_steps(report: dict[str, Any], *, ci_mode: bool) -> list[str]:
    steps: list[str] = []
    if not report["python"]["ok"]:
        if is_windows():
            steps.append(r"Install Python 3.12, make sure the `py` launcher can find it, then rerun .\bootstrap_oracle_tts.ps1.")
        else:
            steps.append("Install Python 3.12 with venv support: sudo apt install python3.12 python3.12-venv")

    runtime_packages: list[str] = []
    if not report["ffmpeg"]["ok"] and not ci_mode:
        if is_windows():
            steps.append("Install FFmpeg and add it to PATH, then rerun the doctor.")
        else:
            runtime_packages.append("ffmpeg")
    runtime_packages.extend(report["qt"]["suggested_packages"] if not ci_mode else [])
    if runtime_packages and is_linux():
        unique_packages = " ".join(sorted(set(runtime_packages)))
        steps.append(f"Install the missing Linux runtime packages: sudo apt install {unique_packages}")

    if not report["entrypoint"]["ok"] and not ci_mode:
        steps.append(
            f"Re-run {repo_bootstrap_display()} to refresh the project venv and install the managed launcher at {managed_launcher_path()}."
        )
        if not report["entrypoint"]["path_has_local_bin"]:
            if is_windows():
                steps.append(f"Add {managed_launcher_dir()} to PATH, open a new PowerShell session, and retry.")
            else:
                steps.append(f'Add {managed_launcher_dir()} to PATH, open a fresh shell, and retry: export PATH="{managed_launcher_dir()}:$PATH"')

    chatterbox_init_blocked = not report["chatterbox_init"]["ok"] and not report["chatterbox_init"]["skipped"]
    if not report["chatterbox_import"]["ok"] or chatterbox_init_blocked or not report["perth"]["ok"]:
        steps.append(f"Re-run {repo_bootstrap_display()} with internet access so Chatterbox and Perth can be installed and cached on CPU.")

    if not report["deterministic_smoke"]["ok"]:
        steps.append(f"Inspect the deterministic smoke failure above, then retry with {repo_python_display()} scripts/download_models.py or {repo_python_display()} scripts/smoke_render.py as needed.")

    if not report["real_engine_smoke"]["ok"]:
        steps.append("Real-engine smoke becomes ready after the Chatterbox import/init and Perth checks pass.")

    if not report["turbo"]["ok"] and not ci_mode:
        steps.append(f"Optional turbo prefetch: {repo_python_display()} scripts/download_models.py --variant turbo --device cpu")

    vulkan = report["vulkan_backend"]
    if vulkan["binary_built"] and not vulkan["model_file_exists"]:
        if vulkan.get("model_override_set"):
            steps.append(
                f"Vulkan backend: ORACLE_AUDIOCPP_MODEL points at a missing model file "
                f"({vulkan.get('model_path') or '?'}); fix the variable before "
                f"rendering on Vulkan."
            )
        else:
            steps.append(
                "Vulkan backend: the Chatterbox model is not downloaded; run "
                "`the-oracle setup-vulkan` (or select the Vulkan backend in the GUI) "
                "to fetch it automatically, or scripts/download_audio_cpp_model.sh "
                "(scripts/build_audio_cpp.sh --with-model builds and fetches in one)."
            )
    if vulkan["vendored_patch_applied"] is False:
        steps.append(
            "Vulkan backend: re-run scripts/patch_audio_cpp_ggml.sh to (re)apply the vendored "
            "RDNA1 ggml patch, then rebuild with scripts/build_audio_cpp.sh."
        )
    audio_cpp_devices = vulkan.get("audio_cpp_devices") or []
    if len(audio_cpp_devices) > 1 and not vulkan.get("device_index_env"):
        indexes = ", ".join(str(device["index"]) for device in audio_cpp_devices)
        steps.append(
            f"Vulkan backend: {len(audio_cpp_devices)} GPUs detected by audio.cpp "
            f"(indexes {indexes}); set ORACLE_AUDIOCPP_DEVICE to choose which one renders."
        )
    # The per-subprocess request cap is sensible in roughly 1-128; warn when
    # it is outside that, since a tiny cap destroys batching and a huge one
    # risks an oversized requests.json. The reported effective cap is clamped
    # >= 1 (engine semantics), so check the raw env value for the lower bound:
    # a user setting 0 or -5 silently clamps to 1 and must still be surfaced.
    effective_batch_cap = vulkan.get("effective_batch_cap", 32)
    batch_env = vulkan.get("batch_env") or ""
    raw_cap: int | None = None
    try:
        if batch_env:
            raw_cap = int(batch_env)
    except ValueError:
        raw_cap = None
    cap_too_small = raw_cap is not None and raw_cap < 1
    cap_too_large = effective_batch_cap > 128
    if cap_too_small or cap_too_large:
        steps.append(
            f"Vulkan backend: effective batch cap is {effective_batch_cap} "
            f"(ORACLE_AUDIOCPP_MAX_BATCH={batch_env or 'unset'}); set it "
            f"to a value in the sensible 1-128 range or unset it for the default 32."
        )
    if report["voice_sources"]["primary_source"] != "seashells":
        steps.append("Add curated local reference clips to ./Seashells so the GUI stops defaulting to smoke/build fallback voices.")

    if not steps:
        steps.append(f"Ready to launch: {repo_run_display()}")
    return steps


def run(repo_root: Path, *, model_timeout: float, qt_timeout: float, skip_model_init: bool, ci_mode: bool) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    _prepend_repo_src(repo_root)
    from the_oracle.voice_catalog import voice_catalog_audit

    chatterbox_probe = _chatterbox_probe(repo_root, timeout=model_timeout, skip_model_init=skip_model_init)
    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "platform": platform.platform(),
        "ci_mode": ci_mode,
        "python": _python_status(),
        "ffmpeg": _ffmpeg_status(),
        "entrypoint": _entrypoint_status(repo_root),
        "chatterbox_import": {
            "ok": bool(chatterbox_probe.get("import_ok")),
            "target": chatterbox_probe.get("import_target", "from chatterbox.tts import ChatterboxTTS"),
            "constructor_symbol": chatterbox_probe.get("constructor_symbol", ""),
            "error": chatterbox_probe.get("import_error", ""),
        },
        "chatterbox_init": {
            "ok": bool(chatterbox_probe.get("init_ok")),
            "device": "cpu",
            "seconds": chatterbox_probe.get("init_seconds"),
            "sample_rate": chatterbox_probe.get("sample_rate"),
            "skipped": bool(chatterbox_probe.get("init_skipped")),
            "error": chatterbox_probe.get("init_error") or chatterbox_probe.get("error", ""),
        },
        "perth": {
            "ok": bool(chatterbox_probe.get("perth_ok")) and bool(chatterbox_probe.get("watermarker_callable")),
            "watermarker_callable": bool(chatterbox_probe.get("watermarker_callable")),
            "watermarker_symbol": chatterbox_probe.get("watermarker_symbol", ""),
            "error": chatterbox_probe.get("perth_error", ""),
        },
        "turbo": _turbo_status(repo_root, timeout=model_timeout),
        "qt": _qt_status(repo_root, timeout=qt_timeout),
        "voice_sources": voice_catalog_audit(repo_root),
        "deterministic_smoke": _deterministic_smoke_status(repo_root),
        "real_engine_smoke": _real_engine_smoke_status(repo_root),
        "vulkan_backend": _vulkan_backend_status(repo_root),
    }
    required_checks = [
        report["python"]["ok"],
        report["chatterbox_import"]["ok"],
        report["perth"]["ok"],
        skip_model_init or report["chatterbox_init"]["ok"],
        report["qt"]["ok"],
        report["deterministic_smoke"]["ok"],
        report["real_engine_smoke"]["ok"],
    ]
    if not ci_mode:
        required_checks.extend([report["ffmpeg"]["ok"], report["entrypoint"]["ok"]])
    report["overall_ready"] = all(required_checks)
    report["next_steps"] = _build_next_steps(report, ci_mode=ci_mode)
    return report


def _print_human_report(report: dict[str, Any]) -> None:
    optional_status = "WARN" if report.get("ci_mode") else "FAIL"
    print(f"Repo root: {report['repo_root']}")
    print(f"Platform: {report['platform']}")
    print(f"{_status(report['python']['ok'])} Python: {report['python']['executable']} ({report['python']['version']})")

    ffmpeg_detail = report["ffmpeg"]["path"] or "ffmpeg not found on PATH"
    ffmpeg_label = _status(report["ffmpeg"]["ok"]) if report["ffmpeg"]["ok"] or not report.get("ci_mode") else optional_status
    print(f"{ffmpeg_label} Runtime tool `ffmpeg`: {ffmpeg_detail}")

    entrypoint = report["entrypoint"]
    entrypoint_detail = entrypoint["fresh_shell_path"] or entrypoint["path_entrypoint"] or entrypoint["venv_entrypoint"]
    if entrypoint["ok"]:
        print(f"{_status(True)} the-oracle entrypoint: {entrypoint_detail}")
    else:
        detail = entrypoint["fresh_shell_error"] or entrypoint["help_error"] or "the-oracle --help failed"
        label = _status(False) if not report.get("ci_mode") else optional_status
        print(f"{label} the-oracle entrypoint: {detail}")

    chatterbox_import = report["chatterbox_import"]
    if chatterbox_import["ok"]:
        print(f"{_status(True)} Chatterbox import: {chatterbox_import['target']}")
    else:
        print(f"{_status(False)} Chatterbox import: {chatterbox_import['error']}")

    chatterbox_init = report["chatterbox_init"]
    if chatterbox_init["ok"]:
        print(
            f"{_status(True)} Chatterbox CPU init: from_pretrained(device=\"cpu\") in {chatterbox_init['seconds']}s"
        )
    elif chatterbox_init["skipped"]:
        print("SKIP Chatterbox CPU init: skipped")
    else:
        print(f"{_status(False)} Chatterbox CPU init: {chatterbox_init['error']}")

    perth = report["perth"]
    if perth["ok"]:
        print(f"{_status(True)} Perth watermarker: {perth['watermarker_symbol']}")
    else:
        detail = perth["error"] or "PerthImplicitWatermarker is unavailable"
        print(f"{_status(False)} Perth watermarker: {detail}")

    turbo = report["turbo"]
    if turbo["ok"]:
        detail = turbo["checkpoint_dir"] or "cached checkpoint available"
        print(f"{_status(True)} Turbo readiness: {detail}")
    else:
        label = _status(False) if not report.get("ci_mode") else optional_status
        print(f"{label} Turbo readiness: {turbo['error']}")

    voice_sources = report["voice_sources"]
    voice_detail = (
        f"{voice_sources['default_voice_assessment']} "
        f"Seashells={voice_sources['seashell_clip_count']}, fallback={voice_sources['fallback_clip_count']}"
    )
    print(f"{_status(voice_sources['ok'])} Default voice sources: {voice_detail}")
    print(f"Voice assets: {voice_sources['better_local_assets_detail']}")
    print(f"Voice mixing: {voice_sources['voice_mixing_detail']}")

    qt = report["qt"]
    if qt["ok"]:
        detail = qt["plugin_path"] or qt["qt_platform"] or "offscreen probe passed"
        print(f"{_status(True)} Qt GUI prerequisites: {detail}")
    else:
        detail = qt["error"] if "error" in qt else qt["offscreen_error"] or qt["ldd_error"] or "Qt prerequisites failed"
        print(f"{_status(False)} Qt GUI prerequisites: {detail}")
        if qt["missing_libraries"]:
            print(f"Missing Qt libraries: {', '.join(qt['missing_libraries'])}")
        if qt["suggested_packages"]:
            print(f"Suggested packages: {' '.join(qt['suggested_packages'])}")

    deterministic = report["deterministic_smoke"]
    if deterministic["ok"]:
        print(f"{_status(True)} Deterministic smoke readiness: {deterministic['output_path']}")
    else:
        print(f"{_status(False)} Deterministic smoke readiness: {deterministic['error']}")

    real_engine = report["real_engine_smoke"]
    if real_engine["ok"]:
        print(f"{_status(True)} Real-engine smoke readiness: {real_engine['expected_paths']['output']}")
    else:
        detail = real_engine.get("error") or str(real_engine.get("chatterbox_import", {}))
        print(f"{_status(False)} Real-engine smoke readiness: {detail}")

    # Opt-in backend: never a hard FAIL (machines without Vulkan are fine), but
    # readiness and the RDNA1 device-lost caveat are surfaced for the user.
    vulkan = report["vulkan_backend"]
    vulkan_label = "PASS" if vulkan["ok"] else "WARN"
    model_state = "set"
    if not vulkan["model_override_set"]:
        model_state = "unset (auto-detected)" if vulkan.get("model_file_exists") else "unset"
    elif vulkan.get("model_file_exists") is False:
        model_state = f"set but file missing ({vulkan.get('model_path') or '?'})"
    parts = [
        f"binary={'built' if vulkan['binary_built'] else 'not built'}",
        f"ORACLE_AUDIOCPP_MODEL={model_state}",
        f"vulkan device={'yes' if vulkan['vulkan_device'] else 'no'}",
    ]
    if vulkan["device_name"]:
        parts.append(vulkan["device_name"])
    if vulkan["rdna1_device"]:
        parts.append("RDNA1")
    if vulkan.get("device_index_env"):
        parts.append(f"device={vulkan['device_index_env']} (env)")
    if vulkan.get("threads_env"):
        parts.append(f"threads={vulkan['threads_env']} (env)")
    # The effective cap always derives from the env var (or the 32 default) --
    # per-render settings like the GUI spin box live in user-chosen settings
    # files, which the env-level doctor does not read.
    if vulkan.get("batch_env"):
        parts.append(f"batch cap={vulkan.get('effective_batch_cap', 32)} (env)")
    if vulkan["vendored_patch_applied"] is None:
        parts.append("vendored patches=not cloned")
    else:
        parts.append(f"vendored patches={'applied' if vulkan['vendored_patch_applied'] else 'MISSING'}")
    print(f"{vulkan_label} Vulkan backend (audio.cpp, opt-in): {', '.join(parts)}")
    if vulkan["caveat"]:
        print(f"      {vulkan['caveat']}")
    if not vulkan["ok"] and vulkan["error"]:
        print(f"      {vulkan['error']}")
    audio_cpp_devices = vulkan.get("audio_cpp_devices") or []
    if audio_cpp_devices:
        labels = ", ".join(f"{device['index']}={device['name']}" for device in audio_cpp_devices)
        print(f"      audio.cpp Vulkan devices: {labels}")
        if len(audio_cpp_devices) > 1 and not vulkan.get("device_index_env"):
            indexes = ", ".join(str(device["index"]) for device in audio_cpp_devices)
            print(f"      Multi-GPU: set ORACLE_AUDIOCPP_DEVICE to one of [{indexes}] to pick a device.")

    print("")
    print("Next steps:")
    for step in report["next_steps"]:
        print(f"- {step}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install and launch diagnostics for The Oracle.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--model-timeout", type=float, default=1800.0)
    parser.add_argument("--qt-timeout", type=float, default=60.0)
    parser.add_argument("--skip-model-init", action="store_true")
    parser.add_argument("--ci", action="store_true", help="Ignore optional environment-only checks such as ffmpeg, wrapper PATH, and turbo prefetch.")
    args = parser.parse_args(argv)

    report = run(
        args.repo_root,
        model_timeout=args.model_timeout,
        qt_timeout=args.qt_timeout,
        skip_model_init=args.skip_model_init,
        ci_mode=args.ci,
    )
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        _print_human_report(report)
    return 0 if report["overall_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
