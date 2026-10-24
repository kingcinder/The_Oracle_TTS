"""Structured, local-only diagnostics for a support report.

The doctor answers "does this install work?"; this module answers "what was
this machine doing?" in a shape that can sit inside a bug report. It is a
read-only snapshot: nothing is modified, nothing is downloaded, and no heavy
dependency is imported (Qt, torch, and the TTS engines are probed through
``importlib`` metadata only — a support report must never load a model).

Privacy contract (shared with ``the_oracle.support_bundle``): no absolute
paths, no usernames, no environment-variable values, no manuscript or audio
content. File *names* and sizes are included because that is what makes a
report actionable; the directory containing them is not.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from the_oracle import __version__
from the_oracle.support_bundle import (
    DOCTOR_TIMEOUT_SECONDS,
    redact_json,
    redact_text,
    run_doctor_report,
)

DIAGNOSTICS_VERSION = 1
FFMPEG_TIMEOUT_SECONDS = 5
MAX_LOGGED_FILES = 5


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return None


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _memory_mb() -> dict[str, int] | None:
    """Total/available RAM on Linux, total on macOS; None where neither read
    is available (Windows without extra dependencies)."""
    try:
        info = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        info = ""
    if info:
        values: dict[str, int] = {}
        for line in info.splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                parts = rest.split()
                if parts and parts[0].isdigit():
                    values[key] = int(parts[0]) // 1024
        if values:
            return {
                "total_mb": values.get("MemTotal", 0),
                "available_mb": values.get("MemAvailable", 0),
            }
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode == 0 and completed.stdout.strip().isdigit():
        return {"total_mb": int(completed.stdout.strip()) // (1024 * 1024)}
    return None


def _disk_mb(root: Path) -> dict[str, int] | None:
    try:
        usage = shutil.disk_usage(root)
    except OSError:
        return None
    return {
        "total_mb": int(usage.total // (1024 * 1024)),
        "free_mb": int(usage.free // (1024 * 1024)),
    }


def _ffmpeg() -> dict[str, Any]:
    binary = shutil.which("ffmpeg")
    if not binary:
        return {"present": False}
    payload: dict[str, Any] = {"present": True, "binary_name": Path(binary).name}
    try:
        completed = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return payload
    first_line = (completed.stdout or "").splitlines()[:1]
    if first_line:
        payload["version_line"] = redact_text(first_line[0])[:160]
    return payload


def _doctor_summary(
    root: Path,
    doctor_report: dict[str, Any] | None,
    run_doctor: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if doctor_report is None and run_doctor:
        doctor_report, problem = run_doctor_report(root, timeout=DOCTOR_TIMEOUT_SECONDS)
        if problem and len(problem) > 400:
            problem = problem[:400] + "…"
    else:
        problem = None if doctor_report is not None else "doctor report excluded"

    if doctor_report is None:
        return {"ran": False, "detail": problem or "doctor did not return a report"}, None

    failing = sorted(
        name
        for name, section in doctor_report.items()
        if isinstance(section, dict) and section.get("ok") is False
    )
    return {
        "ran": True,
        "ok": doctor_report.get("ok"),
        "failing_checks": failing,
        "ci_mode": doctor_report.get("ci_mode"),
    }, doctor_report


def _settings_summary(root: Path) -> dict[str, Any]:
    """A whitelist of harmless UI preferences. Never the recent-file list:
    those are absolute paths the bundle's settings copy handles separately
    (redacted), and a bug report does not need them."""
    path = root / "user_settings" / "app_settings.json"
    if not path.is_file():
        return {"present": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"present": True, "readable": False}
    if not isinstance(data, dict):
        return {"present": True, "readable": False}
    summary = {"present": True, "readable": True}
    for key in ("theme", "backend", "inference_backend", "model_variant", "remember_backend"):
        value = data.get(key)
        if isinstance(value, (str, bool, int, float)) and len(str(value)) <= 80:
            summary[key] = value
    return summary


def _logs_summary(root: Path) -> dict[str, Any]:
    directory = root / "logs"
    if not directory.is_dir():
        return {"present": False}
    entries: list[dict[str, Any]] = []
    try:
        candidates = sorted(
            (path for path in directory.iterdir() if path.is_file()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )[:MAX_LOGGED_FILES]
    except OSError:
        return {"present": True, "readable": False}
    for path in candidates:
        try:
            entries.append({"name": path.name, "bytes": path.stat().st_size})
        except OSError:
            entries.append({"name": path.name, "bytes": None})
    return {"present": True, "readable": True, "files": entries}


def _crash_summary(root: Path) -> dict[str, Any]:
    from the_oracle import crash

    records = crash.list_records(root)
    native = crash.crash_dir(root) / "native-crash.txt"
    native_bytes: int | None = None
    try:
        native_bytes = native.stat().st_size
    except OSError:
        native_bytes = None
    return {
        "capture_consent": crash.read_consent(root),
        "record_count": len(records),
        "newest_record": records[0].name if records else None,
        "native_dump_bytes": native_bytes,
    }


def _environment() -> dict[str, Any]:
    names = sorted(
        name
        for name in os.environ
        if name.startswith("ORACLE_")
        or name in {"HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES", "QT_QPA_PLATFORM", "PYTHONPATH"}
    )
    return {
        "variables_present": names,
        "values_included": False,
        "display_present": bool(os.environ.get("DISPLAY")),
        "wayland_present": bool(os.environ.get("WAYLAND_DISPLAY")),
    }


def _gpu_summary() -> dict[str, Any]:
    return {
        "torch_installed": _module_available("torch"),
        "torch_version": _dist_version("torch"),
        "cuda_visible_devices_set": "CUDA_VISIBLE_DEVICES" in os.environ,
        "vulkan_model_configured": bool(os.environ.get("ORACLE_AUDIOCPP_MODEL")),
        "vulkan_binary_configured": bool(os.environ.get("ORACLE_AUDIOCPP_BIN")),
    }


def _model_cache_summary() -> dict[str, Any]:
    cache_root = Path.home() / ".cache" / "huggingface"
    try:
        present = cache_root.is_dir()
        model_dirs = (
            sorted(child.name for child in cache_root.iterdir() if child.is_dir())[:10]
            if present
            else []
        )
    except OSError:
        present, model_dirs = False, []
    return {"hf_cache_present": present, "top_level_entries": model_dirs}


def collect_diagnostics(
    root: str | Path,
    *,
    doctor_report: dict[str, Any] | None = None,
    run_doctor: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Collect one JSON-safe, redacted diagnostics snapshot.

    ``doctor_report`` short-circuits the doctor subprocess (tests and callers
    that already ran it); ``run_doctor=False`` records that it was skipped.
    The returned dict is what a report stores and a support package writes as
    ``diagnostics.json``.
    """
    root = Path(root).resolve()
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    doctor, full_doctor = _doctor_summary(root, doctor_report, run_doctor)
    payload: dict[str, Any] = {
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "generated_utc": moment.astimezone(timezone.utc).isoformat(),
        "app_version": __version__,
        "install": {
            "root_kind": "install",
            "python_frozen": bool(getattr(sys, "frozen", False)),
            "executable_name": Path(sys.executable).name or "<python>",
        },
        "system": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor()[:120],
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "python_bits": 64 if sys.maxsize > 2**32 else 32,
            "libc": " ".join(part for part in platform.libc_ver() if part),
        },
        "memory": _memory_mb(),
        "disk": _disk_mb(root),
        "qt": {
            "pyside6_version": _dist_version("PySide6"),
            "qpa_platform": os.environ.get("QT_QPA_PLATFORM", ""),
        },
        "engines": {
            "chatterbox_available": _module_available("chatterbox"),
            "chatterbox_version": _dist_version("chatterbox-tts"),
            "sounddevice_available": _module_available("sounddevice"),
            "numpy_version": _dist_version("numpy"),
            "scipy_version": _dist_version("scipy"),
        },
        "gpu": _gpu_summary(),
        "models": _model_cache_summary(),
        "ffmpeg": _ffmpeg(),
        "settings": _settings_summary(root),
        "logs": _logs_summary(root),
        "crash": _crash_summary(root),
        "environment": _environment(),
        "doctor": doctor,
        "redaction": (
            "Absolute paths, usernames, serial numbers, credentials, email "
            "addresses, and environment-variable values are excluded. Log "
            "contents, manuscripts, audio, and recordings are never included "
            "in diagnostics."
        ),
    }
    return redact_json(payload)


def compact_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """The subset worth putting at the top of a bug report: enough for a
    triager to see the environment at a glance, without the long sections."""
    keep = (
        "generated_utc",
        "app_version",
        "system",
        "memory",
        "disk",
        "qt",
        "engines",
        "gpu",
        "ffmpeg",
        "crash",
        "doctor",
    )
    return {key: diagnostics.get(key) for key in keep if key in diagnostics}
