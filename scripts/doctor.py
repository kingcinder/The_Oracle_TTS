#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
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
    "libgobject-2.0.so.0": ["libglib2.0-0t64", "libglib2.0-0"],
    "libgthread-2.0.so.0": ["libglib2.0-0t64", "libglib2.0-0"],
    "libnss3.so": ["libnss3"],
    "libOpenGL.so.0": ["libopengl0"],
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
    except FileNotFoundError as exc:
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
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"
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
    result = _run_command(["dpkg-query", "-W", "-f=${Status}", package_name], timeout=10)
    return result["ok"] and result["stdout"].strip().endswith("installed")


@lru_cache(maxsize=None)
def _package_available(package_name: str) -> bool:
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
    venv_entrypoint = repo_root / ".venv" / "bin" / "the-oracle"
    wrapper_path = Path.home() / ".local" / "bin" / "the-oracle"
    path_entrypoint = shutil.which("the-oracle")
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

    fresh_shell = _run_command(
        ["bash", "-lc", "command -v the-oracle && the-oracle --help >/dev/null"],
        cwd=repo_root,
        timeout=30,
    )
    path_has_local_bin = str(Path.home() / ".local" / "bin") in os.environ.get("PATH", "").split(":")

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
        "fresh_shell_error": fresh_shell["stderr"].strip(),
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
    if plugin_path is None:
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

    ldd_result = _run_command(["ldd", str(plugin_path)], timeout=30)
    missing_libraries: list[str] = []
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
        "ok": bool(plugin_path.exists()) and not missing_libraries and bool(offscreen.get("ok")),
        "import_ok": True,
        "plugin_path": str(plugin_path),
        "plugin_exists": plugin_path.exists(),
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


def _build_next_steps(report: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    if not report["python"]["ok"]:
        steps.append("Install Python 3.12 with venv support: sudo apt install python3.12 python3.12-venv")

    runtime_packages: list[str] = []
    if not report["ffmpeg"]["ok"]:
        runtime_packages.append("ffmpeg")
    runtime_packages.extend(report["qt"]["suggested_packages"])
    if runtime_packages:
        unique_packages = " ".join(sorted(set(runtime_packages)))
        steps.append(f"Install the missing Linux Mint runtime packages: sudo apt install {unique_packages}")

    if not report["entrypoint"]["ok"]:
        steps.append("Re-run ./bootstrap_oracle_tts.sh to refresh the project venv and install the managed ~/.local/bin/the-oracle wrapper.")
        if not report["entrypoint"]["path_has_local_bin"]:
            steps.append('Add ~/.local/bin to PATH, open a fresh shell, and retry: export PATH="$HOME/.local/bin:$PATH"')

    if not report["chatterbox_import"]["ok"] or not report["chatterbox_init"]["ok"] or not report["perth"]["ok"]:
        steps.append("Re-run ./bootstrap_oracle_tts.sh with internet access so Chatterbox and Perth can be installed and cached on CPU.")

    if not report["deterministic_smoke"]["ok"]:
        steps.append("Inspect the deterministic smoke failure above, then retry with ./.venv/bin/python scripts/smoke_render.py.")

    if not report["real_engine_smoke"]["ok"]:
        steps.append("Real-engine smoke becomes ready after the Chatterbox import/init and Perth checks pass.")

    if not report["turbo"]["ok"]:
        steps.append("Optional turbo prefetch: ./.venv/bin/python scripts/download_models.py --variant turbo --device cpu")
    if report["voice_sources"]["primary_source"] != "seashells":
        steps.append("Add curated local reference clips to ./Seashells so the GUI stops defaulting to smoke/build fallback voices.")

    if not steps:
        steps.append("Ready to launch: ./run_oracle_tts.sh")
    return steps


def run(repo_root: Path, *, model_timeout: float, qt_timeout: float, skip_model_init: bool) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    _prepend_repo_src(repo_root)
    from the_oracle.voice_catalog import voice_catalog_audit

    chatterbox_probe = _chatterbox_probe(repo_root, timeout=model_timeout, skip_model_init=skip_model_init)
    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "platform": platform.platform(),
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
    }
    report["overall_ready"] = all(
        [
            report["python"]["ok"],
            report["entrypoint"]["ok"],
            report["chatterbox_import"]["ok"],
            report["perth"]["ok"],
            skip_model_init or report["chatterbox_init"]["ok"],
            report["qt"]["ok"],
            report["deterministic_smoke"]["ok"],
            report["real_engine_smoke"]["ok"],
        ]
    )
    report["next_steps"] = _build_next_steps(report)
    return report


def _print_human_report(report: dict[str, Any]) -> None:
    print(f"Repo root: {report['repo_root']}")
    print(f"Platform: {report['platform']}")
    print(f"{_status(report['python']['ok'])} Python: {report['python']['executable']} ({report['python']['version']})")

    ffmpeg_detail = report["ffmpeg"]["path"] or "ffmpeg not found on PATH"
    print(f"{_status(report['ffmpeg']['ok'])} Runtime tool `ffmpeg`: {ffmpeg_detail}")

    entrypoint = report["entrypoint"]
    entrypoint_detail = entrypoint["fresh_shell_path"] or entrypoint["path_entrypoint"] or entrypoint["venv_entrypoint"]
    if entrypoint["ok"]:
        print(f"{_status(True)} the-oracle entrypoint: {entrypoint_detail}")
    else:
        detail = entrypoint["fresh_shell_error"] or entrypoint["help_error"] or "the-oracle --help failed"
        print(f"{_status(False)} the-oracle entrypoint: {detail}")

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
        print(f"{_status(False)} Chatterbox CPU init: skipped")
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
        print(f"{_status(False)} Turbo readiness: {turbo['error']}")

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
        print(f"{_status(True)} Qt GUI prerequisites: xcb plugin ready at {qt['plugin_path']}")
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

    print("")
    print("Next steps:")
    for step in report["next_steps"]:
        print(f"- {step}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install and launch diagnostics for The Oracle on Linux Mint.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--model-timeout", type=float, default=1800.0)
    parser.add_argument("--qt-timeout", type=float, default=60.0)
    parser.add_argument("--skip-model-init", action="store_true")
    args = parser.parse_args(argv)

    report = run(
        args.repo_root,
        model_timeout=args.model_timeout,
        qt_timeout=args.qt_timeout,
        skip_model_init=args.skip_model_init,
    )
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        _print_human_report(report)
    return 0 if report["overall_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
