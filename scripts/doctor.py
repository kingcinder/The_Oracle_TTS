#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import shutil
import socket
import sys
import tempfile
import urllib.request
from pathlib import Path

from dualvoice_studio.real_engine_smoke import real_engine_smoke_prerequisites


CHECKS = [
    ("PySide6", "PySide6"),
    ("markdown_it", "markdown-it-py"),
    ("symspellpy", "symspellpy"),
    ("mutagen", "mutagen"),
]

OPTIONAL_CHECKS = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("language_tool_python", "language-tool-python"),
    ("chatterbox", "chatterbox-tts"),
    ("perth", "resemble-perth"),
]


def import_status(module_name: str) -> tuple[bool, str | None]:
    try:
        module = __import__(module_name)
        return True, getattr(module, "__version__", None)
    except Exception as exc:
        return False, str(exc)


def flac_write_supported() -> bool:
    try:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "probe.flac"
            sf.write(path, [0.0, 0.0, 0.0], 24000, format="FLAC")
            return path.exists()
    except Exception:
        return False


def http_reachable(url: str, timeout: float) -> bool:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except Exception:
        return False


def chatterbox_constructor_check() -> dict[str, object]:
    try:
        import perth
        from chatterbox.tts import ChatterboxTTS

        return {
            "import_ok": True,
            "import_target": "from chatterbox.tts import ChatterboxTTS",
            "perth_implicit_watermarker": str(getattr(perth, "PerthImplicitWatermarker", None)),
            "module_name": getattr(ChatterboxTTS, "__module__", ""),
            "symbol_name": getattr(ChatterboxTTS, "__name__", ""),
            "constructor_symbol": str(ChatterboxTTS),
        }
    except Exception as exc:
        return {
            "import_ok": False,
            "import_target": "from chatterbox.tts import ChatterboxTTS",
            "error": str(exc),
        }


def install_report_status(repo_root: Path) -> dict[str, object]:
    report_path = repo_root / ".engine-setup" / "chatterbox_install_report.json"
    if not report_path.exists():
        return {"exists": False}
    try:
        return {"exists": True, "content": json.loads(report_path.read_text(encoding="utf-8"))}
    except Exception as exc:
        return {"exists": True, "error": str(exc)}


def run(timeout: float, repo_root: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "soundfile_flac": flac_write_supported(),
        "dns_huggingface": False,
        "imports": {},
        "optional_imports": {},
        "cuda": None,
        "model_endpoints": {},
        "chatterbox_check": chatterbox_constructor_check(),
        "install_report": install_report_status(repo_root),
        "real_engine_smoke": real_engine_smoke_prerequisites(repo_root / "build" / "real_engine_smoke"),
    }
    for module_name, label in CHECKS:
        ok, detail = import_status(module_name)
        result["imports"][label] = {"ok": ok, "detail": detail}
    for module_name, label in OPTIONAL_CHECKS:
        ok, detail = import_status(module_name)
        result["optional_imports"][label] = {"ok": ok, "detail": detail}

    try:
        socket.gethostbyname("huggingface.co")
        result["dns_huggingface"] = True
    except Exception:
        result["dns_huggingface"] = False

    ok, _ = import_status("torch")
    if ok:
        import torch

        result["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }

    endpoints = {
        "huggingface": "https://huggingface.co",
        "chatterbox_model": "https://huggingface.co/ResembleAI/chatterbox",
        "go_emotions": "https://huggingface.co/SamLowe/roberta-base-go_emotions",
    }
    result["model_endpoints"] = {name: http_reachable(url, timeout) for name, url in endpoints.items()}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Environment diagnostics for Chatterbox-only DualVoice Studio.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    report = run(timeout=args.timeout, repo_root=args.repo_root.resolve())
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Python: {report['python']}")
        print(f"Platform: {report['platform']}")
        print(f"ffmpeg: {'ok' if report['ffmpeg'] else 'missing'}")
        print(f"soundfile FLAC: {'ok' if report['soundfile_flac'] else 'missing'}")
        print("Core imports:")
        for name, payload in report["imports"].items():
            print(f"  {name}: {'ok' if payload['ok'] else 'missing'}")
        print("Optional imports:")
        for name, payload in report["optional_imports"].items():
            print(f"  {name}: {'ok' if payload['ok'] else 'missing'}")
        print(f"Hugging Face DNS: {'ok' if report['dns_huggingface'] else 'failed'}")
        print(f"Chatterbox constructor check: {report['chatterbox_check']}")
        print(f"Install report: {report['install_report']}")
        print(f"Real-engine smoke prerequisites: {report['real_engine_smoke']}")
        if report["cuda"] is not None:
            print(f"CUDA: {report['cuda']}")
        print("Model endpoints:")
        for name, ok in report["model_endpoints"].items():
            print(f"  {name}: {'ok' if ok else 'failed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
