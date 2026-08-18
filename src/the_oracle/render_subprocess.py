"""Child-process entry point for native Chatterbox renders started by the GUI.

This module intentionally imports no Qt code. PyTorch/Perth model initialization
can segfault after Qt Multimedia has been initialized in the parent process, so
the desktop GUI delegates the complete render to this clean interpreter.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from the_oracle.models.project import RenderPlan, Utterance, VoiceProfile
from the_oracle.pipeline import OraclePipeline, RenderProgress, RenderSettings
from the_oracle.utils.logging import configure_logging


def _read_job(path: Path) -> tuple[RenderPlan, RenderSettings, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan = RenderPlan.from_dict(payload["plan"])
    settings = RenderSettings(**payload["settings"])
    return plan, settings, payload.get("render_click_wall")


def _emit_progress(progress: RenderProgress) -> None:
    # A single machine-readable line makes progress robust against ordinary
    # model/logging output appearing on the same captured stream.
    print(f"ORACLE_RENDER_PROGRESS {json.dumps(asdict(progress), ensure_ascii=True)}", flush=True)


def _read_preview_job(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_preview_job(job_path: Path, result_path: Path) -> int:
    """Run a single-utterance preview in this clean interpreter.

    Mirrors :func:`run_job` so the Qt GUI process never has to initialize
    Chatterbox/PyTorch itself (the same native SIGSEGV the render child avoids).
    """
    try:
        payload = _read_preview_job(job_path)
        utterance = Utterance.from_dict(payload["utterance"])
        profile = VoiceProfile.from_dict(payload["profile"])
        model_variant = str(payload["model_variant"])
        device_mode = str(payload.get("device_mode") or "cpu")
        inference_backend = str(payload.get("inference_backend") or "pytorch")
        configure_logging()
        pipeline = OraclePipeline(
            use_transformers=False,
            use_language_tool=False,
            use_punctuation_model=False,
        )
        preview_path = pipeline.render_preview(
            utterance,
            profile,
            model_variant,
            device_mode=device_mode,
            inference_backend=inference_backend,
            audio_cpp_device=payload.get("audio_cpp_device"),
            audio_cpp_threads=payload.get("audio_cpp_threads"),
            audio_cpp_timeout=payload.get("audio_cpp_timeout"),
            audio_cpp_max_batch=payload.get("audio_cpp_max_batch"),
            progress_callback=_emit_progress,
        )
        result_path.write_text(
            json.dumps({"ok": True, "preview_path": str(preview_path)}, ensure_ascii=True),
            encoding="utf-8",
        )
        return 0
    except BaseException as exc:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        print(f"ORACLE_RENDER_ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def run_job(job_path: Path, result_path: Path) -> int:
    plan: RenderPlan | None = None
    try:
        plan, settings, render_click_wall = _read_job(job_path)
        configure_logging(Path(plan.output_dir) / "logs" / "render_child.log")
        pipeline = OraclePipeline(
            use_transformers=False,
            use_language_tool=False,
            use_punctuation_model=False,
        )
        output_path = pipeline.render(
            plan,
            settings,
            progress_callback=_emit_progress,
            render_click_wall=render_click_wall,
            force_sequential=True,
        )
        result_path.write_text(
            json.dumps(
                {"ok": True, "output_path": str(output_path), "plan": plan.to_dict()},
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        return 0
    except BaseException as exc:
        # Catch BaseException here because the parent needs a result file for
        # every ordinary Python failure. A native SIGSEGV cannot reach this
        # handler, but the parent still reports that signal accurately.
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "plan": plan.to_dict() if plan is not None else None,
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        print(f"ORACLE_RENDER_ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="the_oracle.render_subprocess")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="Run a single-utterance preview instead of a full render.")
    args = parser.parse_args(argv)
    if args.preview:
        return run_preview_job(args.job, args.result)
    return run_job(args.job, args.result)


if __name__ == "__main__":
    raise SystemExit(main())
