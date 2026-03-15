"""Command line interface for the Chatterbox-only DualVoice Studio app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dualvoice_studio.models.project import VoiceSettings
from dualvoice_studio.pipeline import DualVoicePipeline, RenderSettings, SpeakerSettings
from dualvoice_studio.project_manifest import build_saved_project, load_project_manifest, save_project_manifest
from dualvoice_studio.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dualvoice", description="Render two-speaker dialogue into FLAC with Chatterbox.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("gui", help="Launch the desktop GUI.")

    render = subparsers.add_parser("render", help="Batch render a dialogue file.")
    render.add_argument("--project", help="Load a saved project manifest.")
    render.add_argument("--save-project", dest="save_project", help="Write the current project manifest after preparation/render.")
    render.add_argument("--input", help="Path to .txt or .md dialogue file.")
    render.add_argument("--outdir", help="Output project directory.")
    render.add_argument("--speakerA-ref", dest="speaker_a_ref", help="Reference audio for Speaker A.")
    render.add_argument("--speakerB-ref", dest="speaker_b_ref", help="Reference audio for Speaker B.")
    render.add_argument("--model-variant", choices=["standard", "multilingual", "turbo"], default="standard")
    render.add_argument("--device-mode", choices=["cpu", "vulkan"], default="cpu")
    render.add_argument("--language", default="en", help="Language code for multilingual mode. Ignored for standard/turbo.")
    render.add_argument("--cfg-weight", type=float, default=0.5)
    render.add_argument("--exaggeration", type=float, default=0.5)
    render.add_argument("--temperature", type=float, default=0.8)
    render.add_argument("--repetition-penalty", type=float, default=1.2)
    render.add_argument("--min-p", type=float, default=0.05)
    render.add_argument("--top-p", type=float, default=1.0)
    render.add_argument("--correction-mode", choices=["conservative", "aggressive"], default="conservative")
    render.add_argument("--loudness", choices=["off", "light", "medium"], default="light")
    render.add_argument("--no-stems", action="store_true", help="Skip exporting stems into the project folder.")
    render.add_argument("--title", default="", help="Override exported title metadata.")
    return parser


def _voice_settings_from_args(args: argparse.Namespace) -> VoiceSettings:
    language = args.language if args.model_variant == "multilingual" else "en"
    return VoiceSettings(
        variant=args.model_variant,
        language=language,
        cfg_weight=args.cfg_weight,
        exaggeration=args.exaggeration,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        min_p=args.min_p,
        top_p=args.top_p,
    )


def handle_render(args: argparse.Namespace) -> int:
    pipeline = DualVoicePipeline()
    if args.project:
        saved = load_project_manifest(args.project)
        plan = saved.plan
        settings = saved.render_settings
        speakers = saved.speaker_settings
    else:
        missing = [name for name, value in {"--input": args.input, "--outdir": args.outdir, "--speakerA-ref": args.speaker_a_ref, "--speakerB-ref": args.speaker_b_ref}.items() if not value]
        if missing:
            raise SystemExit(f"render requires either --project or all direct render inputs. Missing: {', '.join(missing)}")
        settings = RenderSettings(
            correction_mode=args.correction_mode,
            model_variant=args.model_variant,
            language=args.language if args.model_variant == "multilingual" else "en",
            export_stems=not args.no_stems,
            loudness_preset=args.loudness,
            device_mode=args.device_mode,
            metadata={"title": args.title} if args.title else {},
        )
        voice_settings = _voice_settings_from_args(args)
        speakers = {
            "A": SpeakerSettings(reference_path=args.speaker_a_ref, voice_settings=voice_settings),
            "B": SpeakerSettings(reference_path=args.speaker_b_ref, voice_settings=voice_settings),
        }
        plan = pipeline.prepare_plan(args.input, args.outdir, speakers, settings)

    configure_logging(Path(plan.output_dir) / "logs" / "cli.log")
    output_path = pipeline.render(plan, settings)
    if args.save_project:
        save_project_manifest(args.save_project, build_saved_project(plan, settings, speakers))
    print(output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gui":
        from dualvoice_studio.app_gui import launch_gui

        configure_logging()
        launch_gui()
        return 0
    if args.command == "render":
        return handle_render(args)
    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
