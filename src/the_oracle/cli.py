"""Command line interface for the Chatterbox-only The Oracle app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import OraclePipeline, RenderSettings, SpeakerSettings
from the_oracle.project_manifest import build_saved_project, load_project_manifest, save_project_manifest
from the_oracle.utils.logging import configure_logging


def _nonnegative_int(value: str) -> int:
    """argparse type: a non-negative integer (Vulkan device indexes start at 0)."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def _positive_int(value: str) -> int:
    """argparse type: a positive integer (thread counts start at 1)."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="the-oracle",
        description="The Oracle renders two-speaker dialogue into FLAC with Chatterbox.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("gui", help="Launch the desktop GUI.")

    setup_vulkan = subparsers.add_parser(
        "setup-vulkan",
        help="One-shot automatic setup for the Vulkan (GPU) backend: build audiocpp_cli and download the Chatterbox model if missing.",
    )

    render = subparsers.add_parser("render", help="Batch render a dialogue file.")
    render.add_argument("--project", help="Load a saved project manifest.")
    render.add_argument("--save-project", dest="save_project", help="Write the current project manifest after preparation/render.")
    render.add_argument("--input", help="Path to .txt or .md dialogue file.")
    render.add_argument("--outdir", help="Output project directory.")
    render.add_argument("--speakerA-ref", dest="speaker_a_ref", help="Reference audio for Speaker A.")
    render.add_argument("--speakerB-ref", dest="speaker_b_ref", help="Reference audio for Speaker B.")
    render.add_argument("--model-variant", choices=["standard", "multilingual", "turbo"], default="standard")
    render.add_argument("--device-mode", choices=["cpu", "vulkan"], default="cpu")
    render.add_argument("--no-audio-cpp-setup", action="store_true", help="Skip the automatic audio.cpp build/model download before a Vulkan render; fail fast instead.")
    render.add_argument("--inference-backend",
        choices=["pytorch", "vulkan"],
        default="pytorch",
        help="Inference backend: 'pytorch' (default, Chatterbox in-process) or 'vulkan' "
        "(opt-in, shells out to audio.cpp built with the Vulkan backend). "
        "Ignored when --project is used (the saved manifest governs).",
    )
    render.add_argument(
        "--audio-cpp-device",
        type=_nonnegative_int,
        default=None,
        metavar="N",
        help="Vulkan device index passed to audio.cpp as --device <N> (constructor arg wins "
        "over ORACLE_AUDIOCPP_DEVICE). Requires --inference-backend vulkan. Ignored when "
        "--project is used (the saved manifest governs).",
    )
    render.add_argument(
        "--audio-cpp-threads",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Thread count passed to audio.cpp as --threads <N> (constructor arg wins over "
        "ORACLE_AUDIOCPP_THREADS). Requires --inference-backend vulkan. Ignored when "
        "--project is used (the saved manifest governs).",
    )
    render.add_argument(
        "--audio-cpp-timeout",
        type=_positive_int,
        default=None,
        metavar="SECONDS",
        help="Per-synthesis timeout in seconds passed to audio.cpp (constructor arg wins "
        "over ORACLE_AUDIOCPP_TIMEOUT, default 600). Requires --inference-backend vulkan. "
        "Ignored when --project is used (the saved manifest governs).",
    )
    render.add_argument(
        "--audio-cpp-max-batch",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Maximum cache-missing stems per audio.cpp --request-sequence subprocess "
        "(constructor arg wins over ORACLE_AUDIOCPP_MAX_BATCH, default 32). Requires "
        "--inference-backend vulkan. Ignored when --project is used (the saved "
        "manifest governs).",
    )
    render.add_argument("--language", default="en", help="Language code for multilingual mode. Ignored for standard/turbo.")
    render.add_argument("--cfg-weight", type=float, default=0.5)
    render.add_argument("--exaggeration", type=float, default=0.5)
    render.add_argument("--temperature", type=float, default=0.8)
    render.add_argument("--repetition-penalty", type=float, default=1.2)
    render.add_argument("--min-p", type=float, default=0.05)
    render.add_argument("--top-p", type=float, default=1.0)
    render.add_argument("--target-wpm", type=float, help="Optional target words-per-minute pacing hint.")
    render.add_argument(
        "--correction-mode",
        choices=["aggressive", "moderate", "mild", "off", "conservative"],
        default="moderate",
        help="Text repair strength. 'conservative' is kept as an alias for 'moderate'.",
    )
    render.add_argument("--loudness", choices=["off", "light", "medium"], default="light")
    render.add_argument("--no-stems", action="store_true", help="Skip exporting stems into the project folder.")
    render.add_argument("--title", default="", help="Override exported title metadata.")
    render.add_argument("--srt", action="store_true", help="Also write an SRT subtitle file next to the rendered FLAC.")
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
    pipeline = OraclePipeline()
    if args.project:
        saved = load_project_manifest(args.project)
        plan = saved.plan
        settings = saved.render_settings
        speakers = saved.speaker_settings
    else:
        missing = [name for name, value in {"--input": args.input, "--outdir": args.outdir, "--speakerA-ref": args.speaker_a_ref, "--speakerB-ref": args.speaker_b_ref}.items() if not value]
        if missing:
            raise SystemExit(f"render requires either --project or all direct render inputs. Missing: {', '.join(missing)}")
        if args.inference_backend != "vulkan" and (
            args.audio_cpp_device is not None
            or args.audio_cpp_threads is not None
            or args.audio_cpp_timeout is not None
            or args.audio_cpp_max_batch is not None
        ):
            raise SystemExit(
                "--audio-cpp-device, --audio-cpp-threads, --audio-cpp-timeout, and "
                "--audio-cpp-max-batch are Vulkan-backend knobs and require "
                "--inference-backend vulkan."
            )
        if args.inference_backend == "vulkan" and args.model_variant == "turbo":
            raise SystemExit(
                "--inference-backend vulkan does not support the turbo variant. "
                "Use --model-variant standard (or multilingual) with Vulkan, or "
                "--inference-backend pytorch for turbo."
            )
        settings = RenderSettings(
            correction_mode=args.correction_mode,
            model_variant=args.model_variant,
            language=args.language if args.model_variant == "multilingual" else "en",
            export_stems=not args.no_stems,
            loudness_preset=args.loudness,
            device_mode=args.device_mode,
            inference_backend=args.inference_backend,
            audio_cpp_device=args.audio_cpp_device,
            audio_cpp_threads=args.audio_cpp_threads,
            audio_cpp_timeout=args.audio_cpp_timeout,
            audio_cpp_max_batch=args.audio_cpp_max_batch,
            target_wpm=args.target_wpm,
            metadata={"title": args.title} if args.title else {},
        )
        voice_settings = _voice_settings_from_args(args)
        speakers = {
            "A": SpeakerSettings(reference_path=args.speaker_a_ref, voice_settings=voice_settings),
            "B": SpeakerSettings(reference_path=args.speaker_b_ref, voice_settings=voice_settings),
        }
        plan = pipeline.prepare_plan(args.input, args.outdir, speakers, settings)

    if settings.inference_backend == "vulkan" and not args.no_audio_cpp_setup:
        # Automatic CPU→GPU switch: build audiocpp_cli and/or download the
        # Chatterbox model when missing, then set the env vars for this process
        # so the existing engine path just works. --no-audio-cpp-setup keeps the
        # old fail-fast behavior (the engine's ensure_model_ready still guards).
        from the_oracle.vulkan_setup import run_vulkan_setup

        def _progress(line: str) -> None:
            print(f"  [vulkan setup] {line}", file=sys.stderr)

        print("Vulkan backend selected: checking/installing prerequisites...", file=sys.stderr)
        result = run_vulkan_setup(progress=_progress)
        if not result.ok:
            raise SystemExit(f"Vulkan backend auto-setup failed: {result.error}")
        for msg in result.messages:
            print(f"  [vulkan setup] {msg}", file=sys.stderr)

    configure_logging(Path(plan.output_dir) / "logs" / "cli.log")
    output_path = pipeline.render(plan, settings)
    if args.srt:
        from the_oracle.audio.export_srt import write_srt

        write_srt(Path(output_path).with_suffix(".srt"), plan.utterances)
    if args.save_project:
        save_project_manifest(args.save_project, build_saved_project(plan, settings, speakers))
    print(output_path)
    return 0


def handle_setup_vulkan() -> int:
    """One-shot automatic setup for the Vulkan (GPU) backend.

    Builds audiocpp_cli and downloads the Chatterbox model when missing, then
    prints the ORACLE_AUDIOCPP_CLI / ORACLE_AUDIOCPP_MODEL values the session
    now uses (env vars are set in-process). Exit code 0 on success, 1 with a
    clear error otherwise.
    """
    from the_oracle.vulkan_setup import run_vulkan_setup

    def _progress(line: str) -> None:
        print(f"  [vulkan setup] {line}", file=sys.stderr)

    print("Vulkan backend setup: checking prerequisites...", file=sys.stderr)
    result = run_vulkan_setup(progress=_progress)
    for msg in result.messages:
        print(f"  [vulkan setup] {msg}", file=sys.stderr)
    if not result.ok:
        print(f"Vulkan backend setup failed: {result.error}", file=sys.stderr)
        return 1
    print()
    print("Vulkan backend ready:")
    if result.binary:
        print(f'    export ORACLE_AUDIOCPP_CLI="{result.binary}"')
    if result.model:
        print(f'    export ORACLE_AUDIOCPP_MODEL="{result.model}"')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gui":
        from the_oracle.app_gui import launch_gui

        configure_logging()
        launch_gui()
        return 0
    if args.command == "render":
        return handle_render(args)
    if args.command == "setup-vulkan":
        return handle_setup_vulkan()
    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
