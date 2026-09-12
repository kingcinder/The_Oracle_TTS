"""Command line interface for the Chatterbox-only The Oracle app."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from the_oracle import __version__
from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import NoAudioToAssembleError, OraclePipeline, PartialRenderError, RenderSettings, SpeakerSettings
from the_oracle.project_manifest import build_saved_project, load_project_manifest, save_project_manifest
from the_oracle.tts_engines.vulkan_backend import AudioCppUnavailableError, RDNA1VulkanError
from the_oracle.utils.logging import configure_logging
from the_oracle.voice_catalog import default_voice_choices


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
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the installed version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("gui", help="Launch the desktop GUI.")

    setup_vulkan = subparsers.add_parser(
        "setup-vulkan",
        help="One-shot automatic setup for the Vulkan (GPU) backend: build audiocpp_cli and download the Chatterbox model if missing.",
    )

    check_input = subparsers.add_parser(
        "check-input",
        help="Lint a dialogue file's formatting without rendering: report speaker turns the engine would misread and, with --fix, correct them in place.",
    )
    check_input.add_argument("file", help="Path to the .txt or .md dialogue file to check.")
    check_input.add_argument(
        "--fix",
        action="store_true",
        help="Correct fixable issues in place (timestamped backup kept) instead of only reporting them.",
    )
    check_input.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as a single JSON document on stdout (issues list, "
        "fixable/warning counts, speaker-voice suggestions) instead of "
        "human-readable text, so CI pipelines can parse it.",
    )

    render = subparsers.add_parser("render", help="Batch render a dialogue file.")
    render.add_argument("--project", help="Load a saved project manifest.")
    render.add_argument("--save-project", dest="save_project", help="Write the current project manifest after preparation/render.")
    render.add_argument("--input", help="Path to .txt or .md dialogue file.")
    render.add_argument("--outdir", help="Output project directory.")
    render.add_argument("--speakerA-ref", dest="speaker_a_ref", help="Reference audio for Speaker A.")
    render.add_argument("--speakerB-ref", dest="speaker_b_ref", help="Reference audio for Speaker B.")
    render.add_argument(
        "--speaker-ref",
        dest="speaker_refs",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Reference audio for an additional character voice, e.g. --speaker-ref C=/path/ref.wav. "
        "Repeatable for up to 24 voices. Characters without a configured reference "
        "fall back to the first provided voice.",
    )
    render.add_argument(
        "--monologue",
        action="store_true",
        help="Render the whole input as a single narrator voice (Speaker A), ignoring per-line attribution.",
    )
    render.add_argument("--model-variant", choices=["standard", "multilingual", "turbo"], default="standard")
    render.add_argument("--device-mode", choices=["cpu", "cuda", "vulkan"], default="cpu", help="PyTorch device: cpu or CUDA; Vulkan is retained for compatibility and uses the audio.cpp backend.")
    render.add_argument("--cuda-device", type=_nonnegative_int, default=None, metavar="N", help="CUDA device index for PyTorch inference (default: CUDA's default device). Requires --device-mode cuda.")
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
    render.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic sampling seed: renders with identical inputs produce "
        "byte-identical audio across runs (PyTorch: torch.manual_seed; "
        "Vulkan: audio.cpp --seed).",
    )
    render.add_argument("--title", default="", help="Override exported title metadata.")
    render.add_argument("--srt", action="store_true", help="Also write an SRT subtitle file next to the rendered FLAC.")
    render.add_argument(
        "--fix-input",
        action="store_true",
        help="Check the input file with the ingestion transformer and automatically correct "
        "fixable formatting issues in place before rendering (dash/pipe separators, "
        "bracketed labels, bulleted turns, orphan labels, chat-export timestamps, "
        "non-UTF-8 encodings). A timestamped backup of the original is kept next to "
        "the file. Without this flag the check still runs and reports issues on "
        "stderr, but never modifies the file. Requires --input.",
    )
    render.add_argument(
        "--fix-input-interactive",
        action="store_true",
        help="Like --fix-input, but review first: show the formatting summary and a "
        "rule-labeled diff of the exact rewrite, then prompt yes/no before writing "
        "(a timestamped backup is kept). Answering no aborts the render with the "
        "file untouched. Requires a terminal; refuses silently in non-interactive "
        "runs (pipes/CI) so automation never blocks on a prompt. Requires --input.",
    )
    render.add_argument(
        "--check-input-json",
        action="store_true",
        help="Emit the input-formatting report as a single JSON document on stdout "
        "(issues list, fixable/warning counts, speaker-voice suggestions, and with "
        "--fix-input the applied fix count and backup path) instead of "
        "human-readable stderr text. Requires --input.",
    )
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


def _subtitle_script_target(file_path: Path) -> Path:
    """The sibling script path a subtitle conversion writes to."""
    suffix = file_path.suffix.lower()
    tail = ".srt.txt" if suffix == ".srt" else ".vtt.txt" if suffix == ".vtt" else ".txt"
    # with_suffix cannot build compound names like ".vtt.txt" (it replaces
    # the whole suffix), so the name is assembled from the stem.
    return file_path.with_name(file_path.stem + tail)


def _srt_script_if_converted(input_path: str) -> str:
    """Return the converted subtitle script when a fix produced one."""
    file_path = Path(input_path)
    if file_path.suffix.lower() in (".srt", ".vtt"):
        script = _subtitle_script_target(file_path)
        if script.is_file():
            return str(script)
    return input_path


def _maybe_convert_srt(input_path: str) -> str:
    """Convert a subtitle input (``.srt``/``.vtt``) to a script, returning the input to use.

    Subtitle files cannot be ingested directly (every cue degrades into
    narration), so a valid SubRip or WebVTT input is converted to a sibling
    ``<name>.srt.txt``/``<name>.vtt.txt`` script and that script is rendered
    instead — the subtitle file itself is never modified. Anything that is
    not a valid subtitle file (including a missing or unreadable file) is
    returned unchanged so the ordinary error paths report it.
    """
    from the_oracle.srt_ingest import convert_srt_file, looks_like_srt

    file_path = Path(input_path)
    if file_path.suffix.lower() not in (".srt", ".vtt") or not file_path.is_file():
        return input_path
    try:
        text = file_path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError:
        text = file_path.read_bytes().decode("cp1252", errors="replace")
    except OSError:
        return input_path
    if not looks_like_srt(text):
        return input_path
    try:
        script_path, cue_count, speaker_count = convert_srt_file(file_path)
    except FileExistsError:
        # A previous conversion already produced the script; reuse it.
        script_path = _subtitle_script_target(file_path)
        print(
            f"SRT input: reusing previously converted script {script_path}",
            file=sys.stderr,
        )
        _print_speaker_ref_hints(str(script_path))
        return str(script_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"SRT conversion failed: {exc}") from exc
    print(
        f"SRT input: converted {cue_count} cue(s) from {file_path.name} into "
        f"{script_path.name} ({speaker_count} speaker(s)); rendering the script.",
        file=sys.stderr,
    )
    # The converted script names a cast; tell the user which voice flags to
    # provide so the render can attribute every speaker.
    _print_speaker_ref_hints(str(script_path))
    return str(script_path)


def _input_issue_json(issue, *, fixable: bool) -> dict:
    """Serialize one transformer issue for the machine-readable report."""
    return {
        "line": issue.line_number,
        "snippet": issue.snippet,
        "fixable": fixable,
        "description": issue.fix_description if fixable else issue.description,
    }


def _input_json_document(
    input_path: str,
    analysis,
    *,
    error: str | None = None,
    fixed_count: int | None = None,
    backup: str | None = None,
    speaker_refs: list[dict] | None = None,
    rejected: list[str] | None = None,
) -> dict:
    """Build the input-formatting JSON report (one document per run).

    ``speaker_refs``/``rejected_labels`` are always present so consumers can
    rely on a stable schema; ``fixed_count``/``backup`` appear only when a
    fix was applied.
    """
    document: dict = {"file": input_path}
    if error is not None:
        document["error"] = error
        document["fixable_count"] = 0
        document["warning_count"] = 0
        document["issues"] = []
    else:
        document["fixable_count"] = len(analysis.fixable_issues)
        document["warning_count"] = len(analysis.warning_issues)
        document["issues"] = [
            _input_issue_json(issue, fixable=True) for issue in analysis.fixable_issues
        ] + [
            _input_issue_json(issue, fixable=False) for issue in analysis.warning_issues
        ]
    if fixed_count is not None:
        document["fixed_count"] = fixed_count
    if backup is not None:
        document["backup"] = backup
    document["speaker_refs"] = speaker_refs or []
    document["rejected_labels"] = rejected or []
    return document


def _speaker_ref_report(file_path: Path, post_fix_text: str | None) -> tuple[list[dict], list[str]]:
    """Compute --speaker-ref suggestions for a script's cast.

    Returns ``(refs, rejected)``: one dict per speaker (``speaker``,
    ``voice_key``, ``flag``, ``new``) plus distinct labels that fail speaker
    validation entirely (no reference audio can attribute those). When
    ``post_fix_text`` is None the file is read and its post-transform text
    is computed on the fly, so a suggested cast always matches what a render
    would see after a fix.
    """
    from the_oracle.ingest_transformer import (
        _decode_best_effort,
        rejected_labels,
        suggest_speaker_refs,
        transform_text,
    )

    if post_fix_text is None:
        try:
            raw = file_path.read_bytes()
        except OSError:
            return [], []
        post_fix_text = transform_text(_decode_best_effort(raw))[0]
    refs = [
        {
            "speaker": s.speaker,
            "voice_key": s.voice_key,
            "flag": s.flag,
            "new": s.is_new,
        }
        for s in suggest_speaker_refs(post_fix_text)
    ]
    return refs, rejected_labels(post_fix_text)


def _check_input_formatting(input_path: str, fix: bool, json_output: bool = False) -> None:
    """Run the ingestion transformer over the CLI render's input file.

    Detected issues are always reported: human-readable on stderr, or as a
    JSON document on stdout when ``json_output`` is set (so pipelines can
    parse it; stdout keeps the report separate from render logs on stderr).
    With ``fix=True`` the fixable issues are corrected in place (timestamped
    backup kept) before the render proceeds. Never raises for
    missing/unreadable files: those are reported (as JSON in json mode) and
    ``prepare_plan`` reports them with its own clear error.
    """
    import json

    from the_oracle.ingest_transformer import analyze_input_file, fix_input_file

    def _issue_json(issue, *, fixable: bool) -> dict:
        return _input_issue_json(issue, fixable=fixable)

    def _report(
        analysis,
        *,
        error: str | None = None,
        fixed_count: int | None = None,
        backup: str | None = None,
        speaker_refs: list[dict] | None = None,
        rejected: list[str] | None = None,
    ) -> None:
        document = _input_json_document(
            input_path,
            analysis,
            error=error,
            fixed_count=fixed_count,
            backup=backup,
            speaker_refs=speaker_refs,
            rejected=rejected,
        )
        print(json.dumps(document, ensure_ascii=False))

    if json_output:
        if not Path(input_path).is_file():
            _report(None, error="file not found")
            return
        try:
            analysis = analyze_input_file(input_path)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            _report(None, error=f"unreadable: {exc}")
            return
        # Speaker-voice suggestions come from the post-fix text so a cast
        # member only visible after a transform is still suggested.
        refs, rejected = _speaker_ref_report(Path(input_path), None)
        if not analysis.has_issues or not fix or not analysis.fixable_issues:
            # Exactly one JSON document per run, so a consumer can always
            # parse stdout as a single object.
            _report(analysis, speaker_refs=refs, rejected=rejected)
            return
        try:
            _text, fix_count, backup_path = fix_input_file(input_path)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--fix-input failed: {exc}") from exc
        # One document: the pre-fix issue list plus what was applied, so
        # consumers see both what was wrong and that the file changed on disk.
        _report(analysis, fixed_count=fix_count, backup=backup_path, speaker_refs=refs, rejected=rejected)
        return

    try:
        analysis = analyze_input_file(input_path)
    except (OSError, ValueError, UnicodeDecodeError):
        return
    if not analysis.has_issues:
        _print_speaker_ref_hints(input_path)
        return
    fixable = analysis.fixable_issues
    warnings = analysis.warning_issues
    print(
        f"Input formatting: {len(fixable)} fixable issue(s), "
        f"{len(warnings)} warning(s) in {input_path}",
        file=sys.stderr,
    )
    for issue in fixable[:5]:
        where = f"line {issue.line_number}" if issue.line_number else "file"
        print(f"  - {where}: {issue.fix_description}", file=sys.stderr)
    for issue in warnings[:5]:
        where = f"line {issue.line_number}" if issue.line_number else "file"
        print(f"  - {where}: {issue.description}", file=sys.stderr)
    if len(fixable) > 5 or len(warnings) > 5:
        print("  - ... (see the GUI's Preview Fixed Text dialog for the full list)", file=sys.stderr)
    # The post-fix cast can be suggested even before the user fixes the
    # file — the suggestions come from the transformed text.
    _print_speaker_ref_hints(input_path)
    if not fix:
        if fixable:
            print(
                "Re-run with --fix-input to correct the fixable issues automatically.",
                file=sys.stderr,
            )
        return
    if not fixable:
        print("--fix-input: no fixable issues; the file was left unchanged.", file=sys.stderr)
        return
    try:
        _text, fix_count, backup_path = fix_input_file(input_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--fix-input failed: {exc}") from exc
    print(f"--fix-input: corrected {fix_count} issue(s) in {input_path}", file=sys.stderr)
    if backup_path:
        print(f"    backup: {backup_path}", file=sys.stderr)
    _print_speaker_ref_hints(input_path)


def _check_input_formatting_interactive(input_path: str, *, prompt=input) -> bool:
    """Interactive review-before-fix: the terminal version of the GUI popup.

    Shows the issue summary and a rule-labeled diff of the exact rewrite,
    then asks before writing. Returns True when the render should proceed
    (no issues, or fix applied); False when the user declined the fix. The
    ``prompt`` parameter is injectable so tests can script the answer.
    """
    from the_oracle.ingest_transformer import (
        _decode_best_effort,
        analyze_input_file,
        fix_input_file,
        labeled_fixed_diff,
        transform_text_detailed,
    )

    file_path = Path(input_path)
    try:
        analysis = analyze_input_file(file_path)
    except (OSError, ValueError, UnicodeDecodeError):
        return True  # ordinary error paths report this
    if not analysis.has_issues:
        return True

    fixable = analysis.fixable_issues
    warnings = analysis.warning_issues
    print(f"Input formatting review for {file_path}", file=sys.stderr)
    if fixable:
        print(f"  {len(fixable)} fixable issue(s) can be corrected automatically:", file=sys.stderr)
        for issue in fixable[:8]:
            where = f"line {issue.line_number}" if issue.line_number else "file"
            print(f"    - {where}: {issue.fix_description}", file=sys.stderr)
        if len(fixable) > 8:
            print(f"    ... and {len(fixable) - 8} more", file=sys.stderr)
    if warnings:
        print(f"  {len(warnings)} warning(s) that cannot be fixed automatically:", file=sys.stderr)
        for issue in warnings[:8]:
            where = f"line {issue.line_number}" if issue.line_number else "file"
            print(f"    ! {where}: {issue.description}", file=sys.stderr)

    if not fixable:
        # Warnings only: nothing to review or write.
        return True

    try:
        raw = file_path.read_bytes()
    except OSError:
        return True
    original = (
        analysis.encoding_fixed_text
        if analysis.encoding_fixed_text is not None
        else _decode_best_effort(raw)
    )
    fixed_text, line_fixes = transform_text_detailed(original)
    print(file=sys.stderr)
    print(labeled_fixed_diff(original, fixed_text, line_fixes), file=sys.stderr)

    try:
        answer = prompt("Apply these fixes before rendering? [y/N]: ")
    except EOFError:
        answer = None
    if not answer or answer.strip().lower() not in ("y", "yes"):
        print("Fix declined: the input file was left unchanged; aborting render.", file=sys.stderr)
        return False
    try:
        _text, fix_count, backup_path = fix_input_file(file_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--fix-input-interactive failed: {exc}") from exc
    print(f"--fix-input-interactive: corrected {fix_count} issue(s) in {input_path}", file=sys.stderr)
    if backup_path:
        print(f"    backup: {backup_path}", file=sys.stderr)
    return True


def _print_speaker_ref_hints(input_path: str) -> None:
    """Print the exact --speaker-ref flags for the script's cast (stderr)."""
    refs, rejected = _speaker_ref_report(Path(input_path), None)
    if refs:
        print("Speaker voices to provide (in first-appearance order):", file=sys.stderr)
        for ref in refs:
            marker = "add" if ref["new"] else "use"
            print(f"  - {ref['speaker']} -> voice {ref['voice_key']}: {marker} {ref['flag']}", file=sys.stderr)
    for label in rejected:
        print(
            f"  ! '{label}' is not accepted as a speaker label; no reference audio can "
            "attribute it — edit the label in the file (e.g. to a name or 'Speaker X').",
            file=sys.stderr,
        )


def handle_check_input(args: argparse.Namespace) -> int:
    """Lint a dialogue file's formatting without loading the render pipeline.

    Exit code 0 = no issues (or fixed with --fix); 1 = issues found (or
    warnings remain after --fix); 2 = file could not be read. Nothing is
    written unless --fix is given, and --fix never rewrites warnings.
    """
    from the_oracle.ingest_transformer import analyze_input_file, fix_input_file

    file_path = Path(args.file)
    json_mode = bool(getattr(args, "json", False))
    if not file_path.is_file():
        if json_mode:
            print(json.dumps(_input_json_document(args.file, None, error="file not found"), ensure_ascii=False))
        else:
            print(f"check-input: file not found: {file_path}", file=sys.stderr)
        return 2
    try:
        analysis = analyze_input_file(file_path)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        if json_mode:
            print(json.dumps(_input_json_document(args.file, None, error=f"unreadable: {exc}"), ensure_ascii=False))
        else:
            print(f"check-input: the file could not be read: {exc}", file=sys.stderr)
        return 2

    if json_mode:
        # Exactly one JSON document per run, on stdout; nothing else is
        # printed in json mode so pipelines can parse stdout cleanly.
        refs, rejected = _speaker_ref_report(file_path, None)
        fixed_count: int | None = None
        backup: str | None = None
        if args.fix and analysis.fixable_issues:
            try:
                _written, fixed_count, backup = fix_input_file(file_path)
            except (OSError, ValueError) as exc:
                print(
                    json.dumps(_input_json_document(args.file, None, error=f"fix failed: {exc}"), ensure_ascii=False)
                )
                return 2
            # Re-analyze so the document describes the file's current state;
            # the exit code and the document then always agree.
            try:
                analysis = analyze_input_file(file_path)
            except (OSError, ValueError, UnicodeDecodeError):
                return 0
        document = _input_json_document(
            str(file_path),
            analysis,
            fixed_count=fixed_count,
            backup=backup,
            speaker_refs=refs,
            rejected=rejected,
        )
        print(json.dumps(document, ensure_ascii=False))
        return 0 if not analysis.has_issues else 1

    if not analysis.has_issues:
        print(f"{file_path}: no formatting issues found.")
        _print_speaker_ref_hints(str(file_path))
        return 0

    fixable = analysis.fixable_issues
    warnings = analysis.warning_issues

    if args.fix and fixable:
        try:
            _text, fix_count, backup_path = fix_input_file(file_path)
        except (OSError, ValueError) as exc:
            print(f"check-input: the file could not be corrected: {exc}", file=sys.stderr)
            return 2
        print(f"check-input: corrected {fix_count} issue(s) in {file_path}")
        if backup_path:
            print(f"    backup: {backup_path}")
        # Re-analyze the corrected file so the exit code reflects what is
        # left, not what was there.
        try:
            analysis = analyze_input_file(file_path)
        except (OSError, ValueError, UnicodeDecodeError):
            return 0
        if not analysis.has_issues:
            print(f"{file_path}: no formatting issues remain.")
            _print_speaker_ref_hints(str(file_path))
            return 0
        fixable = analysis.fixable_issues
        warnings = analysis.warning_issues

    if fixable:
        print(f"{file_path}: {len(fixable)} fixable formatting issue(s), {len(warnings)} warning(s):")
    else:
        print(f"{file_path}: {len(warnings)} formatting warning(s):")
    for issue in fixable[:10]:
        where = f"line {issue.line_number}" if issue.line_number else "file"
        print(f"  - {where}: {issue.fix_description}")
    for issue in warnings[:10]:
        where = f"line {issue.line_number}" if issue.line_number else "file"
        print(f"  ! {where}: {issue.description}")
    if len(fixable) > 10 or len(warnings) > 10:
        print("  - ... (truncated)")
    if fixable:
        hint = "remain (never rewritten automatically)" if args.fix else "corrected automatically"
        print(f"Re-run with --fix to have the fixable issues {hint}.")
    _print_speaker_ref_hints(str(file_path))
    return 1


def handle_render(args: argparse.Namespace) -> int:
    if args.project:
        saved = load_project_manifest(args.project)
        plan = saved.plan
        settings = saved.render_settings
        speakers = saved.speaker_settings
    else:
        # Validate inputs before constructing the pipeline: OraclePipeline()
        # eagerly spawns the LanguageTool download (hundreds of MB) and waits
        # on it, so a missing-flag mistake should fail fast instead of after an
        # expensive load.
        if args.fix_input and args.fix_input_interactive:
            # An explicit error instead of silently favoring one flag: the
            # combination is almost always a script mistake, and a surprise
            # behavior (whichever fix ran) is worse than a clear refusal.
            raise SystemExit(
                "--fix-input and --fix-input-interactive are mutually exclusive.\n"
                "  --fix-input corrects fixable issues in place without asking.\n"
                "  --fix-input-interactive shows the diff and prompts before writing.\n"
                "Pick the one you want; drop the other."
            )
        missing = [name for name, value in {"--input": args.input, "--outdir": args.outdir}.items() if not value]
        if missing:
            raise SystemExit(
                "render requires either --project, or --input and --outdir. "
                f"Missing: {', '.join(missing)}"
            )
        if args.device_mode != "cuda" and args.cuda_device is not None:
            raise SystemExit("--cuda-device requires --device-mode cuda.")
        if args.device_mode == "cuda" and args.inference_backend == "vulkan":
            raise SystemExit("--device-mode cuda cannot be combined with --inference-backend vulkan; use --inference-backend pytorch.")
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

        # Speaker references default to the repo-local Seashells clips (the
        # GUI's "Default Voices" list) when the flags are omitted.
        speaker_a_ref = args.speaker_a_ref
        speaker_b_ref = args.speaker_b_ref
        if not speaker_a_ref or not speaker_b_ref:
            defaults = default_voice_choices(Path(__file__).resolve().parents[2])
            if defaults:
                speaker_a_ref = speaker_a_ref or defaults[0].path
                speaker_b_ref = speaker_b_ref or (defaults[1].path if len(defaults) > 1 else defaults[0].path)
        # Monologue mode renders every line in Speaker A's voice, so it only
        # requires A; a missing B must not block a one-narrator render.
        if not speaker_a_ref or (not speaker_b_ref and not args.monologue):
            raise SystemExit(
                "render requires either --project, or --speakerA-ref/--speakerB-ref. "
                "No default Seashells voices were found to fall back on."
            )
        if not speaker_b_ref:
            speaker_b_ref = speaker_a_ref

        # SRT auto-detection: an .srt input is converted to a canonical
        # dialogue script before any analysis, because subtitles ingested
        # raw degrade into narration.
        args.input = _maybe_convert_srt(args.input)

        # Ingestion transformer: report input-formatting issues before the
        # (expensive) pipeline load; with --fix-input, correct them in place,
        # or with --fix-input-interactive show the review + prompt first.
        # A declined fix aborts here, before the pipeline loads anything.
        if args.fix_input_interactive:
            if not sys.stdin.isatty():
                print(
                    "--fix-input-interactive requires a terminal; continuing without "
                    "the review prompt (the file is not modified).",
                    file=sys.stderr,
                )
                _check_input_formatting(args.input, fix=False, json_output=bool(args.check_input_json))
            elif not _check_input_formatting_interactive(args.input):
                return 2
            else:
                # An interactive fix may have converted an .srt input to a
                # script; render the corrected file.
                args.input = _srt_script_if_converted(args.input)
        else:
            _check_input_formatting(args.input, fix=bool(args.fix_input), json_output=bool(args.check_input_json))
            if args.fix_input:
                args.input = _srt_script_if_converted(args.input)

        settings = RenderSettings(
            correction_mode=args.correction_mode,
            model_variant=args.model_variant,
            language=args.language if args.model_variant == "multilingual" else "en",
            export_stems=not args.no_stems,
            loudness_preset=args.loudness,
            device_mode=args.device_mode,
            cuda_device=args.cuda_device,
            inference_backend=args.inference_backend,
            audio_cpp_device=args.audio_cpp_device,
            audio_cpp_threads=args.audio_cpp_threads,
            audio_cpp_timeout=args.audio_cpp_timeout,
            audio_cpp_max_batch=args.audio_cpp_max_batch,
            seed=args.seed,
            target_wpm=args.target_wpm,
            monologue=args.monologue,
            metadata={"title": args.title} if args.title else {},
        )
        voice_settings = _voice_settings_from_args(args)
        speakers = {
            "A": SpeakerSettings(reference_path=speaker_a_ref, voice_settings=voice_settings),
            "B": SpeakerSettings(reference_path=speaker_b_ref, voice_settings=voice_settings),
        }
        for raw in args.speaker_refs:
            key, sep, path = raw.partition("=")
            if not sep or not key.strip() or not path.strip():
                raise SystemExit(
                    f"Invalid --speaker-ref {raw!r}: expected KEY=PATH, e.g. --speaker-ref C=/path/ref.wav"
                )
            key = key.strip().upper()
            if key in ("A", "B"):
                raise SystemExit(
                    f"--speaker-ref {key} duplicates --speakerA-ref/--speakerB-ref; use those flags for A and B."
                )
            if not re.fullmatch(r"[A-X]", key):
                raise SystemExit(
                    f"Invalid --speaker-ref key {key!r}: voices are A..X (up to 24)."
                )
            speakers[key] = SpeakerSettings(reference_path=path.strip(), voice_settings=voice_settings)

    pipeline = OraclePipeline()

    if not args.project:
        plan = pipeline.prepare_plan(args.input, args.outdir, speakers, settings)

    if settings.inference_backend == "vulkan" and not args.no_audio_cpp_setup:
        # Automatic CPU→GPU switch: build audiocpp_cli and/or download the
        # Chatterbox model when missing, then set the env vars for this process
        # so the existing engine path just works. --no-audio-cpp-setup keeps the
        # old fail-fast behavior (the engine's ensure_model_ready still guards).
        from the_oracle.vulkan_setup import run_vulkan_setup, vulkan_setup_needed

        def _progress(line: str) -> None:
            print(f"  [vulkan setup] {line}", file=sys.stderr)

        # When everything is already in place this prints a single "ready"
        # line instead of a no-op setup round-trip. run_vulkan_setup's
        # progress callback already streams every script line, so the result's
        # messages list must NOT be re-printed (that doubled the output).
        if vulkan_setup_needed():
            print("Vulkan backend selected: installing missing prerequisites...", file=sys.stderr)
            result = run_vulkan_setup(progress=_progress)
            if not result.ok:
                raise SystemExit(f"Vulkan backend auto-setup failed: {result.error}")
        print("Vulkan backend ready.", file=sys.stderr)

    configure_logging(Path(plan.output_dir) / "logs" / "cli.log")
    try:
        output_path = pipeline.render(plan, settings)
    except PartialRenderError as exc:
        # The per-chunk failures are in the log; tell the user which rows
        # failed so they can inspect rather than staring at a traceback.
        rows = ", ".join(str(index) for index in exc.failed_rows)
        raise SystemExit(f"{exc} Failed rows: {rows}.")
    except (AudioCppUnavailableError, RDNA1VulkanError, NoAudioToAssembleError) as exc:
        # These messages already say how to recover (download/build the model,
        # fall back to --inference-backend pytorch, or fix the input); surface
        # them cleanly instead of a raw traceback.
        raise SystemExit(str(exc))
    if args.srt:
        from the_oracle.audio.export_srt import write_srt

        write_srt(Path(output_path).with_suffix(".srt"), plan.utterances)
    if args.save_project:
        save_project_manifest(args.save_project, build_saved_project(plan, settings, speakers))
    print(output_path)
    return 0


def handle_setup_vulkan() -> int:
    """One-shot automatic setup for the Vulkan (GPU) backend.

    Builds audiocpp_cli and downloads the Chatterbox model when missing and
    applies them to the session. The CLI render path and the GUI apply the
    same paths automatically, so no shell exports are needed. Exit code 0 on
    success, 1 with a clear error otherwise.
    """
    from the_oracle.vulkan_setup import run_vulkan_setup

    def _progress(line: str) -> None:
        print(f"  [vulkan setup] {line}", file=sys.stderr)

    print("Vulkan backend setup: checking prerequisites...", file=sys.stderr)
    result = run_vulkan_setup(progress=_progress)
    if not result.ok:
        print(f"Vulkan backend setup failed: {result.error}", file=sys.stderr)
        return 1
    print()
    print("Vulkan backend ready:")
    if result.binary:
        print(f"    binary: {result.binary}")
    if result.model:
        print(f"    model:  {result.model}")
    print()
    print("The Oracle applies these automatically for --inference-backend vulkan")
    print("renders and in the GUI -- no shell exports are required.")
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
    if args.command == "check-input":
        return handle_check_input(args)
    if args.command == "setup-vulkan":
        return handle_setup_vulkan()
    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
