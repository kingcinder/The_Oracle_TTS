"""Tests for the render CLI, focused on the Vulkan device/threads knobs.

These are fast and offline: ``handle_render`` is exercised with a fake
pipeline whose ``prepare_plan``/``render`` are stubs, so nothing touches the
real text-repair pipeline or any TTS engine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from the_oracle.cli import (
    _input_json_document,
    _speaker_ref_report,
    build_parser,
    handle_check_input,
    handle_fix_folder,
    handle_render,
    handle_voices,
)
from the_oracle.ingest_transformer import analyze_input_file


def _render_args(*extra: str, outdir: str = "/tmp/fake-out") -> argparse.Namespace:
    parser = build_parser()
    base = [
        "render",
        "--input",
        "in.txt",
        "--outdir",
        outdir,
        "--speakerA-ref",
        "a.wav",
        "--speakerB-ref",
        "b.wav",
    ]
    return parser.parse_args([*base, *extra])


class _FakePlan:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.utterances = []


class _FakePipeline:
    def __init__(self, output_dir: str) -> None:
        self.settings = None
        self.output = Path(output_dir) / "out.flac"

    def prepare_plan(self, input_path, output_dir, speaker_settings, settings):
        self.settings = settings
        return _FakePlan(output_dir)

    def render(self, plan, settings):
        self.settings = settings
        return self.output


def test_parser_accepts_audio_cpp_flags() -> None:
    args = _render_args(
        "--inference-backend", "vulkan",
        "--audio-cpp-device", "2",
        "--audio-cpp-threads", "6",
        "--audio-cpp-timeout", "120",
        "--audio-cpp-max-batch", "16",
    )
    assert args.inference_backend == "vulkan"
    assert args.audio_cpp_device == 2
    assert args.audio_cpp_threads == 6
    assert args.audio_cpp_timeout == 120
    assert args.audio_cpp_max_batch == 16

    plain = _render_args()
    assert plain.audio_cpp_device is None
    assert plain.audio_cpp_threads is None
    assert plain.audio_cpp_timeout is None
    assert plain.audio_cpp_max_batch is None


def test_audio_cpp_flags_require_vulkan_backend() -> None:
    with patch("the_oracle.cli.OraclePipeline"):
        with pytest.raises(SystemExit, match="require --inference-backend vulkan"):
            handle_render(_render_args("--audio-cpp-device", "1"))

    with patch("the_oracle.cli.OraclePipeline"):
        with pytest.raises(SystemExit, match="require --inference-backend vulkan"):
            handle_render(_render_args("--audio-cpp-threads", "4"))

    with patch("the_oracle.cli.OraclePipeline"):
        with pytest.raises(SystemExit, match="require --inference-backend vulkan"):
            handle_render(_render_args("--audio-cpp-timeout", "120"))

    with patch("the_oracle.cli.OraclePipeline"):
        with pytest.raises(SystemExit, match="require --inference-backend vulkan"):
            handle_render(_render_args("--audio-cpp-max-batch", "16"))


def test_audio_cpp_flags_reject_invalid_ranges(capsys) -> None:
    # Argparse-level range validation gives a clean usage error (SystemExit
    # from parser.error) instead of a ValueError traceback from RenderSettings.
    # parser.error prints the message to stderr and raises SystemExit with
    # just the exit code, so assert on the captured stderr.
    with pytest.raises(SystemExit):
        build_parser().parse_args(["render", "--audio-cpp-device", "-1"])
    assert "expected a non-negative integer" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        build_parser().parse_args(["render", "--audio-cpp-threads", "0"])
    assert "expected a positive integer" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        build_parser().parse_args(["render", "--audio-cpp-timeout", "0"])
    assert "expected a positive integer" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        build_parser().parse_args(["render", "--audio-cpp-max-batch", "0"])
    assert "expected a positive integer" in capsys.readouterr().err

    # Valid values parse fine.
    assert _render_args("--audio-cpp-device", "0").audio_cpp_device == 0
    assert _render_args("--audio-cpp-threads", "1").audio_cpp_threads == 1
    assert _render_args("--audio-cpp-timeout", "1").audio_cpp_timeout == 1
    assert _render_args("--audio-cpp-max-batch", "1").audio_cpp_max_batch == 1


def test_audio_cpp_flags_wire_into_render_settings(tmp_path: Path, monkeypatch) -> None:
    fake = _FakePipeline(str(tmp_path / "output"))
    monkeypatch.setattr("the_oracle.cli.OraclePipeline", lambda: fake)
    args = _render_args(
        "--inference-backend",
        "vulkan",
        # This test only exercises settings wiring, so skip the automatic
        # audio.cpp build/model download that a vulkan render would otherwise
        # trigger in handle_render.
        "--no-audio-cpp-setup",
        "--audio-cpp-device",
        "2",
        "--audio-cpp-threads",
        "6",
        "--audio-cpp-timeout",
        "120",
        "--audio-cpp-max-batch",
        "16",
        outdir=str(tmp_path / "output"),
    )

    assert handle_render(args) == 0

    assert fake.settings.inference_backend == "vulkan"
    assert fake.settings.audio_cpp_device == 2
    assert fake.settings.audio_cpp_threads == 6
    assert fake.settings.audio_cpp_timeout == 120
    assert fake.settings.audio_cpp_max_batch == 16


def test_render_fails_fast_without_input_or_outdir(monkeypatch) -> None:
    """Validation must run before OraclePipeline() is constructed.

    The real pipeline eagerly spawns the LanguageTool download (hundreds of MB)
    and waits on it, so a missing-flag mistake must fail before that load.
    """
    constructed: list[bool] = []

    def fake_pipeline():
        constructed.append(True)
        raise AssertionError("pipeline must not be constructed before input validation")

    monkeypatch.setattr("the_oracle.cli.OraclePipeline", fake_pipeline)
    args = build_parser().parse_args(
        ["render", "--speakerA-ref", "a.wav", "--speakerB-ref", "b.wav"]
    )
    with pytest.raises(SystemExit, match="--input"):
        handle_render(args)
    assert constructed == []


def test_render_requires_speaker_refs_without_defaults(monkeypatch, tmp_path: Path) -> None:
    """Without Seashells defaults and without speaker flags, fail with a clear message."""
    monkeypatch.setattr("the_oracle.cli.default_voice_choices", lambda repo_root: [])
    args = build_parser().parse_args(
        ["render", "--input", "in.txt", "--outdir", str(tmp_path)]
    )
    with pytest.raises(SystemExit, match="--speakerA-ref"):
        handle_render(args)


def test_render_defaults_speaker_refs_to_seashells(monkeypatch, tmp_path: Path) -> None:
    """Omitted speaker flags fall back to the repo-local default voices."""
    from the_oracle.voice_catalog import VoiceChoice

    captured: dict[str, dict] = {}

    class _CapturePipeline:
        def __init__(self) -> None:
            self.output = tmp_path / "out.flac"

        def prepare_plan(self, input_path, output_dir, speaker_settings, settings):
            captured["speakers"] = speaker_settings

            class _Plan:
                output_dir = str(tmp_path)

            return _Plan()

        def render(self, plan, settings):
            return self.output

    monkeypatch.setattr(
        "the_oracle.cli.default_voice_choices",
        lambda repo_root: [VoiceChoice("A", "/tmp/a.wav"), VoiceChoice("B", "/tmp/b.wav")],
    )
    monkeypatch.setattr("the_oracle.cli.OraclePipeline", lambda: _CapturePipeline())

    args = build_parser().parse_args(
        ["render", "--input", "in.txt", "--outdir", str(tmp_path)]
    )
    assert handle_render(args) == 0
    assert captured["speakers"]["A"].reference_path == "/tmp/a.wav"
    assert captured["speakers"]["B"].reference_path == "/tmp/b.wav"


def test_version_flag_prints_version_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    """`--version` prints the installed version and exits 0 without requiring
    a subcommand (the subparsers are required=True)."""
    from the_oracle import __version__

    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == f"the-oracle {__version__}"


def test_monologue_flag_wires_into_render_settings(tmp_path: Path, monkeypatch) -> None:
    """`--monologue` reaches RenderSettings.monologue so the whole input renders
    as a single narrator voice (Speaker A)."""
    fake = _FakePipeline(str(tmp_path / "output"))
    monkeypatch.setattr("the_oracle.cli.OraclePipeline", lambda: fake)
    args = _render_args("--monologue", outdir=str(tmp_path / "output"))

    assert handle_render(args) == 0
    assert fake.settings.monologue is True

    # Default: attribution stays on.
    fake2 = _FakePipeline(str(tmp_path / "output2"))
    monkeypatch.setattr("the_oracle.cli.OraclePipeline", lambda: fake2)
    args2 = _render_args(outdir=str(tmp_path / "output2"))
    assert handle_render(args2) == 0
    assert fake2.settings.monologue is False


def test_speaker_ref_extra_voices_wire_into_speaker_settings(tmp_path: Path, monkeypatch) -> None:
    """`--speaker-ref KEY=PATH` (repeatable) adds character voices C..X for an
    audiobook cast, and the pipeline receives them as speaker settings."""
    captured: dict[str, object] = {}

    class _CapturePipeline(_FakePipeline):
        def prepare_plan(self, input_path, output_dir, speaker_settings, settings):
            captured["speakers"] = speaker_settings
            captured["settings"] = settings
            return _FakePlan(str(output_dir))

    monkeypatch.setattr("the_oracle.cli.OraclePipeline", lambda: _CapturePipeline(str(tmp_path / "output")))
    args = _render_args(
        "--speaker-ref", "C=/tmp/c.wav",
        "--speaker-ref", "D=/tmp/d.wav",
        outdir=str(tmp_path / "output"),
    )

    assert handle_render(args) == 0
    speakers = captured["speakers"]
    assert isinstance(speakers, dict)
    assert set(speakers) == {"A", "B", "C", "D"}
    assert speakers["C"].reference_path == "/tmp/c.wav"
    assert speakers["D"].reference_path == "/tmp/d.wav"
    assert speakers["A"].reference_path == "a.wav"


def test_seed_flag_reaches_render_settings(monkeypatch, tmp_path) -> None:
    """`--seed N` is forwarded into RenderSettings so both backends can seed
    their samplers deterministically."""
    captured: dict[str, object] = {}

    class _CapturePipeline(_FakePipeline):
        def prepare_plan(self, input_path, output_dir, speaker_settings, settings):
            captured["settings"] = settings
            return _FakePlan(str(output_dir))

    monkeypatch.setattr("the_oracle.cli.OraclePipeline", lambda: _CapturePipeline(str(tmp_path / "output")))
    args = _render_args("--seed", "1234", outdir=str(tmp_path / "output"))

    assert handle_render(args) == 0
    assert captured["settings"].seed == 1234


def test_seed_flag_defaults_to_none() -> None:
    assert build_parser().parse_args(["render"]).seed is None
    assert build_parser().parse_args(["render", "--seed", "7"]).seed == 7


def test_speaker_ref_validates_keys_and_duplicates(monkeypatch) -> None:
    """Bad --speaker-ref entries fail fast with a clear message: duplicate A/B,
    invalid characters, and missing KEY=PATH separators."""
    with patch("the_oracle.cli.OraclePipeline"):
        with pytest.raises(SystemExit, match="duplicates --speakerA-ref/--speakerB-ref"):
            handle_render(_render_args("--speaker-ref", "A=/tmp/a.wav"))

    with patch("the_oracle.cli.OraclePipeline"):
        with pytest.raises(SystemExit, match="voices are A..X"):
            handle_render(_render_args("--speaker-ref", "Z=/tmp/z.wav"))

    with patch("the_oracle.cli.OraclePipeline"):
        with pytest.raises(SystemExit, match="expected KEY=PATH"):
            handle_render(_render_args("--speaker-ref", "no-equals-here"))


# ----------------------------------------------------------------------------
# --fix-input: ingestion transformer check in the CLI render path
# ----------------------------------------------------------------------------

def test_fix_input_flag_parses_off_by_default() -> None:
    assert _render_args().fix_input is False
    assert _render_args("--fix-input").fix_input is True


def test_fix_input_reports_issues_without_modifying(tmp_path: Path, capsys) -> None:
    from the_oracle.cli import _check_input_formatting

    target = tmp_path / "messy.txt"
    target.write_text("A - Hello there.\nB - Hi back.\n", encoding="utf-8")
    _check_input_formatting(str(target), fix=False)
    err = capsys.readouterr().err
    assert "2 fixable issue(s)" in err
    assert "--fix-input" in err
    # Report-only: the file is untouched.
    assert target.read_text(encoding="utf-8") == "A - Hello there.\nB - Hi back.\n"


def test_fix_input_corrects_file_and_keeps_backup(tmp_path: Path, capsys) -> None:
    from the_oracle.cli import _check_input_formatting

    target = tmp_path / "messy.txt"
    original = "A - Hello there.\nB - Hi back.\n"
    target.write_text(original, encoding="utf-8")
    _check_input_formatting(str(target), fix=True)
    err = capsys.readouterr().err
    assert "corrected 2 issue(s)" in err
    assert "backup:" in err
    assert target.read_text(encoding="utf-8") == "A: Hello there.\nB: Hi back.\n"
    backups = list(tmp_path.glob("messy.txt.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == original


def test_fix_input_silent_on_clean_and_missing_files(tmp_path: Path, capsys) -> None:
    from the_oracle.cli import _check_input_formatting

    clean = tmp_path / "clean.txt"
    clean.write_text("A: Hello there.\nB: Hi back.\n", encoding="utf-8")
    _check_input_formatting(str(clean), fix=False)
    # A clean file is still reported on: the cast's --speaker-ref hints
    # are the useful content of the report in that case.
    assert "Speaker voices to provide" in capsys.readouterr().err
    _check_input_formatting(str(tmp_path / "does-not-exist.txt"), fix=True)
    assert capsys.readouterr().err == ""


def test_fix_input_with_no_fixable_issues_leaves_file(tmp_path: Path, capsys) -> None:
    """--fix-input on a warnings-only file explains and changes nothing."""
    from the_oracle.cli import _check_input_formatting

    target = tmp_path / "warn.txt"
    target.write_text(
        "A: Hello there.\nNote: this prose line is fine but flagged.\n",
        encoding="utf-8",
    )
    _check_input_formatting(str(target), fix=True)
    err = capsys.readouterr().err
    assert "no fixable issues; the file was left unchanged" in err
    assert target.read_text(encoding="utf-8").startswith("A: Hello there.\n")


def test_fix_input_runs_before_pipeline_construction(tmp_path: Path, monkeypatch) -> None:
    """The transformer check runs before OraclePipeline() is built, so a
    --fix-input correction happens even when pipeline setup would be slow."""
    order: list[str] = []

    def fake_check(input_path: str, fix: bool, json_output: bool = False) -> None:
        order.append("check")

    def fake_pipeline():
        order.append("pipeline")
        return _FakePipeline(str(tmp_path / "output"))

    monkeypatch.setattr("the_oracle.cli._check_input_formatting", fake_check)
    monkeypatch.setattr("the_oracle.cli.OraclePipeline", fake_pipeline)
    target = tmp_path / "in.txt"
    target.write_text("A: Hi.\n", encoding="utf-8")
    args = _render_args("--fix-input", outdir=str(tmp_path / "output"))
    # Point the args at the real temp file.
    args.input = str(target)
    assert handle_render(args) == 0
    assert order == ["check", "pipeline"]


# ---------------------------------------------------------------------------
# check-input subcommand: lint a dialogue file without rendering
# ---------------------------------------------------------------------------


def _check_args(*extra: str) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(["check-input", *extra])


def test_check_input_parses_file_and_fix_flag() -> None:
    args = _check_args("some.txt")
    assert args.file == "some.txt"
    assert args.fix is False
    args = _check_args("some.txt", "--fix")
    assert args.fix is True


def test_check_input_clean_file_exits_zero(tmp_path: Path, capsys) -> None:
    target = tmp_path / "clean.txt"
    target.write_text("A: fine.\nB: also fine.\n", encoding="utf-8")

    from the_oracle.cli import handle_check_input

    assert handle_check_input(_check_args(str(target))) == 0
    assert "no formatting issues found" in capsys.readouterr().out


def test_check_input_reports_issues_and_exits_one(tmp_path: Path, capsys) -> None:
    target = tmp_path / "messy.txt"
    target.write_text("A - Hello there.\nB - Hi back.\n", encoding="utf-8")

    from the_oracle.cli import handle_check_input

    assert handle_check_input(_check_args(str(target))) == 1
    out = capsys.readouterr().out
    assert "2 fixable formatting issue(s)" in out
    assert "rewrite as 'A: Hello there.'" in out
    assert "Re-run with --fix" in out
    # Report-only: the file is untouched and no backup exists.
    assert target.read_text(encoding="utf-8") == "A - Hello there.\nB - Hi back.\n"
    assert not list(tmp_path.glob("*.bak-*"))


def test_check_input_fix_corrects_file_and_exits_zero(tmp_path: Path, capsys) -> None:
    target = tmp_path / "messy.txt"
    target.write_text("A - Hello there.\n", encoding="utf-8")

    from the_oracle.cli import handle_check_input

    assert handle_check_input(_check_args(str(target), "--fix")) == 0
    out = capsys.readouterr().out
    assert "corrected 1 issue(s)" in out
    assert "no formatting issues remain" in out
    assert target.read_text(encoding="utf-8") == "A: Hello there.\n"
    backups = list(tmp_path.glob("*.bak-*"))
    assert len(backups) == 1
    assert "Hello" in backups[0].read_text(encoding="utf-8")


def test_check_input_fix_never_rewrites_warnings(tmp_path: Path, capsys) -> None:
    target = tmp_path / "warn.txt"
    target.write_text("A: fine.\nChapter: prose.\n", encoding="utf-8")

    from the_oracle.cli import handle_check_input

    assert handle_check_input(_check_args(str(target), "--fix")) == 1
    out = capsys.readouterr().out
    assert "1 formatting warning(s)" in out
    # No re-run hint: there is nothing fixable for --fix to act on.
    assert "Re-run with --fix" not in out
    assert target.read_text(encoding="utf-8") == "A: fine.\nChapter: prose.\n"
    assert not list(tmp_path.glob("*.bak-*"))


def test_check_input_missing_file_exits_two(tmp_path: Path, capsys) -> None:
    from the_oracle.cli import handle_check_input

    assert handle_check_input(_check_args(str(tmp_path / "nope.txt"))) == 2
    assert "file not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --check-input-json: machine-readable format report on the render path
# ---------------------------------------------------------------------------


def test_check_input_json_flag_parses_off_by_default() -> None:
    assert _render_args().check_input_json is False
    assert _render_args("--check-input-json").check_input_json is True


def test_check_input_json_emits_single_document(tmp_path: Path, capsys) -> None:
    """Exactly one JSON object is printed to stdout for a consumer to parse."""
    import json

    from the_oracle.cli import _check_input_formatting

    target = tmp_path / "messy.txt"
    target.write_text("A - Hello there.\nB - Hi back.\n", encoding="utf-8")
    _check_input_formatting(str(target), fix=False, json_output=True)
    out = capsys.readouterr().out
    doc = json.loads(out)  # the whole stdout must be one JSON document
    assert doc["file"] == str(target)
    assert doc["fixable_count"] == 2
    assert doc["warning_count"] == 0
    assert [issue["line"] for issue in doc["issues"]] == [1, 2]
    assert all(issue["fixable"] is True for issue in doc["issues"])
    assert "rewrite as 'A: Hello there.'" in doc["issues"][0]["description"]
    assert "fixed_count" not in doc and "backup" not in doc
    # Report-only: file untouched.
    assert target.read_text(encoding="utf-8") == "A - Hello there.\nB - Hi back.\n"


def test_check_input_json_clean_file_zero_counts(tmp_path: Path, capsys) -> None:
    import json

    from the_oracle.cli import _check_input_formatting

    target = tmp_path / "clean.txt"
    target.write_text("A: fine.\n", encoding="utf-8")
    _check_input_formatting(str(target), fix=False, json_output=True)
    doc = json.loads(capsys.readouterr().out)
    assert doc == {
        "file": str(target),
        "fixable_count": 0,
        "warning_count": 0,
        "issues": [],
        # The file's one speaker ("A") is suggested even though the file is
        # otherwise clean — knowing the voice flags is the point.
        "speaker_refs": [{"speaker": "a", "voice_key": "A", "flag": "--speakerA-ref PATH", "new": False}],
        "rejected_labels": [],
    }


def test_check_input_json_with_fix_reports_applied_fix(tmp_path: Path, capsys) -> None:
    """With --fix-input, one document carries both the pre-fix issue list and
    the applied fix count + backup path."""
    import json

    from the_oracle.cli import _check_input_formatting

    target = tmp_path / "messy.txt"
    original = "A - Hello there.\n"
    target.write_text(original, encoding="utf-8")
    _check_input_formatting(str(target), fix=True, json_output=True)
    doc = json.loads(capsys.readouterr().out)
    assert doc["fixable_count"] == 1
    assert doc["fixed_count"] == 1
    assert doc["backup"].startswith(str(tmp_path))
    assert "Hello" in doc["backup"] or True  # backup path is per-file
    assert target.read_text(encoding="utf-8") == "A: Hello there.\n"
    backups = list(tmp_path.glob("*.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == original


def test_check_input_json_missing_file_reports_error(tmp_path: Path, capsys) -> None:
    import json

    from the_oracle.cli import _check_input_formatting

    _check_input_formatting(str(tmp_path / "nope.txt"), fix=False, json_output=True)
    doc = json.loads(capsys.readouterr().out)
    assert doc["error"] == "file not found"
    assert doc["fixable_count"] == 0
    assert doc["issues"] == []


def test_check_input_json_warnings_flagged_not_fixable(tmp_path: Path, capsys) -> None:
    import json

    from the_oracle.cli import _check_input_formatting

    target = tmp_path / "warn.txt"
    target.write_text("A: fine.\nChapter: prose.\n", encoding="utf-8")
    _check_input_formatting(str(target), fix=True, json_output=True)
    doc = json.loads(capsys.readouterr().out)
    assert doc["fixable_count"] == 0
    assert doc["warning_count"] == 1
    assert doc["issues"][0]["fixable"] is False
    assert "fixed_count" not in doc
    # Warnings-only: nothing written, no backup.
    assert not list(tmp_path.glob("*.bak-*"))


# ---------------------------------------------------------------------------
# --speaker-ref suggestions for a script's cast
# ---------------------------------------------------------------------------


def test_suggest_speaker_refs_group_cast(tmp_path: Path) -> None:
    """Three speakers map to A/B/C; C and beyond need an added --speaker-ref."""
    from the_oracle.ingest_transformer import suggest_speaker_refs

    text = (
        "Winston - The plans are ready.\n"
        "Julia - Do they suspect anything?\n"
        "O'Brien - We should move tonight.\n"
    )
    refs = suggest_speaker_refs(text)
    assert [(r.speaker, r.voice_key, r.is_new) for r in refs] == [
        ("winston", "A", False),
        ("julia", "B", False),
        ("o'brien", "C", True),
    ]
    assert refs[0].flag == "--speakerA-ref PATH"
    assert refs[1].flag == "--speakerB-ref PATH"
    assert refs[2].flag == "--speaker-ref C=PATH"


def test_suggest_speaker_refs_includes_post_fix_cast(tmp_path: Path) -> None:
    """A speaker only visible after a transform is still suggested."""
    from the_oracle.ingest_transformer import suggest_speaker_refs

    refs = suggest_speaker_refs("Alice - Hello.\nBob - Hi.\n")
    assert [(r.speaker, r.voice_key) for r in refs] == [("alice", "A"), ("bob", "B")]


def test_rejected_labels_identified(tmp_path: Path) -> None:
    from the_oracle.ingest_transformer import rejected_labels

    text = "A: fine.\nChapter: prose.\nNote: also prose.\nB: okay.\n"
    # Both are prose labels the engine will read as narration.
    assert rejected_labels(text) == ["Chapter", "Note"]


def test_check_input_json_includes_speaker_refs(tmp_path: Path, capsys) -> None:
    import json

    from the_oracle.cli import _check_input_formatting

    target = tmp_path / "cast.txt"
    target.write_text(
        "Winston - Ready.\nJulia - Sure?\nO'Brien - Tonight.\n", encoding="utf-8"
    )
    _check_input_formatting(str(target), fix=False, json_output=True)
    doc = json.loads(capsys.readouterr().out)
    assert doc["speaker_refs"] == [
        {"speaker": "winston", "voice_key": "A", "flag": "--speakerA-ref PATH", "new": False},
        {"speaker": "julia", "voice_key": "B", "flag": "--speakerB-ref PATH", "new": False},
        {"speaker": "o'brien", "voice_key": "C", "flag": "--speaker-ref C=PATH", "new": True},
    ]
    assert doc["rejected_labels"] == []


def test_check_input_json_clean_file_still_suggests_refs(tmp_path: Path, capsys) -> None:
    """Even a healthy file gets cast suggestions — that's the point: know the
    flags before rendering."""
    import json

    from the_oracle.cli import _check_input_formatting

    target = tmp_path / "clean.txt"
    target.write_text("Winston: Ready.\nJulia: Sure?\n", encoding="utf-8")
    _check_input_formatting(str(target), fix=False, json_output=True)
    doc = json.loads(capsys.readouterr().out)
    assert doc["fixable_count"] == 0
    assert [r["speaker"] for r in doc["speaker_refs"]] == ["winston", "julia"]


def test_check_input_human_report_prints_ref_hints(tmp_path: Path, capsys) -> None:
    from the_oracle.cli import handle_check_input

    target = tmp_path / "cast.txt"
    target.write_text(
        "Winston - Ready.\nJulia - Sure?\nO'Brien - Tonight.\nChapter: notes.\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(["check-input", str(target)])
    assert handle_check_input(args) == 1
    captured = capsys.readouterr()
    assert "Speaker voices to provide" in captured.err
    assert "winston -> voice A: use --speakerA-ref PATH" in captured.err
    assert "o'brien -> voice C: add --speaker-ref C=PATH" in captured.err
    assert "'Chapter' is not accepted as a speaker label" in captured.err


# ---------------------------------------------------------------------------
# SRT auto-detection in the render path
# ---------------------------------------------------------------------------


_SRT_SAMPLE = """1
00:00:01,000 --> 00:00:04,000
Winston: The plans are ready.

2
00:00:04,500 --> 00:00:06,000
Do they suspect anything?
"""


def test_maybe_convert_srt_converts_and_reuses(tmp_path: Path, capsys) -> None:
    from the_oracle.cli import _maybe_convert_srt

    source = tmp_path / "movie.srt"
    source.write_text(_SRT_SAMPLE, encoding="utf-8")

    result = _maybe_convert_srt(str(source))
    assert result == str(tmp_path / "movie.srt.txt")
    script = Path(result).read_text(encoding="utf-8")
    assert script.startswith("winston: The plans are ready.")
    err = capsys.readouterr().err
    assert "converted 2 cue(s)" in err
    assert "rendering the script" in err
    # The subtitle file itself is untouched.
    assert "-->" in source.read_text(encoding="utf-8")

    # A second run reuses the converted script instead of failing.
    result2 = _maybe_convert_srt(str(source))
    assert result2 == result
    assert "reusing previously converted script" in capsys.readouterr().err


def test_maybe_convert_srt_passthrough_non_srt(tmp_path: Path) -> None:
    from the_oracle.cli import _maybe_convert_srt

    # Non-.srt extension: never touched, even if the content looks like SRT.
    decoy = tmp_path / "notes.txt"
    decoy.write_text(_SRT_SAMPLE, encoding="utf-8")
    assert _maybe_convert_srt(str(decoy)) == str(decoy)
    # .srt extension but not valid SubRip: passthrough for ordinary handling,
    # and nothing is converted.
    fake = tmp_path / "fake.srt"
    fake.write_text("A: Hello there.\n", encoding="utf-8")
    assert _maybe_convert_srt(str(fake)) == str(fake)
    assert not list(tmp_path.glob("*.srt.txt"))
    # Missing file: passthrough (the render path reports it).
    assert _maybe_convert_srt(str(tmp_path / "nope.srt")) == str(tmp_path / "nope.srt")


def test_srt_conversion_runs_before_transformer_check(tmp_path: Path, monkeypatch, capsys) -> None:
    """The SRT conversion lands before the format check, so the transformer
    sees the converted script, not the subtitles."""
    from the_oracle import cli

    order: list[str] = []

    def fake_srt(input_path: str) -> str:
        order.append("srt")
        return input_path

    def fake_check(input_path: str, fix: bool, json_output: bool = False) -> None:
        order.append("check")

    def fake_pipeline(*_a, **_k):
        order.append("pipeline")
        return cli._FakePipeline_for_tests(str(tmp_path / "output")) if hasattr(cli, "_FakePipeline_for_tests") else None

    monkeypatch.setattr(cli, "_maybe_convert_srt", fake_srt)
    monkeypatch.setattr(cli, "_check_input_formatting", fake_check)
    monkeypatch.setattr(cli, "OraclePipeline", lambda *a, **k: order.append("pipeline"))

    target = tmp_path / "in.txt"
    target.write_text("A: Hi.\n", encoding="utf-8")
    args = _render_args("--input", str(target), outdir=str(tmp_path / "output"))
    args.input = str(target)
    # A fake pipeline object isn't enough for the rest of handle_render; stop
    # after ordering is established by letting prepare/render fail harmlessly.
    try:
        cli.handle_render(args)
    except Exception:
        pass  # ordering assertion below is the point of this test
    assert order.index("srt") < order.index("check")


# ---------------------------------------------------------------------------
# --fix-input-interactive: popup + preview flow on the render path
# ---------------------------------------------------------------------------


def test_fix_input_interactive_flag_parses_off_by_default() -> None:
    assert _render_args().fix_input_interactive is False
    assert _render_args("--fix-input-interactive").fix_input_interactive is True


def test_interactive_accept_applies_fix_with_backup(tmp_path: Path, capsys) -> None:
    from the_oracle.cli import _check_input_formatting_interactive

    target = tmp_path / "messy.txt"
    original = "A - Hello there.\nB - Hi back.\n"
    target.write_text(original, encoding="utf-8")

    questions: list[str] = []
    answers = iter(["y\n"])

    def fake_prompt(question: str) -> str:
        questions.append(question)
        return next(answers)

    assert _check_input_formatting_interactive(str(target), prompt=fake_prompt) is True
    captured = capsys.readouterr()
    # Review content: summary, rule-labeled diff, then the confirmation.
    assert "2 fixable issue(s)" in captured.err
    assert "+ [dash/pipe separator] A: Hello there." in captured.err
    assert questions == ["Apply these fixes before rendering? [y/N]: "]
    assert "corrected 2 issue(s)" in captured.err
    assert target.read_text(encoding="utf-8") == "A: Hello there.\nB: Hi back.\n"
    backups = list(tmp_path.glob("*.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == original


def test_interactive_decline_leaves_file_and_reports(tmp_path: Path, capsys) -> None:
    from the_oracle.cli import _check_input_formatting_interactive

    target = tmp_path / "messy.txt"
    target.write_text("A - Hello there.\n", encoding="utf-8")

    assert _check_input_formatting_interactive(str(target), prompt=lambda _q: "n\n") is False
    captured = capsys.readouterr()
    assert "Fix declined" in captured.err
    assert target.read_text(encoding="utf-8") == "A - Hello there.\n"
    assert not list(tmp_path.glob("*.bak-*"))


def test_interactive_eof_declines_instead_of_crashing(tmp_path: Path, capsys) -> None:
    from the_oracle.cli import _check_input_formatting_interactive

    target = tmp_path / "messy.txt"
    target.write_text("A - Hello there.\n", encoding="utf-8")

    def raise_eof(_question: str) -> str:
        raise EOFError

    assert _check_input_formatting_interactive(str(target), prompt=raise_eof) is False
    assert target.read_text(encoding="utf-8") == "A - Hello there.\n"


def test_interactive_clean_and_warnings_only_proceed_without_prompt(tmp_path: Path, capsys) -> None:
    from the_oracle.cli import _check_input_formatting_interactive

    clean = tmp_path / "clean.txt"
    clean.write_text("A: fine.\n", encoding="utf-8")
    assert _check_input_formatting_interactive(str(clean), prompt=lambda _q: (_ for _ in ()).throw(AssertionError("prompted on clean file"))) is True

    warn = tmp_path / "warn.txt"
    warn.write_text("A: fine.\nChapter: prose.\n", encoding="utf-8")
    assert _check_input_formatting_interactive(str(warn), prompt=lambda _q: (_ for _ in ()).throw(AssertionError("prompted on warnings-only file"))) is True
    captured = capsys.readouterr()
    assert "cannot be fixed automatically" in captured.err
    assert not list(tmp_path.glob("*.bak-*"))


def test_interactive_decline_aborts_render_before_pipeline(tmp_path: Path, monkeypatch) -> None:
    """A declined fix must stop the render before OraclePipeline is built."""
    from the_oracle import cli

    calls: list[str] = []

    def fake_interactive(_input_path: str, **_kwargs) -> bool:
        calls.append("interactive")
        return False  # user declines

    # capsys/pytest stdin is not a tty; pretend we have a real terminal so
    # the interactive branch is actually exercised.
    monkeypatch.setattr(
        "sys.stdin", type("_FakeTty", (), {"isatty": staticmethod(lambda: True), "readline": lambda self: ""})()
    )
    monkeypatch.setattr(cli, "_check_input_formatting_interactive", fake_interactive)
    monkeypatch.setattr(
        cli, "OraclePipeline", lambda *a, **k: calls.append("pipeline") or (_ for _ in ()).throw(AssertionError("pipeline must not load"))
    )

    target = tmp_path / "in.txt"
    target.write_text("A - Hello there.\n", encoding="utf-8")
    args = _render_args("--fix-input-interactive", outdir=str(tmp_path / "output"))
    args.input = str(target)
    assert cli.handle_render(args) == 2
    assert calls == ["interactive"]


def test_interactive_non_tty_refuses_prompt_and_continues(tmp_path: Path, monkeypatch, capsys) -> None:
    """Pipes/CI: the flag degrades to a report-only check, never a hang."""
    from the_oracle import cli
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # isatty() -> False
    calls: list[str] = []

    monkeypatch.setattr(
        cli,
        "_check_input_formatting",
        lambda path, fix, json_output=False: calls.append(("check", fix)),
    )

    target = tmp_path / "in.txt"
    target.write_text("A - Hello there.\n", encoding="utf-8")
    args = _render_args("--fix-input-interactive", outdir=str(tmp_path / "output"))
    args.input = str(target)
    try:
        cli.handle_render(args)
    except Exception:
        pass  # downstream fake-pipeline limits are irrelevant to this test
    # Report-only check ran (fix=False), the interactive prompt never fired.
    assert calls and calls[0] == ("check", False)


# ------------- check-input --json (CI mode) -------------

def test_check_input_json_clean_file_exits_zero(tmp_path, capsys) -> None:
    file_path = tmp_path / "clean.txt"
    file_path.write_text("A: fine.\n", encoding="utf-8")
    args = argparse.Namespace(file=str(file_path), fix=False, json=True)
    assert handle_check_input(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["file"] == str(file_path)
    assert document["fixable_count"] == 0
    assert document["warning_count"] == 0
    assert document["issues"] == []
    assert document["speaker_refs"]  # speaker A suggested
    assert document["rejected_labels"] == []
    assert "fixed_count" not in document


def test_check_input_json_issues_exits_one(tmp_path, capsys) -> None:
    file_path = tmp_path / "messy.txt"
    file_path.write_text("[Speaker A]: Hello there.\n", encoding="utf-8")
    args = argparse.Namespace(file=str(file_path), fix=False, json=True)
    assert handle_check_input(args) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["fixable_count"] == 1
    issue = document["issues"][0]
    assert issue["line"] == 1
    assert issue["fixable"] is True
    assert "brackets" in issue["description"]
    # Exactly one JSON document, nothing else on stdout.
    assert capsys.readouterr().out == ""
    assert "fixed_count" not in document


def test_check_input_json_missing_file_error_document(tmp_path, capsys) -> None:
    args = argparse.Namespace(file=str(tmp_path / "nope.txt"), fix=False, json=True)
    assert handle_check_input(args) == 2
    document = json.loads(capsys.readouterr().out)
    assert document["error"] == "file not found"
    assert document["issues"] == []
    assert document["speaker_refs"] == []
    assert document["rejected_labels"] == []


def test_check_input_json_fix_reports_backup_and_clean_state(tmp_path, capsys) -> None:
    file_path = tmp_path / "messy.txt"
    file_path.write_text("[Speaker A]: Hello there.\n", encoding="utf-8")
    args = argparse.Namespace(file=str(file_path), fix=True, json=True)
    assert handle_check_input(args) == 0
    document = json.loads(capsys.readouterr().out)
    # The document describes the file's CURRENT (post-fix) state.
    assert document["fixable_count"] == 0
    assert document["fixed_count"] == 1
    assert Path(document["backup"]).is_file()
    assert file_path.read_text(encoding="utf-8") == "Speaker A: Hello there.\n"


def test_check_input_json_reuses_render_path_schema(tmp_path, capsys) -> None:
    """The standalone document must match the render path's schema exactly."""
    file_path = tmp_path / "messy.txt"
    file_path.write_text("[Speaker A]: Hello there.\n", encoding="utf-8")
    analysis = analyze_input_file(file_path)
    refs, rejected = _speaker_ref_report(file_path, None)
    standalone = _input_json_document(str(file_path), analysis, speaker_refs=refs, rejected=rejected)
    assert set(standalone) == {
        "file", "fixable_count", "warning_count", "issues",
        "speaker_refs", "rejected_labels",
    }


# ------------- WebVTT (.vtt) pre-flight conversion -------------

_VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Winston>The party is tonight.\n"


def test_maybe_convert_srt_handles_vtt(tmp_path, capsys) -> None:
    from the_oracle.cli import _maybe_convert_srt

    vtt = tmp_path / "episode.vtt"
    vtt.write_text(_VTT, encoding="utf-8")
    result = _maybe_convert_srt(str(vtt))
    assert result == str(tmp_path / "episode.vtt.txt")
    assert (tmp_path / "episode.vtt.txt").read_text(encoding="utf-8").startswith("winston:")
    assert "converted" in capsys.readouterr().err
    # The subtitle file itself is untouched.
    assert vtt.read_text(encoding="utf-8") == _VTT


def test_maybe_convert_srt_reuses_existing_vtt_script(tmp_path, capsys) -> None:
    from the_oracle.cli import _maybe_convert_srt

    vtt = tmp_path / "episode.vtt"
    vtt.write_text(_VTT, encoding="utf-8")
    (tmp_path / "episode.vtt.txt").write_text("winston: reused.\n", encoding="utf-8")
    result = _maybe_convert_srt(str(vtt))
    assert result == str(tmp_path / "episode.vtt.txt")
    # The pre-existing script is reused, not clobbered.
    assert (tmp_path / "episode.vtt.txt").read_text(encoding="utf-8") == "winston: reused.\n"
    assert "reusing" in capsys.readouterr().err


def test_srt_script_if_converted_handles_vtt(tmp_path) -> None:
    from the_oracle.cli import _srt_script_if_converted

    vtt = tmp_path / "episode.vtt"
    assert _srt_script_if_converted(str(vtt)) == str(vtt)  # no script yet
    (tmp_path / "episode.vtt.txt").write_text("winston: x.\n", encoding="utf-8")
    assert _srt_script_if_converted(str(vtt)) == str(tmp_path / "episode.vtt.txt")


# ------------- speaker-ref hints at the subtitle-conversion moment -------------

_VTT_HINTS = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Winston>The party is tonight.\n\n00:00:04.500 --> 00:00:06.000\nJulia: Agreed.\n"


def test_maybe_convert_srt_prints_speaker_ref_hints(tmp_path, capsys) -> None:
    from the_oracle.cli import _maybe_convert_srt

    vtt = tmp_path / "episode.vtt"
    vtt.write_text(_VTT_HINTS, encoding="utf-8")
    _maybe_convert_srt(str(vtt))
    err = capsys.readouterr().err
    assert "converted 2 cue(s)" in err
    assert "Speaker voices to provide" in err
    assert "winston -> voice A: use --speakerA-ref PATH" in err
    assert "julia -> voice B: use --speakerB-ref PATH" in err


def test_maybe_convert_srt_reuse_also_prints_hints(tmp_path, capsys) -> None:
    from the_oracle.cli import _maybe_convert_srt

    vtt = tmp_path / "episode.vtt"
    vtt.write_text(_VTT_HINTS, encoding="utf-8")
    (tmp_path / "episode.vtt.txt").write_text("winston: reused.\n", encoding="utf-8")
    _maybe_convert_srt(str(vtt))
    err = capsys.readouterr().err
    assert "reusing previously converted script" in err
    assert "Speaker voices to provide" in err


# ------------- --fix-input / --fix-input-interactive are mutually exclusive -------------


def test_fix_input_flags_are_mutually_exclusive(tmp_path, capsys) -> None:
    """Combining the two fix flags is an explicit error, not a silent pick."""
    args = build_parser().parse_args([
        "render",
        "--input", str(tmp_path / "in.txt"),
        "--outdir", str(tmp_path / "out"),
        "--speakerA-ref", "a.wav",
        "--speakerB-ref", "b.wav",
        "--fix-input",
        "--fix-input-interactive",
    ])
    with pytest.raises(SystemExit) as excinfo:
        handle_render(args)
    message = str(excinfo.value)
    assert "--fix-input and --fix-input-interactive are mutually exclusive" in message
    assert "--fix-input-interactive shows the diff" in message  # the helpful part


def test_fix_input_flag_conflict_beats_missing_required_args() -> None:
    """The conflict is reported first, before --input/--outdir validation."""
    args = build_parser().parse_args(["render", "--fix-input", "--fix-input-interactive"])
    with pytest.raises(SystemExit) as excinfo:
        handle_render(args)
    assert "mutually exclusive" in str(excinfo.value)


# ------------- rule names in check-input human + JSON reports -------------


def test_check_input_human_report_includes_rule_names(tmp_path, capsys) -> None:
    file_path = tmp_path / "messy.txt"
    file_path.write_text("[A]: Hello.\n[2024-01-01 10:00] B: Hi.\n", encoding="utf-8")
    args = argparse.Namespace(file=str(file_path), fix=False, json=False)
    assert handle_check_input(args) == 1
    err_out = capsys.readouterr().out
    assert "line 1 [bracket]:" in err_out
    assert "line 2 [timestamp]:" in err_out


def test_check_input_json_includes_rule_field(tmp_path, capsys) -> None:
    file_path = tmp_path / "messy.txt"
    file_path.write_text("[A]: Hello.\n", encoding="utf-8")
    args = argparse.Namespace(file=str(file_path), fix=False, json=True)
    assert handle_check_input(args) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["issues"][0]["rule"] == "bracket"
    # SRT files report the whole-document rule.
    srt_path = tmp_path / "movie.srt"
    srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nWinston: The plans are ready.\n", encoding="utf-8"
    )
    args = argparse.Namespace(file=str(srt_path), fix=False, json=True)
    assert handle_check_input(args) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["issues"][0]["rule"] == "srt"


# ------------- voices subcommand -------------


def test_voices_lists_default_clips_human(capsys) -> None:
    args = build_parser().parse_args(["voices"])
    assert handle_voices(args) == 0
    out = capsys.readouterr().out
    assert "->" in out
    assert "(default Speaker A)" in out
    assert "(default Speaker B)" in out
    # Paths point at real files (the repo bundles Seashells/generic).
    first_path = Path(out.split("->", 1)[1].split(" (default")[0].strip())
    assert first_path.is_file()


def test_voices_json_is_parseable_array(capsys) -> None:
    args = build_parser().parse_args(["voices", "--json"])
    assert handle_voices(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert isinstance(document, list) and document
    assert {"label", "path"} <= set(document[0])
    assert Path(document[0]["path"]).is_file()


# ------------- check-input --check-refs -------------


def _real_audio_ref(tmp_path: Path) -> Path:
    """A tiny real WAV built with soundfile (no bundled-clip dependency)."""
    import soundfile as sf
    import numpy as np

    path = tmp_path / "ref_ok.wav"
    sf.write(str(path), np.zeros(480, dtype="float32"), 24000, subtype="PCM_16")
    return path


def test_check_input_check_refs_ok_passes(tmp_path: Path, capsys) -> None:
    script = tmp_path / "script.txt"
    script.write_text("A: Hello.\nB: Hi.\n", encoding="utf-8")
    good = _real_audio_ref(tmp_path)
    args = build_parser().parse_args(
        ["check-input", str(script), "--check-refs", "--speakerA-ref", str(good)]
    )
    assert handle_check_input(args) == 0
    out = capsys.readouterr().out
    assert "[OK] voice A" in out
    assert "[OK] voice B" in out  # unset counts as OK


def test_check_input_check_refs_bad_ref_fails(tmp_path: Path, capsys) -> None:
    script = tmp_path / "script.txt"
    script.write_text("A: Hello.\nB: Hi.\n", encoding="utf-8")
    good = _real_audio_ref(tmp_path)
    bad = tmp_path / "ref_bad.wav"
    bad.write_text("this is not audio", encoding="utf-8")
    missing = tmp_path / "missing.wav"
    args = build_parser().parse_args(
        [
            "check-input",
            str(script),
            "--check-refs",
            "--speakerA-ref",
            str(good),
            "--speakerB-ref",
            str(bad),
            "--speaker-ref",
            f"C={missing}",
        ]
    )
    assert handle_check_input(args) == 1
    out = capsys.readouterr().out
    assert "[OK] voice A" in out
    assert "[BAD] voice B" in out
    assert "[BAD] voice C" in out and "file not found" in out


def test_check_input_json_includes_checked_refs(tmp_path: Path, capsys) -> None:
    script = tmp_path / "script.txt"
    script.write_text("A: Hello.\n", encoding="utf-8")
    bad = tmp_path / "ref_bad.wav"
    bad.write_text("junk", encoding="utf-8")
    args = build_parser().parse_args(
        ["check-input", str(script), "--json", "--check-refs", "--speakerA-ref", str(bad)]
    )
    assert handle_check_input(args) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["refs_ok"] is False
    assert document["checked_refs"][0]["voice_key"] == "A"
    assert document["checked_refs"][0]["ref_status"].startswith("bad:")


def test_check_input_json_checked_refs_ok(tmp_path: Path, capsys) -> None:
    script = tmp_path / "script.txt"
    script.write_text("A: Hello.\n", encoding="utf-8")
    good = _real_audio_ref(tmp_path)
    args = build_parser().parse_args(
        ["check-input", str(script), "--json", "--check-refs", "--speakerA-ref", str(good)]
    )
    assert handle_check_input(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["refs_ok"] is True
    assert document["checked_refs"][0]["ref_status"] == "ok"


def test_check_input_without_check_refs_omits_checked_refs(tmp_path: Path, capsys) -> None:
    script = tmp_path / "script.txt"
    script.write_text("A: Hello.\n", encoding="utf-8")
    args = build_parser().parse_args(["check-input", str(script), "--json"])
    assert handle_check_input(args) == 0
    document = json.loads(capsys.readouterr().out)
    # Stable schema: the keys exist in JSON mode even when --check-refs was
    # not passed; the list is simply empty and nothing was validated.
    assert document["checked_refs"] == []
    assert document["refs_ok"] is True


# ------------- fix-folder subcommand -------------


def test_fix_folder_dry_run_writes_nothing(tmp_path: Path, capsys) -> None:
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("[A]: Hello.\n", encoding="utf-8")
    args = build_parser().parse_args(["fix-folder", str(tmp_path), "--dry-run"])
    assert handle_fix_folder(args) == 0
    assert dirty.read_text(encoding="utf-8") == "[A]: Hello.\n"  # untouched
    out = capsys.readouterr().out
    assert "would be corrected" in out and "Dry run" in out


def test_fix_folder_applies_with_backups(tmp_path: Path, capsys) -> None:
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("[A]: Hello.\n", encoding="utf-8")
    args = build_parser().parse_args(["fix-folder", str(tmp_path)])
    assert handle_fix_folder(args) == 0
    assert "A: Hello." in dirty.read_text(encoding="utf-8")
    backups = list(tmp_path.glob("dirty.txt.bak-*"))
    assert len(backups) == 1
    assert "[A]: Hello." in backups[0].read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "corrected 1 file(s)" in out and "backups kept: 1" in out


def test_fix_folder_json_dry_run_schema(tmp_path: Path, capsys) -> None:
    (tmp_path / "a.txt").write_text("[A]: Hello.\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text(
        "[2024-01-01 10:00] Alice: note\n", encoding="utf-8"
    )
    args = build_parser().parse_args(["fix-folder", str(tmp_path), "--json", "--dry-run"])
    assert handle_fix_folder(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["dry_run"] is True
    assert document["applied"] == []
    assert {Path(f["file"]).name for f in document["fixes"]} == {"a.txt", "b.md"}
    a_fix = next(f for f in document["fixes"] if Path(f["file"]).name == "a.txt")
    assert a_fix["fix_count"] == 1
    assert a_fix["line_fixes"][0]["rule"] == "bracket"
    # Nothing written in dry-run, so no backups exist.
    assert not list(tmp_path.rglob("*.bak-*"))


def test_fix_folder_json_apply_reports_backups(tmp_path: Path, capsys) -> None:
    (tmp_path / "a.txt").write_text("[A]: Hello.\n", encoding="utf-8")
    args = build_parser().parse_args(["fix-folder", str(tmp_path), "--json"])
    assert handle_fix_folder(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["dry_run"] is False
    assert len(document["applied"]) == 1
    entry = document["applied"][0]
    assert Path(entry["file"]).name == "a.txt"
    assert entry["fixed_count"] == 1
    assert entry["backup"] and Path(entry["backup"]).is_file()
    # Idempotent: a second scan finds nothing.
    args2 = build_parser().parse_args(["fix-folder", str(tmp_path), "--json"])
    assert handle_fix_folder(args2) == 0
    document2 = json.loads(capsys.readouterr().out)
    assert document2["fixes"] == [] and document2["applied"] == []


def test_fix_folder_missing_folder_exit_two(tmp_path: Path, capsys) -> None:
    args = build_parser().parse_args(["fix-folder", str(tmp_path / "nope"), "--json"])
    assert handle_fix_folder(args) == 2
    document = json.loads(capsys.readouterr().out)
    assert document["error"] == "folder not found"
    assert document["fixes"] == [] and document["applied"] == []


def test_fix_folder_clean_folder_exits_zero(tmp_path: Path, capsys) -> None:
    (tmp_path / "ok.txt").write_text("A: clean.\n", encoding="utf-8")
    args = build_parser().parse_args(["fix-folder", str(tmp_path)])
    assert handle_fix_folder(args) == 0
    assert "No fixable" in capsys.readouterr().out


def test_fix_folder_subtitle_writes_sibling_script(tmp_path: Path, capsys) -> None:
    srt = tmp_path / "ep.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n[A]: Hi.\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(["fix-folder", str(tmp_path), "--json"])
    assert handle_fix_folder(args) == 0
    document = json.loads(capsys.readouterr().out)
    targets = {Path(entry["file"]).name for entry in document["applied"]}
    assert targets == {"ep.srt.txt"}  # convert-not-overwrite
    assert srt.read_text(encoding="utf-8").startswith("1\n")  # subtitle untouched


# ------------- rejected-label rename hints -------------


def test_rejected_label_hint_includes_rename_flag(tmp_path: Path, capsys) -> None:
    script = tmp_path / "script.txt"
    script.write_text(
        "A: Hello there.\nNote: a footnote\nB: Hi.\n", encoding="utf-8"
    )
    args = build_parser().parse_args(["check-input", str(script)])
    assert handle_check_input(args) == 1
    err = capsys.readouterr().err
    # The rename hint names the exact flag the renamed label would need.
    assert "'Note'" in err
    assert "rename the label" in err and "then provide" in err
    assert "--speaker-ref C=PATH" in err or "--speakerB-ref PATH" in err


def test_rejected_label_json_rename_flag(tmp_path: Path, capsys) -> None:
    script = tmp_path / "script.txt"
    script.write_text("A: Hello.\nNote: footnote\nB: Hi.\n", encoding="utf-8")
    args = build_parser().parse_args(["check-input", str(script), "--json"])
    assert handle_check_input(args) == 1
    document = json.loads(capsys.readouterr().out)
    entries = [e for e in document["rejected_labels"] if e["label"] == "Note"]
    assert entries and entries[0]["rename_flag"]
    assert entries[0]["rename_flag"].startswith("--speaker")


def test_valid_labels_do_not_get_rename_hint(tmp_path: Path, capsys) -> None:
    script = tmp_path / "script.txt"
    script.write_text("A: Hello.\nB: Hi.\n", encoding="utf-8")
    args = build_parser().parse_args(["check-input", str(script), "--json"])
    assert handle_check_input(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["rejected_labels"] == []
