"""Tests for the render CLI, focused on the Vulkan device/threads knobs.

These are fast and offline: ``handle_render`` is exercised with a fake
pipeline whose ``prepare_plan``/``render`` are stubs, so nothing touches the
real text-repair pipeline or any TTS engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from the_oracle.cli import build_parser, handle_render


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
