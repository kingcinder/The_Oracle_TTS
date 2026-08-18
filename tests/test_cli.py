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


