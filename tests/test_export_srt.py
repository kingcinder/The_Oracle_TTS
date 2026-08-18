"""Tests for SRT subtitle export from a render plan's utterances."""

from __future__ import annotations

from pathlib import Path

from the_oracle.audio.export_srt import format_timestamp, utterances_to_srt, write_srt
from the_oracle.models.project import Utterance


def test_format_timestamp_seconds_to_srt_clock() -> None:
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(61.5) == "00:01:01,500"
    assert format_timestamp(3661.25) == "01:01:01,250"


def test_utterances_to_srt_uses_durations_and_pauses() -> None:
    first = Utterance(
        index=0,
        original_text="Hello there.",
        repaired_text="Hello there.",
        speaker="A",
        duration_seconds=1.5,
        pause_after_ms=180,
    )
    second = Utterance(
        index=1,
        original_text="Hi!",
        repaired_text="Hi!",
        speaker="B",
        duration_seconds=0.8,
        pause_after_ms=180,
    )
    lines = utterances_to_srt([first, second]).splitlines()

    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:01,500"
    assert lines[2] == "A: Hello there."
    assert lines[3] == ""
    assert lines[4] == "2"
    # second cue starts after first utterance duration + its 180 ms pause
    assert lines[5] == "00:00:01,680 --> 00:00:02,480"
    assert lines[6] == "B: Hi!"


def test_utterances_to_srt_handles_missing_duration_as_zero() -> None:
    utterance = Utterance(index=0, original_text="Only.", speaker="A", duration_seconds=None)
    srt = utterances_to_srt([utterance])
    assert "00:00:00,000 --> 00:00:00,000" in srt
    assert "A: Only." in srt


def test_write_srt_writes_file(tmp_path: Path) -> None:
    utterance = Utterance(
        index=0,
        original_text="Hi.",
        repaired_text="Hi.",
        speaker="A",
        duration_seconds=1.0,
        pause_after_ms=0,
    )
    destination = write_srt(tmp_path / "subtitles.srt", [utterance])
    assert destination.exists()
    assert "A: Hi." in destination.read_text(encoding="utf-8")
