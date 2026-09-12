"""Tests for SRT subtitle detection and dialogue-script conversion."""

from __future__ import annotations

from pathlib import Path

import pytest

from the_oracle.srt_ingest import (
    convert_srt_file,
    looks_like_srt,
    parse_srt,
    srt_to_dialogue_text,
)
from the_oracle.text_ingest import TextIngestor

SAMPLE = """1
00:00:01,000 --> 00:00:04,000
<i>Winston:</i> The plans are ready.

2
00:00:04,500 --> 00:00:06,000
Do they suspect anything?

3
00:00:06,200 --> 00:00:09,000
- Not yet.
- We should move tonight.

4
00:00:09,100 --> 00:00:12,000
Julia: Agreed.
"""


# ---------------------------------------------------------------------------
# Detection: SRT yes, everything else no
# ---------------------------------------------------------------------------


def test_looks_like_srt_accepts_subtitles() -> None:
    assert looks_like_srt(SAMPLE) is True


def test_looks_like_srt_rejects_prose_and_dialogue() -> None:
    prose = "Note: the arrow --> here means implies.\nSee: chapter 2.\n"
    dialogue = "A: Hello there.\nB: Hi back.\n"
    assert looks_like_srt(prose) is False
    assert looks_like_srt(dialogue) is False
    assert looks_like_srt("") is False


def test_parse_srt_requires_complete_cues() -> None:
    # An arrow-like line alone is not a cue; no cues, no detection.
    assert parse_srt("warning 1 --> 2\n") == []
    # Clock lines without text lines are empty cues: dropped.
    clock_only = "1\n00:00:01,000 --> 00:00:02,000\n"
    assert parse_srt(clock_only) == []


def test_parse_srt_tolerates_missing_index_and_dots() -> None:
    text = "00:00:01.000 --> 00:00:02.500\nHello there.\n"
    cues = parse_srt(text)
    assert len(cues) == 1
    assert cues[0].start_seconds == 1.0
    assert cues[0].end_seconds == 2.5
    assert cues[0].lines == ["Hello there."]


# ---------------------------------------------------------------------------
# Conversion semantics
# ---------------------------------------------------------------------------


def test_conversion_strips_markup_and_timestamps() -> None:
    script, cue_count, speaker_count = srt_to_dialogue_text(SAMPLE)
    assert "<i>" not in script and "-->" not in script
    assert cue_count == 4


def test_conversion_names_speakers_and_merges_consecutive() -> None:
    script, _cues, speaker_count = srt_to_dialogue_text(SAMPLE)
    lines = script.strip().splitlines()
    # Winston's cue + unattributed continuation + his dashed line merge.
    assert lines[0].startswith("winston: The plans are ready. Do they suspect anything? Not yet.")
    # The dashed second line is a separate turn.
    assert lines[1].startswith("Narrator: We should move tonight.")
    assert lines[2] == "julia: Agreed."
    assert speaker_count == 3


def test_conversion_dashed_cue_splits_turns() -> None:
    text = "1\n00:00:01,000 --> 00:00:02,000\n- Hello.\n- Hi back.\n"
    script, _cues, speaker_count = srt_to_dialogue_text(text)
    lines = script.strip().splitlines()
    assert len(lines) == 2
    assert lines[0] == "Narrator: Hello."
    assert lines[1] == "Narrator: Hi back."
    assert speaker_count == 1


def test_conversion_alternates_dashes_between_named_pair() -> None:
    text = (
        "1\n00:00:01,000 --> 00:00:02,000\nWinston: Ready?\n"
        "\n2\n00:00:02,100 --> 00:00:04,000\n- Always.\n- Prove it.\n"
    )
    script, _cues, _speakers = srt_to_dialogue_text(text)
    lines = script.strip().splitlines()
    # First dashed line continues the current speaker (and merges across the
    # cue boundary, like any same-speaker continuation); the second dash
    # alternates to the other voice of the pair.
    assert lines[0] == "winston: Ready? Always."
    assert lines[1] == "Narrator: Prove it."


def test_conversion_prose_prefix_stays_speech() -> None:
    """A 'Note:' prefix inside a subtitle is prose, not a phantom speaker."""
    text = "1\n00:00:01,000 --> 00:00:02,000\nNote: this stays spoken.\n"
    script, _cues, _speakers = srt_to_dialogue_text(text)
    assert script == "Narrator: Note: this stays spoken.\n"


def test_conversion_no_cues_raises_in_file_mode(tmp_path: Path) -> None:
    target = tmp_path / "not subs.srt"
    target.write_text("A: Hello there.\nB: Hi back.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        convert_srt_file(target)


# ---------------------------------------------------------------------------
# File conversion + real ingester attribution
# ---------------------------------------------------------------------------


def test_convert_srt_file_writes_script_and_round_trips(tmp_path: Path) -> None:
    source = tmp_path / "movie.srt"
    source.write_text(SAMPLE, encoding="utf-8")

    target, cue_count, speaker_count = convert_srt_file(source)

    assert target == tmp_path / "movie.srt.txt"
    assert cue_count == 4 and speaker_count == 3
    # The subtitle file itself is untouched.
    assert "<i>" in source.read_text(encoding="utf-8")
    # The script attributes through the real ingester.
    document = TextIngestor().ingest(target)
    speakers = {segment.explicit_speaker for segment in document.segments}
    assert speakers == {"winston", "julia", "Narrator"}


def test_convert_srt_file_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "movie.srt"
    source.write_text(SAMPLE, encoding="utf-8")
    convert_srt_file(source)
    with pytest.raises(FileExistsError):
        convert_srt_file(source)
    # overwrite=True is allowed for regeneration.
    _target, _cues, _speakers = convert_srt_file(source, overwrite=True)
