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


# ---------------------------------------------------------------------------
# WebVTT (.vtt)
# ---------------------------------------------------------------------------

VTT_SAMPLE = """WEBVTT

NOTE This is a comment block
spanning two lines.

STYLE
::cue { color: white }

crackers
00:01.000 --> 00:04.000 align:start position:0%
- What gives?

00:00:07.000 --> 00:00:09.000
<v Winston>The party <i>is</i> tonight.

00:00:09.500 --> 00:00:12.000
Julia: Do they suspect anything?
"""


def test_looks_like_srt_accepts_webvtt() -> None:
    assert looks_like_srt(VTT_SAMPLE) is True


def test_looks_like_srt_rejects_prose_with_vtt_marker() -> None:
    # A WEBVTT mention alone is not a subtitle file.
    assert looks_like_srt("WEBVTT was mentioned in the letter.\n") is False


def test_webvtt_parse_skips_metadata_and_identifiers() -> None:
    cues = parse_srt(VTT_SAMPLE)
    assert len(cues) == 3
    # MM:SS.mmm clocks parse (no hours component).
    assert cues[0].start_seconds == pytest.approx(1.0)
    assert cues[0].end_seconds == pytest.approx(4.0)
    # Cue settings after the end time are dropped.
    assert cues[0].lines == ["- What gives?"]


def test_webvtt_voice_span_becomes_speaker() -> None:
    """<v Winston> is WebVTT's speaker convention; it must survive markup stripping."""
    script, cue_count, speaker_count = srt_to_dialogue_text(VTT_SAMPLE)
    assert cue_count == 3
    assert "winston: [pause=2000] The party is tonight." in script
    assert "julia: [pause=500] Do they suspect anything?" in script
    assert speaker_count == 3  # Narrator fallback + winston + julia


def test_webvtt_conversion_writes_vtt_txt_script(tmp_path) -> None:
    vtt_path = tmp_path / "episode.vtt"
    vtt_path.write_text(VTT_SAMPLE, encoding="utf-8")
    target, cue_count, speaker_count = convert_srt_file(vtt_path)
    assert target == tmp_path / "episode.vtt.txt"
    assert target.read_text(encoding="utf-8").startswith("Narrator: What gives?")
    assert cue_count == 3
    # The subtitle file itself is never modified.
    assert "WEBVTT" in vtt_path.read_text(encoding="utf-8")


def test_webvtt_cast_round_trips_through_ingester(tmp_path) -> None:
    vtt_path = tmp_path / "episode.vtt"
    vtt_path.write_text(VTT_SAMPLE, encoding="utf-8")
    target, _cues, _speakers = convert_srt_file(vtt_path)
    document = TextIngestor().ingest(target)
    speakers = {segment.explicit_speaker for segment in document.segments}
    assert {"narrator", "winston", "julia"} <= {s.lower() for s in speakers}


# ------------- cue-timing pauses -------------


def test_cue_gaps_become_pause_directives() -> None:
    """A notable silence between cues becomes a [pause=N] on the next turn."""
    srt = (
        "1\n00:00:01,000 --> 00:00:04,000\nWinston: The plans are ready.\n"
        "\n"
        "2\n00:00:07,500 --> 00:00:09,000\nJulia: Agreed.\n"  # 3.5 s gap
        "\n"
        "3\n00:00:09,100 --> 00:00:10,000\nJulia: Tomorrow.\n"  # 100 ms gap
    )
    script, _cues, _speakers = srt_to_dialogue_text(srt)
    assert "winston: The plans are ready." in script  # first turn: no directive
    assert "julia: [pause=2000] Agreed." in script  # clamped to the 2000 ms domain
    assert "julia: [pause=100] Tomorrow." not in script
    # The sub-threshold-gap cue merges into the previous same-speaker turn
    # (existing merge semantics) without adding a pause directive.
    assert "julia: [pause=2000] Agreed. Tomorrow." in script


def test_pause_directives_round_trip_through_ingester(tmp_path) -> None:
    """The emitted directives parse into real pause settings for the pipeline."""
    srt = (
        "1\n00:00:01,000 --> 00:00:04,000\nWinston: The plans are ready.\n"
        "\n"
        "2\n00:00:05,000 --> 00:00:06,500\nJulia: Agreed.\n"  # 1 s gap
    )
    source = tmp_path / "movie.srt"
    source.write_text(srt, encoding="utf-8")
    target, _cues, _speakers = convert_srt_file(source)
    assert "[pause=1000]" in target.read_text(encoding="utf-8")
    from the_oracle.text_repair.directives import apply_directives, parse_directives
    from the_oracle.models.project import VoiceSettings

    line = next(
        line for line in target.read_text(encoding="utf-8").splitlines() if "[pause=" in line
    )
    text, directives = parse_directives(line)
    assert directives["pause_ms"] == 1000
    assert "[pause=" not in text  # the directive is stripped from spoken text
    assert apply_directives(VoiceSettings(), directives).pause_ms == 1000
