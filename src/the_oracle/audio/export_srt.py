"""SRT subtitle export from a render plan's utterances."""

from __future__ import annotations

from pathlib import Path

from the_oracle.models.project import Utterance


def format_timestamp(seconds: float) -> str:
    """Format fractional seconds as an SRT clock: ``HH:MM:SS,mmm``."""
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def utterances_to_srt(utterances: list[Utterance]) -> str:
    """Serialize utterances into SRT, accumulating durations and pauses.

    Each cue starts where the previous utterance's audio ended plus its
    configured ``pause_after_ms``. Missing durations are treated as zero.
    """
    blocks: list[str] = []
    cursor_seconds = 0.0
    for cue_number, utterance in enumerate(utterances, start=1):
        duration = float(utterance.duration_seconds or 0.0)
        start = cursor_seconds
        end = start + duration
        cursor_seconds = end + (float(utterance.pause_after_ms or 0) / 1000.0)
        text = (utterance.repaired_text or utterance.original_text or "").replace("\n", " ").strip()
        speaker = utterance.speaker or "?"
        blocks.append(
            f"{cue_number}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{speaker}: {text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(path: str | Path, utterances: list[Utterance]) -> Path:
    """Write SRT subtitles to ``path`` and return the destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(utterances_to_srt(utterances), encoding="utf-8")
    return destination
