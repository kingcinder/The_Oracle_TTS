"""Subtitle ingestion: detect, parse, and convert to dialogue scripts.

Subtitle files are a common source of "dialogue" material, but their format
(numbered cue blocks with clock timestamps) is nothing like the ``Label:
dialogue`` scripts the ingester understands — ingested directly, every cue
degrades into narration. This module converts a valid subtitle file —
SubRip (``.srt``) or WebVTT (``.vtt``) — into a canonical dialogue script
before analysis/rendering:

* Timestamps, cue indexes, HTML markup (``<i>``, ``<font>``), and ASS-style
  overrides (``{\\an8}``) are stripped.
* A ``Name:`` prefix on a cue's first line (validated by the same
  :func:`canonical_speaker_label` the ingester uses) becomes the speaker;
  cues without one inherit the previous cue's speaker, and a script with no
  names at all becomes a single ``Narrator``.
* Dashed lines inside a cue (``- Hello`` / ``- Hi``) are split into separate
  turns, matching the two-speaker subtitle convention.
* Consecutive cues by the same speaker are merged into one turn — subtitle
  cues are line fragments, and one turn per 2-second cue would render
  terribly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from the_oracle.speaker_attribution.heuristics import canonical_speaker_label

# A cue clock line: "00:00:01,000 --> 00:00:04,000" (SubRip, comma or dot
# millis) or "01:02.500 --> 01:05.000" (WebVTT: optional hours, dot millis,
# optional cue settings after the end time).
_SRT_TIME_RE = re.compile(
    r"^\s*(?:(?P<h1>\d{1,2}):)?(?P<m1>\d{1,2}):(?P<s1>\d{2})[,.](?P<ms1>\d{1,3})\s*-->\s*"
    r"(?:(?P<h2>\d{1,2}):)?(?P<m2>\d{1,2}):(?P<s2>\d{2})[,.](?P<ms2>\d{1,3})"
    r"(?P<settings>\s.*)?$"
)

# A numbered cue index line.
_SRT_INDEX_RE = re.compile(r"^\s*\d+\s*$")

# HTML/font markup, WebVTT timestamps tags (<00:00:01.000> karaoke),
# WebVTT class/voice spans, and ASS override blocks inside cue text.
_SRT_MARKUP_RE = re.compile(r"<[^>]+>|\{\\[^}]*\}")

# WebVTT voice spans: ``<v Winston>`` names the cue's speaker. Lifted to a
# ``Winston:`` prefix *before* generic markup stripping, so the name is not
# deleted along with the tags.
_VTT_VOICE_SPAN_RE = re.compile(r"<v\s+([^>]+)>")

# WebVTT metadata blocks that carry no dialogue.
_VTT_BLOCK_RE = re.compile(r"^\s*(NOTE|STYLE|REGION)\b")

# A dashed subtitle line: "- Hello there." (two-speaker convention).
_SRT_DASH_RE = re.compile(r"^\s*[-\u2013\u2014]\s+")

# Fallback speaker when a subtitle names nobody.
_NARRATOR = "Narrator"

# Speaker-label prefix on a cue's first text line: "Winston: The plans...".
_SPEAKER_PREFIX_RE = re.compile(r"^(?P<label>[A-Za-z][\w .'\-]{0,48}?)\s*:\s*(?P<rest>\S.*)$")

# Gaps below this are treated as no silence at all (subtitle cues routinely
# butt up against each other with tens of milliseconds between them; those
# would only add directive noise). Above it, the gap is emitted as a
# [pause=N] directive on the following turn.
_MIN_NOTABLE_GAP_MS = 400.0

# The pacing engine clamps turn pauses to this domain; emitting anything
# larger would be silently clipped downstream anyway.
_MAX_PAUSE_MS = 2000


@dataclass(slots=True)
class SrtCue:
    """One subtitle cue with its text lines cleaned of markup."""

    index: int
    start_seconds: float
    end_seconds: float
    lines: list[str]


def looks_like_srt(text: str) -> bool:
    """True when *text* parses as SubRip or WebVTT subtitles.    Requires at least one complete cue (clock line and text), so ordinary
    prose or dialogue scripts never match — an arrow-like string inside
    prose is not enough.
    """
    return len(parse_srt(text)) >= 1


def parse_srt(text: str) -> list[SrtCue]:
    """Parse SubRip/WebVTT cues from *text*, tolerating missing index lines.

    WebVTT specifics handled: the ``WEBVTT`` header line, ``NOTE``/``STYLE``
    /``REGION`` metadata blocks (skipped, no dialogue there), cue
    identifiers (skipped), cue settings after the end time (dropped), and
    ``MM:SS.mmm`` clocks without an hours component. Raises
    :class:`ValueError` when the text contains cue-looking blocks whose
    clock lines are malformed — that indicates a corrupt subtitle file
    rather than a non-subtitle document.
    """
    blocks: list[tuple[list[str], int]] = []  # (raw lines, block start line no)
    current: list[str] = []
    start_line = 1
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if raw.strip():
            if not current:
                start_line = line_number
            current.append(raw)
        elif current:
            blocks.append((current, start_line))
            current = []
    if current:
        blocks.append((current, start_line))

    # WebVTT header: a lone first block containing just the magic line.
    if blocks and len(blocks[0][0]) == 1 and blocks[0][0][0].strip().startswith("WEBVTT"):
        blocks = blocks[1:]

    cues: list[SrtCue] = []
    for raw_lines, block_start in blocks:
        lines = list(raw_lines)
        # WebVTT metadata blocks (NOTE/STYLE/REGION) carry no dialogue.
        if lines and _VTT_BLOCK_RE.match(lines[0]):
            continue
        # WebVTT cue identifiers: an optional line before the clock that is
        # neither an index number nor a clock line.
        if len(lines) >= 2 and not _SRT_INDEX_RE.match(lines[0]) and not _SRT_TIME_RE.match(lines[0]) and _SRT_TIME_RE.match(lines[1]):
            lines = lines[1:]
        if lines and _SRT_INDEX_RE.match(lines[0]):
            lines = lines[1:]
        if not lines:
            continue
        clock = _SRT_TIME_RE.match(lines[0])
        if not clock:
            continue  # not a cue block (trailing credits text, stray numbers)
        hours_a = int(clock.group("h1") or 0)
        min_a = int(clock.group("m1"))
        sec_a = int(clock.group("s1"))
        ms_a = int(clock.group("ms1"))
        hours_b = int(clock.group("h2") or 0)
        min_b = int(clock.group("m2"))
        sec_b = int(clock.group("s2"))
        ms_b = int(clock.group("ms2"))
        start = hours_a * 3600 + min_a * 60 + sec_a + ms_a / 1000
        end = hours_b * 3600 + min_b * 60 + sec_b + ms_b / 1000
        text_lines = [
            _SRT_MARKUP_RE.sub("", _VTT_VOICE_SPAN_RE.sub(r"\1: ", line)).strip()
            for line in lines[1:]
        ]
        text_lines = [line for line in text_lines if line]
        if not text_lines:
            continue  # an empty cue carries no dialogue
        cues.append(
            SrtCue(
                index=len(cues) + 1,
                start_seconds=start,
                end_seconds=end,
                lines=text_lines,
            )
        )
    return cues


def _speaker_of(line: str) -> tuple[str | None, str]:
    """Split a leading ``Name:`` prefix off a cue line.

    Returns ``(speaker_or_None, remaining_text)``. The label must pass the
    ingester's own speaker validation, so ``Note:``-style prose prefixes in
    subtitles are kept as speech text rather than becoming phantom speakers.
    """
    match = _SPEAKER_PREFIX_RE.match(line)
    if match:
        label = match.group("label").strip()
        if canonical_speaker_label(label) is not None:
            return canonical_speaker_label(label), match.group("rest").strip()
    return None, line


def _cue_turns(
    cue: SrtCue, current_speaker: str | None, last_other_speaker: str | None
) -> list[tuple[str, str]]:
    """Resolve one cue into (speaker, text) turns.

    Dashed lines split into separate turns (the two-speaker subtitle
    convention): the first line belongs to the current speaker and each
    subsequent line alternates to the *other* voice of the pair — the last
    speaker different from the current one, or the ``Narrator`` fallback
    when nobody else has been named yet. Splitting preserves the turn
    boundary so misattributions are at least visible and editable. A
    ``Name:`` prefix on a line always wins over the alternation.
    """
    turns: list[tuple[str, str]] = []
    dashed = any(_SRT_DASH_RE.match(line) for line in cue.lines)
    if dashed:
        speaker = current_speaker or _NARRATOR
        for line in cue.lines:
            body = _SRT_DASH_RE.sub("", line).strip()
            if not body:
                continue
            named, text = _speaker_of(body)
            if named:
                speaker = named
            elif turns:
                # Alternate to the other voice of the pair.
                speaker = last_other_speaker or _NARRATOR
            turns.append((speaker, text))
        return turns

    first_line = cue.lines[0]
    speaker, text = _speaker_of(first_line)
    if len(cue.lines) > 1:
        text = " ".join([text, *[line.strip() for line in cue.lines[1:] if line.strip()]])
    return [(speaker or current_speaker or _NARRATOR, text)]


def srt_to_dialogue_text(text: str) -> tuple[str, int, int]:
    """Convert SRT *text* into a canonical dialogue script.

    Returns ``(script_text, cue_count, speaker_count)``. Consecutive cues by
    the same speaker are merged into one turn, because subtitle cues are
    sentence fragments, not complete dialogue lines.

    Cue *timings* are preserved where they matter: the gap between one
    cue's end and the next cue's start becomes a ``[pause=N]`` directive on
    the following turn when the gap is large enough to be audible (above
    ``_MIN_NOTABLE_GAP_MS``; subtitles routinely carry sub-100-ms gaps that
    would be noise) and is clamped to the pacing engine's domain (``
    _MAX_PAUSE_MS``).
    """
    cues = parse_srt(text)
    turns: list[tuple[str, str]] = []
    turn_gaps: list[float] = []  # ms of source silence before each turn
    current_speaker: str | None = None
    last_other_speaker: str | None = None
    for cue_index, cue in enumerate(cues):
        gap_ms = 0.0
        if cue_index > 0:
            gap_ms = max(0.0, (cue.start_seconds - cues[cue_index - 1].end_seconds) * 1000.0)
        dashed_cue = any(_SRT_DASH_RE.match(line) for line in cue.lines)
        cue_turns = _cue_turns(cue, current_speaker, last_other_speaker)
        for turn_index, (speaker, utterance) in enumerate(cue_turns):
            same_speaker = turns and turns[-1][0] == speaker
            # Consecutive cues by the same speaker merge into one turn (cue
            # fragments read terribly one-by-one), but turns *within* one
            # dashed cue must never re-merge — the dash split them because
            # they are different people.
            may_merge = same_speaker and not (dashed_cue and turn_index > 0)
            if may_merge:
                turns[-1] = (speaker, f"{turns[-1][1]} {utterance}".strip())
                # The merged turn spans both cues; its leading pause is the
                # largest notable gap across everything folded into it.
                turn_gaps[-1] = max(turn_gaps[-1], gap_ms if turn_index == 0 else 0.0)
            else:
                # Within one dashed cue the split turns are simultaneous in
                # the source, so only the first turn inherits the cue gap.
                turn_gaps.append(gap_ms if turn_index == 0 else 0.0)
                turns.append((speaker, utterance))
            if current_speaker and speaker != current_speaker:
                last_other_speaker = current_speaker
            current_speaker = speaker
    lines: list[str] = []
    for turn_index, (speaker, utterance) in enumerate(turns):
        gap = turn_gaps[turn_index] if turn_index < len(turn_gaps) else 0.0
        if turn_index > 0 and gap >= _MIN_NOTABLE_GAP_MS:
            lines.append(f"{speaker}: [pause={int(round(min(gap, _MAX_PAUSE_MS)))}] {utterance}")
        else:
            lines.append(f"{speaker}: {utterance}")
    script = "\n".join(lines) + ("\n" if lines else "")
    speakers = {speaker for speaker, _utterance in turns}
    return script, len(cues), len(speakers)


def convert_srt_file(path: str | Path, *, overwrite: bool = False) -> tuple[Path, int, int]:
    """Convert an ``.srt`` file on disk into a sibling ``.txt`` dialogue script.

    The converted script is written next to the subtitle file as
    ``<name>.srt.txt`` (or ``<name>.txt`` when the file is already named
    that way) so the render pipeline ingests the script, not the subtitles.
    Returns ``(script_path, cue_count, speaker_count)``. Raises
    :class:`ValueError` when the file is not valid SubRip, and
    :class:`FileExistsError` when the target script already exists unless
    ``overwrite`` is set.
    """
    file_path = Path(path)
    raw = file_path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    script, cue_count, speaker_count = srt_to_dialogue_text(text)
    if not script:
        raise ValueError(f"No valid SubRip cues found in {file_path}")
    stem_tail = ".srt.txt" if file_path.suffix.lower() == ".srt" else ".vtt.txt" if file_path.suffix.lower() == ".vtt" else ".txt"
    # with_suffix cannot build compound names like ".vtt.txt" (it replaces
    # the whole suffix), so the target is assembled from the stem.
    target = file_path.with_name(file_path.stem + stem_tail)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Converted script already exists: {target}")
    target.write_text(script, encoding="utf-8")
    return target, cue_count, speaker_count
