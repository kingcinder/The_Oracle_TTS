"""Import plain text and markdown into reviewable dialogue segments.

Speaker parsing is deliberately conservative:

* ``Name: dialogue`` lines are only treated as dialogue turns when the label
  actually looks like a speaker (``Speaker A``, ``Alice``, ``A``, ``1``, all-
  caps screenplay names). Common prose colon-labels such as ``Note:``,
  ``Warning:`` and ``See:`` are NOT speakers, so prose never spawns phantom
  dialogue turns.
* Screenplay format (an ALL-CAPS name line followed by its dialogue line) is
  recognised so scripts parse correctly.
* Quoted speech (``"..." said Alice.`` / ``Alice said, "..."``) attributes the
  line to the named speaker.
* Documents without any speaker markers are treated as prose narration: they
  are split into sentences, and every sentence carries no explicit speaker (the
  attributor assigns the whole block to one narrator voice).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt

from the_oracle.speaker_attribution.heuristics import canonical_speaker_label


# A speaker label at the start of a line, capturing label and content.
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][\w .'-]{0,48})\s*:\s*(.+)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Screenplay style: an ALL-CAPS name line (no colon) followed by dialogue.
SCREENPLAY_NAME_RE = re.compile(r"^\s*([A-Z][A-Z .'-]{1,40})\s*$")
# Quoted speech attribution, both orders:
#   "Hello," said Alice.   /   Alice said, "Hello."
_QUOTE_CHARS = "\"'“”‘’"
_SAID_VERBS = (
    "said|asked|replied|answered|whispered|shouted|cried|yelled|responded|"
    "murmured|muttered|called|stated|exclaimed|added|continued|began|"
    "started|noted|offered|suggested|insisted|confirmed|agreed"
)
QUOTED_SPEECH_RE = re.compile(
    rf"^\s*[{_QUOTE_CHARS}](?P<text>.+?)[{_QUOTE_CHARS}],?\s*(?:{_SAID_VERBS})\s+"
    r"(?P<speaker>[A-Z][\w .'-]{0,40})\s*[.!?]?\s*$",
    re.IGNORECASE,
)
QUOTED_SPEECH_REVERSE_RE = re.compile(
    rf"^\s*(?P<speaker>[A-Z][\w .'-]{{0,40}})\s+(?:{_SAID_VERBS})\s*,\s*"
    rf"[{_QUOTE_CHARS}](?P<text>.+?)[{_QUOTE_CHARS}]\s*[.!?]?\s*$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class TextSegment:
    index: int
    text: str
    explicit_speaker: str | None = None
    source_line: int | None = None


@dataclass(slots=True)
class IngestedDocument:
    title: str
    source_path: str
    raw_text: str
    segments: list[TextSegment]


class TextIngestor:
    def __init__(self) -> None:
        self._markdown = MarkdownIt("commonmark")

    def ingest(self, source_path: str | Path) -> IngestedDocument:
        path = Path(source_path)
        raw_text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".md":
            readable_text = self._extract_markdown_text(raw_text)
        else:
            readable_text = raw_text
        segments = self._segment_text(readable_text)
        title = path.stem.replace("_", " ").strip() or "Untitled Dialogue"
        return IngestedDocument(title=title, source_path=str(path), raw_text=readable_text, segments=segments)

    def _extract_markdown_text(self, markdown_text: str) -> str:
        tokens = self._markdown.parse(markdown_text)
        lines: list[str] = []
        blockquote_depth = 0
        for index, token in enumerate(tokens):
            if token.type == "blockquote_open":
                blockquote_depth += 1
                continue
            if token.type == "blockquote_close":
                blockquote_depth = max(0, blockquote_depth - 1)
                continue
            if blockquote_depth:
                continue
            if token.type == "inline" and index > 0 and tokens[index - 1].type == "heading_open":
                continue
            if token.type == "inline":
                if not token.children:
                    for line in token.content.splitlines():
                        stripped = line.strip()
                        if stripped:
                            lines.append(stripped)
                    continue
                pieces: list[str] = []
                for child in token.children:
                    if child.type == "text":
                        pieces.append(child.content)
                    elif child.type in {"softbreak", "hardbreak"}:
                        pieces.append("\n")
                for line in "".join(pieces).splitlines():
                    stripped = line.strip()
                    if stripped:
                        lines.append(stripped)
        return "\n".join(lines)

    def _segment_text(self, text: str) -> list[TextSegment]:
        raw_lines = [line.rstrip() for line in text.splitlines()]
        non_empty = [(idx + 1, line.strip()) for idx, line in enumerate(raw_lines) if line.strip()]
        if not non_empty:
            return []

        marker_count = sum(1 for _, line in non_empty if self._match_speaker_marker(line))
        screenplay_names = [self._match_screenplay_name(line) for _, line in non_empty]
        screenplay_count = len([name for name in screenplay_names if name is not None])
        avg_words = sum(len(line.split()) for _, line in non_empty) / len(non_empty)
        max_words = max(len(line.split()) for _, line in non_empty)

        # Dialogue-style documents (speaker markers present, or short chat-like
        # lines) are segmented line-wise. Prose narration is segmented by
        # sentence within paragraphs instead.
        looks_like_dialogue = marker_count > 0 or (
            len(non_empty) >= 2 and avg_words <= 18 and max_words <= 40
        )
        if looks_like_dialogue:
            return self._segment_lines(non_empty, screenplay_count)
        return self._segment_prose(text)

    @staticmethod
    def _match_speaker_marker(line: str) -> tuple[str, str] | None:
        match = SPEAKER_RE.match(line)
        if not match:
            return None
        label = match.group(1).strip()
        if canonical_speaker_label(label) is None:
            return None
        return label, match.group(2).strip()

    @staticmethod
    def _match_screenplay_name(line: str) -> str | None:
        """An ALL-CAPS screenplay name line, or None."""
        match = SCREENPLAY_NAME_RE.match(line)
        if not match:
            return None
        name = match.group(1).strip()
        # Skip common non-speaker ALL-CAPS lines (scene headings, transitions).
        if re.match(r"^(INT|EXT|FADE|CUT|TITLE|CHAPTER|ACT|SCENE|END)\b", name, re.IGNORECASE):
            return None
        if not any(char.isalpha() for char in name):
            return None
        return name

    @staticmethod
    def _match_quoted_speech(line: str) -> tuple[str, str] | None:
        match = QUOTED_SPEECH_RE.match(line) or QUOTED_SPEECH_REVERSE_RE.match(line)
        if not match:
            return None
        speaker = match.group("speaker").strip().rstrip(".!?").strip()
        text = match.group("text").strip()
        if canonical_speaker_label(speaker) is None:
            return None
        return speaker, text

    def _segment_lines(self, non_empty: list[tuple[int, str]], screenplay_count: int) -> list[TextSegment]:
        segments: list[TextSegment] = []
        index = 0
        position = 0
        while position < len(non_empty):
            line_number, line = non_empty[position]

            marker = self._match_speaker_marker(line)
            if marker:
                label, content = marker
                segments.append(
                    TextSegment(index=index, text=content, explicit_speaker=label, source_line=line_number)
                )
                index += 1
                position += 1
                continue

            # Screenplay format: an ALL-CAPS name line followed by its line of
            # dialogue. Requires at least two screenplay names in the document
            # to avoid misreading a stray heading as a speaker.
            screenplay = self._match_screenplay_name(line)
            if screenplay and screenplay_count >= 2 and position + 1 < len(non_empty):
                next_line_number, next_line = non_empty[position + 1]
                if self._match_speaker_marker(next_line) is None and not self._match_screenplay_name(next_line):
                    segments.append(
                        TextSegment(
                            index=index,
                            text=next_line,
                            explicit_speaker=screenplay,
                            source_line=line_number,
                        )
                    )
                    index += 1
                    position += 2
                    continue

            quoted = self._match_quoted_speech(line)
            if quoted:
                speaker, content = quoted
                segments.append(
                    TextSegment(index=index, text=content, explicit_speaker=speaker, source_line=line_number)
                )
                index += 1
                position += 1
                continue

            segments.append(TextSegment(index=index, text=line, source_line=line_number))
            index += 1
            position += 1
        return segments

    def _segment_prose(self, text: str) -> list[TextSegment]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        segments: list[TextSegment] = []
        index = 0
        for block in blocks:
            marker = self._match_speaker_marker(block)
            if marker:
                segments.append(
                    TextSegment(index=index, text=marker[1], explicit_speaker=marker[0])
                )
                index += 1
                continue
            quoted = self._match_quoted_speech(block)
            if quoted:
                segments.append(
                    TextSegment(index=index, text=quoted[1], explicit_speaker=quoted[0])
                )
                index += 1
                continue
            for sentence in SENTENCE_SPLIT_RE.split(block):
                sentence = sentence.strip()
                if sentence:
                    segments.append(TextSegment(index=index, text=sentence))
                    index += 1
        return segments


def load_document(path: str | Path) -> IngestedDocument:
    return TextIngestor().ingest(path)


def ingest_text_file(path: str | Path) -> IngestedDocument:
    return load_document(path)


parse_input_file = ingest_text_file
ingest_file = ingest_text_file
read_input_file = ingest_text_file
