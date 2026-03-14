"""Import plain text and markdown into reviewable dialogue segments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt


SPEAKER_RE = re.compile(r"^\s*([A-Za-z][\w .'-]{0,48})\s*:\s*(.+)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


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
        segments: list[TextSegment] = []
        if non_empty and len(non_empty) >= 2:
            for index, (line_number, line) in enumerate(non_empty):
                explicit = None
                content = line
                match = SPEAKER_RE.match(line)
                if match:
                    explicit = match.group(1).strip()
                    content = match.group(2).strip()
                segments.append(TextSegment(index=index, text=content, explicit_speaker=explicit, source_line=line_number))
            return segments

        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        index = 0
        for block in blocks:
            match = SPEAKER_RE.match(block)
            if match:
                segments.append(TextSegment(index=index, text=match.group(2).strip(), explicit_speaker=match.group(1).strip()))
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
