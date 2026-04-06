"""Import plain text and markdown into reviewable dialogue segments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt


_SPEAKER_LABEL_PATTERN = r"[A-Za-z][\w .'/&-]{0,48}"
_SPEAKER_CAPTURE_PATTERN = rf"{_SPEAKER_LABEL_PATTERN}(?:\s*\([^)]+\))?"
_NAME_TOKEN_PATTERN = r"[A-Z][A-Za-z0-9'&.-]*"
_NAME_PATTERN = rf"{_NAME_TOKEN_PATTERN}(?:\s+{_NAME_TOKEN_PATTERN}){{0,2}}"
_ATTRIBUTION_ENTITY_PATTERN = rf"(?:{_NAME_PATTERN}|he|she|they|we|i)"
_SPEECH_VERBS = (
    "admitted",
    "added",
    "agreed",
    "answered",
    "asked",
    "began",
    "called",
    "continued",
    "cried",
    "explained",
    "groaned",
    "insisted",
    "laughed",
    "murmured",
    "muttered",
    "noted",
    "observed",
    "promised",
    "replied",
    "said",
    "shouted",
    "sighed",
    "smiled",
    "snapped",
    "told",
    "warned",
    "went on",
    "whispered",
    "wrote",
    "yelled",
)
_SPEECH_VERB_PATTERN = "|".join(sorted((re.escape(verb) for verb in _SPEECH_VERBS), key=len, reverse=True))
_INLINE_SPEAKER_RE = re.compile(
    rf"^\s*(?:[-*+]\s+)?(?:\[(?P<bracket>{_SPEAKER_CAPTURE_PATTERN})\]|\((?P<paren>{_SPEAKER_CAPTURE_PATTERN})\)|(?P<name>{_SPEAKER_CAPTURE_PATTERN}))\s*(?:[:\-–—]\s+)(?P<content>.+?)\s*$"
)
_HEADING_SPEAKER_RE = re.compile(
    rf"^\s*(?:[-*+]\s+)?(?:\[(?P<bracket>{_SPEAKER_CAPTURE_PATTERN})\]|\((?P<paren>{_SPEAKER_CAPTURE_PATTERN})\)|(?P<name>[A-Z][A-Z0-9 .'/&-]{{0,48}}(?:\s*\([^)]+\))?))\s*$"
)
_NARRATIVE_DIALOGUE_RE = re.compile(
    rf"^\s*(?P<name>{_NAME_PATTERN})(?:\s+\w+){{0,3}}\s+(?P<verb>{_SPEECH_VERB_PATTERN})\s*[:\-–—]\s*(?P<content>.+?)\s*$",
    re.IGNORECASE,
)
_NAME_VERB_RE = re.compile(
    rf"(?P<name>{_ATTRIBUTION_ENTITY_PATTERN})(?:\s+\w+){{0,4}}\s+(?P<verb>{_SPEECH_VERB_PATTERN})\b",
    re.IGNORECASE,
)
_VERB_NAME_RE = re.compile(
    rf"(?P<verb>{_SPEECH_VERB_PATTERN})(?:\s+\w+){{0,2}}\s+(?P<name>{_ATTRIBUTION_ENTITY_PATTERN})\b",
    re.IGNORECASE,
)
_PROPER_NAME_RE = re.compile(rf"\b{_NAME_PATTERN}\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z']+")
_DIALOGUE_DASH_RE = re.compile(r"^\s*[—–]\s*(?P<content>.+?)\s*$")
_NON_SPEAKER_LABELS = {"scene", "chapter", "title", "summary", "transcript"}
_STOPWORD_LIKE_NAMES = {"speaker", "voice", "scene", "chapter", "title", "note"}
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
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
        raw_lines = text.splitlines()
        segments: list[TextSegment] = []
        pending_unlabeled: list[str] = []
        pending_start_line: int | None = None
        index = 0
        line_index = 0

        def flush_unlabeled() -> None:
            nonlocal index, pending_unlabeled, pending_start_line
            if not pending_unlabeled:
                pending_start_line = None
                return
            for chunk, explicit_speaker, source_line in self._segments_from_unlabeled_block(
                pending_unlabeled,
                pending_start_line,
            ):
                segments.append(
                    TextSegment(
                        index=index,
                        text=chunk,
                        explicit_speaker=explicit_speaker,
                        source_line=source_line,
                    )
                )
                index += 1
            pending_unlabeled = []
            pending_start_line = None

        while line_index < len(raw_lines):
            stripped = raw_lines[line_index].strip()
            if not stripped:
                flush_unlabeled()
                line_index += 1
                continue

            introduced = self._parse_narrative_intro_dialogue(stripped)
            if introduced:
                flush_unlabeled()
                speaker, content = introduced
                collected = [content]
                start_line = line_index + 1
                line_index += 1
                while line_index < len(raw_lines):
                    continuation = raw_lines[line_index].strip()
                    if not continuation:
                        break
                    if (
                        self._parse_narrative_intro_dialogue(continuation)
                        or self._parse_inline_speaker(continuation)
                        or self._parse_heading_speaker(continuation)
                    ):
                        break
                    collected.append(continuation)
                    line_index += 1
                segments.append(
                    TextSegment(
                        index=index,
                        text=self._join_lines(collected),
                        explicit_speaker=speaker,
                        source_line=start_line,
                    )
                )
                index += 1
                continue

            inline = self._parse_inline_speaker(stripped)
            if inline:
                flush_unlabeled()
                speaker, content = inline
                collected = [content]
                start_line = line_index + 1
                line_index += 1
                while line_index < len(raw_lines):
                    continuation = raw_lines[line_index].strip()
                    if not continuation:
                        break
                    if self._parse_inline_speaker(continuation) or self._parse_heading_speaker(continuation):
                        break
                    collected.append(continuation)
                    line_index += 1
                segments.append(
                    TextSegment(
                        index=index,
                        text=self._join_lines(collected),
                        explicit_speaker=speaker,
                        source_line=start_line,
                    )
                )
                index += 1
                continue

            heading = self._parse_heading_speaker(stripped)
            if heading:
                flush_unlabeled()
                speaker = heading
                start_line = line_index + 1
                collected: list[str] = []
                line_index += 1
                while line_index < len(raw_lines):
                    continuation = raw_lines[line_index].strip()
                    if not continuation:
                        if collected:
                            break
                        line_index += 1
                        continue
                    if self._parse_inline_speaker(continuation) or self._parse_heading_speaker(continuation):
                        break
                    collected.append(continuation)
                    line_index += 1
                if collected:
                    segments.append(
                        TextSegment(
                            index=index,
                            text=self._join_lines(collected),
                            explicit_speaker=speaker,
                            source_line=start_line,
                        )
                    )
                    index += 1
                    continue

                if pending_start_line is None:
                    pending_start_line = start_line
                pending_unlabeled.append(stripped)
                continue

            if pending_start_line is None:
                pending_start_line = line_index + 1
            pending_unlabeled.append(stripped)
            line_index += 1

        flush_unlabeled()
        return segments

    def _segments_from_unlabeled_block(
        self,
        lines: list[str],
        start_line: int | None,
    ) -> list[tuple[str, str | None, int | None]]:
        cleaned = [line.strip() for line in lines if line.strip()]
        if not cleaned:
            return []

        dashed_segments = self._extract_dialogue_dash_segments(cleaned, start_line)
        if dashed_segments:
            return dashed_segments

        quoted_segments = self._extract_quoted_dialogue("\n".join(cleaned), start_line)
        if quoted_segments:
            return quoted_segments

        if len(cleaned) >= 2 and all(len(line.split()) <= 24 for line in cleaned):
            return [(line, None, start_line) for line in cleaned]

        block = self._join_lines(cleaned)
        sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(block) if sentence.strip()]
        return [(sentence, None, start_line) for sentence in (sentences or [block])]

    def _extract_quoted_dialogue(
        self,
        block_text: str,
        start_line: int | None,
    ) -> list[tuple[str, str | None, int | None]]:
        normalized = self._normalize_quotes(block_text)
        spans = self._iter_quote_spans(normalized)
        if not spans:
            return []

        extracted: list[tuple[str, str | None, int | None]] = []
        previous_named_speaker: str | None = None
        previous_span: tuple[int, int, str] | None = None
        for span_start, span_end, raw_content in spans:
            content = self._clean_quote_content(raw_content)
            if not content:
                previous_span = (span_start, span_end, raw_content)
                continue

            before_clause = self._nearest_clause_before(normalized, span_start)
            after_clause = self._nearest_clause_after(normalized, span_end)
            if not self._looks_like_spoken_quote(content, before_clause, after_clause):
                previous_span = (span_start, span_end, raw_content)
                continue

            speaker = self._infer_quote_speaker(before_clause, after_clause, previous_named_speaker)
            if speaker:
                previous_named_speaker = speaker

            source_line = self._source_line_for_offset(start_line, normalized, span_start)

            if extracted and previous_span is not None:
                bridge = normalized[previous_span[1] : span_start]
                last_text, last_speaker, last_line = extracted[-1]
                if (
                    last_speaker
                    and speaker
                    and self._normalize_speaker_name(last_speaker) == self._normalize_speaker_name(speaker)
                    and self._is_attribution_bridge(bridge)
                ):
                    extracted[-1] = (self._join_lines([last_text, content]), last_speaker, last_line)
                    previous_span = (span_start, span_end, raw_content)
                    continue

            extracted.append((content, speaker, source_line))
            previous_span = (span_start, span_end, raw_content)

        if not extracted:
            return []
        if len(extracted) == 1 and extracted[0][1] is None and not self._contains_speech_verb(normalized):
            return []
        return extracted

    def _extract_dialogue_dash_segments(
        self,
        lines: list[str],
        start_line: int | None,
    ) -> list[tuple[str, str | None, int | None]]:
        if not lines:
            return []
        dash_indices = [index for index, line in enumerate(lines) if _DIALOGUE_DASH_RE.match(line)]
        if len(dash_indices) < 2 or len(dash_indices) < max(2, len(lines) // 2):
            return []

        extracted: list[tuple[str, str | None, int | None]] = []
        previous_named_speaker: str | None = None
        for line_index, line in enumerate(lines):
            match = _DIALOGUE_DASH_RE.match(line)
            if match:
                content = match.group("content").strip()
                dialogue, speaker = self._split_trailing_attribution(content, previous_named_speaker)
                if speaker:
                    previous_named_speaker = speaker
                extracted.append(
                    (
                        self._clean_quote_content(dialogue),
                        speaker,
                        start_line + line_index if start_line is not None else start_line,
                    )
                )
                continue
            if not extracted:
                return []
            last_text, last_speaker, last_line = extracted[-1]
            extracted[-1] = (f"{last_text} {line.strip()}".strip(), last_speaker, last_line)
        return [(text, speaker, line_number) for text, speaker, line_number in extracted if text]

    def _parse_inline_speaker(self, line: str) -> tuple[str, str] | None:
        match = _INLINE_SPEAKER_RE.match(line)
        if not match:
            return None
        speaker = next((value for key, value in match.groupdict().items() if key != "content" and value), "")
        content = (match.group("content") or "").strip()
        speaker = self._clean_speaker_label(speaker)
        if not content or not self._is_probable_speaker_label(speaker, heading=False):
            return None
        return speaker, content

    def _parse_heading_speaker(self, line: str) -> str | None:
        match = _HEADING_SPEAKER_RE.match(line)
        if not match:
            return None
        speaker = next((value for value in match.groupdict().values() if value), "")
        speaker = self._clean_speaker_label(speaker)
        if not self._is_probable_speaker_label(speaker, heading=True):
            return None
        return speaker

    def _parse_narrative_intro_dialogue(self, line: str) -> tuple[str, str] | None:
        match = _NARRATIVE_DIALOGUE_RE.match(line)
        if not match:
            return None
        speaker = self._clean_speaker_label(match.group("name") or "")
        content = (match.group("content") or "").strip()
        if not content or not self._is_probable_speaker_label(speaker, heading=False):
            return None
        return speaker, content

    def _is_probable_speaker_label(self, label: str, *, heading: bool) -> bool:
        cleaned = re.sub(r"\s+", " ", label).strip(" -–—:").strip()
        if not cleaned:
            return False
        lowered = cleaned.lower()
        if lowered in _NON_SPEAKER_LABELS:
            return False
        parts = [part for part in cleaned.split() if part]
        if len(parts) > 5:
            return False
        if heading:
            return lowered.startswith("speaker ") or cleaned.isupper() or all(part[:1].isupper() for part in parts)
        return True

    def _normalize_quotes(self, text: str) -> str:
        return text.translate(_QUOTE_TRANSLATION)

    def _clean_speaker_label(self, label: str) -> str:
        cleaned = re.sub(r"\s*\((?:o\.s\.|v\.o\.|off|offscreen|whispering|quietly|softly|aside|beat)\)\s*$", "", label, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _iter_quote_spans(self, text: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        open_index: int | None = None
        for index, char in enumerate(text):
            if char != '"':
                continue
            if open_index is None:
                open_index = index
                continue
            spans.append((open_index, index, text[open_index + 1 : index]))
            open_index = None
        if open_index is not None:
            trailing = text[open_index + 1 :]
            if self._looks_like_unclosed_quote(trailing):
                spans.append((open_index, len(text), trailing))
        return spans

    def _looks_like_unclosed_quote(self, trailing: str) -> bool:
        cleaned = trailing.strip()
        if not cleaned:
            return False
        return len(_WORD_RE.findall(cleaned)) >= 3

    def _clean_quote_content(self, content: str) -> str:
        cleaned = re.sub(r"\s+", " ", content.strip())
        return cleaned.strip(" -")

    def _nearest_clause_before(self, text: str, start: int) -> str:
        window = text[max(0, start - 160) : start]
        parts = [part.strip() for part in _CLAUSE_SPLIT_RE.split(window) if part.strip()]
        return parts[-1] if parts else window.strip()

    def _nearest_clause_after(self, text: str, end: int) -> str:
        window = text[end : min(len(text), end + 160)]
        parts = [part.strip() for part in _CLAUSE_SPLIT_RE.split(window) if part.strip()]
        return parts[0] if parts else window.strip()

    def _contains_speech_verb(self, text: str) -> bool:
        if not text.strip():
            return False
        return bool(_NAME_VERB_RE.search(text) or _VERB_NAME_RE.search(text))

    def _looks_like_spoken_quote(self, content: str, before_clause: str, after_clause: str) -> bool:
        token_count = len(_WORD_RE.findall(content))
        if token_count >= 2:
            return True
        if any(char in content for char in ".?!,"):
            return True
        return self._contains_speech_verb(before_clause) or self._contains_speech_verb(after_clause)

    def _infer_quote_speaker(
        self,
        before_clause: str,
        after_clause: str,
        previous_named_speaker: str | None,
    ) -> str | None:
        for clause in (before_clause, after_clause):
            speaker = self._speaker_from_attribution_clause(clause, previous_named_speaker)
            if speaker:
                return speaker
        return self._speaker_from_narrative_subject(before_clause)

    def _split_trailing_attribution(
        self,
        text: str,
        previous_named_speaker: str | None,
    ) -> tuple[str, str | None]:
        parts = [part.strip() for part in re.split(r"(?<=[,;])\s+", text) if part.strip()]
        if len(parts) >= 2:
            trailing = parts[-1]
            speaker = self._speaker_from_attribution_clause(trailing, previous_named_speaker)
            if speaker:
                dialogue = " ".join(parts[:-1]).strip()
                return dialogue or text, speaker
        return text, None

    def _speaker_from_attribution_clause(self, clause: str, previous_named_speaker: str | None) -> str | None:
        if not clause:
            return None
        for pattern in (_NAME_VERB_RE, _VERB_NAME_RE):
            matches = list(pattern.finditer(clause))
            if not matches:
                continue
            candidate = matches[-1].group("name").strip()
            resolved = self._resolve_attribution_entity(candidate, previous_named_speaker)
            if resolved:
                return resolved
        return None

    def _resolve_attribution_entity(self, entity: str, previous_named_speaker: str | None) -> str | None:
        cleaned = re.sub(r"\s+", " ", entity).strip(" ,;:.!?-")
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered in {"he", "she", "they", "we", "i"}:
            return previous_named_speaker
        if self._normalize_speaker_name(cleaned) in _STOPWORD_LIKE_NAMES:
            return None
        return cleaned

    def _speaker_from_narrative_subject(self, clause: str) -> str | None:
        if not clause:
            return None
        names = []
        seen: set[str] = set()
        for candidate in _PROPER_NAME_RE.findall(clause):
            normalized = self._normalize_speaker_name(candidate)
            if not normalized or normalized in seen or normalized in _STOPWORD_LIKE_NAMES:
                continue
            seen.add(normalized)
            names.append(candidate.strip())
        if len(names) == 1:
            return names[0]
        return None

    def _normalize_speaker_name(self, name: str | None) -> str:
        if not name:
            return ""
        cleaned = re.sub(r"[\[\]():\-–—]+", " ", name).strip().lower()
        return re.sub(r"\s+", " ", cleaned)

    def _source_line_for_offset(self, start_line: int | None, text: str, offset: int) -> int | None:
        if start_line is None:
            return None
        return start_line + text[:offset].count("\n")

    def _is_attribution_bridge(self, text: str) -> bool:
        stripped = text.strip(" ,;:-")
        if not stripped:
            return True
        tokens = _WORD_RE.findall(stripped)
        if len(tokens) > 10:
            return False
        return self._contains_speech_verb(stripped)

    def _split_unlabeled_block(self, lines: list[str]) -> list[str]:
        cleaned = [line.strip() for line in lines if line.strip()]
        if not cleaned:
            return []
        if len(cleaned) >= 2 and all(len(line.split()) <= 24 for line in cleaned):
            return cleaned
        block = self._join_lines(cleaned)
        sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(block) if sentence.strip()]
        return sentences or [block]

    def _join_lines(self, lines: list[str]) -> str:
        if not lines:
            return ""
        joined = lines[0].strip()
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            if joined.endswith((",", ";", ":", "-", "—", "–")):
                joined = f"{joined} {stripped}".strip()
            elif joined.endswith((".", "!", "?", '"', "'", "”", "’")):
                joined = f"{joined}\n{stripped}"
            else:
                joined = f"{joined} {stripped}"
        return joined.strip()


def load_document(path: str | Path) -> IngestedDocument:
    return TextIngestor().ingest(path)


def ingest_text_file(path: str | Path) -> IngestedDocument:
    return load_document(path)


parse_input_file = ingest_text_file
ingest_file = ingest_text_file
read_input_file = ingest_text_file
