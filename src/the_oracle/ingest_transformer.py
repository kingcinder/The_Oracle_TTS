"""Ingestion transformer: analyze an input script's formatting before use.

The render pipeline's :class:`~the_oracle.text_ingest.TextIngestor` is
deliberately conservative: a ``Name: dialogue`` line is only a speaker turn
when the label looks like a speaker. That correctness has a cost — files
written in *near-miss* formats (``A - dialogue``, ``[A]: dialogue``,
bullet-prefixed turns, chat exports with timestamps, label lines with the
dialogue on the next line, UTF-16 encodings) silently degrade into
narration, and the user only notices after a render.

The transformer closes that gap in two steps:

1. **Analyze** — classify every line that looks like a speaker turn the
   engine will NOT recognize, plus encoding problems, into structured
   issues. Each issue is either *fixable* (the transformer knows a safe
   rewrite) or a plain *warning* (explained to the user, never touched).
2. **Transform** — apply only the safe rewrites, producing the canonical
   ``Label: dialogue`` format the ingester already understands. Fixes are
   line-local, conservative (every candidate label must pass the same
   :func:`canonical_speaker_label` validation the ingester uses), and
   idempotent: transforming an already-fixed file changes nothing.

The GUI surfaces analysis results before Analyze/Render: fixable issues get
a warning popup offering a one-click in-place fix (with a timestamped
backup of the original), and unfixable problems get an explanatory popup.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from the_oracle.speaker_attribution.heuristics import canonical_speaker_label

# A plausible speaker label as it appears at the start of a line. Mirrors the
# ingester's SPEAKER_RE label part so "looks like a speaker to the
# transformer" == "looks like a speaker to the ingester".
_LABEL_PART = r"[A-Za-z][\w .'\-]{0,48}"

# Leading list/quote decoration that hides a speaker line from the ingester
# (e.g. "- A: hello", "> B: hi", "• A: yo").
_BULLET_RE = re.compile(r"^\s*(?:[-*•>+]+|\&gt;+)\s+")

# Dash/pipe/ellipsis used instead of the colon separator: "A - text",
# "Speaker B – text", "Alice | text". The label part is validated separately,
# so prose em-dash sentences never match (their "label" fails the speaker
# check). (Note: this pattern intentionally avoids f-string interpolation so
# the quantifier braces cannot be mis-parsed.)
_DASH_SEP_RE = re.compile(
    "^\\s*(?P<label>[A-Za-z][\\w .'-]{0,48}?)\\s*[-\u2013\u2014|]{1,2}\\s+(?P<text>\\S.*)$"
)

# A trailing period/ellipsis after a short, speaker-like label:
# "A. text", "Speaker B. text", "1. text". Deliberately restricted to short
# labels (single letter/digit or "Speaker X") so ordinary sentences that
# happen to start with a word and a period are never rewritten.
_PERIOD_SEP_RE = re.compile(
    r"^\s*(?P<label>Speaker\s+[A-Za-z0-9]{1,2}|[A-Za-z]|\d{1,2})\.(?:\s+|\.\.\s+)(?P<text>\S.*)$",
    re.IGNORECASE,
)

# Bracketed speaker label: "[A]: text", "[Speaker A] text", "[Alice]: text".
_BRACKETED_LABEL_RE = re.compile(
    rf"^\s*\[\s*(?P<label>{_LABEL_PART})\s*\]\s*:?\s+(?P<text>\S.*)$"
)

# Chat-export timestamp prefix, e.g. "[2024-01-01 10:00] Alice: hi". The
# inside is only dropped when the remainder is itself a well-formed speaker
# turn (validated below), so ordinary bracketed asides are untouched.
_TIMESTAMP_INSIDE_RE = re.compile(r"^\d{1,4}[-/:. ]*\d{0,2}[-/:. ]*\d{0,2}[:\d ]*$")

_BRACKETED_PREFIX_RE = re.compile(r"^\s*\[\s*(?P<inside>[^\[\]]+)\s*\]\s*:?\s*(?P<rest>\S.*)$")

# A label with nothing after the colon; the dialogue lives on the next line:
#   A:
#   Hello there.
_LABEL_ONLY_RE = re.compile(rf"^\s*(?P<label>{_LABEL_PART})\s*:\s*$")

# The canonical form the ingester accepts.
_SPEAKER_RE = re.compile(rf"^\s*(?P<label>{_LABEL_PART})\s*:\s*(?P<text>\S.*)$")


@dataclass(slots=True)
class FormatIssue:
    """One formatting problem found in an input script."""

    line_number: int  # 1-based, 0 for file-level (encoding) issues
    description: str
    snippet: str
    fixable: bool = False
    fix_description: str = ""


@dataclass(slots=True)
class FileAnalysis:
    """Result of analyzing one input file."""

    path: str
    issues: list[FormatIssue] = field(default_factory=list)
    encoding_fixed_text: str | None = None  # set when the file needs transcoding

    @property
    def fixable_issues(self) -> list[FormatIssue]:
        return [issue for issue in self.issues if issue.fixable]

    @property
    def warning_issues(self) -> list[FormatIssue]:
        return [issue for issue in self.issues if not issue.fixable]

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)


def _looks_like_speaker(label: str) -> bool:
    """Same test the ingester applies: does this label name a speaker?"""
    return canonical_speaker_label(label) is not None


# A spaced dash inside the would-be dialogue text. A dash *separator* is
# followed by clean speech; a dash inside the text means the line is prose
# with a parenthetical em-dash ("The storm — heavy and gray — rolled in") and
# must never be rewritten.
_SPACED_DASH_RE = re.compile(r"\s[-\u2013\u2014]\s")


def _is_strong_label(label: str) -> bool:
    """Labels that are unambiguous speaker names on their own: a single
    letter/digit, "Speaker X", or an ALL-CAPS word (screenplay style)."""
    compact = label.strip()
    lowered = compact.lower()
    if re.fullmatch(r"[a-z0-9]{1,2}", lowered):
        return True
    if re.fullmatch(r"speaker\s+\S{1,2}", lowered):
        return True
    return bool(compact) and compact.isupper() and any(c.isalpha() for c in compact)


def _dash_separator_plausible(label: str, text: str, known_labels: set[str], dash_document: bool) -> bool:
    """Decide whether a dash/pipe match is really a speaker-turn separator.

    Requires the extracted text to be free of further spaced dashes (a
    parenthetical em-dash keeps going) and the label to be backed by
    document-level evidence: an unambiguous strong label, the same label in
    a canonical colon line, or a document that is predominantly dash-form
    dialogue.
    """
    if _SPACED_DASH_RE.search(text):
        return False
    compact = label.strip()
    if _is_strong_label(compact) or compact in known_labels or dash_document:
        return True
    return False


def _known_canonical_labels(lines: list[str]) -> set[str]:
    known: set[str] = set()
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        marker = _SPEAKER_RE.match(stripped)
        if marker and _looks_like_speaker(marker.group("label")):
            known.add(marker.group("label").strip())
    return known


def _is_dash_dialogue_document(lines: list[str]) -> bool:
    """True when most non-empty lines are dash/pipe-form speaker turns —
    a document-level signal that dashes are the author's separator of choice."""
    non_empty = 0
    dash_turns = 0
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        non_empty += 1
        dash = _DASH_SEP_RE.match(stripped)
        if dash:
            text = dash.group("text").strip()
            if _looks_like_speaker(dash.group("label").strip()) and not _SPACED_DASH_RE.search(text):
                dash_turns += 1
    return non_empty >= 2 and dash_turns >= 2 and dash_turns * 2 >= non_empty


def _strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line, count=1)


def analyze_text(text: str) -> list[FormatIssue]:
    """Find speaker-turn lines the ingester would misread, in file order."""
    issues: list[FormatIssue] = []
    lines = text.splitlines()
    known_labels = _known_canonical_labels(lines)
    dash_document = _is_dash_dialogue_document(lines)

    for index, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()

        # Already canonical (or screenplay/quoted-speech handled downstream):
        # nothing to report.
        marker = _SPEAKER_RE.match(stripped)
        if marker and _looks_like_speaker(marker.group("label")):
            continue

        candidate = _strip_bullet(line).strip()
        if candidate == stripped:
            candidate = stripped

        fix_description = ""

        if _BULLET_RE.match(line):
            inner = _strip_bullet(line).strip()
            inner_marker = _SPEAKER_RE.match(inner)
            if inner_marker and _looks_like_speaker(inner_marker.group("label")):
                fix_description = f"Remove the list/quote marker so the line reads '{inner}'."

        if not fix_description:
            dash = _DASH_SEP_RE.match(candidate)
            if dash:
                label = dash.group("label").strip()
                text_part = dash.group("text").strip()
                if _looks_like_speaker(label) and _dash_separator_plausible(
                    label, text_part, known_labels, dash_document
                ):
                    fixed = f"{label}: {text_part}"
                    fix_description = (
                        "A dash or pipe is used instead of a colon; rewrite as "
                        f"'{fixed}'."
                    )

        if not fix_description:
            period = _PERIOD_SEP_RE.match(candidate)
            if period and _looks_like_speaker(period.group("label").strip()):
                fixed = f"{period.group('label').strip()}: {period.group('text').strip()}"
                fix_description = f"A period is used after the speaker name; rewrite as '{fixed}'."

        if not fix_description:
            bracketed = _BRACKETED_LABEL_RE.match(candidate)
            if bracketed and _looks_like_speaker(bracketed.group("label").strip()):
                fixed = f"{bracketed.group('label').strip()}: {bracketed.group('text').strip()}"
                fix_description = f"Remove the brackets; rewrite as '{fixed}'."

        if not fix_description:
            prefixed = _BRACKETED_PREFIX_RE.match(candidate)
            if prefixed and _TIMESTAMP_INSIDE_RE.match(prefixed.group("inside").strip()):
                rest = prefixed.group("rest").strip()
                rest_marker = _SPEAKER_RE.match(rest)
                if rest_marker and _looks_like_speaker(rest_marker.group("label")):
                    fix_description = (
                        "A chat-export timestamp prefixes the line; drop it and keep "
                        f"'{rest}'."
                    )

        if not fix_description:
            orphan = _LABEL_ONLY_RE.match(candidate)
            if orphan and _looks_like_speaker(orphan.group("label").strip()):
                next_line = _next_content_line(lines, index)
                if next_line is not None and not _SPEAKER_RE.match(next_line):
                    fixed = f"{orphan.group('label').strip()}: {next_line}"
                    fix_description = (
                        "The speaker label has no dialogue on its line; join it with "
                        f"the next line as '{fixed}'."
                    )

        if fix_description:
            issues.append(
                FormatIssue(
                    line_number=index,
                    description="Speaker turn in a format the engine will not attribute.",
                    snippet=stripped[:120],
                    fixable=True,
                    fix_description=fix_description,
                )
            )
            continue

        # Not auto-fixable: a line that strongly resembles a dialogue turn but
        # whose label fails speaker validation would be read as narration.
        # Only flag it when the document otherwise looks like labelled
        # dialogue, so prose with an occasional "Note:" line stays quiet.
        if marker and not _looks_like_speaker(marker.group("label")):
            if _document_has_valid_markers(lines) and _label_resembles_name(marker.group("label")):
                issues.append(
                    FormatIssue(
                        line_number=index,
                        description=(
                            f"'{marker.group('label').strip()}' does not look like a "
                            "speaker label the engine accepts, so this line will be "
                            "read as narration."
                        ),
                        snippet=stripped[:120],
                        fixable=False,
                    )
                )

    return issues


def _next_content_line(lines: list[str], from_index_1based: int) -> str | None:
    for raw in lines[from_index_1based:]:
        stripped = raw.strip()
        if stripped:
            return stripped
        # A blank line right after the label breaks the visual pairing; treat
        # it as "no dialogue follows".
        return None
    return None


def _document_has_valid_markers(lines: list[str]) -> bool:
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        marker = _SPEAKER_RE.match(stripped)
        if marker and _looks_like_speaker(marker.group("label")):
            return True
    return False


def _label_resembles_name(label: str) -> bool:
    """Loose test: could this rejected label have been meant as a speaker?"""
    compact = label.strip()
    if not compact or len(compact) > 40:
        return False
    # Reject the well-known prose colon-labels ("Note", "See", ...) — those
    # are intentional narration, not misformats.
    return canonical_speaker_label(compact) is None and compact[:1].isalpha()


def _detect_encoding(raw: bytes) -> str | None:
    """Return the encoding to transcode from, or None if already plain UTF-8."""
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"  # normalize a BOM away
    if b"\x00" in raw[:4096]:
        return "utf-16-le"  # BOM-less UTF-16 (NUL bytes are its fingerprint)
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return "cp1252"
    return None


def analyze_input_file(path: str | Path) -> FileAnalysis:
    """Analyze a file on disk: encoding health + per-line format issues."""
    file_path = Path(path)
    analysis = FileAnalysis(path=str(file_path))
    raw = file_path.read_bytes()

    encoding = _detect_encoding(raw)
    if encoding is not None:
        try:
            decoded = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            decoded = None
        if decoded is not None:
            analysis.encoding_fixed_text = decoded
            analysis.issues.append(
                FormatIssue(
                    line_number=0,
                    description=(
                        f"The file is not saved as plain UTF-8 text (detected {encoding}); "
                        "it may read as garbled characters."
                    ),
                    snippet=file_path.name,
                    fixable=True,
                    fix_description="Transcode the file to plain UTF-8.",
                )
            )
        else:
            analysis.issues.append(
                FormatIssue(
                    line_number=0,
                    description="The file's text encoding could not be identified.",
                    snippet=file_path.name,
                    fixable=False,
                )
            )

    if analysis.encoding_fixed_text is not None:
        text = analysis.encoding_fixed_text
    else:
        text = _decode_best_effort(raw)

    # Subtitle files degrade entirely into narration if ingested directly;
    # detect them structurally (not by extension) and offer a conversion.
    try:
        is_srt = _is_srt_text(text)
    except Exception:
        is_srt = False
    if is_srt:
        analysis.issues.append(
            FormatIssue(
                line_number=0,
                description=(
                    "This file is SubRip subtitles; ingested directly, every cue "
                    "would be read as narration instead of attributed dialogue."
                ),
                snippet=file_path.name,
                fixable=True,
                fix_description="Convert the subtitles into a canonical dialogue script.",
            )
        )
        analysis.issues.sort(key=lambda issue: 0 if issue.line_number == 0 else 1)
        return analysis

    analysis.issues.extend(analyze_text(text))
    return analysis


def _decode_best_effort(raw: bytes) -> str:
    """Mirror the pipeline's resilient decode (UTF-8 with BOM, then CP1252)."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


@dataclass(slots=True)
class LineFix:
    """One rewrite applied by the transform, with the rule that produced it."""

    output_line: int  # 1-based line number in the *fixed* text
    rule: str  # "dash", "bracket", "timestamp", "bullet", "orphan", "period", "encoding"
    original: str  # the source text this rewrite replaced (for diff labeling)


# Human-readable names for the fix rules, shown in the preview dialogs.
RULE_LABELS: dict[str, str] = {
    "dash": "dash/pipe separator",
    "period": "period separator",
    "bracket": "bracketed label",
    "timestamp": "chat-export timestamp",
    "orphan": "orphan label line",
    "bullet": "list/quote marker",
    "encoding": "text encoding",
    "srt": "SRT subtitle",
}


def rule_label(rule: str) -> str:
    """Human-readable name for a fix rule (falls back to the raw name)."""
    return RULE_LABELS.get(rule, rule)


def transform_text(text: str) -> tuple[str, int]:
    """Rewrite every fixable format into canonical ``Label: dialogue`` lines.

    Returns ``(fixed_text, fix_count)``. The transform is idempotent: the
    output of a transform has no remaining fixable issues.
    """
    fixed, fixes = transform_text_detailed(text)
    return fixed, len(fixes)


def _is_srt_text(text: str) -> bool:
    """True when *text* parses as SubRip subtitles.

    The import is lazy so the transformer stays importable in slim contexts
    (and to avoid any import-order coupling with the SRT module).
    """
    from the_oracle.srt_ingest import looks_like_srt

    return looks_like_srt(text)


def transform_text_detailed(text: str) -> tuple[str, list[LineFix]]:
    """Transform with per-fix provenance: which rule rewrote which line.

    Returns ``(fixed_text, fixes)`` where each :class:`LineFix` records the
    1-based line number in the *output* text and the name of the rule that
    produced it. The transform is idempotent: the output of a transform has
    no remaining fixable issues.
    """
    lines = text.splitlines()
    out: list[str] = []
    fixes: list[LineFix] = []
    known_labels = _known_canonical_labels(lines)
    dash_document = _is_dash_dialogue_document(lines)

    def _emit(text_line: str, rule: str | None = None, original: str | None = None) -> None:
        out.append(text_line)
        if rule is not None:
            fixes.append(
                LineFix(output_line=len(out), rule=rule, original=original if original is not None else "")
            )

    # SRT subtitles: the whole document is one structural rewrite — cues,
    # timestamps, and markup become canonical ``Label: dialogue`` lines.
    # Checked first because nothing else in a subtitle file is a text line.
    if _is_srt_text(text):
        from the_oracle.srt_ingest import srt_to_dialogue_text

        script, _cues, _speakers = srt_to_dialogue_text(text)
        if script.strip():
            first_rule = "srt"
            fixes.append(LineFix(output_line=1, rule=first_rule, original=lines[0] if lines else ""))
            return script, fixes

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            out.append("")
            index += 1
            continue

        working = _strip_bullet(line).strip()
        had_bullet = working != stripped

        # Orphan label line: join with the next content line.
        orphan = _LABEL_ONLY_RE.match(working)
        if orphan and _looks_like_speaker(orphan.group("label").strip()):
            next_line = _next_content_line(lines, index + 1)
            if next_line is not None and not _SPEAKER_RE.match(next_line):
                _emit(f"{orphan.group('label').strip()}: {next_line}", "orphan", stripped)
                # Advance past the label line and any blanks, then past the
                # joined content line and any blanks that preceded it.
                index += 1  # the label line itself
                while index < len(lines) and not lines[index].strip():
                    index += 1
                if index < len(lines):
                    index += 1  # the content line that was joined
                continue

        marker = _SPEAKER_RE.match(working)
        if marker and _looks_like_speaker(marker.group("label")):
            _emit(
                f"{marker.group('label').strip()}: {marker.group('text').strip()}",
                "bullet" if had_bullet else None,
                stripped,
            )
            index += 1
            continue

        dash = _DASH_SEP_RE.match(working)
        if dash:
            label = dash.group("label").strip()
            text_part = dash.group("text").strip()
            if _looks_like_speaker(label) and _dash_separator_plausible(
                label, text_part, known_labels, dash_document
            ):
                _emit(f"{label}: {text_part}", "dash", stripped)
                index += 1
                continue

        period = _PERIOD_SEP_RE.match(working)
        if period and _looks_like_speaker(period.group("label").strip()):
            _emit(f"{period.group('label').strip()}: {period.group('text').strip()}", "period", stripped)
            index += 1
            continue

        bracketed = _BRACKETED_LABEL_RE.match(working)
        if bracketed and _looks_like_speaker(bracketed.group("label").strip()):
            _emit(f"{bracketed.group('label').strip()}: {bracketed.group('text').strip()}", "bracket", stripped)
            index += 1
            continue

        prefixed = _BRACKETED_PREFIX_RE.match(working)
        if prefixed and _TIMESTAMP_INSIDE_RE.match(prefixed.group("inside").strip()):
            rest = prefixed.group("rest").strip()
            rest_marker = _SPEAKER_RE.match(rest)
            if rest_marker and _looks_like_speaker(rest_marker.group("label")):
                _emit(f"{rest_marker.group('label').strip()}: {rest_marker.group('text').strip()}", "timestamp", stripped)
                index += 1
                continue

        # Nothing matched: keep the line exactly as written (blank-marker
        # stripping is the only lossless decoration removal).
        if working != stripped and stripped:
            # A bullet was stripped but the remainder is not a speaker turn;
            # keep the original line untouched to stay lossless.
            out.append(line)
        else:
            out.append(line)
        index += 1

    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), fixes


@dataclass(slots=True)
class SpeakerRefSuggestion:
    """One speaker's voice-key mapping and the exact CLI flag to voice it."""

    speaker: str  # canonical speaker label, e.g. "winston"
    voice_key: str  # A..X
    flag: str  # exact flag, e.g. "--speaker-ref C=PATH" or "--speakerA-ref PATH"
    is_new: bool  # True when the key is beyond A/B, i.e. a flag the user must ADD


# Imported lazily at call time to avoid a circular import at module load:
# speaker_attribution.heuristics does not import this module, but keeping the
# heavy numpy-dependent import out of the transformer's import path keeps GUI
# startup unchanged.
def _voice_mapping_for(canonical_labels: list[str]) -> dict[str, str] | None:
    from the_oracle.speaker_attribution.heuristics import DualSpeakerAttributor

    return DualSpeakerAttributor._map_labels_to_voices(canonical_labels)


def suggest_speaker_refs(text: str) -> list[SpeakerRefSuggestion]:
    """Suggest the exact ``--speaker-ref`` flags for a script's cast.

    The cast is read from the *post-fix* text, so a speaker only visible
    after a transform (``Winston - Hello.`` → ``Winston: Hello.``) is
    included. The mapping is computed by the pipeline's own voice mapper
    (first-appearance order onto keys A..X), so the suggestion can never
    drift from what a render will actually do. Flags for A/B use the
    dedicated ``--speakerA-ref``/``--speakerB-ref`` forms; keys C and beyond
    are marked ``is_new`` because those are the flags a user must ADD.
    """
    from the_oracle.speaker_attribution.heuristics import canonical_speaker_label

    fixed_text, _fix_count = transform_text(text)
    seen: list[str] = []
    for raw in fixed_text.splitlines():
        marker = _SPEAKER_RE.match(raw.strip())
        if not marker:
            continue
        canonical = canonical_speaker_label(marker.group("label"))
        if canonical and canonical not in seen:
            seen.append(canonical)
    if not seen:
        return []
    mapping = _voice_mapping_for(seen)
    if not mapping:
        return []
    suggestions: list[SpeakerRefSuggestion] = []
    for label in seen:
        key = mapping.get(label)
        if key is None:
            continue
        if key == "A":
            flag = "--speakerA-ref PATH"
        elif key == "B":
            flag = "--speakerB-ref PATH"
        else:
            flag = f"--speaker-ref {key}=PATH"
        suggestions.append(
            SpeakerRefSuggestion(
                speaker=label,
                voice_key=key,
                flag=flag,
                is_new=key not in ("A", "B"),
            )
        )
    return suggestions


def rejected_labels(text: str) -> list[str]:
    """Distinct flagged labels that fail speaker validation entirely.

    These read as narration no matter what reference audio is supplied, so
    no ``--speaker-ref`` can attribute them — the user must edit the label.
    """
    from the_oracle.speaker_attribution.heuristics import canonical_speaker_label

    seen: list[str] = []
    for raw in text.splitlines():
        marker = _SPEAKER_RE.match(raw.strip())
        if not marker:
            continue
        label = marker.group("label").strip()
        if canonical_speaker_label(label) is None and label not in seen:
            if _label_resembles_name(label):
                seen.append(label)
    return seen


@dataclass(slots=True)
class FolderFix:
    """One fixable file inside a folder batch, with its exact rewrite."""

    path: Path
    original_text: str
    fixed_text: str
    fix_count: int
    line_fixes: list[LineFix] = field(default_factory=list)


# Text extensions the batch scan considers input scripts. ``.srt`` is
# included because subtitles are exactly the kind of file a folder scan
# should catch (the transform converts them to ``.txt`` scripts).
# Timestamped fix backups are excluded by name so re-running the batch
# never re-reads its own backups.
_BATCH_EXTENSIONS = {".txt", ".md", ".srt"}


def analyze_folder(folder: str | Path) -> list[FileAnalysis]:
    """Analyze every text file in *folder* (non-recursive, sorted by name).

    Returns only the analyses that actually have issues, in file order, so a
    clean folder yields an empty list and callers never special-case it.
    Backup files written by previous fixes (``*.bak-*``) are skipped.
    """
    folder_path = Path(folder)
    results: list[FileAnalysis] = []
    for file_path in sorted(folder_path.iterdir(), key=lambda p: p.name):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in _BATCH_EXTENSIONS:
            continue
        if file_path.name.endswith(".bak") or ".bak-" in file_path.name:
            continue
        try:
            analysis = analyze_input_file(file_path)
        except (OSError, ValueError, UnicodeDecodeError):
            continue  # unreadable files are the render path's problem to report
        if analysis.has_issues:
            results.append(analysis)
    return results


def preview_folder_fixes(folder: str | Path) -> tuple[list[FolderFix], list[FileAnalysis]]:
    """Compute rewrites for every fixable file in *folder* without writing.

    Returns ``(fixable, warnings_only)``: the :class:`FolderFix` list for a
    combined preview dialog, plus the analyses whose issues are all warnings
    (reported to the user, never rewritten). Raises :class:`ValueError` when
    the folder contains no fixable files at all.
    """
    analyses = analyze_folder(folder)
    fixable: list[FolderFix] = []
    warnings_only: list[FileAnalysis] = []
    for analysis in analyses:
        if not analysis.fixable_issues:
            warnings_only.append(analysis)
            continue
        original = (
            analysis.encoding_fixed_text
            if analysis.encoding_fixed_text is not None
            else _decode_best_effort(analysis.path and Path(analysis.path).read_bytes())
        )
        fixed_text, line_fixes = transform_text_detailed(original)
        fix_count = len(line_fixes)
        if analysis.encoding_fixed_text is not None:
            fix_count += 1
        fixable.append(
            FolderFix(
                path=Path(analysis.path),
                original_text=original,
                fixed_text=fixed_text,
                fix_count=fix_count,
                line_fixes=line_fixes,
            )
        )
    if not fixable:
        raise ValueError(f"No fixable formatting issues found in {folder}")
    return fixable, warnings_only


def apply_folder_fixes(fixes: list[FolderFix], *, backup: bool = True) -> list[tuple[str, int, str | None]]:
    """Write every precomputed rewrite in *fixes* to disk, in order.

    Each file gets a timestamped backup of its original (unless ``backup``
    is False). Returns one ``(path, fix_count, backup_path)`` triple per
    file, in the same order as *fixes*.
    """
    written: list[tuple[str, int, str | None]] = []
    for fix in fixes:
        backup_path: str | None = None
        if backup:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_file = fix.path.with_name(f"{fix.path.name}.bak-{stamp}")
            backup_file.write_text(fix.original_text, encoding="utf-8")
            backup_path = str(backup_file)
        fix.path.write_text(fix.fixed_text, encoding="utf-8")
        written.append((str(fix.path), fix.fix_count, backup_path))
    return written


def labeled_fixed_diff(original_text: str, fixed_text: str, line_fixes: list[LineFix]) -> str:
    """Build a unified diff with each changed line labeled by its fix rule.

    ``+``/``-`` diff lines produced by a rewrite gain a ``[rule label]``
    prefix (context and file-header lines are untouched). Line attribution
    is exact: the diff's hunk headers are walked to track output line
    numbers rather than guessing from text matching, so two identical lines
    fixed by different rules are still labeled correctly.
    """
    rules_by_output_line = {fix.output_line: fix.rule for fix in line_fixes}
    labeled: list[str] = []
    output_line = 0
    for diff_line in difflib.unified_diff(
        original_text.splitlines(keepends=True),
        fixed_text.splitlines(keepends=True),
        fromfile="original",
        tofile="fixed",
        n=2,
    ):
        if diff_line.startswith("---") or diff_line.startswith("+++"):
            labeled.append(diff_line)
            continue
        if diff_line.startswith("@@"):
            labeled.append(diff_line)
            match = re.search(r"\+(\d+)", diff_line)
            output_line = (int(match.group(1)) - 1) if match else 0
            continue
        marker, _body = diff_line[:1], diff_line[1:]
        if marker == "+":
            output_line += 1
            rule = rules_by_output_line.get(output_line)
            if rule:
                labeled.append(f"+ [{rule_label(rule)}] {_body}")
            else:
                labeled.append(diff_line)
        elif marker == "-":
            labeled.append(diff_line)
        else:
            output_line += 1
            labeled.append(diff_line)
    return "".join(labeled)


@dataclass(slots=True)
class DiffRow:
    """One aligned row of a side-by-side diff.

    ``kind`` is ``"same"``, ``"removed"`` (only left), ``"added"`` (only
    right), or ``"changed"`` (pair: removed line on the left, its rewrite on
    the right). ``rule`` is set on the right-hand cell of changed/added rows
    produced by a rewrite.
    """

    left: str | None
    right: str | None
    kind: str
    rule: str | None = None


def side_by_side_diff_rows(
    original_text: str, fixed_text: str, line_fixes: list[LineFix]
) -> list[DiffRow]:
    """Align original and fixed texts into side-by-side rows.

    Built on SequenceMatcher opcodes (whole-file, no hunk windows), so the
    alignment is exact regardless of script length: equal blocks pair up
    line-for-line, and each rewrite becomes a ``changed`` row pairing the
    original line with the rule-labeled rewrite. Insertions/deletions away
    from any fix (rare, since rewrites are line-local) fall back to
    ``added``/``removed`` rows with no pairing.
    """
    rules_by_output_line = {fix.output_line: fix.rule for fix in line_fixes}
    original_lines = original_text.splitlines()
    fixed_lines = fixed_text.splitlines()
    matcher = difflib.SequenceMatcher(a=original_lines, b=fixed_lines, autojunk=False)
    rows: list[DiffRow] = []

    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(a2 - a1):
                rows.append(
                    DiffRow(
                        left=original_lines[a1 + offset],
                        right=fixed_lines[b1 + offset],
                        kind="same",
                    )
                )
        elif tag == "replace":
            # A rewrite is line-local: pair removed/added lines positionally.
            for offset in range(max(a2 - a1, b2 - b1)):
                left = original_lines[a1 + offset] if a1 + offset < a2 else None
                right = fixed_lines[b1 + offset] if b1 + offset < b2 else None
                if left is not None and right is not None:
                    rule = rules_by_output_line.get(b1 + offset + 1)
                    rows.append(
                        DiffRow(left=left, right=right, kind="changed", rule=rule)
                    )
                elif left is not None:
                    rows.append(DiffRow(left=left, right=None, kind="removed"))
                else:
                    rule = rules_by_output_line.get(b1 + offset + 1)
                    rows.append(
                        DiffRow(left=None, right=right, kind="added", rule=rule)
                    )
        elif tag == "delete":
            for offset in range(a1, a2):
                rows.append(DiffRow(left=original_lines[offset], right=None, kind="removed"))
        elif tag == "insert":
            for offset in range(b1, b2):
                rule = rules_by_output_line.get(offset + 1)
                rows.append(
                    DiffRow(left=None, right=fixed_lines[offset], kind="added", rule=rule)
                )
    return rows


def preview_fixed_text(path: str | Path) -> tuple[str, str, int, list[FormatIssue], list[LineFix]]:
    """Compute the fix for a file without writing anything.

    Returns ``(original_text, fixed_text, fix_count, analysis_issues,
    line_fixes)`` so a caller (e.g. the GUI's preview dialog) can show the
    exact rewrite — each changed line labeled with the rule that produced it
    via :func:`labeled_fixed_diff` — before the user accepts it. Raises
    :class:`ValueError` when the file has no fixable issues.
    """
    file_path = Path(path)
    analysis = analyze_input_file(file_path)
    if not analysis.fixable_issues:
        raise ValueError(f"No fixable formatting issues found in {file_path}")
    original = (
        analysis.encoding_fixed_text
        if analysis.encoding_fixed_text is not None
        else _decode_best_effort(file_path.read_bytes())
    )
    fixed_text, line_fixes = transform_text_detailed(original)
    fix_count = len(line_fixes)
    if analysis.encoding_fixed_text is not None:
        fix_count += 1
    return original, fixed_text, fix_count, analysis.issues, line_fixes


def fix_input_file(path: str | Path, *, backup: bool = True) -> tuple[Path, int, str | None]:
    """Analyze, transform, and rewrite a file in place.

    Returns ``(written_path, fix_count, backup_path)``. A timestamped backup
    of the original is written next to the file unless ``backup`` is False.
    Raises :class:`ValueError` when the file has no fixable issues, so a
    caller can never silently "fix" a healthy file.

    SubRip inputs are the one non-in-place case: the conversion is written
    to a sibling ``<name>.srt.txt`` script (the subtitle file is backed up
    but never modified), so the render pipeline ingests the script. The
    returned path is the converted script in that case, and the original
    file for every in-place rewrite.
    """
    file_path = Path(path)
    analysis = analyze_input_file(file_path)
    fixable = analysis.fixable_issues
    if not fixable:
        raise ValueError(f"No fixable formatting issues found in {file_path}")

    original = (
        analysis.encoding_fixed_text
        if analysis.encoding_fixed_text is not None
        else _decode_best_effort(file_path.read_bytes())
    )
    is_srt = any(issue.fixable and issue.line_number == 0 and issue.snippet == file_path.name and "SubRip" in issue.description for issue in analysis.issues)
    fixed_text, fix_count = transform_text(original)
    if analysis.encoding_fixed_text is not None:
        fix_count += 1  # the transcode itself is a fix the user should hear about

    backup_path: str | None = None
    if backup:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_file = file_path.with_name(f"{file_path.name}.bak-{stamp}")
        backup_file.write_text(original, encoding="utf-8")
        backup_path = str(backup_file)

    if is_srt and file_path.suffix.lower() == ".srt":
        written_path = file_path.with_suffix(".srt.txt")
        written_path.write_text(fixed_text, encoding="utf-8")
        return written_path, fix_count, backup_path

    file_path.write_text(fixed_text, encoding="utf-8")
    return file_path, fix_count, backup_path
