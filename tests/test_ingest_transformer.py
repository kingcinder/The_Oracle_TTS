"""Tests for the ingestion transformer (format analysis + auto-fix)."""

from __future__ import annotations

from pathlib import Path

import pytest

from the_oracle.ingest_transformer import (
    analyze_input_file,
    analyze_text,
    fix_input_file,
    transform_text,
)
from the_oracle.text_ingest import TextIngestor


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "A - Hello there.\nB - Hi back.\n",                      # dash separator
        "Speaker A \u2013 The machine hums.\nSpeaker B \u2013 It does.\n",  # en dash
        "Alice | The gate opened.\nBob | And closed.\n",         # pipe separator
        "[A]: One.\n[B]: Two.\n",                                # bracketed label
        "[Speaker A] One.\n[Speaker B] Two.\n",                  # bracketed, no colon
        "- A: First.\n- B: Second.\n",                           # bulleted turns
        "> A: He said.\n> B: She said.\n",                       # quote-marked turns
        "A. Yes.\nB. No.\n",                                     # period separator
        "A:\nHello there.\nB:\nGoodbye.\n",                      # orphan labels
        "[2024-01-01 10:00] Alice: hi\n[2024-01-01 10:01] Bob: hello\n",  # chat export
    ],
)
def test_detects_fixable_noncanonical_formats(text: str) -> None:
    issues = analyze_text(text)
    assert issues, f"no issues detected for {text!r}"
    assert all(issue.fixable for issue in issues)
    # Analyze and transform agree on the fixable count.
    _fixed, fix_count = transform_text(text)
    assert fix_count == len(issues)


def test_clean_file_has_no_issues() -> None:
    assert analyze_text("A: Hello there.\nB: Hi back.\n") == []
    assert analyze_text("Plain prose paragraph with no markers at all.\n") == []


def test_prose_colon_labels_are_not_flagged() -> None:
    assert analyze_text("Note: this is prose.\nSee: appendix.\n") == []


# ----------------------------------------------------------------------------
# Prose protection (the transformer must never corrupt narration)
# ----------------------------------------------------------------------------

def test_prose_em_dash_not_rewritten() -> None:
    text = "The storm \u2014 heavy and gray \u2014 rolled in over the hills, and everyone stayed inside.\n"
    assert analyze_text(text) == []
    fixed, count = transform_text(text)
    assert fixed == text
    assert count == 0


def test_prose_with_known_label_still_protected_by_spaced_dash() -> None:
    # Even a label the document uses elsewhere must not swallow a line whose
    # "text" contains another spaced dash (parenthetical prose).
    text = "Alice: The gate opened.\nThe storm \u2014 heavy and gray \u2014 rolled in.\n"
    fixed, count = transform_text(text)
    assert fixed == text  # the storm line is prose; the Alice line is canonical
    assert count == 0


# ----------------------------------------------------------------------------
# Transformation
# ----------------------------------------------------------------------------

def test_transform_produces_canonical_lines() -> None:
    text = "A - Hello there.\nSpeaker B \u2013 Hi back.\n[A]: One.\n- B: Two.\n"
    fixed, count = transform_text(text)
    assert count == 4
    assert fixed == "A: Hello there.\nSpeaker B: Hi back.\nA: One.\nB: Two.\n"


def test_transform_is_idempotent() -> None:
    text = "A - Hello.\n[A]: One.\n- B: Two.\nA:\nOrphan.\n[2024-01-01 10:00] Alice: hi\n"
    fixed, first = transform_text(text)
    again, second = transform_text(fixed)
    assert first > 0
    assert second == 0
    assert again == fixed


def test_transform_join_orphan_label_and_next_line() -> None:
    fixed, count = transform_text("A:\nHello there.\nB:\nGoodbye.\n")
    assert count == 2
    assert fixed == "A: Hello there.\nB: Goodbye.\n"


@pytest.mark.parametrize(
    "text,expected_speakers",
    [
        ("A - Hello there.\nB - Hi back.\n", ["A", "B"]),
        ("[A]: One.\n[B]: Two.\n", ["A", "B"]),
        ("- A: First.\n- B: Second.\n", ["A", "B"]),
        ("A:\nOrphan line.\n", ["A"]),
        ("[2024-01-01 10:00] Alice: hi\n", ["Alice"]),
    ],
)
def test_fixed_text_is_attributed_by_real_ingester(text: str, expected_speakers: list[str]) -> None:
    fixed, _ = transform_text(text)
    segments = TextIngestor()._segment_text(fixed)
    assert [s.explicit_speaker for s in segments] == expected_speakers


# ----------------------------------------------------------------------------
# File-level analysis and fix
# ----------------------------------------------------------------------------

def test_fix_input_file_writes_backup_and_utf8(tmp_path) -> None:
    target = tmp_path / "script.txt"
    target.write_text("A - Hello there.\nB - Hi back.\n", encoding="utf-8")

    analysis = analyze_input_file(target)
    assert len(analysis.fixable_issues) == 2

    written_path, count, backup_path = fix_input_file(target)
    assert count == 2
    assert written_path == target  # plain text files are fixed in place
    assert target.read_text(encoding="utf-8") == "A: Hello there.\nB: Hi back.\n"
    assert backup_path is not None
    assert (tmp_path / backup_path.split("/")[-1]).read_text(encoding="utf-8") == (
        "A - Hello there.\nB - Hi back.\n"
    )


def test_fix_input_file_refuses_healthy_file(tmp_path) -> None:
    target = tmp_path / "clean.txt"
    target.write_text("A: Hello there.\nB: Hi back.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fix_input_file(target)


def test_analyze_input_file_detects_utf16(tmp_path) -> None:
    target = tmp_path / "utf16.txt"
    target.write_bytes("A: Hello there.\nB: Hi back.\n".encode("utf-16"))
    analysis = analyze_input_file(target)
    assert analysis.issues
    assert analysis.issues[0].line_number == 0  # file-level
    assert analysis.issues[0].fixable
    assert analysis.encoding_fixed_text == "A: Hello there.\nB: Hi back.\n"


def test_fix_input_file_transcodes_to_utf8(tmp_path) -> None:
    target = tmp_path / "legacy.txt"
    target.write_bytes("A: Caf\u00e9 r\u00e9sum\u00e9.\n".encode("utf-16"))
    _fixed, count, _backup = fix_input_file(target)
    assert count == 1  # the encoding fix
    raw = target.read_bytes()
    assert not raw.startswith(b"\xff\xfe")
    assert "Caf\u00e9" in raw.decode("utf-8")


# ----------------------------------------------------------------------------
# Real repo input files stay clean (no false positives on canonical scripts)
# ----------------------------------------------------------------------------

def test_bundled_inputs_are_issue_free() -> None:
    from pathlib import Path

    for name in ("What is, reality.txt", "cli_short.txt", "test.txt"):
        path = Path("Input") / name
        if not path.is_file():
            continue
        assert analyze_input_file(path).issues == [], f"false positives in {name}"


# ---------------------------------------------------------------------------
# Per-fix rule provenance (LineFix) and labeled diffs
# ---------------------------------------------------------------------------


def test_transform_text_detailed_reports_rule_per_line() -> None:
    from the_oracle.ingest_transformer import transform_text_detailed

    text = (
        "[Speaker A]: Hello there.\n"
        "[2024-01-01 10:00] Julia: Do they suspect anything?\n"
        "A. Period style line.\n"
        "B:\n"
        "Hi back.\n"
    )
    fixed, fixes = transform_text_detailed(text)
    rules = {fix.rule for fix in fixes}
    assert rules == {"bracket", "timestamp", "period", "orphan"}
    # Output line numbers point at the rewritten lines in `fixed`.
    for fix in fixes:
        assert 1 <= fix.output_line <= len(fixed.splitlines())
        assert fix.original  # the source text is recorded


def test_transform_text_detailed_dash_and_bullet_rules() -> None:
    from the_oracle.ingest_transformer import transform_text_detailed

    # "- A - Hello there." resolves through the dash rule (the bullet is
    # stripped first, then the dash branch matches); a bullet over an
    # already-canonical line is a pure bullet fix.
    fixed, fixes = transform_text_detailed("- A - Hello there.\n- B: Hi back.\n")
    rules = sorted(fix.rule for fix in fixes)
    assert rules == ["bullet", "dash"]
    # Plain transform still returns a count equal to the detailed fixes.
    count = transform_text("- A - Hello there.\n- B: Hi back.\n")[1]
    assert count == 2


def test_labeled_diff_labels_each_rule_exactly() -> None:
    from the_oracle.ingest_transformer import labeled_fixed_diff, transform_text_detailed

    original = "[A]: Greetings.\nA - Hello there.\n"
    fixed, fixes = transform_text_detailed(original)
    labeled = labeled_fixed_diff(original, fixed, fixes)
    assert "+ [bracketed label] A: Greetings." in labeled
    assert "+ [dash/pipe separator] A: Hello there." in labeled
    assert "-[A]: Greetings." in labeled
    assert "-A - Hello there." in labeled


def test_labeled_diff_identical_lines_labeled_individually() -> None:
    """Two identical source lines fixed by different rules get their own
    labels — attribution is positional, not text-matched."""
    from the_oracle.ingest_transformer import labeled_fixed_diff, transform_text_detailed

    original = "[A]: Hello.\n[B]: Hello.\n"
    fixed, fixes = transform_text_detailed(original)
    assert [fix.rule for fix in fixes] == ["bracket", "bracket"]
    labeled = labeled_fixed_diff(original, fixed, fixes)
    labeled_lines = [line for line in labeled.splitlines() if line.startswith("+ [")]
    assert len(labeled_lines) == 2


def test_rule_label_falls_back_to_raw_name() -> None:
    from the_oracle.ingest_transformer import rule_label

    assert rule_label("dash") == "dash/pipe separator"
    assert rule_label("mystery") == "mystery"


# ---------------------------------------------------------------------------
# SRT-aware transformer: analyze/fix a subtitle file directly
# ---------------------------------------------------------------------------

_SRT_SAMPLE = """1
00:00:01,000 --> 00:00:04,000
<i>Winston:</i> The plans are ready.

2
00:00:04,500 --> 00:00:06,000
Do they suspect anything?

3
00:00:06,200 --> 00:00:09,000
Julia: Agreed.
"""


def test_analyze_input_file_flags_valid_srt(tmp_path: Path) -> None:
    from the_oracle.ingest_transformer import analyze_input_file

    target = tmp_path / "movie.srt"
    target.write_text(_SRT_SAMPLE, encoding="utf-8")
    analysis = analyze_input_file(target)
    assert analysis.has_issues
    fixable = analysis.fixable_issues
    assert len(fixable) == 1
    assert "SubRip" in fixable[0].description
    assert fixable[0].fix_description.startswith("Convert the subtitles")


def test_analyze_input_file_ignores_non_srt_text(tmp_path: Path) -> None:
    from the_oracle.ingest_transformer import analyze_input_file

    # A .txt file whose content is a dialogue script is not subtitles even
    # if it contains an arrow.
    target = tmp_path / "script.txt"
    target.write_text("A: Hello there.\nB: Hi back.\n", encoding="utf-8")
    assert not analyze_input_file(target).has_issues


def test_fix_input_file_converts_srt_to_script_not_in_place(tmp_path: Path) -> None:
    """The subtitle file itself is never rewritten; the script is sibling .srt.txt."""
    from the_oracle.ingest_transformer import fix_input_file
    from the_oracle.text_ingest import TextIngestor

    target = tmp_path / "movie.srt"
    target.write_text(_SRT_SAMPLE, encoding="utf-8")

    written_path, fix_count, backup_path = fix_input_file(target)

    assert written_path == tmp_path / "movie.srt.txt"
    assert fix_count == 1
    # The subtitle file is untouched; the script exists beside it.
    assert "-->" in target.read_text(encoding="utf-8")
    script = written_path.read_text(encoding="utf-8")
    assert script.startswith("winston: The plans are ready.")
    assert "julia: Agreed." in script
    # The cast attributes through the real ingester.
    document = TextIngestor().ingest(written_path)
    assert {segment.explicit_speaker for segment in document.segments} == {"winston", "julia"}
    # Backup of the original subtitles was still kept.
    assert backup_path is not None and "-->" in Path(backup_path).read_text(encoding="utf-8")


def test_srt_transform_is_idempotent_and_script_is_clean(tmp_path: Path) -> None:
    from the_oracle.ingest_transformer import analyze_input_file, transform_text_detailed

    fixed, fixes = transform_text_detailed(_SRT_SAMPLE)
    assert [fix.rule for fix in fixes] == ["srt"]
    # Re-transforming the script is a no-op.
    _again, refixes = transform_text_detailed(fixed)
    assert refixes == []
    # And analyzing the converted script reports no SRT issue.
    target = tmp_path / "converted.srt.txt"
    target.write_text(fixed, encoding="utf-8")
    assert not analyze_input_file(target).has_issues


def test_batch_folder_includes_srt_files(tmp_path: Path) -> None:
    from the_oracle.ingest_transformer import analyze_folder, preview_folder_fixes

    (tmp_path / "movie.srt").write_text(_SRT_SAMPLE, encoding="utf-8")
    (tmp_path / "clean.txt").write_text("A: fine.\n", encoding="utf-8")

    analyses = analyze_folder(tmp_path)
    assert [Path(a.path).name for a in analyses] == ["movie.srt"]
    fixes, _warnings = preview_folder_fixes(tmp_path)
    assert [fix.path.name for fix in fixes] == ["movie.srt"]
    assert fixes[0].line_fixes[0].rule == "srt"


# ---------------------------------------------------------------------------
# WebVTT through the transformer + batch scan
# ---------------------------------------------------------------------------


def test_analyze_input_file_flags_webvtt(tmp_path) -> None:
    file_path = tmp_path / "episode.vtt"
    file_path.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Winston>The party is tonight.\n",
        encoding="utf-8",
    )
    analysis = analyze_input_file(file_path)
    assert analysis.has_issues
    assert analysis.fixable_issues[0].description.startswith("This file is SubRip/WebVTT subtitles")


def test_fix_input_file_writes_vtt_txt_script(tmp_path) -> None:
    file_path = tmp_path / "episode.vtt"
    file_path.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Winston>The party is tonight.\n",
        encoding="utf-8",
    )
    written_path, _fix_count, _backup = fix_input_file(file_path)
    assert written_path == tmp_path / "episode.vtt.txt"
    assert "winston: The party is tonight." in written_path.read_text(encoding="utf-8")
    # The subtitle file itself is untouched (still has the WEBVTT header).
    assert file_path.read_text(encoding="utf-8").startswith("WEBVTT")


def test_analyze_folder_includes_vtt(tmp_path) -> None:
    from the_oracle.ingest_transformer import apply_folder_fixes, preview_folder_fixes
    (tmp_path / "a.vtt").write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Winston>The party is tonight.\n",
        encoding="utf-8",
    )
    (tmp_path / "clean.txt").write_text("A: already fine.\n", encoding="utf-8")
    fixes, warnings = preview_folder_fixes(tmp_path)
    assert [Path(fix.path).name for fix in fixes] == ["a.vtt"]
    assert warnings == []
    # Rule label mentions subtitles so the preview explains the rewrite.
    written = apply_folder_fixes(fixes)
    assert len(written) == 1
    assert Path(written[0][0]).name == "a.vtt.txt"
