"""Tests for the folder-level batch mode of the ingestion transformer."""

from __future__ import annotations

from pathlib import Path

import pytest

from the_oracle.ingest_transformer import (
    analyze_folder,
    analyze_input_file,
    apply_folder_fixes,
    preview_folder_fixes,
)


def test_analyze_folder_finds_only_problem_files(tmp_path: Path) -> None:
    (tmp_path / "messy.txt").write_text("A - Hello there.\nB - Hi back.\n", encoding="utf-8")
    (tmp_path / "brackets.md").write_text("[A]: Greetings.\n", encoding="utf-8")
    (tmp_path / "clean.txt").write_text("A: All good here.\n", encoding="utf-8")
    (tmp_path / "prose.txt").write_text("Note: this is intentional narration.\n", encoding="utf-8")

    analyses = analyze_folder(tmp_path)

    names = sorted(Path(a.path).name for a in analyses)
    assert names == ["brackets.md", "messy.txt"]
    # Clean and prose files are reported nowhere.
    assert not any("clean" in name or "prose" in name for name in names)


def test_analyze_folder_skips_backups_but_scans_subfolders(tmp_path: Path) -> None:
    """Backups are skipped, subfolders ARE scanned (recursive batch)."""
    (tmp_path / "messy.txt").write_text("A - Hello there.\n", encoding="utf-8")
    (tmp_path / "messy.txt.bak-20260911-000000").write_text("A - old backup.\n", encoding="utf-8")
    (tmp_path / "old.bak").write_text("A - another backup.\n", encoding="utf-8")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("B - scanned now.\n", encoding="utf-8")

    analyses = analyze_folder(tmp_path)

    assert sorted(Path(a.path).name for a in analyses) == ["messy.txt", "nested.txt"]


def test_analyze_folder_skips_hidden_and_vendor_dirs(tmp_path: Path) -> None:
    """Hidden dirs (.git, .venv) and vendor caches are never descended into."""
    (tmp_path / "top.txt").write_text("A - Hello.\n", encoding="utf-8")
    for hidden in (".git", ".venv", "__pycache__", "node_modules"):
        d = tmp_path / hidden
        d.mkdir()
        (d / "junk.txt").write_text("A - should be ignored.\n", encoding="utf-8")

    analyses = analyze_folder(tmp_path)

    assert [Path(a.path).name for a in analyses] == ["top.txt"]


def test_analyze_folder_sorts_by_relative_path(tmp_path: Path) -> None:
    """Deep files come back in stable tree order, not walk order."""
    deep = tmp_path / "b" / "c"
    deep.mkdir(parents=True)
    (tmp_path / "z.txt").write_text("A - z.\n", encoding="utf-8")
    (deep / "a.txt").write_text("B - a.\n", encoding="utf-8")

    analyses = analyze_folder(tmp_path)

    assert [Path(a.path).relative_to(tmp_path).as_posix() for a in analyses] == [
        "b/c/a.txt",
        "z.txt",
    ]


def test_analyze_folder_ignores_non_text_extensions(tmp_path: Path) -> None:
    (tmp_path / "song.flac").write_bytes(b"fLaC")
    (tmp_path / "data.json").write_text("{}", encoding="utf-8")
    (tmp_path / "script.txt").write_text("B - Needs fixing.\n", encoding="utf-8")

    analyses = analyze_folder(tmp_path)

    assert [Path(a.path).name for a in analyses] == ["script.txt"]


def test_analyze_folder_empty_and_clean(tmp_path: Path) -> None:
    assert analyze_folder(tmp_path) == []
    (tmp_path / "clean.txt").write_text("A: fine.\nB: also fine.\n", encoding="utf-8")
    assert analyze_folder(tmp_path) == []


def test_preview_folder_fixes_returns_rewrites_and_warnings(tmp_path: Path) -> None:
    (tmp_path / "messy.txt").write_text("A - Hello there.\n", encoding="utf-8")
    (tmp_path / "unknown.txt").write_text("A: fine.\nChapter: unclear.\n", encoding="utf-8")

    fixes, warnings_only = preview_folder_fixes(tmp_path)

    assert [f.path.name for f in fixes] == ["messy.txt"]
    assert fixes[0].fix_count == 1
    assert fixes[0].fixed_text == "A: Hello there.\n"
    # The warning-only file is reported but never rewritten.
    assert [Path(w.path).name for w in warnings_only] == ["unknown.txt"]


def test_preview_folder_fixes_raises_when_nothing_fixable(tmp_path: Path) -> None:
    (tmp_path / "clean.txt").write_text("A: fine.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        preview_folder_fixes(tmp_path)


def test_apply_folder_fixes_writes_backups_and_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "messy.txt").write_text("A - Hello there.\nB - Hi back.\n", encoding="utf-8")
    (tmp_path / "brackets.md").write_text("[A]: Greetings.\n", encoding="utf-8")

    fixes, _warnings = preview_folder_fixes(tmp_path)
    written = apply_folder_fixes(fixes)

    assert [Path(p).name for p, _c, _b in written] == ["brackets.md", "messy.txt"]
    for path, count, backup_path in written:
        assert count >= 1
        assert backup_path is not None and Path(backup_path).is_file()
        # The backup holds the original content.
        assert "A:" not in Path(backup_path).read_text(encoding="utf-8")
    # The files themselves are corrected.
    assert (tmp_path / "messy.txt").read_text(encoding="utf-8") == "A: Hello there.\nB: Hi back.\n"
    assert (tmp_path / "brackets.md").read_text(encoding="utf-8") == "A: Greetings.\n"

    # Re-running the batch finds nothing left to fix.
    assert analyze_folder(tmp_path) == []
    with pytest.raises(ValueError):
        preview_folder_fixes(tmp_path)


def test_batch_fix_round_trips_through_real_ingester(tmp_path: Path) -> None:
    from the_oracle.text_ingest import TextIngestor

    (tmp_path / "messy.txt").write_text("A - Hello there.\nB - Hi back.\n", encoding="utf-8")

    fixes, _warnings = preview_folder_fixes(tmp_path)
    apply_folder_fixes(fixes)

    document = TextIngestor().ingest(tmp_path / "messy.txt")
    speakers = {segment.explicit_speaker for segment in document.segments}
    assert speakers == {"A", "B"}
