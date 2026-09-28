"""Tests for the stacked-slice commit tool (scripts/commit_slices.py).

The tool commits several verified-but-uncommitted slices as separate
index-level commits — explicit paths per slice, house-format messages, then
the journal commit — so a session's pile lands bisectable and revertable
instead of as one lump. Its safety properties are the product: a slice file
that double-claims a path, names a typo'd path (which would silently commit
nothing), or lets the journal be skipped without a decision must be refused
BEFORE the first commit; a slice commit must contain exactly its listed
paths (a concurrent session's dirty files must never ride along); and the
journal must come out in the house format with the real commit hashes.

Parsing and validation run against the loaded module; the end-to-end commit
runs in a disposable git repo (``--repo-root``), never this checkout.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

TOOL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "commit_slices.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("commit_slices", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # The tool uses @dataclass, whose decorator looks the module up in
    # sys.modules — register before exec or the load itself fails.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- parsing ------------------------------------------------------------------


def test_parse_slice_file_reads_paths_messages_and_blanks() -> None:
    tool = _load_tool()
    text = """# --- First slice ---
a.py
b.py
>> Subject line.
>> Body paragraph one.

>> Body paragraph two.
# --- Second slice ---
c.py
# a comment line that is not a header
"""
    slices = tool.parse_slice_file(text)
    assert [s.title for s in slices] == ["First slice", "Second slice"]
    assert slices[0].paths == ["a.py", "b.py"]
    assert slices[0].subject == "Subject line."
    # Blank ">>" lines are preserved as paragraph separators.
    assert slices[0].body == "Body paragraph one.\n\nBody paragraph two."
    assert slices[1].paths == ["c.py"] and slices[1].body == ""


def test_parse_slice_file_refuses_orphan_lines_and_empty_titles() -> None:
    tool = _load_tool()
    for bad_text, expected in (
        (">> orphan message\n# --- A ---\na.py", "message before any"),
        ("a.py\n# --- A ---\n", "before any '# ---' header"),
        ("# --- ---\na.py", "empty slice title"),
    ):
        try:
            tool.parse_slice_file(bad_text)
        except ValueError as exc:
            assert expected in str(exc), f"{bad_text!r}: {exc}"
        else:
            raise AssertionError(f"parse accepted bad input: {bad_text!r}")


def test_slice_subject_falls_back_to_title() -> None:
    tool = _load_tool()
    (slices,) = tool.parse_slice_file("# --- Title only ---\na.py\n")
    assert slices.subject == "Title only" and slices.body == ""


# --- validation (refusals happen before any commit) ----------------------------


def _tmp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
    (repo / "tracked.py").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)
    return repo


def test_validate_refuses_duplicate_paths_typos_and_quotes(tmp_path: Path) -> None:
    tool = _load_tool()
    repo = _tmp_repo(tmp_path)
    (repo / "dirty.py").write_text("x", encoding="utf-8")

    slices = tool.parse_slice_file("# --- A ---\ntracked.py\n# --- B ---\ntracked.py\n")
    try:
        tool.validate(slices, repo)
    except ValueError as exc:
        assert "already claimed" in str(exc)
    else:
        raise AssertionError("duplicate path across slices must be refused")

    slices = tool.parse_slice_file("# --- A ---\nno_such_file.py\n")
    try:
        tool.validate(slices, repo)
    except ValueError as exc:
        assert "neither tracked nor dirty" in str(exc)
    else:
        raise AssertionError("typo'd path must be refused")

    slices = tool.parse_slice_file("# --- A ---\n'tracked.py'\n")
    try:
        tool.validate(slices, repo)
    except ValueError as exc:
        assert "quote" in str(exc)
    else:
        raise AssertionError("shell-quoted path must be refused")


def test_journal_entry_specs_parse_and_range_check() -> None:
    tool = _load_tool()
    entries = tool.parse_journal_entries(["1:first", "2:second thing"])
    assert entries == {0: ["first"], 1: ["second thing"]}
    try:
        tool.parse_journal_entries(["no-colon"])
    except ValueError as exc:
        assert "slice-number" in str(exc)
    else:
        raise AssertionError("malformed journal spec must be refused")


# --- end to end (disposable repo) -----------------------------------------------


def test_end_to_end_commits_slices_in_order_and_writes_the_journal(tmp_path: Path) -> None:
    tool = _load_tool()
    repo = _tmp_repo(tmp_path)
    (repo / "a.py").write_text("a", encoding="utf-8")
    (repo / "b.py").write_text("b", encoding="utf-8")
    (repo / "DECOY.txt").write_text("must not be committed", encoding="utf-8")
    (repo / "JUNO_FIXES.log").write_text(
        "2026-09-27 | earlier.txt (commit abc1234) | earlier | Full suite: 1 passed / 0 failed.\n",
        encoding="utf-8",
    )
    slice_file = tmp_path / "slices.txt"
    slice_file.write_text(
        "# --- First slice ---\n"
        "a.py\n"
        ">> Move the first slice.\n"
        ">> Bodies verbatim, verified by the full suite.\n"
        "# --- Second slice ---\n"
        "b.py\n"
        ">> Add the second slice.\n",
        encoding="utf-8",
    )

    exit_code = tool.main(
        [
            str(slice_file),
            "--repo-root",
            str(repo),
            "--journal-entry",
            "1:First slice journal text",
            "--journal-entry",
            "2:Second slice journal text",
            "--suite-note",
            "Full suite: 1487 passed / 0 failed at slice time.",
        ]
    )
    assert exit_code == 0

    log = subprocess.run(
        ["git", "log", "--format=%s"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout.splitlines()
    assert log[0] == "Record the committed slices in the journals"
    assert log[1] == "Add the second slice."
    assert log[2] == "Move the first slice."

    # Slice 1 contains exactly its paths; slice 2 exactly its own; the decoy
    # was never staged.
    show = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD~1"],
        cwd=str(repo), check=True, capture_output=True, text=True,
    ).stdout.split()
    assert show == ["b.py"]
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout
    assert "DECOY.txt" in status

    # The footer rode every commit, and the journal is in the house format
    # with the real short hashes.
    body = subprocess.run(
        ["git", "log", "-2", "--format=%B"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout
    assert body.count("Co-Authored-By: Codebuff <noreply@codebuff.com>") == 2
    journal = (repo / "JUNO_FIXES.log").read_text(encoding="utf-8").splitlines()
    # Look slice 1's hash up by subject rather than a positional offset —
    # the journal commit (and slice 2) sit above it in the history.
    subjects = subprocess.run(
        ["git", "log", "--format=%H %s"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout.splitlines()
    hash_a = next(line.split()[0][:7] for line in subjects if line.endswith("Move the first slice."))
    assert (
        f"(commit {hash_a}) | First slice journal text | "
        "Full suite: 1487 passed / 0 failed at slice time."
    ) in journal[-2]
    assert journal[-1].endswith("Second slice journal text | Full suite: 1487 passed / 0 failed at slice time.")


def test_dry_run_commits_nothing(tmp_path: Path) -> None:
    tool = _load_tool()
    repo = _tmp_repo(tmp_path)
    (repo / "a.py").write_text("a", encoding="utf-8")
    slice_file = tmp_path / "slices.txt"
    slice_file.write_text("# --- A ---\na.py\n", encoding="utf-8")

    exit_code = tool.main([str(slice_file), "--repo-root", str(repo), "--dry-run", "--no-journal"])
    assert exit_code == 0
    count = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout.strip()
    assert count == "1", "a dry run must not create commits"
    # The untracked file is still exactly that — never staged, never committed.
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout
    assert status.strip() == "?? a.py", "a dry run must not stage anything"
