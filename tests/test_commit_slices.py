"""Tests for the stacked-slice commit tool (scripts/commit_slices.py).

The tool commits several verified-but-uncommitted slices as separate
index-level commits — explicit paths per slice, house-format messages, then
the journal commit — so a session's pile lands bisectable and revertable
instead of as one lump. Its safety properties are the product: a slice file
that double-claims a path, names a typo'd path (which would silently commit
nothing), re-lists paths that are already clean (a stale re-run of an old
slice file), or lets the journal be skipped without a decision must be refused
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
REPO_ROOT = TOOL_PATH.parent.parent


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
    # tracked.py must be dirty here: the staleness guard refuses an all-clean
    # slice before the duplicate rule on the second slice could fire, and this
    # test pins the duplicate refusal specifically.
    (repo / "tracked.py").write_text("changed", encoding="utf-8")

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


# --- staleness (already-clean paths refuse loudly) -------------------------------


def test_validate_refuses_a_slice_whose_paths_are_all_already_clean(tmp_path: Path) -> None:
    tool = _load_tool()
    repo = _tmp_repo(tmp_path)
    slices = tool.parse_slice_file("# --- Stale rerun ---\ntracked.py\n")
    try:
        tool.validate(slices, repo)
    except ValueError as exc:
        raised = str(exc)
    else:
        raise AssertionError("an all-clean slice must be refused before any commit")
    assert "already clean" in raised and "commit nothing" in raised
    assert "Stale rerun" in raised


def test_validate_warns_but_proceeds_on_a_partially_clean_slice(tmp_path: Path, capsys) -> None:
    tool = _load_tool()
    repo = _tmp_repo(tmp_path)
    (repo / "dirty.py").write_text("x", encoding="utf-8")
    slices = tool.parse_slice_file("# --- Mixed ---\ntracked.py\ndirty.py\n")
    tool.validate(slices, repo)  # must not raise
    err = capsys.readouterr().err
    assert "WARNING" in err and "already clean" in err and "tracked.py" in err


def test_stale_second_slice_aborts_before_the_first_slice_commits(tmp_path: Path) -> None:
    """The overtake guard's teeth: one stale slice anywhere in the file stops
    the run in pre-flight, so an earlier fresh slice never half-lands."""
    tool = _load_tool()
    repo = _tmp_repo(tmp_path)
    (repo / "a.py").write_text("a", encoding="utf-8")
    slice_file = tmp_path / "slices.txt"
    slice_file.write_text(
        "# --- Fresh ---\n"
        "a.py\n"
        ">> Real work.\n"
        "# --- Stale rerun ---\n"
        "tracked.py\n",
        encoding="utf-8",
    )
    raised = None
    try:
        tool.main([str(slice_file), "--repo-root", str(repo), "--no-journal"])
    except ValueError as exc:
        raised = str(exc)
    assert raised is not None and "already clean" in raised
    count = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout.strip()
    assert count == "1", "the pre-flight refusal must precede every slice commit"


def test_dry_run_of_a_stale_slice_file_also_refuses(tmp_path: Path) -> None:
    tool = _load_tool()
    repo = _tmp_repo(tmp_path)
    slice_file = tmp_path / "slices.txt"
    slice_file.write_text("# --- Stale rerun ---\ntracked.py\n", encoding="utf-8")
    raised = None
    try:
        tool.main([str(slice_file), "--repo-root", str(repo), "--dry-run", "--no-journal"])
    except ValueError as exc:
        raised = str(exc)
    assert raised is not None and "already clean" in raised


def test_validate_refuses_an_all_clean_rerun_of_the_landed_attribution_fix(
    tmp_path: Path,
) -> None:
    """Regression pin against the REAL landed fix, not a synthetic repo: a
    slice file re-describing the cross-record attribution layer (landed as
    fbeb0cb — helpers.attribute_to_every_record_line plus the record net's
    cross-file pin) must be refused as all-clean, because its files are
    committed. The synthetic pins above prove the guard's mechanics; this one
    proves it fires for a real slice file someone would actually re-run — the
    exact mistake this thread nearly made twice (re-landing shipped work).

    Runs against a depth-1 clone of THIS repo's HEAD, never the live
    worktree: HEAD always contains the fix, so the pin is immune to parallel
    sessions' uncommitted churn, and it fails loudly if the fix is ever
    reverted or its files renamed (the premise assertion names that case).
    The contrast case gives the pin teeth: reopening the work (dirtying one
    of the fix's paths) must flip the verdict — the refusal is the all-clean
    property, not something incidental about a fresh clone.
    """
    tool = _load_tool()
    fix_paths = ("tests/test_record_integrity.py", "tests/helpers.py")
    landed = subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", f"file://{REPO_ROOT}", str(tmp_path / "clone")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert landed.returncode == 0
    clone = tmp_path / "clone"

    # Premise, asserted before the pin can mislead: HEAD really carries both
    # of the fix's files. A revert or rename must fail HERE, with this
    # message, not as a confusing different refusal below.
    for path in fix_paths:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=str(clone),
            capture_output=True,
            text=True,
        )
        assert tracked.returncode == 0, (
            f"{path} is no longer tracked at HEAD — the fbeb0cb fix this pin "
            "guards was reverted or renamed; rewrite or retire this pin, "
            "never silently"
        )

    slices = tool.parse_slice_file(
        "# --- Already landed: cross-record attribution layer (fbeb0cb) ---\n"
        "tests/test_record_integrity.py\n"
        "tests/helpers.py\n"
    )
    raised = None
    try:
        tool.validate(slices, clone)
    except ValueError as exc:
        raised = str(exc)
    else:
        raise AssertionError(
            "an all-clean re-run of the landed attribution fix must be refused"
        )
    assert raised is not None
    assert "already clean" in raised and "commit nothing" in raised
    # The refusal names the fix (the slice title) and BOTH of its paths, so
    # the operator sees exactly which landed work the stale file re-describes.
    assert "fbeb0cb" in raised
    for path in fix_paths:
        assert path in raised

    # Contrast (the pin's teeth): reopen the work — dirty one of the fix's
    # paths in the clone — and the same guard must proceed. A different title
    # because validate refuses duplicate titles before path checks.
    (clone / "tests" / "helpers.py").write_text("# reopened\n", encoding="utf-8")
    reopened = tool.parse_slice_file(
        "# --- Fix reopened mid-flight ---\n"
        "tests/test_record_integrity.py\n"
        "tests/helpers.py\n"
    )
    tool.validate(reopened, clone)  # must not raise


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


# --- journal checking -----------------------------------------------------------


def test_check_journal_accepts_the_house_formats_life() -> None:
    """The validator must accept the journal's real shape — headers, the
    Format: doc line, historical free-form notes, sibling-convention entries
    that fold the suite note into the description — and pin only what
    'malformed' means: bad dates, glued/short dates that would silently
    skip, empty fields, and broken commit citations."""
    tool = _load_tool()
    lines = [
        "# JUNO_FIXES.log — Oracle TTS launch-readiness",
        "",
        "Format: `<date> | <file> | <what was wrong> | <what you changed> | <test results>`",
        "2026-09-17 | Noticed, not actioned | stale launcher note (two fields — fine)",
        "2026-09-20 | Release V1.2.0 artifacts built (commit a397943) | description folded into three fields",
        "2026-09-28 | files (commit b5b9683) | description | Full suite: 1517 passed / 0 failed.",
        "2026-09-28 | files (commits 0445a72, 2ba2392) | hash list | green.",
        "2026-09-28 | files (commit pending) | not yet committed | n/a.",
        "Totally unformatted historical note.",
        "",
    ]
    assert tool.check_journal_lines(lines, floors=False) == []


def test_check_journal_flags_malformed_entries_by_name() -> None:
    tool = _load_tool()
    base = 20  # any lines before these are fine; floors disabled below
    problems = tool.check_journal_lines(
        [
            "2026-09-28 | files (commit b5b9683) | ok | green.",
            f"{base} filler",  # keep the list non-trivially short
            "2026-13-45 | files (commit b5b9683) | impossible calendar date | green.",
            "2026-1-5 | glued date and no separator — would silently skip",
            "2026-09-28 |  | empty field | green.",
            "2026-09-28 | files (commit b5b96) | truncated hash | green.",
            "2026-09-28 | files (commit ) | empty citation body | green.",
            "2026-09-28 | files (commits abc1234, b5b9683) | mixed short list is legal | green.",
        ],
        floors=False,
    )
    rendered = "\n".join(problems)
    assert "is not a valid calendar date" in rendered
    assert "does not match the entry shape" in rendered
    assert "empty field" in rendered
    assert "not 7-40 hex" in rendered
    assert "empty (commit ...) citation" in rendered
    # The mixed list must NOT be flagged (lenient citation body rule).
    assert "abc1234" not in rendered


def test_a_malformed_citation_is_attributed_to_every_line_carrying_it() -> None:
    """The 71d8840 completeness precedent, applied to the journal validator:
    a malformed citation cited on SEVERAL lines must be reported on EVERY
    line carrying it, not deduped to the first occurrence — a one-line fix
    would otherwise leave the trail broken elsewhere while the checker kept
    pointing at a single spot. Pinned synthetically (non-adjacent lines, two
    fictional stale hashes, exact ordered assertion, git-independent), with
    vacuity guards so the scan cannot go blind: an empty input yields no
    problems, and each reported problem must carry its own ``line N:``
    attribution so callers can grep the offending lines."""
    tool = _load_tool()
    lines = [
        "2026-10-08 | files (commit b5b9) | first carrier of the stale hash | green.",
        "2026-10-08 | files (commit b5b9683) | clean line between the carriers | green.",
        "2026-10-08 | files (commit b5b9) | second carrier, non-adjacent | green.",
        "2026-10-08 | files (commit b5b9683) | clean again | green.",
        "2026-10-08 | files (commit abc12) | a different stale hash | green.",
    ]
    problems = tool.check_journal_lines(lines, floors=False)
    assert problems == [
        "line 1: commit citation 'b5b9' is not 7-40 hex digits — a truncated "
        "or garbled hash cannot be verified",
        "line 3: commit citation 'b5b9' is not 7-40 hex digits — a truncated "
        "or garbled hash cannot be verified",
        "line 5: commit citation 'abc12' is not 7-40 hex digits — a truncated "
        "or garbled hash cannot be verified",
    ], problems
    # Vacuity: the every-line property is what the exact list above pins — a
    # first-match-only regression collapses the list to [line 1, line 5] and
    # fails the assertion, not silently. And the attribution is greppable:
    assert sum(problem.startswith("line ") for problem in problems) == len(problems)
    # The clean carriers are never flagged:
    assert not any("b5b9683" in problem for problem in problems)
    assert tool.check_journal_lines([], floors=False) == []


def test_check_journal_vacuity_floors_apply_only_in_check_mode() -> None:
    """The floors are a whole-file blindness guard: they must fail a journal
    with too few entries, and equally they must NOT fire on a small pre-write
    batch (the commit-time refusal runs per-line rules only)."""
    tool = _load_tool()
    small_batch = ["2026-09-28 | files (commit b5b9683) | d | green."]
    assert tool.check_journal_lines(small_batch, floors=False) == []
    problems = tool.check_journal_lines(small_batch, floors=True)
    assert any("floor 10" in problem for problem in problems)
    assert any("no (commit" in problem for problem in problems) is False or problems
    # Exact citation floor message present when entries exist but no citation.
    many_no_citation = [f"2026-09-28 | entry {i} | d | green." for i in range(12)]
    problems = tool.check_journal_lines(many_no_citation, floors=True)
    assert any("no (commit <hash>) citation found" in problem for problem in problems)


def test_check_journal_mode_reads_the_real_file(tmp_path: Path, capsys) -> None:
    """The CLI mode: a passing journal exits 0 with the hold message; a
    malformed one exits 1 and renders every problem."""
    tool = _load_tool()
    repo = tmp_path
    (repo / "JUNO_FIXES.log").write_text(
        "# header\n"
        "2026-09-28 | files (commit b5b9683) | d | green.\n"
        + "".join(
            f"2026-09-2{i} | entry {i} | d | green.\n" for i in range(9)
        ),
        encoding="utf-8",
    )
    assert tool.main(["--check-journal", "--repo-root", str(repo)]) == 0
    assert "house format holds" in capsys.readouterr().out

    (repo / "JUNO_FIXES.log").write_text(
        "# header\n"
        + "".join(f"2026-09-2{i} | entry {i} | d | green.\n" for i in range(9))
        + "2026-09-28 | files (commit b5b9) | truncated | green.\n",
        encoding="utf-8",
    )
    assert tool.main(["--check-journal", "--repo-root", str(repo)]) == 1
    err = capsys.readouterr().err
    assert "MALFORMED" in err and "b5b9" in err


def test_commit_refuses_a_malformed_journal_entry_before_writing(tmp_path: Path) -> None:
    """The pre-write refusal: --journal-entry text with a truncated commit
    citation stops the run before JUNO_FIXES.log is touched."""
    tool = _load_tool()
    repo = _tmp_repo(tmp_path)
    (repo / "a.py").write_text("a", encoding="utf-8")
    slice_file = tmp_path / "slices.txt"
    slice_file.write_text("# --- A ---\na.py\n", encoding="utf-8")

    # The refusal raises out of main like every other validate()-class
    # refusal (the __main__ wrapper turns it into a CLI exit 1).
    raised = None
    try:
        tool.main(
            [
                str(slice_file),
                "--repo-root",
                str(repo),
                "--journal-entry",
                "1:d (commit b5b9) with a truncated hash",
                "--suite-note",
                "green.",
            ]
        )
    except ValueError as exc:
        raised = str(exc)
    assert raised is not None and "b5b9" in raised
    assert not (repo / "JUNO_FIXES.log").exists(), "the refusal must precede any write"
    count = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout.strip()
    # The slice commit itself legitimately landed (the tool's contract:
    # already-committed slices stay committed; the run stops at the failing
    # step) — the journal write and its commit are what were refused.
    assert count == "2"


# --- per-slice verify commands ---------------------------------------------------


def test_verify_commands_parse_and_validate() -> None:
    tool = _load_tool()
    text = (
        "# --- A ---\n"
        "a.py\n"
        "! echo one\n"
        "! echo two\n"
        ">> Subject.\n"
    )
    slices = tool.parse_slice_file(text)
    assert slices[0].verify_commands == ["echo one", "echo two"]

    raised = None
    try:
        tool.parse_slice_file("! echo before any header\n")
    except ValueError as exc:
        raised = str(exc)
    assert raised is not None and "before any" in raised

    raised = None
    try:
        tool.parse_slice_file("# --- A ---\na.py\n!\n")
    except ValueError as exc:
        raised = str(exc)
    assert raised is not None and "empty verify command" in raised


def test_verify_flag_without_commands_is_refused(tmp_path: Path) -> None:
    tool = _load_tool()
    repo = _tmp_repo(tmp_path)
    (repo / "a.py").write_text("a", encoding="utf-8")
    slice_file = tmp_path / "slices.txt"
    slice_file.write_text("# --- A ---\na.py\n", encoding="utf-8")
    raised = None
    try:
        tool.main([str(slice_file), "--repo-root", str(repo), "--verify", "--no-journal"])
    except ValueError as exc:
        raised = str(exc)
    assert raised is not None and "no slice defines a '!' verify command" in raised


def test_verify_failure_stops_the_run_and_skips_the_journal(tmp_path: Path) -> None:
    """A red slice stops the run right there: the failed slice stays committed
    (independently revertable), later slices and the journal never land."""
    tool = _load_tool()
    repo = _tmp_repo(tmp_path)
    (repo / "a.py").write_text("a", encoding="utf-8")
    (repo / "b.py").write_text("b", encoding="utf-8")
    slice_file = tmp_path / "slices.txt"
    slice_file.write_text(
        "# --- First ---\n"
        "a.py\n"
        ">> Move the first slice.\n"
        "# --- Second ---\n"
        "b.py\n"
        "! exit 3\n"
        ">> Add the second slice.\n",
        encoding="utf-8",
    )
    raised = None
    try:
        tool.main(
            [
                str(slice_file),
                "--repo-root",
                str(repo),
                "--verify",
                "--journal-entry",
                "1:first slice journal text",
            ]
        )
    except ValueError as exc:
        raised = str(exc)
    assert raised is not None and "exit 3" in raised and "Second" in raised
    subjects = subprocess.run(
        ["git", "log", "--format=%s"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout.splitlines()
    # The failed slice committed before its verify ran — revertable, per the
    # tool's contract — but the journal commit is absent.
    assert subjects[0] == "Add the second slice."
    assert "Move the first slice." in subjects
    assert "Record the committed slices in the journals" not in subjects
    assert not (repo / "JUNO_FIXES.log").exists(), "a failed verify must precede the journal write"


def test_verify_success_continues_to_the_journal(tmp_path: Path) -> None:
    tool = _load_tool()
    repo = _tmp_repo(tmp_path)
    (repo / "a.py").write_text("a", encoding="utf-8")
    slice_file = tmp_path / "slices.txt"
    slice_file.write_text(
        "# --- A ---\na.py\n! test -f a.py\n>> Add the slice.\n", encoding="utf-8"
    )
    exit_code = tool.main(
        [
            str(slice_file),
            "--repo-root",
            str(repo),
            "--verify",
            "--journal-entry",
            "1:journal text",
        ]
    )
    assert exit_code == 0
    subjects = subprocess.run(
        ["git", "log", "--format=%s"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout.splitlines()
    assert subjects[0] == "Record the committed slices in the journals"
    assert (repo / "JUNO_FIXES.log").is_file()
