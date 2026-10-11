"""Tests for the suite-count drift check.

The script's whole value is in three properties, so those are what is pinned:
that a moved count is *drift* rather than a detail to note in passing, that the
skip ids it reports come from the run's own audit rather than a second
implementation of "what skipped", and that an unrecorded commit is refused
instead of passing quietly -- a check that cannot say "I did not verify this" is
a check that eventually reports a green it did not earn.

The slow end-to-end run (a full suite in a fresh archive tree) is deliberately
not repeated here; it takes minutes and is the script's own job. Every decision
the script makes is exercised, plus the real acquisition, so these stay fast
enough to run on every change.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
BASELINE = SCRIPTS_DIR / "suite_counts_baseline.json"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def counts():
    return _load("oracle_certify_suite_counts", "certify_suite_counts.py")


@pytest.fixture(scope="module")
def acceptance():
    return _load("oracle_fresh_clone_acceptance_for_counts", "fresh_clone_acceptance.py")


def _entry(**overrides) -> dict:
    entry = {
        "passed": 1684,
        "skipped": 3,
        "failed": 0,
        "errors": 0,
        "exit_code": 0,
        "skipped_ids": [],
        "skipped_ids_complete": True,
    }
    entry.update(overrides)
    return entry


# --- reading pytest's output ----------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("1684 passed, 3 skipped in 378.72s", {"passed": 1684, "failed": 0, "errors": 0, "skipped": 3}),
        ("1687 passed, 18 warnings in 388.18s", {"passed": 1687, "failed": 0, "errors": 0, "skipped": 0}),
        ("2 failed, 5 passed, 1 error in 1.0s", {"passed": 5, "failed": 2, "errors": 1, "skipped": 0}),
    ],
)
def test_counts_come_out_of_the_summary_line(counts, output: str, expected: dict) -> None:
    assert counts.parse_suite_counts(output) == expected


def test_the_skip_audit_block_is_not_counted_twice(counts) -> None:
    """The audit prints its own line containing "skipped"; only the summary counts."""
    output = (
        "s.....................                                       [100%]\n"
        "1 test(s) did not run here:\n"
        "  tests/test_vulkan_backend.py::test_vulkan_backend_smoke_requires_device\n"
        "      Skipped: audiocpp_cli binary/model not built\n"
        "\n"
        "1684 passed, 3 skipped in 378.72s\n"
    )

    assert counts.parse_suite_counts(output)["skipped"] == 3


def test_skipped_ids_come_from_the_runs_own_audit(counts) -> None:
    """The ids are what the run says about itself, reasons excluded."""
    output = (
        "1 test(s) did not run here:\n"
        "  tests/test_vulkan_backend.py::test_vulkan_backend_smoke_requires_device\n"
        "      Skipped: audiocpp_cli binary/model not built\n"
        "  tests/test_engine_output_sanitizer.py::test_vulkan_engine_rejects_silent_wav\n"
        "      Skipped: audiocpp_cli binary/model not built\n"
        "\n"
        "1684 passed, 2 skipped in 1.0s\n"
    )

    assert counts.parse_skipped_ids(output) == [
        "tests/test_engine_output_sanitizer.py::test_vulkan_engine_rejects_silent_wav",
        "tests/test_vulkan_backend.py::test_vulkan_backend_smoke_requires_device",
    ]


def test_a_run_with_no_skips_has_no_skip_ids(counts) -> None:
    assert counts.parse_skipped_ids("1687 passed, 18 warnings in 388.18s") == []


def test_a_run_with_no_summary_is_an_error_rather_than_zeros(counts) -> None:
    """Zeros would compare equal to a baseline of zeros and read as a pass."""
    assert not any(counts.parse_suite_counts("pytest: command not found").values())


# --- the comparison -------------------------------------------------------------


def test_identical_counts_are_not_drift(counts) -> None:
    assert counts.compare(_entry(), _entry()) == []


def test_a_moved_pass_count_is_drift(counts) -> None:
    drifts = counts.compare(_entry(passed=1684), _entry(passed=1681, skipped=6))

    assert [drift.field for drift in drifts] == ["passed", "skipped"]
    assert drifts[0].recorded == 1684 and drifts[0].observed == 1681


def test_a_run_that_stopped_passing_cannot_satisfy_the_baseline(counts) -> None:
    """Counts alone would miss this if ``failed`` were left out of the compare."""
    drifts = counts.compare(_entry(passed=1684), _entry(passed=1680, failed=4, exit_code=1))

    assert {drift.field for drift in drifts} == {"passed", "failed", "exit_code"}


def test_a_newly_skipped_test_is_named(counts) -> None:
    drifts = counts.compare(
        _entry(skipped=3, skipped_ids=["tests/test_a.py::test_one"]),
        _entry(skipped=4, skipped_ids=["tests/test_a.py::test_one", "tests/test_b.py::test_two"]),
    )

    # Only the direction that moved is reported: an empty "(none) -> (none)"
    # line would be noise in every failure.
    named = {drift.field: drift for drift in drifts}
    assert named["skipped_ids_added"].observed == ["tests/test_b.py::test_two"]
    assert "skipped_ids_removed" not in named


def test_a_test_that_stopped_skipping_is_named_too(counts) -> None:
    """A skip disappearing is drift as well: it means the run changed shape."""
    drifts = counts.compare(
        _entry(skipped=1, skipped_ids=["tests/test_a.py::test_one"]),
        _entry(skipped=0, skipped_ids=[]),
    )

    named = {drift.field: drift for drift in drifts}
    assert named["skipped_ids_removed"].recorded == ["tests/test_a.py::test_one"]
    assert "skipped_ids_added" not in named


def test_skip_ids_compare_as_a_set_not_as_a_list(counts) -> None:
    """Order is not drift; the recorded file is sorted, the parse is sorted too."""
    assert counts.compare(
        _entry(skipped_ids=["tests/a.py::x", "tests/b.py::y"]),
        _entry(skipped_ids=["tests/b.py::y", "tests/a.py::x"]),
    ) == []


def test_completeness_is_read_against_pytests_own_count(counts) -> None:
    assert counts.id_audit_is_complete({"skipped": 3}, ["a", "b", "c"]) is True
    assert counts.id_audit_is_complete({"skipped": 3}, ["a"]) is False
    assert counts.id_audit_is_complete({"skipped": 0}, []) is True


def test_a_short_audit_is_not_compared_as_skip_ids(counts) -> None:
    """A truncated list must not invent drift -- the count still does the work.

    ``tests/test_skip_audit.py``'s autouse fixture clears the skip ledger, so a
    full run's audit is short (measured: 1 listed of 3). Comparing that against a
    complete list would report two ids "removed" that never went anywhere, and
    fixing the fixture would report phantom "added" ones.
    """
    recorded = _entry(skipped=3, skipped_ids=["tests/a.py::x"], skipped_ids_complete=True)
    observed = _entry(
        skipped=3,
        skipped_ids=["tests/a.py::x", "tests/b.py::y", "tests/c.py::z"],
        skipped_ids_complete=False,
    )

    assert counts.compare(recorded, observed) == []
    note = counts.id_comparison_note(recorded, observed)
    assert note is not None and "NOT compared" in note


def test_a_complete_audit_on_both_sides_compares_ids_and_needs_no_note(counts) -> None:
    recorded = _entry(skipped_ids=["tests/a.py::x"], skipped_ids_complete=True)
    observed = _entry(skipped_ids=["tests/a.py::x", "tests/b.py::y"], skipped_ids_complete=True)

    assert [drift.field for drift in counts.compare(recorded, observed)] == ["skipped_ids_added"]
    assert counts.id_comparison_note(recorded, observed) is None


def test_the_note_names_which_side_was_short(counts) -> None:
    note = counts.id_comparison_note(
        _entry(skipped=3, skipped_ids=["a"], skipped_ids_complete=False),
        _entry(skipped=3, skipped_ids=["a", "b", "c"], skipped_ids_complete=True),
    )

    assert note is not None and "recorded listed 1 of 3" in note
    assert "observed" not in note.split(";")[0]


def test_the_record_report_is_wider_than_the_verdict(counts) -> None:
    """``--record`` prints every field it overwrites, including ones the verdict ignores.

    Measured: the entry really did change when ``skipped_ids_complete`` was added,
    and a re-record that printed nothing would misreport itself as a no-op.
    """
    recorded = _entry(skipped_ids=["tests/a.py::x"])
    recorded.pop("skipped_ids_complete")
    observed = _entry(skipped_ids=["tests/a.py::x"])

    assert [drift.field for drift in counts.changed_fields(recorded, observed)] == [
        "skipped_ids_complete"
    ]
    # The verdict stays silent about it: counts matched, so there is no drift.
    assert counts.compare(recorded, observed) == []


def test_the_drift_report_shows_both_values(counts) -> None:
    lines = counts.format_drift("40b8ddbd60589569bca19c171eddd2771ecd35fc", counts.compare(
        _entry(skipped=3), _entry(skipped=6)
    ))
    rendered = "\n".join(lines)

    assert "40b8ddbd6058" in rendered
    assert "skipped" in rendered
    assert "3" in rendered and "6" in rendered


# --- the baseline file ----------------------------------------------------------


def test_recording_round_trips(counts, tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    entry = _entry(passed=1684)

    counts.save_baseline({"a" * 40: entry}, path)

    assert counts.load_baseline(path) == {"a" * 40: entry}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["about"] == counts.BASELINE_ABOUT


def test_a_missing_baseline_file_reads_as_empty(counts, tmp_path: Path) -> None:
    assert counts.load_baseline(tmp_path / "absent.json") == {}


def test_a_malformed_baseline_file_is_refused(counts, tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(counts.SuiteCountsError, match="counts"):
        counts.load_baseline(path)


def test_an_unrecorded_commit_is_refused_rather_than_passed(counts, tmp_path: Path, capsys) -> None:
    """Exit 2, before any worktree exists: a question is not a pass."""
    path = tmp_path / "baseline.json"
    counts.save_baseline({}, path)

    assert counts.main(["--baseline", str(path), "--scratch", str(tmp_path)]) == 2

    # And it must not go on to "remove" a worktree that was never added: that
    # reported a warning about a path git never created (measured before the fix).
    captured = capsys.readouterr()
    assert "no baseline is recorded" in captured.err
    assert "WARNING" not in captured.err
    # Only the baseline file this test wrote is left: no scratch directory.
    assert [entry.name for entry in tmp_path.iterdir()] == ["baseline.json"]


def test_a_scratch_directory_that_does_not_exist_yet_is_created(counts, tmp_path: Path, capsys) -> None:
    """Naming a scratch directory means "put it there", not "it already exists".

    Measured before the fix: ``--scratch`` on a fresh path died in a
    ``FileNotFoundError`` traceback, and an uncaught crash exits 1 -- the same
    code this script reserves for drift, so a user would read a crash as drift.
    """
    path = tmp_path / "baseline.json"
    counts.save_baseline({}, path)
    scratch = tmp_path / "deep" / "scratch"

    assert counts.main(["--baseline", str(path), "--scratch", str(scratch)]) == 2

    assert scratch.is_dir()
    assert "no baseline is recorded" in capsys.readouterr().err


def test_the_check_compares_the_fresh_tree_and_never_the_working_tree(counts) -> None:
    """The baseline describes a commit, so the suite runs in the archived tree.

    Running it in the working tree instead would grade a tree that carries state
    the commit does not -- a venv, a native build, leftover clips -- which is
    exactly the difference that made this count unstable by hand.
    """
    source = (SCRIPTS_DIR / "certify_suite_counts.py").read_text(encoding="utf-8")

    assert source.count("run_suite(") == 2, "expected the definition plus one call site"
    assert "run_suite(fresh_root, interpreter" in source
    assert "run_suite(repo_root" not in source


# --- the recorded baseline itself -----------------------------------------------


def test_the_recorded_baseline_has_the_shape_the_check_reads(counts) -> None:
    """Guards the file against a hand edit that would quietly de-fang the check."""
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert set(payload) == {"about", "counts"}
    assert isinstance(payload["about"], str) and len(payload["about"]) > 40
    assert payload["counts"], "the baseline is empty; nothing would ever be compared"
    for commit, entry in payload["counts"].items():
        assert len(commit) == 40 and commit.islower(), f"{commit} is not a full sha"
        expected = {*counts.COUNT_FIELDS, counts.SKIP_ID_FIELD, counts.SKIP_ID_COMPLETE_FIELD}
        assert set(entry) == expected, (
            f"{commit[:12]} has fields {sorted(entry)}; the check reads {sorted(expected)}"
        )
        assert isinstance(entry[counts.SKIP_ID_COMPLETE_FIELD], bool)
        assert entry["exit_code"] == 0, (
            f"{commit[:12]} recorded a non-zero exit code; a baseline must be a green run"
        )


# --- acquiring the tree ---------------------------------------------------------


def _scratch_repo(root: Path, *, commits: int = 1) -> Path:
    """A tiny repository: tracked file, ignored dir, untracked file.

    Built here rather than reusing this checkout so the acquisition tests run
    wherever the suite does, including in the worktree this script creates.
    """
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "t"], check=True)
    (root / ".gitignore").write_text("ignored_dir/\n", encoding="utf-8")
    (root / "ignored_dir").mkdir()
    (root / "ignored_dir" / "build.bin").write_bytes(b"noise")
    for index in range(commits):
        (root / "tracked.txt").write_text(f"commit {index + 1}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", ".gitignore", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", f"c{index + 1}"], check=True)
    (root / "untracked.txt").write_text("local\n", encoding="utf-8")
    return root


def test_the_acquisition_is_a_real_checkout_that_can_be_cloned(counts, tmp_path: Path) -> None:
    """A worktree, not an archive: the suite has tests that need the repository.

    ``test_commit_slices`` clones ``file://HEAD`` to copy the landed-fix pin, and
    ``test_record_integrity`` reads the commit history; in an archive both behave
    differently (one fails, one skips), so a baseline recorded from one would
    describe a tree no clone ever produces. Measured: an archive of this
    repository yields 1 failure and 7 skips where the worktree yields 0 and 3.
    """
    repo = _scratch_repo(tmp_path / "repo")

    tree = counts.add_fresh_worktree(repo, tmp_path / "wt" / "tree", "HEAD")

    assert (tree / ".git").exists(), "an archive has no .git; a worktree does"
    assert (tree / "tracked.txt").read_text(encoding="utf-8") == "commit 1\n"
    # Local state does not travel, which is what keeps the counts machine-independent.
    assert not (tree / "ignored_dir").exists()
    assert not (tree / "untracked.txt").exists()

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", f"file://{tree}", str(clone)], check=True
    )
    assert (clone / "tracked.txt").is_file()

    assert counts.remove_fresh_worktree(repo, tree) is None
    assert not tree.exists()


def test_the_worktree_is_checked_out_at_the_ref_that_was_asked_for(counts, tmp_path: Path) -> None:
    """``of a given commit`` is the point: HEAD alone would make this a mirror."""
    repo = _scratch_repo(tmp_path / "repo", commits=2)

    tree = counts.add_fresh_worktree(repo, tmp_path / "wt" / "tree", "HEAD~1")

    assert (tree / "tracked.txt").read_text(encoding="utf-8") == "commit 1\n"
    counts.remove_fresh_worktree(repo, tree)


def test_a_failed_removal_is_reported_rather_than_swallowed(counts, tmp_path: Path) -> None:
    """A silent failure leaves the repository tracking a worktree nobody knows about."""
    repo = _scratch_repo(tmp_path / "repo")
    tree = counts.add_fresh_worktree(repo, tmp_path / "wt" / "tree", "HEAD")
    assert counts.remove_fresh_worktree(repo, tree) is None

    complaint = counts.remove_fresh_worktree(repo, tree)

    assert complaint is not None and "worktree remove" in complaint


# --- agreeing with the sibling fresh-tree check ---------------------------------


def test_both_fresh_tree_checks_refuse_a_tree_that_does_not_load_its_own_code(
    counts, acceptance, tmp_path: Path
) -> None:
    """Two scripts assert the same thing about the same trap; they must agree.

    This interpreter has the package installed editable against the working tree,
    so an empty ``src`` still resolves ``the_oracle`` -- to the wrong tree. Both
    scripts have to refuse that rather than report numbers for code that never ran.
    """
    empty = tmp_path / "empty"
    (empty / "src").mkdir(parents=True)

    with pytest.raises(counts.SuiteCountsError, match="did not load its own code"):
        counts.assert_imports_the_tree_under_test(empty, Path(sys.executable))
    with pytest.raises(acceptance.AcceptanceError, match="did not load its own code"):
        acceptance.assert_imports_the_tree_under_test(empty, Path(sys.executable))


def test_both_fresh_tree_checks_read_a_summary_line_the_same_way(counts, acceptance) -> None:
    summary = "1 test(s) did not run here:\n\n1684 passed, 3 skipped in 378.72s\n"

    assert counts.parse_suite_counts(summary) == acceptance.parse_pytest_summary(summary)
