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
import yaml

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


# --- inheriting the entry for the commit that carries it ------------------------


def _git_in(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo_whose_tip_carries_a_recording(root: Path) -> tuple[Path, Path, str]:
    """The shape of the real dead end: a measured commit, and the commit carrying its entry.

    Commit A is what the recording names; its entry is then written into the
    working tree and committed as B. B can never be recorded itself -- an entry
    cannot name a commit whose content contains the entry -- so B is the commit
    the check has to be able to answer for without a run.
    """
    root.mkdir(parents=True, exist_ok=True)
    _git_in(root, "init", "-q")
    _git_in(root, "config", "user.email", "t@t")
    _git_in(root, "config", "user.name", "t")
    baseline = root / "scripts" / "suite_counts_baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({"about": "x", "counts": {}}), encoding="utf-8")
    (root / "suite_test.py").write_text("def test_one(): pass\n", encoding="utf-8")
    _git_in(root, "add", "suite_test.py", "scripts/suite_counts_baseline.json")
    _git_in(root, "commit", "-qm", "a: the tree that was measured")
    measured = _git_in(root, "rev-parse", "HEAD")
    baseline.write_text(
        json.dumps({"about": "x", "counts": {measured: _entry()}}), encoding="utf-8"
    )
    _git_in(root, "add", "scripts/suite_counts_baseline.json")
    _git_in(root, "commit", "-qm", "b: carries a's entry")
    return root, baseline, measured


def _check_scratch(tmp_path: Path) -> Path:
    return tmp_path / "scratch"


def test_the_commit_that_carries_a_recording_inherits_that_recording(
    counts, tmp_path: Path, capsys, monkeypatch
) -> None:
    """Measured on this repository: the tip that carried `dbee328`'s entry exited 2.

    ``--record`` measures a commit, writes the entry, and commits; the follower is
    what a reader has checked out, and no run can ever record it, because an entry
    cannot name a commit whose content contains the entry. So the check has to
    answer for this commit by comparing trees rather than by running -- which is
    what this pins: the suite is replaced with a hard failure here, so a green
    verdict can only have come from the comparison.
    """
    repo, baseline, measured = _repo_whose_tip_carries_a_recording(tmp_path / "repo")

    def _no_run(*args, **kwargs):
        raise AssertionError("an inherited verdict must not run the suite")

    monkeypatch.setattr(counts, "run_suite", _no_run)

    code = counts.main(
        [
            "--repo-root",
            str(repo),
            "--baseline",
            str(baseline),
            "--scratch",
            str(_check_scratch(tmp_path)),
            "--keep",
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    assert measured[:12] in captured.out and "inherit" in captured.out.lower()
    # Nothing was built, and nothing was left registered: a refusal or a green
    # that ran no suite must not leave a worktree behind, nor claim it kept one.
    assert len(_git_in(repo, "worktree", "list").splitlines()) == 1
    assert list(_check_scratch(tmp_path).iterdir()) == []
    assert "kept at" not in captured.err


def test_a_commit_that_really_changed_the_suite_still_refuses(
    counts, tmp_path: Path, capsys
) -> None:
    """Inheritance is tree equality, not "a recorded commit is close enough"."""
    repo, baseline, _ = _repo_whose_tip_carries_a_recording(tmp_path / "repo")
    (repo / "suite_test.py").write_text(
        "def test_one(): pass\ndef test_two(): pass\n", encoding="utf-8"
    )
    _git_in(repo, "add", "suite_test.py")
    _git_in(repo, "commit", "-qm", "c: one more test")

    code = counts.main(
        [
            "--repo-root",
            str(repo),
            "--baseline",
            str(baseline),
            "--scratch",
            str(_check_scratch(tmp_path)),
        ]
    )

    assert code == 2
    assert "no baseline is recorded" in capsys.readouterr().err


def test_an_entry_the_audit_would_reject_inherits_nothing(
    counts, tmp_path: Path, capsys
) -> None:
    """An inherited verdict is only as sound as the file's shape, so a broken entry refuses.

    An entry the shape audit would fail at this commit makes this commit's run
    differ from the recorded one by that failure, which is drift -- the one thing
    an inherited PASS must never paper over.
    """
    repo, baseline, measured = _repo_whose_tip_carries_a_recording(tmp_path / "repo")
    broken = _entry()
    broken.pop("skipped_ids_complete")
    baseline.write_text(
        json.dumps({"about": "x", "counts": {measured: broken}}), encoding="utf-8"
    )
    _git_in(repo, "add", "scripts/suite_counts_baseline.json")
    _git_in(repo, "commit", "-qm", "b: an entry with a field missing")

    code = counts.main(
        [
            "--repo-root",
            str(repo),
            "--baseline",
            str(baseline),
            "--scratch",
            str(_check_scratch(tmp_path)),
        ]
    )

    assert code == 2
    assert "no baseline is recorded" in capsys.readouterr().err


def test_the_shape_guard_refuses_the_entries_the_audit_refuses(counts) -> None:
    assert counts.entries_satisfy_the_recorded_shape({"a" * 40: _entry()}) is True
    assert counts.entries_satisfy_the_recorded_shape({"a" * 40: _entry(exit_code=1)}) is False
    assert counts.entries_satisfy_the_recorded_shape({"A" * 40: _entry()}) is False
    assert counts.entries_satisfy_the_recorded_shape({"HEAD": _entry()}) is False
    short = _entry()
    short.pop("skipped_ids")
    assert counts.entries_satisfy_the_recorded_shape({"a" * 40: short}) is False


def test_the_shape_guard_agrees_with_the_audit_on_the_real_file(counts) -> None:
    """One file, two readers: the guard must not disagree with the pinned audit."""
    assert counts.entries_satisfy_the_recorded_shape(counts.load_baseline(BASELINE)) is True


def test_the_tree_comparison_admits_when_it_cannot_compare(counts, tmp_path: Path) -> None:
    """'Could not tell' must not read as 'the same tree'; a missing commit is no witness."""
    repo = _scratch_repo(tmp_path / "repo", commits=2)

    assert counts.tree_diff_ignoring(repo, "HEAD", "HEAD~1", None) is True
    assert counts.tree_diff_ignoring(repo, "HEAD", "HEAD~1", "tracked.txt") is False
    assert counts.tree_diff_ignoring(repo, "HEAD", "0" * 40, "tracked.txt") is None


def test_recording_is_the_measuring_path_and_never_inherits(counts) -> None:
    """``--record`` promises a measurement, so it must not hand back a copied entry."""
    source = (SCRIPTS_DIR / "certify_suite_counts.py").read_text(encoding="utf-8")

    assert "if not args.record and recorded is None:" in source


# --- the availability gate CI runs on every head -------------------------------


def _no_suite_run(*args, **kwargs):
    raise AssertionError("the availability gate must not run the suite")


def _availability(counts, repo: Path, baseline: Path, tmp_path: Path, *extra: str) -> int:
    return counts.main(
        [
            "--repo-root",
            str(repo),
            "--baseline",
            str(baseline),
            "--scratch",
            str(_check_scratch(tmp_path)),
            "--availability",
            *extra,
        ]
    )


def test_the_availability_gate_passes_a_recorded_head_without_running_the_suite(
    counts, tmp_path: Path, capsys, monkeypatch
) -> None:
    """The gate answers from the baseline file: no run, no interpreter, no worktree.

    The distinction is the whole point of having it in CI: recording costs a
    full suite run (minutes) while answering "is this head answerable" costs a
    JSON lookup plus a two-tree diff (seconds), and only the second is cheap
    enough to run on every push.
    """
    repo, baseline, measured = _repo_whose_tip_carries_a_recording(tmp_path / "repo")
    monkeypatch.setattr(counts, "run_suite", _no_suite_run)

    code = _availability(counts, repo, baseline, tmp_path, measured)

    assert code == 0
    assert "recorded" in capsys.readouterr().out.lower()
    # Nothing was checked out: the gate must not pay the recording path's cost.
    assert len(_git_in(repo, "worktree", "list").splitlines()) == 1


def test_the_availability_gate_passes_the_follower_by_inheritance(
    counts, tmp_path: Path, capsys, monkeypatch
) -> None:
    """The commit that carries a recording is the case the gate exists to keep green."""
    repo, baseline, _ = _repo_whose_tip_carries_a_recording(tmp_path / "repo")
    monkeypatch.setattr(counts, "run_suite", _no_suite_run)

    code = _availability(counts, repo, baseline, tmp_path, "HEAD")

    assert code == 0
    assert "inherit" in capsys.readouterr().out.lower()
    assert len(_git_in(repo, "worktree", "list").splitlines()) == 1


def test_the_availability_gate_refuses_a_head_that_is_neither_recorded_nor_inheritable(
    counts, tmp_path: Path, capsys, monkeypatch
) -> None:
    """This refusal is the CI red the gate exists to produce."""
    repo, baseline, _ = _repo_whose_tip_carries_a_recording(tmp_path / "repo")
    (repo / "suite_test.py").write_text(
        "def test_one(): pass\ndef test_two(): pass\n", encoding="utf-8"
    )
    _git_in(repo, "add", "suite_test.py")
    _git_in(repo, "commit", "-qm", "c: one more test")
    monkeypatch.setattr(counts, "run_suite", _no_suite_run)

    code = _availability(counts, repo, baseline, tmp_path)

    assert code == 2
    error = capsys.readouterr().err
    assert "no baseline is recorded" in error
    assert "nothing to inherit" in error


def test_the_availability_gate_never_records(counts, tmp_path: Path, capsys) -> None:
    """``--availability`` answers whether counts exist; ``--record`` creates them.

    Combining them must refuse rather than silently measure: a gate that could
    record would be a gate that could rewrite the thing it is supposed to check.
    """
    repo, baseline, measured = _repo_whose_tip_carries_a_recording(tmp_path / "repo")

    code = _availability(counts, repo, baseline, tmp_path, "--record", measured)

    assert code == 2
    assert "--availability" in capsys.readouterr().err


def test_ci_refuses_a_head_whose_counts_are_neither_recorded_nor_inheritable() -> None:
    """The gate must run on every push, see real history, and be allowed to fail.

    Four separate ways this could silently stop covering anything: no job at
    all; a job gated to manual dispatch (a hunt job, not a gate); a shallow
    clone (the inheritance comparison cannot read a recorded commit the clone
    never fetched, so the gate would refuse every head for the wrong reason);
    and continue-on-error (a red nobody sees).
    """
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    gated = [
        (name, job)
        for name, job in jobs.items()
        if any(
            "certify_suite_counts.py" in str(step.get("run", ""))
            and "--availability" in str(step.get("run", ""))
            for step in job.get("steps", [])
        )
    ]
    assert gated, (
        "no CI job runs scripts/certify_suite_counts.py --availability, so a head "
        "whose suite counts are neither recorded nor inheritable is not refused anywhere"
    )
    _, job = gated[0]
    assert "workflow_dispatch" not in str(job.get("if", "")), (
        "the gate must run on pushes and pull requests, not only by hand"
    )
    assert not job.get("continue-on-error"), "a gate that cannot fail is not a gate"
    assert "timeout-minutes" in job, "the gate needs a bound of its own"

    checkout = next(
        step for step in job["steps"] if "actions/checkout" in str(step.get("uses", ""))
    )
    assert str(checkout.get("with", {}).get("fetch-depth")) == "0", (
        "the inheritance comparison diffs against recorded commits, which a shallow "
        "clone never fetched; the gate must check out full history"
    )
    gate_step = next(
        step for step in job["steps"] if "--availability" in str(step.get("run", ""))
    )
    assert not gate_step.get("continue-on-error")
