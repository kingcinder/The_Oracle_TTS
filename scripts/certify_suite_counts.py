#!/usr/bin/env python3
"""Catch suite-count drift against a recorded baseline.

The suite's counts are the cheap summary of what it actually covered, and they
are not self-evident: the same commit yields a different skip count depending on
what the machine running it happens to have built. A git-ignored native build
that stops existing turns a smoke test into a skip, and a skip looks exactly
like a pass in a green run -- so the count is the only place it shows up.

That has been noticed twice by hand now (the Vulkan smoke's skip count moving
from 1 to 3 as more audio-gated tests were added), both times by someone
comparing two runs in their head. A number that only a human remembers is not a
check. So: the counts for a commit are recorded here, and re-deriving them later
on a fresh tree of that same commit either reproduces them or fails.

Why a fresh worktree, and why that makes the baseline machine-independent: the
tree is ``git worktree add --detach <ref>``, so it carries the committed files
and nothing else -- no ``.venv``, no git-ignored native build, no reference
clips left behind by an earlier run. A commit therefore always produces the
same counts under this check, on any machine, and any difference is real drift
rather than a description of whose laptop ran it.

It is a worktree rather than a ``git archive`` on purpose. An archive has no
``.git``, and that is not a neutral difference: the suite carries tests that
need the repository, which skip cleanly without it (``test_record_integrity``'s
commit-citation net) and one that *fails* without it
(``test_commit_slices``' landed-fix pin clones ``file://HEAD``). Recording a
baseline from an archive would bake in four spurious skips and a failure, and
would be a baseline of a tree nobody actually gets from a clone.

    python scripts/certify_suite_counts.py                    # HEAD against its baseline
    python scripts/certify_suite_counts.py ebc88f2            # an older commit
    python scripts/certify_suite_counts.py --record           # record/replace HEAD's entry
    python scripts/certify_suite_counts.py --pytest-args "-k vulkan"
    python scripts/certify_suite_counts.py --capture-output /tmp/run.txt

Exit 0 means the counts matched the recorded baseline. Exit 1 means they did
not: the drift is printed field by field with both values, and any skip that
appeared or disappeared is named by node id. Exit 2 means the check could not
run at all -- no such ref, no baseline recorded for that commit, no interpreter,
or no readable pytest summary. There is deliberately no "no baseline, carry on"
path: an unrecorded commit is a question, not a pass.

Re-recording is reported, not silent: ``--record`` prints every field it
*changed* from the previous entry, so accepting drift is a visible act rather
than the reflex that would make this script pointless.

Honest limits: a fresh worktree has no ``.venv``, so the suite runs under one
shared interpreter (this repo's, or ``--interpreter``), and the check proves the
*tree* is sound rather than that any machine has installed it. The suite is run
offscreen with no GPU, audio device, or display, so what it covers is what an
offscreen run covers -- including the skips the missing native build causes,
which are recorded rather than hidden. And a flake is indistinguishable here
from drift *within* one commit: both mean the same commit stopped producing the
same counts, which is the thing worth knowing either way.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "scripts" / "suite_counts_baseline.json"

#: What the baseline file says about itself, so a reader years from now knows
#: what the numbers are numbers *of*.
BASELINE_ABOUT = (
    "Suite counts from `pytest -q` run in a fresh `git worktree add --detach <ref>` tree of "
    "the named commit: a real checkout with .git and the committed files only, so no .venv "
    "and no git-ignored native build. Recorded and verified by scripts/certify_suite_counts.py; "
    "edit it through --record, not by hand."
)

#: The counts compared, in the order they are reported. ``passed`` and
#: ``skipped`` are the two a coverage drift moves; ``failed``, ``errors`` and
#: ``exit_code`` are here so a baseline can never be satisfied by a run that
#: quietly stopped passing.
COUNT_FIELDS = ("passed", "skipped", "failed", "errors", "exit_code")

#: Node ids of the skipped tests, so drift names *which* test started skipping
#: instead of only how many did.
SKIP_ID_FIELD = "skipped_ids"

#: Whether the audit that produced those ids listed every skip the summary
#: counted. It can fall short: ``tests/test_skip_audit.py`` has an autouse
#: fixture that calls ``conftest.pytest_sessionstart`` (which clears the skip
#: ledger) before and after each of its tests, so every skip recorded earlier in
#: the session is lost from the end-of-run audit. Measured: a run of the three
#: audio-gated files reports "1 test(s) did not run here" against pytest's own
#: "3 skipped". Counts come from pytest's summary and are always right; this flag
#: is what keeps a short id list from being read as the whole skip set -- and
#: what stops the list from reporting a wave of phantom "added" ids the day
#: someone fixes that fixture.
SKIP_ID_COMPLETE_FIELD = "skipped_ids_complete"

_SUMMARY_PATTERNS = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "errors": re.compile(r"(\d+) error"),
    "skipped": re.compile(r"(\d+) skipped"),
}

_AUDIT_HEADER = re.compile(r"^\s*(\d+) test\(s\) did not run here:\s*$")
#: The audit pairs each node id (2 spaces) with its reason (6 spaces, indented
#: under it). Both shapes continue the block; anything else ends it.
_AUDIT_ID = re.compile(r"^ {2}(\S+)\s*$")
_AUDIT_REASON = re.compile(r"^ {4,}\S")


class SuiteCountsError(RuntimeError):
    """The check could not be run, as opposed to having found drift."""


@dataclass(frozen=True)
class Drift:
    """One field whose value no longer matches the baseline."""

    field: str
    recorded: Any
    observed: Any


# --------------------------------------------------------------------------- #
# Acquiring the tree and the interpreter
# --------------------------------------------------------------------------- #


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args], capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise SuiteCountsError(
            f"git {' '.join(args)} failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout


def resolve_commit(repo_root: Path, ref: str) -> str:
    """The full commit sha ``ref`` names, or a refusal.

    Resolved to a sha rather than keyed by the ref itself, because a branch name
    means a different commit a week later and a baseline that follows a moving
    ref records nothing.
    """
    sha = _git(repo_root, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SuiteCountsError(f"could not resolve {ref!r} to a commit (got {sha!r})")
    return sha


def add_fresh_worktree(repo_root: Path, destination: Path, ref: str) -> Path:
    """Check ``ref`` out into a fresh detached worktree and return its path.

    A real checkout, so it has ``.git``: the tests that need a repository run
    here instead of skipping or failing (see the module docstring). What it does
    *not* have is any local state -- no ``.venv``, no git-ignored native build,
    no files an earlier run left behind -- because those are not in the commit,
    and that is what makes a recorded count the same on every machine.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "worktree", "add", "--detach", str(destination), ref)
    return destination


def remove_fresh_worktree(repo_root: Path, destination: Path) -> str | None:
    """Delete the worktree; return a complaint if it could not be removed.

    Deliberately not ``git worktree prune``: this repository is a shared
    checkout, and a prune would delete another session's worktree registration
    along with this one's. Only this run's path is named here.
    """
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(destination)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return (
            f"could not remove the worktree at {destination}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}. Remove it by hand "
            "(git worktree remove --force) so the repository stops tracking it."
        )
    return None


def resolve_interpreter(repo_root: Path, explicit: str | None) -> Path:
    """The interpreter the suite runs under.

    A fresh tree has no venv, so this is necessarily shared with the working
    tree; it is named in the report rather than assumed, because every count
    below is really "can *this* interpreter, running that tree's code, do this".
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise SuiteCountsError(f"no such interpreter: {path}")
        return path
    venv = repo_root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv.exists():
        return venv
    return Path(sys.executable)


def assert_imports_the_tree_under_test(tree: Path, interpreter: Path) -> str:
    """Fail unless code imported from ``tree`` resolves inside ``tree``.

    The whole check rests on this. This repo's interpreter has the package
    installed in editable mode pointing at the *working tree*, so a fresh-tree
    run that silently loaded that copy would report counts for code it never
    executed -- a number graded from something it did not measure, which is the
    exact failure mode a baseline exists to prevent.
    """
    # ``src`` is prepended the way pytest's ``pythonpath`` setting does it, so
    # this asks the same question the suite will.
    probe = (
        "import sys;"
        f"sys.path.insert(0, {str(tree / 'src')!r});"
        "import pathlib, the_oracle;"
        "print(pathlib.Path(the_oracle.__file__).resolve())"
    )
    completed = subprocess.run(
        [str(interpreter), "-c", probe],
        cwd=str(tree),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    resolved = (completed.stdout or "").strip().splitlines()
    if completed.returncode != 0 or not resolved:
        raise SuiteCountsError(
            f"could not import the_oracle from {tree}: {completed.stderr.strip()[:400]}"
        )
    imported = Path(resolved[-1])
    if not imported.is_relative_to(tree.resolve()):
        raise SuiteCountsError(
            "the tree under test did not load its own code: "
            f"imported {imported} while testing {tree}. Refusing to record or "
            "compare counts for a tree that was never exercised."
        )
    return str(imported)


# --------------------------------------------------------------------------- #
# Reading pytest's output
# --------------------------------------------------------------------------- #


def parse_suite_counts(output: str) -> dict[str, int]:
    """Counts out of pytest's final summary line.

    Only the last match of each pattern is read, because the skip audit prints
    its own line that also contains the word "skipped" -- counting every match
    would double-count it.
    """
    counts = dict.fromkeys(_SUMMARY_PATTERNS, 0)
    for key, pattern in _SUMMARY_PATTERNS.items():
        matches = pattern.findall(output)
        if matches:
            counts[key] = int(matches[-1])
    return counts


def parse_skipped_ids(output: str) -> list[str]:
    """The node ids the suite's own skip audit listed.

    Read from the audit block rather than recomputed here: the audit is what the
    run says about itself, and a second implementation of "what skipped" would be
    a second thing to keep in step with pytest.
    """
    lines = output.splitlines()
    ids: list[str] = []
    for index, line in enumerate(lines):
        if not _AUDIT_HEADER.match(line):
            continue
        for following in lines[index + 1 :]:
            match = _AUDIT_ID.match(following)
            if match is not None:
                ids.append(match.group(1))
                continue
            # A reason line belongs to the id above it: skip it rather than
            # ending the block, or every skip after the first would be missed.
            if _AUDIT_REASON.match(following):
                continue
            break
    return sorted(set(ids))


def id_audit_is_complete(counts: Mapping[str, Any], ids: list[str]) -> bool:
    """Whether the audit listed as many skips as pytest's summary counted."""
    return len(ids) == counts.get("skipped", 0)


def run_suite(tree: Path, interpreter: Path, extra_args: tuple[str, ...] = ()) -> tuple[dict[str, Any], str]:
    """Run the full suite in ``tree``; return its counts and the raw output.

    The output comes back too, so a mismatch can be explained from the bytes the
    run actually produced (``--capture-output``) rather than by re-running it.
    """
    argv = [str(interpreter), "-m", "pytest", "-q", *extra_args]
    completed = subprocess.run(
        argv,
        cwd=str(tree),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout + completed.stderr
    counts = parse_suite_counts(output)
    if not any(counts.values()):
        raise SuiteCountsError(
            "could not read pytest's summary from the run, so there is nothing to compare:\n"
            + "\n".join(line for line in output.strip().splitlines()[-12:] if line.strip())
        )
    ids = parse_skipped_ids(output)
    return {
        **counts,
        "exit_code": completed.returncode,
        SKIP_ID_FIELD: ids,
        SKIP_ID_COMPLETE_FIELD: id_audit_is_complete(counts, ids),
    }, output


# --------------------------------------------------------------------------- #
# The baseline
# --------------------------------------------------------------------------- #


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, dict[str, Any]]:
    """The recorded counts, keyed by full commit sha."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise SuiteCountsError(f"{path} is not readable JSON: {error}") from error
    # A hand-edited file must be refused with the documented exit code, not crash
    # on an attribute the shape never promised.
    if not isinstance(payload, dict):
        raise SuiteCountsError(f"{path} is a {type(payload).__name__}, not an object with 'counts'")
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise SuiteCountsError(f"{path} has no 'counts' object; expected the shape --record writes")
    return counts


def save_baseline(counts: Mapping[str, Mapping[str, Any]], path: Path = BASELINE_PATH) -> None:
    path.write_text(
        json.dumps({"about": BASELINE_ABOUT, "counts": counts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compare(recorded: Mapping[str, Any], observed: Mapping[str, Any]) -> list[Drift]:
    """Every field whose value no longer matches, in reporting order.

    Skip ids are compared only when both sides reported a complete audit. A
    short list compared against a complete one would invent drift that did not
    happen; the skip *count* is compared either way, so a test that started
    skipping is still caught.
    """
    drifts = [
        Drift(field, recorded.get(field), observed.get(field))
        for field in COUNT_FIELDS
        if recorded.get(field) != observed.get(field)
    ]
    if not (recorded.get(SKIP_ID_COMPLETE_FIELD) and observed.get(SKIP_ID_COMPLETE_FIELD)):
        return drifts
    before = set(recorded.get(SKIP_ID_FIELD) or [])
    after = set(observed.get(SKIP_ID_FIELD) or [])
    if after - before:
        drifts.append(Drift("skipped_ids_added", [], sorted(after - before)))
    if before - after:
        drifts.append(Drift("skipped_ids_removed", sorted(before - after), []))
    return drifts


def changed_fields(recorded: Mapping[str, Any], observed: Mapping[str, Any]) -> list[Drift]:
    """Every baseline field that differs, for the record path's report.

    Wider than :func:`compare` on purpose: this is not a verdict, it is the
    record of what an operator is overwriting, and a field the verdict
    deliberately ignores (the audit's completeness) still belongs in it -- the
    entry really did change when that field was added, and saying nothing would
    make ``--record`` quieter than the docstring promises.
    """
    fields = (*COUNT_FIELDS, SKIP_ID_FIELD, SKIP_ID_COMPLETE_FIELD)
    return [
        Drift(field, recorded.get(field), observed.get(field))
        for field in fields
        if recorded.get(field) != observed.get(field)
    ]


def id_comparison_note(recorded: Mapping[str, Any], observed: Mapping[str, Any]) -> str | None:
    """Why skip ids were not compared, or ``None`` when they were.

    Printed rather than left implicit: a silently un-compared field is how a
    check quietly stops covering something, which is the whole subject of this
    repository's skip doctrine.
    """
    if recorded.get(SKIP_ID_COMPLETE_FIELD) and observed.get(SKIP_ID_COMPLETE_FIELD):
        return None
    sides = []
    for label, entry in (("recorded", recorded), ("observed", observed)):
        if not entry.get(SKIP_ID_COMPLETE_FIELD):
            sides.append(
                f"{label} listed {len(entry.get(SKIP_ID_FIELD) or [])} of {entry.get('skipped', 0)}"
            )
    return (
        "skip ids were NOT compared ("
        + "; ".join(sides)
        + "); the skip count above still was. Cause on this repo: tests/test_skip_audit.py's "
        "autouse fixture clears conftest's skip ledger mid-run."
    )


def format_drift(commit: str, drifts: list[Drift]) -> list[str]:
    """The lines that explain a failure, addressed to whoever has to fix it."""
    lines = [
        f"FAIL: the suite counts for {commit[:12]} no longer match the recorded baseline.",
        "",
        "  field            recorded      observed",
        "  ---------------  ------------  ------------",
    ]
    for drift in drifts:
        lines.append(f"  {drift.field:<15}  {_render(drift.recorded):<12}  {_render(drift.observed)}")
    lines.append("")
    lines.append(
        "A count that moved means this commit no longer covers what it covered when it was "
        "recorded, which is drift rather than a detail: a skip looks like a pass in a green run."
    )
    lines.append(
        f"If the new counts are the honest ones, record them (--record) and say why in the "
        f"commit message; if they are not, fix what stopped running."
    )
    return lines


def _render(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "(none)"
    return str(value)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ref", nargs="?", default="HEAD", help="Commit to check (default: HEAD).")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record this commit's counts instead of comparing them.",
    )
    parser.add_argument("--interpreter", default=None, help="Python the suite runs under.")
    parser.add_argument("--pytest-args", default="", help="Extra arguments for the suite.")
    parser.add_argument(
        "--capture-output",
        type=Path,
        default=None,
        help="Write the suite's raw output here, so a mismatch can be read rather than re-run.",
    )
    parser.add_argument("--scratch", type=Path, default=None, help="Where to build the fresh tree.")
    parser.add_argument("--keep", action="store_true", help="Keep the fresh tree instead of deleting it.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.expanduser().resolve()
    scratch_parent = args.scratch.expanduser().resolve() if args.scratch else None
    if scratch_parent is not None:
        # ``mkdtemp(dir=...)`` needs its parent to exist, and naming a scratch
        # directory means "put it there" rather than "it is already there" --
        # without this the run died in a traceback, which also collided with the
        # exit code for drift.
        scratch_parent.mkdir(parents=True, exist_ok=True)
    fresh_root = Path(
        tempfile.mkdtemp(prefix="oracle-suite-counts-", dir=str(scratch_parent) if scratch_parent else None)
    ) / "tree"

    # Whether a worktree was actually added: every refusal above the add (no
    # such ref, no baseline, no interpreter) must not go on to "remove" one,
    # which reported a spurious warning about a path git never created.
    added = False
    try:
        commit = resolve_commit(repo_root, args.ref)
        interpreter = resolve_interpreter(repo_root, args.interpreter)
        counts = load_baseline(args.baseline)
        recorded = counts.get(commit)
        if not args.record and recorded is None:
            raise SuiteCountsError(
                f"no baseline is recorded for {commit[:12]} ({args.ref}). Record it with "
                f"`python scripts/certify_suite_counts.py {args.ref} --record` -- an unrecorded "
                "commit is a question, not a pass."
            )

        add_fresh_worktree(repo_root, fresh_root, commit)
        added = True
        imported = assert_imports_the_tree_under_test(fresh_root, interpreter)
        print(f"Checking {args.ref} -> {commit[:12]} in a fresh worktree of that commit")
        print(f"  tree        : {fresh_root}  (detached, no .venv, no git-ignored build)")
        print(f"  interpreter : {interpreter}  (shared: a fresh tree has no venv)")
        print(f"  imported    : {imported}  (asserted to be inside the fresh tree)")

        observed, output = run_suite(fresh_root, interpreter, tuple(args.pytest_args.split()))
        if args.capture_output:
            args.capture_output.expanduser().write_text(output, encoding="utf-8")
            print(f"  raw output  : {args.capture_output}")
        print(
            "  observed    : "
            + " ".join(f"{field}={observed[field]}" for field in COUNT_FIELDS)
        )
        note = id_comparison_note(recorded or {}, observed)
        if note:
            print(f"  note        : {note}")

        if args.record:
            if recorded == observed:
                print(f"Baseline for {commit[:12]} already matches; nothing changed.")
                return 0
            for drift in changed_fields(recorded or {}, observed):
                print(
                    f"  RE-RECORDED {drift.field}: {_render(drift.recorded)} -> "
                    f"{_render(drift.observed)}"
                )
            counts[commit] = observed
            save_baseline(counts, args.baseline)
            print(f"Recorded the baseline for {commit[:12]} in {args.baseline}.")
            return 0

        drifts = compare(recorded or {}, observed)
        if drifts:
            for line in format_drift(commit, drifts):
                print(line)
            return 1
        print(
            f"  recorded    : "
            + " ".join(f"{field}={recorded[field]}" for field in COUNT_FIELDS)
        )
        print(f"PASS: {commit[:12]} reproduces its recorded suite counts.")
        return 0
    except SuiteCountsError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    finally:
        if args.keep:
            print(f"Fresh worktree kept at {fresh_root}", file=sys.stderr)
        elif not added:
            # Nothing was checked out (the run refused first); drop the empty
            # scratch directory rather than reporting a removal that cannot work.
            shutil.rmtree(fresh_root.parent, ignore_errors=True)
        else:
            complaint = remove_fresh_worktree(repo_root, fresh_root)
            if complaint:
                # Keep the tree: deleting its files while git still tracks the
                # registration would leave a half-removed worktree behind.
                print(f"WARNING: {complaint}", file=sys.stderr)
            else:
                shutil.rmtree(fresh_root.parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
