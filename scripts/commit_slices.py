#!/usr/bin/env python3
"""Commit stacked uncommitted work as separate index-level slice commits.

The extraction/hardening workflow this repo runs produces sessions where
several independent slices sit uncommitted in one tree, each already verified
against the full suite. Committing them as one lump destroys bisectability and
revertability; committing them by hand is tedious and error-prone when the
pile mixes my files with a concurrent session's. This script makes the
decomposition the unit of work: a *slice file* lists each slice's explicit
paths and message, and the script stages each slice's paths into the index
(nothing more — ``git add -A`` is exactly what this exists to prevent),
commits it in the house format, and finishes with the journal entry commit.

Nothing here inspects, stages, or touches any path not named in the slice
file. Untracked files are added only when a slice names them explicitly.
If any step fails, the script stops with the index exactly as the failed
step left it — already-committed slices stay committed (each is independently
revertable), and nothing beyond the failing slice was touched.

Usage:
    python scripts/commit_slices.py slices.txt [--dry-run] [--no-journal] [--verify]

The slice file format (``#`` starts a comment; blank lines are skipped):

    # --- <slice title shown in the summary> ---
    src/the_oracle/gui_vulkan.py
    tests/test_gui_vulkan_owner.py
    ! QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_gui_vulkan_owner.py -q
    >> Commit message body line one.
    >> Line two. (Blank ">>" lines become blank lines.)

Each slice is a ``# ---`` header, then its paths (one per line), then an
optional ``>>``-prefixed message body. ``!`` lines are the slice's *verify
commands*: with ``--verify``, each command runs (via the shell, in the repo
root, output streaming) right after the slice commits and before the next
slice is touched — a non-zero exit stops the run there. The failed slice
stays committed (it is independently revertable) and neither the remaining
slices nor the journal are touched, so a red slice never hides behind a
green journal. Without ``--verify``, ``!`` lines are parsed and validated
but not run. The first line of the body becomes the
commit subject; the rest (joined with blank-line separation rules below)
becomes the body. Slices that list no ``>>`` lines get a one-line commit from
the slice title.

The journal: ``--journal-entry "text"`` (repeatable) appends a
``YYYY-MM-DD | <files> (commit <hash>) | <text> | Committed via
commit_slices.py.`` line to JUNO_FIXES.log per slice, committed as its own
"Record ... in the journals" commit, matching the house pattern. Without it,
pass ``--no-journal`` explicitly so skipping the journal is a decision, not
an oversight.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT_FOOTER = (
    "🤖 Generated with Codebuff\n"
    "Co-Authored-By: Codebuff <noreply@codebuff.com>"
)


@dataclass
class Slice:
    """One index-level commit: explicit paths + a house-format message."""

    title: str
    paths: list[str] = field(default_factory=list)
    body_lines: list[str] = field(default_factory=list)
    journal_entries: list[str] = field(default_factory=list)
    verify_commands: list[str] = field(default_factory=list)

    @property
    def subject(self) -> str:
        """Commit subject: the first body line, or the slice title."""
        return self.body_lines[0] if self.body_lines else self.title

    @property
    def body(self) -> str:
        """Commit body: remaining ``>>`` lines, blank lines preserved."""
        return "\n\n".join(self.body_lines[1:])


def _run(args: list[str], repo: Path, *, dry: bool, capture: bool = False) -> str:
    """Run a git command in the repo, printing it first."""
    print(f"  $ git {' '.join(args)}")
    if dry:
        return ""
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=capture,
        text=True,
    )
    return result.stdout if capture else ""


def parse_slice_file(text: str) -> list[Slice]:
    """Parse the slice file into ordered slices.

    A ``# ---`` line starts a slice; every other non-comment line is a path
    or a ``>>`` message line. A path listed before the first header is a
    parsing error, not silent attachment to a phantom slice.
    """
    slices: list[Slice] = []
    current: Slice | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        is_comment = line.startswith("#") and not line.startswith("# ---")
        if not line or is_comment:
            continue
        if line.startswith("# ---"):
            title = line.lstrip("# ").strip().strip("-").strip()
            if not title:
                raise ValueError(f"slice file line {lineno}: empty slice title")
            current = Slice(title=title)
            slices.append(current)
            continue
        if line.startswith(">>"):
            message = line[2:].strip()
            if current is None:
                raise ValueError(f"slice file line {lineno}: message before any '# ---' header")
            current.body_lines.append(message)
            continue
        if line.startswith("!"):
            command = line[1:].strip()
            if current is None:
                raise ValueError(
                    f"slice file line {lineno}: verify command before any '# ---' header"
                )
            if not command:
                raise ValueError(f"slice file line {lineno}: empty verify command")
            current.verify_commands.append(command)
            continue
        if current is None:
            raise ValueError(f"slice file line {lineno}: path {line!r} before any '# ---' header")
        current.paths.append(line)
    return slices


def parse_journal_entries(slice_spec: list[str]) -> dict[int, list[str]]:
    """Map slice index (0-based, in file order) to journal entry texts."""
    out: dict[int, list[str]] = {}
    for spec in slice_spec:
        index_text, sep, entry = spec.partition(":")
        if not sep or not entry.strip():
            raise ValueError(
                f"--journal-entry must be '<slice-number>:<text>' (slice 1 = "
                f"first), got {spec!r}"
            )
        index = int(index_text) - 1
        out.setdefault(index, []).append(entry.strip())
    return out


def _dirty_paths(repo: Path) -> set[str]:
    """Every path git reports as dirty, tracked or untracked."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )
    paths: set[str] = set()
    for line in result.stdout.splitlines():
        # Porcelain v1: XY <path> (renames: 'R  old -> new'); we only need to
        # know what exists, so both sides of a rename count as dirty.
        path_field = line[3:]
        paths.update(part.strip() for part in path_field.split(" -> "))
    return paths


def validate(slices: list[Slice], repo: Path) -> None:
    """Refuse to start on a slice file that cannot be committed cleanly.

    Empty slices, paths that are neither dirty nor tracked, duplicates, and
    paths quoted with shell quoting (a copy-paste artifact) all stop the run
    before the first commit.
    """
    seen: set[str] = set()
    titles: set[str] = set()
    dirty = _dirty_paths(repo)
    for index, slice_ in enumerate(slices, start=1):
        if slice_.title in titles:
            raise ValueError(f"slice {index}: duplicate title {slice_.title!r}")
        titles.add(slice_.title)
        if not slice_.paths:
            raise ValueError(f"slice {index} ({slice_.title}): no paths listed")
        for path in slice_.paths:
            if "'" in path or '"' in path:
                raise ValueError(
                    f"slice {index}: path {path!r} contains a quote character — "
                    "list it unquoted; shell quoting is a copy-paste artifact"
                )
            if path in seen:
                raise ValueError(
                    f"slice {index}: path {path!r} is already claimed by an "
                    "earlier slice — each path commits exactly once"
                )
            seen.add(path)
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", path],
                cwd=str(repo),
                capture_output=True,
                text=True,
            ).returncode == 0
            if not tracked and path not in dirty:
                raise ValueError(
                    f"slice {index}: path {path!r} is neither tracked nor "
                    "dirty — a typo would silently commit nothing"
                )


_DATED_LINE_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2}) \| ")
_DATEISH_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}\b")
_COMMIT_CITATION_RE = re.compile(r"\(commits? (?P<body>[^)]*)\)")


def check_journal_lines(lines: list[str], *, floors: bool = True) -> list[str]:
    """Validate journal lines against the house format; return problem strings.

    The journal's convention has drifted over its life (a five-field era, a
    free-form era, the current four-field house format, and a sibling
    convention that folds the suite note into the description), so this
    validates what "malformed" means without rewriting history. A line is an
    ENTRY only when it starts with ``<ISO date> | ``; a line that looks
    date-ish but does not match that shape exactly (``2026-1-5``, glued
    dates) is malformed by definition — the silent-skip gap would hide
    exactly the entries most worth catching. Every other line (headers, the
    Format: doc line, historical free-form notes) is out of contract.

    Per-entry rules (always applied — this is what the pre-write refusal and
    --check-journal both police): a valid calendar date, at least two `` | ``
    fields, no empty field, and commit-citation integrity — a ``(commit ...)",
    ``(commits ...)`` body that is pure hex must be 7-40 digits (truncation
    breaks ``git rev-parse`` verification and the record-integrity net);
    non-hex bodies (``pending``, notes) and slash/comma-separated hash lists
    pass. Whole-file vacuity floors (``floors=True``, check mode only — a
    two-line batch at commit time would false-positive) keep the scan honest:
    fewer than ten entries or no citation at all means it went blind.
    """
    problems: list[str] = []
    entries = 0
    citations = 0
    for lineno, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _DATED_LINE_RE.match(line)
        if not match:
            if _DATEISH_RE.match(line):
                problems.append(
                    f"line {lineno}: date-ish line does not match the entry shape "
                    "'YYYY-MM-DD | ...' — malformed date or missing separator"
                )
            continue  # historical free-form: out of contract, not an error
        entries += 1
        fields = line.split(" | ")
        try:
            date.fromisoformat(match.group("date"))
        except ValueError:
            problems.append(
                f"line {lineno}: {match.group('date')!r} is not a valid calendar date"
            )
        if len(fields) < 2:
            problems.append(f"line {lineno}: entry has no content after the date")
        elif any(not field.strip() for field in fields):
            problems.append(f"line {lineno}: entry has an empty field")
        for citation in _COMMIT_CITATION_RE.finditer(line):
            body = citation.group("body").strip()
            citations += 1
            if not body:
                problems.append(f"line {lineno}: empty (commit ...) citation")
            elif re.fullmatch(r"[0-9a-f]+", body) and not 7 <= len(body) <= 40:
                problems.append(
                    f"line {lineno}: commit citation {body!r} is not 7-40 hex "
                    "digits — a truncated or garbled hash cannot be verified"
                )
    if floors:
        if entries < 10:
            problems.append(
                f"only {entries} dated entries found (floor 10) — the scan went blind"
            )
        if citations < 1:
            problems.append("no (commit <hash>) citation found — the scan went blind")
    return problems


def check_journal_file(journal: Path) -> list[str]:
    """Run :func:`check_journal_lines` over the journal file on disk."""
    return check_journal_lines(journal.read_text(encoding="utf-8").splitlines())


def run_verify(slice_: Slice, repo: Path, *, dry: bool) -> None:
    """Run the slice's verify commands; a failure stops the run.

    Commands run via the shell in the repo root (so env prefixes and pipes
    work) with output streaming to the caller. Non-zero exit means the slice
    did not prove itself green: the run stops here — the slice stays
    committed (independently revertable), the remaining slices are not
    touched, and the journal is not written.
    """
    for command in slice_.verify_commands:
        print(f"  $ {command}")
        if dry:
            continue
        result = subprocess.run(command, shell=True, cwd=str(repo))
        if result.returncode != 0:
            raise ValueError(
                f"verify command failed (exit {result.returncode}): {command!r} — "
                f"slice {slice_.title!r} stayed committed (revert it or fix "
                "forward); the remaining slices and the journal were not touched"
            )


def commit_slice(slice_: Slice, repo: Path, *, dry: bool) -> str:
    """Stage the slice's paths and commit them; return the commit hash."""
    _run(["add", *slice_.paths], repo, dry=dry)
    message = slice_.subject
    if slice_.body:
        message = f"{slice_.subject}\n\n{slice_.body}"
    message = f"{message}\n\n{COMMIT_FOOTER}"
    print(f"  message: {slice_.subject}")
    _run(["commit", "-m", message], repo, dry=dry)
    if dry:
        return ""
    hash_result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )
    return hash_result.stdout.strip()


def commit_journal(
    slices: list[Slice],
    hashes: list[str],
    entries_by_index: dict[int, list[str]],
    *,
    repo: Path,
    suite_note: str,
    dry: bool,
) -> None:
    """Append one journal line per slice and commit JUNO_FIXES.log.

    House format: ``YYYY-MM-DD | <files> (commit <hash>) | <entry text> |
    <suite note>`` — the caller supplies the entry text per slice and the
    suite note (the slice-time verification record) once for the run.
    """
    if dry:
        print("  $ (journal append + commit — skipped in dry run)")
        return
    today = date.today().isoformat()
    lines: list[str] = []
    for index, slice_ in enumerate(slices):
        for entry in entries_by_index.get(index, []):
            files = ", ".join(slice_.paths)
            hash_part = f"commit {hashes[index]}" if hashes[index] else "commit <uncommitted>"
            lines.append(
                f"{today} | {files} ({hash_part}) | {entry} | {suite_note}"
            )
    if not lines:
        print("  no journal entries configured — skipping the journal commit")
        return
    # Refuse BEFORE touching the file: a malformed entry must fail the run,
    # not silently degrade the record the --check-journal mode then polices.
    # Per-line rules only — the whole-file vacuity floors are check-mode
    # concerns, and a small batch at commit time would false-positive them.
    problems = check_journal_lines(lines, floors=False)
    if problems:
        raise ValueError(
            "refusing to write malformed journal entries:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
    journal = repo / "JUNO_FIXES.log"
    with journal.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    _run(["add", "JUNO_FIXES.log"], repo, dry=dry)
    _run(
        ["commit", "-m", "Record the committed slices in the journals\n\n" + COMMIT_FOOTER],
        repo,
        dry=dry,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "slice_file",
        type=Path,
        nargs="?",
        help="path to the slice file (required unless --check-journal)",
    )
    parser.add_argument(
        "--check-journal",
        action="store_true",
        help="validate JUNO_FIXES.log against the house format and exit "
        "non-zero on any malformed entry (no slice file needed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan without touching git"
    )
    parser.add_argument(
        "--journal-entry",
        action="append",
        dest="journal_entries",
        default=[],
        metavar="N:TEXT",
        help="journal text for slice N (1-based, in file order); repeatable",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run each slice's '!' verify commands after its commit; a "
        "failure stops the run before the remaining slices or the journal land",
    )
    parser.add_argument(
        "--no-journal",
        action="store_true",
        help="skip the journal commit deliberately (refuses to run without "
        "this or at least one --journal-entry)",
    )
    parser.add_argument(
        "--suite-note",
        default="Full suite green at slice time (see each slice's commit message).",
        help="the journal line's verification field (the house format's "
        "fourth field, e.g. 'Full suite: 1487 passed / 0 failed.')",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repo root (default: the checkout this script lives in)",
    )
    args = parser.parse_args(argv)
    repo = args.repo_root if args.repo_root is not None else DEFAULT_REPO_ROOT

    if args.check_journal:
        journal = repo / "JUNO_FIXES.log"
        if not journal.is_file():
            print(f"commit_slices: {journal} does not exist", file=sys.stderr)
            return 1
        problems = check_journal_file(journal)
        if problems:
            print(f"JUNO_FIXES.log: {len(problems)} malformed entr(y|ies) render:", file=sys.stderr)
            for problem in problems:
                print(f"  MALFORMED: {problem}", file=sys.stderr)
            return 1
        print("JUNO_FIXES.log: house format holds")
        return 0

    if args.slice_file is None:
        parser.error("a slice file is required unless --check-journal is passed")

    if not args.journal_entries and not args.no_journal:
        parser.error("pass --journal-entry or --no-journal explicitly")

    text = args.slice_file.read_text(encoding="utf-8")
    slices = parse_slice_file(text)
    if args.verify and not any(slice_.verify_commands for slice_ in slices):
        raise ValueError(
            "--verify passed but no slice defines a '!' verify command — "
            "add one per slice or drop the flag"
        )
    if not slices:
        print("slice file parsed to zero slices; nothing to do", file=sys.stderr)
        return 1
    validate(slices, repo)
    entries_by_index = parse_journal_entries(args.journal_entries)
    for index in entries_by_index:
        if index >= len(slices):
            raise ValueError(f"--journal-entry references slice {index + 1}, but only {len(slices)} slices exist")

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}{len(slices)} slice(s), in order:")
    for index, slice_ in enumerate(slices, start=1):
        print(f"  {index}. {slice_.title}")
        for path in slice_.paths:
            print(f"       {path}")

    hashes: list[str] = []
    for index, slice_ in enumerate(slices, start=1):
        print(f"\n[{index}/{len(slices)}] {slice_.title}")
        hashes.append(commit_slice(slice_, repo, dry=args.dry_run))
        if args.verify and slice_.verify_commands:
            run_verify(slice_, repo, dry=args.dry_run)

    if not args.no_journal:
        print("\n[journal]")
        commit_journal(slices, hashes, entries_by_index, repo=repo, suite_note=args.suite_note, dry=args.dry_run)

    print("\ndone:")
    for index, slice_ in enumerate(slices, start=1):
        print(f"  {hashes[index - 1] or '(dry)'} {slice_.subject}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"commit_slices: {exc}", file=sys.stderr)
        stderr = str(getattr(exc, "stderr", "") or "")
        if "index.lock" in stderr:
            print(
                "commit_slices: another git process holds .git/index.lock — "
                "nothing was committed; re-run once the other git operation "
                "finishes (a concurrent session may be committing).",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc
