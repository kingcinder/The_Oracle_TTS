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
    python scripts/commit_slices.py slices.txt [--dry-run] [--no-journal]

The slice file format (``#`` starts a comment; blank lines are skipped):

    # --- <slice title shown in the summary> ---
    src/the_oracle/gui_vulkan.py
    tests/test_gui_vulkan_owner.py
    >> Commit message body line one.
    >> Line two. (Blank ">>" lines become blank lines.)

Each slice is a ``# ---`` header, then its paths (one per line), then an
optional ``>>``-prefixed message body. The first line of the body becomes the
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
    parser.add_argument("slice_file", type=Path, help="path to the slice file")
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

    if not args.journal_entries and not args.no_journal:
        parser.error("pass --journal-entry or --no-journal explicitly")

    text = args.slice_file.read_text(encoding="utf-8")
    slices = parse_slice_file(text)
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
