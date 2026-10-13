#!/usr/bin/env python3
"""Prove a check catches a mutation — apply, run, revert, hash-guarded.

The project's mutation discipline is done by hand: edit one line, watch exactly
one check fail, put the bytes back. That discipline has a trap, and it has
already bitten once (STATE.md, 2026-10-10): CPython keys cached bytecode on
**(source mtime in whole seconds, source size)**. A mutation that rewrites a
file and a revert that restores the SAME LENGTH inside the SAME SECOND leave the
``.pyc`` looking current, so the next run executes bytecode that no longer
matches any file on disk — a measurement that lies, silently, while every hash
the tool can see reads clean. That run was caught by luck (a runtime value had
been asserted) and thrown away and redone with ``PYTHONPYCACHEPREFIX`` by hand.

This makes the hand-repair the default. Every child runs with
``PYTHONPYCACHEPREFIX`` pointed at a **fresh throwaway cache root per run**, so
it can read neither the repo's ``__pycache__`` nor an entry written by an earlier
phase of the same probe — an isolated-but-REUSED prefix is not enough, because
the same-second key collides inside a private cache too. Nested processes
inherit the variable, which is what makes probing ``pytest`` safe for pytest's
own subprocesses. On top of that, the moment this module rewrites a source it
also drops any cached ``.pyc`` that would still be considered current for it,
which protects a caller who checks the mutation from its own process.

Every phase is hash-guarded. The file's sha256 is taken before the mutation; the
mutated bytes must differ; the restore must reproduce the original digest
exactly. A restore that does not is reported as a refusal, never swallowed. A
file that changed under the probe — this is a shared checkout, other agents edit
it — is NOT clobbered: the revert refuses, names the path and both digests, and
leaves the bytes exactly as found for a human.

Usage::

    python scripts/mutation_probe.py --spec /tmp/mut.json --expect-rc 1 -- \\
        .venv/bin/python -m pytest -q tests/test_gui_sections_chrome.py

A mutation this project has already run by hand does not need a spec file:
``--list-presets`` prints the catalog and ``--preset NAME`` runs one with its
proof command, its expected status and its expected evidence already attached —
the certifier-level gate mutation from STATE.md's 2026-10-12 entry, for
instance: ungate the Live header, sweep all six themes, watch the gate fail.
Every mutation a preset names still carries this module's own guards.

Or from Python::

    with Probe([Mutation(Path("src/x.py"), "old", "new")]) as probe:
        proc = probe.run([sys.executable, "-m", "pytest", "-q", "tests/y.py"])
        assert proc.returncode == 1

Spec file: a JSON list of mutation objects, each
``{"path": ..., "old": ..., "new": ..., "count"?: 1, "expect_sha256"?: ...}``.
``count`` is how many occurrences must be replaced — the mutation has to be
unambiguous; ``expect_sha256`` refuses to touch a revision the caller did not
measure against.

Exit codes: **0** the check CAUGHT the mutation (the child exited nonzero, or
exactly ``--expect-rc`` when one is named); **1** the mutation SURVIVED (the child
exited 0), so the check does not cover that line; **2** nothing was concluded — a
guard refused, the spec was unusable, the child timed out, or the child exited
nonzero with a status OTHER than the ``--expect-rc`` you named. That last one
matters: a collection error or a crash is not a caught mutation, and reporting it
as one is the same class of lie as reading stale bytecode.

``--require-stdout TEXT`` narrows CAUGHT further: the status has to be the one
asked for AND the child's output (stdout then stderr) has to contain TEXT, or
the run is INVALID rather than caught. A nonzero status means *something*
failed; the text is what says the thing this mutation targets is what failed.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

#: Prefix of the probe's throwaway root under the system temp directory. The
#: root holds one cache directory per run and nothing else.
_SCRATCH_PREFIX = "oracle_mutation_probe_"

#: The repository this script ships in. A preset names its files and its proof
#: command relative to HERE, so ``--preset`` works from any working directory
#: (the proof command runs with this as its cwd), while a ``--spec`` keeps
#: whatever meaning the caller's own relative paths have.
REPO_ROOT = Path(__file__).resolve().parents[1]


class GuardRefused(Exception):
    """A hash or spec guard refused to touch (or restore) a file."""


def digest(data: bytes) -> str:
    """The sha256 every guard in this module compares."""
    return hashlib.sha256(data).hexdigest()


def _reap_scratch(root: Path) -> None:
    """Remove a scratch directory, retrying, and REPORT a failure.

    Same doctrine as the certifier family's ``reap_isolated_root`` in
    ``scripts/certifier_isolation.py``: a directory that survives one ``rmtree``
    is reported on stderr rather than swallowed, and the removal is registered
    with ``atexit`` so the exits that skip ``finally`` (a usage error, a
    signal-time exit) do not plant the root in ``/tmp`` for good.
    """
    for _ in range(5):
        shutil.rmtree(root, ignore_errors=True)
        if not root.exists():
            return
        time.sleep(0.05)
    print(
        f"mutation_probe: could not remove its scratch root {root}; remove it by hand",
        file=sys.stderr,
    )


def default_cache_path(source: Path) -> Path:
    """The ``.pyc`` an interpreter with NO pycache prefix would use for *source*."""
    tag = sys.implementation.cache_tag or "cpython"
    return source.parent / "__pycache__" / f"{source.stem}.{tag}.pyc"


def candidate_cache_paths(source: Path) -> tuple[Path, ...]:
    """Every cache file the import system could execute for *source*.

    The default (repo-side) location first, then the one
    ``importlib.util.cache_from_source`` reports — which honors the CALLING
    process's ``sys.pycache_prefix``, and a prefix is exactly what this module
    sets for its children. Asking that function alone would therefore look only
    at the isolated root and go blind to the repo-side cache that a plain
    ``pytest`` reads, which is where the recorded trap lived (found by running
    this module's own tests under a prefix).
    """
    paths = [default_cache_path(source)]
    try:
        prefixed = Path(importlib.util.cache_from_source(str(source)))
    except (NotImplementedError, ValueError):
        return tuple(paths)
    if prefixed != paths[0]:
        paths.append(prefixed)
    return tuple(paths)


def _cache_matches_source(cache: Path, source: Path) -> bool:
    """The import system's own freshness test: header key == source's key.

    The 16-byte header of the default (timestamp) invalidation mode is magic,
    flags, source mtime in WHOLE SECONDS, and source size. Equal means the
    interpreter executes that bytecode without reading the source — the state a
    fast mutate/revert cycle can leave behind.
    """
    try:
        header = cache.read_bytes()[:16]
        stat = source.stat()
    except OSError:
        return False
    if len(header) != 16:
        return False
    _magic, _flags, mtime, size = struct.unpack("<4sIII", header)
    return mtime == (int(stat.st_mtime) & 0xFFFFFFFF) and size == (stat.st_size & 0xFFFFFFFF)


def cached_bytecode_is_current(source: Path) -> Path | None:
    """A cached ``.pyc`` the import system would treat as current, or None."""
    for cache in candidate_cache_paths(source):
        if cache.is_file() and _cache_matches_source(cache, source):
            return cache
    return None


def forget_hazardous_bytecode(source: Path) -> list[str]:
    """Drop every cached ``.pyc`` the import system would still treat as current.

    Called right after this module rewrites *source*, which is the one moment a
    matching key can only mean trouble: the entry is either stale (the trap) or
    freshly compiled from the bytes just written, and dropping either costs one
    recompile. Returns one human-readable line per hazard found, for the report —
    an unlink that fails is REPORTED, never swallowed.
    """
    hazards: list[str] = []
    for cache in dict.fromkeys(candidate_cache_paths(source)):
        if not cache.is_file() or not _cache_matches_source(cache, source):
            continue
        try:
            cache.unlink()
        except OSError as exc:
            hazards.append(
                f"{cache.name} still looks current for {source.name} and could not be "
                f"removed ({exc})"
            )
        else:
            hazards.append(f"{cache.name} still looked current for {source.name}; dropped it")
    return hazards


@dataclass(frozen=True)
class Mutation:
    """One source edit: replace every occurrence of ``old`` with ``new``."""

    path: Path
    old: str
    new: str
    count: int = 1
    expect_sha256: str | None = None

    @classmethod
    def from_payload(cls, payload: object, *, index: int) -> Mutation:
        """Build from one JSON object, refusing anything ambiguous."""
        if not isinstance(payload, dict):
            raise GuardRefused(
                f"mutation {index}: expected a JSON object, got {type(payload).__name__}"
            )
        missing = [key for key in ("path", "old", "new") if key not in payload]
        if missing:
            raise GuardRefused(f"mutation {index}: missing {', '.join(missing)}")
        count = payload.get("count", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise GuardRefused(f"mutation {index}: count must be a positive integer")
        if payload["old"] == payload["new"]:
            raise GuardRefused(f"mutation {index}: old and new are identical")
        expected = payload.get("expect_sha256")
        if expected is not None and not isinstance(expected, str):
            raise GuardRefused(f"mutation {index}: expect_sha256 must be a string")
        return cls(
            path=Path(str(payload["path"])),
            old=str(payload["old"]),
            new=str(payload["new"]),
            count=count,
            expect_sha256=expected,
        )


@dataclass(frozen=True)
class Preset:
    """A mutation this repo has already run by hand, packaged with its check.

    The four things a hand run leaves in a commit message instead of in a tool:
    which bytes to change, which command proves the change is seen, what status
    that command must exit with, and what its output must say. Recording all
    four is what makes a later run MEASURE the same thing rather than
    approximately it — a preset's ``require_stdout`` is the difference between
    "the gate fired" and "something broke".

    ``mutations`` may name repo-relative paths (resolved against
    ``REPO_ROOT``); the preset's proof command runs with ``REPO_ROOT`` as its
    cwd, so it is written the way the documentation writes it. Everything the
    spec path already guards still applies — this is a spec with a command
    attached, not a bypass of one.
    """

    name: str
    what: str
    mutations: tuple[Mutation, ...]
    command: tuple[str, ...]
    expect_rc: int | None = None
    require_stdout: str | None = None
    evidence: str = ""

    def resolve(self) -> list[Mutation]:
        """The mutations, with repo-relative paths anchored at the repo root."""
        return [
            mutation
            if mutation.path.is_absolute()
            else replace(mutation, path=REPO_ROOT / mutation.path)
            for mutation in self.mutations
        ]

    def describe(self) -> str:
        """One catalog line: what it mutates, and the proof it runs."""
        lines = [f"{self.name}: {self.what}"]
        for mutation in self.mutations:
            lines.append(f"    mutates {mutation.path} (count {mutation.count})")
        lines.append(f"    proves  {shlex.join(self.command)}")
        if self.expect_rc is not None:
            lines.append(f"    expects status {self.expect_rc}")
        if self.require_stdout is not None:
            lines.append(f"    expects output containing {self.require_stdout!r}")
        if self.evidence:
            lines.append(f"    recorded: {self.evidence}")
        return "\n".join(lines)


# The certifier-level gate mutation from STATE.md's 2026-10-12 (second slice)
# entry, recorded as the tool it used to be: ungate gui_chrome's Live header
# and watch certify_gui's chrome aspect fail all six themes. The replacement
# DEDENTS rather than merely deleting the ``with`` line — a 12-space suite
# directly after an 8-space statement is an IndentationError, and a mutation
# that cannot import proves nothing (pytest exits 4, which this tool already
# refuses to call caught).
_LIVE_HEADER_GATE_OLD = (
    "        with bulk_restyle():\n"
    '            header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px 0;")\n'
)
_LIVE_HEADER_GATE_NEW = (
    '        header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px 0;")\n'
)

_PRESETS: tuple[Preset, ...] = (
    Preset(
        name="certifier-gate-live-header",
        what="ungate gui_chrome's Live header setStyleSheet, so the certifier's "
        "chrome aspect has to fail the sweep on a CLOSED bulk-restyle gate",
        mutations=(
            Mutation(
                path=Path("src/the_oracle/gui_chrome.py"),
                old=_LIVE_HEADER_GATE_OLD,
                new=_LIVE_HEADER_GATE_NEW,
                count=1,
            ),
        ),
        command=(sys.executable, "scripts/certify_gui.py", "--aspects", "chrome"),
        expect_rc=1,
        require_stdout="bulk-restyle gate CLOSED",
        evidence="STATE.md 2026-10-12 (second slice): all six themes failed "
        "with the gate CLOSED, 0 \"all open\" notes, restored byte-identical (sha256)",
    ),
)

#: Name to preset. Built from the tuple so a ``name`` that disagrees with its
#: own catalog key cannot exist.
PRESETS: dict[str, Preset] = {preset.name: preset for preset in _PRESETS}


@dataclass
class FileRecord:
    """The hash trail of one mutated file, as reported."""

    path: Path
    original_sha256: str
    mutated_sha256: str
    restored_sha256: str | None = None
    conflict: str | None = None
    hazards: list[str] = field(default_factory=list)

    @property
    def byte_identical(self) -> bool:
        return self.restored_sha256 == self.original_sha256


@dataclass
class RunRecord:
    """One child run: what was executed, what it exited with, how long."""

    command: list[str]
    returncode: int
    seconds: float
    timed_out: bool
    stdout: str
    stderr: str


class Probe:
    """Applies mutations, runs a child on isolated bytecode, reverts.

    Drive it as a context manager (``with Probe(specs) as probe: ...``) or by
    hand (``apply()``, ``run()``, ``revert()``, ``close()``). The revert is
    attempted on every exit path and its outcome is part of the report rather
    than an assumption — "byte-identical" is a measured fact here, not a hope,
    because that is exactly the property a mutation run needs to be reusable.
    """

    def __init__(
        self,
        mutations: list[Mutation] | tuple[Mutation, ...] = (),
        *,
        scratch: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        self.mutations = list(mutations)
        self.timeout = timeout
        self._env_extra = dict(env or {})
        self._scratch = Path(scratch) if scratch is not None else None
        self._owns_scratch = scratch is None
        self.records: list[FileRecord] = []
        self.runs: list[RunRecord] = []
        self._by_path: dict[Path, FileRecord] = {}
        self._originals: dict[Path, bytes] = {}
        self._mutated: dict[Path, bytes] = {}
        self._applied = False

    # -- scratch ---------------------------------------------------------

    @property
    def scratch(self) -> Path:
        """The throwaway root for bytecode caches, created on first use."""
        if self._scratch is None:
            self._scratch = Path(tempfile.mkdtemp(prefix=_SCRATCH_PREFIX))
            atexit.register(_reap_scratch, self._scratch)
        return self._scratch

    def close(self) -> None:
        """Reap a scratch root this probe created (a caller's root is kept)."""
        if self._owns_scratch and self._scratch is not None:
            _reap_scratch(self._scratch)

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> Probe:
        self.apply()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.revert()
        except GuardRefused as refusal:
            if exc_type is None:
                raise
            # Never mask the exception in flight; report the refusal instead.
            print(
                f"mutation_probe: revert refused while handling {exc_type.__name__}: {refusal}",
                file=sys.stderr,
            )
        finally:
            self.close()
        return False

    def apply(self) -> None:
        """Write every mutation, or leave the tree as it was.

        All-or-nothing on purpose: a probe that applied half of a spec and then
        refused would leave the checkout mutated, which is the one state this
        harness must never create.
        """
        if self._applied:
            raise GuardRefused("apply() was already called on this probe")
        try:
            for mutation in self.mutations:
                self._apply_one(mutation)
        except Exception:
            try:
                self.revert()
            except GuardRefused as refusal:
                print(f"mutation_probe: could not clean up a partial apply: {refusal}",
                      file=sys.stderr)
            raise
        self._applied = True

    def _apply_one(self, mutation: Mutation) -> None:
        path = mutation.path
        if not path.is_file():
            raise GuardRefused(f"{path}: not a file")
        original = path.read_bytes()
        before = digest(original)
        if mutation.expect_sha256 is not None and before != mutation.expect_sha256:
            raise GuardRefused(
                f"{path}: sha256 {before} is not the expected {mutation.expect_sha256} — "
                "refusing to mutate a revision the caller did not measure against"
            )
        text = original.decode("utf-8")
        found = text.count(mutation.old)
        if found != mutation.count:
            raise GuardRefused(
                f"{path}: the target appears {found} time(s), expected {mutation.count} — "
                "the mutation has to be unambiguous"
            )
        mutated = text.replace(mutation.old, mutation.new).encode("utf-8")
        if mutated == original:
            raise GuardRefused(f"{path}: the replacement does not change the file's bytes")
        existing = self._by_path.get(path)
        if existing is None:
            record = FileRecord(path=path, original_sha256=before, mutated_sha256=digest(mutated))
            self.records.append(record)
            self._by_path[path] = record
            # The FIRST pre-image for this path. Every later bite of the same
            # file has to revert to this one; see the branch below.
            self._originals[path] = original
        else:
            # A second bite at a file this probe already holds. The record and
            # the original STAY as they are — the original is the pre-image
            # from before the first bite, which is what the restore owes the
            # caller — while only the mutated bytes move forward. Overwriting
            # them here (as this did until 2026-10-13, found driving a
            # two-hunk spec at .github/workflows/ci.yml) made revert() put back
            # the MID-CHAIN bytes: the run then reported CAUGHT and exited 0
            # with the checkout left half-mutated.
            record = existing
            record.mutated_sha256 = digest(mutated)
        path.write_bytes(mutated)
        self._mutated[path] = mutated
        record.hazards.extend(forget_hazardous_bytecode(path))

    def revert(self) -> None:
        """Put the original bytes back — verified, and never over a foreign edit."""
        refusals: list[str] = []
        for path, mutated in list(self._mutated.items()):
            current = path.read_bytes()
            if current != mutated:
                message = (
                    f"{path}: the file changed while the probe ran (sha256 {digest(current)} "
                    f"is neither the mutated {digest(mutated)} nor this probe's to overwrite) — "
                    "left exactly as found"
                )
                self._by_path[path].conflict = message
                refusals.append(message)
                continue
            record = self._by_path[path]
            path.write_bytes(self._originals[path])
            record.restored_sha256 = digest(path.read_bytes())
            record.hazards.extend(forget_hazardous_bytecode(path))
            del self._mutated[path]
            if record.restored_sha256 != record.original_sha256:
                refusals.append(
                    f"{path}: the restore produced {record.restored_sha256}, not the original "
                    f"{record.original_sha256}"
                )
        if refusals:
            raise GuardRefused("; ".join(refusals))

    # -- running ---------------------------------------------------------

    def run(
        self,
        command: list[str] | tuple[str, ...],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        cwd: str | Path | None = None,
    ) -> subprocess.CompletedProcess:
        """Run *command* with a fresh throwaway bytecode cache root.

        The fresh root per run is the whole guarantee: the interpreter cannot
        read the repo's ``__pycache__`` (``PYTHONPYCACHEPREFIX`` moves the
        lookup) and cannot read an entry left by an earlier phase of this probe
        (every run gets an empty one), so the bytes it executes are the bytes on
        disk. The root is reaped after the run and the result is recorded, so a
        timeout is reported as a timeout rather than a verdict.
        """
        cache_root = self.scratch / f"pycache-{len(self.runs) + 1}"
        cache_root.mkdir(parents=True, exist_ok=True)
        child_env = {
            **os.environ,
            **self._env_extra,
            **(env or {}),
            "PYTHONPYCACHEPREFIX": str(cache_root),
        }
        limit = self.timeout if timeout is None else timeout
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                list(command),
                env=child_env,
                cwd=None if cwd is None else str(cwd),
                capture_output=True,
                text=True,
                timeout=limit,
            )
        except subprocess.TimeoutExpired as expired:
            timed_out = True
            proc = subprocess.CompletedProcess(
                list(command), -1, _as_text(expired.stdout), _as_text(expired.stderr)
            )
        seconds = time.monotonic() - started
        self.runs.append(
            RunRecord(
                command=list(command),
                returncode=proc.returncode,
                seconds=seconds,
                timed_out=timed_out,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
            )
        )
        _reap_scratch(cache_root)
        return proc

    # -- reporting -------------------------------------------------------

    def report(self, *, expect_rc: int | None = None, require_stdout: str | None = None) -> str:
        """The text report: the hash trail, the hazards, and every run."""
        lines = [f"mutation probe: {len(self.records)} mutation(s), {len(self.runs)} run(s)"]
        for record in self.records:
            restored = record.restored_sha256 or "NOT RESTORED"
            state = "byte-identical" if record.byte_identical else "NOT byte-identical"
            lines.append(
                f"  {record.path}: sha256 {record.original_sha256[:12]} -> "
                f"mutated {record.mutated_sha256[:12]} -> restored {restored[:12]} ({state})"
            )
            for hazard in record.hazards:
                lines.append(f"    cached bytecode: {hazard}")
            if record.conflict:
                lines.append(f"    CONFLICT: {record.conflict}")
        for index, run in enumerate(self.runs, start=1):
            note = " TIMED OUT" if run.timed_out else ""
            lines.append(
                f"  run {index}: rc={run.returncode}{note} in {run.seconds:.1f}s — "
                f"{shlex.join(run.command)}"
            )
        expected = "any nonzero" if expect_rc is None else str(expect_rc)
        lines.append(f"  expected child status: {expected}")
        if require_stdout is not None:
            lines.append(f"  expected child output: {require_stdout!r}")
        return "\n".join(lines)

    def as_json(self, *, expect_rc: int | None = None, require_stdout: str | None = None) -> dict:
        """The same facts, for a script or a CI annotation."""
        return {
            "mutations": [
                {
                    "path": str(record.path),
                    "original_sha256": record.original_sha256,
                    "mutated_sha256": record.mutated_sha256,
                    "restored_sha256": record.restored_sha256,
                    "byte_identical": record.byte_identical,
                    "hazards": record.hazards,
                    "conflict": record.conflict,
                }
                for record in self.records
            ],
            "runs": [
                {
                    "command": run.command,
                    "returncode": run.returncode,
                    "seconds": round(run.seconds, 3),
                    "timed_out": run.timed_out,
                }
                for run in self.runs
            ],
            "expect_rc": expect_rc,
            "require_stdout": require_stdout,
        }


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def load_spec(source: str) -> list[Mutation]:
    """Read the JSON spec from *source* (a path, or ``-`` for stdin)."""
    if source == "-":
        payload = json.load(sys.stdin)
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise GuardRefused("the spec must be a JSON list of mutation objects")
    return [Mutation.from_payload(item, index=index) for index, item in enumerate(payload)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mutation_probe.py",
        description=(
            "Apply a source mutation, run a check under it, and revert it — hash-guarded, "
            "with bytecode isolated so a fast restore cannot be read as clean."
        ),
    )
    parser.add_argument(
        "--spec",
        metavar="FILE",
        help='JSON list of {"path","old","new"[,"count"[,"expect_sha256"]]}; "-" reads stdin',
    )
    parser.add_argument(
        "--expect-rc",
        type=int,
        default=None,
        metavar="STATUS",
        help=(
            "the child status that counts as CAUGHT (default: any nonzero; a different "
            "nonzero status means nothing is concluded, exit 2)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help="a check that does not finish by then is inconclusive, not a verdict (exit 2)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print one JSON document instead of the text report"
    )
    parser.add_argument(
        "--preset",
        metavar="NAME",
        help="a mutation this repo has already recorded; it supplies its own proof "
        "command, expected status and expected output (--list-presets prints them)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="print the preset catalog and exit",
    )
    parser.add_argument(
        "--require-stdout",
        metavar="TEXT",
        default=None,
        help="CAUGHT only if the child's output (stdout then stderr) contains TEXT; "
        "a matching status without the text is inconclusive (exit 2)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="the check to run under the mutation, after --",
    )
    args = parser.parse_args(argv)

    if args.list_presets:
        for name in sorted(PRESETS):
            print(PRESETS[name].describe())
        return 0
    if args.spec is not None and args.preset is not None:
        print("mutation_probe: --spec and --preset are alternatives — pass one", file=sys.stderr)
        return 2
    if args.spec is None and args.preset is None:
        print(
            "mutation_probe: no mutation given — pass --spec FILE or --preset NAME "
            "(--list-presets prints the catalog)",
            file=sys.stderr,
        )
        return 2

    preset: Preset | None = None
    run_cwd: Path | None = None
    if args.spec is not None:
        try:
            mutations = load_spec(args.spec)
        except (OSError, ValueError, GuardRefused) as exc:
            print(f"mutation_probe: unusable spec: {exc}", file=sys.stderr)
            return 2
        expect_rc = args.expect_rc
        require_stdout = args.require_stdout
    else:
        assert args.preset is not None  # the two refusal branches above
        preset = PRESETS.get(args.preset)
        if preset is None:
            print(
                f"mutation_probe: no preset named {args.preset!r} — known: "
                + ", ".join(sorted(PRESETS)),
                file=sys.stderr,
            )
            return 2
        mutations = preset.resolve()
        # A preset's proof command is written the way the documentation writes
        # it (repo-relative), so it means the same thing from any directory.
        run_cwd = REPO_ROOT
        expect_rc = args.expect_rc if args.expect_rc is not None else preset.expect_rc
        require_stdout = (
            args.require_stdout if args.require_stdout is not None else preset.require_stdout
        )

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        if preset is None:
            print("mutation_probe: no check given — pass it after --", file=sys.stderr)
            return 2
        command = list(preset.command)

    probe = Probe(mutations, timeout=args.timeout)
    child: subprocess.CompletedProcess | None = None
    refused: str | None = None
    try:
        probe.apply()
    except GuardRefused as exc:
        refused = str(exc)
    if refused is None:
        try:
            child = probe.run(command, cwd=run_cwd)
        except Exception as exc:  # the probe must never die holding mutated bytes
            refused = f"the check could not be run: {exc}"
        try:
            probe.revert()
        except GuardRefused as exc:
            refused = str(exc) if refused is None else f"{refused}; {exc}"
    if refused is None:
        # Exit 0 with an un-restored file is the verdict that must not exist:
        # revert() raises on the failure modes it can see, so this catches the
        # ones it cannot — a record with no verified restore still stands
        # between the run and a CAUGHT.
        unrestored = [record for record in probe.records if not record.byte_identical]
        if unrestored:
            refused = "; ".join(
                f"{record.path}: restore never verified (sha256 "
                f"{record.restored_sha256}, expected {record.original_sha256})"
                for record in unrestored
            )
    probe.close()

    if refused is not None:
        token, code = "invalid", 2
        verdict = f"INVALID — {refused}"
    elif any(run.timed_out for run in probe.runs):
        token, code = "invalid", 2
        verdict = (
            f"INVALID — the check did not finish within {args.timeout:g}s, so nothing is "
            "concluded about the mutation"
        )
    else:
        assert child is not None
        status = child.returncode
        if status == 0:
            token, code = "survived", 1
            verdict = (
                "SURVIVED — the check exited 0 under the mutation, so it does not cover this "
                "line"
            )
        elif expect_rc is not None and status != expect_rc:
            # A nonzero status that is not the one asked for is not a caught
            # mutation: pytest exits 4 on a collection error and 1 on a failing
            # test, and only the second one means the check saw the change.
            token, code = "invalid", 2
            verdict = (
                f"INVALID — the check exited {status}, not the expected {expect_rc}: it "
                "did not report the failure this probe asked for, so nothing is concluded "
                f"(pass --expect-rc {status} if that is the signal you want)"
            )
        elif require_stdout is not None and require_stdout not in (
            _as_text(child.stdout) + _as_text(child.stderr)
        ):
            # The status matched but the evidence never printed: the check broke
            # some OTHER way (a bad flag, a missing dependency, a theme failing
            # the table audit before any window is built). That is not a caught
            # mutation, and calling it one is the same class of lie as calling a
            # collection error one.
            token, code = "invalid", 2
            verdict = (
                f"INVALID — the check exited {status} but never printed "
                f"{require_stdout!r}, so nothing was concluded about the mutation"
            )
        else:
            token, code = "caught", 0
            expected = "any nonzero" if expect_rc is None else str(expect_rc)
            evidence = f" and printed {require_stdout!r}" if require_stdout else ""
            verdict = (
                f"CAUGHT — the check exited {status} under the mutation ({expected} was "
                f"asked for){evidence}, which is what it has to do"
            )

    if args.json:
        document = probe.as_json(expect_rc=expect_rc, require_stdout=require_stdout)
        document.update({"verdict": token, "exit_code": code})
        print(json.dumps(document, indent=2))
    else:
        print(probe.report(expect_rc=expect_rc, require_stdout=require_stdout))
        if child is not None and child.stdout.strip():
            print("--- child stdout ---")
            print(child.stdout.rstrip())
        if child is not None and child.stderr.strip():
            print("--- child stderr ---")
            print(child.stderr.rstrip())
        print(f"verdict: {verdict}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
