#!/usr/bin/env python3
"""Exercise the launch path from a clean checkout and report what differed.

``git archive HEAD`` produces exactly what a fresh clone starts from: the
tracked files, no ``.venv``, and none of the git-ignored build output a working
tree accumulates. Running the acceptance path there is the only way to see which
of the doctor's PASSes are capability and which are leftovers, because a check
that grades an artifact an earlier run left behind passes on a machine that has
already run it and fails on one that never has.

That comparison was done by hand twice while this repo was being made
launch-ready, so the checks and the comparison live here instead. Both legs run
the same three checks -- the full suite, the doctor gate, and the wrapper
dispatch matrix -- and every field that differs is reported with a category:

* a value naming the tree under test is structural, so it is substituted rather
  than registered: rewriting the fresh tree's root to the working tree's must
  reproduce the other value;
* a field in ``MEASURED_FIELDS`` is a duration of the machine, not a statement
  about either tree;
* anything else must be registered in ``EXPECTED_DELTAS`` with a reason. An
  unregistered difference fails the run, because it means the two trees behave
  differently for a reason nobody has written down.

Registration is per field and never per container, for the same reason the
doctor idempotence check does it that way: registering ``vulkan_backend`` would
excuse every field beneath it, including ones that have nothing to do with the
native build.

What this does not prove, stated plainly: a clean clone has no ``.venv``, so
both legs run under one interpreter (this repo's ``.venv`` by default, or
``--interpreter``). The checks therefore prove the *tree* is sound -- that its
code passes, its gate runs, and its entry points dispatch -- not that this
machine has already installed it. The suite also has to import the tree under
test and not this tree's editable install, so that is asserted before anything
else runs rather than assumed; a run where the wrong source loaded would
otherwise report a green tree that was never exercised.

    python scripts/fresh_clone_acceptance.py
    python scripts/fresh_clone_acceptance.py --only doctor,wrappers
    python scripts/fresh_clone_acceptance.py --working-tree --keep

Exit 0 means every check passed in both trees and every delta was registered;
exit 1 means a check failed or a delta was unexpected; exit 2 means a check
could not run at all (no interpreter, no ``git``, unreadable report).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Report fields that measure the machine rather than describing either tree.
#: Deliberately the same set the doctor idempotence check allows between two
#: runs, and pinned equal to it by a test: both answer "is this a duration or a
#: fact", and one answer drifting from the other would be a defect in itself.
MEASURED_FIELDS = (
    "chatterbox_init.seconds",
    "deterministic_smoke.runtime_seconds",
)

#: Deltas that are expected between a clean checkout and this working tree, each
#: with the reason it is not a defect. Everything here is a local capability or a
#: local artifact: the working tree has a venv, a native build, and reference
#: clips that a clean clone does not. Anything not listed fails the run.
EXPECTED_DELTAS: dict[str, str] = {
    # A clean checkout has no venv until bootstrap creates one.
    "doctor.entrypoint.venv_entrypoint_exists":
        "a clean checkout has no .venv; `manage_install.py bootstrap` creates it",
    # The doctor prefers the venv entrypoint when it exists, so the working tree
    # resolves its own launcher while a clean clone falls back to whatever is on
    # PATH -- on this machine a stale out-of-repo launcher from an older checkout.
    "doctor.entrypoint.help_ok":
        "the working tree's own venv entrypoint answers; a clean clone has none",
    "doctor.entrypoint.help_error":
        "a clean clone has no venv entrypoint, so the doctor reports whatever the "
        "PATH resolves instead of the tree's own launcher",
    # The real-engine smoke's inputs are the artifacts an earlier (pre-fix) run
    # wrote. ensure_real_engine_inputs() no longer writes them, so a clean clone
    # correctly reports them absent.
    "doctor.real_engine_smoke.dialogue_exists":
        "leftovers from earlier runs; the check no longer creates them",
    "doctor.real_engine_smoke.output_exists":
        "leftovers from earlier runs; the check no longer creates them",
    "doctor.real_engine_smoke.speaker_a_reference_exists":
        "leftovers from earlier runs; the check no longer creates them",
    "doctor.real_engine_smoke.speaker_b_reference_exists":
        "leftovers from earlier runs; the check no longer creates them",
    # Reference clips: the working tree carries an untracked Seashell_No_1.wav
    # plus fallback clips under build/, which a clean clone does not.
    "doctor.voice_sources.seashell_clip_count":
        "untracked reference clips placed in Seashells/ by hand",
    "doctor.voice_sources.fallback_clip_count":
        "fallback clips left in build/ by earlier runs",
    # The Vulkan backend needs the git-ignored native build and its GGUF model;
    # neither is in the tracked tree, so a clean clone has neither.
    "doctor.vulkan_backend.ok":
        "the native audio.cpp build and its GGUF model are git-ignored",
    "doctor.vulkan_backend.binary":
        "the native audio.cpp build and its GGUF model are git-ignored",
    "doctor.vulkan_backend.binary_built":
        "the native audio.cpp build and its GGUF model are git-ignored",
    "doctor.vulkan_backend.binary_runs":
        "there is no binary to execute without the git-ignored native build",
    "doctor.vulkan_backend.model_auto_found":
        "the native audio.cpp build and its GGUF model are git-ignored",
    "doctor.vulkan_backend.model_file_exists":
        "the native audio.cpp build and its GGUF model are git-ignored",
    "doctor.vulkan_backend.model_path":
        "the native audio.cpp build and its GGUF model are git-ignored",
    "doctor.vulkan_backend.audio_cpp_devices":
        "device enumeration is gated on the git-ignored native build",
    "doctor.vulkan_backend.vendored_patch_applied":
        "the patch is applied to the git-ignored audio.cpp clone",
    # The suite's one skip is the Vulkan smoke's own guard, which is exactly the
    # difference the native build makes -- so the pass count moves with it.
    "suite.skipped":
        "the Vulkan smoke skips without the git-ignored native build",
    "suite.passed":
        "the Vulkan smoke is one more passing test where the native build exists",
}

#: Every entry point the documentation tells a user to run directly, and the
#: ``manage_install.py`` subcommand each must reach. ``oracle start`` is here
#: because it is an alias the manager translates to ``run`` -- a row that only
#: proves an identity mapping would not catch the translation breaking.
WRAPPER_MATRIX: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("./oracle", ("doctor", "--help"), "doctor"),
    ("./oracle", ("start", "--help"), "run"),
    ("./oracle", ("install", "--help"), "install"),
    ("./oracle", ("uninstall", "--help"), "uninstall"),
    ("./bootstrap_oracle_tts.sh", ("--help",), "bootstrap"),
    ("./install_oracle_tts.sh", ("--help",), "install"),
    ("./doctor_oracle_tts.sh", ("--help",), "doctor"),
    ("./run_oracle_tts.sh", ("--help",), "run"),
    ("./uninstall_oracle_tts.sh", ("--help",), "uninstall"),
)

#: ``manage_install.py doctor`` flags the gate is exercised with in both trees.
#: ``--ci`` is what the CI matrix passes, and it is the mode that ignores
#: environment-only checks such as ffmpeg and the wrapper PATH, so a difference
#: here is about the tree rather than about whose laptop it is.
DOCTOR_FLAGS = ("--json", "--skip-model-init", "--ci")

_SUMMARY_COUNTS = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "errors": re.compile(r"(\d+) error"),
    "skipped": re.compile(r"(\d+) skipped"),
}

_USAGE_SUBCOMMAND = re.compile(r"^usage:\s+manage_install\.py\s+(\S+)", re.MULTILINE)


class AcceptanceError(RuntimeError):
    """A check could not be run, as opposed to having failed."""


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome in one tree.

    ``facts`` is what the two legs are compared on: counts, statuses and the
    doctor's report, and nothing else. ``detail`` is for the reader and is never
    compared, which is what keeps the comparison from being a diff of two logs.
    """

    name: str
    ok: bool
    facts: dict[str, Any] = field(default_factory=dict)
    detail: str = ""


@dataclass(frozen=True)
class Delta:
    """One field that differs between the two legs, and its verdict."""

    path: str
    fresh: Any
    current: Any
    reason: str | None

    @property
    def registered(self) -> bool:
        return self.reason is not None


# --------------------------------------------------------------------------- #
# Acquiring the clean tree
# --------------------------------------------------------------------------- #


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def export_fresh_tree(repo_root: Path, destination: Path, *, working_tree: bool = False) -> Path:
    """Write the tracked tree into ``destination`` and return it.

    ``git archive HEAD`` is used rather than ``git worktree`` so the result has
    no ``.git`` and cannot be committed into by accident, which also means a
    check that misbehaves cannot touch the real repository.

    With ``working_tree``, uncommitted modifications and untracked files are
    copied over the archive afterwards. That is for development runs -- a run
    without it is the one that answers "what does a fresh clone do".
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / f"{destination.name}.tar"
    try:
        _git(repo_root, "archive", "HEAD", "-o", str(archive))
        with tarfile.open(archive) as handle:
            try:
                # The archive is this repository's own tracked tree, produced by
                # `git archive` a moment ago -- there is no untrusted input here,
                # so the trustworthy filter is the honest one. It also preserves
                # the executable bits, which matter: a wrapper's mode is part of
                # what this script checks.
                handle.extractall(destination, filter="fully_trusted")
            except TypeError:  # pragma: no cover - Python without the filter argument
                handle.extractall(destination)
    finally:
        archive.unlink(missing_ok=True)

    if working_tree:
        names: list[str] = []
        names += _git(repo_root, "diff", "HEAD", "--name-only").stdout.split("\n")
        names += _git(repo_root, "ls-files", "--others", "--exclude-standard").stdout.split("\n")
        for name in sorted({line for line in names if line.strip()}):
            source = repo_root / name
            if not source.is_file():
                continue
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return destination


def _resolve_interpreter(repo_root: Path, explicit: str | None) -> Path:
    """The interpreter both legs run under.

    A clean clone has no venv, so the interpreter is necessarily shared. It is
    named in the report rather than assumed, because every capability check below
    is really "can *this* interpreter, running that tree's code, do this".
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise AcceptanceError(f"no such interpreter: {path}")
        return path
    venv = repo_root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv.exists():
        return venv
    return Path(sys.executable)


def assert_imports_the_tree_under_test(tree: Path, interpreter: Path) -> str:
    """Fail unless code imported from ``tree`` resolves inside ``tree``.

    The whole comparison rests on this. This repo's interpreter has the package
    installed in editable mode pointing at the *working tree*, so a fresh-tree
    run that silently loaded that copy would report a green result for code it
    never executed -- a PASS graded from something it did not verify, which is
    the exact failure mode the launch validation exists to catch.
    """
    # ``src`` is prepended the way pytest's ``pythonpath`` setting and
    # ``doctor.py``'s own ``_prepend_repo_src`` do it, so this asks the same
    # question the checks will: which source does the tree under test load?
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
        raise AcceptanceError(
            f"could not import the_oracle from {tree}: {completed.stderr.strip()[:400]}"
        )
    imported = Path(resolved[-1])
    if not imported.is_relative_to(tree.resolve()):
        raise AcceptanceError(
            "the tree under test did not load its own code: "
            f"imported {imported} while testing {tree}. "
            "Refusing to report on a tree that was never exercised."
        )
    return str(imported)


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #


def parse_pytest_summary(output: str) -> dict[str, int]:
    """Counts out of pytest's final summary line.

    Only the last matching line is read, because the skip audit prints its own
    line that also contains the word "skipped" -- counting every match would
    double-count it.
    """
    counts = dict.fromkeys(_SUMMARY_COUNTS, 0)
    for key, pattern in _SUMMARY_COUNTS.items():
        matches = pattern.findall(output)
        if matches:
            counts[key] = int(matches[-1])
    return counts


def run_suite(tree: Path, interpreter: Path, extra_args: tuple[str, ...] = ()) -> CheckResult:
    """Run the full suite in ``tree`` and compare its counts."""
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
    counts = parse_pytest_summary(output)
    if not any(counts.values()):
        raise AcceptanceError(f"could not read pytest's summary from:\n{_tail(output)}")
    facts: dict[str, Any] = {"exit_code": completed.returncode, **counts}
    detail = " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    return CheckResult("suite", completed.returncode == 0, facts, detail)


def run_doctor(tree: Path, interpreter: Path, extra_args: tuple[str, ...] = ()) -> CheckResult:
    """Run the doctor gate in ``tree`` and compare its report field by field."""
    argv = [
        str(interpreter),
        str(tree / "scripts" / "doctor.py"),
        "--repo-root",
        str(tree),
        *DOCTOR_FLAGS,
    ]
    completed = subprocess.run(
        argv,
        cwd=str(tree),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        report = json.loads(completed.stdout)
    except ValueError as error:
        raise AcceptanceError(
            f"the doctor produced no report (exit {completed.returncode}): {error}\n"
            f"{_tail(completed.stderr)}"
        ) from error
    if not isinstance(report, dict):
        raise AcceptanceError(f"the doctor report was {type(report).__name__}, not an object")
    ready = report.get("overall_ready")
    return CheckResult(
        "doctor",
        completed.returncode == 0,
        {"exit_code": completed.returncode, **report},
        f"overall_ready={ready} in {' '.join(DOCTOR_FLAGS)} mode",
    )


def parse_dispatched_subcommand(output: str) -> str | None:
    """The subcommand a wrapper reached, read off ``manage_install``'s usage line.

    Read structurally rather than by searching the text: a wrapper that failed
    before dispatch can still print the word "bootstrap" in an echoed command
    line, and a substring search would call that a success.
    """
    match = _USAGE_SUBCOMMAND.search(output)
    return match.group(1) if match else None


def run_wrappers(tree: Path, _interpreter: Path | None = None, _extra_args: tuple[str, ...] = ()) -> CheckResult:
    """Run the documented entry points and check where each one dispatches."""
    facts: dict[str, Any] = {}
    failures: list[str] = []
    for index, (script, args, expected) in enumerate(WRAPPER_MATRIX):
        row = f"{script} {' '.join(args)}"
        path = tree / script.lstrip("./")
        if not path.exists():
            facts[row] = "missing"
            failures.append(f"{row}: {script} does not exist")
            continue
        if not os.access(path, os.X_OK):
            facts[row] = "not executable"
            failures.append(
                f"{row}: {script} is not executable, though the documentation "
                "invokes it directly (git mode is not 100755)"
            )
            continue
        completed = subprocess.run(
            [str(path), *args],
            cwd=str(tree),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        observed = parse_dispatched_subcommand(completed.stdout + completed.stderr)
        facts[row] = observed or f"no dispatch (exit {completed.returncode})"
        if observed != expected:
            failures.append(f"{row}: dispatched to {observed or 'nothing'}, expected {expected}")
    detail = f"{len(facts) - len(failures)}/{len(facts)} rows dispatched correctly"
    return CheckResult("wrappers", not failures, facts, detail + ("; " + "; ".join(failures) if failures else ""))


#: The checks, by name, each ``(tree, interpreter, extra_args) -> CheckResult``.
#: Adding one here puts it in both legs, in the report, and in ``--only``.
CHECKS: dict[str, Callable[[Path, Path, tuple[str, ...]], CheckResult]] = {
    "suite": run_suite,
    "doctor": run_doctor,
    "wrappers": run_wrappers,
}


# --------------------------------------------------------------------------- #
# Comparing the two legs
# --------------------------------------------------------------------------- #


def substitute_root(value: Any, fresh_root: Path, current_root: Path) -> Any:
    """Rewrite the fresh tree's path into the working tree's, throughout.

    This is what makes a path a *structural* difference rather than one needing
    a registry entry: a value that is the same once the tree root is substituted
    is naming the tree under test, which is expected, and a value that is not
    says something about the tree itself.
    """
    if isinstance(value, dict):
        return {key: substitute_root(item, fresh_root, current_root) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute_root(item, fresh_root, current_root) for item in value]
    if isinstance(value, str) and str(fresh_root) in value:
        return value.replace(str(fresh_root), str(current_root))
    return value


def _is_measurement(value: Any) -> bool:
    # bool is an int subclass: a measurement is a number, a flag is not.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _walk(left: Any, right: Any, path: str, found: list[tuple[str, Any, Any]]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            _walk(left.get(key), right.get(key), child, found)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            found.append((path, len(left), len(right)))
            return
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            _walk(item_left, item_right, f"{path}[{index}]", found)
        return
    if left != right:
        found.append((path, left, right))


def _is_measured(path: str, measured: tuple[str, ...]) -> bool:
    """Whether ``path`` is a registered measurement.

    Payloads are nested under their check name, so ``doctor.entrypoint``'s fields
    arrive as ``doctor.some.field`` while the registry -- which is kept in the
    doctor's own field names so it can be asserted equal to the doctor
    idempotence check's copy -- holds ``some.field``. Both spellings match.
    """
    if path in measured:
        return True
    return "." in path and path.split(".", 1)[1] in measured


def differences(
    fresh: dict[str, Any],
    current: dict[str, Any],
    *,
    fresh_root: Path,
    current_root: Path,
    measured: tuple[str, ...] = MEASURED_FIELDS,
) -> list[Delta]:
    """Every field that differs, each marked registered or unexpected."""
    normalized = substitute_root(fresh, fresh_root, current_root)
    raw: list[tuple[str, Any, Any]] = []
    _walk(normalized, current, "", raw)

    deltas: list[Delta] = []
    for path, fresh_value, current_value in raw:
        if _is_measured(path, measured) and _is_measurement(fresh_value) and _is_measurement(current_value):
            continue
        # Report the value the fresh tree actually produced, not the rewritten one.
        original = _lookup(fresh, path)
        deltas.append(Delta(path, original, current_value, EXPECTED_DELTAS.get(path)))
    return deltas


def _lookup(payload: Any, dotted: str) -> Any:
    node = payload
    for part in dotted.split("."):
        index = None
        if part.endswith("]") and "[" in part:
            part, _, suffix = part.partition("[")
            index = int(suffix[:-1])
        if not isinstance(node, dict) or part not in node:
            return "<missing>"
        node = node[part]
        if index is not None:
            if not isinstance(node, list) or index >= len(node):
                return "<missing>"
            node = node[index]
    return node


def _render(value: Any) -> str:
    text = json.dumps(value) if not isinstance(value, str) else value
    return text if len(text) <= 120 else text[:117] + "..."


def _tail(text: str, lines: int = 12) -> str:
    stripped = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(stripped[-lines:])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _print_report(
    fresh_root: Path,
    current_root: Path,
    interpreter: Path,
    imported: str,
    results: dict[str, tuple[CheckResult, CheckResult]],
    deltas: dict[str, list[Delta]],
    *,
    working_tree: bool,
) -> None:
    print("Fresh-clone acceptance")
    print(f"  clean tree : {fresh_root}{' (with your uncommitted work overlaid)' if working_tree else ''}")
    print(f"  working tree: {current_root}")
    print(f"  interpreter : {interpreter}  (shared: a clean clone has no venv)")
    print(f"  imported    : {imported}  (asserted to be inside the clean tree)")
    print()

    for name, (fresh, current) in results.items():
        mark = "PASS" if fresh.ok and current.ok else "FAIL"
        print(f"{mark} {name}: clean tree {fresh.detail} | working tree {current.detail}")
        for delta in deltas.get(name, []):
            verdict = "expected" if delta.registered else "UNEXPECTED"
            print(f"    delta [{verdict}] {delta.path}")
            print(f"        clean tree : {_render(delta.fresh)}")
            print(f"        working    : {_render(delta.current)}")
            if delta.reason:
                print(f"        because    : {delta.reason}")
    print()

    unexpected = [delta for group in deltas.values() for delta in group if not delta.registered]
    if unexpected:
        print(f"FAIL: {len(unexpected)} unexpected delta(s). Each means the clean tree and the")
        print("working tree behave differently for a reason nobody has written down. Either")
        print("fix the difference, or register the field in EXPECTED_DELTAS in this script")
        print("with the reason it is not a defect.")
    else:
        registered = sum(1 for group in deltas.values() for delta in group if delta.registered)
        print(f"PASS: every delta registered ({registered} field(s) explained, measurements excluded).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--interpreter", default=None, help="Python to run both legs with.")
    parser.add_argument(
        "--only",
        default=",".join(CHECKS),
        help=f"Comma-separated subset of: {', '.join(CHECKS)}.",
    )
    parser.add_argument("--working-tree", action="store_true", help="Overlay uncommitted work onto the clean tree.")
    parser.add_argument("--keep", action="store_true", help="Keep the clean tree instead of deleting it.")
    parser.add_argument("--scratch", type=Path, default=None, help="Where to build the clean tree.")
    parser.add_argument("--pytest-args", default="", help="Extra arguments for the suite, space-separated.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.expanduser().resolve()
    selected = [name.strip() for name in args.only.split(",") if name.strip()]
    unknown = [name for name in selected if name not in CHECKS]
    if unknown:
        parser.error(f"unknown check(s): {', '.join(unknown)}")

    try:
        interpreter = _resolve_interpreter(repo_root, args.interpreter)
    except AcceptanceError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    scratch_parent = args.scratch.expanduser().resolve() if args.scratch else None
    fresh_root = Path(
        tempfile.mkdtemp(prefix="oracle-fresh-acceptance-", dir=str(scratch_parent) if scratch_parent else None)
    ) / "tree"
    extra_args = tuple(args.pytest_args.split())
    try:
        export_fresh_tree(repo_root, fresh_root, working_tree=args.working_tree)
        imported = assert_imports_the_tree_under_test(fresh_root, interpreter)

        results: dict[str, tuple[CheckResult, CheckResult]] = {}
        deltas: dict[str, list[Delta]] = {}
        could_not_run: list[str] = []
        for name in selected:
            check = CHECKS[name]
            try:
                fresh = check(fresh_root, interpreter, extra_args)
                current = check(repo_root, interpreter, extra_args)
            except AcceptanceError as error:
                could_not_run.append(f"{name}: {error}")
                continue
            results[name] = (fresh, current)
            # Nested under the check name, so a delta path reads as
            # ``doctor.vulkan_backend.ok`` and matches its registry entry -- and
            # so two checks that both report a field called ``exit_code`` cannot
            # collide in the registry.
            deltas[name] = differences(
                {name: fresh.facts},
                {name: current.facts},
                fresh_root=fresh_root,
                current_root=repo_root,
            )

        _print_report(
            fresh_root, repo_root, interpreter, imported, results, deltas,
            working_tree=args.working_tree,
        )

        if could_not_run:
            for line in could_not_run:
                print(f"FAIL: check could not run -- {line}", file=sys.stderr)
            return 2
        failed = [name for name, (fresh, current) in results.items() if not (fresh.ok and current.ok)]
        unexpected = [delta for group in deltas.values() for delta in group if not delta.registered]
        if failed or unexpected:
            for name in failed:
                print(f"FAIL: {name} did not pass in both trees.", file=sys.stderr)
            return 1

        print(f"\nPASS: {', '.join(results)} exercised from a clean checkout, all deltas registered.")
        return 0
    finally:
        if args.keep:
            print(f"\nClean tree kept at {fresh_root}", file=sys.stderr)
        else:
            shutil.rmtree(fresh_root.parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
