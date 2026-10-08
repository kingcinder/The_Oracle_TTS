from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from dataclasses import is_dataclass, asdict
import importlib
from pathlib import Path
from typing import Any


def import_module(name: str):
    return importlib.import_module(name)


def resolve_callable(module: Any, *names: str):
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    raise AssertionError(f"None of {names!r} exists on module {module.__name__}")


def maybe_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump()
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    return value


def extract_text(value: Any) -> str:
    value = maybe_to_dict(value)

    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return value.read_text(encoding="utf-8")
    if isinstance(value, dict):
        for key in ("text", "content", "raw_text", "clean_text", "markdown_text", "source_text"):
            if key in value and isinstance(value[key], str):
                return value[key]
        if "utterances" in value:
            return "\n".join(extract_text(item) for item in value["utterances"])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(extract_text(item) for item in value)

    for attr in ("text", "content", "raw_text", "clean_text", "markdown_text", "source_text"):
        data = getattr(value, attr, None)
        if isinstance(data, str):
            return data
    utterances = getattr(value, "utterances", None)
    if utterances is not None:
        return "\n".join(extract_text(item) for item in utterances)

    raise AssertionError(f"Could not extract text from {type(value)!r}")


def extract_speakers(items: Any) -> list[str]:
    items = maybe_to_dict(items)
    if isinstance(items, dict) and "utterances" in items:
        items = items["utterances"]
    if not isinstance(items, Iterable) or isinstance(items, (str, bytes, bytearray)):
        raise AssertionError(f"Expected an iterable of utterances, got {type(items)!r}")

    speakers: list[str] = []
    for item in items:
        item = maybe_to_dict(item)
        if isinstance(item, dict):
            for key in ("speaker", "speaker_id", "assigned_speaker", "label"):
                value = item.get(key)
                if isinstance(value, str):
                    speakers.append(value)
                    break
            else:
                raise AssertionError(f"No speaker field found in {item!r}")
            continue

        for attr in ("speaker", "speaker_id", "assigned_speaker", "label"):
            value = getattr(item, attr, None)
            if isinstance(value, str):
                speakers.append(value)
                break
        else:
            raise AssertionError(f"No speaker attribute found in {item!r}")
    return speakers


def normalise_speaker_label(label: str) -> str:
    upper = label.strip().upper()
    if upper.endswith("A"):
        return "A"
    if upper.endswith("B"):
        return "B"
    return upper


# --- every-occurrence attribution idiom (the 71d8840 completeness rule) -----


def attribute_to_every_line(
    lines: Sequence[str],
    offenders: Iterable[str],
    *,
    is_offender: Any = None,
) -> list[str]:
    """One ``<line number>: <token>`` attribution per occurrence — the idiom
    every reporting net here must share: a first-match-only report names one
    line and hides the rest, and the thing being reported is broken on ALL
    the lines that carry it, not just the first.

    ``is_offender`` is the net's own brokenness test (the record-integrity
    net passes git resolution: a token is an offender when git cannot
    resolve it); with ``None``, every token is an offender (pure synthetic
    pinning). Output is token-major (sorted) then line order, so the exact
    list a synthetic pin asserts is stable.

    The idiom's second half is TEST-side, and this helper alone does not
    provide it: every net that uses this attribution MUST also pin the
    property synthetically — a multi-occurrence case with a non-adjacent
    repeat and an exact ordered assertion — plus a vacuity guard proving the
    scan reads real input. Exemplars to copy: the record net's two-line pin
    (tests/test_record_integrity.py,
    test_a_broken_citation_is_attributed_to_every_line_carrying_it), the
    journal checker's multi-line pin (tests/test_commit_slices.py), and the
    doctor's per-entry next-steps pin (tests/test_doctor_input_subtitles.py).
    """
    broken = is_offender if is_offender is not None else (lambda token: True)
    attributions: list[str] = []
    for token in sorted(offenders):
        if not broken(token):
            continue
        attributions.extend(
            f"{i + 1}: {token}" for i, row in enumerate(lines) if token in row
        )
    return attributions


def attribute_to_every_record_line(
    record_lines: dict[str, Sequence[str]],
    offenders: Iterable[str],
    *,
    is_offender: Any = None,
) -> list[str]:
    """The CROSS-FILE layer of the same idiom: one offender cited in several
    records is broken in all of them, so the report is one
    ``<record>:<line>: <token>`` entry per occurrence across ALL records —
    never per-record first-match. File-major order (dict insertion order),
    token-major then line order within each record, so a synthetic pin's
    exact list is stable.

    Same contract as :func:`attribute_to_every_line`, whose test-side rules
    (synthetic multi-occurrence pin + vacuity guard per consuming net) apply
    here too — the record-integrity net's cross-record pin is the exemplar.
    """
    entries: list[str] = []
    for name, lines in record_lines.items():
        entries.extend(
            f"{name}:{attribution}"
            for attribution in attribute_to_every_line(lines, offenders, is_offender=is_offender)
        )
    return entries


#: Directories that hold no repository content: VCS metadata, the virtualenv,
#: interpreter caches, and the agent harness's own state. Everything else in the
#: tree is included, git-ignored or not -- the checks that regressed wrote into
#: git-ignored space (``build/real_engine_smoke/inputs``), so honouring
#: ``.gitignore`` here would have hidden the very writes this looks for.
DEFAULT_SNAPSHOT_EXCLUDED_DIRS = frozenset(
    {".git", ".venv", ".freebuff", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}
)

#: Files larger than this are fingerprinted by size and mtime instead of by
#: content. The repository carries multi-gigabyte native build output
#: (``audio.cpp/``), and hashing it twice per run costs seconds for no extra
#: detection, because a rewrite changes size or mtime. Content is hashed
#: everywhere else -- which is where checks actually write: plans, JSON, text.
SNAPSHOT_HASH_LIMIT_BYTES = 1 << 20


def repo_tree_snapshot(root, *, excluded_dirs=DEFAULT_SNAPSHOT_EXCLUDED_DIRS) -> dict[str, str]:
    """Fingerprint every file under ``root``, for before/after comparison.

    Returns ``{relative path: signature}``. Two snapshots of an unchanged tree are
    equal, and any file created, removed, rewritten or merely touched between them
    shows up as a difference -- so a check that writes where it should only read
    is caught by comparing snapshots taken either side of it. Paths are relative
    to ``root`` with forward slashes, so a diff reads the same on every platform.
    """
    import hashlib
    import os
    from pathlib import Path

    root = Path(root)
    snapshot: dict[str, str] = {}
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [name for name in subdirectories if name not in excluded_dirs]
        for filename in filenames:
            path = Path(directory) / filename
            relative = path.relative_to(root).as_posix()
            try:
                stat = path.stat()
                size = stat.st_size
                if size <= SNAPSHOT_HASH_LIMIT_BYTES:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                else:
                    digest = f"not-hashed-above-{SNAPSHOT_HASH_LIMIT_BYTES}"
                snapshot[relative] = f"size={size} mtime={stat.st_mtime_ns} sha256={digest}"
            except OSError as exc:  # pragma: no cover - unreadable entries are reported, not fatal
                snapshot[relative] = f"unreadable: {type(exc).__name__}"
    return snapshot


def snapshot_differences(before: dict[str, str], after: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """``(created, changed, removed)`` paths between two snapshots, each sorted."""
    created = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    changed = sorted(path for path in before.keys() & after.keys() if before[path] != after[path])
    return created, changed, removed


# --- Shiboken import-hook invariant (U4.2) --------------------------------
#
# A module whose FIRST import happens inside a QThread races the GUI's main
# thread through shiboken6's signature import hook (inspect.getsource runs
# during import) and the pair segfaults natively — reproduced and
# faulthandler-verified for mutagen (see
# tests/test_gui_render_import_safety.py and
# .serpent-circle/04-debug/root-causes.md). The reusable guard below pins the
# invariant for every QThread subclass and every module a worker's call graph
# can reach one hop out.

_QTHREAD_BASE_NAMES = {"QThread"}


def function_level_imports(tree: ast.AST) -> set[str]:
    """Module names imported inside function or method bodies in *tree*.

    A lazy import is safe only when some earlier import in the same process
    has already loaded the module on the main thread; the sweep treats them
    as violations so that safety is structural, not incidental.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    found.update(alias.name for alias in sub.names)
                elif isinstance(sub, ast.ImportFrom) and sub.module:
                    found.add(sub.module)
    return found


def qthread_class_imports(source_root: Path) -> list[tuple[str, str, set[str]]]:
    """Lazy imports inside each QThread subclass's OWN methods.

    Only the worker's execution path matters: lazy imports elsewhere in the
    same file (e.g. MainWindow's main-thread startup methods) are a
    deliberate startup-latency pattern and are not part of the invariant.
    Returns (relative path, class name, lazy module names) triples.
    """
    results: list[tuple[str, str, set[str]]] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(getattr(base, "id", getattr(base, "attr", "")) in _QTHREAD_BASE_NAMES for base in node.bases):
                continue
            lazy: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Import):
                            lazy.update(alias.name for alias in sub.names)
                        elif isinstance(sub, ast.ImportFrom) and sub.module:
                            lazy.add(sub.module)
            results.append((path.relative_to(source_root).as_posix(), node.name, lazy))
    return results


def iter_qthread_classes(source_root: Path) -> list[tuple[str, str]]:
    """Every QThread subclass in *source_root*, as (relative path, class name)."""
    hits: list[tuple[str, str]] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if any(getattr(base, "id", getattr(base, "attr", "")) in _QTHREAD_BASE_NAMES for base in node.bases):
                hits.append((path.relative_to(source_root).as_posix(), node.name))
    return hits


def worker_module_import_violations(
    source_root: Path,
    *,
    one_hop_targets: tuple[str, ...] = (),
) -> list[str]:
    """The shiboken invariant, as a list of violation strings (empty = clean).

    Two rules:

    1. every QThread subclass's OWN methods must contain no function-level
       imports — the worker's execution path (run() and every helper it
       calls in-class) must be fully module-scoped so no first import can
       happen on the worker thread;
    2. every module in *one_hop_targets* — the modules a worker's call graph
       reaches one hop out (export helpers, engine frontends) — must be fully
       module-scoped for the same reason.

    Static by design: cheap enough for the fast suite, and it fails on the
    edit that introduces the hazard rather than on a heisen-segfault later.
    """
    violations: list[str] = []
    checked: set[Path] = set()

    for rel, cls, lazy in qthread_class_imports(source_root):
        if lazy:
            violations.append(f"{rel}::{cls}: QThread methods carry function-level imports ({sorted(lazy)})")

    for target in one_hop_targets:
        path = source_root / target
        if path in checked:
            continue
        checked.add(path)
        if not path.exists():
            violations.append(f"{target}: one-hop target missing")
            continue
        lazy = function_level_imports(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if lazy:
            violations.append(f"{target}: worker-reachable module carries function-level imports ({sorted(lazy)})")

    return violations


def isolate_user_config(monkeypatch, config_dir) -> None:
    """Point the platform config root at ``config_dir`` for one test.

    ``XDG_CONFIG_HOME`` is only the POSIX config root: on Windows
    :func:`the_oracle.platform_support.user_config_root` reads ``%APPDATA%``
    instead. Patching just ``XDG_CONFIG_HOME`` therefore still lets Windows
    tests read and *write* the developer's real ``%APPDATA%\\the_oracle``
    settings — and leak state between test files (e.g. a remembered backend
    flag written by one file breaking another file's defaults test). Patch
    both roots so the suite is hermetic on every platform.
    """
    import os
    import sys

    config_dir = os.fspath(config_dir)
    monkeypatch.setenv("XDG_CONFIG_HOME", config_dir)
    if sys.platform == "win32":
        monkeypatch.setenv("APPDATA", config_dir)
