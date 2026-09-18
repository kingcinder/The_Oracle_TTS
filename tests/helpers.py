from __future__ import annotations

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
