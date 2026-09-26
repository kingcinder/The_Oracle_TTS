"""Token persistence — one JSON file, repo-local, atomic write.

No XDG user-data dir exists in this suite (app_paths is repo-local by design),
so the license store is repo-local too, consistent with app_settings.json and
the design doc's correction (docs/LICENSING_DESIGN.md, store.py note). The
write is atomic (temp file + os.replace) so a crash mid-write can never leave
a half-written token that would masquerade as corruption.

Load failures are typed, never raised: corrupted_content (parseable JSON but
wrong shape), unreadable (permission/OS error). A missing file is simply
"no license" — the valid, unlicensed community state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

STORE_FILENAME = "oracle_license.json"


@dataclass(frozen=True)
class StoredToken:
    token: str


@dataclass(frozen=True)
class LoadResult:
    state: str  # "no_token" | "loaded" | "corrupted_content" | "unreadable"
    stored: StoredToken | None = None
    detail: str = ""


def store_path(repo_root: str | Path) -> Path:
    return Path(repo_root) / STORE_FILENAME


def save_token(repo_root: str | Path, token: str, *, verify_before_save) -> LoadResult:
    """Atomically persist ``token`` after ``verify_before_save(token)`` passes.

    The verify callback keeps this module honest: a token that does not verify
    is refused *before* any bytes are written, so the store can never hold a
    token the verifier would reject on next load (M4 pins this).
    """
    verdict = verify_before_save(token)
    if not verdict.ok:
        return LoadResult(state="refused", detail=verdict.detail or verdict.state)

    path = store_path(repo_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump({"token": token}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except OSError as error:
        return LoadResult(state="unreadable", detail=f"could not write {path}: {error}")
    return LoadResult(state="saved", stored=StoredToken(token=token))


def load_token(repo_root: str | Path) -> LoadResult:
    """Read the stored token. Missing file -> no_token (the normal case)."""
    path = store_path(repo_root)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return LoadResult(state="no_token")
    except OSError as error:
        return LoadResult(state="unreadable", detail=f"could not read {path}: {error}")

    try:
        data = json.loads(raw)
    except ValueError:
        return LoadResult(
            state="corrupted_content",
            detail=f"{path} is not valid JSON. Re-run `the-oracle activate` with your token to restore it.",
        )
    if not isinstance(data, dict) or not isinstance(data.get("token"), str) or not data["token"]:
        return LoadResult(
            state="corrupted_content",
            detail=f"{path} does not contain a license token. Re-run `the-oracle activate`.",
        )
    return LoadResult(state="loaded", stored=StoredToken(token=data["token"]))


def clear_token(repo_root: str | Path) -> bool:
    """Deactivation is local: remove the file, move the seat. True if removed."""
    path = store_path(repo_root)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
