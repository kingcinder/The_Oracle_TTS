#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT_DEFAULT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from the_oracle.platform_support import (
    is_linux,
    is_windows,
    managed_launcher_dir,
    managed_launcher_path,
    path_entries,
    repo_bootstrap_display,
    repo_python_display,
    repo_run_display,
    venv_entrypoint_path,
)


JSON_PREFIX = "__ORACLE_TTS_JSON__"
MANAGED_WRAPPER_MARKER = "ORACLE_TTS_WRAPPER"
SUPPORTED_PYTHON_MIN = (3, 11)
SUPPORTED_PYTHON_MAX = (3, 13)
LIBRARY_PACKAGE_CANDIDATES: dict[str, list[str]] = {
    "libasound.so.2": ["libasound2t64", "libasound2"],
    "libdbus-1.so.3": ["libdbus-1-3"],
    "libEGL.so.1": ["libegl1"],
    "libfontconfig.so.1": ["libfontconfig1"],
    "libglib-2.0.so.0": ["libglib2.0-0t64", "libglib2.0-0"],
    "libGL.so.1": ["libgl1"],
    "libgobject-2.0.so.0": ["libglib2.0-0t64", "libglib2.0-0"],
    "libgthread-2.0.so.0": ["libglib2.0-0t64", "libglib2.0-0"],
    "libnss3.so": ["libnss3"],
    "libOpenGL.so.0": ["libopengl0"],
    "libpulse.so.0": ["libpulse0"],
    "libxcb-cursor.so.0": ["libxcb-cursor0"],
    "libxcb-icccm.so.4": ["libxcb-icccm4"],
    "libxcb-image.so.0": ["libxcb-image0"],
    "libxcb-keysyms.so.1": ["libxcb-keysyms1"],
    "libxcb-randr.so.0": ["libxcb-randr0"],
    "libxcb-render-util.so.0": ["libxcb-render-util0"],
    "libxcb-shape.so.0": ["libxcb-shape0"],
    "libxcb-sync.so.1": ["libxcb-sync1"],
    "libxcb-xfixes.so.0": ["libxcb-xfixes0"],
    "libxcb-xinerama.so.0": ["libxcb-xinerama0"],
    "libxkbcommon-x11.so.0": ["libxkbcommon-x11-0"],
}


def _prepend_repo_src(repo_root: Path) -> None:
    src_path = repo_root / "src"
    if src_path.exists():
        src_text = str(src_path)
        if src_text not in sys.path:
            sys.path.insert(0, src_text)


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _tail(text: str, lines: int = 8) -> str:
    if not text:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


def _run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except OSError as exc:
        # FileNotFoundError and PermissionError are both OSError subclasses;
        # a missing *or non-executable* probe binary must never crash the
        # doctor -- it returns a structured failure like any other probe.
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "error": str(exc),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"Timed out after {timeout:.0f}s",
            "timed_out": True,
        }

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": False,
    }


def _probe_environment(repo_root: Path, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    if extra_env:
        env.update(extra_env)
    return env


def _run_python_probe(
    repo_root: Path,
    code: str,
    *,
    timeout: float,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = _run_command(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=_probe_environment(repo_root, extra_env),
        timeout=timeout,
    )
    if result["timed_out"]:
        return {"ok": False, "error": result["error"], "stdout_tail": _tail(result["stdout"]), "stderr_tail": _tail(result["stderr"])}

    payload = None
    for stream in (result["stdout"], result["stderr"]):
        for line in reversed(stream.splitlines()):
            if line.startswith(JSON_PREFIX):
                payload = json.loads(line[len(JSON_PREFIX) :])
                break
        if payload is not None:
            break
    if payload is None:
        payload = {
            "ok": result["ok"],
            "error": result.get("error") or f"Probe returned {result['returncode']}",
        }
    payload["returncode"] = result["returncode"]
    payload["stdout_tail"] = _tail(result["stdout"])
    payload["stderr_tail"] = _tail(result["stderr"])
    return payload


@lru_cache(maxsize=None)
def _package_installed(package_name: str) -> bool:
    if not is_linux():
        return False
    result = _run_command(["dpkg-query", "-W", "-f=${Status}", package_name], timeout=10)
    return result["ok"] and result["stdout"].strip().endswith("installed")


@lru_cache(maxsize=None)
def _package_available(package_name: str) -> bool:
    if not is_linux():
        return False
    result = _run_command(["apt-cache", "show", package_name], timeout=10)
    return result["ok"] and bool(result["stdout"].strip())


def _preferred_package(candidates: list[str]) -> str:
    for candidate in candidates:
        if _package_installed(candidate):
            return candidate
    for candidate in candidates:
        if _package_available(candidate):
            return candidate
    return candidates[0]


def _qt_package_suggestions(missing_libraries: list[str]) -> list[str]:
    suggestions: list[str] = []
    for library in missing_libraries:
        candidates = LIBRARY_PACKAGE_CANDIDATES.get(library)
        if not candidates:
            continue
        suggestions.append(_preferred_package(candidates))
    return sorted(set(suggestions))


def _dependency_pin_status(repo_root: Path) -> dict[str, Any]:
    """Verify the running venv matches the pyproject dependency pins.

    Full-suite green claims are only valid against the declared dependency
    set: an out-of-band `pip install` can silently replace a pinned package
    (this actually happened -- a stray upgrade of huggingface_hub/transformers
    broke offline-guarantee tests that were green the day before, see
    JUNO_FIXES.log 2026-09-20 docs entry). This check compares every pinned
    requirement in pyproject.toml against the installed distribution metadata
    via importlib.metadata, so drift is caught by the gate instead of by a
    mysterious test failure days later.

    Reports one entry per requirement plus the importability of tomllib and
    packaging (both stdlib/vendored-stdlib in practice; listed for honesty).
    """
    import tomllib

    import packaging.requirements

    pyproject = repo_root / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        return {
            "ok": False,
            "error": f"could not read {pyproject}: {error}",
            "mismatches": [],
            "missing": [],
            "checked_count": 0,
        }

    project = data.get("project", {})
    requirement_groups: list[tuple[str, list[str]]] = [("dependencies", project.get("dependencies") or [])]
    for group_name, group in (project.get("optional-dependencies") or {}).items():
        requirement_groups.append((f"optional:{group_name}", group or []))

    mismatches: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    checked = 0
    for group_name, requirement_strings in requirement_groups:
        for requirement_string in requirement_strings:
            try:
                requirement = packaging.requirements.Requirement(requirement_string)
            except packaging.requirements.InvalidRequirement as error:
                mismatches.append({"group": group_name, "requirement": requirement_string, "error": str(error)})
                continue
            checked += 1
            try:
                installed = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                missing.append({"group": group_name, "requirement": requirement_string})
                continue
            if not requirement.specifier.contains(installed, prereleases=True):
                mismatches.append(
                    {
                        "group": group_name,
                        "requirement": requirement_string,
                        "installed": installed,
                        "error": f"installed {requirement.name} {installed} does not satisfy {requirement_string}",
                    }
                )

    error = ""
    if missing:
        error = "; ".join(f"{entry['requirement']} not installed" for entry in missing)
    if mismatches:
        error = "; ".join(filter(None, [error, *(entry.get("error", "") for entry in mismatches)]))
    return {
        "ok": not mismatches and not missing,
        "error": error,
        "mismatches": mismatches,
        "missing": missing,
        "checked_count": checked,
    }


def _crash_reports_status(repo_root: Path) -> dict[str, Any]:
    """Report the local crash-capture state (docs/CRASH_TELEMETRY_DESIGN §7).

    Opted-out (the default) is a valid state: ``ok=true`` and informational —
    the doctor never treats a privacy preference as a problem. Deliberately
    NOT part of ``overall_ready``: crash capture is diagnostics, and its
    consent state is not a broken install (same reasoning as the licensing
    check's exclusion). All remedy strings are offline-safe.

    Writability is inferred with os.access rather than a write probe because
    this check must stay read-only (the doctor's read-only pin); a directory
    that does not exist yet is judged by its parent, so an opted-in install
    that never crashed is not a failure.

    This check is also where the GUI segfault watch item surfaces: a
    faulthandler native-crash dump is named in the human report and next
    steps, turning "blocked on repro" into "reproducible with data".
    """
    import json as _json
    import os as _os

    try:
        from the_oracle import crash
    except ImportError as error:
        return {
            "ok": False,
            "consent": None,
            "record_count": 0,
            "at_cap": False,
            "newest": None,
            "newest_exception": "",
            "native_dump_present": False,
            "total_bytes": 0,
            "detail": f"the_oracle.crash is not importable: {error}",
        }

    consent_on = crash.read_consent(repo_root)
    directory = crash.crash_dir(repo_root)
    records = crash.list_records(repo_root)

    native_path = directory / "native-crash.txt"
    try:
        native_present = native_path.exists() and native_path.stat().st_size > 0
    except OSError:
        native_present = False

    write_problem = ""
    if consent_on:
        probe_target = directory if directory.exists() else directory.parent
        if not _os.access(probe_target, _os.W_OK):
            write_problem = (
                f"crash report directory is not writable: {directory} — "
                "check the folder's permissions; capture will fail silently until fixed"
            )

    total_bytes = 0
    newest_exception = ""
    for record in records:
        try:
            total_bytes += record.stat().st_size
        except OSError:
            continue
    if records:
        try:
            data = _json.loads(records[0].read_text(encoding="utf-8"))
            newest_exception = str((data.get("exception") or {}).get("type") or "")
        except (OSError, ValueError):
            newest_exception = "unreadable"

    at_cap = len(records) >= crash.MAX_RECORDS
    if write_problem:
        detail = write_problem
    elif not consent_on:
        detail = "local crash reporting disabled — enable with `the-oracle privacy-opt-in` (reports stay local; nothing is ever uploaded)"
    elif at_cap:
        detail = f"capture enabled; {len(records)} report(s) on disk — at cap, the oldest will be dropped as new ones arrive"
    else:
        detail = f"capture enabled; {len(records)} report(s) on disk"

    return {
        "ok": not write_problem,
        "consent": consent_on,
        "record_count": len(records),
        "at_cap": at_cap,
        "newest": records[0].name if records else None,
        "newest_exception": newest_exception,
        "native_dump_present": native_present,
        "total_bytes": total_bytes,
        "detail": detail,
    }


def _licensing_status(repo_root: Path) -> dict[str, Any]:
    """Report the install's license state (docs/LICENSING_DESIGN.md §5).

    Unlicensed is a valid state: ``ok=true`` with state ``no_token`` — the
    doctor never fails an install for lacking a license and never suggests
    purchasing as a "problem". Verification is a pure function of the stored
    token bytes, the embedded public keys, and the wall clock, so this check
    is idempotent and history-independent like the rest; the clock is the
    only external input, exactly like any time-derived check.

    Deliberately NOT part of ``overall_ready``: an expired or mismatched
    license is a billing state, not a broken install — the suite still runs,
    degraded to community entitlements (licensing/policy.py). The check's own
    verdict and the human-readable line carry the signal instead.
    """
    try:
        from the_oracle.licensing import current_license
    except ImportError as error:
        return {
            "ok": False,
            "state": "package_missing",
            "edition": None,
            "licensee": None,
            "key_id": None,
            "exp": None,
            "machine_locked": False,
            "detail": f"the_oracle.licensing is not importable: {error}",
        }

    status = current_license(repo_root)
    return {
        "ok": status.ok,
        "state": status.state,
        "edition": status.edition,
        "licensee": status.licensee,
        "key_id": status.key_id,
        "exp": status.exp,
        "machine_locked": status.machine_locked,
        "detail": status.detail,
    }


#: Cue separator: the blank-line runs that end one SubRip/WebVTT cue and
#: start the next. Mirror of ``srt_ingest._CUE_SEPARATOR_RE`` — the doctor
#: never imports the_oracle, so the decode chain's shapes are mirrored here
#: and pinned against the real chain by tests/test_doctor_input_subtitles.py.
_CUE_SEPARATOR_RE = re.compile(rb"(\r?\n\r?\n+)")

#: Characters cp1252 can express beyond Latin-1, DERIVED from the codec rather
#: than hand-typed (mirror of ``srt_ingest._CP1252_EXTRA_CHARACTERS``).
_CP1252_EXTRA_CHARACTERS = frozenset(
    ch
    for ch in (
        bytes([value]).decode("cp1252", errors="replace")
        for value in range(0x80, 0x100)
    )
    if ch != "\ufffd"
)


def _utf8_guard(text: str) -> bool:
    """Mirror of ``srt_ingest._utf8_guard``: True when the UTF-8 reading
    produces only cp1252-representable text, keeping a coincidentally-valid
    UTF-8 cp1252 segment on its real encoding."""
    return all(
        ord(ch) <= 0xFF or ch in _CP1252_EXTRA_CHARACTERS
        for ch in text
    )


def _mixed_encoding_under_per_cue_recovery(raw: bytes) -> bool:
    """True when the per-cue recovery (``srt_ingest._decode_subtitle_bytes``,
    landed 2026-09-28) rescues at least one cue of ``raw`` whose guarded
    UTF-8 reading differs from its CP1252 reading — i.e. the file mixes
    encodings and the recovery changes the outcome: the UTF-8 cue arrives
    intact instead of as whole-file-CP1252 mojibake.

    Only meaningful for bytes that fail the whole-file UTF-8 decode and pass
    the whole-file CP1252 decode (the caller guarantees both); within that
    branch no cue can carry CP1252-hole bytes, so the strict probe below
    cannot raise."""
    for index, part in enumerate(_CUE_SEPARATOR_RE.split(raw)):
        if index % 2 == 1:
            continue  # captured separator bytes: pure ASCII
        try:
            text = part.decode("utf-8-sig")
        except UnicodeDecodeError:
            continue
        if _utf8_guard(text) and text != part.decode("cp1252"):
            return True
    return False


def _input_subtitles_status(repo_root: Path) -> dict[str, Any]:
    """Scan Input/ for subtitle files whose encoding needs a salvage path.

    The subtitle decode chain (``srt_ingest._decode_subtitle_bytes``) reads
    UTF-8 with BOM first, then decodes each blank-line-delimited cue on its
    own encoding — guarded UTF-8 wins per cue, CP1252 next, replacement as
    the last resort (the 2026-09-28 per-cue recovery). Three shapes are
    worth surfacing *before* a render: a cleanly legacy-encoded
    ``.srt``/``.vtt`` converts via the whole-file CP1252 fallback; a
    MIXED-encoding file (a UTF-8 cue inside a legacy file — the classic
    two-editors concatenation) is no longer a lossy mojibake case, because
    the per-cue recovery decodes that cue's UTF-8 intact, which is worth
    naming so the user knows the salvage will engage; and a file containing
    bytes CP1252 does not define (0x81, 0x8D, 0x8F, 0x90, 0x9D) passes the
    lossy pre-check and degrades that cue to replacement marks in the
    converted script instead of failing conversion.

    Read-only by contract: the path is computed without
    ``ensure_repo_default_paths`` (which mkdirs), and a missing Input/ is
    reported, never created. Verdict policy: fallback files are fine
    (informational); blocked or unreadable files make ``ok`` False so the
    human report and next-steps flag them.
    """
    input_dir = repo_root / "Input"
    report: dict[str, Any] = {
        "ok": True,
        "input_dir": str(input_dir),
        "exists": input_dir.is_dir(),
        "scanned": 0,
        "utf8_count": 0,
        "fallback": [],
        "mixed": [],
        "blocked": [],
        "unreadable": [],
        "error": "",
    }
    if not input_dir.is_dir():
        return report
    subtitle_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in (".srt", ".vtt")
    )
    report["scanned"] = len(subtitle_files)
    for path in subtitle_files:
        rel = path.relative_to(repo_root).as_posix()
        try:
            raw = path.read_bytes()
        except OSError as exc:
            report["unreadable"].append({"path": rel, "error": str(exc)})
            continue
        try:
            raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            # Would take the CP1252 fallback. Mirror convert_srt_file's strict
            # re-decode to distinguish convertible fallback files from ones
            # the conversion gate would reject.
            try:
                raw.decode("cp1252")
            except UnicodeDecodeError:
                report["blocked"].append(
                    {
                        "path": rel,
                        "error": (
                            "contains byte(s) CP1252 cannot decode; per-cue "
                            "recovery degrades that cue to replacement marks "
                            "(U+FFFD) instead of failing. Re-save the file as "
                            "UTF-8 for clean output."
                        ),
                    }
                )
            else:
                if _mixed_encoding_under_per_cue_recovery(raw):
                    report["mixed"].append({"path": rel})
                else:
                    report["fallback"].append({"path": rel})
        else:
            report["utf8_count"] += 1
    report["ok"] = not report["blocked"] and not report["unreadable"]
    if report["blocked"] or report["unreadable"]:
        report["error"] = "; ".join(
            [f"{entry['path']}: {entry['error']}" for entry in report["blocked"]]
            + [f"{entry['path']}: {entry['error']}" for entry in report["unreadable"]]
        )
    return report


def _preview_writers_status(repo_root: Path) -> dict[str, Any]:
    """Scan project ``previews/`` directories for files outside the gated
    owner's naming scheme.

    The writer-manifest net (tests/test_stem_cache_write_path.py) pins the
    SOURCE side: preview audio is produced by ``OraclePipeline.render_preview``
    alone, which names every file through ``ProjectCache.preview_path`` —
    ``preview_<alnum-speaker>_<index>.wav`` inside the project's ``previews/``
    directory. This check is the DISK side: any file in a ``previews/``
    directory that fails ``ProjectCache.is_gated_preview_name`` is runtime
    evidence of a writer that bypassed the gated owner (or a renamed scheme
    the recognizer no longer matches).

    Read-only by contract: directories are probed without
    ``ensure_repo_default_paths`` (which mkdirs), and no directory is created.
    Disposable ``build/`` artifacts are excluded — the smoke harnesses create
    throwaway project trees whose naming drift is meaningless. Verdict
    policy: a foreign file is informational, not a broken install (the gated
    owner's own output is unaffected), so the check reports ``ok=True`` with
    the files listed for review; ``ok`` goes False only if a directory
    cannot be read.
    """
    report: dict[str, Any] = {
        "ok": True,
        "scanned": 0,
        "dirs_scanned": 0,
        "foreign": [],
        "unreadable": [],
        "error": "",
    }
    candidates: list[Path] = []
    try:
        candidates = [
            path
            for path in sorted(repo_root.rglob("previews"))
            if path.is_dir() and "build" not in path.relative_to(repo_root).parts
        ]
    except OSError as exc:
        report["ok"] = False
        report["error"] = f"could not scan for previews/ directories: {exc}"
        return report
    report["dirs_scanned"] = len(candidates)
    from the_oracle.models.cache import ProjectCache

    for previews_dir in candidates:
        try:
            entries = sorted(previews_dir.iterdir())
        except OSError as exc:
            report["ok"] = False
            report["unreadable"].append(
                {"path": previews_dir.relative_to(repo_root).as_posix(), "error": str(exc)}
            )
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            report["scanned"] += 1
            if not ProjectCache.is_gated_preview_name(entry.name):
                report["foreign"].append(
                    {
                        "path": entry.relative_to(repo_root).as_posix(),
                        "name": entry.name,
                    }
                )
    report["ok"] = not report["unreadable"]
    if report["unreadable"]:
        report["error"] = "; ".join(
            f"{entry['path']}: {entry['error']}" for entry in report["unreadable"]
        )
    return report


def _release_metadata_status(repo_root: Path) -> dict[str, Any]:
    """Surface release-metadata drift (scripts/release.py --check) in the report.

    The release invariant — ``__version__`` is the single version source and
    pyproject.toml, the README/STATE banners, and the CHANGELOG section for
    the current version must all agree (the changelog dated the release day)
    — is exactly the kind of per-project drift a doctor should surface before
    a release attempt fails on it. The probes are reused, not reimplemented:
    ``release.py`` is loaded from beside this script and its ``check()`` runs
    read-only (ast/toml/text reads; no writes, no imports of the package).

    Verdict policy: problems make ``ok`` False and drive next-steps, but the
    check deliberately does NOT join ``required_checks`` — a changelog not
    dated today is expected between releases (the release-day gate), not a
    broken install.
    """
    report: dict[str, Any] = {"ok": True, "problems": [], "version": "", "error": ""}
    release_script = Path(__file__).resolve().parent / "release.py"
    spec = importlib.util.spec_from_file_location("oracle_release_tool", release_script)
    if spec is None or spec.loader is None:
        report["ok"] = False
        report["error"] = f"release.py could not be loaded from {release_script}"
        return report
    release = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(release)
        report["version"] = release.read_version(repo_root)
        report["problems"] = release.check(repo_root)
    except Exception as exc:  # a broken probe must never crash the doctor
        report["ok"] = False
        report["error"] = f"release check failed to run: {exc}"
        return report
    report["ok"] = not report["problems"]
    if report["problems"]:
        report["error"] = "; ".join(report["problems"])
    return report


_RESOLVED_MARKER_RE = re.compile(r"resolved", re.IGNORECASE)
_COMMIT_CITE_RE = re.compile(r"\(commits? (?P<body>[^)]*)\)|`(?P<ticked>[0-9a-fA-F]{7,40})`")
_HEX_TOKEN_RE = re.compile(r"[0-9a-f]{7,40}")


def _resolved_records_status(repo_root: Path) -> dict[str, Any]:
    """Surface stale-resolved records: entries marked RESOLVED whose cited
    commit is not an ancestor of HEAD.

    A RESOLVED marker claims that specific landed work closed the entry; when
    the cited commit still resolves in git but sits outside HEAD's history
    (rewritten history, a rebased-away branch), the record reads green while
    the evidence HEAD shows is gone — exactly the drift the records-hygiene
    doctrine's "its holding commit is cited" rule exists to catch.

    Division of labor with the record-integrity net (suite-side): the net
    fails on UNRESOLVABLE citations, so this runtime check reports those
    informationally only (a shallow clone cannot verify either) and treats
    the resolvable-but-not-ancestor case as its finding. Ancestry probes are
    read-only (git rev-parse / merge-base); verdicts are cached per token so
    a token cited in several records is verified once and attributed
    everywhere (the every-occurrence idiom).

    Scan scope: root ``*.md``/``*.log`` records plus ``docs/**.md``, split on
    blank lines into paragraphs; a paragraph containing "RESOLVED"
    (case-insensitive) is a resolved entry, and its citations are the
    ``(commit ...)`` bodies and backticked 7-40 hex tokens inside it.
    Deliberately NOT part of ``overall_ready``: records hygiene is not
    install health (same precedent as release_metadata/licensing) — it
    surfaces via the human report and next steps.
    """
    if not (repo_root / ".git").exists():
        return {
            "ok": True,
            "skipped": True,
            "reason": "no .git here — commit ancestry cannot be verified",
            "scanned": 0,
            "stale": [],
            "unverifiable": [],
        }

    def _git(*args: str) -> int:
        result = _run_command(["git", "-C", str(repo_root), *args], timeout=30)
        return int(result.get("returncode", 1))

    verdicts: dict[str, str] = {}

    def _verdict(token: str) -> str:
        if token not in verdicts:
            if _git("rev-parse", "--verify", "--quiet", f"{token}^{{commit}}") == 0:
                verdicts[token] = "stale" if _git("merge-base", "--is-ancestor", token, "HEAD") != 0 else "ancestor"
            else:
                verdicts[token] = "unverifiable"
        return verdicts[token]

    stale: list[dict[str, str]] = []
    unverifiable: list[dict[str, str]] = []
    scanned = 0
    files = sorted(repo_root.glob("*.md")) + sorted(repo_root.glob("*.log"))
    docs = repo_root / "docs"
    if docs.is_dir():
        files.extend(sorted(docs.rglob("*.md")))
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(repo_root).as_posix()
        offset = 0
        for paragraph in text.split("\n\n"):
            lineno = text.count("\n", 0, offset) + 1
            offset += len(paragraph) + 2
            if not _RESOLVED_MARKER_RE.search(paragraph):
                continue
            scanned += 1
            tokens: set[str] = set()
            for match in _COMMIT_CITE_RE.finditer(paragraph):
                body = match.group("body") or match.group("ticked") or ""
                tokens.update(t.lower() for t in _HEX_TOKEN_RE.findall(body))
            for token in sorted(tokens):
                verdict = _verdict(token)
                entry = {"file": rel, "line": str(lineno), "token": token}
                if verdict == "stale":
                    stale.append(entry)
                elif verdict == "unverifiable":
                    unverifiable.append(entry)
    return {
        "ok": not stale,
        "skipped": False,
        "scanned": scanned,
        "stale": stale,
        "unverifiable": unverifiable,
        "detail": (
            f"{scanned} RESOLVED entry(ies) checked"
            + (f"; {len(stale)} stale" if stale else "")
            + (f"; {len(unverifiable)} unverifiable here" if unverifiable else "")
        ),
    }


def _python_status() -> dict[str, Any]:
    version_tuple = sys.version_info[:3]
    ok = SUPPORTED_PYTHON_MIN <= version_tuple < SUPPORTED_PYTHON_MAX
    return {
        "ok": ok,
        "executable": sys.executable,
        "version": platform.python_version(),
    }


def _ffmpeg_status() -> dict[str, Any]:
    path = shutil.which("ffmpeg")
    return {"ok": path is not None, "path": path or ""}


def _entrypoint_status(repo_root: Path) -> dict[str, Any]:
    venv_entrypoint = venv_entrypoint_path(repo_root, "the-oracle")
    wrapper_path = managed_launcher_path("the-oracle")
    path_entrypoint = shutil.which("the-oracle")
    if is_windows() and not path_entrypoint:
        path_entrypoint = shutil.which("the-oracle.cmd")
    managed_wrapper = False
    if wrapper_path.exists():
        try:
            managed_wrapper = MANAGED_WRAPPER_MARKER in wrapper_path.read_text(encoding="utf-8")
        except Exception:
            managed_wrapper = False

    help_target = None
    if venv_entrypoint.exists():
        help_target = str(venv_entrypoint)
    elif path_entrypoint:
        help_target = path_entrypoint

    help_result = {"ok": False, "returncode": 127, "stdout": "", "stderr": ""}
    if help_target:
        help_result = _run_command([help_target, "--help"], cwd=repo_root, timeout=30)

    if is_windows():
        fresh_shell = _run_command(
            ["cmd", "/d", "/c", "where the-oracle >nul 2>nul && the-oracle --help >nul 2>nul"],
            cwd=repo_root,
            timeout=30,
        )
    elif shutil.which("bash"):
        fresh_shell = _run_command(
            ["bash", "-lc", "command -v the-oracle && the-oracle --help >/dev/null"],
            cwd=repo_root,
            timeout=30,
        )
    else:
        fresh_shell = {"ok": help_result["ok"], "stdout": help_target or "", "stderr": help_result["stderr"]}
    normalized_entries = {entry.lower() if is_windows() else entry for entry in path_entries()}
    launcher_dir = str(managed_launcher_dir())
    launcher_entry = launcher_dir.lower() if is_windows() else launcher_dir
    path_has_local_bin = launcher_entry in normalized_entries

    return {
        "ok": bool(help_target) and help_result["ok"] and fresh_shell["ok"],
        "venv_entrypoint": str(venv_entrypoint),
        "venv_entrypoint_exists": venv_entrypoint.exists(),
        "path_entrypoint": path_entrypoint or "",
        "managed_wrapper_path": str(wrapper_path),
        "managed_wrapper_installed": managed_wrapper,
        "help_ok": help_result["ok"],
        "help_error": help_result["stderr"] or help_result["stdout"],
        "fresh_shell_help_ok": fresh_shell["ok"],
        "fresh_shell_path": fresh_shell["stdout"].strip(),
        "fresh_shell_error": fresh_shell["stderr"].strip() or ("the-oracle is not available in a fresh shell PATH" if not fresh_shell["ok"] else ""),
        "path_has_local_bin": path_has_local_bin,
    }


def _chatterbox_probe(repo_root: Path, timeout: float, skip_model_init: bool) -> dict[str, Any]:
    code = f"""
from __future__ import annotations
import json
import time

payload = {{}}
try:
    import perth
except Exception as exc:
    payload["perth_ok"] = False
    payload["perth_error"] = f"{{type(exc).__name__}}: {{exc}}"
    payload["watermarker_callable"] = False
else:
    watermarker = getattr(perth, "PerthImplicitWatermarker", None)
    payload["perth_ok"] = True
    payload["watermarker_callable"] = callable(watermarker)
    payload["watermarker_symbol"] = str(watermarker)

try:
    from chatterbox.tts import ChatterboxTTS
except Exception as exc:
    payload["import_ok"] = False
    payload["import_error"] = f"{{type(exc).__name__}}: {{exc}}"
    payload["init_ok"] = False
else:
    payload["import_ok"] = True
    payload["import_target"] = "from chatterbox.tts import ChatterboxTTS"
    payload["constructor_symbol"] = str(ChatterboxTTS)
    if {skip_model_init!r}:
        payload["init_ok"] = False
        payload["init_skipped"] = True
    else:
        try:
            started = time.perf_counter()
            model = ChatterboxTTS.from_pretrained(device="cpu")
        except Exception as exc:
            payload["init_ok"] = False
            payload["init_error"] = f"{{type(exc).__name__}}: {{exc}}"
        else:
            payload["init_ok"] = True
            payload["init_seconds"] = round(time.perf_counter() - started, 3)
            payload["sample_rate"] = int(getattr(model, "sr", 0) or 0)

print({JSON_PREFIX!r} + json.dumps(payload))
"""
    if not skip_model_init:
        # BUG-1 (2026-09-28): the model-init probe can legitimately download
        # the model for minutes on a first run while the doctor printed
        # nothing — a healthy bootstrap was indistinguishable from a hang.
        # Announce BEFORE spawning and name both escape hatches; the
        # timeout itself stays the operator's choice (CI always pairs --ci
        # with --skip-model-init, and a dev checkout's bootstrap download is
        # designed behavior, so no offline clamp is forced here).
        print(
            "Chatterbox model check: a first run may download the model "
            f"(this can take several minutes; bounded by --model-timeout, "
            f"currently {timeout:g}s; skip entirely with --skip-model-init).",
            file=sys.stderr,
        )
    probe = _run_python_probe(repo_root, code, timeout=timeout, extra_env={"PYTHONWARNINGS": "ignore"})
    probe["ok"] = bool(probe.get("import_ok")) and (skip_model_init or bool(probe.get("init_ok"))) and bool(probe.get("perth_ok"))
    return probe


def _find_qt_xcb_plugin() -> Path | None:
    if not is_linux():
        return None
    try:
        from PySide6 import __file__ as pyside_file
        from PySide6.QtCore import QLibraryInfo
    except Exception:
        return None

    candidates = []
    try:
        plugins_root = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
        candidates.append(plugins_root / "platforms" / "libqxcb.so")
    except Exception:
        pass

    pyside_root = Path(pyside_file).resolve().parent
    candidates.append(pyside_root / "Qt" / "plugins" / "platforms" / "libqxcb.so")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1] if candidates else None


def _qt_status(repo_root: Path, timeout: float) -> dict[str, Any]:
    try:
        import PySide6  # noqa: F401
    except Exception as exc:
        return {
            "ok": False,
            "import_ok": False,
            "plugin_path": "",
            "plugin_exists": False,
            "missing_libraries": [],
            "suggested_packages": [],
            "offscreen_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    plugin_path = _find_qt_xcb_plugin()
    if is_linux() and plugin_path is None:
        return {
            "ok": False,
            "import_ok": True,
            "plugin_path": "",
            "plugin_exists": False,
            "missing_libraries": [],
            "suggested_packages": [],
            "offscreen_ok": False,
            "error": "Could not locate PySide6 xcb platform plugin.",
        }

    missing_libraries: list[str] = []
    ldd_result = {"ok": True, "stderr": "", "stdout": ""}
    if is_linux() and plugin_path is not None:
        ldd_result = _run_command(["ldd", str(plugin_path)], timeout=30)
        if ldd_result["ok"]:
            for line in ldd_result["stdout"].splitlines():
                if "=> not found" in line:
                    missing_libraries.append(line.split("=>", 1)[0].strip())

    offscreen_code = f"""
from __future__ import annotations
import json
import os

payload = {{}}
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    player = QMediaPlayer()
    audio = QAudioOutput()
    player.setAudioOutput(audio)
except Exception as exc:
    payload["ok"] = False
    payload["error"] = f"{{type(exc).__name__}}: {{exc}}"
else:
    payload["ok"] = True
    payload["qt_platform"] = app.platformName()
    payload["qmedia_player"] = str(type(player).__name__)
    app.quit()

print({JSON_PREFIX!r} + json.dumps(payload))
"""
    offscreen = _run_python_probe(repo_root, offscreen_code, timeout=timeout, extra_env={"QT_QPA_PLATFORM": "offscreen"})
    suggested_packages = _qt_package_suggestions(missing_libraries)
    return {
        "ok": (not is_linux() or bool(plugin_path and plugin_path.exists())) and not missing_libraries and bool(offscreen.get("ok")),
        "import_ok": True,
        "plugin_path": str(plugin_path) if plugin_path is not None else "",
        "plugin_exists": bool(plugin_path and plugin_path.exists()) if is_linux() else True,
        "missing_libraries": missing_libraries,
        "suggested_packages": suggested_packages,
        "offscreen_ok": bool(offscreen.get("ok")),
        "offscreen_error": offscreen.get("error") or offscreen.get("stderr_tail", ""),
        "qt_platform": offscreen.get("qt_platform", ""),
        "ldd_error": "" if ldd_result["ok"] else ldd_result["stderr"] or ldd_result["stdout"],
    }


def _deterministic_smoke_status(repo_root: Path) -> dict[str, Any]:
    _prepend_repo_src(repo_root)
    try:
        from unittest.mock import patch

        from the_oracle.smoke import run_deterministic_smoke_render, smoke_output_problem
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    output_root = repo_root / "build" / "doctor_deterministic_smoke"
    started = time.perf_counter()
    try:
        # Keep the doctor smoke deterministic and lightweight by forcing the
        # text-repair helpers onto their built-in fallback paths.
        with (
            patch("the_oracle.text_repair.grammar.GrammarCorrector._try_load_language_tool", return_value=None),
            patch("the_oracle.text_repair.punctuation.PunctuationRestorer._try_load_punctuator", return_value=None),
        ):
            result = run_deterministic_smoke_render(output_root, source_format="txt")
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # The render call can return without raising yet produce no usable audio
    # (deleted/moved output, empty file). The verdict policy is owned by the
    # smoke module so the standalone runner applies the same standard.
    problem = smoke_output_problem(result)
    if problem is not None:
        return {"ok": False, "error": problem}

    return {
        "ok": True,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "output_path": str(result.output_path),
        "project_dir": str(result.project_dir),
        "cache_reused_on_second_pass": result.cache_reused_on_second_pass,
    }


def _real_engine_smoke_status(repo_root: Path) -> dict[str, Any]:
    _prepend_repo_src(repo_root)
    try:
        from the_oracle.real_engine_smoke import real_engine_smoke_prerequisites
    except Exception as exc:
        return {"ok": False, "ready": False, "error": f"{type(exc).__name__}: {exc}"}

    output_root = repo_root / "build" / "real_engine_smoke"
    try:
        # Read-only: a check must not create state. Generating the smoke's inputs
        # here wrote reference clips into build/real_engine_smoke/inputs, which
        # the voice-source audit counts — so run 1 reported fallback=0 and run 2
        # fallback=2 on an unchanged machine. The smoke script generates its own
        # inputs (real_engine_smoke.py) when it is actually run, and
        # real_engine_smoke_prerequisites already reports input existence and
        # whether they can be generated.
        readiness = real_engine_smoke_prerequisites(output_root)
    except Exception as exc:
        return {"ok": False, "ready": False, "error": f"{type(exc).__name__}: {exc}"}

    result = dict(readiness)
    # This check evaluates the prerequisites to run the real-engine smoke; it
    # never runs the render itself. Report whether an output actually exists so
    # the human report cannot present a path that was never produced (a fresh
    # install has no build/real_engine_smoke/real_engine_smoke.flac at all).
    expected = readiness.get("expected_paths") or {}
    output_path = expected.get("output")
    result["ok"] = bool(readiness.get("ready"))
    result["output_exists"] = bool(output_path) and Path(str(output_path)).is_file()
    return result


_VULKAN_PATCH_MARKER = "ORACLE VENDORED PATCH"


def _vulkaninfo_summary() -> tuple[bool, str]:
    """Return (vulkan_device_visible, vulkaninfo_text).

    ``vulkaninfo --summary`` exits 0 only when at least one device is visible,
    so the exit code is the device-visibility probe.
    """
    if not shutil.which("vulkaninfo"):
        return False, ""
    result = _run_command(["vulkaninfo", "--summary"], timeout=15)
    return result["ok"], f"{result['stdout']}\n{result['stderr']}"


def _first_device_name(vulkaninfo_text: str) -> str:
    for line in vulkaninfo_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("deviceName"):
            return stripped.split("=", 1)[-1].strip()
    return ""


_AUDIOCPP_DEVICE_LINE = re.compile(r'^Vulkan:(\d+)\s+"([^"]+)"')


def _probe_audiocpp_devices(binary: Path | None) -> dict[str, Any]:
    """Run ``audiocpp_cli --list-devices`` and report what actually happened.

    Returns ``{"ran": bool, "devices": [...], "detail": str}``.

    ``ran`` is the evidence a verdict needs, and it is why this reports the probe
    rather than just its output. A path that merely *exists* proves nothing: with
    ``ORACLE_AUDIOCPP_CLI`` pointing at a file that cannot execute, the doctor
    reported ``ok: True`` because ``find_audiocpp_binary`` returns anything that
    ``exists()``, while this very probe returned no devices -- a PASS graded from
    a filename, next to evidence gathered in the same run that contradicted it.

    Each device is ``{"index": <n>, "name": "..."}`` where ``<n>`` is the value
    to pass as ``ORACLE_AUDIOCPP_DEVICE`` / ``--device <n>``. audio.cpp's own
    indexes are what the backend uses, so this is the authoritative answer for
    multi-GPU machines (vulkaninfo alone cannot tell us which index audio.cpp
    picks).
    """
    if binary is None or not Path(binary).exists():
        return {"ran": False, "devices": [], "detail": "no audio.cpp binary found"}
    result = _run_command([str(binary), "--backend", "vulkan", "--list-devices"], timeout=30)
    if not result["ok"]:
        return {
            "ran": False,
            "devices": [],
            "detail": (
                f"{binary} exists but did not run: --list-devices exited "
                f"{result.get('returncode')}"
                + (" (timed out)" if result.get("timed_out") else "")
            ),
        }
    devices: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    # ggml builds often print device discovery lines to stderr; parse both
    # streams so a stream change can't silently report zero devices, and
    # dedupe by index in case a build echoes the same device to both.
    for stream in (result["stdout"], result["stderr"]):
        for line in (stream or "").splitlines():
            match = _AUDIOCPP_DEVICE_LINE.match(line.strip())
            if match:
                index = int(match.group(1))
                if index in seen_indexes:
                    continue
                seen_indexes.add(index)
                devices.append({"index": index, "name": match.group(2)})
    return {"ran": True, "devices": devices, "detail": ""}


def _vulkan_patches_applied(repo_root: Path) -> bool | None:
    """None when audio.cpp is not cloned; True/False when it is."""
    markers = [
        repo_root / "audio.cpp" / "external" / "ggml" / "src" / "ggml-vulkan" / "ggml-vulkan.cpp",
        repo_root / "audio.cpp" / "external" / "sentencepiece" / "CMakeLists.txt",
    ]
    if not markers[0].exists():
        return None
    try:
        return all(_VULKAN_PATCH_MARKER in path.read_text(encoding="utf-8", errors="ignore") for path in markers)
    except Exception:
        return False


def _vulkan_caveat(*, rdna1_device: bool, patched: bool | None, binary_built: bool) -> str:
    notes: list[str] = []
    if rdna1_device:
        notes.append(
            "RDNA1 (gfx1010/gfx1012) GPU detected: audio.cpp's ggml can hit "
            "VK_ERROR_DEVICE_LOST during buffer init unless the vendored "
            "ORACLE_VENDORED ggml patch is applied (whisper.cpp#3611)."
        )
    if binary_built and patched is False:
        notes.append(
            "The vendored RDNA1 ggml patch is NOT applied to audio.cpp/external/ggml; "
            "re-run scripts/patch_audio_cpp_ggml.sh and rebuild before using "
            "--inference-backend vulkan on RDNA1."
        )
    if notes:
        notes.append("Fall back to --inference-backend pytorch if the device-lost error still fires.")
    return " ".join(notes)


def _vulkan_backend_status(repo_root: Path) -> dict[str, Any]:
    """Informational (opt-in) check: binary, model env, Vulkan device, RDNA1 caveat.

    The Vulkan backend is opt-in (inference_backend: vulkan, default pytorch), so
    this never gates overall_ready -- it reports readiness and surfaces the RDNA1
    device-lost caveat when an RDNA1 GPU or an unpatched clone is detected.
    """
    _prepend_repo_src(repo_root)
    try:
        from the_oracle.tts_engines.vulkan_backend import _vulkan_batch_max_requests, find_audiocpp_binary, find_audiocpp_model
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "binary_built": False,
            "binary_runs": False,
            "binary": "",
            "cli_override": "",
            "model_override_set": False,
            "model_path": "",
            "model_file_exists": False,
            "model_auto_found": False,
            "vulkan_device": False,
            "device_name": "",
            "rdna1_device": False,
            "vendored_patch_applied": None,
            "device_index_env": "",
            "threads_env": "",
            "batch_env": "",
            "effective_batch_cap": 32,
            "audio_cpp_devices": [],
            "caveat": "",
        }

    binary = find_audiocpp_binary()
    model_env = os.environ.get("ORACLE_AUDIOCPP_MODEL", "")
    model_override = Path(model_env).expanduser() if model_env else None
    # Mirrors AudioCppVulkanEngine.ensure_model_ready: the env override always
    # wins (a dangling value must surface, not silently fall through); without
    # one, the repo-local install the download script writes to is
    # auto-detected -- no export required.
    if model_override is not None:
        model_path = model_override
        auto_model = None
    else:
        auto_model = find_audiocpp_model()
        model_path = auto_model
    model_file_exists = bool(model_path and model_path.exists())
    model_auto_found = auto_model is not None
    device_available, vulkaninfo_text = _vulkaninfo_summary()
    lowered = vulkaninfo_text.lower()
    rdna1_device = any(token in lowered for token in ("rdna1", "navi1", "navi10", "navi14"))
    patched = _vulkan_patches_applied(repo_root)
    # The effective per-subprocess request cap: same single source of truth as
    # the engine (ORACLE_AUDIOCPP_MAX_BATCH, clamped >= 1, default 32).
    batch_env = os.environ.get("ORACLE_AUDIOCPP_MAX_BATCH", "")
    effective_batch_cap = _vulkan_batch_max_requests()
    # The one execution-backed fact in this check: the binary is not merely on
    # disk, it ran. Everything else it reports is a stat of one kind or another,
    # so without this a corrupt or incompatible build passes as readily as a
    # working one -- and the RDNA1 caveat below warns about exactly that case.
    device_probe = _probe_audiocpp_devices(binary)
    binary_runs = bool(device_probe["ran"])
    return {
        "ok": bool(binary) and binary_runs and model_file_exists and device_available,
        "binary_built": binary is not None,
        "binary_runs": binary_runs,
        "binary": str(binary) if binary else "",
        "cli_override": os.environ.get("ORACLE_AUDIOCPP_CLI", ""),
        "model_override_set": bool(model_env),
        "model_path": str(model_path) if model_path else "",
        "model_file_exists": model_file_exists,
        "model_auto_found": model_auto_found,
        "vulkan_device": device_available,
        "device_name": _first_device_name(vulkaninfo_text),
        "rdna1_device": rdna1_device,
        "vendored_patch_applied": patched,
        "device_index_env": os.environ.get("ORACLE_AUDIOCPP_DEVICE", ""),
        "threads_env": os.environ.get("ORACLE_AUDIOCPP_THREADS", ""),
        "batch_env": batch_env,
        "effective_batch_cap": effective_batch_cap,
        "audio_cpp_devices": device_probe["devices"],
        "caveat": _vulkan_caveat(rdna1_device=rdna1_device, patched=patched, binary_built=binary is not None),
        # Surface the probe's own failure, so a not-ok verdict always says why.
        "error": ("" if not binary or binary_runs else str(device_probe["detail"])),
    }


def _cuda_backend_status(repo_root: Path) -> dict[str, Any]:
    """Report NVIDIA/CUDA readiness without making CUDA a required check."""
    _prepend_repo_src(repo_root)
    try:
        from the_oracle.device_support import cuda_devices, cuda_reason, cuda_runtime_available

        devices = cuda_devices()
        runtime_available = cuda_runtime_available()
        return {
            "ok": runtime_available,
            "runtime_available": runtime_available,
            "reason": cuda_reason(),
            "devices": [
                {
                    "index": device.index,
                    "name": device.name,
                    "vram_gib": device.vram_gib,
                    "driver_version": device.driver_version,
                    "torch_available": device.torch_available,
                    "suitable": device.suitable,
                    "reason": device.reason,
                }
                for device in devices
            ],
        }
    except Exception as exc:
        return {
            "ok": False,
            "runtime_available": False,
            "reason": f"CUDA probe failed: {type(exc).__name__}: {exc}",
            "devices": [],
        }


def _turbo_status(repo_root: Path, timeout: float) -> dict[str, Any]:
    code = f"""
from __future__ import annotations
import json

from the_oracle.tts_engines.chatterbox_engine import turbo_readiness_report

payload = turbo_readiness_report(device="cpu")
print({JSON_PREFIX!r} + json.dumps(payload))
"""
    probe = _run_python_probe(repo_root, code, timeout=timeout, extra_env={"PYTHONWARNINGS": "ignore"})
    return {
        "ok": bool(probe.get("ok")),
        "cached": bool(probe.get("cached")),
        "checkpoint_dir": probe.get("checkpoint_dir", ""),
        "sample_rate": probe.get("sample_rate"),
        "error": probe.get("error") or probe.get("stderr_tail", "") or probe.get("stdout_tail", ""),
    }


def _build_next_steps(report: dict[str, Any], *, ci_mode: bool) -> list[str]:
    steps: list[str] = []
    if not report["python"]["ok"]:
        if is_windows():
            steps.append(r"Install Python 3.12, make sure the `py` launcher can find it, then rerun .\bootstrap_oracle_tts.ps1.")
        else:
            steps.append("Install Python 3.12 with venv support: sudo apt install python3.12 python3.12-venv")

    runtime_packages: list[str] = []
    if not report["ffmpeg"]["ok"] and not ci_mode:
        if is_windows():
            steps.append("Install FFmpeg and add it to PATH, then rerun the doctor.")
        else:
            runtime_packages.append("ffmpeg")
    runtime_packages.extend(report["qt"]["suggested_packages"] if not ci_mode else [])
    if runtime_packages and is_linux():
        unique_packages = " ".join(sorted(set(runtime_packages)))
        steps.append(f"Install the missing Linux runtime packages: sudo apt install {unique_packages}")

    if not report["entrypoint"]["ok"] and not ci_mode:
        steps.append(
            f"Re-run {repo_bootstrap_display()} to refresh the project venv and install the managed launcher at {managed_launcher_path()}."
        )
        if not report["entrypoint"]["path_has_local_bin"]:
            if is_windows():
                steps.append(f"Add {managed_launcher_dir()} to PATH, open a new PowerShell session, and retry.")
            else:
                steps.append(f'Add {managed_launcher_dir()} to PATH, open a fresh shell, and retry: export PATH="{managed_launcher_dir()}:$PATH"')

    chatterbox_init_blocked = not report["chatterbox_init"]["ok"] and not report["chatterbox_init"]["skipped"]
    if not report["chatterbox_import"]["ok"] or chatterbox_init_blocked or not report["perth"]["ok"]:
        steps.append(f"Re-run {repo_bootstrap_display()} with internet access so Chatterbox and Perth can be installed and cached on CPU.")

    pins = report.get("dependency_pins") or {"ok": True, "error": ""}
    if not pins["ok"]:
        detail = pins["error"] or "the installed venv does not match the pyproject pins"
        steps.append(
            "Dependency drift detected: " + detail
            + f" Re-run {repo_bootstrap_display()} (or the oracle update) to restore the declared dependency set."
        )

    lic = report.get("licensing") or {"ok": True, "state": "no_token"}
    if not lic["ok"]:
        steps.append(
            f"License issue ({lic.get('state')}): {lic.get('detail') or 'see the license line above.'}"
        )

    crash_state = report.get("crash_reports") or {"ok": True, "native_dump_present": False}
    if not crash_state["ok"]:
        steps.append(f"Crash capture problem: {crash_state.get('detail')}")
    if crash_state.get("native_dump_present"):
        steps.append(
            "A native-crash dump exists in crash_reports/native-crash.txt — this is the "
            "data the GUI segfault watch item needed; attach it to a support report."
        )

    if not report["deterministic_smoke"]["ok"]:
        steps.append(f"Inspect the deterministic smoke failure above, then retry with {repo_python_display()} scripts/download_models.py or {repo_python_display()} scripts/smoke_render.py as needed.")

    if not report["real_engine_smoke"]["ok"]:
        steps.append("Real-engine smoke becomes ready after the Chatterbox import/init and Perth checks pass.")

    if not report["turbo"]["ok"] and not ci_mode:
        steps.append(f"Optional turbo prefetch: {repo_python_display()} scripts/download_models.py --variant turbo --device cpu")

    vulkan = report["vulkan_backend"]
    if vulkan["binary_built"] and not vulkan["model_file_exists"]:
        if vulkan.get("model_override_set"):
            steps.append(
                f"Vulkan backend: ORACLE_AUDIOCPP_MODEL points at a missing model file "
                f"({vulkan.get('model_path') or '?'}); fix the variable before "
                f"rendering on Vulkan."
            )
        else:
            steps.append(
                "Vulkan backend: the Chatterbox model is not downloaded; run "
                "`the-oracle setup-vulkan` (or select the Vulkan backend in the GUI) "
                "to fetch it automatically, or scripts/download_audio_cpp_model.sh "
                "(scripts/build_audio_cpp.sh --with-model builds and fetches in one)."
            )
    if vulkan["vendored_patch_applied"] is False:
        steps.append(
            "Vulkan backend: re-run scripts/patch_audio_cpp_ggml.sh to (re)apply the vendored "
            "RDNA1 ggml patch, then rebuild with scripts/build_audio_cpp.sh."
        )
    audio_cpp_devices = vulkan.get("audio_cpp_devices") or []
    if len(audio_cpp_devices) > 1 and not vulkan.get("device_index_env"):
        indexes = ", ".join(str(device["index"]) for device in audio_cpp_devices)
        steps.append(
            f"Vulkan backend: {len(audio_cpp_devices)} GPUs detected by audio.cpp "
            f"(indexes {indexes}); set ORACLE_AUDIOCPP_DEVICE to choose which one renders."
        )
    # The per-subprocess request cap is sensible in roughly 1-128; warn when
    # it is outside that, since a tiny cap destroys batching and a huge one
    # risks an oversized requests.json. The reported effective cap is clamped
    # >= 1 (engine semantics), so check the raw env value for the lower bound:
    # a user setting 0 or -5 silently clamps to 1 and must still be surfaced.
    effective_batch_cap = vulkan.get("effective_batch_cap", 32)
    batch_env = vulkan.get("batch_env") or ""
    raw_cap: int | None = None
    try:
        if batch_env:
            raw_cap = int(batch_env)
    except ValueError:
        raw_cap = None
    cap_too_small = raw_cap is not None and raw_cap < 1
    cap_too_large = effective_batch_cap > 128
    if cap_too_small or cap_too_large:
        steps.append(
            f"Vulkan backend: effective batch cap is {effective_batch_cap} "
            f"(ORACLE_AUDIOCPP_MAX_BATCH={batch_env or 'unset'}); set it "
            f"to a value in the sensible 1-128 range or unset it for the default 32."
        )
    cuda = report.get("cuda_backend", {"devices": [], "runtime_available": True, "reason": "CUDA probe not requested."})
    if cuda.get("devices") and not cuda.get("runtime_available"):
        cuda_update_command = "oracle.ps1 update --pytorch-runtime cuda" if is_windows() else "./oracle update --pytorch-runtime cuda"
        steps.append(
            "CUDA is not currently usable: " + cuda.get("reason", "unknown reason") +
            " Choose CPU, install the CUDA PyTorch runtime with " +
            f"{cuda_update_command}, or replace an undersized GPU."
        )

    input_subs = report.get("input_subtitles") or {"blocked": [], "unreadable": []}
    for entry in input_subs.get("blocked") or []:
        steps.append(
            f"Input/ {entry['path']}: {entry['error']}"
        )
    for entry in input_subs.get("unreadable") or []:
        steps.append(f"Input/ {entry['path']}: could not be read ({entry['error']}).")

    preview_writers = report.get("preview_writers") or {"foreign": []}
    for entry in preview_writers.get("foreign") or []:
        steps.append(
            f"Preview cache: {entry['path']} was not written by the gated "
            "preview owner (OraclePipeline.render_preview names every preview "
            "via ProjectCache.preview_path). Review the file and whatever "
            "wrote it."
        )

    release_meta = report.get("release_metadata") or {}
    for problem in release_meta.get("problems") or []:
        steps.append(f"Release metadata: {problem}")
    if release_meta.get("error") and not release_meta.get("problems"):
        steps.append(f"Release metadata: {release_meta['error']}")

    resolved_state = report.get("resolved_records") or {"ok": True, "stale": []}
    for entry in resolved_state.get("stale") or []:
        steps.append(
            f"Stale-resolved record: {entry['file']}:{entry['line']} cites {entry['token']}, "
            "which is not an ancestor of HEAD (rewritten history or a discarded branch) — "
            "re-verify the resolution and update the record."
        )

    if report["voice_sources"]["primary_source"] != "seashells":
        steps.append("Add curated local reference clips to ./Seashells so the GUI stops defaulting to smoke/build fallback voices.")

    if not steps:
        steps.append(f"Ready to launch: {repo_run_display()}")
    return steps


def run(repo_root: Path, *, model_timeout: float, qt_timeout: float, skip_model_init: bool, ci_mode: bool) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    _prepend_repo_src(repo_root)
    from the_oracle.voice_catalog import voice_catalog_audit

    chatterbox_probe = _chatterbox_probe(repo_root, timeout=model_timeout, skip_model_init=skip_model_init)
    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "platform": platform.platform(),
        "ci_mode": ci_mode,
        "python": _python_status(),
        "ffmpeg": _ffmpeg_status(),
        "entrypoint": _entrypoint_status(repo_root),
        "chatterbox_import": {
            "ok": bool(chatterbox_probe.get("import_ok")),
            "target": chatterbox_probe.get("import_target", "from chatterbox.tts import ChatterboxTTS"),
            "constructor_symbol": chatterbox_probe.get("constructor_symbol", ""),
            "error": chatterbox_probe.get("import_error", ""),
        },
        "chatterbox_init": {
            "ok": bool(chatterbox_probe.get("init_ok")),
            "device": "cpu",
            "seconds": chatterbox_probe.get("init_seconds"),
            "sample_rate": chatterbox_probe.get("sample_rate"),
            "skipped": bool(chatterbox_probe.get("init_skipped")),
            "error": chatterbox_probe.get("init_error") or chatterbox_probe.get("error", ""),
        },
        "perth": {
            "ok": bool(chatterbox_probe.get("perth_ok")) and bool(chatterbox_probe.get("watermarker_callable")),
            "watermarker_callable": bool(chatterbox_probe.get("watermarker_callable")),
            "watermarker_symbol": chatterbox_probe.get("watermarker_symbol", ""),
            "error": chatterbox_probe.get("perth_error", ""),
        },
        "turbo": _turbo_status(repo_root, timeout=model_timeout),
        "cuda_backend": _cuda_backend_status(repo_root),
        "qt": _qt_status(repo_root, timeout=qt_timeout),
        "voice_sources": voice_catalog_audit(repo_root),
        "deterministic_smoke": _deterministic_smoke_status(repo_root),
        "real_engine_smoke": _real_engine_smoke_status(repo_root),
        "vulkan_backend": _vulkan_backend_status(repo_root),
        "dependency_pins": _dependency_pin_status(repo_root),
        "input_subtitles": _input_subtitles_status(repo_root),
        "preview_writers": _preview_writers_status(repo_root),
        "release_metadata": _release_metadata_status(repo_root),
        "licensing": _licensing_status(repo_root),
        "crash_reports": _crash_reports_status(repo_root),
        "resolved_records": _resolved_records_status(repo_root),
    }
    required_checks = [
        report["python"]["ok"],
        report["dependency_pins"]["ok"],
        report["chatterbox_import"]["ok"],
        report["perth"]["ok"],
        skip_model_init or report["chatterbox_init"]["ok"],
        report["qt"]["ok"],
        report["deterministic_smoke"]["ok"],
        report["real_engine_smoke"]["ok"],
    ]
    if not ci_mode:
        required_checks.extend([report["ffmpeg"]["ok"], report["entrypoint"]["ok"]])
    report["overall_ready"] = all(required_checks)
    report["next_steps"] = _build_next_steps(report, ci_mode=ci_mode)
    return report


def _print_human_report(report: dict[str, Any]) -> None:
    optional_status = "WARN" if report.get("ci_mode") else "FAIL"
    print(f"Repo root: {report['repo_root']}")
    print(f"Platform: {report['platform']}")
    print(f"{_status(report['python']['ok'])} Python: {report['python']['executable']} ({report['python']['version']})")

    pins = report.get("dependency_pins") or {"ok": True, "error": "", "checked_count": 0}
    if pins["ok"]:
        print(f"{_status(True)} Dependency pins: {pins['checked_count']} requirements match the installed venv")
    else:
        print(f"{_status(False)} Dependency pins: {pins['error']}")

    lic = report.get("licensing") or {"ok": True, "state": "no_token"}
    if lic["ok"] and lic.get("state") == "no_token":
        print(f"{_status(True)} License: not activated — community edition")
    elif lic["ok"]:
        expiry = f", expires epoch {lic['exp']}" if lic.get("exp") else ", perpetual"
        licensee = lic.get("licensee") or "no licensee recorded"
        print(f"{_status(True)} License: {lic.get('edition')} edition for {licensee} ({expiry})")
    else:
        print(f"{_status(False)} License: {lic.get('state')} — {lic.get('detail')}")

    crash_state = report.get("crash_reports") or {"ok": True, "consent": False}
    if not crash_state.get("ok"):
        print(f"{_status(False)} Crash reports: {crash_state.get('detail')}")
    elif crash_state.get("consent"):
        extras = []
        if crash_state.get("record_count"):
            extras.append(f"{crash_state['record_count']} report(s), newest exception: {crash_state.get('newest_exception') or '?'}")
        if crash_state.get("at_cap"):
            extras.append("at cap — oldest will be dropped")
        if crash_state.get("native_dump_present"):
            extras.append("a native-crash dump exists")
        suffix = f" ({'; '.join(extras)})" if extras else ""
        print(f"{_status(True)} Crash reports: enabled{suffix}")
    else:
        print(f"{_status(True)} Crash reports: disabled — local capture off (opt in: the-oracle privacy-opt-in)")

    ffmpeg_detail = report["ffmpeg"]["path"] or "ffmpeg not found on PATH"
    ffmpeg_label = _status(report["ffmpeg"]["ok"]) if report["ffmpeg"]["ok"] or not report.get("ci_mode") else optional_status
    print(f"{ffmpeg_label} Runtime tool `ffmpeg`: {ffmpeg_detail}")

    entrypoint = report["entrypoint"]
    entrypoint_detail = entrypoint["fresh_shell_path"] or entrypoint["path_entrypoint"] or entrypoint["venv_entrypoint"]
    if entrypoint["ok"]:
        print(f"{_status(True)} the-oracle entrypoint: {entrypoint_detail}")
    else:
        detail = entrypoint["fresh_shell_error"] or entrypoint["help_error"] or "the-oracle --help failed"
        label = _status(False) if not report.get("ci_mode") else optional_status
        print(f"{label} the-oracle entrypoint: {detail}")

    chatterbox_import = report["chatterbox_import"]
    if chatterbox_import["ok"]:
        print(f"{_status(True)} Chatterbox import: {chatterbox_import['target']}")
    else:
        print(f"{_status(False)} Chatterbox import: {chatterbox_import['error']}")

    chatterbox_init = report["chatterbox_init"]
    if chatterbox_init["ok"]:
        print(
            f"{_status(True)} Chatterbox CPU init: from_pretrained(device=\"cpu\") in {chatterbox_init['seconds']}s"
        )
    elif chatterbox_init["skipped"]:
        print("SKIP Chatterbox CPU init: skipped")
    else:
        print(f"{_status(False)} Chatterbox CPU init: {chatterbox_init['error']}")

    perth = report["perth"]
    if perth["ok"]:
        print(f"{_status(True)} Perth watermarker: {perth['watermarker_symbol']}")
    else:
        detail = perth["error"] or "PerthImplicitWatermarker is unavailable"
        print(f"{_status(False)} Perth watermarker: {detail}")

    turbo = report["turbo"]
    if turbo["ok"]:
        detail = turbo["checkpoint_dir"] or "cached checkpoint available"
        print(f"{_status(True)} Turbo readiness: {detail}")
    else:
        label = _status(False) if not report.get("ci_mode") else optional_status
        print(f"{label} Turbo readiness: {turbo['error']}")

    voice_sources = report["voice_sources"]
    voice_detail = (
        f"{voice_sources['default_voice_assessment']} "
        f"Seashells={voice_sources['seashell_clip_count']}, fallback={voice_sources['fallback_clip_count']}"
    )
    print(f"{_status(voice_sources['ok'])} Default voice sources: {voice_detail}")
    print(f"Voice assets: {voice_sources['better_local_assets_detail']}")
    print(f"Voice mixing: {voice_sources['voice_mixing_detail']}")

    qt = report["qt"]
    if qt["ok"]:
        detail = qt["plugin_path"] or qt["qt_platform"] or "offscreen probe passed"
        print(f"{_status(True)} Qt GUI prerequisites: {detail}")
    else:
        detail = qt["error"] if "error" in qt else qt["offscreen_error"] or qt["ldd_error"] or "Qt prerequisites failed"
        print(f"{_status(False)} Qt GUI prerequisites: {detail}")
        if qt["missing_libraries"]:
            print(f"Missing Qt libraries: {', '.join(qt['missing_libraries'])}")
        if qt["suggested_packages"]:
            print(f"Suggested packages: {' '.join(qt['suggested_packages'])}")

    deterministic = report["deterministic_smoke"]
    if deterministic["ok"]:
        print(f"{_status(True)} Deterministic smoke readiness: {deterministic['output_path']}")
    else:
        print(f"{_status(False)} Deterministic smoke readiness: {deterministic['error']}")

    real_engine = report["real_engine_smoke"]
    if real_engine["ok"]:
        # "Ready" means the prerequisites are met — this check does not run the
        # smoke. Cite the output path only when a real smoke run has left one,
        # so a freshly installed machine is never told an artifact exists that
        # never did.
        output_display = (real_engine.get("expected_paths") or {}).get("output", "the smoke output")
        if real_engine.get("output_exists"):
            detail = f"prerequisites ready; smoke output present at {output_display}"
        else:
            detail = (
                "prerequisites ready; no smoke output yet — run "
                f"{repo_python_display()} scripts/real_engine_smoke.py to produce {output_display}"
            )
        print(f"{_status(True)} Real-engine smoke readiness: {detail}")
    else:
        detail = real_engine.get("error") or str(real_engine.get("chatterbox_import", {}))
        print(f"{_status(False)} Real-engine smoke readiness: {detail}")

    print("\nCUDA / NVIDIA backend (optional):")
    cuda = report.get("cuda_backend", {"runtime_available": False, "reason": "CUDA probe not included in this report.", "devices": []})
    cuda_label = "PASS" if cuda.get("runtime_available") else "WARN"
    print(f"{cuda_label} {cuda.get('reason', 'CUDA probe unavailable')}")
    for device in cuda.get("devices", []):
        vram = "unknown VRAM" if device.get("vram_gib") is None else f"{device['vram_gib']:.1f} GiB VRAM"
        state = "usable" if device.get("torch_available") and device.get("suitable") else "not suitable"
        print(f"      CUDA {device['index']}: {device['name']} ({vram}) — {state}")

    # readiness and the RDNA1 device-lost caveat are surfaced for the user.
    vulkan = report["vulkan_backend"]
    vulkan_label = "PASS" if vulkan["ok"] else "WARN"
    model_state = "set"
    if not vulkan["model_override_set"]:
        model_state = "unset (auto-detected)" if vulkan.get("model_file_exists") else "unset"
    elif vulkan.get("model_file_exists") is False:
        model_state = f"set but file missing ({vulkan.get('model_path') or '?'})"
    parts = [
        f"binary={'built' if vulkan['binary_built'] else 'not built'}",
        f"ORACLE_AUDIOCPP_MODEL={model_state}",
        f"vulkan device={'yes' if vulkan['vulkan_device'] else 'no'}",
    ]
    if vulkan["device_name"]:
        parts.append(vulkan["device_name"])
    if vulkan["rdna1_device"]:
        parts.append("RDNA1")
    if vulkan.get("device_index_env"):
        parts.append(f"device={vulkan['device_index_env']} (env)")
    if vulkan.get("threads_env"):
        parts.append(f"threads={vulkan['threads_env']} (env)")
    # The effective cap always derives from the env var (or the 32 default) --
    # per-render settings like the GUI spin box live in user-chosen settings
    # files, which the env-level doctor does not read.
    if vulkan.get("batch_env"):
        parts.append(f"batch cap={vulkan.get('effective_batch_cap', 32)} (env)")
    if vulkan["vendored_patch_applied"] is None:
        parts.append("vendored patches=not cloned")
    else:
        parts.append(f"vendored patches={'applied' if vulkan['vendored_patch_applied'] else 'MISSING'}")
    print(f"{vulkan_label} Vulkan backend (audio.cpp, opt-in): {', '.join(parts)}")
    if vulkan["caveat"]:
        print(f"      {vulkan['caveat']}")
    if not vulkan["ok"] and vulkan["error"]:
        print(f"      {vulkan['error']}")
    audio_cpp_devices = vulkan.get("audio_cpp_devices") or []
    if audio_cpp_devices:
        labels = ", ".join(f"{device['index']}={device['name']}" for device in audio_cpp_devices)
        print(f"      audio.cpp Vulkan devices: {labels}")
        if len(audio_cpp_devices) > 1 and not vulkan.get("device_index_env"):
            indexes = ", ".join(str(device["index"]) for device in audio_cpp_devices)
            print(f"      Multi-GPU: set ORACLE_AUDIOCPP_DEVICE to one of [{indexes}] to pick a device.")

    release_meta = report.get("release_metadata")
    if release_meta is not None:
        if release_meta.get("ok"):
            print(
                f"PASS Release metadata: version {release_meta.get('version') or '?'} "
                "consistent (pyproject, banners, CHANGELOG section dated today)"
            )
        elif release_meta.get("problems"):
            print("WARN Release metadata drift:")
            for problem in release_meta["problems"]:
                print(f"      {problem}")
        else:
            print(f"WARN Release metadata: {release_meta.get('error', 'probe failed')}")

    resolved_state = report.get("resolved_records")
    if resolved_state is not None and not resolved_state.get("skipped"):
        if resolved_state["ok"]:
            print(f"PASS Resolved-record ancestry: {resolved_state.get('detail', '')}")
        else:
            print(f"FAIL Resolved-record ancestry: {resolved_state.get('detail', '')}")
        for entry in resolved_state.get("stale") or []:
            print(
                f"      {entry['file']}:{entry['line']}: cites {entry['token']} — "
                "resolves in git but is NOT an ancestor of HEAD"
            )
        for entry in resolved_state.get("unverifiable") or []:
            print(
                f"      {entry['file']}:{entry['line']}: {entry['token']} cannot be "
                "verified here (the suite-side record net owns strictness)"
            )

    subtitles = report.get("input_subtitles")
    if subtitles is None:
        # An older report predating this check: render nothing rather than a
        # misleading SKIP (the check never ran for it).
        return
    if not subtitles.get("exists"):
        print(f"SKIP Input subtitle encoding: Input/ not present ({subtitles.get('input_dir', '')})")
    else:
        parts = [f"{subtitles['scanned']} subtitle file(s) scanned"]
        if subtitles["utf8_count"]:
            parts.append(f"{subtitles['utf8_count']} UTF-8")
        fallback_names = [entry["path"] for entry in subtitles["fallback"]]
        if fallback_names:
            parts.append("CP1252 fallback: " + ", ".join(fallback_names))
        mixed_names = [entry["path"] for entry in subtitles.get("mixed") or []]
        if mixed_names:
            parts.append(
                "mixed-encoding, recovered per cue: " + ", ".join(mixed_names)
            )
        label = "PASS" if subtitles["ok"] else "WARN"
        print(f"{label} Input subtitle encoding: {'; '.join(parts)}")
        for entry in subtitles["blocked"]:
            print(f"      {entry['path']}: {entry['error']}")
        for entry in subtitles["unreadable"]:
            print(f"      {entry['path']}: {entry['error']}")

    previews = report.get("preview_writers")
    if previews is None:
        # An older report predating this check: render nothing rather than a
        # misleading SKIP (the check never ran for it).
        return
    if previews.get("error"):
        print(f"WARN Preview cache audit: {previews['error']}")
    elif not previews.get("dirs_scanned"):
        print("SKIP Preview cache audit: no project previews/ directories present")
    else:
        parts = [
            f"{previews['dirs_scanned']} previews/ dir(s) scanned, "
            f"{previews['scanned']} file(s)"
        ]
        foreign_names = [entry["name"] for entry in previews["foreign"]]
        if foreign_names:
            parts.append("outside the gated naming scheme: " + ", ".join(foreign_names))
        label = "WARN" if previews["foreign"] else "PASS"
        print(f"{label} Preview cache audit: {'; '.join(parts)}")
        for entry in previews["foreign"]:
            print(
                f"      {entry['path']}: not written by the gated preview owner "
                "(OraclePipeline.render_preview via ProjectCache.preview_path) "
                "— review the file and its writer."
            )

    print("")
    print("Next steps:")
    for step in report["next_steps"]:
        print(f"- {step}")


def main(argv: list[str] | None = None) -> int:
    # The doctor constructs the Chatterbox model and prefetches the turbo
    # checkpoint, so on an offline install those probes must resolve from the
    # seeded cache. Applied before run() so every subprocess probe inherits it
    # through _probe_environment().
    from the_oracle.offline import apply_offline_environment

    apply_offline_environment()
    parser = argparse.ArgumentParser(description="Install and launch diagnostics for The Oracle.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--model-timeout", type=float, default=1800.0)
    parser.add_argument("--qt-timeout", type=float, default=60.0)
    parser.add_argument("--skip-model-init", action="store_true")
    parser.add_argument("--ci", action="store_true", help="Ignore optional environment-only checks such as ffmpeg, wrapper PATH, and turbo prefetch.")
    args = parser.parse_args(argv)

    report = run(
        args.repo_root,
        model_timeout=args.model_timeout,
        qt_timeout=args.qt_timeout,
        skip_model_init=args.skip_model_init,
        ci_mode=args.ci,
    )
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        _print_human_report(report)
    return 0 if report["overall_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
