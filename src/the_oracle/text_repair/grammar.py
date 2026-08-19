"""Grammar correction with a local LanguageTool-first strategy and a safe fallback."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time

# First use of language_tool_python downloads the LanguageTool server
# (hundreds of MB). On a slow link that would otherwise block every render for
# 30+ minutes; bound the attempt so the local fallback takes over instead.
# Override for CI with ORACLE_LANGUAGE_TOOL_TIMEOUT (seconds).
try:
    LANGUAGE_TOOL_LOAD_TIMEOUT_SECONDS = float(os.environ.get("ORACLE_LANGUAGE_TOOL_TIMEOUT", "25"))
except ValueError:
    LANGUAGE_TOOL_LOAD_TIMEOUT_SECONDS = 25.0

# Set once a load attempt is abandoned this process (either the download is
# missing entirely or a load timed out): another attempt would only race the
# abandoned background download, so fall back for the remainder of the
# process. A fresh process picks the tool up once the download completes and
# populates the cache.
_LANGUAGE_TOOL_ABANDONED = threading.Event()


def _language_tool_download_ready() -> bool:
    """True when the exact LanguageTool version this library build needs is
    already on disk, so a first use won't trigger the hundreds-of-MB download.

    Checks the same extract path the library's own download_lt() looks for, so
    a stale or mismatched cache directory (e.g. a leftover snapshot from a
    different library version) is correctly treated as "not ready". Any
    uncertainty returns True: the bounded load attempt below is the safety
    net and must never be skipped because of our own probe.
    """
    try:
        from language_tool_python import download_lt as lt_download
    except Exception:
        return True
    try:
        # An explicitly configured jar directory means no download is needed.
        if os.environ.get(lt_download.LTP_JAR_DIR_PATH_ENV_VAR):
            return True
        download_folder = lt_download.get_language_tool_download_path()
        version = lt_download.LTP_DOWNLOAD_VERSION
        if version == "latest":
            dirname = f"LanguageTool-{lt_download.LT_SNAPSHOT_CURRENT_VERSION}"
        else:
            filename = lt_download.FILENAME_RELEASE.format(version=version)
            dirname = os.path.splitext(filename)[0]
        extract_path = os.path.join(download_folder, dirname)
        return extract_path in lt_download.find_existing_language_tool_downloads(
            download_folder
        )
    except Exception:
        # Unusual cache layout or library version: assume cached and let the
        # bounded load attempt decide.
        return True


# A stale lock (from a helper that was SIGKILLed) older than this is reaped
# so future renders can retry the download instead of skipping forever.
_WARM_LOCK_STALE_SECONDS = 12 * 60 * 60


def _warm_language_tool_download() -> None:
    """Kick off the first-use download in a detached helper process so the cache
    warms for a later run without blocking this render or printing over its
    progress bars. The helper survives this process, and a lock file in the
    cache dir keeps rapid renders from starting duplicate downloads."""
    if not sys.executable:
        return
    try:
        from language_tool_python import download_lt as lt_download
    except Exception:
        return
    try:
        download_folder = lt_download.get_language_tool_download_path()
    except Exception:
        return
    os.makedirs(download_folder, exist_ok=True)
    lock_path = os.path.join(download_folder, ".oracle_lt_warm.lock")

    # Reap a stale lock from a helper that was killed before it could clean up.
    try:
        if time.time() - os.stat(lock_path).st_mtime > _WARM_LOCK_STALE_SECONDS:
            os.remove(lock_path)
    except FileNotFoundError:
        pass

    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return  # a warm download is already running
    os.close(lock_fd)

    child = (
        "import os\n"
        "from language_tool_python import download_lt\n"
        f"lock = {lock_path!r}\n"
        "try:\n"
        "    download_lt.download_lt()\n"
        "finally:\n"
        "    try:\n"
        "        os.remove(lock)\n"
        "    except OSError:\n"
        "        pass\n"
    )
    kwargs: dict[str, object] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "posix":
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([sys.executable, "-c", child], **kwargs)
    except Exception:
        # Spawning failed; don't leave the lock behind to block later runs.
        try:
            os.remove(lock_path)
        except OSError:
            pass


COMMON_FIXES = {
    " i ": " I ",
    " im ": " I'm ",
    " dont ": " don't ",
    " cant ": " can't ",
    " wont ": " won't ",
    " ive ": " I've ",
    " id ": " I'd ",
}


class GrammarCorrector:
    def __init__(self, *, use_language_tool: bool = True) -> None:
        self._tool = self._try_load_language_tool() if use_language_tool else None

    def _try_load_language_tool(self):
        if _LANGUAGE_TOOL_ABANDONED.is_set():
            return None
        try:
            import language_tool_python
        except Exception:
            return None

        # The first-use download hasn't completed yet: don't stall the render
        # waiting for hundreds of MB on a slow link. Fall back to the local
        # fixes now and warm the cache in the background for a later run.
        if not _language_tool_download_ready():
            _warm_language_tool_download()
            _LANGUAGE_TOOL_ABANDONED.set()
            return None

        result: dict[str, object] = {}

        def _load() -> None:
            try:
                result["tool"] = language_tool_python.LanguageTool("en-US")
            except Exception:
                pass

        loader = threading.Thread(target=_load, name="language-tool-load", daemon=True)
        loader.start()
        loader.join(timeout=LANGUAGE_TOOL_LOAD_TIMEOUT_SECONDS)
        if loader.is_alive():
            # The first-use download is still running; don't hold up the
            # render. The abandoned background download may still finish and
            # cache the tool, so a later run (fresh process) gets the real
            # LanguageTool instead of the local fallback.
            _LANGUAGE_TOOL_ABANDONED.set()
            return None
        return result.get("tool")

    def correct(self, text: str, aggressive: bool = False) -> str:
        if not text.strip():
            return text
        if self._tool is not None:
            try:
                corrected = self._tool.correct(text)
                if corrected.strip():
                    return corrected.strip()
            except Exception:
                pass

        corrected = f" {text.strip()} "
        for source, target in COMMON_FIXES.items():
            corrected = corrected.replace(source, target)
        corrected = corrected.strip()
        corrected = re.sub(r"\s+([,.!?;:])", r"\1", corrected)
        corrected = re.sub(r"\s+", " ", corrected)
        if corrected:
            corrected = corrected[0].upper() + corrected[1:]
        if aggressive:
            corrected = corrected.replace("..", ".")
        return corrected
