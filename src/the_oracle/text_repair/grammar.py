"""Grammar correction with a local LanguageTool-first strategy and a safe fallback."""

from __future__ import annotations

import os
import re
import threading

# First use of language_tool_python downloads the LanguageTool server
# (hundreds of MB). On a slow link that would otherwise block every render for
# 30+ minutes; bound the attempt so the local fallback takes over instead.
# Override for CI with ORACLE_LANGUAGE_TOOL_TIMEOUT (seconds).
try:
    LANGUAGE_TOOL_LOAD_TIMEOUT_SECONDS = float(os.environ.get("ORACLE_LANGUAGE_TOOL_TIMEOUT", "25"))
except ValueError:
    LANGUAGE_TOOL_LOAD_TIMEOUT_SECONDS = 25.0

# Set once a load attempt times out this process: another attempt would only
# race the abandoned background download, so fall back for the remainder of
# the process. A fresh process picks the tool up once the download completes
# and populates the cache.
_LANGUAGE_TOOL_ABANDONED = threading.Event()


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
