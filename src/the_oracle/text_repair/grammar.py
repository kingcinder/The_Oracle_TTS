"""Grammar correction with a local LanguageTool-first strategy and a safe fallback."""

from __future__ import annotations

import atexit
import re


COMMON_FIXES = {
    " i ": " I ",
    " im ": " I'm ",
    " dont ": " don't ",
    " cant ": " can't ",
    " wont ": " won't ",
    " ive ": " I've ",
    " id ": " I'd ",
}

_SHARED_LANGUAGE_TOOL = None
_LANGUAGE_TOOL_CLOSE_REGISTERED = False


def _close_shared_language_tool() -> None:
    global _SHARED_LANGUAGE_TOOL
    tool = _SHARED_LANGUAGE_TOOL
    _SHARED_LANGUAGE_TOOL = None
    if tool is None:
        return
    try:
        tool.close()
    except Exception:
        pass


class GrammarCorrector:
    def __init__(self) -> None:
        self._tool = self._try_load_language_tool()

    def _try_load_language_tool(self):
        global _SHARED_LANGUAGE_TOOL, _LANGUAGE_TOOL_CLOSE_REGISTERED
        if _SHARED_LANGUAGE_TOOL is not None:
            return _SHARED_LANGUAGE_TOOL
        try:
            import language_tool_python
        except Exception:
            return None
        try:
            _SHARED_LANGUAGE_TOOL = language_tool_python.LanguageTool("en-US")
        except Exception:
            return None
        if not _LANGUAGE_TOOL_CLOSE_REGISTERED:
            atexit.register(_close_shared_language_tool)
            _LANGUAGE_TOOL_CLOSE_REGISTERED = True
        return _SHARED_LANGUAGE_TOOL

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
