"""Grammar correction with a local LanguageTool-first strategy and a safe fallback."""

from __future__ import annotations

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


class GrammarCorrector:
    def __init__(self) -> None:
        self._tool = self._try_load_language_tool()

    def _try_load_language_tool(self):
        try:
            import language_tool_python
        except Exception:
            return None
        try:
            return language_tool_python.LanguageTool("en-US")
        except Exception:
            return None

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
