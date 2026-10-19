"""Grammar correction with the local fallback.

The optional ``language_tool_python`` client was removed in 1.4.1: its Python
wrapper is GPL-3.0, which is not safe for the proprietary alpha. The local
fixes below are the supported behavior; they are deterministic, offline, and
were already the designed degraded path when LanguageTool was unavailable.

``GrammarCorrector(use_language_tool=...)`` and the ``_try_load_language_tool``
method remain as no-op compatibility points. Existing callers and patch sites
keep working, and a test double can still inject a ``_tool`` object with a
``correct()`` method. No network, download, subprocess, or thread machinery
lives here anymore.
"""

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

# Placeholder that shields ellipses while aggressive mode collapses stray
# double dots. A private-use codepoint keeps it out of real text.
_ELLIPSIS_TOKEN = "\ue000"


def _collapse_double_dots(text: str) -> str:
    """Collapse ``..`` to ``.`` without mangling ellipses.

    Runs of 4+ dots normalise to a single ellipsis first; the ellipsis is
    then shielded while stray double dots collapse, so ``...`` survives and
    ``..`` becomes ``.``.
    """
    text = re.sub(r"\.{4,}", "...", text)
    text = text.replace("...", _ELLIPSIS_TOKEN)
    text = text.replace("..", ".")
    return text.replace(_ELLIPSIS_TOKEN, "...")


def _correct_local_line(line: str, aggressive: bool) -> str:
    corrected = f" {line.strip()} "
    for source, target in COMMON_FIXES.items():
        corrected = corrected.replace(source, target)
    corrected = corrected.strip()
    corrected = re.sub(r"[^\S\n]+([,.!?;:])", r"\1", corrected)
    corrected = re.sub(r"[^\S\n]+", " ", corrected)
    if aggressive:
        corrected = _collapse_double_dots(corrected)
    return corrected


def _capitalize_first(text: str) -> str:
    for index, character in enumerate(text):
        if character.isalpha():
            return text[:index] + character.upper() + text[index + 1 :]
    return text


class GrammarCorrector:
    def __init__(self, *, use_language_tool: bool = True) -> None:
        self._tool = self._try_load_language_tool() if use_language_tool else None

    def _try_load_language_tool(self):
        """Compatibility hook: the external client is no longer installed."""
        return None

    def correct(self, text: str, aggressive: bool = False, *, preserve_newlines: bool = False) -> str:
        """Correct grammar with the local fixes.

        ``preserve_newlines`` keeps paragraph structure: newlines (and blank
        lines between paragraphs) survive instead of being collapsed to
        spaces. Defaults to False, preserving the historical single-line
        output for existing callers. An injected ``_tool`` is still honored
        for test doubles and downstream integrations.
        """
        if not text.strip():
            return text
        if self._tool is not None:
            try:
                corrected = self._tool.correct(text)
                if corrected.strip():
                    return corrected.strip()
            except Exception:
                pass

        if preserve_newlines:
            lines = [_correct_local_line(line, aggressive) for line in text.split("\n")]
            # Mirror the old .strip() on the ends without collapsing
            # interior paragraph breaks. Each line is corrected (and
            # capitalised) independently, like its own utterance.
            while lines and not lines[0]:
                lines.pop(0)
            while lines and not lines[-1]:
                lines.pop()
            return "\n".join(_capitalize_first(line) for line in lines)

        corrected = f" {text.strip()} "
        for source, target in COMMON_FIXES.items():
            corrected = corrected.replace(source, target)
        corrected = corrected.strip()
        corrected = re.sub(r"\s+([,.!?;:])", r"\1", corrected)
        corrected = re.sub(r"\s+", " ", corrected)
        if corrected:
            corrected = corrected[0].upper() + corrected[1:]
        if aggressive:
            corrected = _collapse_double_dots(corrected)
        return corrected
