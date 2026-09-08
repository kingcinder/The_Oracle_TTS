"""Punctuation-aware pause shaping for utterance pacing.

Chatterbox's *internal* prosody at punctuation is canonical model behavior (identical
on both backends). What The Oracle controls is the silence inserted between
synthesized stems. These helpers shape that silence so pacing reflects how the line
actually ends instead of applying one flat gap to everything.

``pause_for_utterance`` scales a speaker's base turn pause by the utterance's
terminal punctuation; ``chunk_seam_pause_ms`` returns the short breath used between
chunks *of the same utterance* (a full turn pause between chunks of one sentence was
the source of the stacked, punctuation-blind gaps).
"""

from __future__ import annotations

import re

# Pause multiplier by the terminal punctuation category of the synthesized text.
# A plain period keeps the speaker's base pause (factor 1.0), so existing renders
# with period-terminated lines are unchanged by construction.
_TERMINAL_PUNCTUATION: dict[str, float] = {
    ".": 1.0,
    "!": 1.3,
    "?": 1.25,
    "…": 1.6,
}

# Pause multiplier for lines that trail off without a strong terminal (comma,
# semicolon, colon, dash, closing bracket, or no punctuation at all): the line
# continues into the next, so the gap should feel shorter, not longer.
_WEAK_TERMINAL_FACTOR = 0.7

# Trailing quotes/closing brackets/parens that may follow the true terminal.
_TRAILING_SKIP = set("\"'”’)]}»")

_MAX_PAUSE_MS = 2000


def _trailing_punctuation(text: str) -> str:
    """Return the effective terminal punctuation character of ``text``.

    Scans back over quotes/closing brackets so ``"No!"`` and ``(Yes.)`` resolve to
    their real terminal. Returns an empty string when the line ends weakly.
    """
    stripped = text.rstrip()
    if not stripped:
        return ""
    index = len(stripped) - 1
    while index >= 0 and stripped[index] in _TRAILING_SKIP:
        index -= 1
    if index < 0:
        return ""
    char = stripped[index]
    # Ellipsis as three dots (not the single … character).
    if char == "." and index >= 2 and stripped[index - 2 : index + 1] == "...":
        return "…"
    return char if char in _TERMINAL_PUNCTUATION else ""


def pause_for_utterance(text: str, base_pause_ms: int) -> int:
    """Scale ``base_pause_ms`` by the terminal punctuation of ``text``.

    Deterministic and clamped to the pause slider's domain (0-2000 ms). Author
    directives applied later may override the result entirely.
    """
    factor = _TERMINAL_PUNCTUATION.get(_trailing_punctuation(text), _WEAK_TERMINAL_FACTOR)
    scaled = int(round(float(base_pause_ms) * factor))
    return max(0, min(_MAX_PAUSE_MS, scaled))


def chunk_seam_pause_ms(turn_pause_ms: int) -> int:
    """Short breath between chunks of the same utterance (not a full turn pause).

    A chunked utterance is one spoken unit; only its final chunk should carry the
    full ``turn_pause_ms`` gap before the *next* utterance. Seams get roughly a
    third of the turn pause so the flow does not stack double pauses at the seam,
    but never less than a 40 ms breath.
    """
    return max(40, int(round(float(max(0, turn_pause_ms)) * 0.35)))
