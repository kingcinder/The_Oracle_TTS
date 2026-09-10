from __future__ import annotations

import re
import unicodedata


QUOTE_TRANSLATIONS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
)


def normalize_text(text: str, *, preserve_newlines: bool = False) -> str:
    """Normalize quotes, dashes, spacing, and punctuation for TTS input.

    ``preserve_newlines`` keeps paragraph structure: newlines (and blank
    lines between paragraphs) survive instead of being collapsed to spaces.
    Defaults to False, preserving the historical single-line output for
    existing callers.
    """
    if preserve_newlines:
        lines = [_normalize_line(line) for line in text.split("\n")]
        # Mirror .strip() on the ends without collapsing interior
        # paragraph breaks.
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        # Each paragraph is treated like its own text: capitalize its first
        # letter, mirroring the single-line behaviour below.
        normalized = "\n".join(_capitalize_first(line) for line in lines)
    else:
        # Historical single-line behaviour, unchanged for existing callers.
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.translate(QUOTE_TRANSLATIONS)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
        normalized = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.strip()
        normalized = _capitalize_first(normalized)
    return normalized


def _capitalize_first(text: str) -> str:
    for index, character in enumerate(text):
        if character.isalpha():
            return text[:index] + character.upper() + text[index + 1 :]
    return text


def _normalize_line(line: str) -> str:
    normalized = unicodedata.normalize("NFKC", line)
    normalized = normalized.translate(QUOTE_TRANSLATIONS)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"[^\S\n]+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", normalized)
    normalized = re.sub(r"[^\S\n]+", " ", normalized)
    return normalized.strip()
