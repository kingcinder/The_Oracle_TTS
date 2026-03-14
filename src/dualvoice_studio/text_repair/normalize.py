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


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(QUOTE_TRANSLATIONS)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()
    for index, character in enumerate(normalized):
        if character.isalpha():
            normalized = normalized[:index] + character.upper() + normalized[index + 1 :]
            break
    return normalized
