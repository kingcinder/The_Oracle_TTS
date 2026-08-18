"""Prosody and stage directives parsed inline from dialogue text.

Directives are author hints written inside the script, for example:

- ``(tone: sad)`` / ``(emotion: joy)`` — override the detected emotion
- ``(whisper)`` — soften energy and lower temperature
- ``(laughs)`` / ``(sighs)`` — map stage directions to emotions
- ``[pause=500]`` — explicit pause in milliseconds
- ``[exaggeration=0.9]``, ``[temperature=0.9]``, ``[cfg_weight=0.6]`` — sampling overrides
- ``[rate=slow]`` / ``[rate=fast]`` — pacing hint that scales the pause

Matched directives are stripped from the text sent to the TTS engine and
folded into the per-utterance :class:`VoiceSettings`.
"""

from __future__ import annotations

import re
from typing import Any

from the_oracle.models.project import VoiceSettings

_EMOTION_RE = re.compile(r"\(\s*(?:tone|emotion)\s*[:=]\s*([a-z][a-z-]*)\s*\)", re.IGNORECASE)
_STAGE_RE = re.compile(r"\(\s*(whisper|laughs|laughing|sighs)\s*\)", re.IGNORECASE)
_TAG_RE = re.compile(
    r"\[\s*(pause|exaggeration|temperature|cfg_weight|rate)\s*[:=]\s*([0-9.]+|slow|fast)\s*\]",
    re.IGNORECASE,
)

_STAGE_EMOTIONS: dict[str, str] = {"laughs": "joy", "laughing": "joy", "sighs": "sadness"}
_NUMERIC_KEYS = ("pause_ms", "exaggeration", "temperature", "cfg_weight")
# Tag keys that take a number. ``rate`` is the only key that accepts the
# literal ``slow``/``fast`` values, but the shared regex also lets them past
# for other keys, so numeric conversion is guarded below.
_NUMERIC_TAG_KEYS = {"pause", "exaggeration", "temperature", "cfg_weight"}

# Emotion labels authors may write in directives, mapped to the canonical
# labels used by the classifier and GUI (``SUPPORTED_EMOTIONS`` in
# ``emotion/goemotions.py``). Unknown labels are passed through and validated
# by the pipeline, which falls back to the classifier result.
_EMOTION_ALIASES: dict[str, str] = {
    "angry": "anger",
    "mad": "anger",
    "furious": "anger",
    "annoyed": "anger",
    "irritated": "anger",
    "afraid": "fear",
    "scared": "fear",
    "terrified": "fear",
    "nervous": "fear",
    "worried": "fear",
    "happy": "joy",
    "glad": "joy",
    "excited": "joy",
    "delighted": "joy",
    "cheerful": "joy",
    "sad": "sadness",
    "sadly": "sadness",
    "upset": "sadness",
    "heartbroken": "sadness",
    "surprised": "surprise",
    "amazed": "surprise",
    "astonished": "surprise",
    "curious": "curiosity",
    "wondering": "curiosity",
}


def _remove_one(cleaned: str, match: re.Match[str]) -> str:
    return re.sub(r"\s{2,}", " ", (cleaned[: match.start()] + " " + cleaned[match.end() :])).strip()


def parse_directives(text: str) -> tuple[str, dict[str, Any]]:
    """Return ``(cleaned_text, overrides)`` with directives removed from text.

    ``overrides`` may contain ``emotion``, ``whisper``, ``rate``, ``pause_ms``,
    ``exaggeration``, ``temperature``, and ``cfg_weight`` keys.
    """
    cleaned = text.strip()
    overrides: dict[str, Any] = {}

    while True:
        match = _EMOTION_RE.search(cleaned)
        if match is None:
            break
        label = match.group(1).lower()
        overrides["emotion"] = _EMOTION_ALIASES.get(label, label)
        cleaned = _remove_one(cleaned, match)

    while True:
        match = _STAGE_RE.search(cleaned)
        if match is None:
            break
        stage = match.group(1).lower()
        if stage == "whisper":
            overrides["whisper"] = True
        elif "emotion" not in overrides:
            # Stage directions only set emotion when the author hasn't already
            # given an explicit (tone:) or (emotion:) directive for the line.
            overrides["emotion"] = _STAGE_EMOTIONS[stage]
        cleaned = _remove_one(cleaned, match)

    while True:
        match = _TAG_RE.search(cleaned)
        if match is None:
            break
        key = match.group(1).lower()
        value = match.group(2).lower()
        if key == "rate":
            overrides["rate"] = value
        elif key in _NUMERIC_TAG_KEYS:
            # The shared regex also admits ``slow``/``fast`` here; treat a
            # non-numeric value for a numeric key as a typo and skip it
            # rather than crashing the whole render. The tag is still
            # stripped below so the loop always makes progress.
            try:
                number = float(value)
            except ValueError:
                pass
            else:
                if key == "pause":
                    overrides["pause_ms"] = int(round(number))
                else:
                    overrides[key] = number
        cleaned = _remove_one(cleaned, match)

    return cleaned, overrides


def apply_directives(settings: VoiceSettings, overrides: dict[str, Any]) -> VoiceSettings:
    """Return a copy of ``settings`` with numeric directive overrides applied.

    ``emotion`` is intentionally not handled here — the caller decides how an
    explicit emotion label interacts with the classifier.
    """
    merged = VoiceSettings.from_mapping(settings)
    for key in _NUMERIC_KEYS:
        if key in overrides:
            if isinstance(getattr(merged, key), int):
                setattr(merged, key, int(round(float(overrides[key]))))
            else:
                setattr(merged, key, float(overrides[key]))
    if overrides.get("whisper"):
        merged.exaggeration = max(0.1, float(merged.exaggeration) * 0.6)
        merged.temperature = min(1.5, float(merged.temperature) * 0.85)
    rate = overrides.get("rate")
    if rate == "slow":
        merged.pause_ms = max(0, int(round(merged.pause_ms * 1.5)))
    elif rate == "fast":
        merged.pause_ms = max(0, int(round(merged.pause_ms * 0.7)))
    return merged
