"""The redaction sanitizer — the privacy core of the crash unit.

Cardinal rule (CRASH_TELEMETRY_DESIGN §4): **drop what it cannot classify.**
Every string that enters a crash record passes through here. Deterministic,
ordered rules:

1. The OS user-home prefix becomes ``~`` (never leaks a username).
2. Known repo-runtime path prefixes become kind labels: ``Input/`` →
   ``<input>``, ``Output/`` → ``<output>``, ``Seashells/`` → ``<voice>``,
   ``Profiles/`` → ``<profile>``.
3. Any remaining absolute-looking path becomes ``<path>``.
4. Anything longer than MAX_FIELD_CHARS is truncated — long free text is
   where transcripts and cast names would hide.

**Mutation contract (M-SANITIZE):** deleting rule 1 (the home-prefix rule)
must fail the property test in tests/test_crash_sanitize.py.
"""

from __future__ import annotations

import re
from pathlib import Path

MAX_FIELD_CHARS = 200
MAX_LOG_TAIL_LINES = 50

_PATH_KIND_RULES: tuple[tuple[str, str], ...] = (
    ("input/", "<input>"),
    ("output/", "<output>"),
    ("seashells/", "<voice>"),
    ("profiles/", "<profile>"),
)

_ABSOLUTE_PATH_RE = re.compile(r"(?:/[A-Za-z0-9._\-]+){2,}")


def _home() -> str:
    try:
        return str(Path.home())
    except (OSError, RuntimeError):  # pragma: no cover - pathological
        return ""


def sanitize_text(text: str, *, home: str | None = None) -> str:
    """Redact one string for a crash record. Pure function of its inputs."""
    if not text:
        return ""
    value = text
    home_prefix = _home() if home is None else home
    if home_prefix and home_prefix != "/" and home_prefix in value:
        value = value.replace(home_prefix, "~")
    lowered = value.lower()
    for marker, label in _PATH_KIND_RULES:
        if marker in lowered:
            # Replace case-preservingly by scanning the lowered copy.
            start = 0
            while True:
                index = lowered.find(marker, start)
                if index < 0:
                    break
                value = value[:index] + label + value[index + len(marker):]
                lowered = lowered[:index] + label + lowered[index + len(marker):]
                start = index + len(label)
    value = _ABSOLUTE_PATH_RE.sub("<path>", value)
    if len(value) > MAX_FIELD_CHARS:
        value = value[:MAX_FIELD_CHARS]
    return value


def sanitize_log_tail(lines: list[str], *, home: str | None = None) -> list[str]:
    """The last MAX_LOG_TAIL_LINES lines, each sanitized and length-capped."""
    return [sanitize_text(line, home=home) for line in lines[-MAX_LOG_TAIL_LINES:]]


def sanitize_frame(filename: str, function: str, lineno: int, *, home: str | None = None) -> str:
    """One traceback frame as ``file:lineno: function`` — no source text,
    no local variables, paths reduced to kinds (design §4's traceback row)."""
    safe_file = sanitize_text(filename, home=home)
    safe_function = sanitize_text(function, home=home)
    return f"{safe_file}:{lineno}: {safe_function}"
