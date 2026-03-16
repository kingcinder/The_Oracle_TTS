from __future__ import annotations

VALID_CORRECTION_MODES = {"aggressive", "moderate", "mild", "off"}
LEGACY_CORRECTION_ALIASES = {"conservative": "moderate"}
DEFAULT_CORRECTION_MODE = "moderate"
CORRECTION_MODE_OPTIONS = (
    ("Aggressive", "aggressive"),
    ("Moderate", "moderate"),
    ("Mild", "mild"),
    ("Off", "off"),
)
LABEL_TO_VALUE = {label.lower(): value for label, value in CORRECTION_MODE_OPTIONS}
VALUE_TO_LABEL = {value: label for label, value in CORRECTION_MODE_OPTIONS}


def normalize_correction_mode(value: str | None) -> str:
    if not value:
        return DEFAULT_CORRECTION_MODE
    candidate = value.strip().lower()
    candidate = LABEL_TO_VALUE.get(candidate, candidate)
    candidate = LEGACY_CORRECTION_ALIASES.get(candidate, candidate)
    return candidate if candidate in VALID_CORRECTION_MODES else DEFAULT_CORRECTION_MODE


def correction_mode_label(value: str | None) -> str:
    normalized = normalize_correction_mode(value)
    return VALUE_TO_LABEL.get(normalized, VALUE_TO_LABEL[DEFAULT_CORRECTION_MODE])
