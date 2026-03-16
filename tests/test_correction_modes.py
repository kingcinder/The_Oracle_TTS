from the_oracle.correction_modes import (
    CORRECTION_MODE_OPTIONS,
    correction_mode_label,
    normalize_correction_mode,
)


def test_normalize_handles_alias_and_unknown() -> None:
    assert normalize_correction_mode("aggressive") == "aggressive"
    assert normalize_correction_mode("Aggressive") == "aggressive"
    assert normalize_correction_mode("conservative") == "moderate"
    assert normalize_correction_mode("unknown") == "moderate"
    assert normalize_correction_mode(None) == "moderate"


def test_label_round_trip_matches_options() -> None:
    for label, value in CORRECTION_MODE_OPTIONS:
        assert correction_mode_label(value) == label
        assert normalize_correction_mode(label) == value
