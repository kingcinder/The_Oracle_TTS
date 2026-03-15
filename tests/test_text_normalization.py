from the_oracle.text_repair.normalization import normalize_text


def test_normalize_text_fixes_quotes_spacing_and_case() -> None:
    value = "  “hello   world”  ,isnt it  "
    normalized = normalize_text(value)

    assert normalized == '"Hello world", isnt it'
