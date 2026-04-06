from pathlib import Path

from the_oracle.text_ingest import ingest_text_file


def test_markdown_ingest_preserves_readable_text(tmp_path: Path) -> None:
    source = tmp_path / "scene.md"
    source.write_text(
        "# Scene\n\n**Alice:** hello there\n\n- Bob: how are you\n\n`ignored`\n",
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert document.title == "scene"
    assert len(document.segments) >= 2
    assert document.segments[0].text == "hello there"
    assert document.segments[0].explicit_speaker == "Alice"


def test_markdown_dialogue_preserves_linewise_turns(tmp_path: Path) -> None:
    source = tmp_path / "dialogue.md"
    source.write_text(
        "# Dialogue\n\n"
        "Speaker A: The Oracle is online.\n"
        "Speaker B: Confirm the signal path.\n"
        "Speaker A: Chatterbox is the only backend now.\n"
        "Speaker B: Render complete.\n",
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert len(document.segments) == 4
    assert [segment.explicit_speaker for segment in document.segments] == ["Speaker A", "Speaker B", "Speaker A", "Speaker B"]
    assert document.segments[0].text == "The Oracle is online."
    assert document.segments[3].text == "Render complete."


def test_plaintext_ingest_supports_screenplay_headings(tmp_path: Path) -> None:
    source = tmp_path / "screenplay.txt"
    source.write_text(
        "ALICE\n"
        "We need a cleaner parse.\n"
        "The current transcript is a mess.\n\n"
        "BOB\n"
        "I'll fix the attribution layer.\n",
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert len(document.segments) == 2
    assert document.segments[0].explicit_speaker == "ALICE"
    assert document.segments[0].text == "We need a cleaner parse.\nThe current transcript is a mess."
    assert document.segments[1].explicit_speaker == "BOB"


def test_plaintext_ingest_merges_continuations_after_inline_speaker(tmp_path: Path) -> None:
    source = tmp_path / "continuations.txt"
    source.write_text(
        "Alice: I started the parser rewrite\n"
        "and kept the continuation on the next line.\n"
        "Bob: Good.\n",
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert len(document.segments) == 2
    assert document.segments[0].explicit_speaker == "Alice"
    assert document.segments[0].text == "I started the parser rewrite and kept the continuation on the next line."


def test_plaintext_ingest_extracts_narrative_quotes_with_attribution(tmp_path: Path) -> None:
    source = tmp_path / "narrative_quotes.txt"
    source.write_text(
        'Alice said, "We need a better parser." Bob replied, "I can fix the attribution next."',
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert [segment.text for segment in document.segments] == [
        "We need a better parser.",
        "I can fix the attribution next.",
    ]
    assert [segment.explicit_speaker for segment in document.segments] == ["Alice", "Bob"]


def test_plaintext_ingest_extracts_trailing_quote_attribution(tmp_path: Path) -> None:
    source = tmp_path / "trailing_attribution.txt"
    source.write_text('"We need a cleaner signal path," Alice said.', encoding="utf-8")

    document = ingest_text_file(source)

    assert len(document.segments) == 1
    assert document.segments[0].explicit_speaker == "Alice"
    assert document.segments[0].text == "We need a cleaner signal path,"


def test_plaintext_ingest_merges_split_quotes_with_same_speaker(tmp_path: Path) -> None:
    source = tmp_path / "split_quotes.txt"
    source.write_text('"I know," Alice said, "and I already repaired the anchors."', encoding="utf-8")

    document = ingest_text_file(source)

    assert len(document.segments) == 1
    assert document.segments[0].explicit_speaker == "Alice"
    assert document.segments[0].text == "I know, and I already repaired the anchors."


def test_plaintext_ingest_extracts_unattributed_quote_turns(tmp_path: Path) -> None:
    source = tmp_path / "quote_turns.txt"
    source.write_text('"Hello there." Bob frowned. "Hi."', encoding="utf-8")

    document = ingest_text_file(source)

    assert [segment.text for segment in document.segments] == ["Hello there.", "Hi."]
    assert [segment.explicit_speaker for segment in document.segments] == [None, None]


def test_plaintext_ingest_extracts_multiline_open_quote_with_attribution(tmp_path: Path) -> None:
    source = tmp_path / "open_quote.txt"
    source.write_text(
        'Alice said, "We need a cleaner parser.\n'
        'The current transcript is a mess."\n'
        'Bob replied, "I can take the attribution layer."',
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert len(document.segments) == 2
    assert document.segments[0].explicit_speaker == "Alice"
    assert document.segments[0].text == "We need a cleaner parser. The current transcript is a mess."
    assert document.segments[0].source_line == 1
    assert document.segments[1].explicit_speaker == "Bob"
    assert document.segments[1].source_line == 3


def test_plaintext_ingest_extracts_dialogue_dash_turns(tmp_path: Path) -> None:
    source = tmp_path / "dialogue_dashes.txt"
    source.write_text(
        "— We need to move now.\n"
        "— I know. I'm already on it.\n"
        "The room fell silent.\n"
        "— Then go.\n",
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert [segment.text for segment in document.segments] == [
        "We need to move now.",
        "I know. I'm already on it. The room fell silent.",
        "Then go.",
    ]


def test_plaintext_ingest_parses_narrative_intro_dialogue(tmp_path: Path) -> None:
    source = tmp_path / "narrative_intro.txt"
    source.write_text(
        "Alice said: We need a cleaner parser.\n"
        "Bob replied: I'll take the attribution layer.\n",
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert [segment.explicit_speaker for segment in document.segments] == ["Alice", "Bob"]
    assert [segment.text for segment in document.segments] == [
        "We need a cleaner parser.",
        "I'll take the attribution layer.",
    ]


def test_plaintext_ingest_cleans_stage_suffixes_from_labels(tmp_path: Path) -> None:
    source = tmp_path / "stage_suffixes.txt"
    source.write_text(
        "ALICE (O.S.)\n"
        "We need to move now.\n\n"
        "Bob (whispering): Keep your voice down.\n",
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert [segment.explicit_speaker for segment in document.segments] == ["ALICE", "Bob"]
