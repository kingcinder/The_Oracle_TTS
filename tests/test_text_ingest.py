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


def test_prose_colon_labels_are_not_segmented_as_speakers(tmp_path: Path) -> None:
    """A prose file with 'Note:'/'Warning:' lines must not spawn phantom
    dialogue turns -- those lines stay as ordinary narration segments."""
    source = tmp_path / "prose.txt"
    source.write_text(
        "Note: this is a long explanatory paragraph that goes on for a while "
        "and certainly has more than forty words in it to push it past the "
        "chat-line threshold so it is treated as prose narration instead of "
        "dialogue, which is exactly what we want to verify here.\n\n"
        "Warning: another paragraph that is also deliberately long and wordy "
        "so the file clearly reads as prose rather than a script or chat log.\n",
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert len(document.segments) >= 2
    assert all(segment.explicit_speaker is None for segment in document.segments)


def test_screenplay_format_attributes_allcaps_names(tmp_path: Path) -> None:
    """Screenplay style (an ALL-CAPS name line followed by its dialogue) is
    parsed into explicit speaker segments."""
    source = tmp_path / "script.txt"
    source.write_text(
        "ALICE\n"
        "Hello there.\n"
        "BOB\n"
        "Hi back.\n"
        "ALICE\n"
        "Let's begin.\n",
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert [segment.explicit_speaker for segment in document.segments] == ["ALICE", "BOB", "ALICE"]
    assert [segment.text for segment in document.segments] == ["Hello there.", "Hi back.", "Let's begin."]


def test_quoted_speech_is_attributed_to_the_named_speaker(tmp_path: Path) -> None:
    source = tmp_path / "quotes.txt"
    source.write_text(
        "\"Hello there,\" said Alice.\n"
        "Bob replied, \"And hello to you.\"\n",
        encoding="utf-8",
    )

    document = ingest_text_file(source)

    assert document.segments[0].explicit_speaker == "Alice"
    assert document.segments[0].text == "Hello there,"
    assert document.segments[1].explicit_speaker == "Bob"
    assert document.segments[1].text == "And hello to you."
