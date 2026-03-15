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
