from the_oracle.models.project import Utterance
from the_oracle.pipeline import compute_incremental_changes
from the_oracle.utils.hashing import build_chunk_hash


def test_chunk_hash_changes_when_text_changes() -> None:
    first = build_chunk_hash(
        speaker="A",
        repaired_text="Hello there.",
        engine_key="chatterbox:standard",
        engine_params={"speed": 1.0},
        engine_version="1.0",
        reference_audio_hash="abc",
    )
    second = build_chunk_hash(
        speaker="A",
        repaired_text="Hello again.",
        engine_key="chatterbox:standard",
        engine_params={"speed": 1.0},
        engine_version="1.0",
        reference_audio_hash="abc",
    )

    assert first != second


def test_incremental_changes_only_returns_modified_indices() -> None:
    previous_plan = {
        "utterances": [
            {"index": 0, "chunk_hash": "same"},
            {"index": 1, "chunk_hash": "old"},
        ]
    }
    current = [
        Utterance(index=0, original_text="a", repaired_text="a", chunk_hash="same"),
        Utterance(index=1, original_text="b", repaired_text="b", chunk_hash="new"),
    ]

    assert compute_incremental_changes(previous_plan, current) == [1]
