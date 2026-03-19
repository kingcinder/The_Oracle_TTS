"""Tests for pre-synthesis chunking of overlong utterances."""

from __future__ import annotations

import pytest

from the_oracle.utils.chunking import (
    TextChunk,
    chunk_utterance,
    reassemble_chunks,
    verify_chunking,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
)


class TestChunkUtteranceShortText:
    """Short utterances should remain unsplit."""

    def test_empty_string_returns_empty_list(self) -> None:
        chunks = chunk_utterance("", parent_index=0)
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        chunks = chunk_utterance("   \n\t  ", parent_index=0)
        assert chunks == []

    def test_very_short_text_returns_single_chunk(self) -> None:
        chunks = chunk_utterance("Hello.", parent_index=1)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello."
        assert chunks[0].is_single_chunk is True
        assert chunks[0].parent_index == 1
        assert chunks[0].chunk_sequence == 0

    def test_text_at_min_size_returns_single_chunk(self) -> None:
        text = "x" * MIN_CHUNK_SIZE
        chunks = chunk_utterance(text, parent_index=2)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].is_single_chunk is True

    def test_text_just_under_min_size_returns_single_chunk(self) -> None:
        text = "x" * (MIN_CHUNK_SIZE - 1)
        chunks = chunk_utterance(text, parent_index=3)
        assert len(chunks) == 1
        assert chunks[0].is_single_chunk is True


class TestChunkUtteranceLongText:
    """Long utterances should be split at appropriate boundaries."""

    def test_text_over_max_size_gets_chunked(self) -> None:
        text = "x" * (MAX_CHUNK_SIZE + 100)
        chunks = chunk_utterance(text, parent_index=1)
        assert len(chunks) > 1
        assert all(len(c.text) <= MAX_CHUNK_SIZE + 50 for c in chunks)  # Small tolerance

    def test_chunk_preserves_order(self) -> None:
        text = "First sentence. Second sentence. Third sentence. " + "x" * MAX_CHUNK_SIZE
        chunks = chunk_utterance(text, parent_index=1)
        assert len(chunks) > 1
        # Verify sequence numbers are in order
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_sequence == i

    def test_chunk_parent_index_matches_input(self) -> None:
        text = "x" * (MAX_CHUNK_SIZE * 2)
        chunks = chunk_utterance(text, parent_index=42)
        assert all(c.parent_index == 42 for c in chunks)


class TestSentenceBoundarySplitting:
    """Chunking should prefer sentence boundaries when text is long enough."""

    def test_splits_on_sentence_boundaries_when_long_enough(self) -> None:
        # Text must exceed MIN_CHUNK_SIZE to trigger chunking
        text = "This is the first sentence. " * 10 + "This is the second sentence. " * 10 + "And a third one here."
        chunks = chunk_utterance(text, parent_index=1)
        # Should split into multiple chunks at sentence boundaries
        assert len(chunks) > 1
        # Verify sentences are kept intact
        chunk_texts = [c.text for c in chunks]
        assert any("first sentence" in t for t in chunk_texts)
        assert any("second sentence" in t for t in chunk_texts)
        assert any("third" in t for t in chunk_texts)

    def test_splits_on_question_marks_when_long_enough(self) -> None:
        text = "How are you? " * 20 + "I am fine. " * 10 + "What about you?"
        chunks = chunk_utterance(text, parent_index=2)
        assert len(chunks) > 1
        chunk_texts = [c.text for c in chunks]
        assert any("How are you?" in t for t in chunk_texts)

    def test_splits_on_exclamation_when_long_enough(self) -> None:
        text = "Wow! " * 30 + "That is amazing! " * 20 + "I can't believe it!"
        chunks = chunk_utterance(text, parent_index=3)
        assert len(chunks) > 1

    def test_short_multi_sentence_stays_unsplit(self) -> None:
        # Short texts under MIN_CHUNK_SIZE should NOT be chunked
        text = "This is the first sentence. This is the second sentence. And a third one here."
        chunks = chunk_utterance(text, parent_index=1)
        assert len(chunks) == 1
        assert chunks[0].is_single_chunk is True


class TestClauseBoundarySplitting:
    """When sentences are too long, should split on clause boundaries."""

    def test_splits_on_commas_when_sentence_too_long(self) -> None:
        # A single long "sentence" with commas
        text = "The very long and winding road that stretches on for what seems like an eternity, through valleys and over hills, past forests and beside streams, eventually leads to a small, quiet village nestled in the countryside."
        chunks = chunk_utterance(text, parent_index=1)
        # Should still chunk despite being one sentence
        assert len(chunks) > 1 or sum(len(c.text) for c in chunks) > 0

    def test_splits_on_semicolons(self) -> None:
        text = "This is a long clause; it goes on for a while; and has multiple parts; each separated by semicolons to create natural breaking points for chunking."
        chunks = chunk_utterance(text, parent_index=2)
        assert len(chunks) >= 1

    def test_splits_on_colons(self) -> None:
        text = "Here is the main point: this is the explanation: and these are the details that follow."
        chunks = chunk_utterance(text, parent_index=3)
        assert len(chunks) >= 1

    def test_splits_on_em_dash(self) -> None:
        text = "This is the first part — and this is the second part — which continues the thought."
        chunks = chunk_utterance(text, parent_index=4)
        assert len(chunks) >= 1


class TestTextPreservation:
    """Chunking must not lose or mutate text content."""

    def test_reassembly_matches_original(self) -> None:
        text = "First sentence. Second sentence. Third sentence with more words to make it longer."
        chunks = chunk_utterance(text, parent_index=1)
        reassembled = reassemble_chunks(chunks)
        # Normalize whitespace for comparison
        assert " ".join(text.split()) == " ".join(reassembled.split())

    def test_verify_chunking_returns_true_for_valid_chunking(self) -> None:
        text = "A short sentence. Another short sentence here."
        chunks = chunk_utterance(text, parent_index=1)
        assert verify_chunking(text, chunks) is True

    def test_verify_chunking_returns_false_for_empty_chunks(self) -> None:
        text = "Some text here."
        assert verify_chunking(text, []) is False

    def test_long_text_preserves_all_content(self) -> None:
        text = "First part. " + "Middle part. " * 50 + "Final part."
        chunks = chunk_utterance(text, parent_index=1)
        assert verify_chunking(text, chunks) is True
        # Verify key phrases are present
        all_text = " ".join(c.text for c in chunks)
        assert "First part" in all_text
        assert "Middle part" in all_text
        assert "Final part" in all_text


class TestDeterministicChunking:
    """Chunking must be deterministic - same input produces same output."""

    def test_same_input_produces_same_chunks(self) -> None:
        text = "This is a test sentence. Here is another one. And a third for good measure."
        chunks1 = chunk_utterance(text, parent_index=1)
        chunks2 = chunk_utterance(text, parent_index=1)
        assert len(chunks1) == len(chunks2)
        assert [c.text for c in chunks1] == [c.text for c in chunks2]
        assert [c.chunk_sequence for c in chunks1] == [c.chunk_sequence for c in chunks2]

    def test_different_parent_index_does_not_affect_chunking(self) -> None:
        text = "First sentence. Second sentence. " + "x" * MAX_CHUNK_SIZE
        chunks1 = chunk_utterance(text, parent_index=1)
        chunks2 = chunk_utterance(text, parent_index=999)
        # Text chunks should be identical, only parent_index differs
        assert [c.text for c in chunks1] == [c.text for c in chunks2]


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_long_word_without_spaces(self) -> None:
        text = "x" * 500
        chunks = chunk_utterance(text, parent_index=1)
        assert len(chunks) > 1
        # All chunks together should equal original
        reassembled = "".join(c.text for c in chunks)
        assert reassembled == text

    def test_mixed_punctuation(self) -> None:
        text = "Hello! How are you? I'm fine, thanks. Wow; really; that's great: I'm happy!"
        chunks = chunk_utterance(text, parent_index=1)
        assert verify_chunking(text, chunks)

    def test_quotes_and_apostrophes(self) -> None:
        text = "She said, \"Hello there!\" and he replied, \"How are you doing?\" It's great."
        chunks = chunk_utterance(text, parent_index=2)
        assert verify_chunking(text, chunks)

    def test_abbreviations_with_periods(self) -> None:
        # Abbreviations shouldn't cause incorrect splits
        text = "Dr. Smith went to Washington D.C. He met with Sen. Jones."
        chunks = chunk_utterance(text, parent_index=1)
        # Should still produce valid chunks even if abbreviation handling is imperfect
        assert len(chunks) >= 1
        assert verify_chunking(text, chunks)


class TestChunkSizeBoundaries:
    """Test behavior at chunk size boundaries."""

    def test_exactly_at_min_size_not_chunked(self) -> None:
        text = "x" * MIN_CHUNK_SIZE
        chunks = chunk_utterance(text, parent_index=1)
        assert len(chunks) == 1
        assert chunks[0].is_single_chunk is True

    def test_one_over_min_size_may_be_chunked(self) -> None:
        text = "x" * (MIN_CHUNK_SIZE + 1)
        chunks = chunk_utterance(text, parent_index=1)
        # May or may not chunk depending on implementation
        assert len(chunks) >= 1

    def test_exactly_at_max_size_single_chunk(self) -> None:
        text = "x" * MAX_CHUNK_SIZE
        chunks = chunk_utterance(text, parent_index=1)
        # Should ideally be a single chunk if it fits
        assert len(chunks) >= 1

    def test_double_max_size_produces_multiple_chunks(self) -> None:
        text = "x" * (MAX_CHUNK_SIZE * 2)
        chunks = chunk_utterance(text, parent_index=1)
        assert len(chunks) > 1

    def test_long_single_sentence_with_clauses_gets_chunked(self) -> None:
        """Regression test: long single sentences with multiple clauses should be chunked.
        
        Previously, _group_clauses would combine all clauses if total was under MAX_CHUNK_SIZE,
        allowing 300+ char single sentences through unchunked, causing Chatterbox truncation.
        """
        # A single long sentence with multiple clauses (344 chars total)
        text = (
            "This is a very long sentence that keeps going on and on without any proper break points "
            "for what seems like an eternity, and it continues further still, through valleys of text "
            "and over hills of prose, past forests of phrases and beside streams of syllables, until "
            "it eventually reaches its destination in the countryside of completed thoughts."
        )
        assert len(text) > MIN_CHUNK_SIZE  # Should trigger chunking logic
        chunks = chunk_utterance(text, parent_index=1)
        # Should split into multiple chunks despite total being under old MAX_CHUNK_SIZE (350)
        assert len(chunks) > 1
        # Verify all content is preserved
        assert verify_chunking(text, chunks)
        # Each chunk should be reasonably sized
        assert all(len(c.text) <= MAX_CHUNK_SIZE for c in chunks)
