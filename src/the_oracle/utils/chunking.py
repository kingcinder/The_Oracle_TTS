"""Surgical pre-synthesis chunking for overlong utterances.

This module provides deterministic chunking of long text before synthesis
to reduce truncation risk from Chatterbox. Chunking prefers clean linguistic
boundaries (sentence first, then clause) and preserves spoken order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Conservative chunk size threshold in characters.
# Chatterbox has shown truncation issues with utterances > ~250 chars.
# We use 250 to provide headroom while avoiding excessive fragmentation.
MAX_CHUNK_SIZE = 250

# Minimum size to consider chunking. Utterances at or below this remain unsplit.
MIN_CHUNK_SIZE = 200

# Sentence boundary regex - splits on .!? followed by whitespace or end
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Clause boundary punctuation - used as fallback splitting points
_CLAUSE_DELIMS = re.compile(r"([;,:\u2014\u2013-]+)")

# Dash variants for clause splitting
_EM_DASH = "\u2014"
_EN_DASH = "\u2013"


@dataclass(slots=True)
class TextChunk:
    """A chunk of text derived from a parent utterance.
    
    Attributes:
        text: The chunk text content.
        parent_index: Original utterance index this chunk came from.
        chunk_sequence: Zero-based sequence number within the parent.
        is_single_chunk: True if this is the only chunk (parent was short enough).
    """
    text: str
    parent_index: int
    chunk_sequence: int
    is_single_chunk: bool = True


def chunk_utterance(text: str, parent_index: int, max_size: int = MAX_CHUNK_SIZE, min_size: int = MIN_CHUNK_SIZE) -> list[TextChunk]:
    """Split overlong utterances into smaller chunks before synthesis.
    
    Args:
        text: The utterance text to potentially chunk.
        parent_index: The original utterance index for tracking.
        max_size: Maximum chunk size in characters before forcing split.
        min_size: Minimum size - utterances at or below this stay unsplit.
    
    Returns:
        List of TextChunk objects. Single chunk if text <= min_size.
        Multiple chunks preserving original order if text > max_size.
    """
    text = text.strip()
    if not text:
        return []
    
    # Short utterances pass through unchanged
    if len(text) <= min_size:
        return [TextChunk(text=text, parent_index=parent_index, chunk_sequence=0, is_single_chunk=True)]
    
    # Try sentence splitting first
    sentences = _split_sentences(text)
    
    # If we got multiple sentences and they're all manageable, use sentence splitting
    if len(sentences) > 1 and all(len(s) <= max_size for s in sentences):
        chunks = []
        for i, sentence in enumerate(sentences):
            chunks.append(TextChunk(
                text=sentence,
                parent_index=parent_index,
                chunk_sequence=i,
                is_single_chunk=False
            ))
        return chunks
    
    # Either single sentence or some sentences are too long.
    # Fall back to clause-based splitting with size awareness.
    chunks = _split_by_size_with_boundaries(text, max_size)
    
    # Verify chunks are actually smaller than max_size
    # If not, force hard splitting
    if len(chunks) == 1 and len(chunks[0]) > max_size:
        chunks = _hard_split(text, max_size)
    elif not chunks:
        # Edge case: splitting returned empty, use hard split
        chunks = _hard_split(text, max_size) if len(text) > max_size else [text]
    
    result = []
    for i, chunk_text in enumerate(chunks):
        result.append(TextChunk(
            text=chunk_text,
            parent_index=parent_index,
            chunk_sequence=i,
            is_single_chunk=len(chunks) == 1
        ))
    
    return result


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries (.!? followed by space).
    
    Returns list of non-empty sentence strings.
    """
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_size_with_boundaries(text: str, max_size: int) -> list[str]:
    """Split overlong text respecting linguistic boundaries.
    
    Strategy:
    1. Try clause splitting on ;,:- first
    2. If clauses are still too long, split on word boundaries
    3. As last resort, hard split at max_size
    
    All splits preserve order and avoid text loss.
    """
    # First try clause boundaries
    clauses = _split_clauses(text)
    
    # If clause splitting didn't help, go straight to word splitting
    if len(clauses) == 1 and len(clauses[0]) > max_size:
        return _split_by_words(text, max_size)
    
    # Group clauses into chunks that fit max_size
    grouped = _group_clauses(clauses, max_size)
    
    # If grouping produced valid chunks, return them
    if grouped and all(len(g) <= max_size for g in grouped):
        return grouped
    
    # Fallback: word-boundary splitting for very long clauses
    return _split_by_words(text, max_size)


def _split_clauses(text: str) -> list[str]:
    """Split text on clause boundaries (;,:- and em/en dashes).
    
    Preserves the delimiters with the preceding clause for natural reading.
    """
    # Split on clause delimiters but keep them
    parts = _CLAUSE_DELIMS.split(text)
    
    if len(parts) <= 1:
        return [text] if text.strip() else []
    
    # Recombine delimiter with preceding clause
    result = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and _CLAUSE_DELIMS.fullmatch(parts[i + 1].strip()):
            # Delimiter follows - attach to current clause
            combined = parts[i] + parts[i + 1]
            if combined.strip():
                result.append(combined.strip())
            i += 2
        else:
            if parts[i].strip():
                result.append(parts[i].strip())
            i += 1
    
    return result if result else [text]


def _group_clauses(clauses: list[str], max_size: int) -> list[str]:
    """Group clauses into chunks that respect max_size.

    Combines adjacent clauses until adding another would exceed max_size.
    For long single sentences, keeps clauses separate to ensure chunking occurs.
    """
    if not clauses:
        return []

    # If total length is under max_size but we have multiple clauses,
    # still split them to avoid passing long single sentences to synthesis.
    # This prevents truncation when a sentence has many clauses that together
    # fit under max_size but individually are reasonable chunks.
    total_len = sum(len(c) for c in clauses)
    if total_len <= max_size and len(clauses) > 1:
        # Check if any individual clause is long enough to warrant splitting
        longest_clause = max(len(c) for c in clauses)
        if longest_clause > max_size * 0.4:  # If any clause is >40% of max, split
            return clauses

    chunks = []
    current = clauses[0]

    for clause in clauses[1:]:
        # Try adding this clause to current chunk
        combined = current + " " + clause
        if len(combined) <= max_size:
            current = combined
        else:
            # Current is full, start new chunk
            chunks.append(current)
            current = clause

    # Don't forget the last chunk
    if current:
        chunks.append(current)

    return chunks


def _split_by_words(text: str, max_size: int) -> list[str]:
    """Split text on word boundaries when clause splitting fails.
    
    This is a fallback for extremely long clauses without natural breaks.
    If no spaces exist, performs hard character-based splitting.
    """
    words = text.split()
    if not words:
        return [text] if text.strip() else []
    
    # If there's only one "word" (no spaces), do hard character split
    if len(words) == 1 and len(words[0]) > max_size:
        return _hard_split(text, max_size)
    
    chunks = []
    current = words[0]
    
    for word in words[1:]:
        combined = current + " " + word
        if len(combined) <= max_size:
            current = combined
        else:
            chunks.append(current)
            current = word
    
    if current:
        chunks.append(current)
    
    return chunks


def _hard_split(text: str, max_size: int) -> list[str]:
    """Hard split text at exact character boundaries.
    
    Last resort for continuous text without natural break points.
    Preserves order and all content.
    """
    if len(text) <= max_size:
        return [text]
    
    result = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        result.append(text[start:end])
        start = end
    
    return result


def reassemble_chunks(chunks: list[TextChunk]) -> str:
    """Reassemble chunks back into original text.
    
    Used for verification that no text was lost during chunking.
    """
    if not chunks:
        return ""
    
    # Group by parent and sort by sequence
    by_parent: dict[int, list[TextChunk]] = {}
    for chunk in chunks:
        if chunk.parent_index not in by_parent:
            by_parent[chunk.parent_index] = []
        by_parent[chunk.parent_index].append(chunk)
    
    # Reassemble in order
    result = []
    for parent_idx in sorted(by_parent.keys()):
        parent_chunks = sorted(by_parent[parent_idx], key=lambda c: c.chunk_sequence)
        reassembled = " ".join(c.text for c in parent_chunks)
        result.append(reassembled)
    
    return " ".join(result)


def verify_chunking(original: str, chunks: list[TextChunk]) -> bool:
    """Verify that chunking preserved all text content.
    
    Returns True if reassembled chunks match original (ignoring whitespace).
    """
    if not chunks:
        return original.strip() == ""
    
    reassembled = reassemble_chunks(chunks)
    return _normalize_whitespace(original) == _normalize_whitespace(reassembled)


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for comparison."""
    return " ".join(text.split())
