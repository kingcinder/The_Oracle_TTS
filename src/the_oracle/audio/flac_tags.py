"""Minimal FLAC Vorbis-comment reader/writer — stdlib only.

The release used to depend on ``mutagen`` solely to write FLAC tags. That is a
GPL-2.0-or-later dependency in a proprietary distribution, and the actual
surface was small: parse the metadata block list, replace or insert the
VORBIS_COMMENT block, and leave the audio frames untouched. This module owns
that surface so the export path carries no third-party tagger.

FLAC layout (https://xiph.org/flac/format.html): ``fLaC`` followed by metadata
blocks. Each block is a one-byte header (high bit = last block, low seven bits
= type) and a 24-bit big-endian length. Type 0 is STREAMINFO; type 4 is a
Vorbis comment. The audio frames start after the last metadata block.

Vorbis comments in FLAC are little-endian: vendor length, vendor string,
comment count, then length-prefixed ``FIELD=value`` UTF-8 strings. Field names
are ASCII 0x20-0x7D excluding ``=``; this writer uppercases them and replaces
anything outside that set with ``_``.

The writer preserves every non-comment block and every existing comment whose
field name is not being updated, so a re-tag is additive rather than
destructive. It writes through a temporary file and replaces atomically: a
tagging failure leaves the original audio untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

_MAGIC = b"fLaC"
_STREAMINFO = 0
_VORBIS_COMMENT = 4
_MAX_BLOCK_TYPE = 126
_VENDOR = "The Oracle"


def _read_blocks(data: bytes) -> tuple[list[tuple[int, bytes]], bytes]:
    """Split a FLAC stream into ``(blocks, audio_frames)``."""
    if not data.startswith(_MAGIC):
        raise ValueError("not a FLAC stream (missing fLaC marker)")
    offset = len(_MAGIC)
    blocks: list[tuple[int, bytes]] = []
    while True:
        if offset + 4 > len(data):
            raise ValueError("truncated FLAC metadata header")
        header = data[offset]
        block_type = header & 0x7F
        if block_type == 127:
            raise ValueError("invalid FLAC metadata block type 127")
        length = int.from_bytes(data[offset + 1 : offset + 4], "big")
        start = offset + 4
        end = start + length
        if end > len(data):
            raise ValueError("truncated FLAC metadata block")
        blocks.append((block_type, data[start:end]))
        offset = end
        if header & 0x80:
            break
    return blocks, data[offset:]


def _read_u32(payload: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(payload):
        raise ValueError("truncated Vorbis comment length")
    return int.from_bytes(payload[offset : offset + 4], "little"), offset + 4


def _parse_vorbis_comments(payload: bytes) -> list[str]:
    offset = 0
    vendor_length, offset = _read_u32(payload, offset)
    offset += vendor_length
    count, offset = _read_u32(payload, offset)
    comments: list[str] = []
    for _ in range(count):
        length, offset = _read_u32(payload, offset)
        raw = payload[offset : offset + length]
        if len(raw) != length:
            raise ValueError("truncated Vorbis comment")
        offset += length
        comments.append(raw.decode("utf-8"))
    return comments


def read_vorbis_comments(path: str | Path) -> dict[str, list[str]]:
    """Return the file's Vorbis comments as ``{UPPERCASE_FIELD: [values]}``.

    A file with no VORBIS_COMMENT block returns an empty mapping. Malformed
    blocks raise ``ValueError`` rather than being silently ignored.
    """
    blocks, _audio = _read_blocks(Path(path).read_bytes())
    comments: dict[str, list[str]] = {}
    for block_type, payload in blocks:
        if block_type != _VORBIS_COMMENT:
            continue
        for comment in _parse_vorbis_comments(payload):
            field, separator, value = comment.partition("=")
            if not separator:
                continue
            comments.setdefault(field.upper(), []).append(value)
        break
    return comments


def _normalize_field(name: str) -> str:
    """A legal Vorbis field name, or ``""`` when nothing usable remains."""
    text = str(name).strip().upper()
    if not text:
        return ""
    return "".join(
        character if 0x20 <= ord(character) <= 0x7D and character != "=" else "_"
        for character in text
    )


def _comment_payload(comments: list[str]) -> bytes:
    vendor = _VENDOR.encode("utf-8")
    parts = [len(vendor).to_bytes(4, "little"), vendor, len(comments).to_bytes(4, "little")]
    for comment in comments:
        raw = comment.encode("utf-8")
        parts.extend((len(raw).to_bytes(4, "little"), raw))
    return b"".join(parts)


def write_vorbis_comments(path: str | Path, metadata: dict[str, str]) -> None:
    """Write ``metadata`` into a FLAC file, preserving unrelated comments.

    Empty values are skipped, matching the old tagger's behavior. The file is
    replaced atomically only after the complete new stream has been built.
    """
    updates: dict[str, str] = {}
    for name, value in metadata.items():
        field = _normalize_field(name)
        text = str(value)
        if field and text:
            updates[field] = text
    if not updates:
        return

    destination = Path(path)
    original = destination.read_bytes()
    blocks, audio = _read_blocks(original)

    existing: list[str] = []
    kept: list[tuple[int, bytes]] = []
    for block_type, payload in blocks:
        if block_type == _VORBIS_COMMENT:
            if not existing:
                existing = _parse_vorbis_comments(payload)
            continue
        kept.append((block_type, payload))

    comments = [
        comment
        for comment in existing
        if _normalize_field(comment.partition("=")[0]) not in updates
    ]
    comments.extend(f"{field}={value}" for field, value in updates.items())

    new_block = (_VORBIS_COMMENT, _comment_payload(comments))
    if kept and kept[0][0] == _STREAMINFO:
        kept.insert(1, new_block)
    else:
        kept.append(new_block)

    output = bytearray(_MAGIC)
    for index, (block_type, payload) in enumerate(kept):
        if block_type > _MAX_BLOCK_TYPE:
            raise ValueError(f"invalid FLAC metadata block type {block_type}")
        last = 0x80 if index == len(kept) - 1 else 0
        output.append(last | block_type)
        output.extend(len(payload).to_bytes(3, "big"))
        output.extend(payload)
    output.extend(audio)

    temporary = destination.with_name(destination.name + ".tags.tmp")
    try:
        temporary.write_bytes(bytes(output))
        os.chmod(temporary, destination.stat().st_mode)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
