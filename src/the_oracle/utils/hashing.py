"""Stable hashing helpers used by render plans and per-utterance cache keys."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_text(value: str) -> str:
    return hash_bytes(value.encode("utf-8"))


def hash_payload(payload: Any) -> str:
    return hash_text(stable_json_dumps(payload))


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_chunk_hash(
    *,
    speaker: str,
    repaired_text: str,
    engine_key: str,
    engine_params: dict[str, Any],
    engine_version: str,
    reference_audio_hash: str,
) -> str:
    return hash_payload(
        {
            "speaker": speaker,
            "repaired_text": repaired_text,
            "engine_key": engine_key,
            "engine_params": engine_params,
            "engine_version": engine_version,
            "reference_audio_hash": reference_audio_hash,
        }
    )


def render_chunk_hash(
    speaker: str,
    repaired_text: str,
    engine_params: dict[str, Any],
    engine_version: str,
    reference_audio_hash: str,
) -> str:
    return build_chunk_hash(
        speaker=speaker,
        repaired_text=repaired_text,
        engine_key="render",
        engine_params=engine_params,
        engine_version=engine_version,
        reference_audio_hash=reference_audio_hash,
    )


def compute_chunk_hash(
    *,
    speaker: str,
    repaired_text: str,
    engine_params: dict[str, Any],
    engine_version: str,
    reference_audio_hash: str,
    engine_key: str = "render",
) -> str:
    return build_chunk_hash(
        speaker=speaker,
        repaired_text=repaired_text,
        engine_key=engine_key,
        engine_params=engine_params,
        engine_version=engine_version,
        reference_audio_hash=reference_audio_hash,
    )


stable_hash_bytes = hash_bytes
stable_hash_text = hash_text
stable_hash_mapping = hash_payload
stable_hash_file = hash_file
hash_render_chunk = compute_chunk_hash
hash_utterance = compute_chunk_hash
compute_render_hash = compute_chunk_hash
