from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import is_dataclass, asdict
import importlib
from pathlib import Path
from typing import Any


def import_module(name: str):
    return importlib.import_module(name)


def resolve_callable(module: Any, *names: str):
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    raise AssertionError(f"None of {names!r} exists on module {module.__name__}")


def maybe_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump()
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    return value


def extract_text(value: Any) -> str:
    value = maybe_to_dict(value)

    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return value.read_text(encoding="utf-8")
    if isinstance(value, dict):
        for key in ("text", "content", "raw_text", "clean_text", "markdown_text", "source_text"):
            if key in value and isinstance(value[key], str):
                return value[key]
        if "utterances" in value:
            return "\n".join(extract_text(item) for item in value["utterances"])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(extract_text(item) for item in value)

    for attr in ("text", "content", "raw_text", "clean_text", "markdown_text", "source_text"):
        data = getattr(value, attr, None)
        if isinstance(data, str):
            return data
    utterances = getattr(value, "utterances", None)
    if utterances is not None:
        return "\n".join(extract_text(item) for item in utterances)

    raise AssertionError(f"Could not extract text from {type(value)!r}")


def extract_speakers(items: Any) -> list[str]:
    items = maybe_to_dict(items)
    if isinstance(items, dict) and "utterances" in items:
        items = items["utterances"]
    if not isinstance(items, Iterable) or isinstance(items, (str, bytes, bytearray)):
        raise AssertionError(f"Expected an iterable of utterances, got {type(items)!r}")

    speakers: list[str] = []
    for item in items:
        item = maybe_to_dict(item)
        if isinstance(item, dict):
            for key in ("speaker", "speaker_id", "assigned_speaker", "label"):
                value = item.get(key)
                if isinstance(value, str):
                    speakers.append(value)
                    break
            else:
                raise AssertionError(f"No speaker field found in {item!r}")
            continue

        for attr in ("speaker", "speaker_id", "assigned_speaker", "label"):
            value = getattr(item, attr, None)
            if isinstance(value, str):
                speakers.append(value)
                break
        else:
            raise AssertionError(f"No speaker attribute found in {item!r}")
    return speakers


def normalise_speaker_label(label: str) -> str:
    upper = label.strip().upper()
    if upper.endswith("A"):
        return "A"
    if upper.endswith("B"):
        return "B"
    return upper
