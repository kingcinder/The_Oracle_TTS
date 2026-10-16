"""Align Chatterbox's installed requirements with the audited Oracle stack."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
from pathlib import Path
from importlib.metadata import distribution


_REQUIREMENT_UPDATES = {
    "transformers==5.2.0": "transformers==5.20.0",
    "diffusers==0.29.0": "diffusers==0.38.0",
    "safetensors==0.5.3": "safetensors==0.8.0",
    "gradio==6.8.0": "gradio==6.16.0",
}


def _replace_metadata(path: Path) -> None:
    metadata_path = path / "METADATA"
    text = metadata_path.read_text(encoding="utf-8")
    if "Version: 0.1.7\n" in text:
        text = text.replace("Version: 0.1.7\n", "Version: 0.1.7+oracle1\n", 1)
    elif "Version: 0.1.7+oracle1\n" not in text:
        raise ValueError(f"unexpected Chatterbox version metadata in {metadata_path}")

    for old, new in _REQUIREMENT_UPDATES.items():
        old_line = f"Requires-Dist: {old}\n"
        new_line = f"Requires-Dist: {new}\n"
        if old_line in text:
            text = text.replace(old_line, new_line, 1)
        elif new_line not in text:
            raise ValueError(f"missing expected dependency {old!r} in {metadata_path}")

    _atomic_write(metadata_path, text.encode("utf-8"))
    _update_record(path, metadata_path)


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _update_record(dist_info: Path, metadata_path: Path) -> None:
    record_path = dist_info / "RECORD"
    rows = list(csv.reader(io.StringIO(record_path.read_text(encoding="utf-8"))))
    relative_metadata = metadata_path.relative_to(dist_info.parent).as_posix()
    digest = base64.urlsafe_b64encode(hashlib.sha256(metadata_path.read_bytes()).digest()).decode("ascii").rstrip("=")
    for row in rows:
        if row and row[0] == relative_metadata:
            row[1] = f"sha256={digest}"
            row[2] = str(metadata_path.stat().st_size)
        elif row and row[0] == record_path.relative_to(dist_info.parent).as_posix():
            row[1:] = ["", ""]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    _atomic_write(record_path, output.getvalue().encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    installed = distribution("chatterbox-tts")
    _replace_metadata(Path(installed._path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
