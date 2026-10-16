from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path

from scripts.patch_chatterbox_metadata import _replace_metadata


def test_chatterbox_metadata_matches_the_audited_oracle_stack(tmp_path: Path) -> None:
    dist_info = tmp_path / "chatterbox_tts-0.1.7.dist-info"
    dist_info.mkdir()
    metadata = dist_info / "METADATA"
    metadata.write_text(
        "Metadata-Version: 2.1\nName: chatterbox-tts\nVersion: 0.1.7\n"
        "Requires-Dist: transformers==5.2.0\nRequires-Dist: diffusers==0.29.0\n"
        "Requires-Dist: safetensors==0.5.3\nRequires-Dist: gradio==6.8.0\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text(
        f"{dist_info.name}/METADATA,,\n{dist_info.name}/RECORD,,\n",
        encoding="utf-8",
    )

    _replace_metadata(dist_info)
    _replace_metadata(dist_info)

    content = metadata.read_bytes()
    assert b"Version: 0.1.7+oracle1" in content
    assert b"Requires-Dist: transformers==5.20.0" in content
    assert b"Requires-Dist: diffusers==0.38.0" in content
    assert b"Requires-Dist: safetensors==0.8.0" in content
    assert b"Requires-Dist: gradio==6.16.0" in content
    rows = list(csv.reader(io.StringIO((dist_info / "RECORD").read_text(encoding="utf-8"))))
    expected = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode("ascii").rstrip("=")
    assert rows[0][1] == f"sha256={expected}"
    assert rows[0][2] == str(len(content))
