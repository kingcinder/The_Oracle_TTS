from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.build_linux_artifacts import _sha256


def test_sha256_matches_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"oracle")
    assert _sha256(path) == hashlib.sha256(b"oracle").hexdigest()

