from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.distribution_locks import build_lock


def test_build_lock_writes_require_hashes_and_manifest(tmp_path: Path) -> None:
    wheels = tmp_path / "wheels" / "linux"
    wheels.mkdir(parents=True)
    first = wheels / "demo_pkg-1.2.3-py3-none-any.whl"
    second = wheels / "other_pkg-2.0.0-py3-none-any.whl"
    first.write_bytes(b"first wheel")
    second.write_bytes(b"second wheel")
    lock = tmp_path / "locks" / "linux-py3.12.txt"
    manifest = tmp_path / "manifest.json"

    build_lock(wheels, lock, manifest, platform="linux", python_version="3.12")

    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    text = lock.read_text(encoding="utf-8")
    assert "--require-hashes" in text
    assert "--no-index" in text
    assert f"demo-pkg==1.2.3 --hash=sha256:{digest}" in text
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["python_version"] == "3.12"
    assert payload["platforms"]["linux"]["wheels"][0]["sha256"] == digest

