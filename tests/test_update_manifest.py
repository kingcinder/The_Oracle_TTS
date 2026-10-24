from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "oracle_check_update_manifest", REPO_ROOT / "scripts" / "check_update_manifest.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verify_manifest_artifact_and_report_update(tmp_path: Path) -> None:
    checker = _load()
    artifact = tmp_path / "the-oracle-1.5.0.AppImage"
    artifact.write_bytes(b"verified artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "update-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": 1,
                "version": "1.5.0",
                "artifacts": [{"name": artifact.name, "sha256": digest, "size": artifact.stat().st_size}],
                "rollback": "Install the previous verified artifact.",
            }
        ),
        encoding="utf-8",
    )

    payload = checker.load_manifest(manifest)
    assert checker.verify_artifact(payload, tmp_path, artifact.name) == artifact
    assert checker.update_available("1.4.2", payload) is True
    assert checker.update_available("1.5.0", payload) is False


def test_verify_manifest_rejects_changed_artifact(tmp_path: Path) -> None:
    checker = _load()
    artifact = tmp_path / "the-oracle.AppImage"
    artifact.write_bytes(b"changed")
    payload = {
        "format": 1,
        "version": "1.5.0",
        "artifacts": [{"name": artifact.name, "sha256": "0" * 64, "size": 7}],
    }
    with pytest.raises(checker.ManifestError, match="SHA-256"):
        checker.verify_artifact(payload, tmp_path, artifact.name)

