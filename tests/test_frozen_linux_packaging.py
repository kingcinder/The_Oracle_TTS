from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = REPO_ROOT / "scripts" / "build_frozen_linux.py"
    spec = importlib.util.spec_from_file_location("oracle_frozen_linux_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_spec_has_no_model_or_secret_inputs() -> None:
    spec = (REPO_ROOT / "packaging" / "linux" / "the-oracle.spec").read_text(encoding="utf-8")
    assert "MODEL_PATH" not in spec
    assert "hf_cache" not in spec
    assert "HF_HOME" not in spec
    assert "license-signing-seed" not in spec
    assert "packaging" in spec and "entrypoint.py" in spec


def test_update_manifest_and_checksums_are_deterministic(tmp_path: Path) -> None:
    builder = _load_builder()
    artifact = tmp_path / "the-oracle.AppImage"
    artifact.write_bytes(b"rehearsal")

    checksums = builder.write_checksums(tmp_path, [artifact], "1.5.0")
    assert checksums.name == "checksums-1.5.0.sha256"
    digest = hashlib.sha256(b"rehearsal").hexdigest()
    assert f"{digest}  {artifact.name}" in checksums.read_text(encoding="utf-8")

    manifest = builder.write_update_manifest(tmp_path, [artifact], "1.5.0")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload == {
        "format": 1,
        "version": "1.5.0",
        "artifacts": [{"name": artifact.name, "sha256": digest, "size": len(b"rehearsal")}],
        "rollback": "Install the previous verified artifact; no automatic replacement is performed.",
    }


def test_stage_appdir_uses_replaceable_runtime_and_keeps_user_data(tmp_path: Path) -> None:
    builder = _load_builder()
    frozen = tmp_path / "frozen"
    (frozen / "the-oracle").mkdir(parents=True)
    (frozen / "the-oracle" / "the-oracle").write_bytes(b"binary")
    appdir = builder.stage_appdir(tmp_path / "AppDir", frozen)

    assert (appdir / "AppRun").is_file()
    assert (appdir / "usr" / "bin" / "the-oracle").is_file()
    desktop = (appdir / "usr" / "share" / "applications" / "the-oracle.desktop").read_text(
        encoding="utf-8"
    )
    assert "Exec=the-oracle" in desktop
    assert "models" not in desktop.lower()
    assert "~/.config/the-oracle" in (appdir / "AppRun").read_text(encoding="utf-8")


def test_build_refuses_non_linux_or_non_x86_64(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    builder = _load_builder()
    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")
    with pytest.raises(builder.BuildError, match="Linux x86_64"):
        builder.ensure_host()
