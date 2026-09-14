"""Tests for pinned model revisions and the offline install bundle.

Covers:
  * the_oracle.models.pins: every pin is a real 40-char hex commit SHA.
  * scripts/download_models.py: helper-model revisions match the central pins.
  * the_oracle.tts_engines.chatterbox_engine: turbo download uses the pinned revision.
  * scripts/manage_install.py: --offline-bundle plumbing — pip gets
    --no-index/--find-links, models seed into the HF cache with refs/main
    pointed at the pins, the managed wrappers export HF_HUB_OFFLINE=1 on
    offline installs.
  * scripts/build_offline_bundle.py: launcher scripts reference the offline flag.

No test downloads models, touches the network, or spends anything.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def manage_install():
    return _load_script_module("oracle_manage_install_offline", "manage_install.py")


@pytest.fixture()
def bundle_builder():
    return _load_script_module("oracle_build_offline_bundle", "build_offline_bundle.py")


@pytest.fixture()
def download_models(monkeypatch):
    """Load scripts/download_models.py with the heavy chatterbox engine stubbed."""
    tts_pkg = types.ModuleType("the_oracle.tts_engines")
    tts_pkg.__path__ = []  # type: ignore[attr-defined]
    engine_mod = types.ModuleType("the_oracle.tts_engines.chatterbox_engine")

    class FakeChatterboxEngine:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_model_ready(self):
            pass

    engine_mod.ChatterboxEngine = FakeChatterboxEngine
    monkeypatch.setitem(sys.modules, "the_oracle.tts_engines", tts_pkg)
    monkeypatch.setitem(sys.modules, "the_oracle.tts_engines.chatterbox_engine", engine_mod)
    return _load_script_module("oracle_download_models_offline", "download_models.py")


# --- central pins ------------------------------------------------------------


def test_model_pins_are_real_commit_shas() -> None:
    from the_oracle.models.pins import MODEL_PINS

    assert len(MODEL_PINS) >= 4
    for repo_id, sha in MODEL_PINS.items():
        assert "/" in repo_id, f"not a repo id: {repo_id}"
        assert SHA_RE.match(sha), f"{repo_id}: pin must be a 40-char hex SHA, got {sha!r}"


def test_download_models_revisions_match_central_pins(download_models) -> None:
    from the_oracle.models.pins import MODEL_PINS

    for name, repo_id in download_models.HF_MODELS.items():
        assert download_models.HF_MODEL_REVISIONS[name] == MODEL_PINS[repo_id]


def _stub_chatterbox_engine_imports(monkeypatch) -> None:
    """Stub the heavy third-party / sibling modules chatterbox_engine imports."""
    import unittest.mock as mock

    for name in ("numpy", "huggingface_hub", "huggingface_hub.errors", "huggingface_hub.utils"):
        monkeypatch.setitem(sys.modules, name, mock.MagicMock())
    cache_mod = types.ModuleType("the_oracle.models.cache")
    cache_mod.CachedReference = object
    cache_mod.ProjectCache = object
    project_mod = types.ModuleType("the_oracle.models.project")
    project_mod.VoiceSettings = object
    project_mod.strip_pain_point_markers = lambda s: s
    platform_mod = types.ModuleType("the_oracle.platform_support")
    platform_mod.repo_python_display = lambda: "python"
    hashing_mod = types.ModuleType("the_oracle.utils.hashing")
    hashing_mod.hash_payload = lambda *a, **k: "hash"
    for mod in (cache_mod, project_mod, platform_mod, hashing_mod):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)


def test_turbo_revision_is_pinned(monkeypatch) -> None:
    _stub_chatterbox_engine_imports(monkeypatch)
    from the_oracle.models.pins import MODEL_PINS, TURBO_REPO_ID

    spec = importlib.util.spec_from_file_location(
        "the_oracle.tts_engines.chatterbox_engine",
        REPO_ROOT / "src" / "the_oracle" / "tts_engines" / "chatterbox_engine.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.TURBO_REPO_ID == TURBO_REPO_ID
    assert module.TURBO_REVISION == MODEL_PINS[TURBO_REPO_ID]
    assert SHA_RE.match(module.TURBO_REVISION)


# --- offline pip args --------------------------------------------------------


def test_pip_base_args_online_is_empty(manage_install) -> None:
    assert manage_install._pip_base_args(None) == []


def test_pip_base_args_offline_uses_no_index_find_links(manage_install, tmp_path: Path) -> None:
    wheels = tmp_path / "bundle" / "wheels" / "linux"
    wheels.mkdir(parents=True)
    args = manage_install._pip_base_args(tmp_path / "bundle")
    assert args[:2] == ["--no-index", "--find-links"]
    assert args[2] == str(wheels)


def test_offline_wheels_dir_missing_platform_fails(manage_install, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "wheels").mkdir(parents=True)
    with pytest.raises(SystemExit):
        manage_install.offline_wheels_dir(bundle)


def test_resolve_offline_bundle_requires_manifest(manage_install, tmp_path: Path) -> None:
    assert manage_install._resolve_offline_bundle(None) is None
    with pytest.raises(SystemExit):
        manage_install._resolve_offline_bundle(str(tmp_path))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    assert manage_install._resolve_offline_bundle(str(bundle)) == bundle.resolve()


# --- model seeding -----------------------------------------------------------


def _make_fake_bundle_cache(bundle: Path, pins: dict[str, str]) -> None:
    hf_cache = bundle / "hf_cache"
    for repo_id, sha in pins.items():
        model_dir = hf_cache / ("models--" + repo_id.replace("/", "--"))
        snap = model_dir / "snapshots" / sha
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")
        refs = model_dir / "refs"
        refs.mkdir(exist_ok=True)
        (refs / sha).write_text(sha + "\n", encoding="utf-8")


def test_seed_offline_models_points_refs_main_at_pins(
    manage_install, tmp_path: Path, monkeypatch
) -> None:
    from the_oracle.models.pins import MODEL_PINS

    bundle = tmp_path / "bundle"
    _make_fake_bundle_cache(bundle, MODEL_PINS)
    fake_cache = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(fake_cache))

    manage_install.seed_offline_models(bundle)

    for repo_id, sha in MODEL_PINS.items():
        model_dir = fake_cache / ("models--" + repo_id.replace("/", "--"))
        assert (model_dir / "snapshots" / sha / "config.json").is_file()
        assert (model_dir / "refs" / "main").read_text(encoding="utf-8").strip() == sha


def test_seed_offline_models_missing_model_fails(manage_install, tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "hf_cache").mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    with pytest.raises(SystemExit):
        manage_install.seed_offline_models(bundle)


def test_write_offline_marker(manage_install, tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"app": "the-oracle"}), encoding="utf-8")
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    monkeypatch.setattr(manage_install, "REPO_ROOT", fake_root)

    manage_install.write_offline_marker(bundle)

    marker = fake_root / manage_install.OFFLINE_MARKER_FILENAME
    assert marker.is_file()
    assert "the-oracle" in marker.read_text(encoding="utf-8")


# --- wrappers export HF_HUB_OFFLINE ------------------------------------------


def _wrapper_contents(manage_install, monkeypatch, tmp_path: Path, *, windows: bool) -> str:
    monkeypatch.setattr(manage_install, "is_windows", lambda: windows)
    monkeypatch.setattr(manage_install, "REPO_ROOT", tmp_path)
    return manage_install.managed_wrapper_contents()


def test_bash_wrapper_exports_hf_hub_offline_on_offline_install(
    manage_install, tmp_path: Path, monkeypatch
) -> None:
    body = _wrapper_contents(manage_install, monkeypatch, tmp_path, windows=False)
    assert "HF_HUB_OFFLINE=1" in body
    assert manage_install.OFFLINE_MARKER_FILENAME in body


def test_windows_wrapper_exports_hf_hub_offline_on_offline_install(
    manage_install, tmp_path: Path, monkeypatch
) -> None:
    body = _wrapper_contents(manage_install, monkeypatch, tmp_path, windows=True)
    assert "HF_HUB_OFFLINE=1" in body
    assert manage_install.OFFLINE_MARKER_FILENAME in body


# --- bundle launchers --------------------------------------------------------


def test_bundle_launchers_invoke_offline_install(bundle_builder, tmp_path: Path) -> None:
    bundle_builder.write_launchers(tmp_path)
    sh = (tmp_path / "install.sh").read_text(encoding="utf-8")
    bat = (tmp_path / "install.bat").read_text(encoding="utf-8")
    assert "--offline-bundle" in sh
    assert "--offline-bundle" in bat
    assert "repo.tar.gz" in sh and "repo.tar.gz" in bat
