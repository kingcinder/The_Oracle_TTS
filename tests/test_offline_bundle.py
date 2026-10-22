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
import os
import re
import sys
import types
import zipfile
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

    # Load under a UNIQUE module name: exec-ing the engine under the real
    # module's name while numpy/huggingface_hub are stubbed would poison the
    # real engine's module globals (its `import numpy as np` binds the stub)
    # for every later test in the process. The pin constants do not depend on
    # the stubs, so a private alias module is sufficient here.
    spec = importlib.util.spec_from_file_location(
        "oracle_test_chatterbox_engine_pins",
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
    # The product looks for wheels/<platform> matching the *current* OS.
    platform_dir = "windows" if sys.platform == "win32" else "linux"
    wheels = tmp_path / "bundle" / "wheels" / platform_dir
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
        # Byte-exact: huggingface_hub treats the file's contents as the revision,
        # so a trailing newline hides the snapshot from every hub-level loader.
        assert (model_dir / "refs" / "main").read_bytes() == sha.encode("ascii")


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


# --- pinned chatterbox loaders ----------------------------------------------


def _load_chatterbox_engine_module(monkeypatch):
    """Load chatterbox_engine.py with heavy deps stubbed, as the real module."""
    import unittest.mock as mock

    _stub_chatterbox_engine_imports(monkeypatch)
    monkeypatch.setitem(sys.modules, "numpy", mock.MagicMock())
    hf_mod = types.ModuleType("huggingface_hub")
    hf_mod.snapshot_download = mock.MagicMock(return_value="/tmp/fake-ckpt")
    hf_errors = types.ModuleType("huggingface_hub.errors")
    hf_errors.LocalEntryNotFoundError = type("LocalEntryNotFoundError", (Exception,), {})
    hf_utils = types.ModuleType("huggingface_hub.utils")
    hf_utils.LocalTokenNotFoundError = type("LocalTokenNotFoundError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_mod)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", hf_errors)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", hf_utils)
    spec = importlib.util.spec_from_file_location(
        "oracle_test_chatterbox_engine",
        REPO_ROOT / "src" / "the_oracle" / "tts_engines" / "chatterbox_engine.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stub_chatterbox_variant(monkeypatch, dotted: str, cls_name: str, calls: list):
    pkg = types.ModuleType("chatterbox")
    pkg.__path__ = []  # type: ignore[attr-defined]
    mod = types.ModuleType(dotted)

    class FakeTTS:
        @classmethod
        def from_local(cls, ckpt_dir, device):
            calls.append((dotted, str(ckpt_dir), device))
            return f"<{cls_name} model>"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # must NOT be called
            raise AssertionError(f"{cls_name}.from_pretrained must not be used (offline!)")

    mod.__dict__[cls_name] = FakeTTS
    mod.Conditionals = object
    if "mtl" in dotted:
        mod.SUPPORTED_LANGUAGES = {"en": "English"}
    monkeypatch.setitem(sys.modules, "chatterbox", pkg)
    monkeypatch.setitem(sys.modules, dotted, mod)


@pytest.mark.parametrize("variant,dotted,cls_name", [
    ("standard", "chatterbox.tts", "ChatterboxTTS"),
    ("multilingual", "chatterbox.mtl_tts", "ChatterboxMultilingualTTS"),
])
def test_chatterbox_variants_load_from_pinned_local_snapshot(monkeypatch, variant, dotted, cls_name) -> None:
    from the_oracle.models.pins import CHATTERBOX_REPO, MODEL_PINS

    engine_mod = _load_chatterbox_engine_module(monkeypatch)
    calls: list = []
    _stub_chatterbox_variant(monkeypatch, dotted, cls_name, calls)

    engine = engine_mod.ChatterboxEngine(variant=variant, device="cpu")
    engine.ensure_model_ready()

    assert len(calls) == 1
    _, ckpt_dir, device = calls[0]
    # The engine normalizes through Path, so compare platform-normalized.
    assert Path(ckpt_dir) == Path("/tmp/fake-ckpt")
    assert device == "cpu"
    snap_kwargs = engine_mod.snapshot_download.call_args.kwargs
    assert snap_kwargs["repo_id"] == CHATTERBOX_REPO
    assert snap_kwargs["revision"] == MODEL_PINS[CHATTERBOX_REPO]
    assert SHA_RE.match(snap_kwargs["revision"])


def test_cache_symlinks_are_materialized_for_windows(bundle_builder, tmp_path: Path) -> None:
    """A symlinked snapshot entry would break an ordinary Windows install.

    ``snapshot_download`` links cached files with relative symlinks, and the
    installer seeds the local cache with ``copytree(symlinks=True)``. Creating a
    symlink on Windows needs Developer Mode or an elevated token, so every
    symlink in the bundle's cache must be a real file before the bundle ships.
    """
    cache = tmp_path / "hf_cache"
    repo = cache / "models--ResembleAI--chatterbox" / "snapshots" / "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
    blob = cache / "models--ResembleAI--chatterbox" / "blobs" / "deadbeef"
    repo.mkdir(parents=True)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"weights")
    os.symlink(os.path.relpath(blob, repo), repo / "conds.pt")
    (repo / "tokenizer.json").write_text("{}", encoding="utf-8")

    assert bundle_builder.dereference_cache_symlinks(cache) == 1
    materialized = repo / "conds.pt"
    assert not materialized.is_symlink()
    assert materialized.read_bytes() == b"weights"
    # A second pass is a no-op, so an already-clean bundle stays untouched.
    assert bundle_builder.dereference_cache_symlinks(cache) == 0


# --- bundle launchers --------------------------------------------------------


def test_bundle_launchers_invoke_offline_install(bundle_builder, tmp_path: Path) -> None:
    bundle_builder.write_launchers(tmp_path)
    sh = (tmp_path / "install.sh").read_text(encoding="utf-8")
    bat = (tmp_path / "install.bat").read_text(encoding="utf-8")
    assert "--offline-bundle" in sh
    assert "--offline-bundle" in bat

    assert "repo.tar.gz" in sh and "repo.tar.gz" in bat
    assert ':~-1%' in bat and ':~0,-1%' in bat
    assert '--offline-bundle "%BUNDLE_DIR%"' in bat
    assert 'tar -xzf "%BUNDLE_DIR%\\repo.tar.gz"' in bat

def test_model_cache_seeds_the_bundle_instead_of_redownloading(bundle_builder, tmp_path, monkeypatch) -> None:
    """A builder that already holds the pinned models must reuse them.

    Without this, building a bundle next to an existing hub cache re-downloads
    every pinned repo (~13 GB) into the bundle, which is exactly the cost the
    --model-cache flag exists to avoid.
    """
    existing = tmp_path / "existing"
    repo_dir = existing / "models--ResembleAI--chatterbox"
    snapshot = repo_dir / "snapshots" / "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
    (repo_dir / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18", encoding="ascii")
    (snapshot / "conds.pt").write_bytes(b"weights")
    blob = repo_dir / "blobs" / "deadbeef"
    blob.write_bytes(b"weights")
    (snapshot / "conds.pt").unlink()
    os.link(blob, snapshot / "conds.pt")
    (existing / "not-a-repo").mkdir()
    # A developer's hub cache also holds models the bundle must never carry:
    # only the pinned repos are seeded, or the bundle silently grows by the
    # size of every unrelated checkpoint on the builder.
    unrelated = existing / "models--AtomicChat--Qwen3.5-9B-GGUF"
    (unrelated / "snapshots" / "abc123").mkdir(parents=True)
    (unrelated / "snapshots" / "abc123" / "model.gguf").write_bytes(b"x" * 32)

    hf_cache = tmp_path / "bundle" / "hf_cache"
    assert bundle_builder.seed_cache(existing, hf_cache) == 1
    assert not (hf_cache / unrelated.name).exists(), "an unpinned repo was seeded into the bundle"
    seeded_repo = hf_cache / "models--ResembleAI--chatterbox"
    seeded = seeded_repo / "snapshots" / "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18" / "conds.pt"
    assert seeded.read_bytes() == b"weights"
    assert not (hf_cache / "not-a-repo").exists()
    # The blob is copied once; the snapshot entry is linked to it, so a linked
    # source cache does not double in the bundle.
    copied_blob = seeded_repo / "blobs" / "deadbeef"
    assert copied_blob.is_file()
    assert copied_blob.stat().st_ino == seeded.stat().st_ino
    # Idempotent: an existing destination is left alone rather than merged.
    assert bundle_builder.seed_cache(existing, hf_cache) == 0


def test_download_models_skips_the_network_for_seeded_repos(bundle_builder, tmp_path, monkeypatch) -> None:
    """Every repo already present in the seed cache must not hit the network."""
    import huggingface_hub

    existing = tmp_path / "existing"
    hf_cache = tmp_path / "bundle" / "hf_cache"
    repos = {
        bundle_builder.GO_EMOTIONS_REPO: ["config.json"],
        bundle_builder.PUNCTUATION_REPO: ["config.json"],
        bundle_builder.CHATTERBOX_REPO: ["conds.pt"],
        bundle_builder.TURBO_REPO_ID: ["ve.safetensors"],
    }
    for repo_id, files in repos.items():
        revision = bundle_builder.MODEL_PINS[repo_id]
        snapshot = existing / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        for name in files:
            (snapshot / name).write_bytes(b"weights")

    download_calls: list[str] = []

    def fake_snapshot_download(*, repo_id, revision, allow_patterns, cache_dir):
        download_calls.append(repo_id)
        return str(Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    sizes = bundle_builder.download_models(hf_cache, seed_from=existing)

    assert set(sizes) == set(repos)
    assert all(size for size in sizes.values())
    # snapshot_download is still called (it verifies the cache), but it cannot
    # have downloaded anything: every file it needs is already in hf_cache.
    for repo_id in download_calls:
        assert (hf_cache / f"models--{repo_id.replace('/', '--')}").is_dir()
    assert not list(hf_cache.rglob("*.incomplete"))


def test_offline_wheel_builder_packs_the_pinned_perth_git_commit(bundle_builder, tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(bundle_builder, "run", lambda args: calls.append(list(args)))

    bundle_builder.download_wheels(
        tmp_path / "wheels",
        platform="linux",
        python_version="3.12",
        pytorch="cpu",
    )

    assert len(calls) == 5
    perth_wheel = calls[2]
    assert perth_wheel[1:3] == ["-m", "pip"]
    assert "wheel" in perth_wheel
    assert bundle_builder.manage_install.PERTH_GIT_PACKAGE in perth_wheel
    pypi_download = calls[3]
    assert bundle_builder.manage_install.PERTH_GIT_PACKAGE not in pypi_download
    assert bundle_builder.manage_install.PERTH_OFFLINE_PACKAGE not in pypi_download
    assert "PyYAML>=6.0" in pypi_download


def test_torch_and_chatterbox_wheels_are_downloaded_without_deps(bundle_builder, tmp_path, monkeypatch) -> None:
    """Cross-platform torch/chatterbox downloads must not resolve dependencies.

    pip evaluates torch's ``platform_system == "Linux"`` nvidia markers against
    the building host, so a resolving `pip download --platform win_amd64
    torch==2.6.0` fails on a Linux builder ("No matching distribution found for
    nvidia-cudnn-cu12"). chatterbox-tts has the same effect one level up: it
    requires torch, which would pull the PyPI wheel family instead of the
    bundled one. Both are installed --no-deps by manage_install, so both are
    downloaded --no-deps here.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(bundle_builder, "run", lambda args: calls.append(list(args)))

    bundle_builder.download_wheels(
        tmp_path / "wheels",
        platform="windows",
        python_version="3.12",
        pytorch="cpu",
    )

    torch_download, torch_deps, _perth, resolving_download, no_deps_download = calls
    assert "--no-deps" in torch_download
    assert "--no-deps" not in torch_deps, "torch's runtime deps must resolve normally"
    assert set(bundle_builder.TORCH_RUNTIME_DEPS) <= set(torch_deps)
    assert all(pkg in torch_deps for pkg in ("sympy", "networkx", "jinja2"))
    assert "--platform" in torch_deps and "win_amd64" in torch_deps

    # Every package that requires torch or torchaudio is bundled --no-deps:
    # resolving them would make pip evaluate torch's Linux-only nvidia markers
    # on the building host and fail for a win_amd64 target.
    assert "--no-deps" in no_deps_download
    for name in bundle_builder.TORCH_CONSUMER_NAMES:
        assert any(req.lower().startswith(name) for req in no_deps_download), (
            f"{name} must be bundled --no-deps"
        )
        assert not any(req.lower().startswith(name) for req in resolving_download), (
            f"{name} must not be part of the resolving PyPI download"
        )
    assert bundle_builder.manage_install.CHATTERBOX_TTS_PACKAGE in no_deps_download
    # The resolving download must be pointed at the bundled torch wheels AND
    # constrained to their exact local versions: any package requiring torch
    # (conformer, deepmultilingualpunctuation, ...) would otherwise resolve
    # PyPI's default wheel, whose Linux-only nvidia pins cannot be satisfied
    # for a win_amd64 target.
    assert "--find-links" in resolving_download
    assert resolving_download[resolving_download.index("--find-links") + 1] == str(tmp_path / "wheels")
    assert "--constraint" in resolving_download
    constraints = Path(resolving_download[resolving_download.index("--constraint") + 1])
    pinned = constraints.read_text(encoding="utf-8")
    assert "torch==2.6.0" in pinned


def _write_fake_torch_wheel(wheels: Path, metadata_lines: list[str]) -> Path:
    wheel = wheels / "torch-2.6.0+cpu-cp312-cp312-win_amd64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "torch-2.6.0+cpu.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: torch\nVersion: 2.6.0+cpu\n" + "".join(metadata_lines),
        )
    return wheel


def test_torch_runtime_deps_come_from_the_bundled_wheel_metadata(bundle_builder, tmp_path) -> None:
    """torch pins its deps exactly; the loose list downloaded a sympy torch refuses.

    The offline install failed on `sympy==1.13.1` because the unconstrained
    download had fetched sympy 1.14.0. Linux-only nvidia requirements declared
    by torch must be dropped for a Windows target.
    """
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _write_fake_torch_wheel(
        wheels,
        [
            "Requires-Dist: filelock\n",
            "Requires-Dist: sympy==1.13.1; python_version >= \"3.9\"\n",
            "Requires-Dist: nvidia-nccl-cu12==2.21.5; platform_system == \"Linux\" and platform_machine == \"x86_64\"\n",
            "Requires-Dist: triton==3.2.0; platform_system == \"Linux\"\n",
            "Requires-Dist: opt-einsum>=3.3; extra == \"opt-einsum\"\n",
        ],
    )

    pins = bundle_builder.torch_dependency_pins(wheels, platform="windows", python_version="3.12")

    assert "sympy==1.13.1" in pins
    assert "filelock" in pins
    assert not any("nvidia" in pin for pin in pins), "a Linux-only nvidia pin leaked into a Windows bundle"
    assert not any("triton" in pin for pin in pins)
    assert not any("opt-einsum" in pin for pin in pins), "extra-gated requirements must be skipped"


def test_torch_runtime_deps_fall_back_without_a_torch_wheel(bundle_builder, tmp_path) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    assert bundle_builder.torch_dependency_pins(
        wheels, platform="windows", python_version="3.12"
    ) == list(bundle_builder.TORCH_RUNTIME_DEPS)


def _write_wheel(wheels: Path, dist: str, version: str, requires: list[str]) -> Path:
    wheel = wheels / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{dist}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {line}\n" for line in requires),
        )
    return wheel


def test_missing_target_requirements_finds_windows_only_dependencies(bundle_builder, tmp_path) -> None:
    """pip drops Windows-only markers on a Linux builder; the bundle must not.

    The real failure: tqdm requires ``colorama; platform_system == "Windows"``,
    so the wheel was never downloaded and the air-gapped install died on it.
    """
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _write_wheel(
        wheels,
        "tqdm",
        "4.67.1",
        [
            "colorama; platform_system == \"Windows\"",
            "numpy; platform_system == \"Linux\"",
            "typing-extensions>=4",
        ],
    )
    _write_wheel(wheels, "typing_extensions", "4.16.0", [])

    missing = bundle_builder.missing_target_requirements(wheels, platform="windows", python_version="3.12")

    assert "colorama" in missing
    assert not any("numpy" in item for item in missing), "a Linux-only requirement leaked into a Windows list"
    assert not any("typing" in item for item in missing), "a present requirement was reported missing"

    # Once the wheel is in the set the closure is closed.
    _write_wheel(wheels, "colorama", "0.4.6", [])
    assert bundle_builder.missing_target_requirements(wheels, platform="windows", python_version="3.12") == []


def test_missing_target_requirements_checks_version_specifiers(bundle_builder, tmp_path) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _write_wheel(wheels, "tokenizers", "0.22.0", [])
    _write_wheel(wheels, "transformers", "5.20.0", ["tokenizers<0.24.0,>=0.23.1"])

    missing = bundle_builder.missing_target_requirements(wheels, platform="windows", python_version="3.12")
    assert any(item.startswith("tokenizers") for item in missing), (
        "a present-but-too-old requirement must be reported as missing"
    )


def test_closure_ignores_metadata_the_installer_patches(bundle_builder, tmp_path) -> None:
    """chatterbox-tts's published pins are not what the installer installs.

    manage_install installs it --no-deps and patches its metadata to the audited
    stack, so following its declared pins would bundle transformers==5.2.0 and
    the GPL-3.0 pykakasi that the patch deliberately drops.
    """
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _write_wheel(
        wheels,
        "chatterbox_tts",
        "0.1.7",
        ["transformers==5.2.0", "pykakasi==2.3.0", "diffusers==0.29.0"],
    )
    _write_wheel(wheels, "transformers", "5.20.0", [])

    assert bundle_builder.missing_target_requirements(wheels, platform="windows", python_version="3.12") == []


def test_torch_constraints_use_the_local_cpu_wheel_versions(bundle_builder, tmp_path) -> None:
    """The constraint pins the +cpu local versions, which PyPI never publishes.

    A plain ``torch==2.6.0`` pin still lets pip read PyPI's torch metadata and
    fail on its Linux-only nvidia requirements; only the local version string
    forces the bundled wheel (no nvidia deps) to be the single candidate.
    """
    wheels = tmp_path / "wheels" / "windows"
    wheels.mkdir(parents=True)
    for name in (
        "torch-2.6.0+cpu-cp312-cp312-win_amd64.whl",
        "torchaudio-2.6.0+cpu-cp312-cp312-win_amd64.whl",
        "torchvision-0.21.0+cpu-cp312-cp312-win_amd64.whl",
        "numpy-2.4.5-cp312-cp312-win_amd64.whl",
    ):
        (wheels / name).write_bytes(b"")

    constraints = bundle_builder.torch_constraints(wheels)
    pins = constraints.read_text(encoding="utf-8").split()
    assert pins == [
        "torch==2.6.0+cpu",
        "torchaudio==2.6.0+cpu",
        "torchvision==0.21.0+cpu",
    ]
def _make_fake_run_capture():
    captured: dict = {}

    def fake_run(args, **kwargs):
        captured["args"] = list(args)
        captured["env"] = dict(kwargs.get("env") or {})
        return types.SimpleNamespace(returncode=0)

    return captured, fake_run


def _prepare_run_repo(manage_install, tmp_path: Path, monkeypatch, *, offline: bool) -> None:
    repo = tmp_path / "repo with space"
    entrypoint = manage_install.venv_entrypoint_path(repo, "the-oracle")
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text("#!/bin/sh\n", encoding="utf-8")
    if offline:
        (repo / manage_install.OFFLINE_MARKER_FILENAME).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(manage_install, "REPO_ROOT", repo)
    monkeypatch.setenv("DISPLAY", ":0")


def test_run_path_exports_hf_hub_offline_on_offline_install(
    manage_install, tmp_path: Path, monkeypatch
) -> None:
    _prepare_run_repo(manage_install, tmp_path, monkeypatch, offline=True)
    captured, fake_run = _make_fake_run_capture()
    monkeypatch.setattr(manage_install.subprocess, "run", fake_run)
    assert manage_install.run_gui() == 0
    assert captured["env"].get("HF_HUB_OFFLINE") == "1"
    assert captured["args"][-1] == "gui"


def test_run_path_leaves_offline_flag_unset_without_marker(
    manage_install, tmp_path: Path, monkeypatch
) -> None:
    _prepare_run_repo(manage_install, tmp_path, monkeypatch, offline=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    captured, fake_run = _make_fake_run_capture()
    monkeypatch.setattr(manage_install.subprocess, "run", fake_run)
    assert manage_install.run_gui() == 0
    assert "HF_HUB_OFFLINE" not in captured["env"]


# --- an invalid bundle must not leave a half-made cache behind -----------------


def test_seed_offline_models_validates_before_creating_cache(
    manage_install, tmp_path: Path, monkeypatch
) -> None:
    """An incomplete bundle must leave the machine untouched: the HF cache is
    created only once every pinned model is known to be present."""
    bundle = tmp_path / "bundle"
    (bundle / "hf_cache").mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    hf_cache = tmp_path / "hf" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache))
    with pytest.raises(SystemExit):
        manage_install.seed_offline_models(bundle)
    assert not hf_cache.exists(), "an invalid bundle created the HF cache"
