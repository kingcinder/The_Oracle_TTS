"""Release tooling: the single-source version invariant, banner sync, and checksums.

The contract under test: ``src/the_oracle/__init__.py`` is the only tracked
place a release version is written; pyproject.toml reads it dynamically and
the README / STATE.md banners agree with it. scripts/release.py enforces that
invariant and produces versioned artifacts with a sha256 manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release.py"


def _load_release():
    spec = importlib.util.spec_from_file_location("release_tool", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


release = _load_release()


def _write_fake_repo(
    tmp_path: Path,
    *,
    version: str = "9.9.9",
    readme_version: str | None = None,
    state_version: str | None = None,
    pyproject_literal: bool = False,
    banner_in_state: bool = True,
) -> Path:
    """A minimal tracked tree with the same shape the script expects."""
    (tmp_path / "src" / "the_oracle").mkdir(parents=True)
    (tmp_path / "src" / "the_oracle" / "__init__.py").write_text(
        f'__all__ = ["__version__"]\n\n__version__ = "{version}"\n', encoding="utf-8"
    )
    if pyproject_literal:
        project = f'[project]\nname = "the-oracle"\nversion = "1.2.3"\n'
        dynamic = ""
    else:
        project = '[project]\nname = "the-oracle"\ndynamic = ["version"]\n'
        dynamic = '[tool.setuptools.dynamic]\nversion = { attr = "the_oracle.__version__" }\n'
    (tmp_path / "pyproject.toml").write_text(project + "\n" + dynamic, encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# The Oracle\n\n"
        f"**Release V{readme_version or version}** — some release description.\n\n"
        "Body text that must never be touched.\n",
        encoding="utf-8",
    )
    state_banner = f"**Current release: V{state_version or version} (notes)**\n" if banner_in_state else ""
    (tmp_path / "STATE.md").write_text(
        "# State\n\n" + state_banner + "\nMore body text.\n", encoding="utf-8"
    )
    return tmp_path


def test_pyproject_is_generated_from_the_package_version() -> None:
    """The collapse pin: no version literal in pyproject; it reads __version__."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert "version" not in project
    assert "version" in project["dynamic"]
    attr = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "the_oracle.__version__"


def test_check_passes_on_this_repo() -> None:
    assert release.check(REPO_ROOT) == []


def test_check_flags_banner_drift(tmp_path: Path) -> None:
    repo = _write_fake_repo(tmp_path, readme_version="8.8.8")
    problems = release.check(repo)
    assert len(problems) == 1
    assert "README.md" in problems[0]
    assert "8.8.8" in problems[0]
    assert "9.9.9" in problems[0]


def test_check_flags_pyproject_version_literal(tmp_path: Path) -> None:
    repo = _write_fake_repo(tmp_path, pyproject_literal=True)
    problems = release.check(repo)
    assert any("pyproject.toml" in problem for problem in problems)


def test_sync_banners_updates_only_the_version(tmp_path: Path) -> None:
    repo = _write_fake_repo(tmp_path, readme_version="8.8.8", state_version="8.8.8")
    before = {
        "README.md": (repo / "README.md").read_text(encoding="utf-8"),
        "STATE.md": (repo / "STATE.md").read_text(encoding="utf-8"),
    }

    changed = release.sync_banners(repo, "9.9.9")

    assert changed == ["README.md", "STATE.md"]
    for filename, old_text in before.items():
        new_text = (repo / filename).read_text(encoding="utf-8")
        old_lines, new_lines = old_text.splitlines(), new_text.splitlines()
        assert len(old_lines) == len(new_lines)
        diffs = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
        assert len(diffs) == 1
        old_line, new_line = diffs[0]
        assert old_line.replace("V8.8.8", "V9.9.9") == new_line
        assert new_text.count("9.9.9") == 1
    assert release.check(repo) == []


def test_sync_banners_refuses_missing_banner(tmp_path: Path) -> None:
    repo = _write_fake_repo(tmp_path, banner_in_state=False)
    with pytest.raises(release.ReleaseError, match="STATE.md"):
        release.sync_banners(repo, "9.9.9")





def test_write_checksums_manifest_is_sha256sum_compatible(tmp_path: Path) -> None:
    (tmp_path / "the_oracle-9.9.9.tar.gz").write_bytes(b"sdist bytes")
    (tmp_path / "the_oracle-9.9.9-py3-none-any.whl").write_bytes(b"wheel bytes")
    # A previous release run left its manifest (and an unrelated file) behind:
    (tmp_path / "checksums-9.9.9.sha256").write_text("stale\n", encoding="utf-8")
    (tmp_path / "unrelated-notes.txt").write_text("not an artifact\n", encoding="utf-8")
    artifacts = [
        tmp_path / "the_oracle-9.9.9-py3-none-any.whl",
        tmp_path / "the_oracle-9.9.9.tar.gz",
    ]

    manifest = release.write_checksums(tmp_path, "9.9.9", artifacts)

    assert manifest.name == "checksums-9.9.9.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"{hashlib.sha256(b'wheel bytes').hexdigest()}  the_oracle-9.9.9-py3-none-any.whl",
        f"{hashlib.sha256(b'sdist bytes').hexdigest()}  the_oracle-9.9.9.tar.gz",
    ]
    # The manifest must never list itself or unrelated files, or `sha256sum -c`
    # could never pass on a rebuilt release directory.
    assert not any("checksums-9.9.9.sha256" in line or "unrelated" in line for line in lines)
    # Where the tool exists (Linux CI), also prove the format end-to-end;
    # the hashlib assertion above is the primary pin on every platform.
    if shutil.which("sha256sum"):
        subprocess.run(["sha256sum", "-c", manifest.name], cwd=str(tmp_path), check=True)


def test_build_artifacts_rejects_unversioned_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResult:
        stdout = "ARTIFACT:the_oracle-py3-none-any.whl\nARTIFACT:the_oracle.tar.gz\n"

    calls: list[tuple[list[str], str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("cwd")))
        return _FakeResult()

    monkeypatch.setattr(release.subprocess, "run", _fake_run)
    with pytest.raises(release.ReleaseError, match="not versioned"):
        release.build_artifacts(tmp_path, tmp_path / "out", "9.9.9")
    cmd, cwd = calls[0]
    assert cmd[0] == release.sys.executable
    assert cmd[1] == "-c"
    assert "build_meta" in cmd[2] and "build_sdist" in cmd[2] and "build_wheel" in cmd[2]
    assert cmd[3] == str(tmp_path / "out")
    assert cwd == str(tmp_path)


def test_build_artifacts_returns_existing_versioned_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "the_oracle-9.9.9.tar.gz").write_bytes(b"s")
    (outdir / "the_oracle-9.9.9-py3-none-any.whl").write_bytes(b"w")

    class _FakeResult:
        stdout = "ARTIFACT:the_oracle-9.9.9.tar.gz\nARTIFACT:the_oracle-9.9.9-py3-none-any.whl\n"

    monkeypatch.setattr(release.subprocess, "run", lambda cmd, **kwargs: _FakeResult())
    artifacts = release.build_artifacts(tmp_path, outdir, "9.9.9")
    assert [path.name for path in artifacts] == ["the_oracle-9.9.9.tar.gz", "the_oracle-9.9.9-py3-none-any.whl"]


def test_main_refuses_dirty_tree_before_building_or_testing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _write_fake_repo(tmp_path)
    monkeypatch.setattr(release, "tree_is_dirty", lambda _repo: True)
    tested = []
    monkeypatch.setattr(release, "run_tests", lambda _repo: tested.append(1))

    status = release.main(["--skip-tests", "--repo-root", str(repo)])

    assert status == 1
    assert tested == []  # the refusal happens before any expensive step
    assert not (repo / "release_artifacts").exists()
