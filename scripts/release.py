#!/usr/bin/env python3
"""Release tooling for The Oracle.

Version metadata lives in exactly one place: ``__version__`` in
``src/the_oracle/__init__.py``. ``pyproject.toml`` declares it dynamically
(``[tool.setuptools.dynamic]``), and the README / STATE.md release banners
are rewritten from it by ``--sync-banners``. No other tracked file may carry
the number — ``--check`` enforces that invariant.

Usage:
    python scripts/release.py --check
        Verify the single-source invariant only (read-only; CI-safe).

    python scripts/release.py --sync-banners
        Rewrite the version inside the README / STATE.md release banners
        from ``__version__``. The only metadata-writing mode.

    python scripts/release.py [--outdir DIR] [--skip-tests]
        Full release: check, refuse a dirty tree, run the test suite,
        build the sdist + wheel with the in-venv setuptools PEP 517 hooks
        (no network, no ``build`` package needed), and write a versioned
        ``checksums-<version>.sha256`` manifest beside the artifacts.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path


class ReleaseError(Exception):
    """A release step cannot proceed; message is user-facing."""


#: Files carrying a human-facing release banner, with the exact line prefix
#: that identifies the banner. The version inside the line is rewritten by
#: --sync-banners and verified by --check.
BANNER_SITES: tuple[tuple[str, str], ...] = (
    ("README.md", "**Release V"),
    ("STATE.md", "**Current release: V"),
)

_BANNER_VERSION_RE = re.compile(r"V(\d+\.\d+\.\d+)")

#: The single source of truth, as setuptools must read it.
_VERSION_ATTR = "the_oracle.__version__"

# Printed by the build subprocess so artifact names survive any unrelated
# setuptools chatter on stdout.
_ARTIFACT_PREFIX = "ARTIFACT:"

_BUILD_SNIPPET = (
    "import sys\n"
    "from setuptools import build_meta\n"
    f"print('{_ARTIFACT_PREFIX}' + build_meta.build_sdist(sys.argv[1]))\n"
    f"print('{_ARTIFACT_PREFIX}' + build_meta.build_wheel(sys.argv[1]))\n"
)


def read_version(repo_root: Path) -> str:
    """Return the ``__version__`` literal, without importing the package."""
    init = repo_root / "src" / "the_oracle" / "__init__.py"
    if not init.is_file():
        return ""
    tree = ast.parse(init.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    return ""


def pyproject_problems(repo_root: Path) -> list[str]:
    """Drift between pyproject.toml and the single source of truth."""
    path = repo_root / "pyproject.toml"
    if not path.is_file():
        return ["pyproject.toml: missing"]
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    problems: list[str] = []
    if "version" in project:
        problems.append(
            'pyproject.toml: [project] sets a version literal; declare dynamic = ["version"] '
            "and let [tool.setuptools.dynamic] read the_oracle.__version__"
        )
    if "version" not in project.get("dynamic", []):
        problems.append("pyproject.toml: [project] does not declare 'version' in dynamic")
    attr = data.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("version", {}).get("attr")
    if attr != _VERSION_ATTR:
        problems.append(
            f"pyproject.toml: [tool.setuptools.dynamic] version.attr must be {_VERSION_ATTR!r}"
        )
    return problems


def banner_problems(repo_root: Path, version: str) -> list[str]:
    """Drift between the release banners and the single source of truth."""
    problems: list[str] = []
    for filename, prefix in BANNER_SITES:
        path = repo_root / filename
        if not path.is_file():
            problems.append(f"{filename}: missing (expected a banner line starting {prefix!r})")
            continue
        banner_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(prefix)]
        if len(banner_lines) != 1:
            problems.append(
                f"{filename}: expected exactly one banner line starting {prefix!r}, found {len(banner_lines)}"
            )
            continue
        match = _BANNER_VERSION_RE.search(banner_lines[0])
        if not match:
            problems.append(f"{filename}: banner line carries no V<major>.<minor>.<patch>: {banner_lines[0]!r}")
        elif match.group(1) != version:
            problems.append(
                f"{filename}: banner says V{match.group(1)} but __version__ (the single source) says {version}"
            )
    return problems


def check(repo_root: Path) -> list[str]:
    """Every way the tracked tree disagrees about the version."""
    problems: list[str] = []
    version = read_version(repo_root)
    if not version:
        problems.append("src/the_oracle/__init__.py: no __version__ string literal found")
    problems.extend(pyproject_problems(repo_root))
    if version:
        problems.extend(banner_problems(repo_root, version))
    return problems


def sync_banners(repo_root: Path, version: str) -> list[str]:
    """Rewrite the version inside each banner line; return changed files."""
    changed: list[str] = []
    for filename, prefix in BANNER_SITES:
        path = repo_root / filename
        if not path.is_file():
            raise ReleaseError(f"{filename}: missing; cannot sync its release banner")
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        hits = [index for index, line in enumerate(lines) if line.startswith(prefix)]
        if len(hits) != 1:
            raise ReleaseError(
                f"{filename}: expected exactly one banner line starting {prefix!r}, found {len(hits)}"
            )
        new_line = _BANNER_VERSION_RE.sub(f"V{version}", lines[hits[0]], count=1)
        if new_line != lines[hits[0]]:
            lines[hits[0]] = new_line
            path.write_text("".join(lines), encoding="utf-8")
            changed.append(filename)
    return changed


def tree_is_dirty(repo_root: Path) -> bool:
    """True when git reports any uncommitted change (staged, unstaged, or untracked)."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def run_tests(repo_root: Path) -> None:
    """Run the full suite; QT_QPA_PLATFORM defaults to offscreen for headless hosts."""
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    subprocess.run([sys.executable, "-m", "pytest"], cwd=str(repo_root), env=env, check=True)


def build_artifacts(repo_root: Path, outdir: Path, version: str) -> list[Path]:
    """Build sdist + wheel via the PEP 517 hooks; return versioned artifact paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-c", _BUILD_SNIPPET, str(outdir)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    names = [line[len(_ARTIFACT_PREFIX):].strip() for line in result.stdout.splitlines() if line.startswith(_ARTIFACT_PREFIX)]
    if len(names) != 2:
        raise ReleaseError(f"expected the build to report one sdist and one wheel, got {names!r}")
    artifacts: list[Path] = []
    for name in names:
        if version not in name:
            raise ReleaseError(f"artifact {name!r} is not versioned (expected {version} in the file name)")
        path = outdir / name
        if not path.is_file():
            raise ReleaseError(f"the build reported {name!r} but it is missing from {outdir}")
        artifacts.append(path)
    return artifacts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()





def write_checksums(outdir: Path, version: str, artifacts: list[Path]) -> Path:
    """Write a sha256sum -c manifest over exactly this run's artifacts.

    Scoped to the passed artifacts rather than a directory scan: a rebuild in
    a non-empty outdir must not checksum stale manifests or unrelated files.
    """
    if not artifacts:
        raise ReleaseError(f"no artifacts to checksum in {outdir}")
    manifest = outdir / f"checksums-{version}.sha256"
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in sorted(artifacts, key=lambda p: p.name)),
        encoding="utf-8",
    )
    return manifest


def _require_version(repo_root: Path) -> str:
    version = read_version(repo_root)
    if not version:
        raise ReleaseError("src/the_oracle/__init__.py: no __version__ string literal found")
    return version


def _report(problems: list[str]) -> int:
    if problems:
        for problem in problems:
            print(f"MISMATCH: {problem}", file=sys.stderr)
        return 1
    print("single-source version invariant holds")
    return 0


def _release(repo: Path, outdir_arg: str, *, skip_tests: bool) -> int:
    status = _report(check(repo))
    if status != 0:
        return status
    version = _require_version(repo)
    if tree_is_dirty(repo):
        print(
            "refusing to release from a dirty tree — run --sync-banners, commit, then rerun",
            file=sys.stderr,
        )
        return 1
    if not skip_tests:
        print("running the test suite ...")
        run_tests(repo)
    outdir = repo / outdir_arg
    print(f"building sdist + wheel into {outdir} ...")
    artifacts = build_artifacts(repo, outdir, version)
    manifest = write_checksums(outdir, version, artifacts)
    print("artifacts:")
    for path in artifacts:
        print(f"  {path}")
    print(f"checksums: {manifest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build verified release artifacts for The Oracle.")
    parser.add_argument("--check", action="store_true", help="verify the single-source version invariant only")
    parser.add_argument("--sync-banners", action="store_true", help="rewrite README/STATE release banners from __version__")
    parser.add_argument("--outdir", default="release_artifacts", help="artifact output directory, relative to the repo root")
    parser.add_argument("--skip-tests", action="store_true", help="skip the pytest run before building")
    parser.add_argument("--repo-root", type=Path, default=None, help="repo root (default: the checkout this script lives in)")
    args = parser.parse_args(argv)
    repo = args.repo_root if args.repo_root is not None else Path(__file__).resolve().parents[1]

    try:
        if args.check:
            return _report(check(repo))
        if args.sync_banners:
            version = _require_version(repo)
            changed = sync_banners(repo, version)
            print(f"synced {', '.join(changed)} to V{version}" if changed else f"banners already at V{version}")
            return _report(check(repo))
        return _release(repo, args.outdir, skip_tests=args.skip_tests)
    except ReleaseError as exc:
        print(f"release: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"release: command failed ({exc.returncode}): {' '.join(map(str, exc.cmd))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
