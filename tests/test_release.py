"""Release tooling: the single-source version invariant, banner sync, and checksums.

The contract under test: ``src/the_oracle/__init__.py`` is the only tracked
place a release version is written; pyproject.toml reads it dynamically and
the README / STATE.md banners agree with it. scripts/release.py enforces that
invariant and produces versioned artifacts with a sha256 manifest. The
invariant also covers CHANGELOG.md: the current version must have exactly one
section, dated the release day, and the file's structure is audited too —
exactly one ``[Unreleased]`` placeholder, released sections newest-first, and
no empty shipped section.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tomllib
from datetime import date
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
    changelog_version: str | None = None,
    changelog_date: str | None = None,
    changelog_text: str | None = None,
) -> Path:
    """A minimal tracked tree with the same shape the script expects.

    The CHANGELOG defaults to a section for ``version`` dated *today*, which
    is what a release-morning tree looks like; the date parameters let tests
    build yesterday's tree and drift scenarios.
    """
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
    if changelog_text is None:
        stamped_version = changelog_version if changelog_version is not None else version
        stamped_date = changelog_date if changelog_date is not None else date.today().isoformat()
        changelog_text = (
            "# Changelog\n\n"
            "## [Unreleased]\n\n### Changed\n\n- (nothing yet)\n\n"
            f"## [{stamped_version}] — {stamped_date}\n\n### Added\n\n- Something.\n"
        )
    if changelog_text != "":
        (tmp_path / "CHANGELOG.md").write_text(changelog_text, encoding="utf-8")
    return tmp_path


def test_pyproject_is_generated_from_the_package_version() -> None:
    """The collapse pin: no version literal in pyproject; it reads __version__."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert "version" not in project
    assert "version" in project["dynamic"]
    attr = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "the_oracle.__version__"


def test_check_passes_on_this_repo_on_its_own_release_day() -> None:
    """The full invariant, evaluated on the day the changelog itself records.

    Undated --check (real ``date.today()``) passes only on the release day by
    design; between releases the tree legitimately fails it. This pin runs
    the real repo's check at the date stamped on its current section, so the
    invariant itself (pyproject + banners + changelog heading) is verified on
    real content without being hostage to the calendar.
    """
    version = release.read_version(REPO_ROOT)
    assert version, "the real repo must carry a __version__"
    headings = [
        match
        for match in (
            release._CHANGELOG_HEADING_RE.match(line)
            for line in (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
        )
        if match
    ]
    stamped = {match.group("version"): match.group("date") for match in headings}
    assert version in stamped, f"CHANGELOG.md has no section for the current version {version}"
    assert release.check(REPO_ROOT, today=date.fromisoformat(stamped[version])) == []


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


def test_check_flags_missing_changelog_section(tmp_path: Path) -> None:
    """A version bump without a changelog section for the new version fails."""
    repo = _write_fake_repo(tmp_path, changelog_version="8.0.0")
    problems = release.check(repo)
    assert len(problems) == 1
    assert "CHANGELOG.md" in problems[0]
    assert "9.9.9" in problems[0]
    assert date.today().isoformat() in problems[0], (
        "the fix message must name the exact heading to add"
    )


def test_check_flags_stale_changelog_date(tmp_path: Path) -> None:
    """The section exists but is not dated the release day."""
    yesterday = date.today().toordinal() - 1
    yesterday_iso = date.fromordinal(yesterday).isoformat()
    repo = _write_fake_repo(tmp_path, changelog_date=yesterday_iso)
    problems = release.check(repo)
    assert len(problems) == 1
    assert yesterday_iso in problems[0]
    assert date.today().isoformat() in problems[0]


def test_check_flags_duplicate_changelog_sections(tmp_path: Path) -> None:
    repo = _write_fake_repo(
        tmp_path,
        changelog_text=(
            "# Changelog\n\n"
            "## [9.9.9] — 2026-01-01\n\n- First.\n\n"
            "## [9.9.9] — 2026-01-02\n\n- Second.\n"
        ),
    )
    problems = release.check(repo)
    assert len(problems) == 1
    assert "2 sections" in problems[0]
    assert "lines 3, 7" in problems[0]


def test_check_flags_missing_changelog_file(tmp_path: Path) -> None:
    repo = _write_fake_repo(tmp_path, changelog_text="")
    problems = release.check(repo)
    assert any("CHANGELOG.md: missing" in problem for problem in problems)


def test_check_flags_invalid_changelog_date(tmp_path: Path) -> None:
    """The heading shape matches but the date is not a real calendar day."""
    repo = _write_fake_repo(
        tmp_path,
        changelog_text="# Changelog\n\n## [9.9.9] — 2026-13-45\n\n- Something.\n",
    )
    problems = release.check(repo)
    assert len(problems) == 1
    assert "not a valid calendar date" in problems[0]


def test_check_tolerates_unreleased_and_other_versions(tmp_path: Path) -> None:
    """Only the CURRENT version's section is date-checked; [Unreleased] and
    older sections are none of --check's business."""
    repo = _write_fake_repo(
        tmp_path,
        changelog_text=(
            "# Changelog\n\n"
            "## [Unreleased]\n\n### Changed\n\n- (nothing yet)\n\n"
            "## [9.9.9] — " + date.today().isoformat() + "\n\n### Added\n\n- Now.\n\n"
            "## [0.9.0] — 2025-12-25\n\n### Added\n\n- Before.\n"
        ),
    )
    assert release.check(repo) == []


def test_check_passes_when_unreleased_sits_below_the_stamped_release(tmp_path: Path) -> None:
    """--sync-changelog's own output shape: the stamped version section goes
    ABOVE the re-inserted empty [Unreleased] placeholder, so the placeholder
    must be exempt from the newest-first rule — otherwise the release tool
    could never pass its own gate."""
    repo = _write_fake_repo(
        tmp_path,
        changelog_text=(
            "# Changelog\n\n"
            "## [9.9.9] — " + date.today().isoformat() + "\n\n### Added\n\n- Now.\n\n"
            "## [Unreleased]\n\n### Changed\n\n- (nothing yet)\n\n"
            "## [0.9.0] — 2025-12-25\n\n### Added\n\n- Before.\n"
        ),
    )
    assert release.check(repo) == []


def test_check_flags_duplicate_unreleased_sections(tmp_path: Path) -> None:
    """The misfire this gate exists for: a hand-edit (or a partial sync) leaves
    two [Unreleased] placeholders, and later release notes accumulate in the
    wrong one."""
    repo = _write_fake_repo(
        tmp_path,
        changelog_text=(
            "# Changelog\n\n"
            "## [Unreleased]\n\n### Changed\n\n- (nothing yet)\n\n"
            "## [Unreleased]\n\n### Fixed\n\n- More.\n\n"
            "## [9.9.9] — " + date.today().isoformat() + "\n\n### Added\n\n- Now.\n"
        ),
    )
    problems = release.check(repo)
    assert len(problems) == 1
    assert "2 '## [Unreleased]' sections" in problems[0]
    assert "merge them" in problems[0]


def test_check_flags_misordered_version_sections(tmp_path: Path) -> None:
    """A newer release appearing below an older one is structural damage —
    the newest-first contract is what keeps the reading order trustworthy."""
    repo = _write_fake_repo(
        tmp_path,
        changelog_text=(
            "# Changelog\n\n"
            "## [Unreleased]\n\n### Changed\n\n- (nothing yet)\n\n"
            "## [9.9.8] — 2026-09-01\n\n### Added\n\n- Older.\n\n"
            "## [9.9.9] — " + date.today().isoformat() + "\n\n### Added\n\n- Now.\n"
        ),
    )
    problems = release.check(repo)
    assert len(problems) == 1
    assert "out of order" in problems[0]
    assert "[9.9.9]" in problems[0] and "[9.9.8]" in problems[0]
    assert "newest-first" in problems[0]


def test_check_flags_an_empty_shipped_section(tmp_path: Path) -> None:
    """A version stamped into the changelog that records nothing is the empty
    1.0.0-placeholder shape; a shipped section must carry at least one entry."""
    repo = _write_fake_repo(
        tmp_path,
        changelog_text=(
            "# Changelog\n\n"
            "## [Unreleased]\n\n### Changed\n\n- (nothing yet)\n\n"
            "## [9.9.9] — " + date.today().isoformat() + "\n\n### Added\n\n- Now.\n\n"
            "## [9.9.8] — 2026-09-01\n\n### Fixed\n\n## [9.9.7] — 2026-08-30\n\n### Added\n\n- Old.\n"
        ),
    )
    problems = release.check(repo)
    assert len(problems) == 1
    assert "the [9.9.8] section is empty" in problems[0]


def test_check_subheadings_and_blockquote_prose_do_not_count_as_entries(tmp_path: Path) -> None:
    """### sub-headings and blockquote notes give a section shape without
    recording a single change — the emptiness rule must see through them."""
    repo = _write_fake_repo(
        tmp_path,
        changelog_text=(
            "# Changelog\n\n"
            "## [9.9.9] — " + date.today().isoformat() + "\n\n"
            "### Added\n\n> Some retrospective prose, no actual entries.\n\n"
        ),
    )
    problems = release.check(repo)
    assert len(problems) == 1
    assert "is empty" in problems[0]


def test_check_changelog_date_follows_injected_today(tmp_path: Path) -> None:
    """The date comparison evaluates against the injected release day.

    Undated --check uses the real calendar, so a section dated tomorrow
    legitimately fails it; only the matching injected ``today`` passes.
    """
    tomorrow = date.fromordinal(date.today().toordinal() + 1)
    repo = _write_fake_repo(tmp_path, changelog_date=tomorrow.isoformat())
    assert release.check(repo) != []
    assert release.check(repo, today=tomorrow) == []
    assert release.check(repo, today=date.today()) != []


def test_changelog_heading_requires_the_canonical_form(tmp_path: Path) -> None:
    """A near-miss heading (hyphen, bolded, missing brackets) is 'missing',
    so hand-written variants cannot silently satisfy the invariant."""
    for index, bad_heading in enumerate(
        (
            "## [9.9.9] - 2026-01-01",
            "**## [9.9.9] — 2026-01-01**",
            "## 9.9.9 — 2026-01-01",
            "## [9.9.9] — 2026-1-1",
        )
    ):
        repo = _write_fake_repo(
            tmp_path / f"case-{index}",
            changelog_text=f"# Changelog\n\n{bad_heading}\n\n- Something.\n",
        )
        problems = release.check(repo)
        assert len(problems) == 1, bad_heading
        assert "no section for version 9.9.9" in problems[0], bad_heading


def test_sync_changelog_retitles_unreleased_body(tmp_path: Path) -> None:
    """The release-day edit, as one command: [Unreleased]'s body (including
    its ### sub-headings) becomes the dated version section, and [Unreleased]
    survives below as an empty placeholder for future work."""
    repo = _write_fake_repo(tmp_path)
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n### Added\n\n- New thing.\n- Another.\n\n"
        "## [1.0.0] — 2026-01-01\n\n### Added\n\n- Old.\n",
        encoding="utf-8",
    )

    changed = release.sync_changelog(repo, "9.9.9", today=date(2026, 9, 21))

    assert changed == ["CHANGELOG.md"]
    text = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.startswith(
        "# Changelog\n\n"
        "## [9.9.9] — 2026-09-21\n\n### Added\n\n- New thing.\n- Another.\n\n"
        "## [Unreleased]\n\n### Changed\n\n- (nothing yet)\n\n"
        "## [1.0.0] — 2026-01-01"
    )
    # The stamped tree must satisfy the full invariant on the stamped day.
    assert release.check(repo, today=date(2026, 9, 21)) == []


def test_sync_changelog_rewrites_an_existing_sections_date(tmp_path: Path) -> None:
    """When the section already exists (e.g. written ahead), only its date
    moves; the body and any neighbors are untouched."""
    repo = _write_fake_repo(tmp_path, changelog_date="2026-01-01")
    changed = release.sync_changelog(repo, "9.9.9", today=date(2026, 9, 21))
    assert changed == ["CHANGELOG.md"]
    text = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [9.9.9] — 2026-09-21" in text
    assert "2026-01-01" not in text
    assert release.check(repo, today=date(2026, 9, 21)) == []


def test_sync_changelog_is_idempotent(tmp_path: Path) -> None:
    repo = _write_fake_repo(tmp_path, changelog_date="2026-01-01")
    assert release.sync_changelog(repo, "9.9.9", today=date(2026, 9, 21)) == ["CHANGELOG.md"]
    assert release.sync_changelog(repo, "9.9.9", today=date(2026, 9, 21)) == []


def test_sync_changelog_refuses_when_nothing_to_retitle(tmp_path: Path) -> None:
    repo = _write_fake_repo(tmp_path, changelog_text="")
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.0.0] — 2026-01-01\n\n- Old.\n", encoding="utf-8"
    )
    with pytest.raises(release.ReleaseError, match="Unreleased"):
        release.sync_changelog(repo, "9.9.9", today=date(2026, 9, 21))


def test_sync_changelog_refuses_missing_file(tmp_path: Path) -> None:
    repo = _write_fake_repo(tmp_path, changelog_text="")
    (repo / "CHANGELOG.md").write_text("placeholder", encoding="utf-8")  # ensure it existed
    (repo / "CHANGELOG.md").unlink()
    with pytest.raises(release.ReleaseError, match="missing"):
        release.sync_changelog(repo, "9.9.9", today=date(2026, 9, 21))


def test_main_sync_changelog_stamps_and_then_check_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real CLI mode: sync_changelog actually runs, and the follow-up
    --check report inside the same invocation holds (evaluated today, since
    main stamps with the real calendar)."""
    repo = _write_fake_repo(tmp_path, changelog_date="2026-01-01")
    reports: list[list[str]] = []
    monkeypatch.setattr(release, "_report", lambda problems: reports.append(problems) or 0)

    status = release.main(["--sync-changelog", "--repo-root", str(repo)])

    assert status == 0
    assert reports == [[]], f"check inside the CLI run must hold: {reports}"
    text = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [9.9.9] — {date.today().isoformat()}" in text


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


def test_build_snippet_binds_outdir_before_setuptools_mutates_argv(tmp_path: Path) -> None:
    """setuptools' build_meta reassigns sys.argv in place without restoring it;
    the snippet must bind the outdir to a local before the first hook call, or
    build_wheel receives a setuptools-internal flag as its directory."""
    project = tmp_path / "proj"
    (project / "src" / "tiny_pkg").mkdir(parents=True)
    (project / "src" / "tiny_pkg" / "__init__.py").write_text('__version__ = "0.0.1"\n', encoding="utf-8")
    (project / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=75.8.0", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        '[project]\nname = "tiny-pkg"\nversion = "0.0.1"\n',
        encoding="utf-8",
    )
    outdir = tmp_path / "out"

    result = subprocess.run(
        [sys.executable, "-c", release._BUILD_SNIPPET, str(outdir)],
        cwd=str(project),
        capture_output=True,
        text=True,
        check=True,
    )

    names = [line[len("ARTIFACT:"):].strip() for line in result.stdout.splitlines() if line.startswith("ARTIFACT:")]
    assert len(names) == 2
    for name in names:
        assert (outdir / name).is_file(), f"{name} never landed in {outdir}"
    # The old bug's signature: the cwd gains a stray hook-named directory.
    assert not (project / "sdist").exists()


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


def test_store_release_checksums_records_a_copy_verifiable_from_a_clone(tmp_path: Path) -> None:
    """The point of the tracked copy: a published artifact must be verifiable
    from the repository alone. The build folder is gitignored, so a manifest
    that only lives there proves nothing to someone who cloned the repo."""
    repo = _write_fake_repo(tmp_path / "repo")
    outdir = repo / "release_artifacts"
    outdir.mkdir()
    (outdir / "the_oracle-9.9.9.tar.gz").write_bytes(b"sdist bytes")
    (outdir / "the_oracle-9.9.9-py3-none-any.whl").write_bytes(b"wheel bytes")
    artifacts = [outdir / "the_oracle-9.9.9.tar.gz", outdir / "the_oracle-9.9.9-py3-none-any.whl"]
    manifest = release.write_checksums(outdir, "9.9.9", artifacts)

    tracked = release.store_release_checksums(repo, "9.9.9", manifest)

    assert tracked == repo / "release_checksums" / "checksums-9.9.9.sha256"
    # Byte-identical, so `sha256sum -c` reads the same manifest either way.
    assert tracked.read_bytes() == manifest.read_bytes()

    # The real workflow: both files downloaded into an unrelated folder,
    # verified against the manifest that came with the repository.
    if not shutil.which("sha256sum"):
        return
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for artifact in artifacts:
        shutil.copy2(artifact, downloads / artifact.name)
    subprocess.run(["sha256sum", "-c", str(tracked)], cwd=str(downloads), check=True)


def test_store_release_checksums_is_idempotent_for_identical_content(tmp_path: Path) -> None:
    repo = _write_fake_repo(tmp_path)
    outdir = tmp_path / "release_artifacts"
    outdir.mkdir()
    artifact = outdir / "the_oracle-9.9.9.tar.gz"
    artifact.write_bytes(b"same bytes")
    manifest = release.write_checksums(outdir, "9.9.9", [artifact])

    first = release.store_release_checksums(repo, "9.9.9", manifest)
    recorded = first.read_bytes()
    second = release.store_release_checksums(repo, "9.9.9", manifest)

    assert first == second
    assert second.read_bytes() == recorded


def test_store_release_checksums_refuses_to_rewrite_a_recorded_manifest(tmp_path: Path) -> None:
    """Changed hashes for an already-published version are exactly what a
    checksum manifest exists to make impossible to do unnoticed."""
    repo = _write_fake_repo(tmp_path)
    tracked_dir = repo / "release_checksums"
    tracked_dir.mkdir()
    published = tracked_dir / "checksums-9.9.9.sha256"
    published.write_text("deadbeef  the_oracle-9.9.9.tar.gz\n", encoding="utf-8")

    outdir = tmp_path / "release_artifacts"
    outdir.mkdir()
    artifact = outdir / "the_oracle-9.9.9.tar.gz"
    artifact.write_bytes(b"rebuilt from different inputs")
    manifest = release.write_checksums(outdir, "9.9.9", [artifact])
    assert manifest.read_bytes() != published.read_bytes()

    with pytest.raises(release.ReleaseError, match="must not be rewritten"):
        release.store_release_checksums(repo, "9.9.9", manifest)

    # The refusal leaves the recorded manifest exactly as it was.
    assert published.read_text(encoding="utf-8") == "deadbeef  the_oracle-9.9.9.tar.gz\n"


def test_the_tracked_checksums_path_is_not_gitignored() -> None:
    """The feature's premise: this directory has to be committable.

    A broad ignore rule would quietly reduce the tracked copy to a local file
    that never reaches git — the release would look successful while the
    hashes stayed as unverifiable as before. (``release_artifacts/**/*.sha256``
    covers the build folder; this is the other half.)
    """
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    path = f"{release.RELEASE_CHECKSUMS_DIR}/checksums-9.9.9.sha256"
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", path],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    # check-ignore exits 1 when nothing matched, 0 when the path is ignored.
    assert result.returncode == 1, (
        f"{path} is gitignored, so a release's recorded hashes would never "
        "reach the repository"
    )


def test_release_mode_records_the_tracked_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tail of the release path: build -> outdir manifest -> tracked copy."""
    repo = _write_fake_repo(tmp_path)
    outdir = repo / "release_artifacts"
    monkeypatch.setattr(release, "tree_is_dirty", lambda _repo: False)
    monkeypatch.setattr(release, "run_tests", lambda _repo: None)

    def _fake_build(_repo_root, out, version):
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name, payload in (
            (f"the_oracle-{version}.tar.gz", b"sdist"),
            (f"the_oracle-{version}-py3-none-any.whl", b"wheel"),
        ):
            path = out / name
            path.write_bytes(payload)
            written.append(path)
        return written

    monkeypatch.setattr(release, "build_artifacts", _fake_build)

    status = release.main(["--repo-root", str(repo)])

    assert status == 0
    tracked = repo / "release_checksums" / "checksums-9.9.9.sha256"
    assert tracked.is_file()
    assert tracked.read_bytes() == (outdir / "checksums-9.9.9.sha256").read_bytes()
