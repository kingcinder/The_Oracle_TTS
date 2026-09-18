"""A doctor run must not write outside the one directory it declares.

Two gate regressions came from checks grading state they had created: reference
clips written into ``build/real_engine_smoke/inputs`` were counted by the
voice-source audit, so run 1 and run 2 disagreed on an unchanged machine, and the
real-engine readiness PASS cited a smoke FLAC it had never produced. Comparing two
reports catches that after the fact. This catches the write itself -- and does so
for every check in the doctor, including checks added later, without anyone having
to think about it.

The comparison is a whole-tree content snapshot either side of a real run, taken
with the real ``doctor.py`` in the mode CI gates on. The single exception is the
deterministic smoke's own project directory: that check renders a real project and
verifies the artifact it produced, so its output there is its subject rather than a
side effect of grading. Everything else -- source, tests, scripts, docs, and every
git-ignored artifact directory a check might be tempted to cache into -- must be
byte-identical afterwards.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import repo_tree_snapshot, snapshot_differences

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"

#: The only place a doctor run may add, change or remove anything. Taken from a
#: measured run rather than assumed: with this directory excluded, a full
#: `--skip-model-init --ci` doctor run leaves a 7000-file tree untouched.
WRITABLE_BY_A_DOCTOR_RUN = "build/doctor_deterministic_smoke/"

#: Where the test parks an existing `build/` so the run starts from the state a
#: fresh clone is in. A rename, not a copy, so it is instant even though this tree
#: carries gigabytes; and the name says what it is if a killed test leaves it.
PARKED_BUILD = REPO_ROOT / "build.__readonly_test_parked"


def _outside_writable(paths: list[str]) -> list[str]:
    return [path for path in paths if not path.startswith(WRITABLE_BY_A_DOCTOR_RUN)]


def _describe(created: list[str], changed: list[str], removed: list[str]) -> str:
    lines = []
    for label, paths in (("created", created), ("changed", changed), ("removed", removed)):
        if paths:
            lines.append(f"  {label}:")
            lines.extend(f"    {path}" for path in paths)
    return "\n".join(lines)


# --- the guard itself ---------------------------------------------------------


def test_snapshot_includes_files_git_would_ignore(tmp_path: Path) -> None:
    """The regressions lived in git-ignored space, so ignoring it must not hide them."""
    build = tmp_path / "build" / "real_engine_smoke" / "inputs"
    build.mkdir(parents=True)
    (build / "speaker_a_ref.wav").write_bytes(b"not really audio")
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")

    snapshot = repo_tree_snapshot(tmp_path)

    assert "build/real_engine_smoke/inputs/speaker_a_ref.wav" in snapshot


def test_snapshot_excludes_vcs_the_venv_and_caches(tmp_path: Path) -> None:
    for directory in (".git", ".venv", "__pycache__", ".pytest_cache", ".freebuff"):
        nested = tmp_path / directory
        nested.mkdir()
        (nested / "noise.bin").write_bytes(b"x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("content\n", encoding="utf-8")

    snapshot = repo_tree_snapshot(tmp_path)

    assert list(snapshot) == ["src/real.py"]


def test_snapshot_differences_reports_creation_change_and_removal(tmp_path: Path) -> None:
    (tmp_path / "kept.txt").write_text("same\n", encoding="utf-8")
    (tmp_path / "edited.txt").write_text("before\n", encoding="utf-8")
    (tmp_path / "deleted.txt").write_text("gone\n", encoding="utf-8")
    before = repo_tree_snapshot(tmp_path)

    (tmp_path / "created.txt").write_text("new\n", encoding="utf-8")
    (tmp_path / "edited.txt").write_text("after\n", encoding="utf-8")
    (tmp_path / "deleted.txt").unlink()
    after = repo_tree_snapshot(tmp_path)

    assert snapshot_differences(before, after) == (
        ["created.txt"],
        ["edited.txt"],
        ["deleted.txt"],
    )


def test_snapshot_detects_a_rewrite_that_keeps_the_size(tmp_path: Path) -> None:
    """A same-length rewrite is the subtle case: only the content hash sees it."""
    target = tmp_path / "plan.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")
    before = repo_tree_snapshot(tmp_path)

    target.write_text('{"a": 2}\n', encoding="utf-8")

    created, changed, removed = snapshot_differences(before, repo_tree_snapshot(tmp_path))
    assert (created, changed, removed) == ([], ["plan.json"], [])


def test_parking_moves_the_build_tree_and_puts_it_back(monkeypatch, tmp_path: Path) -> None:
    """Without this, the check below is blind on any machine that has run before."""
    module = sys.modules[__name__]
    build = tmp_path / "build"
    parked_path = tmp_path / "build.__readonly_test_parked"
    (build / "real_engine_smoke" / "inputs").mkdir(parents=True)
    (build / "real_engine_smoke" / "inputs" / "speaker_a_ref.wav").write_bytes(b"existing")
    monkeypatch.setattr(module, "BUILD_DIR", build)
    monkeypatch.setattr(module, "PARKED_BUILD", parked_path)

    parked = _park_build_tree()

    assert parked == parked_path
    assert not build.exists()
    assert (parked_path / "real_engine_smoke" / "inputs" / "speaker_a_ref.wav").read_bytes() == b"existing"

    # Something the doctor generated while the tree was parked.
    (build / "doctor_deterministic_smoke").mkdir(parents=True)
    _restore_build_tree(parked)

    assert not parked_path.exists()
    assert (build / "real_engine_smoke" / "inputs" / "speaker_a_ref.wav").read_bytes() == b"existing"
    assert not (build / "doctor_deterministic_smoke").exists()


def test_parking_is_a_no_op_without_a_build_tree(monkeypatch, tmp_path: Path) -> None:
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "BUILD_DIR", tmp_path / "absent")
    monkeypatch.setattr(module, "PARKED_BUILD", tmp_path / "parked")

    assert _park_build_tree() is None
    _restore_build_tree(None)  # a fresh clone keeps what the run produced


def test_snapshot_of_this_repository_is_not_vacuous() -> None:
    """An empty or truncated snapshot would make the check below pass for free."""
    snapshot = repo_tree_snapshot(REPO_ROOT)

    assert len(snapshot) > 100
    assert "src/the_oracle/cli.py" in snapshot
    assert "scripts/doctor.py" in snapshot


# --- the property -------------------------------------------------------------


def _park_build_tree() -> Path | None:
    """Move an existing ``build/`` aside, so the run starts as a fresh clone does.

    This is what makes the check able to see the defect at all: a check that writes
    its artifacts only when they are missing writes nothing on a machine that
    already has them, so the write -- and the wrong count it causes -- is invisible
    there. The write is real on the machine that does not have them yet, which is a
    fresh clone and every CI runner, and that is the state to test.
    """
    if not BUILD_DIR.exists():
        return None
    if PARKED_BUILD.exists():  # left behind by an interrupted run
        shutil.rmtree(PARKED_BUILD)
    BUILD_DIR.rename(PARKED_BUILD)
    return PARKED_BUILD


def _restore_build_tree(parked: Path | None) -> None:
    """Put the parked tree back, discarding whatever the run generated.

    The generated output is git-ignored and regenerated on demand, and leaving the
    working tree exactly as this test found it is what keeps a guard measure from
    being a change of its own.
    """
    if parked is None:
        return
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    parked.rename(BUILD_DIR)


@pytest.mark.slow
def test_a_doctor_run_writes_nothing_outside_its_declared_output() -> None:
    """Two runs, starting from a tree with no generated artifacts at all.

    The first run is the fresh-clone case, where a check that writes its own
    artifacts has to do so; the second starts from the tree the first settled, which
    is where the historical defect did its damage -- the write happened on run 1 and
    was *counted* on run 2.
    """
    parked = _park_build_tree()
    try:
        _assert_no_writes_outside_the_declared_output()
    finally:
        _restore_build_tree(parked)


def _assert_no_writes_outside_the_declared_output() -> None:
    for attempt in range(2):
        before = repo_tree_snapshot(REPO_ROOT)

        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "doctor.py"),
                "--repo-root",
                str(REPO_ROOT),
                "--skip-model-init",
                "--ci",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )

        created, changed, removed = snapshot_differences(before, repo_tree_snapshot(REPO_ROOT))
        offending = _outside_writable(created), _outside_writable(changed), _outside_writable(removed)
        detail = _describe(*offending)

        # The doctor must reach a verdict rather than crashing; whether that verdict
        # is "ready" depends on the machine, which is not this test's business.
        assert completed.returncode in (0, 1), f"doctor exited {completed.returncode}: {completed.stderr[-2000:]}"
        assert not any(offending), (
            f"run {attempt + 1} of the doctor touched files outside its own smoke output, so a"
            f" check is writing state instead of only reading it:\n{detail}"
        )
