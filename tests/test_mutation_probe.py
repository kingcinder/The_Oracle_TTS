"""The mutation probe: apply, run, revert — hash-guarded, bytecode isolated.

The trap this file exists for is stdlib behavior rather than a repo defect
(STATE.md, 2026-10-10): CPython keys cached bytecode on **(source mtime in whole
seconds, source size)**, so bytes rewritten to the same length inside the same
second can leave the ``.pyc`` looking current. A run then executes bytecode that
matches no file on disk while every hash a tool can read says clean. The first
test REPRODUCES that collision on purpose, in both directions; the rest pin the
guards that stop the probe from being lied to or from lying about its own
restore — "byte-identical" is measured here, never assumed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "scripts" / "mutation_probe.py"

#: Same length on purpose: the trap needs the size to match, so these two are
#: what a fast mutate/revert cycle looks like to the import system.
BEFORE = 'def value():\n    return "old"\n'
AFTER = 'def value():\n    return "new"\n'


@pytest.fixture(scope="module")
def probe_module():
    """The harness, imported by path — the idiom this repo's tests use."""
    spec = importlib.util.spec_from_file_location("oracle_mutation_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The harness uses @dataclass, whose decorator looks the module up in
    # sys.modules — register before exec or the load itself fails.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _default_cached_pyc(source: Path) -> Path:
    """Where a child with NO ``PYTHONPYCACHEPREFIX`` writes *source*'s bytecode.

    Computed rather than asked for: ``importlib.util.cache_from_source`` honors
    the CALLING process's ``sys.pycache_prefix``, so under the probe — whose
    pytest child runs with a prefix — it names the isolated root instead of the
    repo-side cache the naive children write. That mismatch made the first draft
    of these tests pass for the wrong reason, which is the exact class of
    mistake this file exists to prevent, so the path is derived from the cache
    tag and the source's own directory.
    """
    tag = sys.implementation.cache_tag or "cpython"
    cache = source.parent / "__pycache__" / f"{source.stem}.{tag}.pyc"
    assert cache.is_file(), f"no bytecode cache was written at {cache}"
    return cache


def _cached_key(cache: Path) -> tuple[int, int]:
    """The (source mtime second, source size) the cache recorded."""
    _magic, _flags, mtime, size = struct.unpack("<4sIII", cache.read_bytes()[:16])
    return mtime, size


def _rewrite_cache_key(cache: Path, mtime: int, size: int) -> None:
    """Re-key a cached ``.pyc`` to describe a source it does not contain.

    The import system compares only the header, so this is the collision in its
    purest form: bytecode from one revision, a key that matches another.
    """
    raw = bytearray(cache.read_bytes())
    magic, flags = struct.unpack("<4sI", bytes(raw[:8]))
    raw[:16] = struct.pack("<4sIII", magic, flags, mtime & 0xFFFFFFFF, size & 0xFFFFFFFF)
    cache.write_bytes(bytes(raw))


def _value_child(cwd: Path, *, isolate: Path | None) -> str:
    """Import ``toy`` in a fresh interpreter and print ``toy.value()``.

    ``isolate=None`` leaves the child on the default cache beside the source —
    the state the trap needs. An isolated child is pointed at *isolate*.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX"}
    }
    if isolate is not None:
        env["PYTHONPYCACHEPREFIX"] = str(isolate)
    proc = subprocess.run(
        [sys.executable, "-c", "import toy\nprint(toy.value())"],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_the_same_second_trap_is_real_in_both_directions(tmp_path: Path) -> None:
    """Reproduce the collision that made a recorded mutation run lie.

    Direction 1: the cache holds the OLD function while the file on disk has the
    NEW one, so a probe would look inert — "the check does not catch it" — when
    the check was never shown the mutation. Direction 2 is the incident as
    recorded: every hash on disk says clean, and the interpreter still runs the
    bytes that were reverted. Both are constructed by hand here, because the
    harness's tests then have a state to prove immunity against.
    """
    source = tmp_path / "toy.py"

    # Direction 1: mutate-looking-inert.
    source.write_text(BEFORE, encoding="utf-8")
    assert _value_child(tmp_path, isolate=None) == "old"  # writes the cache
    cache = _default_cached_pyc(source)
    recorded_mtime, recorded_size = _cached_key(cache)
    source.write_text(AFTER, encoding="utf-8")
    assert source.stat().st_size == recorded_size, "the collision needs the same size"
    os.utime(source, (recorded_mtime, recorded_mtime))
    assert source.read_text(encoding="utf-8") == AFTER
    assert _value_child(tmp_path, isolate=None) == "old", "the trap did not reproduce"

    # Direction 2: reverted-looking-clean.
    cache.unlink()
    assert _value_child(tmp_path, isolate=None) == "new"  # recompiles AFTER
    cache = _default_cached_pyc(source)
    recorded_mtime, recorded_size = _cached_key(cache)
    source.write_text(BEFORE, encoding="utf-8")
    assert source.stat().st_size == recorded_size
    os.utime(source, (recorded_mtime, recorded_mtime))
    assert source.read_text(encoding="utf-8") == BEFORE
    assert _value_child(tmp_path, isolate=None) == "new", "the trap did not reproduce"

    # The state that makes the lie possible is exactly a cache the import system
    # considers current — so it is also what the harness refuses to depend on.
    assert source.read_text(encoding="utf-8") == BEFORE
    assert _value_child(tmp_path, isolate=tmp_path / "fresh-cache") == "old", (
        "a child with its own cache root must read the source instead"
    )


def test_forget_hazardous_bytecode_drops_only_a_matching_cache(
    probe_module, tmp_path: Path
) -> None:
    """The safety net for a caller that checks the mutation from its own process."""
    source = tmp_path / "toy.py"
    source.write_text(BEFORE, encoding="utf-8")
    assert _value_child(tmp_path, isolate=None) == "old"
    cache = _default_cached_pyc(source)

    os.utime(source, (1, 1))  # the source moved on: the cache is legitimately stale
    assert probe_module.cached_bytecode_is_current(source) is None
    assert probe_module.forget_hazardous_bytecode(source) == []
    assert cache.exists(), "a genuinely stale cache is not the hazard this drops"

    recorded_mtime, _size = _cached_key(cache)
    os.utime(source, (recorded_mtime, recorded_mtime))
    assert probe_module.cached_bytecode_is_current(source) == cache
    hazards = probe_module.forget_hazardous_bytecode(source)
    assert len(hazards) == 1 and cache.name in hazards[0], hazards
    assert not cache.exists()


def test_the_probe_runs_the_bytes_on_disk_and_restores_them(probe_module, tmp_path: Path) -> None:
    """With a lying cache in place, the probe still sees the mutation — then restores.

    The cache is left holding bytes the file no longer contains (direction 2 of
    the trap, still current-looking). A naive child runs those bytes; the probe's
    child gets a fresh cache root per run and executes the source. The revert is
    then measured, not assumed: the digests have to match the original exactly.
    """
    source = tmp_path / "toy.py"
    source.write_text(AFTER, encoding="utf-8")
    assert _value_child(tmp_path, isolate=None) == "new"
    cache = _default_cached_pyc(source)
    recorded_mtime, recorded_size = _cached_key(cache)
    source.write_text(BEFORE, encoding="utf-8")
    os.utime(source, (recorded_mtime, recorded_mtime))
    assert source.stat().st_size == recorded_size
    assert probe_module.cached_bytecode_is_current(source) is not None
    assert _value_child(tmp_path, isolate=None) == "new", "the lying cache is gone"

    original = source.read_bytes()
    repo_side = {path.name for path in cache.parent.iterdir()}
    probe = probe_module.Probe([probe_module.Mutation(source, "old", "theirs")])
    probe.apply()
    try:
        # Re-key the stale cache to the source as it is NOW — the mutated bytes,
        # the cache's own older bytes, one matching header. Re-keyed AFTER the
        # apply, so the mutation cannot quietly invalidate it by changing the
        # size: the collision has to be live for the run below to mean anything.
        stat = source.stat()
        _rewrite_cache_key(cache, int(stat.st_mtime), stat.st_size)
        assert probe_module.cached_bytecode_is_current(source) is not None, (
            "the collision is not live, so the run below would prove nothing"
        )
        assert _value_child(tmp_path, isolate=None) == "new", "a naive child is not fooled"

        proc = probe.run(
            [sys.executable, "-c", "import toy\nprint(toy.value())"], cwd=tmp_path
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "theirs", "the probe must run the bytes on disk"
        assert {path.name for path in cache.parent.iterdir()} == repo_side, (
            "the probe's child wrote bytecode next to the source it imported"
        )
    finally:
        probe.revert()
        probe.close()

    assert source.read_bytes() == original
    record = probe.records[0]
    assert record.byte_identical, probe.report()
    assert record.restored_sha256 == probe_module.digest(original)
    assert "byte-identical" in probe.report()


def test_the_guards_refuse_an_ambiguous_or_unmeasured_mutation(
    probe_module, tmp_path: Path
) -> None:
    """Every refusal happens before the file is touched, and a partial apply rolls back."""
    source = tmp_path / "toy.py"
    source.write_text("one = 1\none = 1\n", encoding="utf-8")
    before = probe_module.digest(source.read_bytes())

    cases = [
        (probe_module.Mutation(source, "one = 1", "one = 2"), "appears 2 time(s)"),
        (probe_module.Mutation(source, "missing", "x"), "appears 0 time(s)"),
        (
            probe_module.Mutation(source, "one = 1", "one = 2", expect_sha256="0" * 64),
            "is not the expected",
        ),
        (
            probe_module.Mutation(source, "one = 1", "one = 1", count=2),
            "does not change the file's bytes",
        ),
    ]
    for mutation, fragment in cases:
        with pytest.raises(probe_module.GuardRefused) as caught:
            probe_module.Probe([mutation]).apply()
        assert fragment in str(caught.value), caught.value
        assert probe_module.digest(source.read_bytes()) == before, (
            f"a refused spec changed the file: {fragment}"
        )

    # A spec whose FIRST mutation applies and whose second refuses must leave the
    # tree exactly as it was — a probe that dies half-applied is the one state
    # this harness must never create.
    good = probe_module.Mutation(source, "one = 1", "one = 2", count=1)
    ambiguous = probe_module.Mutation(source, "one = 1", "one = 2")
    with pytest.raises(probe_module.GuardRefused):
        probe_module.Probe([good, ambiguous]).apply()
    assert probe_module.digest(source.read_bytes()) == before


def test_a_foreign_edit_is_refused_and_left_exactly_as_found(probe_module, tmp_path: Path) -> None:
    """A shared checkout: another agent's edit during a probe is never clobbered."""
    source = tmp_path / "toy.py"
    source.write_text(BEFORE, encoding="utf-8")
    probe = probe_module.Probe([probe_module.Mutation(source, "old", "new")])
    probe.apply()
    foreign = 'def value():\n    return "theirs"\n'
    source.write_text(foreign, encoding="utf-8")

    with pytest.raises(probe_module.GuardRefused) as caught:
        probe.revert()
    message = str(caught.value)
    assert str(source) in message and "changed while the probe ran" in message
    assert source.read_text(encoding="utf-8") == foreign, "the foreign edit was overwritten"
    record = probe.records[0]
    assert record.conflict is not None
    assert record.byte_identical is False
    assert "NOT byte-identical" in probe.report()
    probe.close()


def test_each_run_gets_a_fresh_cache_root_under_a_reaped_scratch(
    probe_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolation per run, and the /tmp discipline the certifier family already holds."""
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))

    probe = probe_module.Probe([])
    runs = [
        probe.run([sys.executable, "-c", "import sys; print(sys.pycache_prefix)"]),
        probe.run([sys.executable, "-c", "import sys; print(sys.pycache_prefix)"]),
    ]
    roots = [Path(run.stdout.strip()) for run in runs]
    assert roots[0] != roots[1], "every run needs its own cache root"
    for root in roots:
        assert root.is_absolute() and root.name.startswith("pycache-"), root
        assert not root.exists(), "a per-run cache root is reaped when the run ends"

    source_dir = tmp_path / "isolated"
    source_dir.mkdir()
    (source_dir / "toy.py").write_text('VALUE = "x"\n', encoding="utf-8")
    code = f"import sys; sys.path.insert(0, {str(source_dir)!r}); import toy; print(toy.VALUE)"
    imported = probe.run([sys.executable, "-c", code])
    assert imported.stdout.strip() == "x", imported.stderr
    assert not (source_dir / "__pycache__").exists(), (
        "the probe's child wrote bytecode next to the source it imported"
    )

    scratch = probe.scratch
    assert scratch.name.startswith("oracle_mutation_probe_") and scratch.exists()
    probe.close()
    assert not scratch.exists(), "close() reaps the root the probe created"
    assert sorted(path.name for path in temp.iterdir()) == [], "the scratch parent is empty"


def test_the_cli_verdicts_a_caught_and_a_surviving_mutation(
    probe_module, tmp_path: Path
) -> None:
    """The executable contract: exit 0 caught, exit 1 survived, bytes untouched."""
    source = tmp_path / "toy.py"
    source.write_text('VALUE = "old"\nNOTE = "aaaa"\n', encoding="utf-8")
    (tmp_path / "check.py").write_text(
        'import toy\nraise SystemExit(0 if toy.VALUE == "old" else 1)\n', encoding="utf-8"
    )
    before = probe_module.digest(source.read_bytes())
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    caught = _run_cli(
        tmp_path,
        scratch,
        [{"path": str(source), "old": 'VALUE = "old"', "new": 'VALUE = "new"'}],
    )
    assert caught.returncode == 0, caught.stdout + caught.stderr
    assert "CAUGHT" in caught.stdout, caught.stdout
    assert probe_module.digest(source.read_bytes()) == before, "the CLI left the mutation applied"
    assert "byte-identical" in caught.stdout

    survived = _run_cli(
        tmp_path,
        scratch,
        [{"path": str(source), "old": 'NOTE = "aaaa"', "new": 'NOTE = "bbbb"'}],
    )
    assert survived.returncode == 1, survived.stdout + survived.stderr
    assert "SURVIVED" in survived.stdout, survived.stdout
    assert probe_module.digest(source.read_bytes()) == before

    document = _run_cli(
        tmp_path,
        scratch,
        [{"path": str(source), "old": 'VALUE = "old"', "new": 'VALUE = "new"'}],
        extra=["--json"],
    )
    payload = json.loads(document.stdout)
    assert payload["verdict"] == "caught" and payload["exit_code"] == 0
    assert payload["mutations"][0]["byte_identical"] is True
    assert payload["runs"][0]["returncode"] == 1

    assert sorted(path.name for path in scratch.iterdir()) == [], "the CLI left scratch behind"


def test_a_collection_error_is_not_a_caught_mutation(probe_module, tmp_path: Path) -> None:
    """pytest exits 1 for a failing test and 4 for a collection error.

    Only the first means the check saw the mutation. Asking for 1 and getting 4
    has to come back as "nothing concluded", not "the check does not cover this
    line" — the same class of lie as reading stale bytecode, and the shape a
    broken import in the mutated file produces (found while driving the harness
    through a real theme mutation, where the mutated module refused to import).
    """
    source = tmp_path / "toy.py"
    source.write_text('VALUE = "old"\n', encoding="utf-8")
    before = probe_module.digest(source.read_bytes())
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_cli(
        tmp_path,
        scratch,
        [{"path": str(source), "old": "old", "new": "new"}],
        command=[sys.executable, "-m", "pytest", "-q", "tests/does_not_exist.py"],
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "INVALID" in result.stdout and "not the expected 1" in result.stdout, result.stdout
    assert probe_module.digest(source.read_bytes()) == before
    assert sorted(path.name for path in scratch.iterdir()) == []


def _run_cli(
    cwd: Path,
    scratch: Path,
    spec: list[dict],
    *,
    expect_rc: int = 1,
    command: list[str] | None = None,
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Invoke the harness exactly as a developer would, and hold it to ``expect_rc``."""
    (cwd / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    invocation = [
        sys.executable,
        str(PROBE_PATH),
        "--spec",
        str(cwd / "spec.json"),
        "--expect-rc",
        str(expect_rc),
        "--timeout",
        "120",
        *(extra or []),
        "--",
        *(command or [sys.executable, "check.py"]),
    ]
    return subprocess.run(
        invocation,
        cwd=str(cwd),
        env={**os.environ, "TMPDIR": str(scratch)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
