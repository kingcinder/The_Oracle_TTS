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

import ast
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


# --- the preset path ----------------------------------------------------------


def _preset_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the harness with *args*, as a developer would from the repo root."""
    return subprocess.run(
        [sys.executable, str(PROBE_PATH), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_the_certifier_preset_still_describes_this_tree(probe_module) -> None:
    """The recorded certifier mutation must match the tree it is run against.

    A preset is a hand run written down, so the only way it can rot silently is
    the tree moving underneath it: the Live header's gate site edited (the
    anchor then occurs 0 or 2 times and the tool's count guard refuses — good,
    but a red the operator should be able to read as "re-record me" rather than
    as a broken tool), or the site reformatted so the recorded replacement no
    longer produces the recorded ungated statement, or the proof command
    renamed out from under it. The evidence fields are pinned too: the recorded
    run exited 1 on six CLOSED failures, and a looser needle would call any
    breakage a catch — exactly the lie ``--require-stdout`` exists to prevent.

    This does NOT run the sweep (six real windows, ~19 s): the live proof is a
    deliberate act, recorded in the commit message. What is pinned here is that
    the act would still be possible.
    """
    preset = probe_module.PRESETS["certifier-gate-live-header"]
    assert preset.expect_rc == 1, (
        "the recorded certifier run exited 1 — re-measure the sweep, do not relax "
        "the expectation"
    )
    assert preset.require_stdout == "bulk-restyle gate CLOSED", (
        "the recorded evidence was six CLOSED failures; a looser needle would report "
        "CAUGHT for a sweep that broke some other way"
    )
    script = probe_module.REPO_ROOT / preset.command[1]
    assert script.is_file(), f"the proof command names a script that is not there: {script}"
    assert list(preset.command[2:]) == ["--aspects", "chrome"], preset.command

    [mutation] = preset.resolve()
    assert mutation.path.is_file(), f"{mutation.path} is gone — re-record the preset"
    text = mutation.path.read_text(encoding="utf-8")
    assert text.count(mutation.old) == mutation.count, (
        f"the Live header gate site no longer occurs exactly {mutation.count} time(s) in "
        f"{mutation.path.name}: someone edited the site, so the preset must be "
        "re-recorded rather than left to miss"
    )
    mutated = text.replace(mutation.old, mutation.new)
    assert mutated != text, "the replacement changes nothing — it is not a mutation"
    # A mutation that cannot import exits 4 (a collection error), which the
    # verdict logic already refuses to call caught; failing here says WHY.
    ast.parse(mutated)
    header_lines = [line for line in mutated.splitlines() if "header.setStyleSheet(" in line]
    assert header_lines == mutation.new.rstrip("\n").splitlines(), (
        "the mutation no longer produces the recorded ungated statement "
        f"(got {header_lines!r}) — re-record the preset"
    )


# --- the certifier hermeticity presets (M1-M4) --------------------------------

#: The four mutations ``tests/test_certifier_hermeticity.py``'s module docstring
#: records as proven-live-and-reverted, each with the net that caught it and the
#: needle that proves THAT net fired rather than some other breakage.
#:
#: M2's proof deliberately uses the isolation pin instead of the docstring's
#: behavioral half: with the redirect gone, the behavioral run rewrites the
#: developer's REAL app settings six times (the exact damage the redirect
#: exists to stop), which makes it a destructive replay. The isolation pin
#: asserts the same redirect through the real reader, side-effect free, and its
#: own comment says it was found by this very mutation.
_HERMETICITY_PRESETS = {
    "certifier-hermeticity-paths-redirect": {
        "file": "scripts/certify_gui.py",
        "needle": "gui_startup_timing.json",
        "proof": (
            "tests/test_certifier_hermeticity.py::"
            "test_sweep_never_touches_the_real_repo_or_config"
        ),
        "gone": "_app_gui.ensure_repo_default_paths = _hermetic_paths",
    },
    "certifier-hermeticity-config-redirect": {
        "file": "scripts/certifier_isolation.py",
        "needle": "the helper no longer redirects both config variables",
        "proof": (
            "tests/test_certifier_isolation.py::"
            "test_the_helper_redirects_both_config_variables"
        ),
        "gone": 'THROWAWAY_ENV_VARS = ("XDG_CONFIG_HOME", "APPDATA")',
    },
    "certifier-hermeticity-crash-root": {
        "file": "scripts/certify_gui.py",
        "needle": "crash_handlers.install(root=_CONFIG_ROOT)",
        "proof": (
            "tests/test_certifier_hermeticity.py::test_crash_channel_stays_in_the_throwaway"
        ),
        "gone": "crash_handlers.install(root=_CONFIG_ROOT)",
    },
    "certifier-hermeticity-log-entry-points": {
        "file": "scripts/certify_gui.py",
        "needle": "the sweep must not call configure_logging(",
        "proof": "tests/test_certifier_hermeticity.py::test_no_repo_log_entry_points",
        "added": "configure_logging(",
    },
}


@pytest.mark.parametrize("name", sorted(_HERMETICITY_PRESETS))
def test_the_hermeticity_presets_still_describe_this_tree(probe_module, name: str) -> None:
    """Each recorded hermeticity mutation must still be applicable and still be seen.

    The same rot risk as the live-header preset, multiplied by four: the anchor
    edited (the count guard refuses, which must read as "re-record me"), the
    proof command renamed, or the needle loosened until any breakage counts as a
    catch. The needle is pinned to the exact assertion message the net prints,
    which is the whole difference between "the mutation was seen" and "something
    else broke".

    Like the live-header pin, this does NOT run the proof command (M1's is the
    full ~20 s sweep): what is pinned is that the act would still be possible.
    The live runs are recorded in each preset's evidence line.
    """
    expected = _HERMETICITY_PRESETS[name]
    preset = probe_module.PRESETS[name]

    assert preset.expect_rc == 1, (
        "a caught mutation fails one named test, so pytest exits 1 — re-measure, "
        "do not relax"
    )
    assert preset.require_stdout == expected["needle"], (
        "the needle must be the exact message the catching net prints; a looser one "
        "would report CAUGHT for a mutation caught some other way"
    )
    assert preset.evidence, "a preset is a hand run written down — say where it was run"
    proof = list(preset.command)
    assert proof[1:3] == ["-m", "pytest"], proof
    assert expected["proof"] in proof, (
        f"the preset must run {expected['proof']} — the net its evidence names"
    )

    script = probe_module.REPO_ROOT / expected["file"]
    [mutation] = preset.resolve()
    assert mutation.path == script, f"{name} mutates {mutation.path}, expected {script}"
    assert mutation.path.is_file(), f"{mutation.path} is gone — re-record the preset"
    text = mutation.path.read_text(encoding="utf-8")
    assert text.count(mutation.old) == mutation.count, (
        f"the mutation site no longer occurs exactly {mutation.count} time(s) in "
        f"{mutation.path.name}: re-record {name} rather than leave it to miss"
    )
    mutated = text.replace(mutation.old, mutation.new)
    assert mutated != text, "the replacement changes nothing — it is not a mutation"
    # A mutation that cannot import exits 4 (a collection error), which the
    # verdict logic refuses to call caught; failing here says WHY.
    ast.parse(mutated)
    if "gone" in expected:
        assert expected["gone"] not in mutated, (
            "the mutation no longer removes the line it claims to remove — re-record it"
        )
    if "added" in expected:
        assert expected["added"] in mutated, (
            "the mutation no longer introduces the forbidden text it claims to — "
            "re-record it"
        )


def test_the_cli_lists_presets_and_refuses_the_ambiguous_invocations(probe_module) -> None:
    """The catalog is reachable and every half-specified run is a refusal.

    Refusals are exit 2 — "nothing concluded" — because a probe that guessed
    between ``--spec`` and ``--preset``, or invented a preset name, would run a
    mutation nobody asked for.
    """
    listed = _preset_cli("--list-presets")
    assert listed.returncode == 0, listed.stderr
    assert "certifier-gate-live-header" in listed.stdout, listed.stdout
    assert "bulk-restyle gate CLOSED" in listed.stdout, listed.stdout
    assert "--aspects chrome" in listed.stdout, listed.stdout

    unknown = _preset_cli("--preset", "not-a-preset")
    assert unknown.returncode == 2, unknown.stdout
    assert "not-a-preset" in unknown.stderr and "certifier-gate-live-header" in unknown.stderr, (
        unknown.stderr
    )

    neither = _preset_cli()
    assert neither.returncode == 2 and "--preset" in neither.stderr, neither.stderr

    both = _preset_cli(
        "--spec", "spec.json", "--preset", "certifier-gate-live-header"
    )
    assert both.returncode == 2 and "alternatives" in both.stderr, both.stderr


def test_a_preset_run_needs_the_evidence_and_not_just_the_status(
    probe_module, tmp_path: Path, monkeypatch, capsys
) -> None:
    """The three verdicts a preset can reach, driven through ``main()``.

    A fake preset on a temp file stands in for the real one so the suite never
    mutates ``gui_chrome.py`` or runs six real windows; everything else —
    preset resolution, the preset's own command being used when no check is
    passed after ``--``, its ``expect_rc`` and its needle — is the production
    path. The middle case is the point: status right AND evidence present is
    CAUGHT, status right with no evidence is INVALID (exit 2), and a check that
    exits 0 stays SURVIVED however loudly it printed.
    """
    source = tmp_path / "toy.py"
    source.write_text('VALUE = "old"\n', encoding="utf-8")
    before = probe_module.digest(source.read_bytes())
    mutation = probe_module.Mutation(path=source, old="old", new="new", count=1)

    def preset_running(child: str) -> object:
        return probe_module.Preset(
            name="fake",
            what="a temp-file mutation used to pin the verdict shapes",
            mutations=(mutation,),
            command=(sys.executable, "-c", child),
            expect_rc=1,
            require_stdout="bulk-restyle gate CLOSED",
        )

    cases = (
        ("raise SystemExit(1)", 2, "INVALID"),
        ("print('bulk-restyle gate CLOSED'); raise SystemExit(1)", 0, "CAUGHT"),
        ("print('bulk-restyle gate CLOSED'); raise SystemExit(0)", 1, "SURVIVED"),
    )
    for child, expected_code, token in cases:
        monkeypatch.setitem(probe_module.PRESETS, "fake", preset_running(child))
        code = probe_module.main(["--preset", "fake"])
        output = capsys.readouterr().out
        assert code == expected_code, output
        assert token in output, output
        assert probe_module.digest(source.read_bytes()) == before, (
            "every verdict has to leave the file byte-identical"
        )


def test_the_stdout_requirement_applies_to_a_spec_run_too(
    probe_module, tmp_path: Path
) -> None:
    """``--require-stdout`` is the flag's own contract, not a preset perk.

    The same check, same mutation, same expected status: adding a needle the
    child never prints turns CAUGHT into INVALID, so the narrowing cannot be
    switched off by reaching for ``--spec`` instead.
    """
    source = tmp_path / "toy.py"
    source.write_text('VALUE = "old"\n', encoding="utf-8")
    (tmp_path / "check.py").write_text(
        'import toy\nraise SystemExit(0 if toy.VALUE == "old" else 1)\n', encoding="utf-8"
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    before = probe_module.digest(source.read_bytes())
    spec = [{"path": str(source), "old": "old", "new": "new"}]

    without = _run_cli(tmp_path, scratch, spec)
    assert without.returncode == 0, without.stdout + without.stderr

    with_needle = _run_cli(
        tmp_path, scratch, spec, extra=["--require-stdout", "never printed"]
    )
    assert with_needle.returncode == 2, with_needle.stdout + with_needle.stderr
    assert "INVALID" in with_needle.stdout and "never printed" in with_needle.stdout, (
        with_needle.stdout
    )
    assert probe_module.digest(source.read_bytes()) == before, (
        "the narrowed verdict still restores the file byte-identical"
    )


def test_several_bites_of_one_file_revert_to_its_preimage(
    probe_module, tmp_path: Path
) -> None:
    """A spec may take more than one bite of a file; the restore is of the FILE.

    Each bite used to overwrite the path's recorded original, so ``revert()``
    put back the MID-CHAIN bytes: the run reported CAUGHT and exited 0 with the
    checkout left half-mutated — found 2026-10-13 driving a two-hunk spec at
    ``.github/workflows/ci.yml``, which is the one state the harness's own
    docstring says it must never create. Pinned here: one record per file, its
    ``original`` the pre-image from before the FIRST bite, its ``mutated`` the
    last one written, and the bytes afterwards the pre-image exactly.
    """
    source = tmp_path / "multi.py"
    source.write_text('A = "one"\nB = "two"\n', encoding="utf-8")
    preimage = source.read_bytes()
    (tmp_path / "check.py").write_text(
        'import multi\nraise SystemExit(0 if (multi.A, multi.B) == ("uno", "dos") else 1)\n',
        encoding="utf-8",
    )

    probe = probe_module.Probe(
        [
            probe_module.Mutation(path=source, old='A = "one"', new='A = "uno"', count=1),
            probe_module.Mutation(path=source, old='B = "two"', new='B = "dos"', count=1),
        ]
    )
    probe.apply()
    try:
        text = source.read_text(encoding="utf-8")
        assert 'A = "uno"' in text and 'B = "dos"' in text, text
        child = probe.run([sys.executable, str(tmp_path / "check.py")], cwd=tmp_path)
        assert child.returncode == 0, child.stdout + child.stderr
    finally:
        probe.revert()
        probe.close()

    assert source.read_bytes() == preimage, "the file must come back to its pre-image"
    [record] = probe.records  # one record per FILE, not one per bite
    assert record.byte_identical, (record.original_sha256, record.restored_sha256)
    assert record.original_sha256 == probe_module.digest(preimage)
    assert record.mutated_sha256 != record.original_sha256


def test_a_run_whose_restore_was_never_verified_never_reports_caught(
    probe_module, tmp_path: Path, monkeypatch, capsys
) -> None:
    """Exit 0 over an un-restored file is the verdict that must not exist.

    ``revert()`` refuses on the failures it can see; this pins the belt under
    it. If no verified restore stands behind a mutated file, ``main()`` reports
    INVALID (exit 2) even when ``revert()`` returned without complaining — the
    shape a silent revert failure produces, which is exactly how the two-bite
    bug above escaped as a CAUGHT.
    """
    source = tmp_path / "toy.py"
    source.write_text('VALUE = "old"\n', encoding="utf-8")
    pristine = source.read_bytes()
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps([{"path": str(source), "old": "old", "new": "new"}]), encoding="utf-8"
    )

    monkeypatch.setattr(probe_module.Probe, "revert", lambda self: None)
    code = probe_module.main(
        [
            "--spec",
            str(spec),
            "--expect-rc",
            "1",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(1)",
        ]
    )
    output = capsys.readouterr().out
    assert code == 2, output
    assert "restore never verified" in output, output
    # The no-op revert really did leave the mutation applied; undo it here.
    source.write_bytes(pristine)


# --- --witness: the check must be green before its verdict can be believed ----


def _toy_preset_for(probe_module, source: Path, child: str) -> object:
    """A fake preset whose mutation is a temp file and whose check is ``child``."""
    return probe_module.Preset(
        name="fake",
        what="a temp-file mutation used to pin the witness shape",
        mutations=(probe_module.Mutation(path=source, old="old", new="new", count=1),),
        command=(sys.executable, "-c", child),
        expect_rc=1,
        require_stdout="bulk-restyle gate CLOSED",
    )


def _toy_that_imports(tmp_path: Path, body: str) -> str:
    """A ``-c`` child that imports the temp ``toy`` module and runs ``body``."""
    return f"import sys; sys.path.insert(0, {str(tmp_path)!r}); import toy; {body}"


def test_the_witness_refuses_a_check_that_is_already_red(
    probe_module, tmp_path: Path, monkeypatch, capsys
) -> None:
    """A check red with OR without the mutation must not read as CAUGHT.

    The lie this prevents is demonstrated in the same test: the identical run
    WITHOUT ``--witness`` reports CAUGHT and exits 0, because a nonzero status
    plus a printed needle is all the verdict logic can see. The witness asks
    the question the verdict cannot — was the check green to begin with — and
    refuses instead of guessing.
    """
    source = tmp_path / "toy.py"
    source.write_text('VALUE = "old"\n', encoding="utf-8")
    before = probe_module.digest(source.read_bytes())
    monkeypatch.setitem(
        probe_module.PRESETS,
        "fake",
        _toy_preset_for(
            probe_module,
            source,
            "print('bulk-restyle gate CLOSED'); raise SystemExit(1)",
        ),
    )

    without = probe_module.main(["--preset", "fake"])
    plain = capsys.readouterr().out
    assert without == 0 and "CAUGHT" in plain, plain

    with_witness = probe_module.main(["--preset", "fake", "--witness"])
    witnessed = capsys.readouterr().out
    assert with_witness == 2, witnessed
    assert "INVALID" in witnessed and "UNMUTATED" in witnessed, witnessed
    assert probe_module.digest(source.read_bytes()) == before, (
        "a refused witness must leave the file untouched — nothing was mutated"
    )


def test_the_witness_refuses_a_needle_the_clean_run_already_prints(
    probe_module, tmp_path: Path, monkeypatch, capsys
) -> None:
    """Exit 0 on the clean tree is not enough: the needle must be silent there.

    A check that always prints the needle (a stray debug line, an assertion
    copied from the docs) matches it under the mutation too, so the needle
    would prove nothing — the same class of lie as a pre-existing red, one    layer down.
    """
    source = tmp_path / "toy.py"
    source.write_text('VALUE = "old"\n', encoding="utf-8")
    before = probe_module.digest(source.read_bytes())
    monkeypatch.setitem(
        probe_module.PRESETS,
        "fake",
        _toy_preset_for(
            probe_module,
            source,
            _toy_that_imports(
                tmp_path,
                "print('bulk-restyle gate CLOSED'); "
                "raise SystemExit(0 if toy.VALUE == 'old' else 1)",
            ),
        ),
    )

    code = probe_module.main(["--preset", "fake", "--witness"])
    out = capsys.readouterr().out
    assert code == 2, out
    assert "cannot distinguish" in out, out
    assert probe_module.digest(source.read_bytes()) == before


def test_the_witness_runs_first_and_a_green_check_still_catches(
    probe_module, tmp_path: Path, monkeypatch, capsys
) -> None:
    """The flag must not cost a real catch: green and silent clean, red and loud mutated."""
    source = tmp_path / "toy.py"
    source.write_text('VALUE = "old"\n', encoding="utf-8")
    before = probe_module.digest(source.read_bytes())
    monkeypatch.setitem(
        probe_module.PRESETS,
        "fake",
        _toy_preset_for(
            probe_module,
            source,
            _toy_that_imports(
                tmp_path,
                "clean = toy.VALUE == 'old'; "
                "print('bulk-restyle gate CLOSED' if not clean else 'clean run'); "
                "raise SystemExit(0 if clean else 1)",
            ),
        ),
    )

    code = probe_module.main(["--preset", "fake", "--witness"])
    out = capsys.readouterr().out
    assert code == 0 and "CAUGHT" in out, out
    assert "witness" in out.lower() and "unmutated" in out.lower(), (
        f"the report must name the witness run and what it measured:\n{out}"
    )
    assert probe_module.digest(source.read_bytes()) == before


def test_the_witness_applies_to_a_spec_run_too(
    probe_module, tmp_path: Path, capsys
) -> None:
    """``--witness`` is the flag's own contract, not a preset perk."""
    source = tmp_path / "toy.py"
    source.write_text('VALUE = "old"\n', encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps([{"path": str(source), "old": "old", "new": "new"}]),
        encoding="utf-8",
    )

    code = probe_module.main(
        [
            "--spec",
            str(spec),
            "--expect-rc",
            "1",
            "--witness",
            "--",
            sys.executable,
            "-c",
            "print('bulk-restyle gate CLOSED'); raise SystemExit(1)",
        ]
    )
    out = capsys.readouterr().out
    assert code == 2, out
    assert "UNMUTATED" in out, out
    assert source.read_text(encoding="utf-8") == 'VALUE = "old"\n'
