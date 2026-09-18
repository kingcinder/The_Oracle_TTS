"""Tests for the fresh-clone acceptance script and the surface it dispatches.

The script's whole value is in three properties, so those are what is pinned:
that a *path* difference is structural while a difference of substance is not,
that the comparison cannot be widened into uselessness by one broad entry, and
that the documented entry points it probes are really runnable -- which is how
the `./oracle` regression below was found.

The slow end-to-end run (both legs, full suite) is deliberately not repeated
here; it takes minutes and is the script's own job. What is exercised is every
decision the script makes, plus the real `git archive` acquisition, so the tests
stay fast enough to run on every change.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def acceptance():
    return _load("oracle_fresh_clone_acceptance", "fresh_clone_acceptance.py")


@pytest.fixture(scope="module")
def idempotence():
    return _load("oracle_doctor_idempotence_for_acceptance", "doctor_idempotence.py")


# --- the delta comparison -------------------------------------------------------


def test_a_path_naming_the_tree_is_structural_not_a_delta(acceptance, tmp_path: Path) -> None:
    """The same value once the root is substituted describes no real difference."""
    fresh = tmp_path / "clean"
    current = tmp_path / "working"
    deltas = acceptance.differences(
        {"doctor": {"repo_root": str(fresh), "detail": f"rendered into {fresh}/build/out.flac"}},
        {"doctor": {"repo_root": str(current), "detail": f"rendered into {current}/build/out.flac"}},
        fresh_root=fresh,
        current_root=current,
    )

    assert deltas == []


def test_a_difference_of_substance_is_reported_verbatim(acceptance, tmp_path: Path) -> None:
    deltas = acceptance.differences(
        {"doctor": {"vulkan_backend": {"ok": False}}},
        {"doctor": {"vulkan_backend": {"ok": True}}},
        fresh_root=tmp_path / "clean",
        current_root=tmp_path / "working",
    )

    assert len(deltas) == 1
    delta = deltas[0]
    # The dotted path carries the check name, so it reads as its registry entry.
    assert delta.path == "doctor.vulkan_backend.ok"
    assert delta.fresh is False and delta.current is True


def test_a_registered_field_is_marked_and_an_unregistered_one_is_not(acceptance, tmp_path: Path) -> None:
    registered = "doctor.vulkan_backend.ok"
    assert registered in acceptance.EXPECTED_DELTAS

    deltas = acceptance.differences(
        {"doctor": {"vulkan_backend": {"ok": False, "something_new": 1}}},
        {"doctor": {"vulkan_backend": {"ok": True, "something_new": 2}}},
        fresh_root=tmp_path / "clean",
        current_root=tmp_path / "working",
    )
    by_path = {delta.path: delta for delta in deltas}

    assert by_path[registered].registered is True
    assert by_path[registered].reason
    # An unregistered field is what fails the run; it must not inherit the
    # registration of a sibling.
    assert by_path["doctor.vulkan_backend.something_new"].registered is False


def test_registering_a_container_cannot_excuse_its_children(acceptance, tmp_path: Path) -> None:
    """Registration is per field: one broad entry would hide unrelated changes."""
    container = "doctor.vulkan_backend"
    assert container not in acceptance.EXPECTED_DELTAS
    children = {"caveat": 1, "something_new": 2}

    deltas = acceptance.differences(
        {"doctor": {"vulkan_backend": dict(children, extra="a")}},
        {"doctor": {"vulkan_backend": dict(children, extra="b")}},
        fresh_root=tmp_path / "clean",
        current_root=tmp_path / "working",
    )

    assert {delta.path for delta in deltas} == {"doctor.vulkan_backend.extra"}
    assert deltas[0].registered is False


def test_a_measurement_is_excluded_rather_than_reported(acceptance, tmp_path: Path) -> None:
    deltas = acceptance.differences(
        {"doctor": {"deterministic_smoke": {"runtime_seconds": 0.9}}},
        {"doctor": {"deterministic_smoke": {"runtime_seconds": 1.1}}},
        fresh_root=tmp_path / "clean",
        current_root=tmp_path / "working",
    )

    assert deltas == []


def test_a_registered_measurement_still_reports_a_type_change(acceptance, tmp_path: Path) -> None:
    """The exemption is for a number that differs, not for anything at that path."""
    deltas = acceptance.differences(
        {"doctor": {"deterministic_smoke": {"runtime_seconds": None}}},
        {"doctor": {"deterministic_smoke": {"runtime_seconds": 1.1}}},
        fresh_root=tmp_path / "clean",
        current_root=tmp_path / "working",
    )

    assert [delta.path for delta in deltas] == ["doctor.deterministic_smoke.runtime_seconds"]


def test_the_measurement_registry_agrees_with_the_idempotence_check(acceptance, idempotence) -> None:
    """Two scripts answering "is this a duration or a fact" must answer alike."""
    assert tuple(acceptance.MEASURED_FIELDS) == tuple(idempotence.MEASURED_FIELDS)


# --- the registry itself --------------------------------------------------------


def test_every_registered_delta_names_a_check_and_a_reason(acceptance) -> None:
    for path, reason in acceptance.EXPECTED_DELTAS.items():
        check = path.split(".", 1)[0]
        assert check in acceptance.CHECKS, f"{path} does not start with a known check"
        assert len(reason.strip()) > 20, f"{path} needs a real reason, not a placeholder"


def test_the_registry_is_not_silently_widened_by_a_bare_check_name(acceptance) -> None:
    for path in acceptance.EXPECTED_DELTAS:
        assert path.count(".") >= 1, path
        assert path not in acceptance.CHECKS, f"{path} registers a whole check"


# --- reading pytest's output ----------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("1008 passed, 1 skipped in 178.96s", {"passed": 1008, "failed": 0, "errors": 0, "skipped": 1}),
        ("1009 passed, 16 warnings in 191.75s", {"passed": 1009, "failed": 0, "errors": 0, "skipped": 0}),
        ("3 failed, 5 passed, 2 errors in 1.0s", {"passed": 5, "failed": 3, "errors": 2, "skipped": 0}),
        ("1 failed, 2 passed, 1 skipped in 0.5s", {"passed": 2, "failed": 1, "errors": 0, "skipped": 1}),
    ],
)
def test_counts_come_out_of_the_summary_line(acceptance, output: str, expected: dict) -> None:
    assert acceptance.parse_pytest_summary(output) == expected


def test_the_skip_audit_block_is_not_counted_twice(acceptance) -> None:
    """The audit prints its own line containing "skipped"; only the summary counts."""
    real_output = (
        "....................................s                        [100%]\n"
        "1 test(s) did not run here:\n"
        "  tests/test_vulkan_backend.py::test_vulkan_backend_smoke_requires_device\n"
        "      Skipped: audiocpp_cli is not built; run scripts/build_audio_cpp.sh first.\n"
        "\n"
        "1008 passed, 1 skipped in 178.96s\n"
    )

    assert acceptance.parse_pytest_summary(real_output) == {
        "passed": 1008,
        "failed": 0,
        "errors": 0,
        "skipped": 1,
    }


def test_no_counts_at_all_is_an_error_rather_than_a_zero_result() -> None:
    acceptance = _load("oracle_fresh_clone_acceptance_inline", "fresh_clone_acceptance.py")

    assert not any(acceptance.parse_pytest_summary("pytest: command not found").values())


# --- reading a wrapper's dispatch ----------------------------------------------


def test_the_dispatched_subcommand_is_read_off_the_usage_line(acceptance) -> None:
    usage = "usage: manage_install.py doctor [-h] [--skip-model-init] [--ci]\n\noptions:\n"

    assert acceptance.parse_dispatched_subcommand(usage) == "doctor"


def test_a_mention_of_a_subcommand_is_not_a_dispatch(acceptance) -> None:
    """A wrapper that echoed its own command line must not read as success."""
    echoed = "FAIL: Need Python 3.11 or 3.12\n$ python scripts/manage_install.py bootstrap --help\n"

    assert acceptance.parse_dispatched_subcommand(echoed) is None


# --- the documented entry points ------------------------------------------------


def _readme_direct_commands() -> set[str]:
    """Tracked files the README tells a user to run as ``./path``.

    ``./install.sh`` is in the README but refers to the generated offline
    bundle's launcher, not a file in this repository, so it drops out here by
    not being tracked -- which is exactly the distinction that matters.
    """
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    commands = {match.group(1) for match in re.finditer(r"\./([A-Za-z0-9_][A-Za-z0-9_./-]*)", text)}
    return {name for name in commands if (REPO_ROOT / name).is_file()}


def test_the_readme_documents_commands_this_test_can_see() -> None:
    """Guards the guard: a regex that matched nothing would pass everything."""
    documented = _readme_direct_commands()

    assert "oracle" in documented
    assert any(name.endswith(".sh") for name in documented)


def test_every_documented_command_is_executable() -> None:
    """The defect this pass found: ``./oracle`` was tracked 100644.

    A file the documentation invokes directly has to carry the executable bit,
    or ``./oracle install`` -- the README's first instruction -- dies with
    "Permission denied" (exit 126) on a fresh clone.

    Both halves are asserted, and neither is skipped, so this holds in either
    tree the suite runs in. On disk is the *effective* mode, which is what an
    extracted archive has and what a checkout actually gives you. The recorded
    git mode is the *promised* mode -- the thing a fresh clone will get even if
    this particular checkout drifted -- and it is only readable where there is a
    repository, which an archive deliberately is not.
    """
    commands = sorted(_readme_direct_commands())

    not_executable = [name for name in commands if not os.access(REPO_ROOT / name, os.X_OK)]
    assert not not_executable, (
        "these files are documented as `./<path>` but cannot be executed: "
        + "; ".join(not_executable)
    )

    if not (REPO_ROOT / ".git").exists():
        return
    modes = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-s", "--", *commands],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    wrong_mode = []
    for line in modes.splitlines():
        # `git ls-files -s` is `<mode> <sha> <stage>\t<path>`; splitting the tail
        # on the tab keeps a path containing spaces intact instead of printing
        # the stage number alongside it.
        mode, _, tail = line.split(maxsplit=2)
        path = tail.split("\t", 1)[-1]
        if mode != "100755":
            wrong_mode.append(f"{path} is {mode}")

    assert not wrong_mode, (
        "these files are documented as `./<path>` but a fresh clone would not be "
        "able to run them: " + "; ".join(wrong_mode)
    )


def test_the_dispatch_matrix_covers_every_root_wrapper(acceptance) -> None:
    """A new wrapper with no row here would ship undispatch-checked."""
    root_wrappers = {
        path.name for path in REPO_ROOT.glob("*.sh")
    }
    covered = {script.lstrip("./") for script, _, _ in acceptance.WRAPPER_MATRIX}

    assert root_wrappers <= covered
    assert "oracle" in covered


def test_the_matrix_checks_an_alias_rather_than_only_identities(acceptance) -> None:
    """`oracle start` must reach `run`; an identity-only matrix proves less."""
    rows = {(script, args, expected) for script, args, expected in acceptance.WRAPPER_MATRIX}
    aliases = [row for row in rows if row[1] and row[1][0] != row[2]]

    assert any(script == "./oracle" and args == ("start", "--help") and expected == "run"
               for script, args, expected in aliases)


# --- proving the tree under test is the tree that ran --------------------------


def test_a_tree_whose_code_does_not_load_is_refused(acceptance, tmp_path: Path) -> None:
    """Never report on a tree that was not exercised.

    The trap this guards is concrete: this interpreter has the package installed
    in editable mode pointing at the working tree, so a tree with an empty
    ``src`` still resolves ``the_oracle`` -- to the *wrong* tree. That must be a
    refusal, not a green report about code that never ran.
    """
    empty = tmp_path / "empty"
    (empty / "src").mkdir(parents=True)

    with pytest.raises(acceptance.AcceptanceError, match="did not load its own code"):
        acceptance.assert_imports_the_tree_under_test(empty, Path(sys.executable))


def test_this_repository_loads_its_own_code(acceptance) -> None:
    resolved = acceptance.assert_imports_the_tree_under_test(REPO_ROOT, Path(sys.executable))

    assert resolved.startswith(str(REPO_ROOT))


# --- acquiring the clean tree ---------------------------------------------------


def _make_scratch_repo(root: Path) -> Path:
    """A self-contained repository, so these tests run wherever the suite does.

    Built here rather than reusing this checkout because the acceptance run is
    performed *from* a `git archive` tree, which has no repository at all -- a
    test needing this repo's `.git` would fail in exactly the tree the script
    creates, which is how the first version of this file broke.
    """
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "PATH": os.environ.get("PATH", "/usr/bin"),
    }
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True, env=env)
    (root / ".gitignore").write_text("ignored_dir/\n", encoding="utf-8")
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (root / "ignored_dir").mkdir()
    (root / "ignored_dir" / "build_output.bin").write_bytes(b"noise")
    subprocess.run(["git", "-C", str(root), "add", ".gitignore", "tracked.txt"], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True, env=env)
    (root / "untracked.txt").write_text("uncommitted\n", encoding="utf-8")
    return root


def test_the_clean_tree_carries_the_tracked_files_and_no_local_state(acceptance, tmp_path: Path) -> None:
    repo = _make_scratch_repo(tmp_path / "repo")

    tree = acceptance.export_fresh_tree(repo, tmp_path / "tree")

    assert (tree / "tracked.txt").is_file()
    assert (tree / ".gitignore").is_file()
    # No .git, so a check that misbehaved cannot write to the repository it was
    # archived from, and no local state travels with it.
    assert not (tree / ".git").exists()
    assert not (tree / "ignored_dir").exists(), "git-ignored state must not travel"
    assert not (tree / "untracked.txt").exists(), "untracked files must not travel"


def test_the_working_tree_can_be_overlaid_on_request(acceptance, tmp_path: Path) -> None:
    repo = _make_scratch_repo(tmp_path / "repo")

    tree = acceptance.export_fresh_tree(repo, tmp_path / "overlaid", working_tree=True)

    assert (tree / "untracked.txt").read_text(encoding="utf-8") == "uncommitted\n"
    # Overlaying the working tree must not smuggle in the ignored build output,
    # and must not smuggle in a repository either.
    assert not (tree / "ignored_dir").exists()
    assert not (tree / ".git").exists()


@pytest.fixture(scope="module")
def acceptance_step() -> dict:
    """The CI step that runs this script, so its declared contract can be pinned."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    matches = [step for step in steps if "fresh_clone_acceptance.py" in (step.get("run") or "")]
    assert len(matches) == 1, "expected exactly one CI step running the acceptance script"
    return matches[0]


# --- the CI step's declared contract -------------------------------------------


def test_ci_runs_the_check_this_script_exists_for(acceptance_step: dict) -> None:
    run = acceptance_step["run"].strip()

    assert run.startswith("python scripts/fresh_clone_acceptance.py")
    # The suite leg is the test matrix's own job above this step; running two
    # more full suites here would double the job's cost for no new signal.
    assert "--only doctor,wrappers" in run
    assert "--working-tree" not in run, "a CI run must validate the committed tree"


def test_ci_selects_only_checks_that_exist(acceptance, acceptance_step: dict) -> None:
    only = re.search(r"--only\s+(\S+)", acceptance_step["run"]).group(1)

    unknown = [name for name in only.split(",") if name not in acceptance.CHECKS]
    assert not unknown, f"the CI step asks for checks that do not exist: {unknown}"


def test_ci_runs_it_where_the_wrappers_it_probes_exist(acceptance_step: dict) -> None:
    """Every matrix row is a .sh wrapper; those do not exist on Windows."""
    assert acceptance_step["if"] == "runner.os == 'Linux'"



