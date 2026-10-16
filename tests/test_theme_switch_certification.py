"""Tests for the theme-switch certifier and its CI wiring.

The script's value is that a theme switch's cost is *split* rather than timed:
structure, propagation, rule matching, and the part that used to be ours. Three
properties are pinned here:

* every verdict can actually fail -- a term whose bound no measurement can trip
  would be decoration, so each is driven with a synthetic number just over its
  line (the real bounds hold a real run, asserted separately);
* timings are never part of the verdict, because they are of the machine: the
  docstring promises it and ``test_timings_are_never_asserted`` keeps it;
* the sweep writes nothing -- a full subprocess run leaves the real repo tree and
  the real app-config root byte-identical.

The slow end-to-end run is one subprocess in one test (the script's own job);
everything else is pure.
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
SCRIPT = SCRIPTS_DIR / "certify_theme_switch.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def certifier():
    return _load("oracle_certify_theme_switch", "certify_theme_switch.py")


# --- the verdicts ---------------------------------------------------------------


def _good(certifier, **overrides):
    """A measurement that passed, from the real run on this tree.

    The numbers are the ones the certifier printed when it certified HEAD
    (354 widgets, depth 13, 59 rules, ~16k change events per restyle), so the
    bounds are exercised against reality rather than against a convenient story.
    The filter-view numbers are the window-scoped ones the repair produced
    (2920 events, all of them with the scope open on the gated path): the
    filter rides the window and its tooltip owners now, not the QApplication.
    """
    values = {
        "widgets": 354,
        "depth": 13,
        "real_rules": 59,
        "small_rules": 2,
        "storm_change_events": 16210,
        "storm_all_events": 18506,
        "storm_after_events": 19573,
        "switch_change_events": 16210,
        "switch_open_gate_events": 18506,
        "switch_rounds": 4,
        "filter_installed": True,
        "filter_window_scoped": True,
        "filter_app_scope_installs": 0,
        "filter_calls_ungated": 2920,
        "filter_closed_ungated": 2920,
        "filter_calls_switch": 2920,
        "filter_closed_switch": 0,
        "small_change_events": 10806,
        "redundant_change_events": 0,
        "redundant_restyles": 0,
        "qt_warnings": (),
    }
    values.update(overrides)
    return certifier.Measure(**values)


def test_a_real_measurement_is_accepted(certifier) -> None:
    assert certifier.evaluate(_good(certifier)) == []


def test_windows_safe_polling_mode_needs_no_native_filter_events(certifier) -> None:
    measured = _good(
        certifier,
        safe_polling_mode=True,
        storm_change_events=0,
        storm_all_events=0,
        storm_after_events=0,
        small_change_events=0,
        redundant_change_events=0,
        redundant_restyles=1,
        filter_calls_switch=0,
        filter_calls_ungated=0,
        filter_closed_ungated=0,
    )
    assert certifier.evaluate(measured) == []


def test_windows_never_installs_the_python_storm_counter(certifier, monkeypatch) -> None:
    monkeypatch.setattr(certifier, "is_windows", lambda: True)
    assert certifier._counter_for_platform() is None
    monkeypatch.setattr(certifier, "is_windows", lambda: False)
    assert isinstance(certifier._counter_for_platform(), certifier._StormCounter)


def test_a_stub_tree_cannot_certify(certifier) -> None:
    failures = certifier.evaluate(_good(certifier, widgets=40))

    assert any("structure" in failure and "widgets" in failure for failure in failures)


def test_a_shallow_tree_cannot_certify(certifier) -> None:
    """The repolish term scales with depth, so a shallow tree measures nothing."""
    failures = certifier.evaluate(_good(certifier, depth=2))

    assert any("depth" in failure for failure in failures)


def test_a_gutted_theme_is_refused(certifier) -> None:
    failures = certifier.evaluate(_good(certifier, real_rules=2))

    assert any("rules" in failure and "gutted" in failure for failure in failures)


def test_rule_matching_must_stay_separable(certifier) -> None:
    """If fewer rules stop fanning out less, the rule term has stopped existing."""
    failures = certifier.evaluate(_good(certifier, small_change_events=16210))

    assert any("rule matching" in failure and "strictly less" in failure for failure in failures)


def test_a_missing_storm_is_refused(certifier) -> None:
    failures = certifier.evaluate(_good(certifier, storm_change_events=10))

    assert any("propagation" in failure for failure in failures)


def test_a_deferred_storm_is_refused(certifier) -> None:
    """The gate can only cover what happens inside the call."""
    failures = certifier.evaluate(_good(certifier, storm_after_events=200_000))

    assert any("deferred" in failure for failure in failures)


def test_the_dead_gate_is_refused(certifier) -> None:
    """The one mutation this file exists for: apply_theme stops opening the scope."""
    failures = certifier.evaluate(_good(certifier, filter_closed_switch=2920))

    assert any(
        "ours" in failure and "CLOSED" in failure and "must be 0" in failure
        for failure in failures
    )


def test_a_vacuous_gate_is_refused(certifier) -> None:
    """A scope open over nothing would make the dead-gate assertion meaningless."""
    failures = certifier.evaluate(_good(certifier, filter_calls_switch=10))

    assert any("vacuous" in failure for failure in failures)


def test_the_before_picture_must_be_a_storm(certifier) -> None:
    failures = certifier.evaluate(_good(certifier, filter_closed_ungated=120))

    assert any("before-picture" in failure for failure in failures)


def test_a_missing_tooltip_filter_is_refused(certifier) -> None:
    """No filter on the window means the term this file certifies never ran."""
    failures = certifier.evaluate(_good(certifier, filter_installed=False))

    assert any("never exercised" in failure for failure in failures)


def test_a_filter_not_bound_to_the_window_is_refused(certifier) -> None:
    """The term is the window-scoped filter, not merely some filter existing."""
    failures = certifier.evaluate(_good(certifier, filter_window_scoped=False))

    assert any("not bound to the window" in failure for failure in failures)


def test_a_filter_installed_on_the_application_is_refused(certifier) -> None:
    """The 1.4.0 crash contract: app-wide filtering of every object must not
    come back, however healthy the rest of the numbers look."""
    failures = certifier.evaluate(_good(certifier, filter_app_scope_installs=1))

    assert any(
        "installed on the QApplication" in failure and "mid-destruction" in failure
        for failure in failures
    )


def test_a_run_without_a_genuine_switch_is_refused(certifier) -> None:
    """Measured live while mutating: a gutted theme collapsed every key to one css."""
    failures = certifier.evaluate(_good(certifier, switch_rounds=0))

    assert any("not one gated round" in failure for failure in failures)


def test_a_redundant_restyle_is_refused(certifier) -> None:
    failures = certifier.evaluate(
        _good(certifier, redundant_change_events=16196, redundant_restyles=1)
    )

    assert any("re-applying the current theme fanned out" in failure for failure in failures)
    assert any("must be a no-op" in failure for failure in failures)


def test_an_unparsable_stylesheet_is_refused(certifier) -> None:
    failures = certifier.evaluate(
        _good(certifier, qt_warnings=("Could not parse stylesheet of object QWidget(0x1)",))
    )

    assert any("could not apply the stylesheet" in failure for failure in failures)


def test_unrelated_qt_warnings_are_not_failures(certifier) -> None:
    """The handler is installed for diagnostics; only a parse failure is a verdict."""
    assert certifier.evaluate(_good(certifier, qt_warnings=("QFont::setPointSize: Point size <= 0",))) == []


# --- the promises the code makes about itself -----------------------------------


def test_timings_are_never_asserted(certifier) -> None:
    """``evaluate`` must not read a timing: a slow runner is not a defect."""
    import inspect

    source = inspect.getsource(certifier.evaluate)

    assert "timings" not in source
    assert "perf_counter" not in source


def test_the_three_terms_and_ours_are_named_in_the_contract(certifier) -> None:
    """The split is the deliverable; the docstring has to keep saying so."""
    docstring = certifier.__doc__ or ""

    for term in ("structure", "propagation", "rule matching", "ours"):
        assert term in docstring, f"{term} vanished from the certifier's contract"


def test_the_config_redirect_precedes_every_the_oracle_import() -> None:
    """A ``the_oracle`` import resolved first would read the real config root.

    Since 2026-10-11 the redirect itself is the certifier family's shared helper
    (``scripts/certifier_isolation.py``), so this pins the order and the
    delegation: a call to ``isolated_config_root`` above every ``the_oracle``
    import. That the helper redirects both variables — XDG for POSIX, APPDATA
    for Windows — and arms the reap is pinned where it lives, in
    tests/test_certifier_isolation.py.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    first_import = re.search(r"^(?:from|import) the_oracle", source, re.MULTILINE)

    assert first_import is not None, "no the_oracle import found -- the scan went blind"
    assert "from certifier_isolation import isolated_config_root, reap_isolated_root" in source, (
        "the throwaway root no longer comes from the shared helper — a hand-rolled "
        "second copy is what this refactor removed, and a copy has no owner"
    )
    redirect = source.index("_CONFIG_ROOT = isolated_config_root(")
    assert redirect < first_import.start()
    assert 'isolated_config_root("oracle_theme_switch_cfg_")' in source, (
        "the sweep's throwaway root must keep its own prefix, so a leaked root names its sweep"
    )


def test_the_window_paths_are_drained_into_the_throwaway() -> None:
    """The ``Output/logs`` timing write every construction makes must not hit the repo."""
    source = SCRIPT.read_text(encoding="utf-8")

    assert "_app_gui.ensure_repo_default_paths = _throwaway_paths" in source
    assert "return ensure_repo_default_paths(_CONFIG_ROOT)" in source
    # Drift guard: app_gui must still expose the seam the drain patches.
    app_gui = (REPO_ROOT / "src" / "the_oracle" / "app_gui.py").read_text(encoding="utf-8")
    assert "ensure_repo_default_paths(" in app_gui


def test_no_repo_log_entry_points() -> None:
    """``configure_logging`` appends to the repo's logs/; ``launch_gui`` writes timing."""
    source = SCRIPT.read_text(encoding="utf-8")

    for forbidden in ("configure_logging(", "launch_gui("):
        assert forbidden not in source, f"the sweep must not call {forbidden}"


# --- the run itself -------------------------------------------------------------


def _run_in_scratch(
    scratch: Path, *args: str
) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run the certifier with ``TMPDIR`` redirected; report what it left there.

    ``tempfile.mkdtemp`` — the script's throwaway config root — honors
    ``TMPDIR``, so every root a run creates lands under ``scratch``: that is
    what lets this file say "nothing left behind" about its own run instead of
    about whatever happens to be sitting in the shared ``/tmp``.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "TMPDIR": str(scratch)},
    )
    return proc, sorted(path.name for path in scratch.iterdir())


def test_a_fast_exit_reaps_the_throwaway_config_root(tmp_path: Path) -> None:
    """``--help`` leaves through ``SystemExit``, before any of the script's
    own teardown: the config root it made at import must still be reaped.

    Measured on 2026-10-10: the roots this family left in ``/tmp`` were
    exactly the ones no ``finally`` covered (``--help`` and the aspect-typo
    exit in the sibling certifier), each carrying the PulseAudio client's own
    ``pulse/cookie`` from script load.
    """
    helped, leftover = _run_in_scratch(tmp_path / "helped", "--help")

    assert helped.returncode == 0, f"--help must exit 0\n{helped.stderr[-800:]}"
    assert not leftover, (
        f"--help left its throwaway config root behind: {leftover} — the reap "
        "has to cover the exits that never reach main()'s teardown"
    )


def test_a_real_run_certifies_and_writes_nothing(tmp_path: Path) -> None:
    """Run the certifier as CI does, and hold it to the hermeticity promise.

    Deliberately NOT ``slow``-marked: this is the local half of a named CI
    step, and a ``-m "not slow"`` loop that dropped it would report green
    without measuring the theme switch at all. tests/test_fast_subset.py asks
    pytest's own collector whether the fast subset still keeps this gate.
    """
    from the_oracle.platform_support import user_config_root

    from tests.helpers import repo_tree_snapshot, snapshot_differences

    config_root = user_config_root() / "the_oracle"
    before_repo = repo_tree_snapshot(REPO_ROOT)
    before_config = repo_tree_snapshot(config_root) if config_root.exists() else {}

    proc, leftover = _run_in_scratch(tmp_path / "scratch")

    after_repo = repo_tree_snapshot(REPO_ROOT)
    after_config = repo_tree_snapshot(config_root) if config_root.exists() else {}

    detail = f"rc={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
    assert proc.returncode == 0, f"the certifier itself failed\n{detail}"
    assert not leftover, (
        f"the run left its throwaway config root behind: {leftover}\n{detail}"
    )
    for term in ("structure", "propagation", "rule matching", "ours (the gate)"):
        assert term in proc.stdout, f"the report does not mention {term}\n{detail}"

    for label, before, after in (
        ("real repo tree", before_repo, after_repo),
        ("real app-config root", before_config, after_config),
    ):
        created, changed, removed = snapshot_differences(before, after)
        assert not (created or changed or removed), (
            f"the sweep wrote to the {label}: created={created} changed={changed} removed={removed}\n{detail}"
        )


# --- the CI wiring --------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow_steps() -> list[dict]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["test"]["steps"]


def test_ci_wires_the_certifier_on_both_operating_systems(workflow_steps: list[dict]) -> None:
    """The section-chrome sweep's shape, which this file follows: one step per OS."""
    matches = [step for step in workflow_steps if "certify_theme_switch.py" in (step.get("run") or "")]

    assert len(matches) == 2, "expected exactly one Linux step and one Windows step"
    by_os = {step.get("if"): step for step in matches}
    assert set(by_os) == {"runner.os == 'Linux'", "runner.os == 'Windows'"}, by_os.keys()

    linux = by_os["runner.os == 'Linux'"]
    assert "certify_theme_switch.py" in linux["run"] and ".venv/bin/python" in linux["run"]
    assert linux["shell"] == "bash"

    windows = by_os["runner.os == 'Windows'"]
    assert "certify_theme_switch.py" in windows["run"]
    assert ".venv\\Scripts\\python.exe" in windows["run"]
    assert windows["shell"] == "pwsh"


def test_ci_step_names_say_what_is_certified(workflow_steps: list[dict]) -> None:
    names = [
        step.get("name", "")
        for step in workflow_steps
        if "certify_theme_switch.py" in (step.get("run") or "")
    ]

    for name in names:
        assert "Theme Switch" in name, f"{name!r} does not say which sweep it is"
