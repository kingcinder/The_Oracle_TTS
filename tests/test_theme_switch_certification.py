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
        "filter_calls_ungated": 18506,
        "filter_closed_ungated": 18506,
        "filter_calls_switch": 18506,
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
    failures = certifier.evaluate(_good(certifier, filter_closed_switch=18506))

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
    """No filter on the app means the term this file certifies was never exercised."""
    failures = certifier.evaluate(_good(certifier, filter_installed=False))

    assert any("never exercised" in failure for failure in failures)


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
    """A ``the_oracle`` import resolved first would read the real config root."""
    source = SCRIPT.read_text(encoding="utf-8")
    first_import = re.search(r"^(?:from|import) the_oracle", source, re.MULTILINE)

    assert first_import is not None, "no the_oracle import found -- the scan went blind"
    redirect = source.index('os.environ["XDG_CONFIG_HOME"] = _CONFIG_ROOT')
    assert redirect < first_import.start()
    # APPDATA covers the Windows leg -- XDG alone would still touch the real one.
    assert 'os.environ["APPDATA"] = _CONFIG_ROOT' in source


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


@pytest.mark.slow
def test_a_real_run_certifies_and_writes_nothing() -> None:
    """Run the certifier as CI does, and hold it to the hermeticity promise."""
    from the_oracle.platform_support import user_config_root

    from tests.helpers import repo_tree_snapshot, snapshot_differences

    config_root = user_config_root() / "the_oracle"
    before_repo = repo_tree_snapshot(REPO_ROOT)
    before_config = repo_tree_snapshot(config_root) if config_root.exists() else {}

    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    after_repo = repo_tree_snapshot(REPO_ROOT)
    after_config = repo_tree_snapshot(config_root) if config_root.exists() else {}

    detail = f"rc={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
    assert proc.returncode == 0, f"the certifier itself failed\n{detail}"
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
