"""The chrome aspect's widget-restyle gate trace, unit-pinned.

``scripts/certify_gui.py``'s chrome aspect builds one real MainWindow per
theme inside a ``QWidget.setStyleSheet`` trace — the runtime half of
``tests/test_widget_restyle_gate.py`` run on six real themed windows: every
construction-time widget-level stylesheet from the four gated modules must
find ``gui_themes.bulk_restyle`` open, and every one of the four modules must
fire at least once (measured 2026-10-12: gui_sections 8, gui_widgets 14,
gui_cast 6, gui_chrome 1, no other caller).

The local chrome gate (``tests/test_gui_sections_chrome.py``) runs the real
sweep and can only exercise the evaluator's PASSING side. This file pins the
two failure paths a green sweep never shows — the closed gate and the blind
trace — and cross-pins the certifier's gated-module list against the AST
net's own, so the script's copy (scripts cannot import tests) cannot drift
without a deliberate re-pin here.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CERTIFIER = REPO_ROOT / "scripts" / "certify_gui.py"


@pytest.fixture(scope="module")
def certifier():
    """The certifier, imported as a file (the suite's script-test idiom).

    Importing the real module runs its preamble (throwaway config root +
    env redirect + the app_gui path-seam patch) — the same acceptance
    tests/test_gui_theme_certification._load_certifier documents.
    """
    spec = importlib.util.spec_from_file_location(
        "oracle_certify_gui_for_gate_trace", CERTIFIER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fresh_report(certifier) -> Iterator:
    """A clean failures/notes slate, restored after the test.

    The certifier reports through module-level lists (one sweep, one report);
    a unit test that leaves findings behind would fail its neighbours.
    """
    del certifier.failures[:]
    del certifier.notes[:]
    yield certifier
    del certifier.failures[:]
    del certifier.notes[:]


def _all_open_calls(certifier) -> list[tuple[bool, str]]:
    """One gate-open call per gated module — the certified shape."""
    return [(True, module) for module in certifier.GATED_WIDGET_MODULES]


def test_every_gated_module_firing_open_certifies_and_reports_counts(
    fresh_report,
) -> None:
    """The passing side: one open call per module adds no failure and one
    evidence note carrying the per-module counts."""
    certifier = fresh_report
    certifier._certify_restyle_gate("probe_theme", _all_open_calls(certifier))
    assert certifier.failures == [], certifier.failures
    assert len(certifier.notes) == 1
    note = certifier.notes[0]
    assert note.startswith("probe_theme: OK - gate trace: ")
    assert "all open" in note
    for module in certifier.GATED_WIDGET_MODULES:
        assert module in note, f"the trace note omits {module}: {note}"


def test_a_closed_gate_is_reported_and_names_the_module(fresh_report) -> None:
    """The runtime half's real finding: an ungated construction-time
    setStyleSheet fails the sweep and names the module it came from."""
    certifier = fresh_report
    calls = _all_open_calls(certifier) + [(False, "gui_widgets.py")]
    certifier._certify_restyle_gate("probe_theme", calls)
    closed = [f for f in certifier.failures if "gate CLOSED" in f]
    assert len(closed) == 1, certifier.failures
    assert "gui_widgets.py" in closed[0], closed[0]
    assert closed[0].startswith("probe_theme:"), closed[0]
    # The floors are still satisfied (every module fired), so the only
    # finding is the closed gate — a mixed failure list would hide it.
    assert len(certifier.failures) == 1, certifier.failures


def test_a_blind_trace_names_every_module_it_saw_no_call_from(fresh_report) -> None:
    """Vacuity: a trace that observed nothing must fail per module rather
    than pass green over an empty measurement."""
    certifier = fresh_report
    certifier._certify_restyle_gate("probe_theme", [])
    blind = [f for f in certifier.failures if "went blind" in f]
    assert len(blind) == len(certifier.GATE_TRACE_FLOORS), certifier.failures
    for module in certifier.GATE_TRACE_FLOORS:
        assert any(module in failure for failure in blind), (module, blind)
    assert certifier.notes == [], certifier.notes


def test_calls_from_untouched_modules_are_neither_counted_nor_credited(
    fresh_report,
) -> None:
    """Only the four gated modules are in scope: a stylesheet from anywhere
    else neither feeds the floors nor excuses a missing module."""
    certifier = fresh_report
    calls = _all_open_calls(certifier) + [
        (True, "app_gui.py"),
        (False, "app_gui.py"),  # ungated, but out of the net's scope
    ]
    certifier._certify_restyle_gate("probe_theme", calls)
    assert certifier.failures == [], certifier.failures


def test_the_certifiers_gated_module_list_cannot_drift_from_the_nets() -> None:
    """The script's copy of the gated-module list equals the AST net's own.

    Scripts cannot import tests, so the list is duplicated; this is the pin
    that makes the duplication honest — a module added to the net must be
    added to the certifier (and vice versa) in the same change.
    """
    from tests.test_widget_restyle_gate import GATED_MODULES

    spec = importlib.util.spec_from_file_location(
        "oracle_certify_gui_for_gate_list", CERTIFIER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Executed fresh (not via the fixture) so the source constants are read
    # exactly as a sweep would see them.
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    assert set(module.GATED_WIDGET_MODULES) == set(GATED_MODULES), (
        f"certify_gui's gated list {module.GATED_WIDGET_MODULES} drifted from "
        f"the gate net's {GATED_MODULES}: update both and re-pin"
    )
    assert set(module.GATE_TRACE_FLOORS) == set(module.GATED_WIDGET_MODULES), (
        "every gated module needs a construction floor, and a floor for an "
        f"ungated module certifies nothing: floors {sorted(module.GATE_TRACE_FLOORS)}, "
        f"gated {sorted(module.GATED_WIDGET_MODULES)}"
    )
