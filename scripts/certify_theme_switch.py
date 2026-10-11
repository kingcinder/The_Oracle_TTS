#!/usr/bin/env python3
"""Certify the theme-switch cost split on the REAL window, offscreen.

A theme switch is one synchronous ``QApplication.setStyleSheet``, and its cost
has three separable terms plus one that used to be ours:

1. **structure** — how much tree there is to repolish (widget count × depth).
   It is the one term no code change here can shrink, which is exactly why the
   others have to be measured against a *real* window: on a stub tree every
   number below is meaningless.
2. **propagation** — Qt's unpolish/repolish fan-out: one restyle streams tens of
   thousands of palette/font/style events across the whole tree, synchronously
   inside the call (hiding the window or disabling updates changes nothing).
3. **rule matching** — QSS selector work, proportional to the rule count: the
   real stylesheet fans out far more change events than a 2-rule one on the same
   tree, which is what makes a selector diet a real lever.
4. **ours** — the app-wide Ctrl+hover filter used to re-enter Python for every
   one of those events even though no branch in it can fire for an event Qt
   generated inside ``setStyleSheet`` (real input cannot arrive until the call
   returns to the event loop). ``gui_themes.bulk_restyle`` now gates it. Measured
   A/B in one process: gate-off 510 ms vs gate-on 373 ms.

Only the gate is asserted as a *contract*; the rest are bounds and relationships,
because they are facts about Qt and this tree rather than about a machine.
Timings are reported and never asserted — the same numbers on a runner with
different font metrics or a different Qt build are not a defect, and a gate that
failed on them would be noise. What must hold anywhere:

* the sandbox is a real tree (widget/depth floors);
* a genuine switch fans out a storm, and it is synchronous inside the call;
* a 2-rule stylesheet fans out measurably less than the real one;
* during a real ``apply_theme`` switch the **app's own** Ctrl+hover filter is
  invoked for that whole storm and finds the bulk-restyle scope open on every
  one of those calls, while the same measurement on the ungated path (a direct
  ``setStyleSheet``, which is what the code did before the gate) finds it closed
  throughout — the A/B that makes the scope load-bearing here and fails if
  ``apply_theme`` stops opening it;
* a redundant re-apply is free (no events, no restyle).

Scope, stated plainly: this certifies the *scope* across the real storm, which
the unit tests cannot — they drive ``apply_theme`` with a duck-typed app stand-in
and a synthetic event, so no real Qt fan-out is involved. That the filter's own
branches are skipped belongs to
``tests/test_gui_tooltips.py::test_event_filter_short_circuits_during_a_bulk_restyle``,
which is where the line-by-line contract lives.

Writes nothing: the app config root is redirected to a throwaway before any
``the_oracle`` import, and the window's default paths are drained into it, so the
settings rewrites and the ``Output/logs`` timing writes land there. Exit 0 means
every term held.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Offscreen before Qt is imported, and the throwaway config root before any
# the_oracle import (both must precede their readers, not merely be intended).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_CONFIG_ROOT = tempfile.mkdtemp(prefix="oracle_theme_switch_cfg_")
os.environ["XDG_CONFIG_HOME"] = _CONFIG_ROOT
os.environ["APPDATA"] = _CONFIG_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtCore import QCoreApplication, QEvent, QObject, qInstallMessageHandler  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from the_oracle import app_gui as _app_gui  # noqa: E402
from the_oracle import gui_tooltips  # noqa: E402
from the_oracle.app_gui import MainWindow  # noqa: E402
from the_oracle.app_paths import ensure_repo_default_paths  # noqa: E402
from the_oracle.gui_themes import (  # noqa: E402
    DEFAULT_THEME,
    THEMES,
    apply_theme,
    build_stylesheet,
    in_bulk_restyle,
)

#: The event types Qt sends while repolishing. The storm's size is the
#: propagation term; the exact mix is reported, not asserted (Qt's choice).
CHANGE_TYPES = ("PaletteChange", "FontChange", "StyleChange")

#: Bounds, not exact values: these are facts about Qt and this tree, measured on
#: the developer machine (354 widgets, depth 13, 59 rules, ~16k change events per
#: restyle) and floored generously, so a differently-built runner still passes
#: while a stub tree, a dead gate, or a gutted stylesheet cannot.
MIN_WIDGETS = 200
MIN_DEPTH = 5
MIN_REAL_RULES = 40
MAX_SMALL_RULES = 5
MIN_STORM_CHANGE_EVENTS = 5000
#: What fraction of the fan-out must land inside the call for "synchronous" to be
#: the honest word (measured ~0.83; a pump-based storm would be ~0).
MIN_SYNCHRONOUS_SHARE = 0.5
#: A 2-rule stylesheet must fan out noticeably less (measured ~0.66 of the real
#: css). This is the structural separation of the rule-matching term.
MAX_SMALL_SHARE = 0.8

_SMALL_CSS = "QWidget { color: #202020; }\nQPushButton { color: #101010; }\n"


def _throwaway_paths(_repo_root: Path):
    """MainWindow's ``self.paths`` (and its ``Output/logs`` writes) resolve here.

    The same seam the suite patches, so the
    ``Output/logs/gui_startup_timing.json`` write every construction makes lands
    in the throwaway rather than the repo.
    """
    return ensure_repo_default_paths(_CONFIG_ROOT)


_app_gui.ensure_repo_default_paths = _throwaway_paths


@dataclass
class Measure:
    """One run's numbers. Every verdict below reads only these."""

    widgets: int = 0
    depth: int = 0
    real_rules: int = 0
    small_rules: int = 0

    #: Ungated direct restyle (what the code did before the gate).
    storm_change_events: int = 0
    storm_all_events: int = 0
    storm_after_events: int = 0

    #: Gated restyle through apply_theme.
    switch_change_events: int = 0
    switch_open_gate_events: int = 0
    switch_rounds: int = 0

    #: The app's OWN Ctrl+hover filter, counted from inside it (see measure()).
    filter_installed: bool = False
    filter_calls_ungated: int = 0
    filter_closed_ungated: int = 0
    filter_calls_switch: int = 0
    filter_closed_switch: int = 0

    #: 2-rule stylesheet on the same tree, ungated.
    small_change_events: int = 0

    #: Re-applying the current theme: must do nothing at all.
    redundant_change_events: int = 0
    redundant_restyles: int = 0

    qt_warnings: tuple[str, ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)


def evaluate(measured: Measure) -> list[str]:
    """Every contract that failed, named for whoever has to fix it.

    Pure: no Qt, no I/O. The floors and relationships live here so a test can
    drive both verdicts with synthetic numbers instead of a real window.
    """
    failures: list[str] = []

    # 1. structure -- the sandbox has to be a real window.
    if measured.widgets < MIN_WIDGETS:
        failures.append(
            f"structure: only {measured.widgets} widgets (floor {MIN_WIDGETS}) -- every "
            "number below is meaningless on a stub tree"
        )
    if measured.depth < MIN_DEPTH:
        failures.append(
            f"structure: widget depth {measured.depth} (floor {MIN_DEPTH}) -- the repolish "
            "term scales with depth, so a shallow tree cannot certify it"
        )

    # 3a. rule matching -- the stylesheet must still be the real one.
    if measured.real_rules < MIN_REAL_RULES:
        failures.append(
            f"rule matching: the built stylesheet has {measured.real_rules} rules "
            f"(floor {MIN_REAL_RULES}) -- the theme was gutted, so the fan-out below "
            "certifies a stylesheet nobody ships"
        )
    if measured.small_rules > MAX_SMALL_RULES:
        failures.append(
            f"rule matching: the comparison stylesheet has {measured.small_rules} rules; "
            "it is supposed to be a minimal one"
        )

    # 2. propagation -- a genuine switch fans out a storm, inside the call.
    if measured.storm_change_events < MIN_STORM_CHANGE_EVENTS:
        failures.append(
            f"propagation: an ungated restyle fanned out {measured.storm_change_events} "
            f"change events (floor {MIN_STORM_CHANGE_EVENTS}) -- either the tree stopped "
            "being repolished or the measurement stopped seeing it"
        )
    if measured.storm_after_events:
        share = measured.storm_all_events / measured.storm_after_events
        if share < MIN_SYNCHRONOUS_SHARE:
            failures.append(
                f"propagation: only {share:.0%} of the fan-out landed inside setStyleSheet "
                f"(floor {MIN_SYNCHRONOUS_SHARE:.0%}) -- the storm is being deferred, so the "
                "gate cannot cover it and this file's premise is wrong"
            )
    else:
        failures.append("propagation: the ungated restyle produced no event counts at all")

    # 3b. rule matching -- fewer rules must propagate measurably less.
    if not (0 < measured.small_change_events < measured.storm_change_events * MAX_SMALL_SHARE):
        failures.append(
            f"rule matching: a {measured.small_rules}-rule stylesheet fanned out "
            f"{measured.small_change_events} change events against "
            f"{measured.storm_change_events} for the {measured.real_rules}-rule one -- "
            f"expected strictly less than {MAX_SMALL_SHARE:.0%} of it, so selector work is "
            "no longer a separable term of this cost"
        )

    # 4. ours -- the scope must cover the whole storm the app's filter sees.
    if not measured.switch_rounds:
        failures.append(
            "ours: not one gated round performed a genuine switch -- every key resolved "
            "to the stylesheet already applied, so the storm, the scope and the timing "
            "below were never exercised at all (a gutted or collapsed theme)"
        )
    if not measured.filter_installed:
        failures.append(
            "ours: no app-wide Ctrl+hover filter is installed on this application, so "
            "there is nothing for the bulk-restyle scope to spare -- the term this file "
            "exists for was never exercised"
        )
    if measured.filter_closed_switch:
        failures.append(
            f"ours: the app's own Ctrl+hover filter saw {measured.filter_closed_switch} of "
            f"its {measured.filter_calls_switch} events during a real apply_theme switch "
            "with the bulk-restyle scope CLOSED (must be 0) -- apply_theme stopped opening "
            "the scope, so app-wide filters pay the full per-event Python round-trip again"
        )
    if measured.filter_calls_switch < MIN_STORM_CHANGE_EVENTS:
        failures.append(
            f"ours: the app's filter was only invoked {measured.filter_calls_switch} times "
            f"during the switch (floor {MIN_STORM_CHANGE_EVENTS}) -- the gate is open when "
            "there is nothing to skip, which would make the assertion above vacuous"
        )
    if measured.filter_closed_ungated < MIN_STORM_CHANGE_EVENTS:
        failures.append(
            f"ours: the ungated path closed the scope for only "
            f"{measured.filter_closed_ungated} of the filter's "
            f"{measured.filter_calls_ungated} events (floor {MIN_STORM_CHANGE_EVENTS}) -- if "
            "the before-picture is not a full storm, 'the scope spares it' is unproven"
        )
    if measured.redundant_change_events:
        failures.append(
            f"ours: re-applying the current theme fanned out "
            f"{measured.redundant_change_events} events -- the redundant-restyle early "
            "return is gone (it cost ~0.45 s per window before it existed)"
        )
    if measured.redundant_restyles:
        failures.append(
            f"ours: re-applying the current theme restyled the app "
            f"{measured.redundant_restyles} time(s); it must be a no-op"
        )

    # Rule-matching correctness: the stylesheet has to apply cleanly.
    parsing = [
        warning
        for warning in measured.qt_warnings
        if "stylesheet" in warning.lower() or "could not parse" in warning.lower()
    ]
    if parsing:
        failures.append(
            "rule matching: Qt could not apply the stylesheet -- " + "; ".join(parsing[:3])
        )
    return failures


class _QtMessageLog:
    """Collects Qt's own warnings; the default handler would print them all."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, mode, context, message: str) -> None:  # noqa: ANN001
        self.messages.append(message)

    def install(self) -> None:
        qInstallMessageHandler(self)


class _StormCounter(QObject):
    """App-wide counter, split by whether the bulk-restyle scope was open.

    A second filter, installed alongside the app's own, so the *counts* of the
    fan-out come from something whose only job is counting.
    """

    def __init__(self) -> None:
        super().__init__()
        self.all: Counter[str] = Counter()
        self.open_gate = 0
        self.closed_gate = 0

    def eventFilter(self, obj: object, event: QEvent) -> bool:  # noqa: N802 (Qt naming)
        self.all[QEvent.Type(event.type()).name] += 1
        if in_bulk_restyle():
            self.open_gate += 1
        else:
            self.closed_gate += 1
        return False

    def reset(self) -> None:
        self.all.clear()
        self.open_gate = 0
        self.closed_gate = 0

    @property
    def change_events(self) -> int:
        return sum(self.all[name] for name in CHANGE_TYPES)

    @property
    def total_events(self) -> int:
        return sum(self.all.values())


class _FilterView:
    """Counts the app's own filter's view of the scope, from inside that filter.

    ``gui_tooltips`` binds ``in_bulk_restyle`` at import time and calls it as the
    filter's first line, so replacing that module-level name counts exactly the
    invocations of the REAL filter -- including, which is the point, how many of
    them found the scope closed. Counting from a second filter could not answer
    that: Qt invokes both regardless.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.closed = 0
        self._real = gui_tooltips.in_bulk_restyle

    def install(self) -> None:
        real = self._real

        def probe() -> bool:
            open_now = real()
            self.calls += 1
            if not open_now:
                self.closed += 1
            return open_now

        gui_tooltips.in_bulk_restyle = probe

    def uninstall(self) -> None:
        gui_tooltips.in_bulk_restyle = self._real

    def reset(self) -> None:
        self.calls = 0
        self.closed = 0


def _widget_depth(root: QWidget) -> int:
    """Deepest widget nesting under ``root``, counted over QWidget children."""
    best = 0
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        best = max(best, depth)
        for child in node.children():
            if isinstance(child, QWidget):
                stack.append((child, depth + 1))
    return best


def _build_window(app: QApplication) -> MainWindow:
    """A real window, startup dialogs neutralized, pumped like a user session."""
    window = MainWindow()
    window._maybe_run_crash_startup_flow = lambda: None
    window._start_inference_wizard = lambda: None
    window.show()
    window.resize(1500, 1500)
    for _ in range(8):
        app.processEvents()
    # The hide/show cycle exercises the re-layout path a real session hits; the
    # section-chrome sweep certifies the same cycle for the same reason.
    window.hide()
    window.show()
    for _ in range(8):
        app.processEvents()
    return window


def _destroy_window(app: QApplication, window: MainWindow) -> None:
    """Tear down on the suite's discipline: stop timers, close, flush deletes."""
    for attr in ("_gui_shown_timer", "_crash_flow_timer", "_wizard_launch_timer"):
        timer = getattr(window, attr, None)
        if timer is not None:
            timer.stop()
    # close() can be REFUSED when a GUI-owned thread cannot stop; honour it.
    if window.close():
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def measure(
    app: QApplication, window: MainWindow, counter: _StormCounter, warnings: _QtMessageLog
) -> Measure:
    """Run every measurement once against a live window."""
    measured = Measure()
    measured.widgets = len(window.findChildren(QWidget)) + 1
    measured.depth = _widget_depth(window)
    measured.filter_installed = gui_tooltips.install_ctrl_hover_help(app) is not None

    real_css = build_stylesheet(THEMES[DEFAULT_THEME])
    other = next(key for key in sorted(THEMES) if key != DEFAULT_THEME)
    measured.real_rules = real_css.count("{")
    measured.small_rules = _SMALL_CSS.count("{")

    view = _FilterView()

    def during(style_call, *, watch_filter: bool) -> tuple[float, int, int, int, int, int, int]:
        """One restyle, sampled inside the call and again after the pump."""
        counter.reset()
        if watch_filter:
            view.reset()
        started = time.perf_counter()
        style_call()
        elapsed = (time.perf_counter() - started) * 1000
        call_change = counter.change_events
        call_open = counter.open_gate
        call_closed = counter.closed_gate
        call_total = counter.total_events
        filter_calls, filter_closed = view.calls, view.closed
        for _ in range(4):
            app.processEvents()
        return elapsed, call_change, call_open, call_closed, call_total, filter_calls, filter_closed

    # The filter's own view is counted throughout, so the ungated and gated
    # halves are measured by the same instrument.
    view.install()
    try:
        # -- ungated: the pre-gate path, straight into setStyleSheet ------------
        direct: list[tuple] = []
        for index in range(2):
            css = real_css if index % 2 == 0 else build_stylesheet(THEMES[other])
            direct.append(during(lambda c=css: app.setStyleSheet(c), watch_filter=True))
        elapsed, change, _open, closed, after, _calls, _closed = direct[-1]
        measured.storm_change_events = change
        measured.storm_all_events = closed
        measured.storm_after_events = after
        measured.filter_calls_ungated = max(row[5] for row in direct)
        measured.filter_closed_ungated = max(row[6] for row in direct)
        measured.timings_ms["ungated restyle"] = statistics.median(row[0] for row in direct)

        # -- gated: the real apply_theme --------------------------------------
        # Keys alternate so each round is a genuine switch; note that the first
        # round can still be a no-op (the ungated half leaves `other` applied) and
        # correctly early-returns, which is why the aggregates below take maxima
        # rather than the last row -- a no-op round has no calls to report.
        keys = [DEFAULT_THEME, other, DEFAULT_THEME, other]
        gated: list[tuple] = [
            during(lambda k=key: apply_theme(app, k), watch_filter=True) for key in keys
        ]
        measured.switch_change_events = max(row[1] for row in gated)
        measured.switch_open_gate_events = max(row[2] for row in gated)
        measured.filter_calls_switch = max(row[5] for row in gated)
        measured.filter_closed_switch = max(row[6] for row in gated)
        # Only rounds that really restyled count towards the timing: a round that
        # early-returned has a 0.0 in it, and a run where EVERY round did is a
        # failure rather than a timing (median of nothing, measured live as a
        # StatisticsError traceback out of a gutted theme).
        switching = [row for row in gated if row[1]]
        measured.switch_rounds = len(switching)
        measured.timings_ms["gated switch"] = (
            statistics.median(row[0] for row in switching) if switching else 0.0
        )
    finally:
        view.uninstall()

    # -- rule matching: the same tree under a minimal stylesheet ---------------
    small = [during(lambda: app.setStyleSheet(_SMALL_CSS), watch_filter=False) for _ in range(2)]
    measured.small_change_events = small[-1][1]
    measured.timings_ms["2-rule stylesheet"] = statistics.median(row[0] for row in small)

    # -- redundant: the same key twice; the second must be free ---------------
    applied = apply_theme(app, DEFAULT_THEME)
    counter.reset()
    started = time.perf_counter()
    applied_again = apply_theme(app, applied)
    measured.timings_ms["redundant re-apply"] = (time.perf_counter() - started) * 1000
    measured.redundant_change_events = counter.change_events
    measured.redundant_restyles = (
        0 if applied_again == applied and counter.change_events == 0 else 1
    )

    measured.qt_warnings = tuple(warnings.messages)
    return measured


def _report(measured: Measure) -> None:
    print(f"  structure      : {measured.widgets} widgets, depth {measured.depth}")
    print(
        f"  propagation    : {measured.storm_change_events} change events inside the call, "
        f"{measured.storm_after_events} by the end of the pump"
    )
    print(
        f"  rule matching  : {measured.real_rules} rules fan out "
        f"{measured.storm_change_events}; {measured.small_rules} rules fan out "
        f"{measured.small_change_events}"
    )
    print(
        "  ours (the gate): the app's own filter was invoked "
        f"{measured.filter_calls_switch} times during a real switch and found the scope "
        f"open every time ({measured.filter_closed_switch} closed); ungated, the same "
        f"filter saw the scope closed for {measured.filter_closed_ungated} of its "
        f"{measured.filter_calls_ungated} events"
    )
    print(
        f"  redundant switch: {measured.redundant_change_events} events, "
        f"{measured.redundant_restyles} restyle(s)"
    )
    print("  timings (reported, never asserted -- they are of this machine):")
    for name, value in measured.timings_ms.items():
        print(f"    {name:<20} {value:7.1f} ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quiet", action="store_true", help="Only report failures.")
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication([])
    warnings = _QtMessageLog()
    warnings.install()
    counter = _StormCounter()
    app.installEventFilter(counter)
    window = _build_window(app)
    try:
        measured = measure(app, window, counter, warnings)
    finally:
        _destroy_window(app, window)

    if not args.quiet:
        _report(measured)

    failures = evaluate(measured)
    if failures:
        print("THEME-SWITCH CERTIFICATION FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(
        "All good: the real window's switch cost splits into structure, propagation and "
        "rule matching -- and ours is gated out of the whole storm."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
