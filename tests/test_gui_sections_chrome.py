"""Section chrome for the previously bare areas (Phase 3 of the legibility
pass): Script & Cast (path rows + cast bar), Review (the table), and Extra
Voices must be QHSectionGroup sections with registry entries and working
persistence, per the approved design.

The unit pins below drive a faked window. The last two tests are the
delivery of the same contract: the CI steps that run the REAL window per
theme (one named step per aspect, so the job list names the certification
that broke), and — since 2026-10-10 — that same six-theme certification
executed by ``pytest`` itself, so a chrome regression fails a developer's run
and not only the gate.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from the_oracle.gui_sections import QHSectionGroup, _COLLAPSED_HEADER_HEIGHT
from the_oracle.gui_themes import THEMES
from tests.test_app_gui_profiles import _build_window

REPO_ROOT = Path(__file__).resolve().parents[1]
CERTIFIER = REPO_ROOT / "scripts" / "certify_gui.py"


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_review_table_is_collapsible_resizable_section(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert "table" in window._section_registry
        section, splitter, index = window._section_registry["table"]
        assert splitter is window._lower_splitter
        assert index == 0
        assert isinstance(section, QHSectionGroup)
        # The table lives inside the section.
        assert window.table.parentWidget() is section
        assert section.is_collapsed() is False
        section.set_collapsed(True)
        assert section.is_collapsed() is True
        # Collapse clips the section to its header (mechanism proven by the
        # Status/Errors section).
        assert section.maximumHeight() == _COLLAPSED_HEADER_HEIGHT
        section.set_collapsed(False)
        assert section.is_collapsed() is False
        assert section.maximumHeight() > _COLLAPSED_HEADER_HEIGHT
        # Status/Errors keeps its slot and index.
        assert "status" in window._section_registry
        _status, status_splitter, status_index = window._section_registry["status"]
        assert status_splitter is window._lower_splitter
        assert status_index == 1
    finally:
        window.close()


def test_paths_section_wraps_path_rows_and_cast_bar(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert "paths" in window._section_registry
        section, splitter, index = window._section_registry["paths"]
        assert splitter is window._paths_splitter
        assert index == 0
        assert isinstance(section, QHSectionGroup)
        assert "Script & Cast" in section.title()
        # Path fields and the cast bar live inside the section.
        assert window.input_path.parentWidget() is section
        assert window.outdir_path.parentWidget() is section
        assert window.output_name.parentWidget() is section
        assert window.manage_cast_button.parentWidget() is section
        # The Analyze/Render row stays bare: it is outside the section.
        assert window.analyze_button.parentWidget() is not section
        assert section.is_collapsed() is False
        section.set_collapsed(True)
        assert section.maximumHeight() == _COLLAPSED_HEADER_HEIGHT
        section.set_collapsed(False)
    finally:
        window.close()


def test_extra_voices_section_registered_and_hidden_for_two_speakers(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        assert "extra_voices" in window._section_registry
        section, splitter, index = window._section_registry["extra_voices"]
        assert splitter is window._sections_splitter
        assert index == 3
        assert isinstance(section, QHSectionGroup)
        assert window.extra_speaker_scroll.parentWidget() is section
        # A fresh window has only speakers A and B: the column is hidden.
        assert section.isHidden() is True
    finally:
        window.close()


def test_hidden_extra_voices_pane_does_not_break_size_slider(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        shared, _splitter, _index = window._section_registry["shared"]
        assert window._section_registry["extra_voices"][0].isHidden() is True
        # Redistribution must skip the hidden pane rather than misalign or
        # raise (Qt setSizes only honors visible-pane lists).
        shared.set_size_share(70)
        assert shared.size_share() == 70
        sizes = window._sections_splitter.sizes()
        assert len(sizes) == 4
        assert sum(sizes) > 0
        assert sizes[0] > 0
    finally:
        window.close()


def test_section_title_elides_before_header_chrome_at_floor_width(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Regression pin for the title-under-chrome overlap: at the 260 px
    section floor the long "Shared Render Settings" title must be elided
    before it reaches the size slider, and the original text kept for when
    the pane widens again."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window.resize(1160, 760)
        window.show()
        qt_app.processEvents()
        window._sections_splitter.setSizes([260, 260, 389])
        qt_app.processEvents()
        section, _splitter, _index = window._section_registry["shared"]
        assert section.width() <= 270, "expected the pane at its floor"
        assert section._full_title == "Shared Render Settings"
        assert section.title() != section._full_title, (
            "long title was not elided at the section floor — it paints "
            "under the header chrome"
        )
        assert section.title().endswith("...")
        # Widening restores the full text.
        window._sections_splitter.setSizes([460, 360, 360])
        qt_app.processEvents()
        assert section.title() == "Shared Render Settings"
    finally:
        window.close()


def test_paths_collapse_and_splitter_survive_persist_round_trip(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # _persist_workspace_layout no-ops until _on_gui_shown has run (the
    # restore must never be overwritten by widget defaults), so show each
    # window first, exactly like the real app.
    window, _paths = _build_window(monkeypatch, tmp_path)
    try:
        window.show()
        qt_app.processEvents()
        section, _splitter, _index = window._section_registry["paths"]
        assert section.is_collapsed() is False
        window._paths_splitter.setSizes([150, 650])
        window._persist_workspace_layout()
    finally:
        window.close()

    # Fresh window over the same isolated settings: the splitter share is
    # restored while expanded (a later collapse clips the pane to its
    # 30 px header, which is a different state and would overwrite the
    # share, so sizes are asserted before any collapsing).
    window2, _paths2 = _build_window(monkeypatch, tmp_path)
    try:
        window2.show()
        qt_app.processEvents()
        section2, splitter2, index2 = window2._section_registry["paths"]
        assert splitter2 is window2._paths_splitter
        assert index2 == 0
        assert section2.is_collapsed() is False
        sizes = window2._paths_splitter.sizes()
        actual_share = sizes[0] / sum(sizes)
        assert abs(actual_share - 150 / 800) <= 0.04, f"paths pane share not restored: {sizes}"
        # Registry-driven restore covers the new keys too: the Review and
        # Extra Voices entries flow through _apply_workspace_layout.
        assert "table" in window2._section_registry
        assert "extra_voices" in window2._section_registry

        # Second hop: collapse now persists and restores as collapsed.
        section2.set_collapsed(True)
        window2._persist_workspace_layout()
    finally:
        window2.close()

    window3, _paths3 = _build_window(monkeypatch, tmp_path)
    try:
        window3.show()
        qt_app.processEvents()
        section3, _splitter3, _index3 = window3._section_registry["paths"]
        assert section3.is_collapsed() is True
    finally:
        window3.close()


# --- CI: every aspect is a named step on every push ---------------------------


def _certification_step(aspect: str, os_name: str) -> str:
    """The workflow's step name for one aspect on one runner."""
    return f"GUI Six-Theme Certification - {aspect} ({os_name})"


def _harness_aspects() -> tuple[str, ...]:
    """The harness's own aspect inventory, off its ``ASPECTS`` tuple.

    That tuple is what ``--aspects`` validates against, so it is the contract
    CI has to cover. Read from the source rather than imported: importing the
    script makes its throwaway config root and redirects the config environment
    before any ``the_oracle`` import — the isolation that makes the sweep safe
    to run on a developer's machine — so an import here would make the test
    part of what it grades.
    """
    source = CERTIFIER.read_text(encoding="utf-8")
    match = re.search(r"^ASPECTS = \(([^)]*)\)", source, re.MULTILINE)
    assert match, "scripts/certify_gui.py no longer declares an ASPECTS tuple"
    return tuple(re.findall(r'"([a-z_]+)"', match.group(1)))


def test_workflow_runs_each_gui_certification_aspect_as_its_own_named_step() -> None:
    """The unit pins above drive a faked window; the certification certifies
    the same chrome on the REAL MainWindow per theme. Gated only by operating
    system, so it runs on every push and pull_request -- the `on:` block
    needs no per-step opt-in.

    One NAMED step per aspect per OS, because the job list is where a failure
    gets read: run as one sweep, a contrast regression and a chrome regression
    are the same red step and the reason has to be dug out of a log. The step
    set must match the harness's aspects in both directions — an aspect added
    to ``certify_gui.py`` fails here until it has a step, and a step for an
    aspect that no longer exists fails too.
    """
    aspects = _harness_aspects()
    assert aspects == ("contrast", "chrome", "studio", "wizard"), (
        "the harness's aspects changed: wire the new one into ci.yml as its own "
        f"named step on both operating systems, then widen this pin: {aspects}"
    )

    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = {step.get("name"): step for step in workflow["jobs"]["test"]["steps"]}

    named = {name for name in steps if name.startswith("GUI Six-Theme Certification")}
    expected = {
        _certification_step(aspect, os_name)
        for aspect in aspects
        for os_name in ("Linux", "Windows")
    }
    assert named == expected, (
        "CI certifies a different set of aspects than the harness declares: "
        f"missing={sorted(expected - named)} extra={sorted(named - expected)}"
    )

    for aspect in aspects:
        for os_name, interpreter, guard, shell in (
            ("Linux", ".venv/bin/python", "runner.os == 'Linux'", "bash"),
            ("Windows", ".venv\\Scripts\\python.exe", "runner.os == 'Windows'", "pwsh"),
        ):
            step = steps[_certification_step(aspect, os_name)]
            assert "scripts/certify_gui.py" in step["run"]
            assert f"--aspects {aspect}" in step["run"], (
                f"the {aspect} step must select its own aspect by name, else it "
                f"certifies something else under this name: {step['run']}"
            )
            assert interpreter in step["run"]
            assert step["if"] == guard
            assert step["shell"] == shell


# --- ...and the same certification runs locally -------------------------------

#: One theme's summary line: ``  tape_deck: OK - sections render, sliders move,
#: collapse clips, layout round-trips``. A theme that failed prints
#: ``FAILED - see the list below`` instead, so it simply will not match.
_OK_LINE = re.compile(r"^\s+(?P<key>[a-z_]+): OK - sections render", re.MULTILINE)

#: The gate-trace evidence line: ``  tape_deck: OK - gate trace: gui_cast 6,
#: gui_chrome 1, gui_sections 8, gui_widgets 14 - all open``. One per
#: certified theme — the chrome aspect's construction-time bulk-restyle
#: trace reporting its per-module counts on that theme's real window.
_GATE_LINE = re.compile(r"^\s+(?P<key>[a-z_]+): OK - gate trace:", re.MULTILINE)


def test_section_chrome_report_parser_cannot_pass_on_a_truncated_report() -> None:
    """The local gate leans on this parse, so pin it here too: a report cut
    short must yield fewer keys, never the full six, so an empty or mangled
    stdout cannot pass the gate vacuously."""
    sample = (
        "  tape_deck: OK - sections render, sliders move, collapse clips, layout round-trips\n"
        "  sephiroth: FAILED - see the list below\n"
    )
    keys = _OK_LINE.findall(sample)
    assert keys == ["tape_deck"], f"a FAILED theme must not be read as certified: {keys}"
    assert _OK_LINE.findall("") == []


def test_section_chrome_certification_runs_locally_and_passes() -> None:
    """Run the six-theme section-chrome certification as a real subprocess on
    the real MainWindow — the same script CI runs as a named step — so a
    chrome regression fails a developer's ``pytest``, not only the gate.

    ``scripts/certify_gui.py`` is one harness with two aspects: this test
    drives ``chrome`` (~14 s, 13 real windows — the expensive half), and
    tests/test_gui_theme_certification.py drives ``contrast`` (~8 s). CI runs
    the default, both, in one 14 s sweep. tests/test_certifier_hermeticity.py
    executes the same script for its own claim (it must wrap its run in
    snapshots to do so); this test owns the certification verdict itself, and
    the duplication is deliberate: if the hermeticity contract is ever
    re-scoped or deleted, the chrome gate must not disappear with it.

    Exit 0 alone is not the verdict. Every one of the six themes must report
    its own OK line (a theme that failed the sweep prints ``FAILED``), so a
    certifier returning 0 without certifying anything cannot pass here.

    Deliberately NOT ``slow``-marked: this is the local half of a named CI
    step, and a ``-m "not slow"`` loop that dropped it would report green
    without certifying a single theme. tests/test_fast_subset.py asks pytest's
    own collector whether the fast subset still keeps this gate.
    """
    proc = subprocess.run(
        [sys.executable, str(CERTIFIER), "--aspects", "chrome"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    detail = (
        f"rc={proc.returncode}\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )
    assert proc.returncode == 0, f"the section-chrome certification failed\n{detail}"

    certified = _OK_LINE.findall(proc.stdout)
    assert set(certified) == set(THEMES), (
        f"the sweep did not report every theme as OK (got {sorted(certified)})\n{detail}"
    )
    assert "All good across all 6 themes" in proc.stdout, (
        f"the certification did not print its summary line\n{detail}"
    )
    # The chrome aspect's own closing line, not just the prefix: a harness
    # that printed the contrast line's summary while skipping chrome would
    # otherwise pass on the shared wording.
    assert "sections render, sliders move, collapse clips" in proc.stdout, detail
    # The gate-trace evidence, one line per theme: the sweep must actually run
    # the construction-time bulk-restyle trace on each of the six real
    # windows and report its per-module counts — exiting 0 without the trace
    # would drop the runtime gate proof the chrome aspect now carries.
    gate_keys = _GATE_LINE.findall(proc.stdout)
    assert set(gate_keys) == set(THEMES), (
        f"the sweep did not report a gate trace for every theme "
        f"(got {sorted(gate_keys)})\n{detail}"
    )
