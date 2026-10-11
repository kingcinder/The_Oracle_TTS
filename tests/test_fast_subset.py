"""What the fast subset keeps: the GUI certifications must be inside it.

``pytest -m "not slow"`` is the loop a developer runs between changes, and the
six-theme GUI certifications are the suite's only proof that a theme reaches a
rendered window. Behind the ``slow`` marker those gates lived outside it, so a
green fast run could certify nothing at all — the same shape as a skip, and the
same reason tests/skip_audit.py exists (a green that earned itself).

This asks pytest's OWN collector which node ids a ``-m "not slow"`` run picks
up, rather than scanning the sources for a marker. A source scan would be a
second implementation of "what the subset runs", and the doctrine next door in
tests/test_skip_audit.py is that the run's own report is the evidence.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The certification gates: each runs a certifier script as a real subprocess
#: and is the local half of a named CI step. Listing a gate here is the
#: deliberate act that keeps it out of the subset's blind spot.
CERTIFICATION_GATES = (
    "tests/test_gui_theme_certification.py"
    "::test_design_token_certification_runs_locally_and_passes",
    "tests/test_gui_theme_certification.py"
    "::test_the_dialog_aspects_certify_the_studio_and_the_wizard",
    "tests/test_gui_sections_chrome.py"
    "::test_section_chrome_certification_runs_locally_and_passes",
    "tests/test_theme_switch_certification.py::test_a_real_run_certifies_and_writes_nothing",
)


def test_the_fast_subset_collects_every_gui_certification_gate() -> None:
    """The subset's own collection list, with the gates in it — or a named failure."""
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "not slow"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    detail = f"rc={collected.returncode}\nstdout tail:\n{collected.stdout[-2000:]}\nstderr:\n{collected.stderr[-2000:]}"
    assert collected.returncode == 0, f"collecting the fast subset failed\n{detail}"

    listed = set(collected.stdout.splitlines())
    # A truncated or mangled report must not read as agreement: the subset is
    # most of the suite, so a handful of lines is a broken parse, not a subset.
    assert len(listed) > 1000, (
        f"the collection report parsed to {len(listed)} lines, which cannot be the whole "
        f"fast subset — the parse is wrong rather than the subset\n{detail}"
    )
    missing = [node for node in CERTIFICATION_GATES if node not in listed]
    assert not missing, (
        f"`pytest -m \"not slow\"` does not collect {missing}, so the fast loop would "
        "report green without certifying a theme. Take the `slow` marker off the gate, "
        "or remove it from CERTIFICATION_GATES deliberately.\n" + detail
    )
