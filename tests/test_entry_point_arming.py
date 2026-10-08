"""Every GUI-bearing or native-heavy entry point arms the faulthandler net.

**Completeness-parity pin (the 71d8840 / every-occurrence idiom):** one
source-ordered assertion per unarmed entry point found by the 2026-10-08
audit — the whole list, every time, not first-match-only. Dropping any arm
(or reordering it after the entry point's Qt/native work begins) fails its
pin by name.

Coverage of the audit:
- scripts/certify_gui_themes.py   — real MainWindow offscreen, 6 themes
- scripts/gui_drive.py            — real MainWindow + QMediaPlayer cycles
- scripts/u44_accessibility_harness.py — real gui_crash/gui_license dialogs
- scripts/u42_crash_repro.py      — enable_faulthandler_catch BEFORE MainWindow
- scripts/doctor.py probe children — QtMultimedia imports in bare children

Already armed (verified in this audit, not re-pinned here):
- the-oracle console script → cli.main (install + arm_native_capture)
- app_gui.launch_gui (arms before MainWindow)
- render_subprocess.main (arms in the render/preview child)
- real_engine_smoke / smoke_render / perf_baseline: runpy-in-process, not
  separate OS processes — they annotate the parent's dump file. Recorded in
  STATE's audit entry as accepted.
- scripts whose children pass through cli.main (crash_hunt GUI leg,
  windows_install_smoke, vulkan_ci_smoke, fresh_clone_acceptance): armed
  at the child entry, nothing to add.

**Mutation contract:** removing an arm call fails that script's pin;
reordering an arm after the entry point's MainWindow/QApplication/media
line fails its order assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

ARM_CALL = "crash_handlers.arm_native_capture()"


def _source(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _assert_armed_before(source: str, relative: str, before_marker: str) -> None:
    assert ARM_CALL in source, f"{relative}: no arm_native_capture call — net unarmed"
    assert source.index(ARM_CALL) < source.index(before_marker), (
        f"{relative}: arm must run before {before_marker!r} (the entry point's "
        "Qt/native work), not after"
    )
    assert "from the_oracle.crash import handlers as crash_handlers" in source, (
        f"{relative}: arm import missing"
    )


def test_certify_gui_themes_arms_before_mainwindow() -> None:
    _assert_armed_before(
        _source("scripts/certify_gui_themes.py"),
        "certify_gui_themes.py",
        "app = QApplication.instance()",
    )


def test_gui_drive_arms_before_mainwindow() -> None:
    _assert_armed_before(
        _source("scripts/gui_drive.py"),
        "gui_drive.py",
        "app = QApplication(sys.argv)",
    )


def test_u44_accessibility_harness_arms_before_dialogs() -> None:
    _assert_armed_before(
        _source("scripts/u44_accessibility_harness.py"),
        "u44_accessibility_harness.py",
        "parent = QWidget()",
    )


def test_u42_crash_repro_arms_before_mainwindow() -> None:
    # u42 arms via enable_faulthandler_catch (the consent-transition API);
    # the audit standardizes launch-path entries on arm_native_capture, so
    # pin EITHER spelling before MainWindow().
    source = _source("scripts/u42_crash_repro.py")
    armed = any(
        call in source
        for call in (
            "crash_handlers.arm_native_capture()",
            "crash_handlers.enable_faulthandler_catch(repo)",
        )
    )
    assert armed, "u42_crash_repro.py: no arm call before MainWindow()"
    first_arm = min(
        (source.index(call) for call in (
            "crash_handlers.arm_native_capture()",
            "crash_handlers.enable_faulthandler_catch(repo)",
        ) if call in source),
    )
    assert first_arm < source.index("app_gui.MainWindow()"), (
        "u42_crash_repro.py: arm must precede the MainWindow construction"
    )


def test_doctor_probe_children_arm() -> None:
    """Doctor probe children import QtMultimedia and construct QMediaPlayer
    in a bare `python -c` child: the ONE historically-proven crash class
    (crash-evidence §A) that no parent net can see across. The arm belongs
    inside _run_python_probe's child preamble so every probe inherits it."""
    source = _source("scripts/doctor.py")
    assert ".arm_native_capture()" in source, "doctor.py: probe children never arm the net"
    # The arm lives inside _run_python_probe's boot wrapper (it must arm the
    # spawned child before the probe code execs), so: def < arm < the
    # probe-exec line. Dropping the arm fails the first assert; moving it
    # after the exec (or out of the boot wrapper) fails the order assert.
    assert source.index(".arm_native_capture()") > source.index("def _run_python_probe("), (
        "doctor.py: the arm must be inside _run_python_probe's boot wrapper"
    )
    assert source.index(".arm_native_capture()") < source.index("'<doctor-probe>'"), (
        "doctor.py: the arm must run before the probe code execs"
    )
    assert "from the_oracle.crash import handlers" in source, "doctor.py: arm import missing"
