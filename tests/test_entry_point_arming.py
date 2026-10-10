"""Every GUI-bearing or native-heavy entry point arms the faulthandler net.

**Completeness-parity pin (the 71d8840 / every-occurrence idiom):** one
source-ordered assertion per unarmed entry point found by the 2026-10-08
audit — the whole list, every time, not first-match-only. Dropping any arm
(or reordering it after the entry point's Qt/native work begins) fails its
pin by name.

Coverage of the audit:
- scripts/certify_gui_themes.py   — real MainWindow offscreen, 6 themes
- scripts/certify_section_chrome.py — real MainWindow offscreen, 6 themes,
  section chrome (Script & Cast / Review / Extra Voices) — pinned below
- scripts/gui_drive.py            — real MainWindow + QMediaPlayer cycles
- scripts/u44_accessibility_harness.py — real gui_crash/gui_license dialogs
- scripts/u42_crash_repro.py      — enable_faulthandler_catch BEFORE MainWindow
- scripts/doctor.py probe children — QtMultimedia imports in bare children

Already armed (verified in this audit, not re-pinned here):
- the-oracle console script → cli.main (install + arm_native_capture)
- app_gui.launch_gui (arms before MainWindow)
- render_subprocess.main (arms in the render/preview child)
- scripts whose children pass through cli.main (crash_hunt GUI leg,
  windows_install_smoke, vulkan_ci_smoke, fresh_clone_acceptance): armed
  at the child entry, nothing to add.

Armed by the 2026-10-08 smoke-entry extension (pinned below):
- scripts/doctor.py main() — model loads + probe children in THIS process
- scripts/smoke_render.py main(), the_oracle.smoke.main,
  the_oracle.real_engine_smoke.main, scripts/perf_baseline.py main() —
  the deterministic/real-engine smoke stack, in-process native synthesis
  (the audit's earlier runpy annotation is superseded: the entries now arm
  like every other python -m path; the arming is idempotent so the
  runpy-in-process case composes unchanged)

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


def test_certify_section_chrome_arms_before_mainwindow() -> None:
    """The section-chrome sweep builds 12 real MainWindows (2 per theme):
    same native-crash surface as certify_gui_themes, same arm contract,
    pinned by name so dropping the arm fails here."""
    _assert_armed_before(
        _source("scripts/certify_section_chrome.py"),
        "certify_section_chrome.py",
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


# --- the smoke-entry extension (2026-10-08): every python -m path arms ---------


def test_doctor_main_arms_before_run() -> None:
    """The doctor process itself loads the Chatterbox model, spawns probe
    children, and imports QtMultimedia — its main() must arm before run().
    Order pin: the arm precedes apply_offline_environment (the first heavy
    import the entry does), so no model/Qt work runs unarmed."""
    source = _source("scripts/doctor.py")
    assert "crash_handlers.install()" in source and "crash_handlers.arm_native_capture()" in source, (
        "doctor.py main(): entry-point arm missing"
    )
    main_pos = source.index("def main(")
    # The probe-children arm inside _run_python_probe's boot wrapper is an
    # earlier, different arm site — search for main()'s own arm after main starts.
    arm_pos = source.index("crash_handlers.arm_native_capture()", main_pos)
    assert main_pos < arm_pos, (
        "doctor.py: the arm must be in main(), not module top-level (importing\n"
        "       the doctor as a library — tests, wrappers — must not arm anything)"
    )
    assert arm_pos < source.index("apply_offline_environment()", main_pos), (
        "doctor.py: the arm must precede the first heavy import in main()"
    )


def _assert_smoke_entry_arms(relative: str, marker: str) -> None:
    source = _source(relative)
    assert "crash_handlers.install()" in source and "crash_handlers.arm_native_capture()" in source, (
        f"{relative}: entry-point arm missing"
    )
    assert source.index("crash_handlers.arm_native_capture()") < source.index(marker), (
        f"{relative}: the arm must precede {marker!r} (the entry's heavy work)"
    )


def test_smoke_render_entry_arms_before_work() -> None:
    # Marker is the first statement that does work in main() — the bare
    # function name also matches the module-top import, which precedes main.
    _assert_smoke_entry_arms("scripts/smoke_render.py", "results = [")


def test_the_oracle_smoke_module_entry_arms_before_work() -> None:
    _assert_smoke_entry_arms("src/the_oracle/smoke.py", "result = run_deterministic_smoke_render(")


def test_real_engine_smoke_entry_arms_before_offline_env() -> None:
    _assert_smoke_entry_arms("src/the_oracle/real_engine_smoke.py", "apply_offline_environment()")


def test_perf_baseline_entry_arms_before_profiling() -> None:
    """The profiler runs the whole deterministic render in-process via runpy;
    its main() arms (the runpy'd smoke_render re-arm is then idempotent)."""
    _assert_smoke_entry_arms("scripts/perf_baseline.py", "report = profile_render(")


def test_every_python_m_smoke_entry_arm_is_consent_aware(tmp_path: Path) -> None:
    """Behavioral, not just textual: the smoke entry arm must inherit the
    handlers' fail-closed consent contract. Real `python -m` child against a
    consent-off sandbox root (root override via install()'s remember-once,
    the same semantics a launcher uses): no capture machinery at all — no
    crash_reports directory created, faulthandler not enabled. Then consent
    on: the same one-liner arms and holds the dump file. The scripts under
    test are never executed here — this pins the two-line arm idiom every
    new entry point copies, so a future entry written WITHOUT the consent
    gate (a bare faulthandler.enable) fails the idiom check."""
    import json
    import subprocess

    from the_oracle.crash import consent as crash_consent

    wired = tmp_path / "sandbox"
    wired.mkdir()
    crash_consent.write_consent(wired, False)

    off = subprocess.run(
        [sys.executable, "-c", (
            "import sys, os, json, faulthandler; "
            "sys.path.insert(0, 'src'); "
            "from the_oracle.crash import handlers; "
            f"handlers.install(root={str(wired)!r}); "
            "armed = handlers.arm_native_capture(); "
            "print(json.dumps({'armed': armed, 'enabled': faulthandler.is_enabled(), "
            "'dir_exists': os.path.isdir(os.path.join(sys.argv[1], 'crash_reports'))}))"
        ), str(wired)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    verdict = json.loads(off.stdout.strip().splitlines()[-1])
    assert verdict == {"armed": False, "enabled": False, "dir_exists": False}, (
        "the arm idiom must stay fail-closed without consent: no capture "
        "machinery, no crash_reports directory"
    )

    crash_consent.write_consent(wired, True)
    on = subprocess.run(
        [sys.executable, "-c", (
            "import sys, json, faulthandler; "
            "sys.path.insert(0, 'src'); "
            "from the_oracle.crash import handlers; "
            f"handlers.install(root={str(wired)!r}); "
            "armed = handlers.arm_native_capture(); "
            "print(json.dumps({'armed': armed, 'enabled': faulthandler.is_enabled()}))"
        ), str(wired)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    verdict = json.loads(on.stdout.strip().splitlines()[-1])
    assert verdict == {"armed": True, "enabled": True}, (
        "with consent on, the arm idiom must leave faulthandler enabled"
    )
