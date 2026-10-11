"""``certify_gui`` is hermetic: no real config, no repo logs.

The sweep builds thirteen REAL MainWindows (two per theme plus the menu
pass) and rewrites settings six times, so "isolation" cannot rest on intent —
it is a measurable contract with two halves:

1. **Behavioral** — a full subprocess run of the script (the exact shape CI
   runs) must leave the real repo tree AND the real app-config root
   byte-identical: before/after snapshots, git-ignored space included (the
   regression class that mattered lived in git-ignored ``Output/logs/``).
   Measured pre-fix on a consented machine, that run changed exactly
   ``Output/logs/gui_startup_timing.json`` (every ``MainWindow.__init__``
   writes it) and ``Output/logs/gui_prewarm_timing.json`` (the startup
   prewarm) — those two files are what the path redirect exists to stop.
   The theme certifier this sweep absorbed had the same two writes AND
   rewrote the real app settings file once per theme (backed up to disk and
   restored in ``finally``, so only a crash lost the user's theme, layout and
   last input — measured before the merge, on a developer machine); the same
   redirect is what now stops all three writes at the root.
2. **Source** — the channels no snapshot on a typical machine can observe:
   the native-crash dump arm (consent-ON machines append to an existing
   ``crash_reports/native-crash.txt`` without changing its bytes, and a
   consent-OFF machine creates nothing, so both look clean), the config
   redirect's position before the first ``the_oracle`` import, and the
   drift between the script's patch and ``app_gui``'s real seam.

Mutation contract (proven live, reverted byte-identical):
- dropping the ``ensure_repo_default_paths`` redirect fails the behavioral
  test naming ``Output/logs/gui_startup_timing.json``;
- dropping the ``XDG_CONFIG_HOME``/``APPDATA`` redirect fails the behavioral
  test naming real-config files (the sweep rewrites app settings 6×) — that
  redirect is the shared helper's job since 2026-10-11, so the mutation is made
  in ``scripts/certifier_isolation.py`` and this same behavioral run is the
  test that catches it;
- dropping ``install(root=_CONFIG_ROOT)`` or the consent mirror fails
  ``test_crash_channel_stays_in_the_throwaway`` (source-level — the
  behavioral half cannot see this channel on an already-armed machine);
- introducing ``configure_logging``/``launch_gui`` fails
  ``test_no_repo_log_entry_points``.

Honest limits: the snapshot window is ~20 s wide, so a concurrent writer in
this shared checkout (another session committing, editing tracked sources,
or the developer's own GUI) can report a change it did not cause — the
failure names every path, which makes that diagnosable rather than
mysterious. One concrete shape of it, hit while merging the two certifiers
into this one: a sibling agent's in-flight edit to ``gui_sections.py`` lands
inside the window and the behavioral test reports it, correctly, as a change
this run did not make. A park left in the repo root
by a crashed predecessor is merged home *by* the run, which the snapshot
will rightly report: that is recovery writing, not sweep noise. The
throwaway ``/tmp`` litter a SIGSEGV leaves behind and OS-level caches
outside the app config root (fontconfig, Qt plugin caches under ``HOME``)
are out of this contract's scope.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "certify_gui.py"

pytestmark = pytest.mark.slow


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _package_source(name: str) -> str:
    return (REPO_ROOT / "src" / "the_oracle" / name).read_text(encoding="utf-8")


# --- behavioral: a real run writes nothing ----------------------------------


def test_sweep_never_touches_the_real_repo_or_config() -> None:
    """Run the real script as CI does; the repo and the real config root must
    come out byte-identical (see the module docstring for the measured
    pre-fix writes and the honest limits)."""
    from the_oracle.platform_support import user_config_root

    from tests.helpers import repo_tree_snapshot, snapshot_differences

    config_root = user_config_root() / "the_oracle"
    before_repo = repo_tree_snapshot(REPO_ROOT)
    before_config = repo_tree_snapshot(config_root) if config_root.exists() else {}

    proc = subprocess.run(
        [sys.executable, "scripts/certify_gui.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )

    after_repo = repo_tree_snapshot(REPO_ROOT)
    after_config = repo_tree_snapshot(config_root) if config_root.exists() else {}

    detail = (
        f"rc={proc.returncode}\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )
    assert proc.returncode == 0, f"the sweep itself failed\n{detail}"

    for label, before, after in (
        ("real repo tree", before_repo, after_repo),
        ("real app-config root", before_config, after_config),
    ):
        created, changed, removed = snapshot_differences(before, after)
        assert not (created or changed or removed), (
            f"the sweep wrote to the {label}: "
            f"created={created} changed={changed} removed={removed}\n{detail}"
        )


# --- source: channels a snapshot cannot see ---------------------------------


def test_config_root_redirect_precedes_every_the_oracle_import() -> None:
    """The env redirect is the first thing the module does after its
    docstring: a ``the_oracle`` import resolved before it would read (and
    potentially cache) the real config root.

    The redirect itself moved to the certifier family's shared helper on
    2026-10-11, so what this pins is the ORDER and the delegation: a call to the
    shared ``isolated_config_root``, above every ``the_oracle`` import. The
    substance — that the helper redirects BOTH ``XDG_CONFIG_HOME`` and
    ``%APPDATA%`` and arms the reap — is pinned where it lives, in
    tests/test_certifier_isolation.py.
    """
    source = _source()
    first_import = re.search(r"^(?:from|import) the_oracle", source, re.MULTILINE)
    assert first_import is not None, "no the_oracle import found — the scan went blind"
    assert "from certifier_isolation import isolated_config_root, reap_isolated_root" in source, (
        "the throwaway root no longer comes from the shared helper — a hand-rolled "
        "second copy is what this refactor removed, and a copy has no owner"
    )
    redirect = source.index("_CONFIG_ROOT = isolated_config_root(")
    assert redirect < first_import.start(), "config redirect must precede every the_oracle import"
    assert 'isolated_config_root("oracle_gui_cert_cfg_")' in source, (
        "the sweep's throwaway root must keep its own prefix, so a leaked root names its sweep"
    )


def test_window_default_paths_are_rooted_at_the_throwaway() -> None:
    """MainWindow's ``self.paths`` (and the ``Output/logs`` timing writes
    hanging off it) must resolve under the throwaway for the whole run —
    through the REAL builder, not a hand-rolled stub that could drift from
    ``OraclePaths``."""
    source = _source()
    assert "_app_gui.ensure_repo_default_paths = _hermetic_paths" in source, (
        "the paths seam is no longer redirected — MainWindow would write "
        "Output/logs/gui_startup_timing.json into the real repo again"
    )
    assert "return ensure_repo_default_paths(_CONFIG_ROOT)" in source, (
        "_hermetic_paths must delegate to the real builder with the "
        "throwaway root, not reconstruct OraclePaths itself"
    )
    # Redirect before the sweep builds anything.
    assert source.index("_app_gui.ensure_repo_default_paths") < source.index("for key in THEMES:")
    # Drift guard: app_gui must still expose the seam the script patches —
    # a rename there would leave this redirect silently dead.
    assert "ensure_repo_default_paths(" in _package_source("app_gui.py"), (
        "app_gui no longer calls ensure_repo_default_paths — the script's "
        "redirect no-ops and the behavioral test is the only net left"
    )


def test_crash_channel_stays_in_the_throwaway() -> None:
    """Install the root override BEFORE the arm (the bare ``arm_native_capture()``
    resolves ``_root()``, which the override repoints), and mirror an existing
    consent decision byte-for-byte so a consented machine still arms while a
    fresh checkout stays fail-closed. A snapshot cannot see this channel: on
    an already-armed machine the dump file's bytes do not change."""
    source = _source()
    install_pos = source.index("crash_handlers.install(root=_CONFIG_ROOT)")
    mirror_pos = source.index("shutil.copyfile(")
    arm_pos = source.index("crash_handlers.arm_native_capture()")
    qt_pos = source.index("app = QApplication.instance()")
    assert install_pos < mirror_pos < arm_pos < qt_pos, (
        "order must be: hermetic root installed, consent mirrored, arm, "
        "then Qt — an arm before the override writes the repo crash store, "
        "a mirror after the arm leaves a consented machine unarmed"
    )
    assert "crash_consent.read_consent(REPO_ROOT)" in source, (
        "the mirror must be conditional on the REAL consent file — copying "
        "unconditionally (or fabricating consent) would arm machines that "
        "never opted in"
    )
    # Vacuity: the owners really expose what the script leans on.
    assert "def install(root" in _package_source("crash/handlers.py")
    assert "def arm_native_capture(root" in _package_source("crash/handlers.py")
    assert "def consent_path(" in _package_source("crash/consent.py")


def test_no_repo_log_entry_points() -> None:
    """The two writers behind the repo's ``logs/`` and ``Output/logs/`` log
    families must never be reachable from this script: ``configure_logging``
    appends to the repo-local ``logs/oracle.log``, and ``launch_gui`` writes
    ``Output/logs/gui_launch_timing.json`` unconditionally."""
    source = _source()
    for forbidden in ("configure_logging(", "launch_gui(", "gui_launch_timing"):
        assert forbidden not in source, f"the sweep must not call {forbidden}"
