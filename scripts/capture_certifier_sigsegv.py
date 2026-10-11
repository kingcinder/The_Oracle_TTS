#!/usr/bin/env python3
"""Drive the faulthandler-armed certifier until it SIGSEGV's, keep the tree intact,
and root-cause the faulting frame(s) from the captured native dump."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV_PY = REPO / ".venv" / "bin" / "python"

# ── where the capture lands ──────────────────────────────────────────────────
# certify_gui.py arms into its throwaway config root (certifier_isolation.isolated_config_root),
# NOT the repo's crash_reports/.  The faulthandler dump file is created by
# crash.handlers.enable_faulthandler_catch at arm time (before the SIGSEGV), so whatever
# process runs the certifier must itself be consented + armed.  We mirror the certifier's
# own consent-mirror idiom: take the repo's existing consent decision, copy it into the
# throwaway root, so a consented machine still arms and a non-consented machine stays
# fail-closed (no new consent decision is created).
#
# The certifier redirects both XDG_CONFIG_HOME and APPDATA into the throwaway, and
# handlers._root() honors the install-time root_override.  So the dump lands at:
#   <throwaway>/crash_reports/native-crash.txt
#
# We keep the tree intact by running from a throwaway working dir (not the repo root) and
# never letting the certifier's own reap touch the repo.  We want the crash record, so we
# do NOT pass --keep to the certifier (that flag is about the certifier's throwaway config
# root, not about the crash record we are chasing); we instead snapshot the crash_reports/
# tree the certifier creates inside its throwaway at crash time.

# ── consent mirror (faithful to certifier_gui.py:944-952) ───────────────────
def mirror_consent_into(root: Path) -> bool:
    consent_src = REPO / "crash_consent.json"
    if not consent_src.is_file():
        return False
    dst = root / "crash_consent.json"
    shutil.copyfile(consent_src, dst)
    return True


def run_once(aspect: str, *, work: Path, config_root: Path) -> dict:
    """Run the certifier for one aspect; return the parsed outcome."""
    t0 = time.time()
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(config_root)
    env["APPDATA"] = str(config_root)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [str(VENV_PY), str(REPO / "scripts" / "certify_gui.py"), "--aspects", aspect],
        cwd=str(work),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return {
        "aspect": aspect,
        "rc": proc.returncode,
        "wall_s": time.time() - t0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="certify-sigsegv-work-"))
    config_root = Path(tempfile.mkdtemp(prefix="certify-sigsegv-cfg-"))
    crashed = work / "crash-reports"  # snapshot of what the certifier created inside config_root
    snapshot: dict[str, dict] = {}
    try:
        if not mirror_consent_into(config_root):
            print("WARN: no crash_consent.json in repo — certifier will run unarmed; "
                  "aborting the crash hunt", file=sys.stderr)
            return 2

        captured: str | None = None
        for aspect in ("contrast", "chrome", "studio", "wizard"):
            before = set(config_root.glob("crash_reports/**/*"))
            outcome = run_once(aspect, work=work, config_root=config_root)
            after = set(config_root.glob("crash_reports/**/*"))
            new = after - before
            snapshot[aspect] = {
                "rc": outcome["rc"],
                "wall_s": round(outcome["wall_s"], 2),
                "new_crash_artifacts": sorted(p.relative_to(config_root).as_posix() for p in new),
                "stderr_tail": outcome["stderr"][-2000:].splitlines()[-40:],
                "stdout_tail": outcome["stdout"][-400:].splitlines()[-20:],
            }
            # snapshot the certifier's crash_reports tree (the native dump is here, not in
            # the repo's crash_reports/).  Copy it out so the tree stays clean and we keep the
            # evidence even if the throwaway is later reaped.
            if crashed.exists():
                shutil.rmtree(crashed, ignore_errors=True)
            crashed.parent.mkdir(parents=True, exist_ok=True)
            if (config_root / "crash_reports").is_dir():
                shutil.copytree(config_root / "crash_reports", crashed)

            rc = outcome["rc"]
            if captured is None and (rc in (139, 245) or (rc < 0 and os.name == "posix")):
                captured = aspect
                print(f"CAPTURED SIGSEGV-class exit on aspect={aspect!r}: rc={rc} "
                      f"wall={outcome['wall_s']}s", flush=True)
                print("crash artifacts (relative to certifier throwaway config root):", flush=True)
                for p in snapshot[aspect]["new_crash_artifacts"]:
                    print(f"  {p}", flush=True)
                np = config_root / "crash_reports" / "native-crash.txt"
                if np.is_file():
                    print("native-crash.txt present (faulthandler dump captured)", flush=True)
                    print(f"  size={np.stat().st_size} bytes", flush=True)
                else:
                    print("native-crash.txt ABSENT — faulthandler did not write a dump "
                          "(consent/override/path issue, or the fault bypassed faulthandler)", flush=True)

        print("=== per-aspect summary ===", flush=True)
        for aspect, info in snapshot.items():
            star = " <== SIGSEGV captured" if aspect == captured else ""
            print(f"\n--- {aspect}{star} (rc={info['rc']}, wall={info['wall_s']}s) ---", flush=True)
            for line in info["stderr_tail"]:
                print(line, flush=True)
            print("  new crash artifacts:", flush=True)
            for p in info["new_crash_artifacts"]:
                print(f"    {p}", flush=True)
            print("  stdout tail:", flush=True)
            for line in info["stdout_tail"]:
                print(f"    {line}", flush=True)

        print(f"\nCaptured SIGSEGV aspect: {captured!r}", flush=True)

    finally:
        # leave the repo tree clean: the certifier's throwaway config root is ours to remove.
        shutil.rmtree(config_root, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
        # persist captured evidence under a git-ignored repo-local hunt dir
        hunt_dir = REPO / "crash_hunt_captures"
        hunt_dir.mkdir(parents=True, exist_ok=True)
        dest = hunt_dir / f"certify-sigsegv-{time.strftime('%Y%m%d-%H%M%S')}"
        dest.mkdir(parents=True, exist_ok=True)
        if crashed.exists():
            shutil.copytree(crashed, dest / "certifier_crash_reports")
        (dest / "outcome.json").write_text(
            json.dumps({aspect: {k: v for k, v in info.items() if k != "new_crash_artifacts"}
                        for aspect, info in snapshot.items()}, indent=2)
        )
        print(f"\nCaptured evidence retained at: {dest}", flush=True)
        print(f"  {dest / 'certifier_crash_reports'}: certifier's crash_reports/ tree "
              f"(incl. native-crash.txt if captured)", flush=True)
        print(f"  {dest / 'outcome.json'}: per-aspect outcome", flush=True)

    return 0 if captured else 3


if __name__ == "__main__":
    raise SystemExit(main())
