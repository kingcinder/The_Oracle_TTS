#!/usr/bin/env python3
"""Commercial alpha soft-launch gate.

One command that answers: *can this tree be distributed commercially today, and
if not, exactly what is missing?* It composes the repository's existing release
checks with the commercial checks that have no other owner:

* version / CHANGELOG / banner agreement (``scripts/release.py``);
* a registered production license verify key;
* a final, counsel-ready EULA (the draft is a blocker by design);
* generated third-party notices that match the release tree;
* a recorded decision for every strong-copyleft or unknown dependency;
* truthful provenance for every shipped voice asset;
* the version's tracked checksum manifest and built artifacts;
* the documented Windows-experimental scope;
* the support-bundle capability.

Exit status is 0 only when every blocker is cleared. Warnings are printed and
listed but do not change the exit status. The gate never performs network I/O
and never edits the tree.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

import release  # noqa: E402
from the_oracle import third_party  # noqa: E402

BLOCKER = "blocker"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True, slots=True)
class Finding:
    """One gate result. ``owner_action`` is required for a blocker."""

    key: str
    ok: bool
    severity: str
    detail: str
    owner_action: str = ""


def _eula_problems(root: Path) -> list[str]:
    final = root / "EULA.md"
    draft = root / "EULA-DRAFT.md"
    path = final if final.is_file() else draft
    if not path.is_file():
        return ["EULA.md (or EULA-DRAFT.md) is missing"]
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    if "draft" in path.name.lower() or re.search(r"\bdraft\b", text, re.IGNORECASE):
        problems.append(f"{path.name} is still marked as a draft")
    if re.search(r"\[insert", text, re.IGNORECASE):
        problems.append("the governing-law/venue section still contains an [insert ...] placeholder")
    if not re.search(r"governing law", text, re.IGNORECASE):
        problems.append("no governing-law section is present")
    return problems


def _tracked_voice_assets(root: Path) -> set[str]:
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files", "Seashells"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).splitlines()
        return {path for path in tracked if path.lower().endswith(".wav")}
    except (OSError, subprocess.CalledProcessError):
        return {
            str(path.relative_to(root))
            for path in (root / "Seashells").rglob("*.wav")
            if path.is_file()
        }


def _inventory_paths(root: Path) -> set[str]:
    path = root / "Seashells" / "ASSETS.md"
    if not path.is_file():
        return set()
    return set(re.findall(r"`(Seashells/[^`]+\.wav)`", path.read_text(encoding="utf-8")))


def _voice_asset_problems(root: Path) -> list[str]:
    inventory = root / "Seashells" / "ASSETS.md"
    if not inventory.is_file():
        return ["Seashells/ASSETS.md is missing"]
    text = inventory.read_text(encoding="utf-8")
    problems: list[str] = []
    for forbidden in ("undocumented", "unknown", "tbd", "todo"):
        if forbidden in text.lower():
            problems.append(f"Seashells/ASSETS.md still contains {forbidden!r}")
    tracked = _tracked_voice_assets(root)
    documented = _inventory_paths(root)
    missing = sorted(tracked - documented)
    if missing:
        problems.append(f"voice assets without an inventory entry: {', '.join(missing)}")
    stale = sorted(documented - tracked)
    if stale:
        problems.append(f"inventory entries with no tracked file: {', '.join(stale)}")
    root_rows = [
        line
        for line in text.splitlines()
        if line.startswith("| `Seashells/") and "`Seashells/generic/" not in line
    ]
    for row in root_rows:
        if "Owner-owned" not in row or "all rights reserved" not in row.lower():
            problems.append("a root-level recording is not documented as Owner-owned: " + row[:100])
    return problems


def _license_policy_findings(root: Path, inventory: third_party.Inventory) -> list[Finding]:
    path = root / "scripts" / "license_review.json"
    if not path.is_file():
        return [
            Finding(
                "license_policy",
                False,
                BLOCKER,
                "scripts/license_review.json is missing; no dependency decision can be recorded.",
                "Create scripts/license_review.json with the allow/review decisions.",
            )
        ]
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [
            Finding(
                "license_policy",
                False,
                BLOCKER,
                f"scripts/license_review.json cannot be read ({error}).",
                "Repair the JSON policy file.",
            )
        ]
    allowed = {
        third_party.canonicalize_name(name)
        for name in (policy.get("allow") or [])
        if isinstance(name, str)
    }
    review = {
        third_party.canonicalize_name(name): value
        for name, value in (policy.get("review") or {}).items()
        if isinstance(name, str)
    }
    findings: list[Finding] = []
    # The license policy governs the software the app imports or ships as a
    # binary. Voice/model assets have their own attribution sections in the
    # notices; they are not software-license decisions.
    software_kinds = {"python", "bundled-binary"}
    strong = [
        component
        for component in inventory.components
        if component.kind in software_kinds
        and component.classification in {third_party.STRONG_COPYLEFT, third_party.UNKNOWN}
    ]
    for component in strong:
        key = third_party.canonicalize_name(component.name)
        if key in allowed:
            findings.append(
                Finding(
                    "license_policy",
                    True,
                    INFO,
                    f"{component.name} ({component.license}) is explicitly allowed.",
                )
            )
            continue
        reason = ""
        if key in review:
            value = review[key] or {}
            reason = value.get("reason") if isinstance(value, dict) else str(value)
        findings.append(
            Finding(
                "license_policy",
                False,
                BLOCKER,
                f"{component.name} is {component.license} ({component.classification}) and no distribution decision is recorded."
                + (f" Recorded reason: {reason}" if reason else ""),
                "Replace the component, separate it from the proprietary distribution, or record an allow decision in scripts/license_review.json after legal review.",
            )
        )
    weak = [
        component
        for component in inventory.components
        if component.kind in software_kinds
        and component.classification == third_party.WEAK_COPYLEFT
    ]
    if weak:
        findings.append(
            Finding(
                "license_policy",
                True,
                WARNING,
                "Weak-copyleft components require their notices and replacement/source obligations: "
                + ", ".join(sorted({component.name for component in weak})),
                "Keep the notices current and document the LGPL relinking/source offer before distribution.",
            )
        )
    if not strong:
        findings.append(Finding("license_policy", True, INFO, "No strong-copyleft or unknown runtime component."))
    return findings


def _support_bundle_available() -> bool:
    return importlib.util.find_spec("the_oracle.support_bundle") is not None


def audit(root: Path = REPO_ROOT, *, today: date | None = None) -> list[Finding]:
    """Every commercial readiness check, in report order."""
    root = Path(root).resolve()
    findings: list[Finding] = []

    version = release.read_version(root) or "unknown"
    try:
        version_report = release.check_report(root, today=today)
    except Exception as error:  # release metadata must never crash the gate
        version_report = release.CheckReport(problems=[f"release metadata could not be checked: {error}"])
    if version_report.problems:
        findings.append(
            Finding(
                "version",
                False,
                BLOCKER,
                "; ".join(version_report.problems),
                "Fix the version/banner/CHANGELOG disagreement with scripts/release.py --check.",
            )
        )
    else:
        findings.append(
            Finding(
                "version",
                True,
                INFO,
                f"Version {version}, banners, and CHANGELOG agree"
                + (" (post-cut date state is by design)." if version_report.post_cut else "."),
            )
        )

    key_problems = release.licensing_key_problems(root)
    if key_problems:
        findings.append(
            Finding(
                "licensing_key",
                False,
                BLOCKER,
                "; ".join(key_problems),
                "Register the production verify key in src/the_oracle/licensing/keys.py (see docs/LICENSING_OPS.md §1).",
            )
        )
    else:
        findings.append(Finding("licensing_key", True, INFO, "A production verify key is registered."))

    eula_problems = _eula_problems(root)
    if eula_problems:
        findings.append(
            Finding(
                "eula",
                False,
                BLOCKER,
                "; ".join(eula_problems),
                "Complete the governing-law/venue section, remove the DRAFT marker (rename EULA-DRAFT.md to EULA.md), and have counsel review it.",
            )
        )
    else:
        findings.append(Finding("eula", True, INFO, "The EULA is final and names its governing law."))

    inventory = third_party.build_inventory(root, version=version)
    notice_problems = third_party.notice_problems(root, inventory=inventory)
    if inventory.problems:
        notice_problems.extend(inventory.problems)
    if notice_problems:
        findings.append(
            Finding(
                "third_party_notices",
                False,
                BLOCKER,
                "; ".join(notice_problems),
                "Regenerate THIRD_PARTY_NOTICES.md with scripts/generate_third_party_notices.py.",
            )
        )
    else:
        findings.append(
            Finding(
                "third_party_notices",
                True,
                INFO,
                f"{len(inventory.components)} third-party components are inventoried and noticed.",
            )
        )

    findings.extend(_license_policy_findings(root, inventory))

    voice_problems = _voice_asset_problems(root)
    if voice_problems:
        findings.append(
            Finding(
                "voice_assets",
                False,
                BLOCKER,
                "; ".join(voice_problems),
                "Give every shipped recording a documented source/license in Seashells/ASSETS.md; no undocumented or placeholder rows.",
            )
        )
    else:
        findings.append(Finding("voice_assets", True, INFO, "Every shipped voice asset has a documented provenance."))

    manifest = release.tracked_checksums_path(root, version)
    if manifest.is_file():
        findings.append(Finding("checksums", True, INFO, f"{manifest.relative_to(root).as_posix()} is recorded."))
    else:
        findings.append(
            Finding(
                "checksums",
                False,
                BLOCKER,
                f"{manifest.relative_to(root).as_posix()} is missing for version {version}.",
                "Build the release with scripts/release.py and commit the tracked checksum manifest.",
            )
        )

    artifacts = sorted((root / "release_artifacts").glob(f"the_oracle-{version}-*"))
    if artifacts:
        findings.append(Finding("artifacts", True, INFO, f"{len(artifacts)} artifact(s) exist for {version}."))
    else:
        findings.append(
            Finding(
                "artifacts",
                True,
                WARNING,
                f"No release_artifacts/the_oracle-{version}-* files are present yet.",
                "Run scripts/release.py on a clean tree before publishing.",
            )
        )

    runbook = root / "docs" / "SOFT_LAUNCH.md"
    if not runbook.is_file():
        findings.append(
            Finding(
                "windows_scope",
                False,
                BLOCKER,
                "docs/SOFT_LAUNCH.md is missing, so the platform scope and customer path are undocumented.",
                "Write docs/SOFT_LAUNCH.md with Linux supported and Windows experimental.",
            )
        )
    elif "windows" not in runbook.read_text(encoding="utf-8").lower():
        findings.append(
            Finding(
                "windows_scope",
                True,
                WARNING,
                "docs/SOFT_LAUNCH.md does not name Windows; the alpha scope is unclear.",
                "State that Linux is supported and Windows is experimental.",
            )
        )
    else:
        findings.append(Finding("windows_scope", True, INFO, "The alpha runbook names its platform scope."))

    if _support_bundle_available():
        findings.append(Finding("support_bundle", True, INFO, "The support-bundle command is installed."))
    else:
        findings.append(
            Finding(
                "support_bundle",
                False,
                BLOCKER,
                "the_oracle.support_bundle is not importable.",
                "Install or implement the support-bundle capability.",
            )
        )

    if (root / "PRIVACY.md").is_file():
        findings.append(Finding("privacy", True, INFO, "PRIVACY.md is present."))
    else:
        findings.append(
            Finding("privacy", False, BLOCKER, "PRIVACY.md is missing.", "Restore the privacy policy before distribution.")
        )
    return findings


def report_text(findings: list[Finding]) -> str:
    blockers = [finding for finding in findings if not finding.ok and finding.severity == BLOCKER]
    warnings = [finding for finding in findings if finding.severity == WARNING]
    passed = [finding for finding in findings if finding.ok]
    lines = [
        "The Oracle — commercial alpha soft-launch gate",
        "=" * 48,
        f"STATUS: {'READY' if not blockers else 'BLOCKED'}"
        f" ({len(blockers)} blocker(s), {len(warnings)} warning(s), {len(passed)} passed)",
        "",
    ]
    if blockers:
        lines.append("Blockers:")
        for finding in blockers:
            lines.append(f"  [BLOCKER] {finding.key}: {finding.detail}")
            if finding.owner_action:
                lines.append(f"            owner action: {finding.owner_action}")
        lines.append("")
    if warnings:
        lines.append("Warnings:")
        for finding in warnings:
            lines.append(f"  [warning] {finding.key}: {finding.detail}")
            if finding.owner_action:
                lines.append(f"            action: {finding.owner_action}")
        lines.append("")
    lines.append("Passed:")
    for finding in passed:
        lines.append(f"  [ok] {finding.key}: {finding.detail}")
    return "\n".join(lines)


def _json_payload(findings: list[Finding]) -> dict[str, object]:
    blockers = [finding for finding in findings if not finding.ok and finding.severity == BLOCKER]
    return {
        "ok": not blockers,
        "status": "ready" if not blockers else "blocked",
        "blockers": len(blockers),
        "warnings": len([finding for finding in findings if finding.severity == WARNING]),
        "findings": [asdict(finding) for finding in findings],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Commercial soft-launch go/no-go gate.")
    parser.add_argument("--check", action="store_true", help="Run the gate (the default action).")
    parser.add_argument("--root", default=str(REPO_ROOT), help="Tree to audit (default: this repository).")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    args = parser.parse_args(argv)

    findings = audit(Path(args.root))
    payload = _json_payload(findings)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(report_text(findings))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
