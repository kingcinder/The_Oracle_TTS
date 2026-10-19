"""Commercial soft-launch gate: EULA, assets, license policy, and report shape."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from the_oracle import third_party


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "soft_launch.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("soft_launch_gate_module", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def test_draft_eula_is_a_blocker(tmp_path: Path) -> None:
    (tmp_path / "EULA-DRAFT.md").write_text(
        "# End User License Agreement (DRAFT)\n\nGoverning law: [insert jurisdiction]\n",
        encoding="utf-8",
    )
    problems = gate._eula_problems(tmp_path)
    assert any("draft" in problem for problem in problems)
    assert any("insert" in problem for problem in problems)


def test_final_eula_passes(tmp_path: Path) -> None:
    (tmp_path / "EULA.md").write_text(
        "# End User License Agreement\n\n## Governing law and venue\n\nThis agreement is governed by the laws of Testland.\n",
        encoding="utf-8",
    )
    assert gate._eula_problems(tmp_path) == []


def test_missing_governing_law_is_a_blocker(tmp_path: Path) -> None:
    (tmp_path / "EULA.md").write_text("# End User License Agreement\n\nNo choice of law.\n", encoding="utf-8")
    assert any("governing-law" in problem for problem in gate._eula_problems(tmp_path))


def test_voice_asset_problems_detect_missing_and_undocumented(tmp_path: Path) -> None:
    seashells = tmp_path / "Seashells"
    seashells.mkdir()
    (seashells / "one.wav").write_bytes(b"RIFF")
    (seashells / "two.wav").write_bytes(b"RIFF")
    (seashells / "ASSETS.md").write_text(
        "# Assets\n\n| File | Source | License | Note |\n|---|---|---|---|\n"
        "| `Seashells/one.wav` | owner | Undocumented | TODO |\n",
        encoding="utf-8",
    )
    problems = gate._voice_asset_problems(tmp_path)
    assert any("two.wav" in problem for problem in problems)
    assert any("undocumented" in problem.lower() for problem in problems)


def test_voice_asset_problems_accepts_owner_owned_inventory(tmp_path: Path) -> None:
    seashells = tmp_path / "Seashells"
    seashells.mkdir()
    (seashells / "one.wav").write_bytes(b"RIFF")
    (seashells / "ASSETS.md").write_text(
        "# Assets\n\n| File | Source | License | Note |\n|---|---|---|---|\n"
        "| `Seashells/one.wav` | Owner recording | Owner-owned; all rights reserved | Original recording. |\n",
        encoding="utf-8",
    )
    assert gate._voice_asset_problems(tmp_path) == []


def _fake_inventory(*components: third_party.Component) -> third_party.Inventory:
    return third_party.Inventory(components=components, version="test")


def test_strong_copyleft_with_a_recorded_reason_is_a_blocker(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "license_review.json").write_text(
        json.dumps({"allow": [], "review": {"mutagen": {"reason": "replace it"}}}),
        encoding="utf-8",
    )
    inventory = _fake_inventory(third_party.Component("mutagen", "1.0", "GPL-2.0-or-later", "python"))
    findings = gate._license_policy_findings(tmp_path, inventory)
    blocker = next(finding for finding in findings if not finding.ok)
    assert "replace it" in blocker.detail
    assert blocker.owner_action


def test_strong_copyleft_can_be_allowlisted(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "license_review.json").write_text(
        json.dumps({"allow": ["mutagen"], "review": {}}),
        encoding="utf-8",
    )
    inventory = _fake_inventory(third_party.Component("mutagen", "1.0", "GPL-2.0-or-later", "python"))
    findings = gate._license_policy_findings(tmp_path, inventory)
    assert all(finding.ok for finding in findings)


def test_weak_copyleft_is_a_warning_not_a_blocker(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "license_review.json").write_text(json.dumps({"allow": [], "review": {}}), encoding="utf-8")
    inventory = _fake_inventory(third_party.Component("PySide6", "6.8.2.1", "LGPL-3.0", "python"))
    findings = gate._license_policy_findings(tmp_path, inventory)
    assert findings
    assert all(finding.ok for finding in findings)
    assert any(finding.severity == gate.WARNING for finding in findings)


def test_artifact_check_requires_both_the_sdist_and_the_wheel(tmp_path: Path) -> None:
    outdir = tmp_path / "release_artifacts"
    outdir.mkdir()
    wheel = outdir / "the_oracle-9.9.9-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    finding = gate._artifact_finding(tmp_path, "9.9.9")
    assert finding.ok
    assert finding.severity == gate.WARNING
    assert "the_oracle-9.9.9.tar.gz" in finding.detail

    (outdir / "the_oracle-9.9.9.tar.gz").write_bytes(b"sdist")
    finding = gate._artifact_finding(tmp_path, "9.9.9")
    assert finding.ok
    assert finding.severity == gate.INFO
    assert "sdist and wheel" in finding.detail


def test_report_names_the_owner_action() -> None:
    finding = gate.Finding("eula", False, gate.BLOCKER, "EULA is a draft", "Finish the EULA.")
    text = gate.report_text([finding])
    assert "BLOCKED" in text
    assert "owner action: Finish the EULA." in text


def test_gate_json_is_machine_readable_on_the_current_tree() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1)
    payload = json.loads(result.stdout)
    assert payload["status"] in {"ready", "blocked"}
    assert isinstance(payload["findings"], list)
    assert all({"key", "ok", "severity", "detail", "owner_action"} <= set(item) for item in payload["findings"])
