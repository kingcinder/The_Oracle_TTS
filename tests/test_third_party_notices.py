"""Third-party notice inventory: coverage, classification, and freshness."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from the_oracle import third_party
from the_oracle.models.pins import MODEL_PINS


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_third_party_notices.py"


@pytest.fixture(scope="module")
def inventory() -> third_party.Inventory:
    return third_party.build_inventory(REPO_ROOT, version="test")


def test_classify_license_covers_the_five_buckets() -> None:
    assert third_party.classify_license("MIT") == third_party.PERMISSIVE
    assert third_party.classify_license("Apache-2.0 OR BSD-2-Clause") == third_party.PERMISSIVE
    assert third_party.classify_license("LGPL-3.0") == third_party.WEAK_COPYLEFT
    assert third_party.classify_license("MPL-2.0 AND MIT") == third_party.WEAK_COPYLEFT
    assert third_party.classify_license("GPL-3.0-or-later") == third_party.STRONG_COPYLEFT
    assert third_party.classify_license("GNU GENERAL PUBLIC LICENSE Version 3") == third_party.STRONG_COPYLEFT
    assert third_party.classify_license("NVIDIA Proprietary Software") == third_party.PROPRIETARY
    assert third_party.classify_license("CC BY 4.0") == third_party.PERMISSIVE
    assert third_party.classify_license("Owner-owned; all rights reserved") == third_party.PROPRIETARY
    assert third_party.classify_license("") == third_party.UNKNOWN
    assert third_party.classify_license("UNKNOWN") == third_party.UNKNOWN


def test_full_gpl_text_is_not_misread_as_lgpl() -> None:
    gpl3 = (REPO_ROOT / "THIRD_PARTY_LICENSES" / "GPL-3.0.txt").read_text(encoding="utf-8")
    assert third_party.classify_license(gpl3) == third_party.STRONG_COPYLEFT


def test_inventory_covers_every_declared_runtime_dependency(inventory: third_party.Inventory) -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    declared = list(project["dependencies"])
    for extra in third_party.RUNTIME_EXTRAS:
        declared.extend(project.get("optional-dependencies", {}).get(extra, []))
    names = {third_party.canonicalize_name(component.name) for component in inventory.components}
    missing = []
    for raw in declared:
        name = third_party.canonicalize_name(third_party.Requirement(raw).name)
        if name not in names:
            missing.append(name)
    assert missing == []


def test_inventory_covers_models_and_voice_assets(inventory: third_party.Inventory) -> None:
    names = {component.name for component in inventory.components}
    for repo_id in MODEL_PINS:
        assert repo_id in names
    for path in sorted((REPO_ROOT / "Seashells" / "generic").glob("*.wav")):
        assert path.name in names
    for path in sorted((REPO_ROOT / "Seashells").glob("*.wav")):
        assert path.name in names


def test_runtime_closure_excludes_development_tools(inventory: third_party.Inventory) -> None:
    names = {component.name.lower() for component in inventory.components}
    assert "pytest" not in names
    assert "pip-audit" not in names
    assert "detect-secrets" not in names


def test_committed_notices_are_current(inventory: third_party.Inventory) -> None:
    assert third_party.notice_problems(REPO_ROOT, inventory=inventory) == []


def test_render_is_deterministic(inventory: third_party.Inventory) -> None:
    assert third_party.render_notices(inventory) == third_party.render_notices(inventory)


def test_notices_do_not_cite_external_commit_hashes() -> None:
    """The record-integrity net reads hash-shaped tokens in root prose as
    citations of *this* repository. Model revisions are external identifiers,
    so the notices must point at the pin table instead of spelling them."""
    text = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert re.findall(r"\b[0-9a-f]{40}\b", text) == []


def test_committed_notices_have_no_placeholders() -> None:
    text = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8").lower()
    for forbidden in ("draft", "todo", "tbd", "undocumented"):
        assert forbidden not in text


def test_every_referenced_license_text_exists() -> None:
    text = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for filename in third_party._LICENSE_TEXT_FILES.values():
        assert f"THIRD_PARTY_LICENSES/{filename}" in text
        assert (REPO_ROOT / "THIRD_PARTY_LICENSES" / filename).is_file()


def test_packaging_carries_the_notices_and_license_texts() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "THIRD_PARTY_NOTICES.md" in manifest
    assert "recursive-include THIRD_PARTY_LICENSES" in manifest
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    license_files = data["tool"]["setuptools"]["license-files"]
    assert "THIRD_PARTY_NOTICES.md" in license_files
    assert "PRIVACY.md" in license_files
    assert "THIRD_PARTY_LICENSES/*" in license_files


def test_script_check_mode_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_script_json_mode_is_machine_readable() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["version"]
    assert any(component["kind"] == "model" for component in payload["components"])
