"""Distribution hygiene checks for bundled voice assets."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY = REPO_ROOT / "Seashells" / "ASSETS.md"


def _tracked_voice_assets() -> set[str]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "Seashells"], cwd=REPO_ROOT, text=True
    ).splitlines()
    return {path for path in tracked if path.lower().endswith(".wav")}


def _inventory_paths() -> set[str]:
    text = INVENTORY.read_text(encoding="utf-8")
    return set(re.findall(r"`(Seashells/[^`]+\.wav)`", text))


def test_stock_voice_references_are_tracked() -> None:
    tracked = _tracked_voice_assets()
    expected = {
        "Seashells/Cody's Seashell.wav",
        "Seashells/Cody's Seashell1.wav",
        "Seashells/Fransisco_MONSTER.wav",
        "Seashells/Seashell_ fransisco.wav",
        "Seashells/Seashell_No_1.wav",
        "Seashells/Seashell_No_2.wav",
        "Seashells/seashell fransisco.wav",
        "Seashells/seashell fransisco_TTS_clean_v5_aggressive_targeted_cleanup.wav",
    }
    assert expected.issubset(tracked)


def test_every_tracked_voice_asset_has_an_inventory_entry() -> None:
    missing = sorted(_tracked_voice_assets() - _inventory_paths())
    assert missing == []


def test_every_inventory_entry_is_a_tracked_voice_asset() -> None:
    stale = sorted(_inventory_paths() - _tracked_voice_assets())
    assert stale == []


def test_voice_inventory_has_no_undocumented_entries() -> None:
    text = INVENTORY.read_text(encoding="utf-8")
    lowered = text.lower()
    for forbidden in ("undocumented", "unknown", "tbd", "todo"):
        assert forbidden not in lowered, (
            f"Seashells/ASSETS.md still contains {forbidden!r}; every shipped clip "
            "must have a recorded source and redistribution term"
        )
    assert "owner-owned" in lowered


def test_root_level_recordings_are_documented_as_owner_owned() -> None:
    text = INVENTORY.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| `Seashells/")]
    root_rows = [line for line in rows if "`Seashells/generic/" not in line]
    assert root_rows, "expected the root-level Seashells recordings in the inventory"
    for row in root_rows:
        assert "Owner-owned" in row
        assert "all rights reserved" in row.lower()
