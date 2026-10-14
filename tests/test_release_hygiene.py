"""Distribution hygiene checks for private voice assets."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_private_voice_recordings_are_not_tracked() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "Seashells"], cwd=REPO_ROOT, text=True
    ).splitlines()
    forbidden = {
        "Seashells/Cody's Seashell.wav",
        "Seashells/Cody's Seashell1.wav",
        "Seashells/Seashell_ fransisco.wav",
        "Seashells/seashell fransisco.wav",
        "Seashells/seashell fransisco_TTS_clean_v5_aggressive_targeted_cleanup.wav",
        "Input/fransisco help.txt",
        "Seashells/Seashell_No_1.wav",
        "Seashells/Seashell_No_2.wav",
    }
    assert not forbidden.intersection(tracked)


def test_every_tracked_voice_asset_has_an_inventory_entry() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "Seashells/generic/*.wav"], cwd=REPO_ROOT, text=True
    ).splitlines()
    inventory = (REPO_ROOT / "Seashells" / "ASSETS.md").read_text(encoding="utf-8")
    missing = [path for path in tracked if f"`{path}`" not in inventory]
    assert missing == []
