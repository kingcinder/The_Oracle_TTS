"""Ownership tests for the Vulkan/audio.cpp GUI extraction (gui_vulkan.py).

The per-cluster damage assessment said the Vulkan cluster's wholesale patch
surface must be preserved. The first extraction attempt moved the two policy
functions and failed exactly the tests that patch
``app_gui.find_audiocpp_binary`` / ``app_gui._vulkan_preflight_report`` by
name — a function body resolving those names from its own module globals is
invisible to app_gui-level monkeypatching. These tests pin the corrected
split: only the patch-free helpers moved; the patch-coupled policies stay on
app_gui (with their source-level patch behavior intact).
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import(name: str):
    return importlib.import_module(name)


def test_patch_free_helpers_live_in_gui_vulkan() -> None:
    owner = _import("the_oracle.gui_vulkan")
    for name in ("_device_row_text", "_parse_oracle_model_path"):
        assert callable(getattr(owner, name)), name


def test_app_gui_delegates_device_row_text_to_the_owner() -> None:
    owner = _import("the_oracle.gui_vulkan")
    app_gui = _import("the_oracle.app_gui")
    assert app_gui._device_row_text is owner._device_row_text
    assert app_gui._parse_oracle_model_path is owner._parse_oracle_model_path
    assert owner._device_row_text(2, "AMD Radeon") == "Device 2: AMD Radeon"


def test_patch_coupled_policies_stay_on_app_gui() -> None:
    """The two policy functions' bodies resolve find_audiocpp_binary from
    app_gui's own globals — the property the patch surface depends on."""
    app_gui = _import("the_oracle.app_gui")
    assert callable(app_gui._vulkan_prerequisite_missing)
    assert callable(app_gui._vulkan_preflight_report)
    app_gui_source = Path(app_gui.__file__).read_text(encoding="utf-8")
    assert "def _vulkan_prerequisite_missing" in app_gui_source
    assert "def _vulkan_preflight_report" in app_gui_source


def test_prerequisite_missing_sees_app_gui_level_binary_patch(monkeypatch) -> None:
    """The load-bearing patch property, exercised directly: patching
    app_gui.find_audiocpp_binary must change what the policy function sees."""
    app_gui = _import("the_oracle.app_gui")
    from the_oracle.tts_engines import vulkan_backend

    monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: None)
    monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
    # find_audiocpp_model() is patched too so the check isolates the binary probe.
    monkeypatch.setattr(vulkan_backend, "find_audiocpp_model", lambda: None)
    missing = app_gui._vulkan_prerequisite_missing()
    assert missing and "audiocpp_cli is not built" in missing[0]


def test_gui_vulkan_has_no_wholesale_patch_targets() -> None:
    """Nothing in gui_vulkan is monkeypatched by name anywhere in the suite."""
    owner = _import("the_oracle.gui_vulkan")
    owner_names = {name for name in vars(owner) if not name.startswith("__")}
    tests_dir = REPO_ROOT / "tests"
    offenders: list[str] = []
    for test_file in tests_dir.glob("test_*.py"):
        text = test_file.read_text(encoding="utf-8")
        for name in owner_names:
            if f'"gui_vulkan", "{name}"' in text or f"'gui_vulkan', '{name}'" in text:
                offenders.append(f"{test_file.name}:{name}")
    assert offenders == []


def test_gui_vulkan_imports_nothing_from_app_gui() -> None:
    """No import cycle: the support module must not reach back into app_gui."""
    owner = _import("the_oracle.gui_vulkan")
    source = Path(owner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name == "app_gui" or alias.name.startswith("app_gui.") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "app_gui" and not (node.module or "").startswith("app_gui.")
