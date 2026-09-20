"""Vulkan/audio.cpp GUI support surface: device labels and model-path parsing.

Extracted from app_gui.py: only the module-level helpers with NO wholesale
patch surface in the suite. The per-cluster damage assessment (and its
execution attempt) confirmed the two policy functions
(``_vulkan_prerequisite_missing``, ``_vulkan_preflight_report``) must STAY in
app_gui: tests patch ``app_gui.find_audiocpp_binary`` and
``app_gui._vulkan_preflight_report`` by name, and a function body that resolves
those names from its own module globals breaks under that patching — measured,
not assumed (the first extraction attempt failed exactly those tests).
"""

from __future__ import annotations

from the_oracle.vulkan_setup import parse_model_export


def _parse_oracle_model_path(output: str) -> str:
    """Extract the installed model path from the download script's output.

    Delegates to :func:`the_oracle.vulkan_setup.parse_model_export` (the
    single source of truth shared with the auto-setup orchestrator).
    """
    return parse_model_export(output)


def _device_row_text(index: int, name: str) -> str:
    """Human label for one Vulkan device, shared by the dropdown items and
    the picker's summary label so the two can never drift apart."""
    return f"Device {index}: {name}"
