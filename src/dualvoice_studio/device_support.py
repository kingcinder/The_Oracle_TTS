"""Runtime device capability helpers for Chatterbox execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeviceModeOption:
    key: str
    label: str
    available: bool
    reason: str = ""


def available_device_modes() -> list[DeviceModeOption]:
    return [
        DeviceModeOption(key="cpu", label="CPU", available=True, reason="Always available."),
        DeviceModeOption(
            key="vulkan",
            label="Vulkan GPU",
            available=_vulkan_runtime_available(),
            reason=_vulkan_reason(),
        ),
    ]


def resolve_chatterbox_device(device_mode: str) -> str:
    if device_mode == "cpu":
        return "cpu"
    if device_mode == "vulkan":
        raise RuntimeError("Vulkan GPU mode is not verified for the installed Chatterbox runtime in this environment.")
    raise ValueError(f"Unsupported device mode: {device_mode}")


def _vulkan_runtime_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    if not hasattr(torch.backends, "vulkan"):
        return False
    try:
        return bool(torch.backends.vulkan.is_available())
    except Exception:
        return False


def _vulkan_reason() -> str:
    if _vulkan_runtime_available():
        return "Torch reports Vulkan support, but Chatterbox execution on Vulkan remains unverified here."
    return "The installed torch/Chatterbox runtime does not expose a verified Vulkan execution path."
