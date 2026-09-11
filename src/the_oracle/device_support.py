"""Runtime hardware capability helpers for The Oracle inference choices.

CUDA is intentionally modeled as a PyTorch *device mode*, not as a separate
inference backend. This keeps model loading, conditioning, caching, previews,
and renders on the existing ChatterboxEngine path while allowing the GUI and
CLI to explain why a GPU is or is not usable.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass


# Chatterbox is a roughly half-billion-parameter model and needs room for
# weights, activations, and audio buffers. Cards below this floor are not
# presented as viable CUDA choices; in particular, 1 GiB legacy cards are
# correctly identified as unsuitable instead of producing an opaque OOM.
CUDA_MIN_VRAM_BYTES = 4 * 1024**3


@dataclass(frozen=True, slots=True)
class DeviceModeOption:
    key: str
    label: str
    available: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CUDADeviceInfo:
    index: int
    name: str
    vram_bytes: int | None
    driver_version: str = ""
    torch_available: bool = False
    suitable: bool = False
    reason: str = ""

    @property
    def vram_gib(self) -> float | None:
        if self.vram_bytes is None:
            return None
        return self.vram_bytes / (1024**3)

    @property
    def label(self) -> str:
        vram = "unknown VRAM" if self.vram_gib is None else f"{self.vram_gib:.1f} GiB VRAM"
        return f"CUDA {self.index}: {self.name} ({vram})"


def _torch_cuda_devices() -> list[CUDADeviceInfo]:
    try:
        import torch
    except Exception:
        return []
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
    except Exception:
        return []
    devices: list[CUDADeviceInfo] = []
    for index in range(count):
        try:
            properties = torch.cuda.get_device_properties(index)
            name = str(getattr(properties, "name", f"NVIDIA CUDA device {index}"))
            total_memory = int(getattr(properties, "total_memory", 0) or 0) or None
        except Exception as exc:
            devices.append(
                CUDADeviceInfo(
                    index=index,
                    name=f"NVIDIA CUDA device {index}",
                    vram_bytes=None,
                    torch_available=available,
                    suitable=False,
                    reason=f"PyTorch could not inspect this device: {exc}",
                )
            )
            continue
        if total_memory is not None and total_memory < CUDA_MIN_VRAM_BYTES:
            reason = (
                f"Not suitable for Chatterbox: {total_memory / (1024**3):.1f} GiB VRAM "
                f"is below the {CUDA_MIN_VRAM_BYTES / (1024**3):.0f} GiB minimum."
            )
            suitable = False
        else:
            reason = "CUDA is visible to PyTorch and this device meets the VRAM floor."
            suitable = True
        devices.append(
            CUDADeviceInfo(
                index=index,
                name=name,
                vram_bytes=total_memory,
                torch_available=available,
                suitable=suitable,
                reason=reason,
            )
        )
    return devices


_NVIDIA_SMI_LINE = re.compile(r"^\s*(\d+)\s*,\s*(.*?)\s*,\s*([^,]+?)\s*,\s*(.*?)\s*$")


def _parse_vram_bytes(raw: str) -> int | None:
    match = re.search(r"([\d.]+)", raw.replace("MiB", "").replace("MiB", ""))
    if not match:
        return None
    try:
        # nvidia-smi reports memory in MiB for the query below.
        return int(float(match.group(1)) * 1024**2)
    except ValueError:
        return None


def _nvidia_smi_devices() -> list[CUDADeviceInfo]:
    """Inspect installed NVIDIA hardware even when the active torch wheel is CPU-only."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    devices: list[CUDADeviceInfo] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        vram_bytes = _parse_vram_bytes(parts[2] + " MiB")
        if vram_bytes is None:
            reason = "NVIDIA hardware found, but VRAM could not be read."
            suitable = False
        elif vram_bytes < CUDA_MIN_VRAM_BYTES:
            reason = (
                f"Not suitable for Chatterbox: {vram_bytes / (1024**3):.1f} GiB VRAM "
                f"is below the {CUDA_MIN_VRAM_BYTES / (1024**3):.0f} GiB minimum."
            )
            suitable = False
        else:
            reason = "NVIDIA hardware meets the VRAM floor; CUDA still needs a compatible PyTorch runtime."
            suitable = True
        devices.append(
            CUDADeviceInfo(
                index=index,
                name=parts[1],
                vram_bytes=vram_bytes,
                driver_version=parts[3],
                torch_available=False,
                suitable=suitable,
                reason=reason,
            )
        )
    return devices


def cuda_devices() -> list[CUDADeviceInfo]:
    """Return CUDA devices with PyTorch facts preferred over nvidia-smi facts."""
    torch_devices = _torch_cuda_devices()
    if torch_devices:
        return torch_devices
    return _nvidia_smi_devices()


def cuda_runtime_available() -> bool:
    """True only when PyTorch can actually execute CUDA kernels."""
    return any(device.torch_available and device.suitable for device in cuda_devices())


def cuda_reason() -> str:
    devices = cuda_devices()
    if not devices:
        return "No NVIDIA CUDA device was detected (or nvidia-smi is unavailable)."
    suitable = [device for device in devices if device.suitable]
    if not suitable:
        return "Detected NVIDIA GPU(s), but none meet the 4 GiB Chatterbox VRAM minimum."
    if not any(device.torch_available for device in suitable):
        return "A suitable NVIDIA GPU is present, but this installation has no CUDA-enabled PyTorch runtime."
    return "CUDA-enabled PyTorch can use at least one suitable NVIDIA GPU."


def available_device_modes() -> list[DeviceModeOption]:
    return [
        DeviceModeOption(key="cpu", label="CPU / system DRAM", available=True, reason="Always available."),
        DeviceModeOption(
            key="cuda",
            label="CUDA / NVIDIA GPU",
            available=cuda_runtime_available(),
            reason=cuda_reason(),
        ),
        DeviceModeOption(
            key="vulkan",
            label="Vulkan GPU",
            available=_vulkan_runtime_available(),
            reason=_vulkan_reason(),
        ),
    ]


def resolve_chatterbox_device(device_mode: str, cuda_device: int | None = None) -> str:
    """Resolve a persisted/CLI device mode to a PyTorch device string.

    Vulkan remains an audio.cpp backend and is rejected here as before. CUDA
    requires both a suitable card and a CUDA-capable PyTorch runtime; callers
    get a readable recovery message instead of a late tensor/device failure.
    """
    if device_mode == "cpu":
        return "cpu"
    if device_mode == "cuda":
        devices = cuda_devices()
        suitable = [device for device in devices if device.suitable and device.torch_available]
        if not suitable:
            raise RuntimeError(
                "CUDA inference is unavailable: " + cuda_reason() + " Install the CUDA PyTorch bundle "
                "or choose device_mode=cpu."
            )
        if cuda_device is None:
            return "cuda"
        selected = next((device for device in suitable if device.index == cuda_device), None)
        if selected is None:
            available = ", ".join(str(device.index) for device in suitable)
            raise RuntimeError(
                f"CUDA device {cuda_device} is unavailable or unsuitable. Suitable CUDA devices: "
                f"{available or 'none'}. Choose another device or use CPU."
            )
        return f"cuda:{cuda_device}"
    if device_mode == "vulkan":
        raise RuntimeError(
            "Vulkan GPU mode is not verified by PyTorch device resolution; "
            "use the audio.cpp Vulkan backend instead."
        )
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
