"""Engine adapters for Chatterbox inference across PyTorch and Vulkan backends."""

from the_oracle.tts_engines.chatterbox_engine import ChatterboxConditioning, ChatterboxEngine
from the_oracle.tts_engines.vulkan_backend import (
    AudioCppUnavailableError,
    AudioCppVulkanEngine,
    RDNA1VulkanError,
    SUPPORTED_BACKENDS,
    VulkanConditioning,
    vulkan_device_available,
)

__all__ = [
    "AudioCppUnavailableError",
    "AudioCppVulkanEngine",
    "ChatterboxConditioning",
    "ChatterboxEngine",
    "RDNA1VulkanError",
    "SUPPORTED_BACKENDS",
    "VulkanConditioning",
    "vulkan_device_available",
]
