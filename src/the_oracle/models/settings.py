"""Render/speaker settings dataclasses — the single owner of the settings schema.

Extracted verbatim from pipeline.py so settings-adjacent modules
(gui_settings, project_manifest) can import the schema without pulling in the
engine (pipeline.py imports the Chatterbox engine surface, which is exactly
the wrong dependency direction for a settings module — see the
WidgetSnapshot/PayloadDefaults design note in gui_settings.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from the_oracle.correction_modes import normalize_correction_mode
from the_oracle.models.project import VoiceSettings
from the_oracle.speaker_attribution.heuristics import AnchorAssignments
from the_oracle.tts_engines.vulkan_backend import SUPPORTED_BACKENDS


@dataclass(slots=True)
class SpeakerSettings:
    reference_path: str
    voice_settings: VoiceSettings | dict[str, Any] = field(default_factory=VoiceSettings)
    emotion_reference_paths: dict[str, str] = field(default_factory=dict)
    # Voice blending: when two reference paths are given, the speaker's
    # conditioning reference is a deterministic blend of them (see
    # audio/blend.py) instead of ``reference_path`` alone. ``blend_weight``
    # is the proportion (0..1) of the first clip; ``blend_mode`` is one of
    # "mix", "alternate", "layer".
    blend_references: list[str] = field(default_factory=list)
    blend_weight: float = 0.5
    blend_mode: str = "mix"


@dataclass(slots=True)
class RenderSettings:
    correction_mode: str = normalize_correction_mode("moderate")
    model_variant: str = "standard"
    language: str = "en"
    export_stems: bool = True
    loudness_preset: str = "light"
    pause_between_turns_ms: int = 180
    crossfade_ms: int = 20
    device_mode: str = "cpu"
    cuda_device: int | None = None
    inference_backend: str = "pytorch"
    audio_cpp_device: int | None = None
    audio_cpp_threads: int | None = None
    audio_cpp_timeout: int | None = None
    audio_cpp_max_batch: int | None = None
    # Deterministic sampling seed. When set, every engine call seeds its RNG
    # first so a render with identical inputs produces byte-identical audio
    # across runs (PyTorch: torch.manual_seed; Vulkan: audio.cpp --seed).
    seed: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    anchors: AnchorAssignments | None = None
    target_wpm: float | None = None
    # Single-narrator mode: every line is rendered in Speaker A's voice,
    # ignoring per-line attribution (useful for reading a book aloud as one
    # narrator instead of a cast).
    monologue: bool = False

    def __post_init__(self) -> None:
        self.correction_mode = normalize_correction_mode(self.correction_mode)
        if self.inference_backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported inference backend: {self.inference_backend!r}. "
                f"Choose from {SUPPORTED_BACKENDS}."
            )
        # Vulkan device/threads knobs are only meaningful for the audio.cpp
        # backend; reject negative values early so a bad CLI flag or manifest
        # cannot turn into a confusing audio.cpp error mid-render.
        if self.audio_cpp_device is not None and self.audio_cpp_device < 0:
            raise ValueError(
                f"audio_cpp_device must be a non-negative Vulkan device index, got {self.audio_cpp_device!r}."
            )
        if self.audio_cpp_threads is not None and self.audio_cpp_threads < 1:
            raise ValueError(
                f"audio_cpp_threads must be a positive thread count, got {self.audio_cpp_threads!r}."
            )
        if self.audio_cpp_timeout is not None and self.audio_cpp_timeout < 1:
            raise ValueError(
                f"audio_cpp_timeout must be a positive timeout in seconds, got {self.audio_cpp_timeout!r}."
            )
        if self.audio_cpp_max_batch is not None and self.audio_cpp_max_batch < 1:
            raise ValueError(
                f"audio_cpp_max_batch must be a positive request count, got {self.audio_cpp_max_batch!r}."
            )
        # CUDA device selection is only meaningful for the PyTorch path, but
        # validating it here keeps malformed manifests from failing deep inside
        # a worker process.
        if self.cuda_device is not None and self.cuda_device < 0:
            raise ValueError(
                f"cuda_device must be a non-negative CUDA device index, got {self.cuda_device!r}."
            )
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed!r}.")
