"""Single-engine Chatterbox adapter supporting standard, multilingual, and turbo variants."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from dualvoice_studio.models.cache import CachedReference, ProjectCache
from dualvoice_studio.models.project import VoiceSettings
from dualvoice_studio.utils.hashing import hash_payload


SUPPORTED_VARIANTS = ("standard", "multilingual", "turbo")


@dataclass(slots=True)
class ChatterboxConditioning:
    cache_id: str
    path: Path
    reference_hash: str
    speaker: str
    variant: str


class ChatterboxEngine:
    engine_id = "chatterbox"

    def __init__(self, variant: str = "standard", device: str | None = None) -> None:
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported Chatterbox variant: {variant}")
        self.variant = variant
        self.device = device or self._detect_device()
        self._model = None
        self._condition_cls = None
        self._languages = {"en": "English"}
        self._loaded_conditioning: dict[str, Any] = {}

    @property
    def engine_version(self) -> str:
        try:
            return version("chatterbox-tts")
        except PackageNotFoundError:
            return "not-installed"

    @property
    def sample_rate(self) -> int:
        return int(self.model.sr)

    @property
    def model(self):
        if self._model is None:
            self._model, self._condition_cls, self._languages = self._load_variant()
        return self._model

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def supported_languages(self) -> dict[str, str]:
        _ = self.model
        return dict(self._languages)

    def _load_variant(self):
        if self.variant == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals, SUPPORTED_LANGUAGES

            return ChatterboxMultilingualTTS.from_pretrained(device=self.device), Conditionals, SUPPORTED_LANGUAGES
        if self.variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals

            return ChatterboxTurboTTS.from_pretrained(device=self.device), Conditionals, {"en": "English"}
        from chatterbox.tts import ChatterboxTTS, Conditionals

        return ChatterboxTTS.from_pretrained(device=self.device), Conditionals, {"en": "English"}

    def prepare_reference(self, project_cache: ProjectCache, speaker: str, reference_path: str) -> CachedReference:
        return project_cache.cache_reference_audio(reference_path, speaker, self.sample_rate)

    def prepare_conditioning(
        self,
        project_cache: ProjectCache,
        speaker: str,
        cached_reference: CachedReference,
        settings: VoiceSettings,
    ) -> ChatterboxConditioning:
        cache_id = hash_payload(
            {
                "speaker": speaker,
                "reference_hash": cached_reference.original_hash,
                "variant": self.variant,
                "conditioning_exaggeration": settings.exaggeration,
            }
        )
        condition_path = project_cache.conditioning_path(cache_id)
        if not condition_path.exists():
            if self.variant == "turbo":
                self.model.prepare_conditionals(
                    cached_reference.normalized_path,
                    exaggeration=settings.exaggeration,
                    norm_loudness=settings.norm_loudness,
                )
            else:
                self.model.prepare_conditionals(
                    cached_reference.normalized_path,
                    exaggeration=settings.exaggeration,
                )
            self.model.conds.save(condition_path)
        return ChatterboxConditioning(
            cache_id=cache_id,
            path=condition_path,
            reference_hash=cached_reference.original_hash,
            speaker=speaker,
            variant=self.variant,
        )

    def synthesize(self, text: str, conditioning: ChatterboxConditioning, settings: VoiceSettings) -> np.ndarray:
        cache_key = str(conditioning.path)
        conds = self._loaded_conditioning.get(cache_key)
        if conds is None:
            try:
                conds = self._condition_cls.load(conditioning.path, map_location=self.device)
            except TypeError:
                conds = self._condition_cls.load(conditioning.path)
            conds = conds.to(self.device)
            self._loaded_conditioning[cache_key] = conds
        self.model.conds = conds

        kwargs: dict[str, Any] = {
            "text": text,
            "audio_prompt_path": None,
            "cfg_weight": settings.cfg_weight,
            "exaggeration": settings.exaggeration,
            "temperature": settings.temperature,
            "repetition_penalty": settings.repetition_penalty,
            "min_p": settings.min_p,
            "top_p": settings.top_p,
        }
        if self.variant == "multilingual":
            kwargs["language_id"] = settings.language
        if self.variant == "turbo":
            kwargs["top_k"] = settings.top_k
            kwargs["norm_loudness"] = settings.norm_loudness
        audio = self.model.generate(**kwargs)
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio_array = np.asarray(audio, dtype=np.float32).squeeze()
        return audio_array
