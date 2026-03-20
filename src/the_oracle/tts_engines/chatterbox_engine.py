"""Single-engine Chatterbox adapter supporting standard, multilingual, and turbo variants."""

from __future__ import annotations

import os
from time import perf_counter, time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download

try:
    from huggingface_hub.errors import LocalEntryNotFoundError
except Exception:  # pragma: no cover - fallback for older huggingface_hub versions
    LocalEntryNotFoundError = tuple()  # type: ignore[assignment]

try:
    from huggingface_hub.utils import LocalTokenNotFoundError
except Exception:  # pragma: no cover - fallback for older huggingface_hub versions
    LocalTokenNotFoundError = tuple()  # type: ignore[assignment]

from the_oracle.models.cache import CachedReference, ProjectCache
from the_oracle.models.project import VoiceSettings
from the_oracle.utils.hashing import hash_payload


SUPPORTED_VARIANTS = ("standard", "multilingual", "turbo")
TURBO_REPO_ID = "ResembleAI/chatterbox-turbo"
TURBO_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]


class TurboModelError(RuntimeError):
    """Raised when the turbo checkpoint cannot be prepared or initialized."""


@dataclass(slots=True)
class ChatterboxConditioning:
    cache_id: str
    path: Path
    reference_hash: str
    speaker: str
    variant: str


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _format_turbo_error(exc: Exception, *, cached_only: bool = False) -> str:
    prefix = "Turbo checkpoint is not cached locally." if cached_only else "Turbo model initialization failed."
    if isinstance(exc, LocalTokenNotFoundError) or "Token is required (`token=True`)" in str(exc):
        return (
            f"{prefix} chatterbox-tts 0.1.6 forces HF auth for the public turbo checkpoint. "
            "The Oracle bypasses that broken loader, but this environment still does not have a usable turbo checkpoint. "
            "Connect to the internet and run ./.venv/bin/python scripts/download_models.py --variant turbo --device cpu "
            "or set HF_TOKEN if your HF environment requires auth. "
            f"Original error: {type(exc).__name__}: {exc}"
        )
    if isinstance(exc, LocalEntryNotFoundError):
        return (
            f"{prefix} Connect to the internet and run ./.venv/bin/python scripts/download_models.py --variant turbo --device cpu "
            "to prefetch the public turbo checkpoint."
        )
    return (
        f"{prefix} Run ./.venv/bin/python scripts/download_models.py --variant turbo --device cpu while online. "
        f"Original error: {type(exc).__name__}: {exc}"
    )


def download_turbo_checkpoint(*, local_files_only: bool = False) -> Path:
    return Path(
        snapshot_download(
            repo_id=TURBO_REPO_ID,
            token=_hf_token(),
            local_files_only=local_files_only,
            allow_patterns=TURBO_ALLOW_PATTERNS,
        )
    )


def turbo_readiness_report(device: str = "cpu") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "cached": False,
        "checkpoint_dir": "",
        "device": device,
        "error": "",
    }
    try:
        checkpoint_dir = download_turbo_checkpoint(local_files_only=True)
    except Exception as exc:
        payload["error"] = _format_turbo_error(exc, cached_only=True)
        return payload

    payload["cached"] = True
    payload["checkpoint_dir"] = str(checkpoint_dir)
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS.from_local(checkpoint_dir, device)
    except Exception as exc:
        payload["error"] = _format_turbo_error(exc)
        return payload

    payload["ok"] = True
    payload["sample_rate"] = int(getattr(model, "sr", 0) or 0)
    return payload


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
        self._load_seconds: float | None = None
        self._load_wall: float | None = None

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
            load_start = perf_counter()
            load_wall = time()
            self._model, self._condition_cls, self._languages = self._load_variant()
            self._load_seconds = round(perf_counter() - load_start, 6)
            self._load_wall = load_wall
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

    def ensure_model_ready(self) -> None:
        _ = self.model

    def _load_variant(self):
        if self.variant == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals, SUPPORTED_LANGUAGES

            return ChatterboxMultilingualTTS.from_pretrained(device=self.device), Conditionals, SUPPORTED_LANGUAGES
        if self.variant == "turbo":
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals

                checkpoint_dir = download_turbo_checkpoint()
                return ChatterboxTurboTTS.from_local(checkpoint_dir, self.device), Conditionals, {"en": "English"}
            except Exception as exc:
                raise TurboModelError(_format_turbo_error(exc)) from exc
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
