"""Deterministic smoke-render harness for repository verification."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from the_oracle.audio.assemble import save_wav
from the_oracle.models.cache import CachedReference, ProjectCache
from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import ChatterboxConditioning, OraclePipeline, RenderSettings, SpeakerSettings
from the_oracle.utils.hashing import hash_payload


SMOKE_DIALOGUE = """Speaker A: The Oracle is online.
Speaker B: Confirm the signal path.
Speaker A: Chatterbox is the only backend now.
Speaker B: Render complete.
"""

SMOKE_DIALOGUE_MARKDOWN = """# Smoke Dialogue

Speaker A: The Oracle is online.
Speaker B: Confirm the signal path.
Speaker A: Chatterbox is the only backend now.
Speaker B: Render complete.
"""


@dataclass(slots=True)
class SmokeRenderResult:
    source_format: str
    output_path: Path
    project_dir: Path
    cache_reused_on_second_pass: bool
    stem_count: int
    render_plan_path: Path
    dialogue_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "source_format": self.source_format,
            "output_path": str(self.output_path),
            "project_dir": str(self.project_dir),
            "cache_reused_on_second_pass": self.cache_reused_on_second_pass,
            "stem_count": self.stem_count,
            "render_plan_path": str(self.render_plan_path),
            "dialogue_path": str(self.dialogue_path),
        }


class _SmokeEmotionClassifier:
    def classify(self, text: str):
        return type("EmotionResult", (), {"label": "neutral", "confidence": 1.0})()

    def controls_for_emotion(self, label: str) -> dict[str, float | int]:
        return {"cfg_weight": 0.5, "exaggeration": 0.5, "temperature": 0.8, "pause_ms": 120}


class _DeterministicChatterboxEngine:
    engine_id = "chatterbox"
    sample_rate = 24000
    engine_version = "deterministic-smoke-v1"

    def __init__(self, variant: str = "standard", device: str | None = None) -> None:
        self.variant = variant
        self.device = device or "cpu"

    def supported_languages(self) -> dict[str, str]:
        return {"en": "English"}

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
                "cfg_weight": settings.cfg_weight,
                "exaggeration": settings.exaggeration,
                "language": settings.language,
            }
        )
        return ChatterboxConditioning(
            cache_id=cache_id,
            path=Path(project_cache.conditioning_path(cache_id)),
            reference_hash=cached_reference.original_hash,
            speaker=speaker,
            variant=self.variant,
        )

    def synthesize(self, text: str, conditioning: ChatterboxConditioning, settings: VoiceSettings) -> np.ndarray:
        seed = hash_payload(
            {
                "text": text,
                "speaker": conditioning.speaker,
                "reference_hash": conditioning.reference_hash,
                "variant": conditioning.variant,
                "settings": settings.to_dict(),
            }
        )
        base = int(seed[:8], 16)
        duration_seconds = 0.22 + 0.02 * max(1, len(text.split()))
        samples = max(1, int(round(self.sample_rate * duration_seconds)))
        time_axis = np.arange(samples, dtype=np.float32) / np.float32(self.sample_rate)
        frequency = 180.0 + float(base % 240)
        phase = np.float32((base % 360) * np.pi / 180.0)
        amplitude = np.float32(0.12 + ((base >> 8) % 25) / 500.0)
        envelope = np.linspace(1.0, 0.7, num=samples, dtype=np.float32)
        audio = amplitude * np.sin((2.0 * np.pi * frequency * time_axis) + phase) * envelope
        return np.asarray(audio, dtype=np.float32)


def _write_reference(path: Path, frequency: float) -> Path:
    sample_rate = 24000
    seconds = 0.6
    time_axis = np.arange(int(sample_rate * seconds), dtype=np.float32) / np.float32(sample_rate)
    audio = 0.2 * np.sin(2.0 * np.pi * frequency * time_axis)
    save_wav(path, np.asarray(audio, dtype=np.float32), sample_rate)
    return path


def run_deterministic_smoke_render(output_root: str | Path, source_format: str = "txt") -> SmokeRenderResult:
    if source_format not in {"txt", "md"}:
        raise ValueError(f"Unsupported smoke source format: {source_format}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dialogue_name = "smoke_dialogue.md" if source_format == "md" else "smoke_dialogue.txt"
    dialogue_text = SMOKE_DIALOGUE_MARKDOWN if source_format == "md" else SMOKE_DIALOGUE
    dialogue_path = output_root / dialogue_name
    dialogue_path.write_text(dialogue_text, encoding="utf-8")
    speaker_a = _write_reference(output_root / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(output_root / "speaker_b_ref.wav", 330.0)
    project_dir = output_root / f"render_project_{source_format}"
    if project_dir.exists():
        shutil.rmtree(project_dir)

    with (
        patch("the_oracle.pipeline.ChatterboxEngine", _DeterministicChatterboxEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
    ):
        pipeline = OraclePipeline()
        shared_voice = VoiceSettings(variant="standard", language="en")
        speaker_settings = {
            "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=shared_voice),
            "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=shared_voice),
        }
        render_settings = RenderSettings(
            correction_mode="conservative",
            model_variant="standard",
            language="en",
            export_stems=True,
            loudness_preset="off",
            pause_between_turns_ms=120,
            crossfade_ms=10,
            metadata={"title": f"Deterministic Smoke Render ({source_format})"},
        )
        _, output_path = pipeline.render_project(dialogue_path, project_dir, speaker_settings, render_settings)
        _, second_output_path = pipeline.render_project(dialogue_path, project_dir, speaker_settings, render_settings)

    if output_path != second_output_path:
        raise RuntimeError("Smoke render output path changed between runs.")

    render_plan_path = project_dir / "render_plan.json"
    render_plan = json.loads(render_plan_path.read_text(encoding="utf-8"))
    stem_count = len(list((project_dir / "cache" / "utterances").glob("*.wav")))
    return SmokeRenderResult(
        source_format=source_format,
        output_path=output_path,
        project_dir=project_dir,
        cache_reused_on_second_pass=render_plan["metadata"].get("cache_reused_on_second_pass") == "True",
        stem_count=stem_count,
        render_plan_path=render_plan_path,
        dialogue_path=dialogue_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic end-to-end smoke render.")
    parser.add_argument("--output-root", type=Path, default=Path("build/smoke_render"))
    parser.add_argument("--format", choices=["txt", "md"], default="txt", dest="source_format")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    result = run_deterministic_smoke_render(args.output_root, source_format=args.source_format)
    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Source format: {result.source_format}")
        print(f"Smoke render output: {result.output_path}")
        print(f"Project dir: {result.project_dir}")
        print(f"Stem count: {result.stem_count}")
        print(f"Cache reused on second pass: {result.cache_reused_on_second_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
