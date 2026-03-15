"""Opt-in real-engine Chatterbox smoke validation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from the_oracle.audio.assemble import AudioSegment, assemble_dialogue, save_wav
from the_oracle.audio.export_flac import write_flac
from the_oracle.models.cache import ProjectCache
from the_oracle.models.project import VoiceSettings
from the_oracle.tts_engines.chatterbox_engine import ChatterboxEngine


REAL_ENGINE_DIALOGUE = {
    "A": "The Oracle is online.",
    "B": "Confirm the live Chatterbox path.",
}


@dataclass(slots=True)
class RealEngineSmokeResult:
    runtime_seconds: float
    model_variant: str
    device: str
    output_path: Path
    report_path: Path
    dialogue_path: Path
    speaker_a_reference: Path
    speaker_b_reference: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_seconds": round(self.runtime_seconds, 3),
            "model_variant": self.model_variant,
            "device": self.device,
            "output_path": str(self.output_path),
            "report_path": str(self.report_path),
            "dialogue_path": str(self.dialogue_path),
            "speaker_a_reference": str(self.speaker_a_reference),
            "speaker_b_reference": str(self.speaker_b_reference),
        }


def default_real_engine_paths(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    inputs_dir = root / "inputs"
    return {
        "root": root,
        "inputs_dir": inputs_dir,
        "dialogue": inputs_dir / "real_engine_dialogue.txt",
        "speaker_a_reference": inputs_dir / "speaker_a_ref.wav",
        "speaker_b_reference": inputs_dir / "speaker_b_ref.wav",
        "project_dir": root / "project",
        "report": root / "report.json",
        "output": root / "real_engine_smoke.flac",
    }


def _write_reference_clip(path: Path, frequency: float) -> Path:
    sample_rate = 24000
    seconds = 1.0
    time_axis = np.arange(int(sample_rate * seconds), dtype=np.float32) / np.float32(sample_rate)
    audio = 0.18 * np.sin(2.0 * np.pi * frequency * time_axis)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(path, np.asarray(audio, dtype=np.float32), sample_rate)
    return path


def ensure_real_engine_inputs(output_root: str | Path) -> dict[str, Path]:
    paths = default_real_engine_paths(output_root)
    paths["inputs_dir"].mkdir(parents=True, exist_ok=True)
    if not paths["dialogue"].exists():
        dialogue_text = "\n".join(f"Speaker {speaker}: {text}" for speaker, text in REAL_ENGINE_DIALOGUE.items()) + "\n"
        paths["dialogue"].write_text(dialogue_text, encoding="utf-8")
    if not paths["speaker_a_reference"].exists():
        _write_reference_clip(paths["speaker_a_reference"], 220.0)
    if not paths["speaker_b_reference"].exists():
        _write_reference_clip(paths["speaker_b_reference"], 330.0)
    return paths


def real_engine_smoke_prerequisites(output_root: str | Path) -> dict[str, object]:
    paths = default_real_engine_paths(output_root)
    try:
        import perth
        from chatterbox.tts import ChatterboxTTS

        watermarker = getattr(perth, "PerthImplicitWatermarker", None)
        chatterbox_import = {
            "ok": True,
            "symbol": str(ChatterboxTTS),
            "perth_watermarker_callable": callable(watermarker),
        }
    except Exception as exc:
        chatterbox_import = {"ok": False, "error": str(exc)}

    try:
        import soundfile as sf

        soundfile_ok = hasattr(sf, "write")
    except Exception:
        soundfile_ok = False

    return {
        "ready": chatterbox_import["ok"] and chatterbox_import.get("perth_watermarker_callable", False) and soundfile_ok,
        "chatterbox_import": chatterbox_import,
        "soundfile_ok": soundfile_ok,
        "dialogue_exists": paths["dialogue"].exists(),
        "speaker_a_reference_exists": paths["speaker_a_reference"].exists(),
        "speaker_b_reference_exists": paths["speaker_b_reference"].exists(),
        "can_generate_missing_inputs": True,
        "expected_paths": {key: str(value) for key, value in paths.items() if key != "inputs_dir"},
    }


def run_real_engine_smoke(
    output_root: str | Path = Path("build/real_engine_smoke"),
    model_variant: str = "standard",
    device: str | None = None,
    language: str = "en",
) -> RealEngineSmokeResult:
    prerequisites = real_engine_smoke_prerequisites(output_root)
    if not prerequisites["ready"]:
        raise RuntimeError(f"Real-engine smoke prerequisites are not ready: {prerequisites}")
    paths = ensure_real_engine_inputs(output_root)
    engine = ChatterboxEngine(variant=model_variant, device=device)
    voice_settings = VoiceSettings(variant=model_variant, language=language)
    project_cache = ProjectCache(paths["project_dir"])
    start = time.perf_counter()

    conditioning = {}
    references = {
        "A": paths["speaker_a_reference"],
        "B": paths["speaker_b_reference"],
    }
    for speaker, reference_path in references.items():
        cached_reference = engine.prepare_reference(project_cache, speaker, str(reference_path))
        conditioning[speaker] = engine.prepare_conditioning(project_cache, speaker, cached_reference, voice_settings)

    stems: list[AudioSegment] = []
    for index, (speaker, text) in enumerate(REAL_ENGINE_DIALOGUE.items()):
        rendered = engine.synthesize(text, conditioning[speaker], voice_settings)
        stem_path = project_cache.stem_path(f"real_engine_{index:02d}_{speaker}")
        save_wav(stem_path, rendered, engine.sample_rate)
        stems.append(
            AudioSegment(
                path=str(stem_path),
                sample_rate=engine.sample_rate,
                pause_after_ms=150,
                duration_seconds=len(rendered) / engine.sample_rate,
            )
        )

    final_audio, sample_rate = assemble_dialogue(stems, crossfade_ms=20, loudness_preset="off")
    output_path = write_flac(
        paths["output"],
        final_audio,
        sample_rate,
        {
            "title": "Real Engine Smoke",
            "engine": "chatterbox",
            "model_variant": model_variant,
            "device": engine.device,
        },
    )
    runtime_seconds = time.perf_counter() - start
    result = RealEngineSmokeResult(
        runtime_seconds=runtime_seconds,
        model_variant=model_variant,
        device=engine.device,
        output_path=output_path,
        report_path=paths["report"],
        dialogue_path=paths["dialogue"],
        speaker_a_reference=paths["speaker_a_reference"],
        speaker_b_reference=paths["speaker_b_reference"],
    )
    paths["report"].write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an opt-in real-engine Chatterbox smoke render.")
    parser.add_argument("--output-root", type=Path, default=Path("build/real_engine_smoke"))
    parser.add_argument("--model-variant", choices=["standard", "multilingual", "turbo"], default="standard")
    parser.add_argument("--device", default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    result = run_real_engine_smoke(
        output_root=args.output_root,
        model_variant=args.model_variant,
        device=args.device,
        language=args.language,
    )
    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Runtime seconds: {result.runtime_seconds:.3f}")
        print(f"Model variant: {result.model_variant}")
        print(f"Device: {result.device}")
        print(f"Output path: {result.output_path}")
        print(f"Report path: {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
