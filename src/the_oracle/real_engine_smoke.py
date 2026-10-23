"""Opt-in real-engine Chatterbox smoke validation."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from the_oracle.audio.assemble import AudioSegment, assemble_dialogue, save_wav
from the_oracle.audio.export_flac import write_flac
from the_oracle.models.cache import ProjectCache
from the_oracle.models.project import VoiceSettings
from the_oracle.tts_engines.chatterbox_engine import ChatterboxEngine


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REAL_ENGINE_DIALOGUE = (
    ("A", "The recording is ready. Can you hear me clearly?"),
    ("B", "Yes. The room is quiet, and the levels look steady."),
    ("A", "Good. We should check the second microphone before we begin."),
    ("B", "I have it here. The cable is connected, and the meter is moving."),
    ("A", "Please read the first line at a comfortable pace."),
    ("B", "A small boat crossed the harbor just after sunrise."),
    ("A", "That sounded natural. Now try the numbers and abbreviation."),
    ("B", "The package weighs 2.5 kilograms and arrives at 8:45 a.m."),
    ("A", "Excellent. Leave a short pause, then ask me a question."),
    ("B", "Should we save this version before changing the voice settings?"),
    ("A", "Yes. The original stays safe, and we can compare the new take."),
    ("B", "All right. I will export the reviewed dialogue when we finish."),
)


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


def ensure_real_engine_inputs(output_root: str | Path) -> dict[str, Path]:
    paths = default_real_engine_paths(output_root)
    paths["inputs_dir"].mkdir(parents=True, exist_ok=True)
    if not paths["dialogue"].exists():
        dialogue_text = "\n".join(f"Speaker {speaker}: {text}" for speaker, text in REAL_ENGINE_DIALOGUE) + "\n"
        paths["dialogue"].write_text(dialogue_text, encoding="utf-8")
    bundled_references = {
        "speaker_a_reference": REPOSITORY_ROOT / "Seashells" / "generic" / "english_male_1.wav",
        "speaker_b_reference": REPOSITORY_ROOT / "Seashells" / "generic" / "english_reader_1.wav",
    }
    for key, source in bundled_references.items():
        destination = paths[key]
        if destination.exists():
            continue
        if not source.is_file():
            raise RuntimeError(f"Bundled speech reference is missing: {source}")
        shutil.copyfile(source, destination)
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


def _has_speech_like_variation(audio: np.ndarray, sample_rate: int) -> bool:
    """Reject stationary tones while allowing silence between spoken phrases."""
    flattened = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame_size = max(1, int(sample_rate * 0.1))
    frame_count = len(flattened) // frame_size
    if frame_count < 3:
        return False
    frames = flattened[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    peak = float(np.max(np.abs(flattened)))
    active = rms[rms > max(0.005, peak * 0.01)]
    return active.size >= 3 and float(np.std(active) / np.mean(active)) >= 0.02


def _validate_speech_reference(path: Path) -> None:
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Speech reference cannot be decoded: {path} ({exc})") from exc
    if audio.size == 0 or not np.isfinite(audio).all() or not _has_speech_like_variation(audio, sample_rate):
        raise RuntimeError(
            f"Speech reference has no speech-like variation: {path}. "
            "Use a valid recording of spoken language, not a tone or synthetic test signal."
        )


def _validate_smoke_output(
    output_path: Path,
    audio: np.ndarray,
    *,
    expected_sample_rate: int | None = None,
) -> None:
    """Require a decodable, non-silent FLAC with non-stationary speech-like audio."""
    if audio is None or len(audio) == 0:
        raise RuntimeError("Real-engine smoke produced no audio output.")
    if not np.isfinite(audio).all() or float(np.max(np.abs(audio))) <= 0:
        raise RuntimeError("Real-engine smoke produced silent output.")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Real-engine smoke output missing or empty: {output_path}")
    try:
        decoded, sample_rate = sf.read(output_path, dtype="float32", always_2d=False)
        info = sf.info(output_path)
    except Exception as exc:
        raise RuntimeError(f"Real-engine smoke output is not decodable audio: {output_path} ({exc})") from exc
    if info.format != "FLAC" or decoded.size == 0 or not np.isfinite(decoded).all():
        raise RuntimeError("Real-engine smoke output is not a valid FLAC audio stream.")
    if info.channels != 1 or decoded.ndim != 1 or len(decoded) != len(audio):
        raise RuntimeError("Real-engine smoke FLAC has an unexpected channel count or frame count.")
    if expected_sample_rate is not None and info.samplerate != expected_sample_rate:
        raise RuntimeError(
            f"Real-engine smoke FLAC sample rate is {info.samplerate} Hz; expected {expected_sample_rate} Hz."
        )
    if not np.allclose(decoded, np.asarray(audio, dtype=np.float32), atol=2e-5, rtol=0):
        raise RuntimeError("Real-engine smoke FLAC samples do not match the rendered audio.")
    if not _has_speech_like_variation(decoded, sample_rate):
        raise RuntimeError("Real-engine smoke output has no speech-like variation.")


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
    _validate_speech_reference(paths["speaker_a_reference"])
    _validate_speech_reference(paths["speaker_b_reference"])
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
    for index, (speaker, text) in enumerate(REAL_ENGINE_DIALOGUE):
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

    final_audio, sample_rate = assemble_dialogue(stems, crossfade_ms=20, loudness_preset="light")
    if final_audio is None or len(final_audio) == 0:
        raise RuntimeError("Real-engine smoke produced no audio output.")
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
    _validate_smoke_output(output_path, final_audio, expected_sample_rate=sample_rate)
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
    # Entry-point arm first (2026-10-08 audit standard): this process runs the
    # REAL native synthesis stack — the historically heaviest crash surface in
    # the repo. Consent-aware, idempotent, fail-closed — same contract as
    # cli.main and launch_gui.
    from the_oracle.crash import handlers as crash_handlers

    crash_handlers.install()
    crash_handlers.arm_native_capture()
    # This module imports the engine — and so huggingface_hub — at import time,
    # which means the offline constant has already been captured by the time we
    # get here; apply_offline_environment corrects both the env and the constant.
    from the_oracle.offline import apply_offline_environment

    apply_offline_environment()
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
