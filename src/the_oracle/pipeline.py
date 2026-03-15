"""Chatterbox-only orchestration for ingesting, repairing, reviewing, and rendering projects."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

from the_oracle.app_paths import normalize_output_filename
from the_oracle.audio.assemble import AudioSegment, assemble_dialogue, load_audio, save_wav
from the_oracle.audio.export_flac import next_available_output_path, write_flac
from the_oracle.device_support import resolve_chatterbox_device
from the_oracle.emotion.goemotions import GoEmotionsClassifier
from the_oracle.models.cache import ProjectCache
from the_oracle.models.project import RenderPlan, Utterance, VoiceProfile, VoiceSettings
from the_oracle.speaker_attribution.heuristics import AnchorAssignments, DualSpeakerAttributor
from the_oracle.text_ingest import TextIngestor
from the_oracle.text_repair.repairer import TextRepairPipeline
from the_oracle.tts_engines.chatterbox_engine import ChatterboxEngine, ChatterboxConditioning, SUPPORTED_VARIANTS
from the_oracle.utils.hashing import build_chunk_hash
from the_oracle.utils.logging import get_logger


LOGGER = get_logger(__name__)


def chatterbox_version() -> str:
    try:
        return version("chatterbox-tts")
    except PackageNotFoundError:
        return "not-installed"


@dataclass(slots=True)
class SpeakerSettings:
    reference_path: str
    voice_settings: VoiceSettings | dict[str, Any] = field(default_factory=VoiceSettings)
    emotion_reference_paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RenderSettings:
    correction_mode: str = "conservative"
    model_variant: str = "standard"
    language: str = "en"
    export_stems: bool = True
    loudness_preset: str = "light"
    pause_between_turns_ms: int = 180
    crossfade_ms: int = 20
    device_mode: str = "cpu"
    metadata: dict[str, str] = field(default_factory=dict)
    anchors: AnchorAssignments | None = None


@dataclass(slots=True)
class RenderProgress:
    stage: str
    detail: str
    current_step: int
    total_steps: int
    current_segment: int
    total_segments: int
    elapsed_seconds: float
    eta_seconds: float | None = None


class OraclePipeline:
    def __init__(self) -> None:
        self.ingestor = TextIngestor()
        self.repair = TextRepairPipeline()
        self.attributor = DualSpeakerAttributor()
        self.emotions = GoEmotionsClassifier()

    def available_model_variants(self) -> list[str]:
        return list(SUPPORTED_VARIANTS)

    def supported_languages(self, model_variant: str = "standard") -> dict[str, str]:
        if model_variant != "multilingual":
            return {"en": "English"}
        try:
            return ChatterboxEngine(model_variant).supported_languages()
        except Exception:
            return {"en": "English"}

    @staticmethod
    def _coerce_voice_settings(value: VoiceSettings | dict[str, Any]) -> VoiceSettings:
        return VoiceSettings.from_mapping(value)

    @staticmethod
    def _blend_control(base_value: float | int, target_value: float | int, intensity: float) -> float:
        return float(base_value) + ((float(target_value) - float(base_value)) * intensity)

    def _apply_emotion_and_naturalness(self, base: VoiceSettings, emotion_label: str) -> VoiceSettings:
        merged = VoiceSettings.from_mapping(base)
        intensity = max(0.0, min(2.0, merged.emotion_intensity))
        for key, value in self.emotions.controls_for_emotion(emotion_label).items():
            if not hasattr(merged, key):
                continue
            blended = self._blend_control(getattr(base, key), value, intensity)
            setattr(merged, key, int(round(blended)) if isinstance(getattr(base, key), int) else blended)

        naturalness = max(0.0, min(1.0, merged.naturalness))
        if naturalness > 0.0:
            merged.cfg_weight = max(0.2, merged.cfg_weight - (0.12 * naturalness))
            merged.temperature = min(1.5, merged.temperature + (0.18 * naturalness))
            merged.repetition_penalty = max(1.0, merged.repetition_penalty - (0.25 * naturalness))
            merged.min_p = min(0.2, merged.min_p + (0.03 * naturalness))
            merged.pause_ms = int(round(merged.pause_ms * (1.0 + (0.12 * naturalness))))
        return merged

    def prepare_plan(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        speaker_settings: dict[str, SpeakerSettings],
        settings: RenderSettings,
    ) -> RenderPlan:
        document = self.ingestor.ingest(input_path)
        repaired_segments = [self.repair.repair(segment.text, mode=settings.correction_mode) for segment in document.segments]
        decisions = self.attributor.assign(
            [segment.text for segment in document.segments],
            explicit_speakers=[segment.explicit_speaker for segment in document.segments],
            anchors=settings.anchors,
        )

        variant = settings.model_variant

        utterances: list[Utterance] = []
        for segment, repaired, decision in zip(document.segments, repaired_segments, decisions, strict=True):
            base = self._coerce_voice_settings(speaker_settings[decision.speaker].voice_settings)
            base.variant = variant
            base.language = base.language if variant == "multilingual" else "en"
            base.crossfade_ms = settings.crossfade_ms
            emotion = self.emotions.classify(repaired.text)
            merged = self._apply_emotion_and_naturalness(base, emotion.label)
            merged.variant = variant
            merged.language = base.language if variant == "multilingual" else "en"
            merged.pause_ms = max(0, int(round(merged.pause_ms)))
            merged.crossfade_ms = settings.crossfade_ms

            utterances.append(
                Utterance(
                    index=segment.index,
                    original_text=segment.text,
                    repaired_text=repaired.text,
                    speaker=decision.speaker,
                    emotion=emotion.label,
                    emotion_confidence=emotion.confidence,
                    emotion_score=emotion.confidence,
                    speaker_confidence=decision.confidence,
                    speaker_source=decision.reason,
                    explicit_speaker=segment.explicit_speaker,
                    source_line=segment.source_line,
                    pause_after_ms=merged.pause_ms,
                    parameters=merged.to_dict(),
                    corrections=repaired.corrections,
                    engine_settings=merged,
                )
            )

        voice_profiles = {
            speaker: VoiceProfile(
                name=f"Speaker {speaker}",
                speaker=speaker,
                reference_audio=[Path(config.reference_path)],
                neutral_reference=Path(config.reference_path),
                emotion_references={key: Path(value) for key, value in config.emotion_reference_paths.items()},
                engine_params=replace(
                    self._coerce_voice_settings(config.voice_settings),
                    variant=variant,
                    language=self._coerce_voice_settings(config.voice_settings).language if variant == "multilingual" else "en",
                ),
            )
            for speaker, config in speaker_settings.items()
        }

        speaker_languages = {profile.engine_params.language for profile in voice_profiles.values()}
        project_language = next(iter(speaker_languages)) if len(speaker_languages) == 1 else "mixed"

        metadata = {
            "title": document.title,
            "artist": "The Oracle",
            "comment": f"Rendered with Chatterbox ({variant})",
            "date": date.today().isoformat(),
            "software": "The Oracle 0.2.0",
            "engine": "chatterbox",
            "chatterbox_version": chatterbox_version(),
            "model_variant": variant,
            "language": project_language,
            "cfg_weight": str(voice_profiles["A"].engine_params.cfg_weight),
            "exaggeration": str(voice_profiles["A"].engine_params.exaggeration),
            "reference_clips_used": "true",
            "watermark": "Perth watermark embedded by Chatterbox",
            "device_mode": settings.device_mode,
            **settings.metadata,
        }
        plan = RenderPlan(
            title=document.title,
            source_path=str(input_path),
            output_dir=str(output_dir),
            engine="chatterbox",
            correction_mode=settings.correction_mode,
            metadata=metadata,
            utterances=utterances,
            voice_profiles=voice_profiles,
        )
        plan.update_hashes()
        return plan

    def render(
        self,
        plan: RenderPlan,
        settings: RenderSettings,
        progress_callback=None,
    ) -> Path:
        start_time = perf_counter()

        def emit_progress(
            *,
            stage: str,
            detail: str,
            current_step: int,
            total_steps: int,
            current_segment: int = 0,
            total_segments: int = 0,
            eta_seconds: float | None = None,
        ) -> None:
            if progress_callback is None:
                return
            progress_callback(
                RenderProgress(
                    stage=stage,
                    detail=detail,
                    current_step=current_step,
                    total_steps=total_steps,
                    current_segment=current_segment,
                    total_segments=total_segments,
                    elapsed_seconds=perf_counter() - start_time,
                    eta_seconds=eta_seconds,
                )
            )

        variant = settings.model_variant
        project_cache = ProjectCache(plan.output_dir)
        engine = ChatterboxEngine(variant=variant, device=resolve_chatterbox_device(settings.device_mode))
        conditioning: dict[str, ChatterboxConditioning] = {}
        stem_segments: list[AudioSegment] = []
        timing_entries: list[dict[str, Any]] = []
        total_segments = len(plan.utterances)
        total_steps = len(plan.voice_profiles) + total_segments + 3
        completed_steps = 0

        emit_progress(
            stage="Loading model",
            detail=f"Loading Chatterbox {variant} on {engine.device}",
            current_step=completed_steps,
            total_steps=total_steps,
            total_segments=total_segments,
        )
        ensure_model_ready = getattr(engine, "ensure_model_ready", None)
        if callable(ensure_model_ready):
            ensure_model_ready()
        completed_steps += 1
        emit_progress(
            stage="Loading model",
            detail=f"Chatterbox {variant} is ready",
            current_step=completed_steps,
            total_steps=total_steps,
            total_segments=total_segments,
        )

        for speaker, profile in plan.voice_profiles.items():
            emit_progress(
                stage="Preparing speaker",
                detail=f"Preparing speaker {speaker} reference audio and conditioning",
                current_step=completed_steps,
                total_steps=total_steps,
                total_segments=total_segments,
            )
            speaker_start = perf_counter()
            cached_reference = engine.prepare_reference(project_cache, speaker, str(profile.primary_reference))
            profile.normalized_reference_audio = [Path(cached_reference.normalized_path)]
            profile.reference_audio_hash = cached_reference.original_hash
            conditioning_payload = engine.prepare_conditioning(project_cache, speaker, cached_reference, profile.engine_params)
            profile.conditioning_cache_id = conditioning_payload.cache_id
            conditioning[speaker] = conditioning_payload
            project_cache.save_json(f"profiles/{speaker.lower()}_voice_profile.json", profile.to_dict())
            timing_entries.append(
                {
                    "type": "speaker_prep",
                    "speaker": speaker,
                    "seconds": round(perf_counter() - speaker_start, 6),
                }
            )
            completed_steps += 1
            emit_progress(
                stage="Preparing speaker",
                detail=f"Speaker {speaker} is ready",
                current_step=completed_steps,
                total_steps=total_steps,
                total_segments=total_segments,
            )

        previous_plan = self._load_previous_plan(project_cache)
        render_start = perf_counter()
        for utterance_index, utterance in enumerate(plan.utterances, start=1):
            profile = plan.voice_profiles[utterance.speaker]
            utterance.parameters = utterance.engine_settings.to_dict()
            utterance.chunk_hash = build_chunk_hash(
                speaker=utterance.speaker,
                repaired_text=utterance.repaired_text,
                engine_key=f"chatterbox:{variant}",
                engine_params=utterance.parameters,
                engine_version=engine.engine_version,
                reference_audio_hash=profile.reference_audio_hash,
            )
            stem_path = project_cache.stem_path(utterance.chunk_hash)
            average_segment_seconds = None
            if utterance_index > 1:
                average_segment_seconds = (perf_counter() - render_start) / float(utterance_index - 1)
            eta_seconds = None if average_segment_seconds is None else average_segment_seconds * (total_segments - utterance_index + 1)
            emit_progress(
                stage="Rendering segment",
                detail=f"Rendering segment {utterance_index}/{total_segments} for speaker {utterance.speaker}",
                current_step=completed_steps,
                total_steps=total_steps,
                current_segment=utterance_index,
                total_segments=total_segments,
                eta_seconds=eta_seconds,
            )
            segment_start = perf_counter()
            synthesize_seconds = 0.0
            cache_hit = stem_path.exists()
            if not stem_path.exists():
                synth_start = perf_counter()
                rendered = engine.synthesize(utterance.text_for_tts(), conditioning[utterance.speaker], utterance.engine_settings)
                synthesize_seconds = perf_counter() - synth_start
                save_wav(stem_path, rendered, engine.sample_rate)
            load_start = perf_counter()
            audio, sample_rate = load_audio(stem_path)
            load_audio_seconds = perf_counter() - load_start
            utterance.duration_seconds = len(audio) / sample_rate
            stem_segments.append(
                AudioSegment(
                    path=str(stem_path),
                    sample_rate=sample_rate,
                    pause_after_ms=utterance.pause_after_ms,
                    duration_seconds=utterance.duration_seconds,
                )
            )
            if settings.export_stems:
                project_cache.export_stem(stem_path, f"stems/{utterance.index:04d}_{utterance.speaker}.wav")
            segment_seconds = perf_counter() - segment_start
            timing_entries.append(
                {
                    "type": "utterance",
                    "index": utterance.index,
                    "speaker": utterance.speaker,
                    "cache_hit": cache_hit,
                    "synthesize_seconds": round(synthesize_seconds, 6),
                    "load_audio_seconds": round(load_audio_seconds, 6),
                    "segment_total_seconds": round(segment_seconds, 6),
                    "inter_segment_overhead_seconds": round(max(0.0, segment_seconds - synthesize_seconds), 6),
                }
            )
            completed_steps += 1
            emit_progress(
                stage="Rendering segment",
                detail=f"Segment {utterance_index}/{total_segments} ready",
                current_step=completed_steps,
                total_steps=total_steps,
                current_segment=utterance_index,
                total_segments=total_segments,
                eta_seconds=None if utterance_index >= total_segments else average_segment_seconds * (total_segments - utterance_index) if average_segment_seconds is not None else None,
            )

        plan.metadata["cache_reused_on_second_pass"] = str(not compute_incremental_changes(previous_plan, plan))
        emit_progress(
            stage="Assembling audio",
            detail="Applying pauses, crossfades, and loudness settings",
            current_step=completed_steps,
            total_steps=total_steps,
            current_segment=total_segments,
            total_segments=total_segments,
        )
        final_audio, sample_rate = assemble_dialogue(
            stem_segments,
            crossfade_ms=settings.crossfade_ms,
            loudness_preset=settings.loudness_preset,
        )
        completed_steps += 1
        requested_filename = normalize_output_filename(str(settings.metadata.get("output_filename", ""))) or f"{Path(plan.source_path).stem}.flac"
        final_output = next_available_output_path(Path(plan.output_dir) / requested_filename)
        emit_progress(
            stage="Writing output",
            detail=f"Writing {final_output.name}",
            current_step=completed_steps,
            total_steps=total_steps,
            current_segment=total_segments,
            total_segments=total_segments,
        )
        exported = write_flac(final_output, final_audio, sample_rate, plan.metadata)
        plan.update_hashes()
        project_cache.save_json("render_plan.json", plan.to_dict())
        self._write_correction_log(project_cache, plan)
        self._write_render_timing_log(project_cache, timing_entries)
        emit_progress(
            stage="Complete",
            detail=f"Render complete: {exported.name}",
            current_step=total_steps,
            total_steps=total_steps,
            current_segment=total_segments,
            total_segments=total_segments,
            eta_seconds=0.0,
        )
        return exported

    def render_project(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        speaker_settings: dict[str, SpeakerSettings],
        settings: RenderSettings,
    ) -> tuple[RenderPlan, Path]:
        plan = self.prepare_plan(input_path, output_dir, speaker_settings, settings)
        output_path = self.render(plan, settings)
        return plan, output_path

    def render_preview(
        self,
        utterance: Utterance,
        profile: VoiceProfile,
        model_variant: str,
        device_mode: str = "cpu",
        progress_callback=None,
    ) -> Path:
        start_time = perf_counter()

        def emit_preview_progress(stage: str, detail: str, current_step: int, total_steps: int = 4) -> None:
            if progress_callback is None:
                return
            progress_callback(
                RenderProgress(
                    stage=stage,
                    detail=detail,
                    current_step=current_step,
                    total_steps=total_steps,
                    current_segment=0,
                    total_segments=0,
                    elapsed_seconds=perf_counter() - start_time,
                    eta_seconds=None,
                )
            )

        reference_path = profile.primary_reference.resolve()
        project_cache = ProjectCache(reference_path.parent / ".oracle_preview")
        engine = ChatterboxEngine(variant=model_variant, device=resolve_chatterbox_device(device_mode))
        emit_preview_progress("Loading model", f"Loading Chatterbox {model_variant} on {engine.device}", 0)
        ensure_model_ready = getattr(engine, "ensure_model_ready", None)
        if callable(ensure_model_ready):
            ensure_model_ready()
        emit_preview_progress("Preparing reference", f"Preparing speaker {utterance.speaker} reference audio", 1)
        cached_reference = engine.prepare_reference(project_cache, utterance.speaker, str(reference_path))
        emit_preview_progress("Preparing conditioning", f"Preparing speaker {utterance.speaker} conditioning", 2)
        conditioning = engine.prepare_conditioning(project_cache, utterance.speaker, cached_reference, profile.engine_params)
        emit_preview_progress("Generating preview", f"Generating preview for segment {utterance.index}", 3)
        rendered = engine.synthesize(utterance.text_for_tts(), conditioning, utterance.engine_settings)
        preview_path = project_cache.preview_path(utterance.speaker, utterance.index)
        save_wav(preview_path, rendered, engine.sample_rate)
        emit_preview_progress("Complete", f"Preview ready: {preview_path.name}", 4)
        return preview_path

    def _load_previous_plan(self, project_cache: ProjectCache) -> dict[str, Any]:
        path = Path(project_cache.project_dir) / "render_plan.json"
        if not path.exists():
            return {"utterances": []}
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def _write_correction_log(self, project_cache: ProjectCache, plan: RenderPlan) -> None:
        entries: list[dict[str, Any]] = []
        diff_lines: list[str] = []
        for utterance in plan.utterances:
            for correction in utterance.corrections:
                entry = {
                    "index": utterance.index,
                    "speaker": utterance.speaker,
                    "stage": correction.stage,
                    "before": correction.before,
                    "after": correction.after,
                }
                entries.append(entry)
                diff_lines.append(f"[{utterance.index}] {correction.stage}\n- {correction.before}\n+ {correction.after}\n")
        payload = {"count": len(entries), "entries": entries}
        project_cache.save_json("logs/text_corrections.json", payload)
        project_cache.save_json("logs/corrections.json", payload)
        project_cache.write_text("logs/corrections.diff", "\n".join(diff_lines))

    def _write_render_timing_log(self, project_cache: ProjectCache, timing_entries: list[dict[str, Any]]) -> None:
        utterance_entries = [entry for entry in timing_entries if entry.get("type") == "utterance"]
        payload = {
            "count": len(timing_entries),
            "entries": timing_entries,
            "summary": {
                "utterance_count": len(utterance_entries),
                "cache_hits": sum(1 for entry in utterance_entries if entry.get("cache_hit")),
                "cache_misses": sum(1 for entry in utterance_entries if not entry.get("cache_hit")),
                "total_synthesize_seconds": round(sum(entry.get("synthesize_seconds", 0.0) for entry in utterance_entries), 6),
                "total_overhead_seconds": round(sum(entry.get("inter_segment_overhead_seconds", 0.0) for entry in utterance_entries), 6),
            },
        }
        project_cache.save_json("logs/render_timings.json", payload)


def compute_incremental_changes(old_plan: RenderPlan | dict[str, Any], new_plan: RenderPlan | dict[str, Any] | list[Utterance]) -> list[int]:
    def _rows(value: RenderPlan | dict[str, Any] | list[Utterance]) -> list[dict[str, Any]]:
        if isinstance(value, RenderPlan):
            return [utterance.to_dict() for utterance in value.utterances]
        if isinstance(value, list):
            return [utterance.to_dict() if isinstance(utterance, Utterance) else utterance for utterance in value]
        return list(value.get("utterances", []))

    previous = {item["index"]: item.get("chunk_hash", "") or item.get("cache_key", "") for item in _rows(old_plan)}
    return [
        item["index"]
        for item in _rows(new_plan)
        if previous.get(item["index"], "") != (item.get("chunk_hash", "") or item.get("cache_key", ""))
    ]


diff_render_plan = compute_incremental_changes
select_changed_utterances = compute_incremental_changes
find_changed_utterances = compute_incremental_changes
