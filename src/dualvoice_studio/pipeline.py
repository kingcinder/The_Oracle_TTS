"""Chatterbox-only orchestration for ingesting, repairing, reviewing, and rendering projects."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dualvoice_studio.audio.assemble import AudioSegment, assemble_dialogue, load_audio, save_wav
from dualvoice_studio.audio.export_flac import write_flac
from dualvoice_studio.emotion.goemotions import GoEmotionsClassifier
from dualvoice_studio.models.cache import ProjectCache
from dualvoice_studio.models.project import RenderPlan, Utterance, VoiceProfile, VoiceSettings
from dualvoice_studio.speaker_attribution.heuristics import AnchorAssignments, DualSpeakerAttributor
from dualvoice_studio.text_ingest import TextIngestor
from dualvoice_studio.text_repair.repairer import TextRepairPipeline
from dualvoice_studio.tts_engines.chatterbox_engine import ChatterboxEngine, ChatterboxConditioning, SUPPORTED_VARIANTS
from dualvoice_studio.utils.hashing import build_chunk_hash
from dualvoice_studio.utils.logging import get_logger


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
    metadata: dict[str, str] = field(default_factory=dict)
    anchors: AnchorAssignments | None = None


class DualVoicePipeline:
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
        language = settings.language if variant == "multilingual" else "en"

        utterances: list[Utterance] = []
        for segment, repaired, decision in zip(document.segments, repaired_segments, decisions, strict=True):
            base = self._coerce_voice_settings(speaker_settings[decision.speaker].voice_settings)
            base.variant = variant
            base.language = language
            base.pause_ms = settings.pause_between_turns_ms
            base.crossfade_ms = settings.crossfade_ms
            emotion = self.emotions.classify(repaired.text)
            merged = VoiceSettings.from_mapping(base)
            for key, value in self.emotions.controls_for_emotion(emotion.label).items():
                if hasattr(merged, key):
                    setattr(merged, key, value)
            merged.variant = variant
            merged.language = language
            merged.pause_ms = settings.pause_between_turns_ms
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
                engine_params=replace(self._coerce_voice_settings(config.voice_settings), variant=variant, language=language),
            )
            for speaker, config in speaker_settings.items()
        }

        metadata = {
            "title": document.title,
            "artist": "DualVoice Studio",
            "comment": f"Rendered with Chatterbox ({variant})",
            "date": date.today().isoformat(),
            "software": "DualVoice Studio 0.2.0",
            "engine": "chatterbox",
            "chatterbox_version": chatterbox_version(),
            "model_variant": variant,
            "language": language,
            "cfg_weight": str(voice_profiles["A"].engine_params.cfg_weight),
            "exaggeration": str(voice_profiles["A"].engine_params.exaggeration),
            "reference_clips_used": "true",
            "watermark": "Perth watermark embedded by Chatterbox",
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

    def render(self, plan: RenderPlan, settings: RenderSettings) -> Path:
        variant = settings.model_variant
        project_cache = ProjectCache(plan.output_dir)
        engine = ChatterboxEngine(variant=variant)
        conditioning: dict[str, ChatterboxConditioning] = {}
        stem_segments: list[AudioSegment] = []

        for speaker, profile in plan.voice_profiles.items():
            cached_reference = engine.prepare_reference(project_cache, speaker, str(profile.primary_reference))
            profile.normalized_reference_audio = [Path(cached_reference.normalized_path)]
            profile.reference_audio_hash = cached_reference.original_hash
            conditioning_payload = engine.prepare_conditioning(project_cache, speaker, cached_reference, profile.engine_params)
            profile.conditioning_cache_id = conditioning_payload.cache_id
            conditioning[speaker] = conditioning_payload
            project_cache.save_json(f"profiles/{speaker.lower()}_voice_profile.json", profile.to_dict())

        previous_plan = self._load_previous_plan(project_cache)
        for utterance in plan.utterances:
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
            if not stem_path.exists():
                rendered = engine.synthesize(utterance.text_for_tts(), conditioning[utterance.speaker], utterance.engine_settings)
                save_wav(stem_path, rendered, engine.sample_rate)
            audio, sample_rate = load_audio(stem_path)
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

        plan.metadata["cache_reused_on_second_pass"] = str(not compute_incremental_changes(previous_plan, plan))
        final_audio, sample_rate = assemble_dialogue(
            stem_segments,
            crossfade_ms=settings.crossfade_ms,
            loudness_preset=settings.loudness_preset,
        )
        final_output = Path(plan.output_dir) / f"{Path(plan.source_path).stem}.flac"
        exported = write_flac(final_output, final_audio, sample_rate, plan.metadata)
        plan.update_hashes()
        project_cache.save_json("render_plan.json", plan.to_dict())
        self._write_correction_log(project_cache, plan)
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

    def render_preview(self, utterance: Utterance, profile: VoiceProfile, model_variant: str) -> Path:
        project_cache = ProjectCache(profile.primary_reference.resolve().parent / ".dualvoice_preview")
        engine = ChatterboxEngine(variant=model_variant)
        cached_reference = engine.prepare_reference(project_cache, utterance.speaker, str(profile.primary_reference))
        conditioning = engine.prepare_conditioning(project_cache, utterance.speaker, cached_reference, profile.engine_params)
        rendered = engine.synthesize(utterance.text_for_tts(), conditioning, utterance.engine_settings)
        preview_path = project_cache.stem_path(f"preview_{utterance.speaker}_{utterance.index}")
        save_wav(preview_path, rendered, engine.sample_rate)
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
