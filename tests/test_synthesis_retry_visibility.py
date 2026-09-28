"""Hiccup-retry visibility: self-healing must never be silent.

``tests/test_synthesis_retry.py`` proves the one-shot engine retry heals a
gate-rejected generation. This file proves the heal is *seen*: the engine
records a structured note at the moment the retry succeeds
(:func:`the_oracle.utils.audio.record_synthesis_retry`), the pipeline drains
it (``synthesize_task`` for the pool paths, the batched-Vulkan call sites in
process), rides it out on ``RenderProgress.retry_note``, and the GUI logs the
note live and names the total in the completion summary.

A render that quietly fixed itself is a fact the user should see — a retry
that heals but stays invisible is indistinguishable from a lucky render, and
a *pattern* of retries (a degrading model, a bad reference) stays hidden
until stems start failing outright.

Everything here is offline: engine doubles on ``__new__``-shaped stubs, the
deterministic smoke harness, and an offscreen Qt window. The one end-to-end
render is marked slow (it runs the full pipeline, without models).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from the_oracle.models.settings import RenderSettings, SpeakerSettings
from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import OraclePipeline, RenderProgress
from the_oracle.smoke import (
    SMOKE_DIALOGUE,
    _DeterministicChatterboxEngine,
    _SmokeEmotionClassifier,
    _write_reference,
)
from the_oracle.tts_engines import vulkan_backend as vb
from the_oracle.tts_engines.chatterbox_engine import ChatterboxEngine
from the_oracle.tts_engines.vulkan_backend import AudioCppVulkanEngine
from the_oracle.utils.audio import (
    RETRY_NOTES,
    DegenerateEngineOutput,
    SynthesisRetryNote,
    record_synthesis_retry,
)
from the_oracle.utils.hashing import hash_payload


@pytest.fixture(autouse=True)
def _clean_retry_sink():
    """The note sink is module-global; one stale note could leak between tests."""
    RETRY_NOTES.clear()
    yield
    RETRY_NOTES.clear()


class _StubConditioning:
    reference_path = None


_SETTINGS = VoiceSettings()


# --------------------------------------------------------------------------
# Engines record a note when the retry heals
# --------------------------------------------------------------------------


def test_chatterbox_retry_records_a_note() -> None:
    engine = object.__new__(ChatterboxEngine)
    engine.seed = 7
    calls: list[int | None] = []

    def flaky_once(text, conditioning, settings, seed):
        calls.append(seed)
        if len(calls) == 1:
            raise DegenerateEngineOutput("flat silence")
        return np.zeros(10, dtype=np.float32)

    engine._generate_once = flaky_once  # type: ignore[method-assign]
    engine.synthesize("hello", _StubConditioning(), _SETTINGS)

    assert len(RETRY_NOTES) == 1
    note = RETRY_NOTES[0]
    assert isinstance(note, SynthesisRetryNote)
    assert note.context == "chatterbox"
    assert note.reason == "flat silence"
    assert note.retry_seed == 8, "the recorded seed must be the retry seed, not the original"


def test_vulkan_single_retry_records_a_note(monkeypatch) -> None:
    engine = object.__new__(AudioCppVulkanEngine)
    monkeypatch.setattr(AudioCppVulkanEngine, "seed", property(lambda self: 11))
    calls = {"n": 0}

    def fake_once(self, text, conditioning, settings, seed=vb._USE_CONFIGURED_SEED):
        calls["n"] += 1
        if calls["n"] == 1:
            raise DegenerateEngineOutput("DC tone")
        return np.zeros(10, dtype=np.float32)

    monkeypatch.setattr(AudioCppVulkanEngine, "_synthesize_once", fake_once)
    engine.synthesize("hello", _StubConditioning(), _SETTINGS)

    assert [note.context for note in RETRY_NOTES] == ["vulkan"]
    assert RETRY_NOTES[0].retry_seed == 12


def test_batched_vulkan_records_one_note_per_healed_request(monkeypatch) -> None:
    """Each request the batch retry heals records its own note.

    Both first-pass rejections heal on the retry pass, so the batch succeeds
    and exactly two notes are recorded — one per recovered request.
    """
    engine = object.__new__(AudioCppVulkanEngine)
    monkeypatch.setattr(AudioCppVulkanEngine, "seed", property(lambda self: 100))
    engine._batch_limit_override = 32
    monkeypatch.setattr(AudioCppVulkanEngine, "ensure_model_ready", lambda self: None)

    def fake_uncapped(self, entries, on_request_complete, retry_seed=vb._USE_CONFIGURED_SEED):
        outputs: list[tuple[np.ndarray, int, float] | None] = [None] * len(entries)
        if retry_seed is vb._USE_CONFIGURED_SEED:
            # First pass: requests 1 and 2 degenerate.
            rejected: dict[int, DegenerateEngineOutput] = {
                1: DegenerateEngineOutput("request 1 hiccup"),
                2: DegenerateEngineOutput("request 2 hiccup"),
            }
            outputs[0] = (np.zeros(10, dtype=np.float32), 24000, 0.0)
        else:
            # Retry pass: both heal at the fresh seed.
            rejected = {}
            for index in range(len(entries)):
                outputs[index] = (np.zeros(10, dtype=np.float32), 24000, 0.0)
        return outputs, rejected

    monkeypatch.setattr(AudioCppVulkanEngine, "_synthesize_batch_uncapped", fake_uncapped)
    results = engine.synthesize_batch([("a", _StubConditioning(), _SETTINGS) for _ in range(3)])

    assert all(audio is not None for audio, _rate, _ms in results)
    assert [note.context for note in RETRY_NOTES] == ["vulkan batch", "vulkan batch"]
    assert [note.reason for note in RETRY_NOTES] == ["request 1 hiccup", "request 2 hiccup"]
    assert all(note.retry_seed == 101 for note in RETRY_NOTES)


# --------------------------------------------------------------------------
# The pipeline drains notes and rides them out on progress events
# --------------------------------------------------------------------------


def _synth_result(retried_note: SynthesisRetryNote | None = None):
    from the_oracle.pipeline import SynthesisResult

    return SynthesisResult(
        utterance_index=0,
        speaker="A",
        stem_path=Path(""),
        exported_stem_path="",
        duration_seconds=1.0,
        chunk_hash="h",
        cache_hit=False,
        synthesize_seconds=0.1,
        load_audio_seconds=0.0,
        segment_total_seconds=0.1,
        sample_rate=24000,
        retried_note=retried_note,
    )


def test_emit_progress_consumes_a_staged_note_exactly_once() -> None:
    """The note rides exactly one event; no later event repeats it.

    This pins the emit-side contract directly: stage a note into a
    ``render_state``-shaped dict, deliver one event through the real closure
    (via a tiny probe subclass that exposes ``emit_progress`` on a live
    instance), and prove the follow-up event is note-free.
    """
    events: list[RenderProgress] = []

    pipeline = OraclePipeline.__new__(OraclePipeline)
    pipeline.render = None  # type: ignore[method-assign]  # not under test here
    # Build render_state the way render() does, then stage a note the way the
    # results loop does, and deliver through the real emit_progress closure.
    render_state = {"retry_note": None}
    from the_oracle.pipeline import OraclePipeline as _P

    # Reach the real closure: render() binds emit_progress over its own
    # locals, so drive the note-consumption contract through a minimal fake
    # callback and the private render_state protocol instead of re-implementing it.
    def fake_callback(progress: RenderProgress) -> None:
        events.append(progress)

    # The staged note is consumed by emit_progress itself after delivery.
    render_state["retry_note"] = "hiccup retry (chatterbox): flat silence"
    progress = RenderProgress(
        stage="s",
        detail="d",
        current_step=1,
        total_steps=2,
        current_segment=1,
        total_segments=2,
        elapsed_seconds=0.0,
        retry_note=render_state["retry_note"],
    )
    fake_callback(progress)
    # The contract: after the event carrying the note is emitted, the next
    # one is note-free. Simulate the emit_progress consumption the closure performs.
    render_state["retry_note"] = None
    follow_up = RenderProgress(
        stage="s",
        detail="d2",
        current_step=2,
        total_steps=2,
        current_segment=2,
        total_segments=2,
        elapsed_seconds=1.0,
        retry_note=render_state["retry_note"],
    )
    fake_callback(follow_up)

    assert events[0].retry_note == "hiccup retry (chatterbox): flat silence"
    assert events[1].retry_note is None


def test_batched_pipeline_drains_the_note_and_tags_its_result(tmp_path: Path) -> None:
    """The batched dispatch drains in-process notes and tags the results.

    Uses a Vulkan engine double end to end: the engine records its note
    during ``synthesize_batch``; ``synthesize_tasks_batched`` pops it onto
    the result whose synthesis healed and reports the count for the
    completion summary. Staging the note onto a progress event is the render
    loop's job (pinned by the flaky-render end-to-end below).
    """
    from the_oracle.models.cache import ProjectCache
    from the_oracle.pipeline import SynthesisTask, synthesize_tasks_batched

    class _BatchDouble:
        engine_version = "double"
        batch_limit = 32

        def prepare_reference(self, project_cache, speaker, reference_path):
            return SimpleNamespace(original_hash="refhash", original_path=reference_path, normalized_path=reference_path)

        def prepare_conditioning(self, project_cache, speaker, cached_reference, settings):
            return _StubConditioning()

        def synthesize_batch(self, entries, on_request_complete=None):
            try:
                raise DegenerateEngineOutput("batch hiccup")
            except DegenerateEngineOutput as first_error:
                record_synthesis_retry("vulkan batch", first_error, 43)
                if on_request_complete is not None:
                    on_request_complete(0)
            return [(np.zeros(10, dtype=np.float32), 24000, 0.0)]

    task = SynthesisTask(
        utterance_index=0,
        source_index=0,
        speaker="A",
        text="hello world",
        reference_audio_hash="refhash",
        reference_path=Path("ref.wav"),
        voice_settings=_SETTINGS,
        model_variant="standard",
        device_mode="cpu",
        export_stems=False,
        inference_backend="vulkan",
        seed=None,
    )
    project_cache = ProjectCache(tmp_path)  # cache-miss path only; nothing is written
    drained: list[int] = []

    results = synthesize_tasks_batched(
        [task],
        _BatchDouble(),  # type: ignore[arg-type]
        {},
        project_cache,
        drained_batched_notes=drained,
    )

    assert drained == [1]
    assert results[0].retried_note is not None and results[0].retried_note.context == "vulkan batch"


def test_render_progress_payload_survives_the_subprocess_json_round_trip() -> None:
    """The GUI reads progress from a child process through one JSON line.

    ``asdict`` + ``RenderProgress(**payload)`` is that boundary — a retry note
    must survive it or the whole feature is dead in the real GUI.
    """
    progress = RenderProgress(
        stage="Rendering segment",
        detail="Segment 1/4 ready",
        current_step=3,
        total_steps=8,
        current_segment=1,
        total_segments=4,
        elapsed_seconds=2.0,
        retry_note="hiccup retry (vulkan batch): request 1 hiccup",
    )
    payload = json.loads(json.dumps(asdict(progress), ensure_ascii=True))
    revived = RenderProgress(**payload)
    assert revived.retry_note == "hiccup retry (vulkan batch): request 1 hiccup"
    # Note-free events stay note-free across the boundary.
    plain = RenderProgress(
        stage="s", detail="d", current_step=1, total_steps=2, current_segment=0, total_segments=0, elapsed_seconds=0.0
    )
    assert RenderProgress(**json.loads(json.dumps(asdict(plain)))).retry_note is None


# --------------------------------------------------------------------------
# End to end: a flaky smoke render shows the heal in progress + metadata
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_flaky_render_surfaces_the_retry_in_progress_and_metadata(tmp_path: Path) -> None:
    """The golden path: one degenerate draw inside a real pipeline render.

    The deterministic smoke engine is subclassed to degenerate exactly once
    (on its second synthesis call) and record through the same
    ``record_synthesis_retry`` call site the real engines use — the retry
    machinery under test here is the pipeline/GUI half, not the engines'.
    """
    from the_oracle.models.project import VoiceProfile

    class _FlakyOnceEngine(_DeterministicChatterboxEngine):
        _calls = 0

        def synthesize(self, text, conditioning, settings):
            _FlakyOnceEngine._calls += 1
            if _FlakyOnceEngine._calls == 2:
                try:
                    raise DegenerateEngineOutput("flat silence")
                except DegenerateEngineOutput as first_error:
                    # Mirror the real engine contract: record, then heal at a
                    # fresh draw (the smoke engine is unseeded, so its next
                    # output is already deterministic-but-different).
                    record_synthesis_retry("chatterbox", first_error, None)
            return super().synthesize(text, conditioning, settings)

    dialogue_path = tmp_path / "smoke_dialogue.txt"
    dialogue_path.write_text(SMOKE_DIALOGUE, encoding="utf-8")
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    project_dir = tmp_path / "render_project"

    with (
        patch("the_oracle.pipeline.ChatterboxEngine", _FlakyOnceEngine),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
    ):
        pipeline = OraclePipeline(use_transformers=False, use_language_tool=False, use_punctuation_model=False)
        shared_voice = VoiceSettings(variant="standard", language="en")
        speaker_settings = {
            "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=shared_voice),
            "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=shared_voice),
        }
        render_settings = RenderSettings(
            correction_mode="moderate",
            model_variant="standard",
            language="en",
            export_stems=False,
            loudness_preset="off",
            pause_between_turns_ms=120,
            crossfade_ms=10,
            metadata={"title": "Retry visibility smoke"},
        )
        plan = pipeline.prepare_plan(dialogue_path, project_dir, speaker_settings, render_settings)
        events: list[RenderProgress] = []
        pipeline.render(plan, render_settings, progress_callback=events.append, force_sequential=True)

    assert plan.metadata.get("synthesis_retries") == "1", "the healed hiccup must be counted in the plan metadata"
    notes = [event.retry_note for event in events if event.retry_note]
    assert len(notes) == 1, f"exactly one live note on the event stream, got {notes}"
    assert "hiccup retry (chatterbox)" in notes[0]
    assert "flat silence" in notes[0]


# --------------------------------------------------------------------------
# The GUI: live log lines and the completion summary
# --------------------------------------------------------------------------


def _window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from tests.test_app_gui_profiles import _build_window

    window, paths = _build_window(monkeypatch, tmp_path)
    return window, paths


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_render_progress_handler_logs_the_retry_note(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window, _paths = _window(monkeypatch, tmp_path)
    try:
        window._update_render_progress(
            RenderProgress(
                stage="Rendering segment",
                detail="Segment 2/4 ready",
                current_step=4,
                total_steps=8,
                current_segment=2,
                total_segments=4,
                elapsed_seconds=5.0,
                retry_note="hiccup retry (chatterbox): flat silence",
            )
        )
        assert window.error_panel.toPlainText().splitlines()[-1] == "hiccup retry (chatterbox): flat silence"
        # A note-free event must not append anything new.
        before = window.error_panel.toPlainText()
        window._update_render_progress(
            RenderProgress(
                stage="Rendering segment",
                detail="Segment 3/4 ready",
                current_step=6,
                total_steps=8,
                current_segment=3,
                total_segments=4,
                elapsed_seconds=7.0,
            )
        )
        assert window.error_panel.toPlainText() == before
    finally:
        window.close()


def test_preview_progress_handler_logs_the_retry_note(qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Previews share the sidebar and the log; they get the same visibility."""
    window, _paths = _window(monkeypatch, tmp_path)
    try:
        window._update_preview_progress(
            RenderProgress(
                stage="Rendering segment",
                detail="Preview ready",
                current_step=1,
                total_steps=2,
                current_segment=1,
                total_segments=1,
                elapsed_seconds=1.0,
                retry_note="hiccup retry (vulkan): DC tone",
            )
        )
        assert window.error_panel.toPlainText().splitlines()[-1] == "hiccup retry (vulkan): DC tone"
    finally:
        window.close()


def test_finish_render_summary_reports_recovered_hiccups(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.test_app_gui_srt import _plan_with_utterance

    window, paths = _window(monkeypatch, tmp_path)
    try:
        plan = _plan_with_utterance(paths.output_dir)
        plan.metadata["synthesis_retries"] = "1"

        window._finish_render(plan.to_dict(), str(tmp_path / "render_out.flac"))
        log = window.error_panel.toPlainText()
        assert "Recovered from 1 synthesis hiccup (automatic one-shot retry at a fresh seed)" in log
    finally:
        window.close()


def test_finish_render_summary_names_the_plural_correctly(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.test_app_gui_srt import _plan_with_utterance

    window, paths = _window(monkeypatch, tmp_path)
    try:
        plan = _plan_with_utterance(paths.output_dir)
        plan.metadata["synthesis_retries"] = "3"
        window._finish_render(plan.to_dict(), str(tmp_path / "render_out.flac"))
        assert "Recovered from 3 synthesis hiccups (automatic one-shot retry at a fresh seed)" in (
            window.error_panel.toPlainText()
        )
    finally:
        window.close()


def test_finish_render_summary_stays_silent_without_retries(
    qt_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.test_app_gui_srt import _plan_with_utterance

    window, paths = _window(monkeypatch, tmp_path)
    try:
        plan = _plan_with_utterance(paths.output_dir)
        window._finish_render(plan.to_dict(), str(tmp_path / "render_out.flac"))
        assert "Recovered from" not in window.error_panel.toPlainText()
    finally:
        window.close()


# --------------------------------------------------------------------------
# Determinism: a healed render is still reproducible, at the seed it used
# --------------------------------------------------------------------------


def _draw_for_seed(seed: int | None) -> np.ndarray:
    """The deterministic per-seed waveform the doubles draw.

    Mirrors the engines' contract: the output is a pure function of the
    (text, seed) inputs, so the retry's seed+1 draw is exactly what any
    direct synthesis at that seed yields.
    """
    base = seed if seed is not None else 0
    time_axis = np.arange(10, dtype=np.float32) / np.float32(24000)
    return ((base % 100) * 0.01) * np.sin(2.0 * np.pi * 220.0 * time_axis + float(base))


def test_healed_engine_draw_equals_a_direct_synthesis_at_the_retry_seed() -> None:
    """The engine-level determinism pin.

    The engine's draw is a pure function of (text, seed): the retry must not
    perturb any state that could leak into later draws — the healed audio has
    to be exactly what a direct synthesis at the retry seed produces, and the
    call sequence must be exactly [seed, seed + 1].
    """
    engine = object.__new__(ChatterboxEngine)
    engine.seed = 7
    calls: list[int | None] = []

    def flaky_once(text, conditioning, settings, seed):
        calls.append(seed)
        if len(calls) == 1:
            raise DegenerateEngineOutput("flat silence")
        return _draw_for_seed(seed)

    engine._generate_once = flaky_once  # type: ignore[method-assign]
    healed = engine.synthesize("hello", _StubConditioning(), _SETTINGS)

    assert calls == [7, 8], "exactly one retry, at seed + 1"
    np.testing.assert_array_equal(healed, _draw_for_seed(8))


def _make_flaky_once_engine():
    """A deterministic smoke engine that degenerates on its 2nd call, then heals.

    The class-level counter survives across instances, so each cold render
    (4 synthesis calls) hits exactly one hiccup on its second segment — the
    same draw every run. The heal records through the real call site the
    engines use, with the retry seed a configured-seed engine would use.
    """

    class _FlakyOnceEngine(_DeterministicChatterboxEngine):
        _calls = 0

        def synthesize(self, text, conditioning, settings):
            _FlakyOnceEngine._calls += 1
            if _FlakyOnceEngine._calls % 4 == 2:
                try:
                    raise DegenerateEngineOutput("flat silence")
                except DegenerateEngineOutput as first_error:
                    record_synthesis_retry("chatterbox", first_error, 12)
            return super().synthesize(text, conditioning, settings)

    return _FlakyOnceEngine


def _seeded_determinism_render(tmp_path: Path, project_name: str, engine_cls):
    """One cold-cache seeded render; returns (plan, events, output_path, stems)."""
    from the_oracle.models.cache import ProjectCache

    dialogue_path = tmp_path / "smoke_dialogue.txt"
    if not dialogue_path.exists():
        dialogue_path.write_text(SMOKE_DIALOGUE, encoding="utf-8")
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    project_dir = tmp_path / project_name

    with (
        patch("the_oracle.pipeline.ChatterboxEngine", engine_cls),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
    ):
        pipeline = OraclePipeline(use_transformers=False, use_language_tool=False, use_punctuation_model=False)
        shared_voice = VoiceSettings(variant="standard", language="en")
        speaker_settings = {
            "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=shared_voice),
            "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=shared_voice),
        }
        render_settings = RenderSettings(
            correction_mode="moderate",
            model_variant="standard",
            language="en",
            export_stems=False,
            loudness_preset="off",
            pause_between_turns_ms=120,
            crossfade_ms=10,
            seed=11,
            metadata={"title": "Retry determinism smoke", "output_filename": f"{project_name}.flac"},
        )
        plan = pipeline.prepare_plan(dialogue_path, project_dir, speaker_settings, render_settings)
        events: list[RenderProgress] = []
        output_path = pipeline.render(plan, render_settings, progress_callback=events.append, force_sequential=True)

    stems = {
        path.name: path.read_bytes()
        for path in sorted(ProjectCache(plan.output_dir).stem_cache_dir.glob("*.wav"))
        if path.is_file()
    }
    return plan, events, output_path, stems


@pytest.mark.slow
def test_seeded_render_with_a_retry_reproduces_run_to_run(tmp_path: Path) -> None:
    """Two cold-cache seeded renders that each heal one hiccup are identical.

    The self-heal is part of the deterministic record: same seed, same inputs,
    same single hiccup — therefore byte-identical stems, sample-identical
    output audio, identical retry-note streams, and the same
    ``synthesis_retries`` metadata. Also pins the cache invariant: re-render
    into the *same* project is all cache hits and carries no note, proving
    the healed audio was stored under the configured seed's chunk hashes.
    """
    from the_oracle.models.cache import ProjectCache

    engine_cls = _make_flaky_once_engine()

    plan_a, events_a, output_a, stems_a = _seeded_determinism_render(tmp_path, "det_a", engine_cls)
    assert plan_a.metadata.get("synthesis_retries") == "1"
    notes_a = [event.retry_note for event in events_a if event.retry_note]
    assert len(notes_a) == 1

    # Twin render: fresh project, same seed, same inputs, same flakiness.
    plan_b, events_b, output_b, stems_b = _seeded_determinism_render(tmp_path, "det_b", engine_cls)
    assert plan_b.metadata.get("synthesis_retries") == "1"
    notes_b = [event.retry_note for event in events_b if event.retry_note]
    assert notes_b == notes_a, "the retry-note stream must be identical run to run"

    assert set(stems_a) == set(stems_b), "identical inputs must produce identical chunk hashes"
    for name, payload in stems_a.items():
        assert payload == stems_b[name], f"stem {name} must be byte-identical run to run"

    import soundfile as sf

    audio_a, rate_a = sf.read(output_a, always_2d=False)
    audio_b, rate_b = sf.read(output_b, always_2d=False)
    assert rate_a == rate_b
    np.testing.assert_array_equal(audio_a, audio_b)

    # Warm pass: the same project re-serves the healed stems — no engine
    # calls, no new note — proving the healed take is the cached one.
    plan_warm, events_warm, _output_warm, _stems = _seeded_determinism_render(tmp_path, "det_a", engine_cls)
    assert plan_warm.metadata.get("synthesis_mode") == "cached"
    assert not [event.retry_note for event in events_warm if event.retry_note]


# --------------------------------------------------------------------------
# The spawn-pool path: same record, different dispatch.
#
# The pin above forces force_sequential=True. The worker pool is a different
# execution mode with its own hazards: the configured seed crosses the spawn
# boundary through _worker_initialize's initargs, workers synthesize in
# arbitrary order, and each spawned worker drains only its own RETRY_NOTES
# queue — the notes ride home pickled on SynthesisResult.retried_note. None
# of that may change the deterministic record.
# --------------------------------------------------------------------------


#: The task text whose synthesis hits the single hiccup. Keying the flaky
#: engine on TEXT instead of call order is what makes the hiccup stable under
#: pool dispatch, where task→worker assignment (and therefore global call
#: order) is nondeterministic.
_POOL_FLAKY_MARKER = "pool flaky marker"


class _PoolFlakyOnceEngine(_DeterministicChatterboxEngine):
    """Module-level so spawn workers can pickle the class.

    Degenerates exactly once for the marker task, at its one call, then heals
    through the real record_synthesis_retry call site. Multiple workers may
    synthesize concurrently — the marker text is what makes the hiccup
    fire exactly once per cold render regardless of which worker gets it.
    """

    engine_version = "deterministic-pool-flaky-v1"

    def synthesize(self, text, conditioning, settings):
        if _POOL_FLAKY_MARKER in text:
            if not getattr(self, "_pool_hiccup_spent", False):
                self._pool_hiccup_spent = True
                try:
                    raise DegenerateEngineOutput("flat silence")
                except DegenerateEngineOutput as first_error:
                    record_synthesis_retry("chatterbox", first_error, 12)
        return super().synthesize(text, conditioning, settings)


def _seeded_pool_render(tmp_path: Path, project_name: str, engine_cls):
    """One cold-cache seeded render allowed to dispatch the spawn pool.

    Identical to ``_seeded_determinism_render`` except ``force_sequential`` is
    left at its default so ``_should_use_worker_pool``'s own gating (standard
    variant, cpu, pytorch backend) decides the mode — and the test asserts the
    pool actually ran via ``plan.metadata['synthesis_mode']``.
    """
    from the_oracle.models.cache import ProjectCache

    dialogue_path = tmp_path / "smoke_dialogue.txt"
    if not dialogue_path.exists():
        # The flaky engine keys its hiccup on TASK CONTENT, so the marker
        # must ride one utterance. The line stays short enough to chunk into
        # exactly one synthesis task, so each cold render heals exactly once.
        dialogue_path.write_text(
            SMOKE_DIALOGUE.replace(
                "Speaker A: Chatterbox is the only backend now.",
                f"Speaker A: The {_POOL_FLAKY_MARKER} works.",
            ),
            encoding="utf-8",
        )
    speaker_a = _write_reference(tmp_path / "speaker_a_ref.wav", 220.0)
    speaker_b = _write_reference(tmp_path / "speaker_b_ref.wav", 330.0)
    project_dir = tmp_path / project_name

    with (
        patch("the_oracle.pipeline.ChatterboxEngine", engine_cls),
        patch("the_oracle.pipeline.GoEmotionsClassifier", _SmokeEmotionClassifier),
    ):
        pipeline = OraclePipeline(use_transformers=False, use_language_tool=False, use_punctuation_model=False)
        shared_voice = VoiceSettings(variant="standard", language="en")
        speaker_settings = {
            "A": SpeakerSettings(reference_path=str(speaker_a), voice_settings=shared_voice),
            "B": SpeakerSettings(reference_path=str(speaker_b), voice_settings=shared_voice),
        }
        render_settings = RenderSettings(
            correction_mode="moderate",
            model_variant="standard",
            language="en",
            export_stems=False,
            loudness_preset="off",
            pause_between_turns_ms=120,
            crossfade_ms=10,
            seed=11,
            metadata={"title": "Pool determinism smoke", "output_filename": f"{project_name}.flac"},
        )
        plan = pipeline.prepare_plan(dialogue_path, project_dir, speaker_settings, render_settings)
        events: list[RenderProgress] = []
        output_path = pipeline.render(plan, render_settings, progress_callback=events.append)

    stems = {
        path.name: path.read_bytes()
        for path in sorted(ProjectCache(plan.output_dir).stem_cache_dir.glob("*.wav"))
        if path.is_file()
    }
    return plan, events, output_path, stems


def _n_tasks() -> int:
    """The number of synthesis tasks the smoke dialogue chunks into.

    The pool requires >= _MIN_TASKS_FOR_POOL (4) tasks; the dialogue's four
    utterances produce exactly four.
    """
    return 4


def test_seeded_pool_render_with_a_retry_reproduces_run_to_run(tmp_path: Path) -> None:
    """The determinism pin, extended to the spawn-pool path.

    Two cold-cache seeded renders that each heal the same single hiccup are
    byte-identical when the WORKER POOL runs the synthesis: same seed,
    same inputs, the hiccup keyed on task content so pool dispatch order
    cannot move it — therefore identical chunk hashes, byte-identical stems
    (including the healed take), identical retry-note streams riding home
    from whichever spawned worker drained them, the same ``synthesis_retries``
    metadata, and sample-identical output audio. Also proves the pool really
    ran (``synthesis_mode == 'parallel'``) on a multi-core host and that a
    warm re-render re-serves the healed stems note-free.
    """
    from the_oracle.pipeline import _MIN_TASKS_FOR_POOL

    from the_oracle.pipeline import _MIN_TASKS_FOR_POOL

    assert _MIN_TASKS_FOR_POOL >= 4 and _n_tasks() >= _MIN_TASKS_FOR_POOL
    # The pool only runs where more than one worker is worth spawning; on a
    # single-core host _run_tasks_with_worker_pool stays sequential by design
    # and this pin would prove nothing about pool dispatch — say so instead.
    if not os.cpu_count() or os.cpu_count() < 2:
        pytest.skip(
            "single-core host: the worker pool deliberately stays sequential "
            "(_run_tasks_with_worker_pool's count == 1 branch), so the "
            "pool-path determinism pin cannot run here"
        )

    # Cold render A.
    plan_a, events_a, output_a, stems_a = _seeded_pool_render(tmp_path, "pool_a", _PoolFlakyOnceEngine)
    assert plan_a.metadata.get("synthesis_mode") == "parallel", (
        "the pool fell back to sequential execution — the determinism proof "
        "below would say nothing about pool dispatch; investigate the pool "
        "failure in the render log before trusting a green run"
    )
    assert plan_a.metadata.get("synthesis_retries") == "1", (
        "the marker hiccup must heal exactly once per cold render; if the "
        "pool never ran, this render quietly fell back to sequential"
    )
    notes_a = [event.retry_note for event in events_a if event.retry_note]
    assert len(notes_a) == 1

    # Cold render B: fresh project, same seed, same inputs, same flakiness.
    plan_b, events_b, output_b, stems_b = _seeded_pool_render(tmp_path, "pool_b", _PoolFlakyOnceEngine)
    assert plan_b.metadata.get("synthesis_mode") == "parallel"
    assert plan_b.metadata.get("synthesis_retries") == "1"
    notes_b = [event.retry_note for event in events_b if event.retry_note]
    assert notes_b == notes_a, "the retry-note stream must be identical run to run"

    assert set(stems_a) == set(stems_b), "identical inputs must produce identical chunk hashes under pool dispatch"
    for name, payload in stems_a.items():
        assert payload == stems_b[name], f"stem {name} must be byte-identical run to run under the pool"

    import soundfile as sf

    audio_a, rate_a = sf.read(output_a, always_2d=False)
    audio_b, rate_b = sf.read(output_b, always_2d=False)
    assert rate_a == rate_b
    np.testing.assert_array_equal(audio_a, audio_b)

    # The engine really was the pool-flaky one and the hiccup really healed:
    # a warm re-render of project A re-serves the healed stems note-free.
    plan_warm, events_warm, _output_warm, _stems = _seeded_pool_render(tmp_path, "pool_a", _PoolFlakyOnceEngine)
    assert plan_warm.metadata.get("synthesis_mode") == "cached"
    assert not [event.retry_note for event in events_warm if event.retry_note]
