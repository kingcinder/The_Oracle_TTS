"""Tests for the opt-in Vulkan backend (audio.cpp), all deterministic and offline.

The only hardware-touching test (``test_vulkan_backend_smoke_requires_device``)
skips gracefully when no Vulkan device is visible, matching the repo's
GPU-dependent test convention.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from the_oracle.models.cache import CachedReference, ProjectCache
from the_oracle.models.project import VoiceSettings
from the_oracle.pipeline import (
    RenderSettings,
    SynthesisTask,
    _chunk_engine_key,
    _should_use_worker_pool,
    synthesize_task,
    synthesize_tasks_batched,
)
from the_oracle.utils.hashing import build_chunk_hash, hash_file
import the_oracle.tts_engines.vulkan_backend as vulkan_backend
from the_oracle.tts_engines.vulkan_backend import (
    AudioCppUnavailableError,
    AudioCppVulkanEngine,
    RDNA1VulkanError,
    SUPPORTED_BACKENDS,
    VulkanConditioning,
    _vulkan_batch_max_requests,
    vulkan_device_available,
)


def _fake_completed(returncode: int, stderr: str = "", stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _engine_with_fake_binary(tmp_path: Path, **kwargs) -> AudioCppVulkanEngine:
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    model = tmp_path / "chatterbox-ggml"
    model.write_text("model", encoding="utf-8")
    return AudioCppVulkanEngine(binary=binary, model_path=model, **kwargs)


def test_supported_backends_contract() -> None:
    assert SUPPORTED_BACKENDS == ("pytorch", "vulkan")


def test_engine_contract_shape() -> None:
    engine = AudioCppVulkanEngine()
    assert engine.engine_id == "vulkan"
    assert engine.engine_version == "audio.cpp (ggml vulkan) v2"
    assert engine.sample_rate == 24000
    assert engine.device == "vulkan"


def test_turbo_variant_rejected_on_vulkan_backend() -> None:
    with pytest.raises(ValueError, match="turbo"):
        AudioCppVulkanEngine(variant="turbo")


def test_engine_rejects_invalid_batch_limit() -> None:
    """The engine-level batch cap is a hard contract: a constructor
    batch_limit < 1 would make every non-empty batch fail, so it is rejected
    up front for direct callers (RenderSettings/CLI validate too, but the
    engine is the last line of defense)."""
    with pytest.raises(ValueError, match="batch_limit") as excinfo:
        AudioCppVulkanEngine(batch_limit=0)
    assert "positive request count" in str(excinfo.value)
    with pytest.raises(ValueError, match="batch_limit"):
        AudioCppVulkanEngine(batch_limit=-3)
    # Non-int input (e.g. an unparsed config string from a duck-typed caller)
    # is rejected as ValueError, not left to raise TypeError on comparison.
    with pytest.raises(ValueError, match="batch_limit"):
        AudioCppVulkanEngine(batch_limit="8")
    # Valid values parse fine.
    assert AudioCppVulkanEngine(batch_limit=1).batch_limit == 1
    assert AudioCppVulkanEngine(batch_limit=32).batch_limit == 32


def test_render_settings_defaults_to_pytorch_and_validates() -> None:
    assert RenderSettings().inference_backend == "pytorch"
    with pytest.raises(ValueError, match="Unsupported inference backend"):
        RenderSettings(inference_backend="cuda")


def test_render_settings_audio_cpp_knobs_default_and_validate() -> None:
    assert RenderSettings().audio_cpp_device is None
    assert RenderSettings().audio_cpp_threads is None
    assert RenderSettings().audio_cpp_timeout is None
    assert RenderSettings(audio_cpp_device=0).audio_cpp_device == 0
    assert RenderSettings(audio_cpp_threads=4).audio_cpp_threads == 4
    assert RenderSettings(audio_cpp_timeout=300).audio_cpp_timeout == 300
    with pytest.raises(ValueError, match="audio_cpp_device"):
        RenderSettings(audio_cpp_device=-1)
    with pytest.raises(ValueError, match="audio_cpp_threads"):
        RenderSettings(audio_cpp_threads=0)
    with pytest.raises(ValueError, match="audio_cpp_timeout"):
        RenderSettings(audio_cpp_timeout=0)


def test_chunk_engine_key_keeps_pytorch_cache_stable() -> None:
    assert _chunk_engine_key("pytorch", "standard") == "chatterbox:standard"
    assert _chunk_engine_key("pytorch", "multilingual") == "chatterbox:multilingual"
    assert _chunk_engine_key("vulkan", "standard") == "vulkan:chatterbox:standard"
    assert _chunk_engine_key("vulkan", "standard") != _chunk_engine_key("pytorch", "standard")


def test_vulkan_backend_never_uses_worker_pool() -> None:
    settings = RenderSettings(model_variant="standard", device_mode="cpu", inference_backend="vulkan")
    assert _should_use_worker_pool(settings, "cpu") is False
    assert _should_use_worker_pool(RenderSettings(model_variant="standard", device_mode="cpu"), "cpu") is True


def test_gui_render_can_force_sequential_worker_execution() -> None:
    """The GUI-only transient flag bypasses native worker processes.

    Spawning native PyTorch/Perth workers from a live Qt application is the
    crash path seen as child status 245 on larger renders; the CLI remains
    eligible for the historical pool when the flag is absent.
    """
    settings = RenderSettings(model_variant="standard", device_mode="cpu")
    assert _should_use_worker_pool(settings, "cpu", force_sequential=True) is False
    assert _should_use_worker_pool(settings, "cpu") is True


def test_missing_binary_or_model_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hermetic: a dev machine with the repo-local build/model present would
    # otherwise route this through the fully-configured branch.
    monkeypatch.setattr(vulkan_backend, "find_audiocpp_binary", lambda: None)
    monkeypatch.setattr(vulkan_backend, "find_audiocpp_model", lambda: None)
    engine = AudioCppVulkanEngine()
    with pytest.raises(AudioCppUnavailableError, match="pytorch"):
        engine.ensure_model_ready()


def test_ensure_model_ready_flags_nonexistent_model_path(tmp_path: Path) -> None:
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    engine = AudioCppVulkanEngine(binary=binary, model_path=tmp_path / "missing-model")
    with pytest.raises(AudioCppUnavailableError, match="does not exist") as excinfo:
        engine.ensure_model_ready()
    # The recovery command is in the message, so fixing the model is one
    # copy-paste away instead of a README hunt.
    assert "download_audio_cpp_model.sh" in str(excinfo.value)
    assert "ORACLE_AUDIOCPP_MODEL" in str(excinfo.value)


def test_ensure_model_ready_unset_model_includes_recovery_command(monkeypatch, tmp_path: Path) -> None:
    # Hermetic: a dev machine with the env var or the repo-local model present
    # would otherwise route this through a different branch entirely.
    monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
    monkeypatch.setattr(vulkan_backend, "find_audiocpp_model", lambda: None)
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    engine = AudioCppVulkanEngine(binary=binary)
    with pytest.raises(AudioCppUnavailableError) as excinfo:
        engine.ensure_model_ready()
    assert "download_audio_cpp_model.sh" in str(excinfo.value)


def test_prepare_reference_rejects_missing_and_unreadable(tmp_path: Path) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    project_cache = ProjectCache(tmp_path / "project")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        engine.prepare_reference(project_cache, "A", str(tmp_path / "nope.wav"))

    garbage = tmp_path / "garbage.wav"
    garbage.write_bytes(b"not audio at all")
    with pytest.raises(ValueError, match="not readable audio"):
        engine.prepare_reference(project_cache, "A", str(garbage))


def test_prepare_reference_reports_real_sample_rate(tmp_path: Path) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref48k.wav"
    sf.write(reference, np.zeros(4800, dtype=np.float32), 48000)
    project_cache = ProjectCache(tmp_path / "project")

    cached = engine.prepare_reference(project_cache, "A", str(reference))
    assert cached.sample_rate == 48000
    assert cached.normalized_path == str(reference)


def test_prepare_reference_and_conditioning(tmp_path: Path) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    project_cache = ProjectCache(tmp_path / "project")

    cached = engine.prepare_reference(project_cache, "A", str(reference))
    conditioning = engine.prepare_conditioning(project_cache, "A", cached, VoiceSettings())

    assert cached.normalized_path == str(reference)
    assert cached.sample_rate == 24000
    assert isinstance(conditioning, VulkanConditioning)
    assert conditioning.speaker == "A"
    assert conditioning.reference_path == reference
    assert conditioning.cache_id


def test_synthesize_builds_command_and_loads_output(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(
        cache_id="c1",
        reference_path=reference,
        speaker="A",
    )

    recorded: dict[str, list[str]] = {}

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        recorded["command"] = command
        out = Path(command[command.index("--out") + 1])
        sf.write(out, np.linspace(-0.5, 0.5, 4800, dtype=np.float32), 24000)
        return _fake_completed(0)

    monkeypatch.setattr(engine, "_run_command", fake_run)

    audio = engine.synthesize("Hello from the Vulkan backend.", conditioning, VoiceSettings())

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.shape[0] == 4800
    assert engine.sample_rate == 24000

    command = recorded["command"]
    assert command[command.index("--task") + 1] == "clon"
    assert command[command.index("--family") + 1] == "chatterbox"
    assert command[command.index("--backend") + 1] == "vulkan"
    assert command[command.index("--voice-ref") + 1] == str(reference)
    assert Path(command[command.index("--out") + 1]).suffix == ".wav"


def _recorded_command(tmp_path: Path, monkeypatch, engine: AudioCppVulkanEngine | None = None) -> list[str]:
    """Run one synthesize through the engine and return the recorded argv."""
    engine = engine or _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")
    recorded: dict[str, list[str]] = {}

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        recorded["command"] = command
        out = Path(command[command.index("--out") + 1])
        sf.write(out, np.zeros(4800, dtype=np.float32), 24000)
        return _fake_completed(0)

    monkeypatch.setattr(engine, "_run_command", fake_run)
    engine.synthesize("Probe.", conditioning, VoiceSettings())
    return recorded["command"]


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_tuning_settings_forwarded_to_audio_cpp(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    settings = VoiceSettings(cfg_weight=0.7, temperature=0.9, repetition_penalty=1.4, min_p=0.08, top_p=0.95)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")
    recorded: dict[str, list[str]] = {}

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        recorded["command"] = command
        out = Path(command[command.index("--out") + 1])
        sf.write(out, np.zeros(4800, dtype=np.float32), 24000)
        return _fake_completed(0)

    monkeypatch.setattr(engine, "_run_command", fake_run)
    engine.synthesize("Probe.", conditioning, settings)
    command = recorded["command"]

    assert _flag_value(command, "--guidance-scale") == "0.7"
    assert _flag_value(command, "--temperature") == "0.9"
    assert _flag_value(command, "--top-p") == "0.95"
    assert _flag_value(command, "--repetition-penalty") == "1.4"
    assert "min_p=0.08" in command[command.index("--request-option") + 1]
    assert "--language" not in command


def test_language_forwarded_only_for_multilingual(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")
    recorded: dict[str, list[str]] = {}

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        recorded["command"] = command
        out = Path(command[command.index("--out") + 1])
        sf.write(out, np.zeros(4800, dtype=np.float32), 24000)
        return _fake_completed(0)

    monkeypatch.setattr(engine, "_run_command", fake_run)
    engine.synthesize(
        "Probe.",
        conditioning,
        VoiceSettings(variant="multilingual", language="fr"),
    )
    command = recorded["command"]
    assert _flag_value(command, "--language") == "fr"


def test_device_index_from_constructor_and_env(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    command = _recorded_command(tmp_path, monkeypatch, engine)
    assert "--device" not in command

    engine2 = _engine_with_fake_binary(tmp_path, device_index=1)
    command2 = _recorded_command(tmp_path, monkeypatch, engine2)
    assert _flag_value(command2, "--device") == "1"

    monkeypatch.setenv("ORACLE_AUDIOCPP_DEVICE", "0")
    engine3 = _engine_with_fake_binary(tmp_path)
    command3 = _recorded_command(tmp_path, monkeypatch, engine3)
    assert _flag_value(command3, "--device") == "0"


def test_threads_env_forwarded(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    command = _recorded_command(tmp_path, monkeypatch, engine)
    assert "--threads" not in command

    monkeypatch.setenv("ORACLE_AUDIOCPP_THREADS", "8")
    command2 = _recorded_command(tmp_path, monkeypatch, engine)
    assert _flag_value(command2, "--threads") == "8"


def test_constructor_device_index_wins_over_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ORACLE_AUDIOCPP_DEVICE", "0")
    engine = _engine_with_fake_binary(tmp_path, device_index=1)
    command = _recorded_command(tmp_path, monkeypatch, engine)
    assert _flag_value(command, "--device") == "1"


def test_constructor_threads_forwarded_and_wins_over_env(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_with_fake_binary(tmp_path, threads=4)
    command = _recorded_command(tmp_path, monkeypatch, engine)
    assert _flag_value(command, "--threads") == "4"

    monkeypatch.setenv("ORACLE_AUDIOCPP_THREADS", "8")
    engine2 = _engine_with_fake_binary(tmp_path, threads=4)
    command2 = _recorded_command(tmp_path, monkeypatch, engine2)
    assert _flag_value(command2, "--threads") == "4"


def test_constructor_timeout_wins_over_env_and_feeds_run_command(tmp_path: Path, monkeypatch) -> None:
    """The timeout constructor arg (fed by the GUI spin box / CLI flag) wins
    over ORACLE_AUDIOCPP_TIMEOUT and is what _run_command actually uses."""
    engine = _engine_with_fake_binary(tmp_path, timeout=90)
    assert engine.timeout == 90

    monkeypatch.setenv("ORACLE_AUDIOCPP_TIMEOUT", "30")
    assert engine.timeout == 90  # constructor wins

    # The env var is honored when no constructor override is given.
    engine2 = _engine_with_fake_binary(tmp_path)
    assert engine2.timeout == 30

    # Default without env override is 600.
    monkeypatch.delenv("ORACLE_AUDIOCPP_TIMEOUT", raising=False)
    engine3 = _engine_with_fake_binary(tmp_path)
    assert engine3.timeout == 600

    # _run_command passes self.timeout to subprocess.run.
    recorded: dict[str, float] = {}

    def recording_run(command, **kwargs):
        recorded["timeout"] = kwargs.get("timeout")
        return _fake_completed(0)

    monkeypatch.setattr("the_oracle.tts_engines.vulkan_backend.subprocess.run", recording_run)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")
    engine._run_command([str(engine.binary)])
    assert recorded["timeout"] == 90


def test_list_devices_parses_audio_cpp_output(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    sample = (
        "ggml_vulkan: Found 1 Vulkan devices:\n"
        'Vulkan:0 "AMD Radeon RX 5700 XT (RADV NAVI10)" [GPU]\n'
        'CPU:0 "AMD Ryzen 7 3700X 8-Core Processor" [CPU]\n'
    )
    monkeypatch.setattr(
        "the_oracle.tts_engines.vulkan_backend.subprocess.run",
        lambda *a, **k: _fake_completed(0, stdout=sample),
    )
    devices = engine.list_devices()
    assert devices == [{"index": 0, "name": "AMD Radeon RX 5700 XT (RADV NAVI10)"}]


def test_list_devices_raises_without_binary(tmp_path: Path, monkeypatch) -> None:
    # Do not depend on what is on disk (this repo may have the real built
    # binary at audio.cpp/build/...), so force binary discovery to fail.
    monkeypatch.setattr("the_oracle.tts_engines.vulkan_backend.find_audiocpp_binary", lambda: None)
    engine = AudioCppVulkanEngine()
    with pytest.raises(AudioCppUnavailableError, match="build_audio_cpp"):
        engine.list_devices()


def test_rdna1_device_lost_failure_is_visible(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")

    monkeypatch.setattr(
        engine,
        "_run_command",
        lambda command: _fake_completed(1, stderr="ggml_vulkan: VK_ERROR_DEVICE_LOST during buffer init"),
    )

    with pytest.raises(RDNA1VulkanError, match="inference_backend: pytorch"):
        engine.synthesize("Probe.", conditioning, VoiceSettings())


def test_generic_audio_cpp_failure_includes_stderr(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")

    monkeypatch.setattr(
        engine,
        "_run_command",
        lambda command: _fake_completed(7, stderr="model file not found: chatterbox-ggml"),
    )

    with pytest.raises(RuntimeError, match="model file not found"):
        engine.synthesize("Probe.", conditioning, VoiceSettings())


def test_synthesize_batch_empty_returns_empty(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    monkeypatch.setattr(engine, "_run_command", lambda command: pytest.fail("batch must not spawn for empty group"))
    assert engine.synthesize_batch([]) == []


def test_synthesize_batch_runs_one_process_for_many_utterances(tmp_path: Path, monkeypatch) -> None:
    """The whole point of batching: N utterances -> ONE audio.cpp process.

    The command must use --request-sequence + --out-dir (never per-utterance
    --text/--voice-ref/--out), the JSON must carry every request with its
    absolute voice_ref and tuning options, and outputs must come back in
    request order with audio.cpp's per-request wall_ms timing.
    """
    engine = _engine_with_fake_binary(tmp_path, device_index=1, threads=4)
    ref_a = tmp_path / "ref_a.wav"
    ref_b = tmp_path / "ref_b.wav"
    sf.write(ref_a, np.zeros(2400, dtype=np.float32), 24000)
    sf.write(ref_b, np.zeros(2400, dtype=np.float32), 24000)
    cond_a = VulkanConditioning(cache_id="c1", reference_path=ref_a, speaker="A")
    cond_b = VulkanConditioning(cache_id="c2", reference_path=ref_b, speaker="B")
    settings = VoiceSettings(cfg_weight=0.7, temperature=0.9, repetition_penalty=1.4, min_p=0.08, top_p=0.95)

    recorded: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        # The batch passes a scaled timeout (per-synthesis timeout x request
        # count); the fake must accept it. Computed from the engine so the
        # assertion is hermetic even when ORACLE_AUDIOCPP_TIMEOUT is exported.
        assert kwargs.get("timeout") == engine.timeout * 2
        recorded["command"] = command
        sequence_path = Path(command[command.index("--request-sequence") + 1])
        out_dir = Path(command[command.index("--out-dir") + 1])
        recorded["sequence"] = json.loads(sequence_path.read_text(encoding="utf-8"))
        for index in range(2):
            sf.write(out_dir / f"request_{index}.wav", np.linspace(-0.5, 0.5, 4800 * (index + 1), dtype=np.float32), 24000)
        return _fake_completed(
            0,
            stdout="[TIMING] request.request_0.wall_ms 250.5\n[TIMING] request.request_1.wall_ms 512.25\n",
        )

    monkeypatch.setattr(engine, "_run_command", fake_run)

    outputs = engine.synthesize_batch(
        [
            ("Hello from speaker A.", cond_a, settings),
            ("And from speaker B.", cond_b, settings),
        ]
    )

    command = recorded["command"]
    assert command[command.index("--task") + 1] == "clon"
    assert command[command.index("--family") + 1] == "chatterbox"
    assert command[command.index("--backend") + 1] == "vulkan"
    assert command[command.index("--mode") + 1] == "offline"
    assert command[command.index("--device") + 1] == "1"
    assert command[command.index("--threads") + 1] == "4"
    assert "--text" not in command, "batch must not pass per-utterance --text"
    assert "--voice-ref" not in command, "batch must not pass per-utterance --voice-ref"
    assert "--out" not in command, "batch must not pass per-utterance --out"

    requests = recorded["sequence"]["requests"]  # type: ignore[index]
    assert len(requests) == 2
    assert requests[0]["id"] == "request_0"
    assert requests[0]["text"] == "Hello from speaker A."
    assert requests[0]["voice_ref"] == str(ref_a.resolve())
    assert requests[0]["options"]["guidance_scale"] == "0.7"
    assert requests[0]["options"]["min_p"] == "0.08"
    assert requests[1]["id"] == "request_1"
    assert requests[1]["text"] == "And from speaker B."
    assert requests[1]["voice_ref"] == str(ref_b.resolve())
    assert requests[1]["options"]["temperature"] == "0.9"
    assert "language" not in requests[0]["options"]

    assert len(outputs) == 2
    assert outputs[0][0].shape[0] == 4800
    assert outputs[1][0].shape[0] == 9600
    assert outputs[0][1] == 24000
    assert outputs[0][2] == 250.5  # audio.cpp's own per-request timing
    assert outputs[1][2] == 512.25
    assert engine.sample_rate == 24000


def test_synthesize_batch_embeds_language_for_multilingual(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")
    recorded: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        sequence_path = Path(command[command.index("--request-sequence") + 1])
        recorded["sequence"] = json.loads(sequence_path.read_text(encoding="utf-8"))
        out_dir = Path(command[command.index("--out-dir") + 1])
        sf.write(out_dir / "request_0.wav", np.zeros(4800, dtype=np.float32), 24000)
        return _fake_completed(0)

    monkeypatch.setattr(engine, "_run_command", fake_run)
    engine.synthesize_batch(
        [("Bonjour.", conditioning, VoiceSettings(variant="multilingual", language="fr"))]
    )
    requests = recorded["sequence"]["requests"]  # type: ignore[index]
    assert requests[0]["options"]["language"] == "fr"


def test_synthesize_batch_missing_output_raises(tmp_path: Path, monkeypatch) -> None:
    """Exit 0 but no request_N.wav must fail loudly, never produce silence."""
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        out_dir = Path(command[command.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        return _fake_completed(0)  # writes nothing

    monkeypatch.setattr(engine, "_run_command", fake_run)
    with pytest.raises(RuntimeError, match="request_0"):
        engine.synthesize_batch([("Probe.", conditioning, VoiceSettings())])


class _BatchStubEngine:
    """Minimal AudioCppVulkanEngine stand-in for grouping tests.

    Implements just the duck-typed surface ``synthesize_tasks_batched``
    touches: ``engine_version``, ``prepare_reference``, ``prepare_conditioning``,
    and ``synthesize_batch`` (which records each call's size and fails on the
    ``fail_on``-th call so failure isolation can be asserted; pass
    ``fail_on=None`` for the always-succeed grouping/progress tests).
    """

    engine_version = "stub-batch-v1"

    def __init__(self, fail_on: int | None = 2) -> None:
        self.batch_sizes: list[int] = []
        self.fail_on = fail_on

    def prepare_reference(self, project_cache: ProjectCache, speaker: str, reference_path: str) -> CachedReference:
        return CachedReference(reference_path, reference_path, "refhash", 24000)

    def prepare_conditioning(
        self,
        project_cache: ProjectCache,
        speaker: str,
        cached_reference: CachedReference,
        settings: VoiceSettings,
    ) -> VulkanConditioning:
        return VulkanConditioning(cache_id="c1", reference_path=Path(cached_reference.original_path), speaker=speaker)

    def synthesize_batch(self, entries, on_request_complete=None):
        self.batch_sizes.append(len(entries))
        if self.fail_on is not None and len(self.batch_sizes) == self.fail_on:
            raise RuntimeError("second audio.cpp batch failed")
        if on_request_complete is not None:
            for index in range(len(entries)):
                on_request_complete(index)
        return [(np.zeros(4800, dtype=np.float32), 24000, 100.0) for _ in entries]


def test_synthesize_batch_rejects_over_cap_entries_from_constructor(tmp_path: Path, monkeypatch) -> None:
    """The engine's defense-in-depth cap honors the constructor batch_limit
    (not just the env var), so a direct caller can never build an unbounded
    requests.json even when the cap is set via settings."""
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "64")
    engine = _engine_with_fake_binary(tmp_path, batch_limit=2)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")

    with pytest.raises(ValueError, match="at most 2 requests per subprocess"):
        engine.synthesize_batch([("Probe.", conditioning, VoiceSettings()) for _ in range(3)])


def test_synthesize_batch_rejects_over_cap_entries(tmp_path: Path, monkeypatch) -> None:
    """Defense-in-depth: the engine itself refuses a group larger than
    ORACLE_AUDIOCPP_MAX_BATCH so a direct caller (e.g. a future server mode)
    can never build an unbounded requests.json -- even one that skips the
    pipeline's grouping in synthesize_tasks_batched."""
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "2")
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")

    # The guard must fire before any subprocess spawns or JSON is written.
    def _fail_spawn(*_args, **_kwargs):
        raise AssertionError("synthesize_batch must not spawn a subprocess for an over-cap group")

    monkeypatch.setattr(engine, "_run_command", _fail_spawn)
    monkeypatch.setattr(engine, "_run_batch_command_streaming", _fail_spawn)

    entries = [("Probe.", conditioning, VoiceSettings()) for _ in range(3)]
    with pytest.raises(ValueError, match="at most 2 requests per subprocess") as excinfo:
        engine.synthesize_batch(entries)
    assert "ORACLE_AUDIOCPP_MAX_BATCH" in str(excinfo.value)


def test_synthesize_batch_accepts_exactly_the_cap(tmp_path: Path, monkeypatch) -> None:
    """A group of exactly ORACLE_AUDIOCPP_MAX_BATCH entries is allowed (the
    boundary the pipeline's grouping sends per subprocess)."""
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "2")
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        out_dir = Path(command[command.index("--out-dir") + 1])
        for index in range(2):
            sf.write(out_dir / f"request_{index}.wav", np.zeros(2400, dtype=np.float32), 24000)
        return _fake_completed(0)

    monkeypatch.setattr(engine, "_run_command", fake_run)
    outputs = engine.synthesize_batch(
        [("Probe.", conditioning, VoiceSettings()) for _ in range(2)]
    )
    assert len(outputs) == 2


def test_engine_batch_limit_constructor_wins_over_env(monkeypatch) -> None:
    """The batch cap is threaded like device/threads/timeout: a constructor
    arg (fed by the CLI flag / GUI spin / manifest field) wins over
    ORACLE_AUDIOCPP_MAX_BATCH, and without either the default is 32."""
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "64")
    engine = AudioCppVulkanEngine(batch_limit=8)
    assert engine.batch_limit == 8

    engine2 = AudioCppVulkanEngine()
    assert engine2.batch_limit == 64  # env var honored when no ctor arg

    monkeypatch.delenv("ORACLE_AUDIOCPP_MAX_BATCH", raising=False)
    engine3 = AudioCppVulkanEngine()
    assert engine3.batch_limit == 32  # default


def test_vulkan_batch_max_requests_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv("ORACLE_AUDIOCPP_MAX_BATCH", raising=False)
    assert _vulkan_batch_max_requests() == 32
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "8")
    assert _vulkan_batch_max_requests() == 8
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "bogus")
    assert _vulkan_batch_max_requests() == 32  # malformed falls back
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "0")
    assert _vulkan_batch_max_requests() == 1  # clamped to >= 1


def test_synthesize_tasks_batched_groups_and_isolates_failures(tmp_path: Path, monkeypatch) -> None:
    """Cache-missing stems split into bounded groups (one subprocess each), and
    a failed group only takes down its own tasks -- the rest still render."""
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "2")
    project_cache = ProjectCache(tmp_path / "project")
    engine = _BatchStubEngine()
    tasks = [
        SynthesisTask(
            utterance_index=index,
            source_index=index,
            speaker="A",
            text=f"Distinct probe line number {index} with unique words for hashing.",
            reference_path=tmp_path / f"ref_{index}.wav",
            reference_audio_hash="refhash",
            voice_settings=VoiceSettings(),
            model_variant="standard",
            device_mode="cpu",
            export_stems=False,
            inference_backend="vulkan",
        )
        for index in range(1, 6)
    ]

    stats: dict[str, int] = {}
    results = synthesize_tasks_batched(tasks, engine, {}, project_cache, batch_stats=stats)

    # 5 cache misses split into 3 groups (2 + 2 + 1): one subprocess per group.
    assert engine.batch_sizes == [2, 2, 1]
    assert stats == {"processes": 3, "requests": 5}
    # Group 2 (tasks 3 and 4) failed; groups 1 and 3 still synthesized.
    succeeded = {result.utterance_index for result in results if result.error is None}
    failed = {result.utterance_index for result in results if result.error is not None}
    assert succeeded == {1, 2, 5}
    assert failed == {3, 4}
    # Stems were written only for the successful tasks.
    stem_names = {path.stem for path in project_cache.stem_cache_dir.glob("*.wav")}
    assert len(stem_names) == 3


def test_synthesize_tasks_batched_propagates_rdna1_failure(tmp_path: Path, monkeypatch) -> None:
    """The RDNA1 device-lost limitation (and missing binary/model) is
    deterministic: every remaining batch fails the same way, so
    synthesize_tasks_batched must propagate the actionable error instead of
    degrading it into a generic partial-failure row list."""

    class _RDNA1Engine(_BatchStubEngine):
        def synthesize_batch(self, entries, on_request_complete=None):
            self.batch_sizes.append(len(entries))
            raise RDNA1VulkanError(
                "RDNA1 device-lost limitation; re-run with --inference-backend pytorch"
            )

    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "2")
    project_cache = ProjectCache(tmp_path / "project")
    task = SynthesisTask(
        utterance_index=1,
        source_index=1,
        speaker="A",
        text="Probe line with unique words for hashing.",
        reference_path=tmp_path / "ref.wav",
        reference_audio_hash="refhash",
        voice_settings=VoiceSettings(),
        model_variant="standard",
        device_mode="cpu",
        export_stems=False,
        inference_backend="vulkan",
    )

    with pytest.raises(RDNA1VulkanError, match="inference-backend pytorch"):
        synthesize_tasks_batched([task], _RDNA1Engine(), {}, project_cache)


def test_synthesize_tasks_batched_forwards_request_completion(tmp_path: Path, monkeypatch) -> None:
    """The pipeline maps each group-local request index back to the task's
    utterance_index and forwards it to on_request_complete as it lands."""
    monkeypatch.setenv("ORACLE_AUDIOCPP_MAX_BATCH", "2")
    project_cache = ProjectCache(tmp_path / "project")
    engine = _BatchStubEngine(fail_on=None)
    tasks = [
        SynthesisTask(
            utterance_index=index,
            source_index=index,
            speaker="A",
            text=f"Distinct probe line number {index} with unique words for hashing.",
            reference_path=tmp_path / f"ref_{index}.wav",
            reference_audio_hash="refhash",
            voice_settings=VoiceSettings(),
            model_variant="standard",
            device_mode="cpu",
            export_stems=False,
            inference_backend="vulkan",
        )
        for index in range(1, 6)
    ]

    completed: list[int] = []
    synthesize_tasks_batched(tasks, engine, {}, project_cache, on_request_complete=completed.append)

    # 5 misses in groups [2, 2, 1]; the pipeline must report the real task
    # indices (1..5) in completion order, not the group-local 0..n indices.
    assert completed == [1, 2, 3, 4, 5]


def test_synthesize_batch_streams_progress_per_request(tmp_path: Path, monkeypatch) -> None:
    """With on_request_complete, the batch runs via Popen and reports each
    request as its wav lands in --out-dir (out-dir polling), not only at the
    end -- so a GUI can advance its bar during the render. Callbacks fire in
    request order while the process is still running."""
    engine = _engine_with_fake_binary(tmp_path)
    ref_a = tmp_path / "ref_a.wav"
    ref_b = tmp_path / "ref_b.wav"
    sf.write(ref_a, np.zeros(2400, dtype=np.float32), 24000)
    sf.write(ref_b, np.zeros(2400, dtype=np.float32), 24000)
    cond_a = VulkanConditioning(cache_id="c1", reference_path=ref_a, speaker="A")
    cond_b = VulkanConditioning(cache_id="c2", reference_path=ref_b, speaker="B")

    recorded: dict[str, object] = {}

    class _FakePopen:
        """Simulates audiocpp_cli: request_0.wav lands after the first poll
        cycle, request_1.wav lands just before exit."""

        def __init__(self, command, **kwargs):
            recorded["command"] = command
            self.returncode = None
            self._polls = 0
            self.stdout = iter([])
            self.stderr = iter([])
            self.out_dir = Path(command[command.index("--out-dir") + 1])
            self.killed = False

        def poll(self):
            self._polls += 1
            if self._polls == 1:
                return None
            if self._polls == 2:
                sf.write(self.out_dir / "request_0.wav", np.zeros(2400, dtype=np.float32), 24000)
                return None
            sf.write(self.out_dir / "request_1.wav", np.zeros(2400, dtype=np.float32), 24000)
            self.returncode = 0
            return 0

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def kill(self):
            self.killed = True
            self.returncode = 1

    monkeypatch.setattr("the_oracle.tts_engines.vulkan_backend.subprocess.Popen", _FakePopen)

    completed: list[int] = []
    outputs = engine.synthesize_batch(
        [
            ("Hello from speaker A.", cond_a, VoiceSettings()),
            ("And from speaker B.", cond_b, VoiceSettings()),
        ],
        on_request_complete=completed.append,
    )

    # Live progress: request 0 reported mid-run (poll 2), request 1 reported
    # by the final sweep as the process exits -- in request order.
    assert completed == [0, 1]
    assert len(outputs) == 2
    assert outputs[0][0].shape[0] == 2400
    assert outputs[1][0].shape[0] == 2400


def test_synthesize_batch_streaming_timeout_kills_child(tmp_path: Path, monkeypatch) -> None:
    """The streaming path honors the timeout: a child that never produces
    wavs is killed and the same RuntimeError the blocking path raises is
    surfaced (never a hang, never a silent swallow)."""
    engine = _engine_with_fake_binary(tmp_path, timeout=1)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")

    class _HangingPopen:
        def __init__(self, command, **kwargs):
            self.returncode = None
            self.stdout = iter([])
            self.stderr = iter([])
            self.killed = False

        def poll(self):
            return None  # never finishes

        def wait(self, timeout=None):
            self.returncode = 1
            return 1

        def kill(self):
            self.killed = True
            self.returncode = 1

    monkeypatch.setattr("the_oracle.tts_engines.vulkan_backend.subprocess.Popen", _HangingPopen)

    with pytest.raises(RuntimeError, match="timed out after"):
        engine.synthesize_batch(
            [("Probe.", conditioning, VoiceSettings())],
            on_request_complete=lambda _index: None,
        )


def test_synthesize_batch_failure_is_visible_and_rdna1_detected(tmp_path: Path, monkeypatch) -> None:
    """A failing batch subprocess surfaces the same RDNA1/generic failure
    paths as single synthesis -- never silently retried or hidden."""
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    conditioning = VulkanConditioning(cache_id="c1", reference_path=reference, speaker="A")

    monkeypatch.setattr(
        engine,
        "_run_command",
        lambda command, **kwargs: _fake_completed(
            1, stderr="ggml_vulkan: VK_ERROR_DEVICE_LOST during buffer init"
        ),
    )
    with pytest.raises(RDNA1VulkanError, match="inference_backend: pytorch"):
        engine.synthesize_batch([("Probe.", conditioning, VoiceSettings())])

    monkeypatch.setattr(
        engine,
        "_run_command",
        lambda command, **kwargs: _fake_completed(7, stderr="model file not found: chatterbox-ggml"),
    )
    with pytest.raises(RuntimeError, match="model file not found"):
        engine.synthesize_batch([("Probe.", conditioning, VoiceSettings())])

    """Regression: vulkan stems must hash into the vulkan: namespace so the two
    backends never collide in the shared stem cache (the chunked-path bug)."""
    engine = _engine_with_fake_binary(tmp_path)
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
    project_cache = ProjectCache(tmp_path / "project")

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        out = Path(command[command.index("--out") + 1])
        sf.write(out, np.zeros(9600, dtype=np.float32), 24000)
        return _fake_completed(0)

    monkeypatch.setattr(engine, "_run_command", fake_run)

    voice_settings = VoiceSettings()
    task = SynthesisTask(
        utterance_index=1,
        source_index=1,
        speaker="A",
        text="A long line that may be chunked.",
        reference_path=reference,
        reference_audio_hash=hash_file(reference),
        voice_settings=voice_settings,
        model_variant="standard",
        device_mode="cpu",
        export_stems=False,
        inference_backend="vulkan",
    )
    cached = engine.prepare_reference(project_cache, "A", str(reference))
    conditioning = engine.prepare_conditioning(project_cache, "A", cached, voice_settings)

    result = synthesize_task(task, engine, conditioning, project_cache)

    expected = build_chunk_hash(
        speaker=task.speaker,
        repaired_text=task.text,
        engine_key="vulkan:chatterbox:standard",
        engine_params=task.voice_settings.to_dict(),
        engine_version=engine.engine_version,
        reference_audio_hash=task.reference_audio_hash,
    )
    pytorch_expected = build_chunk_hash(
        speaker=task.speaker,
        repaired_text=task.text,
        engine_key="chatterbox:standard",
        engine_params=task.voice_settings.to_dict(),
        engine_version=engine.engine_version,
        reference_audio_hash=task.reference_audio_hash,
    )
    assert result.chunk_hash == expected
    assert result.chunk_hash != pytorch_expected
    assert result.stem_path.name == f"{expected}.wav"
    # The stem must live ONLY in the vulkan namespace: nothing was written
    # under the pytorch key, so the two backends cannot collide in the cache.
    assert not project_cache.stem_path(pytorch_expected).exists()


def test_vulkan_backend_smoke_requires_device() -> None:
    """Hardware-dependent smoke: skip gracefully when no Vulkan device exists."""
    if not vulkan_device_available():
        pytest.skip("No Vulkan device available; Vulkan backend smoke test requires a Vulkan-capable GPU.")

    engine = AudioCppVulkanEngine()
    if engine.binary is None:
        pytest.skip("audiocpp_cli is not built; run scripts/build_audio_cpp.sh first.")
    if engine.model is None:
        pytest.skip("Chatterbox ggml model is not configured; set ORACLE_AUDIOCPP_MODEL.")

    # Reaching here means a Vulkan device exists and the CLI + model are
    # configured. Exercise the deterministic preflight only -- real synthesis
    # needs the ggml model present and is covered by the offline mocks above.
    engine.ensure_model_ready()
