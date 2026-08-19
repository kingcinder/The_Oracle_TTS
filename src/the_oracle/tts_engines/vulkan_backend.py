"""Opt-in Vulkan-backed Chatterbox inference via audio.cpp (ggml).

This backend runs the Chatterbox model family through ``audiocpp_cli`` from
audio.cpp (github.com/0xShug0/audio.cpp), a ggml-based C++ inference engine
built with ``-DENGINE_ENABLE_VULKAN=ON``. It is selected with
``inference_backend: vulkan`` (CLI: ``--inference-backend vulkan``) and does
not touch the default PyTorch/Chatterbox path.

Rationale: AMD RDNA1 GPUs (e.g. RX 5700 XT, gfx1010) have no CUDA and no
official ROCm support, and PyTorch's Vulkan backend is experimental and
mobile-only. audio.cpp's ggml Vulkan backend is the only realistic GPU
acceleration route for Chatterbox-class models on this hardware.

The engine implements the same duck-typed contract the pipeline already uses
with :class:`ChatterboxEngine`:

- ``prepare_reference(...) -> CachedReference``
- ``prepare_conditioning(...) -> conditioning`` (a :class:`VulkanConditioning`)
- ``synthesize(text, conditioning, settings) -> np.ndarray[float32]``

audio.cpp does not need a pre-normalized reference clip (it reads the wav
itself), so ``prepare_reference`` returns the original path and the CLI is
invoked with ``--voice-ref <original.wav>``.

The speaker's tuning settings are forwarded to audio.cpp's chatterbox session
as ``--guidance-scale`` (cfg_weight), ``--temperature``, ``--top-p``,
``--repetition-penalty``, and ``min_p`` via ``--request-option``; multilingual
runs pass ``--language``. ``--device <n>`` is selected by the constructor's
``device_index`` or ``ORACLE_AUDIOCPP_DEVICE``, and ``--threads`` by the
constructor's ``threads`` or ``ORACLE_AUDIOCPP_THREADS`` (constructor args
win). The CLI and project manifests feed those constructor args, so the
knobs work without environment variables.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from the_oracle.models.cache import CachedReference, ProjectCache
from the_oracle.models.project import VoiceSettings
from the_oracle.utils.audio import ensure_mono
from the_oracle.utils.hashing import hash_file, hash_payload

SUPPORTED_BACKENDS = ("pytorch", "vulkan")

# Chatterbox 0.5B native sample rate. Used as the engine's reported rate until
# the first real audio.cpp output is synthesized (its actual rate then takes
# over via _last_sample_rate), so the downstream FLAC pipeline is always right.
DEFAULT_SAMPLE_RATE = 24000

def _synthesis_timeout_seconds() -> float:
    # Parsed lazily so a malformed ORACLE_AUDIOCPP_TIMEOUT env value cannot
    # crash the module at import time.
    try:
        return float(os.environ.get("ORACLE_AUDIOCPP_TIMEOUT", "600"))
    except ValueError:
        return 600.0

_RDNA1_DEVICE_LOST_MARKERS = (
    "VK_ERROR_DEVICE_LOST",
    "ErrorDeviceLost",
    "DeviceLost",
    "device lost",
)


def _timeout_error_message(timeout: float) -> str:
    """The timeout RuntimeError text, shared by the blocking and streaming paths."""
    return (
        "audio.cpp Vulkan synthesis timed out after "
        f"{timeout:g}s (a batch of N utterances scales this to "
        "N x the per-synthesis timeout; raise it with "
        "ORACLE_AUDIOCPP_TIMEOUT / --audio-cpp-timeout). Re-run "
        "with inference_backend=pytorch to use the default path."
    )


class AudioCppUnavailableError(RuntimeError):
    """Raised when the audiocpp_cli binary or the Chatterbox ggml model is missing."""


class RDNA1VulkanError(RuntimeError):
    """Raised when audio.cpp's Vulkan backend hits the documented RDNA1 device-lost failure.

    This is a hardware/driver limitation (RDNA1's SDMA transfer queue rejects
    ``vkCmdFillBuffer`` inside ``ggml_vk_buffer_memset`` -- see
    ggml-org/whisper.cpp#3611), not a bug in The Oracle, so it must be surfaced
    visibly with a clear fallback rather than retried silently.
    """


@dataclass(slots=True)
class VulkanConditioning:
    """Conditioning payload for one audio.cpp Vulkan clone call.

    audio.cpp has no separate conditioning step: voice cloning is performed per
    call by passing ``--voice-ref``. This object just carries the original
    reference wav so ``synthesize`` can build the CLI invocation.
    """

    cache_id: str
    reference_path: Path
    speaker: str


def vulkan_device_available() -> bool:
    """Return True when a Vulkan device is visible to the system.

    Used by tests and diagnostics to skip gracefully on machines without a
    Vulkan device. ``ORACLE_AUDIOCPP_FORCE_VULKAN=1`` overrides the probe.
    """
    if os.environ.get("ORACLE_AUDIOCPP_FORCE_VULKAN") == "1":
        return True
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo is None:
        return False
    try:
        completed = subprocess.run(
            [vulkaninfo, "--summary"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def find_audiocpp_binary() -> Path | None:
    """Locate ``audiocpp_cli``: env override, repo-local build, then PATH."""
    override = os.environ.get("ORACLE_AUDIOCPP_CLI")
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists():
            return candidate
    root = _repo_root()
    for candidate in (
        root / "audio.cpp" / "build" / "linux-vulkan-release" / "bin" / "audiocpp_cli",
        root / "audio.cpp" / "build" / "bin" / "audiocpp_cli",
    ):
        if candidate.exists():
            return candidate
    found = shutil.which("audiocpp_cli")
    return Path(found) if found else None


def find_audiocpp_model() -> Path | None:
    """Locate the Chatterbox ggml model: env override, then the repo-local
    install ``scripts/download_audio_cpp_model.sh`` writes to.

    Mirrors ``find_audiocpp_binary`` so a model downloaded with the official
    script is picked up automatically -- no ORACLE_AUDIOCPP_MODEL export
    required. ``AUDIOCPP_MODELS_ROOT`` (the download script's install-root
    knob) is honored before the repo default.
    """
    override = os.environ.get("ORACLE_AUDIOCPP_MODEL")
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists():
            return candidate
    root = _repo_root()
    folders: list[Path] = []
    models_root = os.environ.get("AUDIOCPP_MODELS_ROOT")
    if models_root:
        folders.append(Path(models_root).expanduser() / "Chatterbox-GGUF")
    folders.append(root / "audio.cpp" / "models" / "Chatterbox-GGUF")
    for folder in folders:
        for name in ("chatterbox-q8_0.gguf", "chatterbox-f16.gguf"):
            candidate = folder / name
            if candidate.exists():
                return candidate
    return None


def _model_from_env() -> str | None:
    return os.environ.get("ORACLE_AUDIOCPP_MODEL")


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# Maximum cache-missing stems per audio.cpp --request-sequence subprocess. The
# pipeline splits renders into groups of this size, and the engine enforces the
# SAME cap as defense-in-depth so a direct caller (e.g. a future server mode)
# can never build an unbounded requests.json. Override with
# ORACLE_AUDIOCPP_MAX_BATCH. This is the single source of truth; the pipeline
# imports it from here rather than redefining it.
_VULKAN_BATCH_MAX_REQUESTS: int = 32


def _vulkan_batch_max_requests() -> int:
    """The Vulkan batch-size cap: ``ORACLE_AUDIOCPP_MAX_BATCH`` or the default."""
    raw = os.environ.get("ORACLE_AUDIOCPP_MAX_BATCH")
    if raw is None:
        return _VULKAN_BATCH_MAX_REQUESTS
    try:
        return max(1, int(raw))
    except ValueError:
        return _VULKAN_BATCH_MAX_REQUESTS


_DEVICE_LINE = re.compile(r'^Vulkan:(\d+)\s+"([^"]+)"')
# audio.cpp's batch sink prints one timing line per request:
#   [TIMING] request.request_0.wall_ms 1234.5
_BATCH_TIMING_LINE = re.compile(r'\[TIMING\] request\.request_(\d+)\.wall_ms\s+([\d.]+)')

class AudioCppVulkanEngine:
    """Audio.cpp-backed Chatterbox clone engine running on the Vulkan backend."""

    engine_id = "vulkan"

    def __init__(
        self,
        variant: str = "standard",
        device: str | None = None,
        *,
        binary: str | Path | None = None,
        model_path: str | Path | None = None,
        device_index: int | None = None,
        threads: int | None = None,
        timeout: int | None = None,
        batch_limit: int | None = None,
        seed: int | None = None,
    ) -> None:
        if variant == "turbo":
            raise ValueError(
                "The turbo Chatterbox variant is not available on the Vulkan backend. "
                "Use inference_backend=pytorch for turbo, or standard/multilingual on Vulkan."
            )
        if seed is not None and seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
        # The engine-level guard treats batch_limit as a hard contract, so a
        # direct caller (e.g. a future server mode) cannot pass a value that
        # would make every non-empty batch fail (batch_limit < 1 means even a
        # single request is over-cap). RenderSettings/CLI also validate, but
        # the engine is the last line of defense for duck-typed callers, so a
        # non-int (e.g. an unparsed config string) is rejected as ValueError
        # rather than letting the comparison raise TypeError.
        if batch_limit is not None and (not isinstance(batch_limit, int) or batch_limit < 1):
            raise ValueError(
                f"batch_limit must be a positive request count, got {batch_limit!r}."
            )
        self.variant = variant
        self.device = device or "vulkan"
        self._binary_override = Path(binary).expanduser() if binary else None
        self._model_override = Path(model_path).expanduser() if model_path else None
        self._device_index_override = device_index
        self._threads_override = threads
        self._timeout_override = timeout
        self._batch_limit_override = batch_limit
        self._seed_override = seed
        self._binary: Path | None = None
        self._model: Path | None = None
        self._last_sample_rate: int | None = None

    @property
    def device_index(self) -> int | None:
        """Vulkan device index passed to ``--device``: constructor arg wins over
        the ``ORACLE_AUDIOCPP_DEVICE`` env var; None selects audio.cpp's default."""
        if self._device_index_override is not None:
            return self._device_index_override
        return _env_int("ORACLE_AUDIOCPP_DEVICE")

    @property
    def threads(self) -> int | None:
        """Optional ``--threads`` value: constructor arg wins over
        ``ORACLE_AUDIOCPP_THREADS``; None lets audio.cpp use its default."""
        if self._threads_override is not None:
            return self._threads_override
        return _env_int("ORACLE_AUDIOCPP_THREADS")

    @property
    def timeout(self) -> float:
        """Per-synthesis timeout in seconds: constructor arg wins over
        ``ORACLE_AUDIOCPP_TIMEOUT``; the env default is 600."""
        if self._timeout_override is not None:
            return float(self._timeout_override)
        return _synthesis_timeout_seconds()

    @property
    def batch_limit(self) -> int:
        """Maximum requests per ``--request-sequence`` subprocess: constructor
        arg (fed by the CLI flag / GUI spin box / manifest field) wins over
        ``ORACLE_AUDIOCPP_MAX_BATCH``; the default is 32."""
        if self._batch_limit_override is not None:
            return self._batch_limit_override
        return _vulkan_batch_max_requests()

    @property
    def seed(self) -> int | None:
        """Deterministic sampling seed passed to audio.cpp's ``--seed``: the
        constructor arg (fed by the CLI ``--seed`` flag / manifest field) wins
        over the ``ORACLE_AUDIOCPP_SEED`` env var; None lets audio.cpp use its
        default (a fresh random seed per process)."""
        if self._seed_override is not None:
            return self._seed_override
        return _env_int("ORACLE_AUDIOCPP_SEED")

    @property
    def binary(self) -> Path | None:
        if self._binary is None:
            if self._binary_override is not None:
                self._binary = self._binary_override if self._binary_override.exists() else None
            else:
                self._binary = find_audiocpp_binary()
        return self._binary

    @property
    def model(self) -> Path | None:
        if self._model is None:
            override = self._model_override or _model_from_env()
            if override:
                # An explicit path (constructor arg or env) is honored as-is
                # so a broken value surfaces clearly in ensure_model_ready.
                self._model = Path(override).expanduser()
            else:
                self._model = find_audiocpp_model()
        return self._model

    @property
    def engine_version(self) -> str:
        # Deterministic and side-effect free so chunk-cache keys stay stable
        # across runs without probing the binary.
        #
        # "v2": the engine now forwards the speaker's tuning settings
        # (guidance-scale/temperature/top-p/repetition-penalty/min_p) to
        # audio.cpp. The chunk hash always included voice_settings, but older
        # Vulkan stems were synthesized with those settings ignored, so the
        # version bump invalidates stale cache entries and lets the newly
        # faithful settings take effect.
        return "audio.cpp (ggml vulkan) v2"

    @property
    def sample_rate(self) -> int:
        return self._last_sample_rate or DEFAULT_SAMPLE_RATE

    def ensure_model_ready(self) -> None:
        missing: list[str] = []
        if self.binary is None:
            missing.append(
                "audiocpp_cli binary (build it with scripts/build_audio_cpp.sh or set ORACLE_AUDIOCPP_CLI)"
            )
        recovery = (
            "download it with scripts/download_audio_cpp_model.sh, "
            "then set ORACLE_AUDIOCPP_MODEL to the printed path"
        )
        if self.model is None:
            missing.append(f"Chatterbox ggml model ({recovery})")
        elif not self.model.exists():
            missing.append(f"Chatterbox ggml model path does not exist: {self.model} ({recovery})")
        if missing:
            raise AudioCppUnavailableError(
                "The Vulkan inference backend needs audio.cpp and its Chatterbox model. Missing: "
                + "; ".join(missing)
                + ". See the README section 'Vulkan backend (audio.cpp)'. "
                "Re-run with inference_backend=pytorch to use the default path."
            )

    def prepare_reference(
        self,
        project_cache: ProjectCache,
        speaker: str,
        reference_path: str,
    ) -> CachedReference:
        source = Path(reference_path)
        if not source.exists():
            raise FileNotFoundError(
                f"Voice reference for the Vulkan backend does not exist: {source}. "
                "audio.cpp reads the reference wav directly, so provide a readable WAV path."
            )
        try:
            sample_rate = int(sf.info(str(source)).samplerate)
        except Exception as exc:
            raise ValueError(
                f"Voice reference for the Vulkan backend is not readable audio: {source} ({exc})"
            ) from exc
        return CachedReference(
            original_path=str(source),
            normalized_path=str(source),
            original_hash=hash_file(source),
            sample_rate=sample_rate,
        )

    def prepare_conditioning(
        self,
        project_cache: ProjectCache,
        speaker: str,
        cached_reference: CachedReference,
        settings: VoiceSettings,
    ) -> VulkanConditioning:
        cache_id = hash_payload(
            {
                "backend": "vulkan",
                "speaker": speaker,
                "reference_hash": cached_reference.original_hash,
            }
        )
        return VulkanConditioning(
            cache_id=cache_id,
            reference_path=Path(cached_reference.original_path),
            speaker=speaker,
        )

    def synthesize(
        self,
        text: str,
        conditioning: VulkanConditioning,
        settings: VoiceSettings,
    ) -> np.ndarray:
        self.ensure_model_ready()
        reference_path = conditioning.reference_path
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Voice reference for the Vulkan backend does not exist: {reference_path}"
            )
        with tempfile.TemporaryDirectory(prefix="oracle_vulkan_") as temp_dir:
            out_wav = Path(temp_dir) / "synthesis.wav"
            command = self._build_command(text, reference_path, out_wav, settings)
            completed = self._run_command(command)
            if completed.returncode != 0:
                self._raise_for_failure(completed)
            if not out_wav.exists():
                raise RuntimeError(
                    "audio.cpp exited 0 but produced no output file at "
                    f"{out_wav} (command: {' '.join(str(part) for part in command)})"
                )
            audio, rate = sf.read(str(out_wav), dtype="float32")
            self._last_sample_rate = int(rate)
            return np.asarray(ensure_mono(audio), dtype=np.float32).squeeze()

    def _build_command(
        self,
        text: str,
        reference_path: Path,
        out_wav: Path,
        settings: VoiceSettings,
    ) -> list[str]:
        command = [
            str(self.binary),
            "--task",
            "clon",
            "--family",
            "chatterbox",
            "--model",
            str(self.model),
            "--backend",
            "vulkan",
        ]
        device_index = self.device_index
        if device_index is not None:
            command += ["--device", str(device_index)]
        threads = self.threads
        if threads is not None:
            command += ["--threads", str(threads)]
        seed = self.seed
        if seed is not None:
            command += ["--seed", str(seed)]
        command += self._tuning_flags(settings)
        command += [
            "--text",
            text,
            "--voice-ref",
            str(reference_path),
            "--out",
            str(out_wav),
        ]
        return command

    @staticmethod
    def _tuning_options(settings: VoiceSettings) -> dict[str, str]:
        """The speaker's tuning settings as a request-option map.

        audio.cpp's chatterbox session (session.cpp) reads exactly these
        option keys, so forwarding them keeps Vulkan renders faithful to the
        same VoiceSettings the PyTorch path honors (including emotion-blended
        values). The map is the single source of truth: ``_tuning_flags``
        renders it to CLI argv for single synthesis, and ``synthesize_batch``
        embeds it per request in the ``--request-sequence`` JSON.
        """
        options = {
            "guidance_scale": f"{settings.cfg_weight:g}",
            "temperature": f"{settings.temperature:g}",
            "top_p": f"{settings.top_p:g}",
            "repetition_penalty": f"{settings.repetition_penalty:g}",
            "min_p": f"{settings.min_p:g}",
        }
        if settings.variant == "multilingual":
            options["language"] = settings.language
        return options

    @classmethod
    def _tuning_flags(cls, settings: VoiceSettings) -> list[str]:
        """Render ``_tuning_options`` to audio.cpp CLI argv.

        ``min_p`` has no dedicated CLI flag, so it goes through
        ``--request-option key=value``; ``language`` is its own flag.
        """
        flags: list[str] = []
        for key, value in cls._tuning_options(settings).items():
            if key == "language":
                flags += ["--language", value]
            elif key == "min_p":
                flags += ["--request-option", f"min_p={value}"]
            else:
                flags += [f"--{key.replace('_', '-')}", value]
        return flags

    def _build_batch_command(self, sequence_path: Path, out_dir: Path) -> list[str]:
        """argv for one audio.cpp process serving many utterances via --request-sequence.

        Device/threads stay CLI-level (they configure the model load); every
        per-request knob (text, voice_ref, tuning options, language) lives in
        the sequence JSON. ``--mode offline`` is explicit because the batch
        path rejects streaming mode.
        """
        command = [
            str(self.binary),
            "--task",
            "clon",
            "--family",
            "chatterbox",
            "--model",
            str(self.model),
            "--backend",
            "vulkan",
            "--mode",
            "offline",
        ]
        device_index = self.device_index
        if device_index is not None:
            command += ["--device", str(device_index)]
        threads = self.threads
        if threads is not None:
            command += ["--threads", str(threads)]
        seed = self.seed
        if seed is not None:
            command += ["--seed", str(seed)]
        command += ["--request-sequence", str(sequence_path), "--out-dir", str(out_dir)]
        return command

    def synthesize_batch(
        self,
        entries: list[tuple[str, VulkanConditioning, VoiceSettings]],
        on_request_complete: Callable[[int], None] | None = None,
    ) -> list[tuple[np.ndarray, int, float]]:
        """Synthesize several utterances in ONE audio.cpp process (one model load).

        The single biggest cost of the Vulkan backend is the per-utterance
        subprocess spawn: every fresh ``audiocpp_cli`` reloads the multi-GB
        GGUF and recompiles Vulkan shaders. ``--request-sequence`` runs the
        whole group through one process, amortizing that cost across the
        render. Each entry is ``(text, conditioning, voice_settings)``; the
        result list matches entry order as ``(audio, sample_rate, wall_ms)``
        triples, where ``wall_ms`` is audio.cpp's own per-request timing
        (parsed from its ``[TIMING]`` lines; 0.0 when unavailable) so the
        pipeline records truthful per-item synthesize seconds.

        ``on_request_complete(request_index)`` (when given) is called as each
        request's ``request_<index>.wav`` lands in ``--out-dir`` while the
        subprocess is still running -- audio.cpp writes each wav inside its
        per-item ``on_result`` callback, so the file's appearance is a reliable
        live-progress signal independent of C++ stdout buffering. The callback
        enables the GUI progress bar to advance during a batched render
        instead of jumping at the end.

        Defense-in-depth: this method also enforces the same batch cap the
        pipeline uses (``ORACLE_AUDIOCPP_MAX_BATCH``, default 32). A group
        larger than the cap raises ``ValueError`` instead of building an
        unbounded requests.json, so even a direct caller that skips the
        pipeline's grouping (e.g. a future server mode) can never ship one
        gigantic request sequence to audio.cpp.
        """
        if not entries:
            return []
        batch_limit = self.batch_limit
        if len(entries) > batch_limit:
            raise ValueError(
                "AudioCppVulkanEngine.synthesize_batch accepts at most "
                f"{batch_limit} requests per subprocess (got {len(entries)}), "
                "so one requests.json can never grow unbounded. Split the "
                "group at this cap (the pipeline's synthesize_tasks_batched "
                "does this for renders) or raise the limit with "
                "ORACLE_AUDIOCPP_MAX_BATCH."
            )
        self.ensure_model_ready()
        with tempfile.TemporaryDirectory(prefix="oracle_vulkan_batch_") as temp_dir:
            temp = Path(temp_dir)
            requests: list[dict[str, Any]] = []
            for index, (text, conditioning, settings) in enumerate(entries):
                reference_path = conditioning.reference_path
                if not reference_path.exists():
                    raise FileNotFoundError(
                        f"Voice reference for the Vulkan backend does not exist: {reference_path}"
                    )
                requests.append(
                    {
                        "id": f"request_{index}",
                        "text": text,
                        # audio.cpp resolves voice_ref relative to the sequence
                        # file's directory, so absolute paths are required.
                        "voice_ref": str(reference_path.resolve()),
                        "options": self._tuning_options(settings),
                    }
                )
            sequence_path = temp / "requests.json"
            sequence_path.write_text(json.dumps({"requests": requests}), encoding="utf-8")
            out_dir = temp / "out"
            out_dir.mkdir()
            command = self._build_batch_command(sequence_path, out_dir)
            # The per-synthesis timeout (default 600s) bounds ONE utterance;
            # a batch of N requests legitimately needs N times the headroom,
            # otherwise long renders would time out as one subprocess even
            # though every individual utterance was fast.
            batch_timeout = self.timeout * max(1, len(entries))
            if on_request_complete is not None:
                completed = self._run_batch_command_streaming(
                    command, out_dir, len(entries), on_request_complete, batch_timeout
                )
            else:
                completed = self._run_command(command, timeout=batch_timeout)
            if completed.returncode != 0:
                self._raise_for_failure(completed)
            wall_ms: dict[int, float] = {}
            for match in _BATCH_TIMING_LINE.finditer(completed.stdout or ""):
                wall_ms[int(match.group(1))] = float(match.group(2))
            outputs: list[tuple[np.ndarray, int, float]] = []
            for index in range(len(entries)):
                wav = out_dir / f"request_{index}.wav"
                if not wav.exists():
                    raise RuntimeError(
                        "audio.cpp batch exited 0 but produced no output for request "
                        f"request_{index} at {wav} "
                        f"(command: {' '.join(str(part) for part in command)})"
                    )
                audio, rate = sf.read(str(wav), dtype="float32")
                self._last_sample_rate = int(rate)
                outputs.append(
                    (
                        np.asarray(ensure_mono(audio), dtype=np.float32).squeeze(),
                        int(rate),
                        wall_ms.get(index, 0.0),
                    )
                )
            return outputs

    def _run_batch_command_streaming(
        self,
        command: list[str],
        out_dir: Path,
        request_count: int,
        on_request_complete: Callable[[int], None],
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        """Run a batch subprocess, reporting each request as its wav lands.

        audio.cpp's batch runner calls its per-item sink immediately after
        each request finishes (``run_offline_batch`` in execution.cpp), and
        that sink writes ``--out-dir/request_<id>.wav`` synchronously before
        the next request starts. Polling for those files is therefore a
        truthful live-completion signal that does not depend on C++ stdout
        buffering. Stdout/stderr are drained by reader threads so a chatty
        binary can never block the child on a full pipe while we poll.

        On timeout the child is killed and the same RuntimeError the blocking
        path raises is thrown, so callers see one consistent failure mode.
        """
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def _drain(stream, sink: list[str]) -> None:
            assert stream is not None
            for line in stream:
                sink.append(line)

        threads = [
            threading.Thread(target=_drain, args=(proc.stdout, stdout_lines), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, stderr_lines), daemon=True),
        ]
        for thread in threads:
            thread.start()

        reported: set[int] = set()
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if time.monotonic() > deadline:
                proc.kill()
                proc.wait(timeout=5)
                raise RuntimeError(_timeout_error_message(timeout))
            for index in range(request_count):
                if index not in reported and (out_dir / f"request_{index}.wav").exists():
                    reported.add(index)
                    on_request_complete(index)
            time.sleep(0.05)
        for thread in threads:
            thread.join(timeout=5)
        # Final sweep: the last wav(s) may land in the instant before exit.
        for index in range(request_count):
            if index not in reported and (out_dir / f"request_{index}.wav").exists():
                reported.add(index)
                on_request_complete(index)
        return subprocess.CompletedProcess(
            args=command,
            returncode=proc.returncode,
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
        )

    def list_devices(self) -> list[dict[str, Any]]:
        """Return the Vulkan devices audio.cpp sees, as [{"index", "name"}, ...].

        Useful for choosing ``--device <n>`` on multi-GPU machines and for the
        doctor. Raises :class:`AudioCppUnavailableError` when the binary is
        missing or cannot be probed.
        """
        if self.binary is None:
            raise AudioCppUnavailableError(
                "audiocpp_cli binary is not built; run scripts/build_audio_cpp.sh to list Vulkan devices."
            )
        try:
            completed = subprocess.run(
                [str(self.binary), "--backend", "vulkan", "--list-devices"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AudioCppUnavailableError(f"audio.cpp --list-devices failed: {exc}") from exc
        devices: list[dict[str, Any]] = []
        for line in (completed.stdout or "").splitlines():
            match = _DEVICE_LINE.match(line.strip())
            if match:
                devices.append({"index": int(match.group(1)), "name": match.group(2)})
        return devices

    def _run_command(
        self,
        command: list[str],
        *,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one audio.cpp invocation, honoring ``self.timeout`` by default.

        ``timeout`` overrides it for a specific call (the batch path passes a
        per-synthesis timeout scaled by request count, since a single
        ``--request-sequence`` process synthesizes many utterances)."""
        effective = self.timeout if timeout is None else timeout
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=effective,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(_timeout_error_message(effective)) from exc

    def _raise_for_failure(self, completed: subprocess.CompletedProcess[str]) -> None:
        output = f"{completed.stderr or ''}\n{completed.stdout or ''}"
        if _is_rdna1_device_lost(output):
            raise RDNA1VulkanError(
                "audio.cpp's Vulkan backend hit the known AMD RDNA1 device-lost failure "
                "(VK_ERROR_DEVICE_LOST during buffer init: the RDNA1 SDMA transfer queue "
                "rejects vkCmdFillBuffer in ggml_vk_buffer_memset -- see "
                "ggml-org/whisper.cpp#3611). This is a hardware/driver limitation, not a "
                "bug in The Oracle. Re-run with --inference-backend pytorch "
                "(inference_backend: pytorch) for this session."
            )
        raise RuntimeError(
            "audio.cpp Vulkan synthesis failed.\n"
            f"Command: {' '.join(str(part) for part in completed.args)}\n"
            f"Exit code: {completed.returncode}\n"
            f"Output: {(output or '(empty)').strip()[-2000:]}"
        )


def _is_rdna1_device_lost(output: str) -> bool:
    lowered = output.lower()
    return any(marker.lower() in lowered for marker in _RDNA1_DEVICE_LOST_MARKERS)
