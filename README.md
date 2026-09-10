# The Oracle

**Release V1.11** — CUDA inference support, guided onboarding, and complete first-run workspace persistence.

The Oracle is a local PySide6 desktop app and CLI for turning a `.txt` or `.md` two-person dialogue into a single FLAC render with Chatterbox. The repository now ships with a cross-platform bootstrap/install surface for both Linux and Windows.

Chatterbox outputs include built-in Perth watermarking by design. This project does not remove or hide that.

## Features

- Chatterbox-only render path with `standard`, `multilingual`, and `turbo` variants
- Voice cloning via per-speaker reference clips passed through `audio_prompt_path`
- Desktop GUI with review, repair, progress, and profile/template workflows
- Six stylized, contrast-certified themes (menu-bar **Theme** menu): Studio Light, Night Deck, Sephiroth, Pulp Science Fiction, Tape Deck, Ocean Depths
- Collapsible GUI sections with per-section size sliders, all persisted across restarts
- The workspace remembers itself: theme, input file, every option and slider position, splitter layout, and window size are restored at launch
- Ctrl+hover help: hold the left **Control** key and hover any control for a short description of what it does; every slider also carries an exact-mechanics tooltip (engine knob, range, and formula-level effect)
- CLI render flow with saved project manifests
- Deterministic smoke render path for repo-local verification without live model generation
- Managed bootstrap, install, doctor, run, and uninstall entrypoints for Linux and Windows, plus a one-stop manager (`./oracle` / `oracle.ps1`) with install, start, update, and uninstall actions
- Opt-in CUDA inference through the existing PyTorch/Chatterbox path, with automatic NVIDIA hardware/VRAM suitability detection, CPU fallback, and selectable CUDA device index
- Opt-in Vulkan inference backend via audio.cpp (`--inference-backend vulkan`) for AMD RDNA1-class GPUs with no CUDA/ROCm path
- Pain-point markers: a `~` glued between two words in an input script (e.g. `syncronized~lockstep`) marks a junction where the engine used to hang up or mispronounce; the marker is stripped from synthesis automatically in every correction mode (including Verbatim) while the review table keeps your annotation — an unattached `~` (e.g. `~42`) is still read verbatim

## One-Stop Manager (Install / Start / Update / Uninstall)

Linux/macOS, from the repository root:

```bash
./oracle install      # set up the venv, register launchers, verify (doctor)
./oracle start        # launch the desktop GUI
./oracle update       # refresh an existing install (keeps your data)
./oracle uninstall    # remove launchers and the local venv
```

Windows (PowerShell, from the repository root):

```powershell
.\oracle.ps1 install
.\oracle.ps1 start
.\oracle.ps1 update
.\oracle.ps1 uninstall
```

`update` reinstalls dependencies and rebuilds the managed launchers while
keeping `Input/`, `Seashells/`, `Profiles/`, `Output/`, and your saved app
settings untouched. The underlying per-action scripts
(`install_oracle_tts.*`, `run_oracle_tts.*`, `doctor_oracle_tts.*`,
`bootstrap_oracle_tts.*`, `uninstall_oracle_tts.*`) remain available.

## Platform Support

- Supported Python: `3.11` or `3.12`
- Supported operating systems: Linux and Windows
- Default execution path: CPU (PyTorch Chatterbox in-process); CUDA is available when a CUDA-enabled PyTorch wheel and a suitable NVIDIA GPU are detected
- Vulkan is available as an opt-in inference backend via audio.cpp (`--inference-backend vulkan`); see "Vulkan Backend (audio.cpp)" below

## Quick Start

### Linux

Install Python 3.11 or 3.12, `venv`, and `ffmpeg`, then run:

```bash
./install_oracle_tts.sh
```

On Ubuntu (and other freedesktop Linux desktops) the installer registers **The
Oracle** in the OS programs/app list (`~/.local/share/applications/the-oracle.desktop`)
and drops a launchable shortcut on your Desktop. Both are removed again by
`./uninstall_oracle_tts.sh`.

For source-only bootstrap without desktop integration:

```bash
./bootstrap_oracle_tts.sh
```

### Windows

Install Python 3.11 or 3.12 and `ffmpeg`, then run from PowerShell:

```powershell
.\install_oracle_tts.ps1
```

If local script execution is blocked by policy, run the same command through a one-shot bypass:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_oracle_tts.ps1
```

For source-only bootstrap without Start Menu integration:

```powershell
.\bootstrap_oracle_tts.ps1
```

The managed launcher is installed into:

- Linux: `~/.local/bin/the-oracle`
- Windows: `%APPDATA%\Python\Scripts\the-oracle.cmd`

If that directory is not already on `PATH`, the doctor will tell you exactly what is missing.

## Diagnostics And Launch

Linux:

```bash
./doctor_oracle_tts.sh
./run_oracle_tts.sh
```

Windows:

```powershell
.\doctor_oracle_tts.ps1
.\run_oracle_tts.ps1
```

The doctor checks:

- Python version support
- `ffmpeg` availability
- managed launcher health
- Chatterbox and Perth importability
- Chatterbox CPU model initialization
- Qt GUI readiness
- deterministic smoke render readiness
- real-engine smoke prerequisites

## Optional Model Prefetch

Linux:

```bash
./.venv/bin/python scripts/download_models.py --variant all --device cpu
```

Windows:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --variant all --device cpu
```

Turbo-only prefetch:

Linux:

```bash
./.venv/bin/python scripts/download_models.py --variant turbo --device cpu
```

Windows:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --variant turbo --device cpu
```

## CLI Usage

`--speakerA-ref`/`--speakerB-ref` are optional: when omitted they default to the
repo-local `Seashells/` clips (the GUI's "Default Voices", including the
bundled `Seashells/generic/` library), so the simplest render is just:

```bash
the-oracle render --input Input/cli_short.txt --outdir Output
```

To override the voices explicitly:

```bash
the-oracle render \
  --input Input/cli_short.txt \
  --outdir Output \
  --speakerA-ref Seashells/<speaker_a>.wav \
  --speakerB-ref Seashells/<speaker_b>.wav
```

Project save/load workflow:

```bash
the-oracle render \
  --input Input/cli_short.txt \
  --outdir Output \
  --speakerA-ref Seashells/<speaker_a>.wav \
  --speakerB-ref Seashells/<speaker_b>.wav \
  --save-project Output/oracle_project.json

the-oracle render --project Output/oracle_project.json
```

GUI launch through the managed launcher:

```bash
the-oracle gui
```

Text repair runs a local LanguageTool grammar pass when available. First use
downloads the LanguageTool server (hundreds of MB). A render never waits on
that download: if a complete LanguageTool snapshot is already on disk it is
reused directly (this also covers the upstream quirk where the library's
pinned snapshot label goes stale vs. what the server actually serves);
otherwise the render falls back to the built-in local fixes immediately and
hands the download to a detached background helper, so a later run picks up
the real tool once the cache is populated (a lock file keeps rapid renders
from starting duplicate downloads). If the tool is cached but slow to start,
the load is bounded to 25 seconds — set `ORACLE_LANGUAGE_TOOL_TIMEOUT=<seconds>`
to tune that bound.

### First-run inference setup and tutorial

On the first launch after installation, The Oracle opens a non-modal **Inference Setup & Tour**. It inventories NVIDIA devices through PyTorch and `nvidia-smi`, reports each card's name/VRAM/runtime status, disables cards that are below the 4 GiB Chatterbox suitability floor, and leaves CPU/system DRAM as the guaranteed fallback. It also explains that AMD GPUs cannot use CUDA and that Vulkan/audio.cpp is the separate GPU path. The Continue button advances through a dependency-ordered tour: hardware discovery, backend choice, input/output, model and correction behavior, voice character, timing, hybrid voices, and finally Analyze/Preview/Render. The tutorial highlights the live control and positions itself beside it so its explanation and the control's tooltip remain readable.

After completion, use **Settings → Replay Setup & Tutorial** to replay the **Entire Setup & Tutorial**, **Hardware Discovery Only**, or **Main GUI Tour Only**. The selected PyTorch CPU/CUDA device or Vulkan backend is applied to the real inference picker and persisted with the existing Remember GPU/CPU choice setting. A skipped first-run tour can be replayed from the same menu at any time. The first-run page also sets independent default Input and Output folders; its checkboxes control whether those folder choices persist, while the last-used input file remains remembered separately.

The first time **Recording Studio** is opened, a second setup guide appears. It configures the microphone and device-supported sample rate, the shared Input/ teleprompter script, the Seashells/ voice-recording folder, and the safe boilerplate naming policy. It explains that a missing microphone disables recording, provides detailed microphone placement, room, breath, plosive, enunciation, and emotional-delivery guidance, and teaches audition/assignment after a take. The guide's **Continue** button applies choices as dependencies are introduced. Its folder, script, microphone, sample rate, filename, generic-name warning, and replay state are remembered whenever the studio closes; the folder/script checkboxes decide which defaults persist. The generic filename caution can be disabled in the popup and restored from Settings. Use **Settings → Replay Recording Studio Setup & Guide** to replay it.

### CUDA / NVIDIA GPU

CUDA is an additional PyTorch device mode, not a second synthesis engine. The existing Chatterbox pipeline, voice conditioning, cache behavior, previews, and render workers are reused unchanged. The GUI's **PyTorch Device** picker lists CPU plus detected NVIDIA cards; cards below the 4 GiB Chatterbox suitability floor are shown but disabled with the reason, and a machine with no usable GPU keeps CPU/DRAM available as the guaranteed fallback.

The installer probes `nvidia-smi` before installing PyTorch. Its default `auto` policy installs the CUDA 12.4 PyTorch wheels only when at least one NVIDIA card reports enough VRAM; otherwise it installs the CPU wheels. You can explicitly choose the runtime:

```bash
./oracle install --pytorch-runtime auto   # recommended: hardware-aware
./oracle install --pytorch-runtime cuda   # force CUDA 12.4 PyTorch wheels
./oracle install --pytorch-runtime cpu    # force CPU wheels
./oracle update --pytorch-runtime cuda
```

The CLI selects a device with `--device-mode cuda` and optionally pins a card with `--cuda-device N`:

```bash
the-oracle render --input Input/cli_short.txt --outdir Output \
  --device-mode cuda --cuda-device 0
```

A CUDA choice is validated again at render time using PyTorch's actual runtime and device properties. The live render panel identifies the selected NVIDIA card rather than merely saying "GPU". If the driver, wheel, or VRAM is unsuitable, The Oracle fails with an actionable message and CPU remains selectable. The GUI remembers the selected PyTorch device when **Remember GPU/CPU choice** is enabled, and saved project manifests/profiles round-trip the CUDA device index. `scripts/doctor.py` reports every detected NVIDIA card, VRAM, driver version, and whether the installed runtime can use it. CUDA cannot accelerate AMD cards; AMD systems should use CPU or the Vulkan/audio.cpp backend described below.


The default render path runs Chatterbox in-process on PyTorch (CPU). An opt-in
`--inference-backend vulkan` path instead shells out to
[audio.cpp](https://github.com/0xShug0/audio.cpp) — a ggml-based C++ inference
engine built with `-DENGINE_ENABLE_VULKAN=ON` — to run the Chatterbox model
family on a Vulkan GPU. This is the only realistic GPU-acceleration route on
AMD RDNA1 hardware (e.g. RX 5700 XT), which has no CUDA and no official ROCm
support, and whose PyTorch Vulkan backend is experimental/mobile-only.

### Automatic setup (one-click CPU→GPU switch)

Selecting the **Vulkan (audio.cpp)** inference backend no longer requires
manual script runs: The Oracle completes the switch by itself.

- **GUI** — switching the Inference Backend dropdown to Vulkan, or clicking
  Render/Preview with it selected, automatically builds `audiocpp_cli`
  (`scripts/build_audio_cpp.sh`) if it is missing and downloads the Chatterbox
  model (`scripts/download_audio_cpp_model.sh`) if it is missing, streaming
  progress into the backend status panel. The queued render/preview starts by
  itself once setup finishes, and `ORACLE_AUDIOCPP_CLI`/`ORACLE_AUDIOCPP_MODEL`
  are set for the session.
- **CLI** — `the-oracle render --inference-backend vulkan` runs the same
  automatic setup first (output streamed to stderr; a ready backend prints a
  single `Vulkan backend ready.` line). Pass `--no-audio-cpp-setup` to keep
  the old fail-fast behavior. A one-shot `the-oracle setup-vulkan` command
  performs just the setup and prints the resolved binary/model paths — the
  CLI and GUI apply them automatically, so no shell exports are needed.

A failed setup is surfaced visibly with the exact manual commands to run, and
never silently retried in a loop; a later Render/Preview click retries it.

### Hardware requirements

- Any Vulkan-capable GPU and driver (primary target: AMD RADV on RDNA1+).
- `vulkaninfo` on `PATH` so tests and diagnostics can detect a Vulkan device
  (they skip gracefully when absent).

### Build

```bash
./scripts/build_audio_cpp.sh
```

This clones audio.cpp into `audio.cpp/` (ggml is vendored in-tree at
`audio.cpp/external/ggml`), applies The Oracle's vendored fixes (see below),
builds `audiocpp_cli` with the Vulkan backend into
`audio.cpp/build/linux-vulkan-release/bin/` (chatterbox family only by default;
override with `AUDIOCPP_MODEL_SET=full`), and points you at
`scripts/download_audio_cpp_model.sh` for the model. It requires GCC 13+,
CMake, and the Vulkan SDK. You normally do not need to run this by hand —
selecting the Vulkan backend (GUI) or `the-oracle setup-vulkan` (CLI) builds
it automatically on first use; this manual path is for pre-installing ahead
of time.

To build the CLI **and** fetch the Chatterbox model in one command, pass
`--with-model`; a single command then gets you from nothing to a renderable
Vulkan setup (the download step is skipped with `SKIP_MODEL_DOWNLOAD=1`, e.g.
for CI that already has the model):

```bash
./scripts/build_audio_cpp.sh --with-model
```

### Enable

Selecting the Vulkan backend already completes the setup for the current
session (see Automatic setup above), so the manual steps below are only needed
when you want to pre-install or persist the environment across sessions:

```bash
./scripts/download_audio_cpp_model.sh   # downloads Chatterbox Q8_0 GGUF; the app applies the path itself

the-oracle render \
  --input Input/cli_short.txt --outdir Output \
  --speakerA-ref "Seashells/Cody's Seashell.wav" \
  --speakerB-ref "Seashells/Cody's Seashell1.wav" \
  --inference-backend vulkan
```

`scripts/download_audio_cpp_model.sh` wraps audio.cpp's own model manager
(`tools/model_manager_v2.py`) so the model lands in the right place and it
prints the exact `export ORACLE_AUDIOCPP_MODEL=...` line to paste. The default
package is the spec's recommended Chatterbox Q8_0 GGUF; override with
`AUDIOCPP_MODEL_PACKAGE=chatterbox_f16` (or set `AUDIOCPP_MODELS_ROOT` to
install elsewhere), and pass `--dry-run` to preview the download or
`--overwrite` to re-download an existing install. Relocate the install with
the `AUDIOCPP_MODELS_ROOT` env var, not a `--models-root` argument — the
script already passes that flag itself and derives the printed path from its
own root.

The `turbo` variant is PyTorch-only: the CLI rejects
`--inference-backend vulkan --model-variant turbo` with a clear error, and the
GUI disables the Vulkan option when turbo is selected (see below).
`--device-mode` is a PyTorch-path flag and is ignored on the Vulkan backend
(leave it at its default `cpu`). Saved project manifests persist
`inference_backend`, so a project saved from a Vulkan render re-renders on
Vulkan.

The speaker's tuning settings are forwarded to audio.cpp's chatterbox session
so Vulkan renders sound the same as PyTorch ones: `cfg_weight` →
`--guidance-scale`, `temperature`, `top_p`, `repetition_penalty`, and `min_p`
(via `--request-option min_p=...`, since audio.cpp has no dedicated `--min-p`
flag). Multilingual runs forward `--language`. (Vulkan stem caches created
before this tuning-settings fix are invalidated and will re-render once.)

**Rendering batches by default.** The single biggest cost of the Vulkan
backend is the per-utterance subprocess spawn: every fresh `audiocpp_cli`
reloads the multi-GB GGUF and recompiles Vulkan shaders. The render path
therefore synthesizes every cache-missing stem through audio.cpp
`--request-sequence` batches (a JSON file of `{id, text, voice_ref, options}`
requests, output per request to `--out-dir/<id>.wav`), so the model load and
shader compile happen once per batch instead of once per line. Batches are
capped at 32 requests per subprocess (`ORACLE_AUDIOCPP_MAX_BATCH=<n>` to
override), so an enormous render is split into several processes instead of
shipping one gigantic `requests.json` — and a failed batch only takes down
its own group, never the whole render. Per-request `[TIMING]` lines keep the
per-utterance synthesize timings in `logs/render_timings.json` truthful, and
the timeline records how many audio.cpp processes the render actually
spawned (`vulkan_batch_processes` / `vulkan_batch_requests`). The GUI's
progress bar advances **live, per request**: because audio.cpp writes each
`request_N.wav` into `--out-dir` as soon as that request finishes, the engine
polls the output directory while the subprocess runs and emits a progress
event for every completed request — so a long batched render shows steady
progress instead of jumping from "starting" to "done" at the end. Cache hits
never touch the engine at all; a failed batch marks every request in it
failed (never silently retried), consistent with the partial-render behavior
of the PyTorch path.

Optional environment knobs:

- `ORACLE_AUDIOCPP_DEVICE=<n>` — select a specific Vulkan device on multi-GPU
  machines (passed as `--device <n>`). See the devices audio.cpp detects with
  `audiocpp_cli --backend vulkan --list-devices` (or the doctor).
- `ORACLE_AUDIOCPP_THREADS=<n>` — pass `--threads <n>` to audio.cpp
  (default is 4).
- `ORACLE_AUDIOCPP_TIMEOUT=<seconds>` — per-synthesis timeout (default 600).
  A batched render scales this by its request count (one `--request-sequence`
  process synthesizes N utterances, so it gets N x the per-synthesis timeout)
  so long renders are not killed just because they are one long subprocess.
- `ORACLE_AUDIOCPP_MAX_BATCH=<n>` — maximum cache-missing stems per
  `--request-sequence` subprocess (default 32). An enormous render is split
  into several processes at this cap, and a failed batch only takes down its
  own group. The engine also enforces the same cap as defense-in-depth:
  `AudioCppVulkanEngine.synthesize_batch` refuses any group larger than this
  (raising a clear error instead of writing an unbounded `requests.json`), so
  even a direct caller that skips the pipeline's grouping — e.g. a future
  server mode — can never ship one gigantic request sequence to audio.cpp.

The same device/threads/timeout/batch-cap knobs work without environment
variables as render CLI flags and are persisted in saved project manifests
(so a project re-renders with the same knobs):

- `--audio-cpp-device <n>` — equivalent of `ORACLE_AUDIOCPP_DEVICE`; requires
  `--inference-backend vulkan`.
- `--audio-cpp-threads <n>` — equivalent of `ORACLE_AUDIOCPP_THREADS`; requires
  `--inference-backend vulkan`.
- `--audio-cpp-timeout <seconds>` — equivalent of `ORACLE_AUDIOCPP_TIMEOUT`;
  requires `--inference-backend vulkan`.
- `--audio-cpp-max-batch <n>` — equivalent of `ORACLE_AUDIOCPP_MAX_BATCH`
  (requests per `--request-sequence` subprocess); requires
  `--inference-backend vulkan`.

Explicit CLI/manifest values take precedence over the environment variables.
`the-oracle render --help` documents all four flags, and the doctor's Vulkan
check reports the same device list to pick `<n>` from.

In the desktop GUI, the backend is chosen with the **Inference Backend**
dropdown in Shared Render Settings: `PyTorch (CPU)` (default) or `Vulkan
(audio.cpp)` (opt-in). It governs both render and preview, so previewing a
row uses the same backend you render with, and it requires the same setup as
the CLI path — a Vulkan-enabled `audiocpp_cli` plus `ORACLE_AUDIOCPP_MODEL`
(see Build/Enable above). If either is missing when you select Vulkan, the
GUI warns inline right at selection time (and blocks render with the same
message), so the failure is surfaced before the render worker starts instead
of deep inside it. Because `turbo` is PyTorch-only, selecting the
turbo model variant disables the `Vulkan (audio.cpp)` option and falls back
to PyTorch, so the GUI never offers the unusable turbo+Vulkan combination.

The GUI can also fetch the model for you: **Settings → Download Vulkan
Model...** runs `scripts/download_audio_cpp_model.sh` in a background thread
(the UI stays responsive — model downloads are large) and, when it finishes,
sets `ORACLE_AUDIOCPP_MODEL` for the current session, clears the inline
prerequisite warning, and shows the exact `export ORACLE_AUDIOCPP_MODEL=...`
line — and applies it automatically for the session.

Next to the backend dropdown is a **Test Vulkan Backend** button. It runs a
quick audio.cpp preflight in a background thread — binary present, model
present, and `audiocpp_cli --backend vulkan --list-devices` — and reports
which GPU the Vulkan backend would use (the selected **Vulkan Device** index,
marked `(selected)` in the device list, or audio.cpp's default when set to
Auto), so you can validate the setup before rendering. It works from either
backend selection, since it only inspects the Vulkan setup; it is disabled
while a preflight runs and when the PyTorch-only `turbo` variant is selected.


The selection is persisted in saved settings profiles/templates (Settings →
Save Settings... / Save Current as Template...) and in saved project
manifests (File → Save Project / Save Project As...), and it is restored when
either is re-opened — so a project saved from a Vulkan render re-renders on
Vulkan, and a profile saved with Vulkan selected re-applies it. It also flows
through the existing `inference_backend` field in the manifest's
`render_settings`.

On top of those manual profiles, the GUI now **remembers the backend choice
across sessions with zero extra steps**: **Settings → Remember GPU/CPU choice**
(checked by default) records which inference backend you last used — plus the
Vulkan Device/Threads/Timeout/Max Batch knobs and the resolved
`audiocpp_cli`/Chatterbox model paths — in the app settings file
(`app_settings.json` under the config dir). Every launch restores the
selection automatically and re-applies the remembered audio.cpp paths to the
environment, so a session that rendered on Vulkan starts on Vulkan with the
GPU already wired up — no re-running the build/download, no shell exports.
A remembered path whose file has since moved or been deleted is surfaced in
the status panel and never silently applied; the normal automatic setup then
re-runs on its own to heal the gap. Uncheck the option to always start on
PyTorch (CPU).

With Vulkan selected, the same **Vulkan Device**, **Vulkan Threads**,
**Vulkan Timeout (s)**, and **Vulkan Max Batch** rows appear (enabled only
then). Selecting Vulkan
probes `audiocpp_cli --backend vulkan --list-devices` in a background thread
and populates the **Vulkan Device** dropdown with audio.cpp's real GPUs by
name (e.g. `Device 0: AMD Radeon RX 5700 XT (RADV NAVI10)`), with **Auto
(audio.cpp default)** always first — so you pick a real device instead of
guessing at a blind 0-15 index range. A stale saved device index that is no
longer detected is kept as an explicit `Device N: (not detected)` row (never
silently changed), and the label under the picker lists everything audio.cpp
sees. The **Vulkan Timeout (s)** spin sets the per-synthesis timeout
(equivalent of `ORACLE_AUDIOCPP_TIMEOUT`, default 600), with 0 meaning
"Default (600s)". The **Vulkan Max Batch** spin sets the requests-per-
subprocess cap (equivalent of `ORACLE_AUDIOCPP_MAX_BATCH`, default 32), with
0 meaning "Default (32)". These values are persisted in the same
profiles/templates and manifests, so a GUI session needs no environment
variables to pin the device, thread count, timeout, or batch cap.

### Known RDNA1 device-lost issue and the vendored fix

RDNA1 (gfx1010/gfx1012) has a documented ggml-vulkan bug: `VK_ERROR_DEVICE_LOST`
during buffer initialization, traced to `ggml_vk_buffer_memset()` using the
SDMA transfer queue (see [ggml-org/whisper.cpp#3611](https://github.com/ggml-org/whisper.cpp/issues/3611)).
The Oracle ships a **vendored fix** for this: `scripts/patch_audio_cpp_ggml.sh`
applies it to the cloned audio.cpp tree before building — `ggml_vk_buffer_memset()`
routes through the compute queue when the device architecture is `AMD_RDNA1`
(`vk_command_pool` binds its queue, so the submission follows it too).

The change is marked in the source with `ORACLE VENDORED PATCH` comment blocks
and documented as reviewable artifacts in `scripts/patches/`:

- `ggml_rdna1_buffer_memset.patch` — the RDNA1 compute-queue fix, vendored
  against the ggml commit printed by `scripts/patch_audio_cpp_ggml.sh` (re-apply
  by hand if a ggml update drifts the anchor; the patch script fails loudly
  instead of silently losing it).
- `audio_cpp_space_safe_build.patch` — quotes sentencepiece's
  `-fmacro-prefix-map` so the build works from checkout paths containing
  spaces (required for this repo's own `the oracle tts` directory).

The patch script is idempotent and is invoked automatically by
`scripts/build_audio_cpp.sh` before compiling. If audio.cpp still reports
`VK_ERROR_DEVICE_LOST` (e.g. on an unpatched build or a different quirk), the
Python backend surfaces the failure with a clear error and you can re-run with
`--inference-backend pytorch` for that session — the error is never retried
silently.

## Repo Layout

- `Input/` sample dialogue files
- `Seashells/` repo-local reference voice clips
- `scripts/` install, doctor, smoke, and model utility entrypoints
- `src/the_oracle/` application code
- `tests/` unit and integration-style coverage

## Sample Inputs

The portable sample inputs under `Input/` include:

- `Read Aloud transcript.txt`
- `What is, reality.txt`
- `cli_short.txt`
- `test.txt`

## Development

Create a local development environment with a supported interpreter:

Linux:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/pytest
```

Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\pytest.exe
```

## Generic Voices And Voice Blending

**Generic voices.** The repo ships a small bundled generic voice library under
`Seashells/generic/` (male and female narrator references copied from
audio.cpp's demo voices — Apache-2.0, attribution in
`Seashells/generic/ATTRIBUTION.md`). These appear at the top of the GUI's
voice picker under **Default Voices**, so a render never needs a custom
recording. Drop your own neutral recordings into `Seashells/generic/` to
extend the library.

**Voice blending.** Chatterbox conditions each speaker on one reference
payload, so combining two voices works by deriving a single **blended
reference clip** from two sources. In each speaker panel: pick a second clip
under **Blend With Voice**, choose how strongly the base voice dominates with
**Base Voice Presence** (0–100%), and pick a **Blend Mode**: `Mix`
(weight-averaged), `Alternate` (the two clips take turns), or `Layer` (base
voice with the second underneath). The blend is deterministic and cached by
its inputs' content hashes, so a given (base, target, weight, mode) always
renders the same voice, on both the PyTorch and Vulkan backends, and stem
caches re-key automatically when the blend changes. Blend settings round-trip
through saved projects and GUI settings profiles.

**Saving blends as named voices.** **Save Blend As...** next to the blend
picker saves the current combination as a named voice that appears under
**Saved Blends** in every speaker panel's picker, right alongside the bundled
generic voices — so a favorite blend becomes a one-click voice like any
other. Saved blends live in `Profiles/blend_voices.json` (user-generated,
gitignored) with deterministic clips in `Profiles/.blends/`; saving the same
name again replaces the entry.

## Product Notes

- Chatterbox standard is the default quality-first backend.
- Multilingual mode requires the multilingual Chatterbox variant and a real language code.
- Turbo is optional and lower-latency, but not the default backend.
- Better voice quality comes from stronger local reference clips, not from a separate packaged voice library.
- Voice blending (combining two references into one conditioning clip) is supported; see "Generic Voices And Voice Blending" above.
- **Correction Mode → Verbatim (no changes)** passes the source text through exactly as written — no spelling, grammar, or punctuation edits — so narration is always faithful to the file. (The TTS engine applies its own required text normalization internally, identically on both backends.)

## Licensing

This project is publicly visible for inspection and evaluation, but it is not open-source. All rights are reserved.

Commercial licensing and leasing inquiries: codysa90@gmail.com
