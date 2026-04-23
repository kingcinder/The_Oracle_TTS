# The Oracle

The Oracle is a local PySide6 desktop app and CLI for turning unstructured `.txt` or `.md` scripts into polished conversation or monologue FLAC renders with Chatterbox. It is built for writers, audiobook experimenters, voice designers, and local-first TTS users who want a controlled workflow: ingest rough prose, review speaker turns, assign voices, record custom reference clips, and render the final audio without sending source material to a hosted service.

This beta branch brings the project to a dual-OS installable product surface with a shared Linux/Windows manager, a compact desktop workflow, stronger deterministic dialogue parsing, and an in-app voice recorder for creating local reference clips directly into `Seashells/`.

Chatterbox outputs include built-in Perth watermarking by design. This project does not remove or hide that.

## Beta Highlights

- Cross-platform bootstrap, doctor, and run entrypoints for Linux and Windows
- Compact desktop GUI with monologue mode and a denser review workspace
- Stronger deterministic text ingest and speaker attribution for messy real-world input
- In-app voice recorder with selectable microphone, hardware-aware sample-rate filtering, mono FLAC export, and playback
- Saved GUI settings, templates, and project manifests
- Deterministic and real-engine smoke validation paths for repository verification

## What It Does Best

- Turns messy dialogue, prose, pasted notes, and screenplay-like text into reviewable speaker turns.
- Supports both two-speaker conversations and single-speaker monologues from the same interface.
- Keeps reference voices repo-local in `Seashells/`, including custom clips recorded inside the app.
- Gives advanced users direct CLI control while keeping the default GUI workflow approachable.
- Treats external runtime expectations such as `ffmpeg`, fresh-shell PATH propagation, and optional model prefetches as visible diagnostics rather than hidden assumptions.

## Product Scope

- Supported Python: `3.11` or `3.12`
- Supported operating systems: Linux and Windows
- Default execution path: CPU
- TTS backend: Chatterbox only
- Package version: `0.9.0b1`
- Primary output format: FLAC
- Primary reference-voice location: `Seashells/`

## Quick Start

Clone the repository, open a terminal in the repo root, and use the platform wrapper for your OS. The wrappers call the shared manager in `scripts/manage_install.py`, so Linux and Windows stay on the same install path instead of drifting into separate setup logic.

### Linux

Install Python `3.11` or `3.12`, `venv`, and `ffmpeg`, then run:

```bash
./install_oracle_tts.sh
```

For repo-local bootstrap without desktop integration:

```bash
./bootstrap_oracle_tts.sh
```

### Windows

Install Python `3.11` or `3.12` and `ffmpeg`, then run from `cmd.exe`:

```cmd
bootstrap_windows.cmd
```

That wrapper creates or reuses `.venv`, installs the project through the shared manager, and runs the Windows doctor path.

Windows entrypoints:

```cmd
doctor_windows.cmd
run_windows.cmd
```

PowerShell entrypoints remain available:

```powershell
.\install_oracle_tts.ps1
.\bootstrap_oracle_tts.ps1
.\doctor_oracle_tts.ps1
.\run_oracle_tts.ps1
```

If local script execution is blocked by policy, use a one-shot bypass:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_oracle_tts.ps1
```

After installation, launch the GUI with:

```powershell
the-oracle gui
```

If the launcher directory has not propagated into the current shell yet, use the platform run wrapper instead:

```powershell
.\run_oracle_tts.ps1
```

## Managed Launcher

The managed launcher is installed into:

- Linux: `~/.local/bin/the-oracle`
- Windows: `%APPDATA%\Python\Scripts\the-oracle.cmd`

If that directory is not already on `PATH`, the doctor reports it clearly. Missing `ffmpeg`, fresh-shell PATH propagation, and unprefetched turbo weights remain visible checks, but they are treated as external environment expectations rather than repository correctness failures in CI-safe mode.

## Diagnostics

Linux:

```bash
./doctor_oracle_tts.sh
./run_oracle_tts.sh
```

Windows:

```cmd
doctor_windows.cmd
run_windows.cmd
```

The doctor checks:

- supported Python version
- `ffmpeg` availability
- managed launcher health
- Chatterbox and Perth importability
- Chatterbox CPU model initialization
- Qt GUI readiness
- deterministic smoke render readiness
- real-engine smoke prerequisites

Doctor results are intentionally split between repository correctness and external environment readiness. Missing `ffmpeg`, launcher PATH warnings in a fresh shell, and unprefetched optional turbo weights are useful warnings, but they are not treated as proof that the repository itself is broken in CI-safe validation.

## GUI Workflow

1. Launch the app with `the-oracle gui` or a platform wrapper.
2. Choose an input `.txt` or `.md` script from `Input/` or another local folder.
3. Choose an output directory and optional output filename.
4. Select the model variant, correction mode, loudness profile, and voice reference clips.
5. Enable `Monologue Mode` when the entire script should be rendered as Speaker A and Speaker B should stay hidden.
6. Click `Analyze` to parse the script into reviewable turns.
7. Review or edit speaker assignment, repaired text, and emotion line by line.
8. Use row insertion/removal and preview where needed to refine the plan.
9. Click `Render FLAC` to produce the final output.

GUI capabilities:

- profile save/load
- saved templates
- project save/load manifests
- row insertion and removal
- per-line preview
- in-app progress dialogs
- direct access to the voice recorder from the main window

## Voice Recorder

The GUI includes a `Voice Recorder` button and `Tools -> Voice Recorder`. The recorder is designed for making "Seashell" reference clips in-house: pick a script, select a microphone, record, stop, and save a local FLAC clip into `Seashells/` for later use as a speaker reference.

Recorder workflow:

1. Choose a text or markdown file to display as the read-aloud script.
2. Select the microphone device.
3. Select a supported sample rate from the recorder dropdown.
4. Click `Record`.
5. Click `Pause` or `Resume` as needed.
6. Click `Stop` to write the FLAC file into `Seashells/`.
7. Use `Playback` to review the saved clip.

Recorder details:

- output format: FLAC
- final saved channel layout: mono
- output location: `Seashells/`
- capture compatibility: accepts mono devices and devices that only expose a native preferred channel mode through Qt
- downmix behavior: native stereo capture is downmixed to mono before FLAC export
- sample-rate menu: common mic/recording frequencies ranging from headset territory through studio-oriented rates, filtered to rates actually supported by the selected microphone
- verified local device class: `Microphone (USB PnP Audio Device)` style USB capture devices that expose 48 kHz stereo float through Qt
- output filename: sanitized from the chosen clip name
- included boilerplate script: `Input/READ_THIS_TO_RECORD_SEASHELLS.txt`
- included canonical reference clip: `Seashells/Cody's Seashell.wav`
- derivative/private variants beyond the one original Cody Seashell asset are intentionally excluded from the repository product surface

The recorder intentionally saves normalized mono FLAC files because the downstream reference-conditioning path expects one clean reference payload per speaker. If a microphone exposes 44.1 kHz, 48 kHz, or another supported common rate, the dropdown shows only the formats Qt reports as recordable for that selected device.

## CLI Usage

Show the installed version:

```bash
the-oracle --version
```

Batch render a dialogue file:

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

Use the CLI when you already trust the parsed input or want repeatable batch renders. Use the GUI when you want to inspect attribution, repair text, tune emotions, or validate speaker assignment before rendering.

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

## Repo Layout

- `Input/` curated sample/read-aloud scripts, including the Seashell recording script
- `Output/` renders, manifests, logs, and smoke outputs
- `Profiles/` saved GUI settings profiles
- `Seashells/` repo-local reference voice clips, including the original `Cody's Seashell.wav`
- `scripts/` bootstrap, doctor, smoke, and model utility entrypoints
- `src/the_oracle/` application code
- `tests/` unit and integration-style validation

## Sample Inputs

- `READ_THIS_TO_RECORD_SEASHELLS.txt`

## Development

Create a local development environment with a supported interpreter.

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

## Validation

The repository is validated with:

- cross-platform CI on Ubuntu and Windows
- Python `3.11` and `3.12`
- portability checks
- deterministic smoke render checks
- real-engine smoke prerequisites
- GUI regression tests
- dialogue parsing and speaker attribution regression coverage
- recorder UI/backend tests including FLAC writes across the supported sample-rate matrix
- recorder compatibility tests for devices that only expose native stereo/float capture while still producing mono output

## Product Notes

- Chatterbox `standard` remains the default quality-first backend.
- `multilingual` mode requires the multilingual Chatterbox variant and a real language code.
- `turbo` is optional and lower-latency, but not the default backend.
- Better voice quality comes from stronger local reference clips, not from a separate packaged voice library.
- Voice mixing is intentionally deferred because the current conditioning pipeline assumes one reference payload per speaker.
- The parser and attribution path are deterministic by design; dense literary multi-party prose is still heuristic rather than full coreference resolution.
- The recorder is intentionally conservative: it exposes only sample rates that the selected Qt audio device reports as recordable, then standardizes the saved reference clip to mono FLAC.

## Licensing

This project is publicly visible for inspection and evaluation, but it is not open-source. All rights are reserved.

Commercial licensing and leasing inquiries: codysa90@gmail.com
