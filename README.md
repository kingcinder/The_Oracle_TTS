# The Oracle

The Oracle is a local PySide6 desktop app and CLI for turning a `.txt` or `.md` two-person dialogue into a single FLAC render with Chatterbox. The repository now ships with a cross-platform bootstrap/install surface for both Linux and Windows.

Chatterbox outputs include built-in Perth watermarking by design. This project does not remove or hide that.

## Features

- Chatterbox-only render path with `standard`, `multilingual`, and `turbo` variants
- Voice cloning via per-speaker reference clips passed through `audio_prompt_path`
- Desktop GUI with review, repair, progress, and profile/template workflows
- CLI render flow with saved project manifests
- Deterministic smoke render path for repo-local verification without live model generation
- Managed bootstrap, install, doctor, run, and uninstall entrypoints for Linux and Windows

## Platform Support

- Supported Python: `3.11` or `3.12`
- Supported operating systems: Linux and Windows
- Default execution path: CPU
- Vulkan remains surfaced only as an availability check, not a claimed stable runtime

## Quick Start

### Linux

Install Python 3.11 or 3.12, `venv`, and `ffmpeg`, then run:

```bash
./install_oracle_tts.sh
```

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

## Product Notes

- Chatterbox standard is the default quality-first backend.
- Multilingual mode requires the multilingual Chatterbox variant and a real language code.
- Turbo is optional and lower-latency, but not the default backend.
- Better voice quality comes from stronger local reference clips, not from a separate packaged voice library.
- Voice mixing is intentionally deferred because the current conditioning pipeline assumes one reference payload per speaker.

## Licensing

This project is publicly visible for inspection and evaluation, but it is not open-source. All rights are reserved.

Commercial licensing and leasing inquiries: codysa90@gmail.com
