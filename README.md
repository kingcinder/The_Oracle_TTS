# DualVoice Studio

DualVoice Studio is a local PySide6 desktop app and CLI for turning a `.txt` or `.md` two-person dialogue into a single FLAC render with Chatterbox. The product is now Chatterbox-exclusive: the standard Chatterbox model is the default quality-first backend, with multilingual and turbo variants available inside the same engine family.

Chatterbox outputs include built-in Perth watermarking by design. This project does not remove or hide that.

## Features

- Chatterbox-only render path with `standard`, `multilingual`, and `turbo` variants
- Voice cloning via per-speaker reference clips passed through `audio_prompt_path`
- GUI review table for `[index | speaker | original text | repaired text | emotion | duration | preview]`
- Background FLAC rendering with a live progress dialog, segment counter, stage text, and ETA when enough timing data exists
- Save/load GUI settings profiles plus reusable local templates for recurring setups
- Reference voice picker with repo-local default clips, recent custom clips, and a custom file chooser path
- Automatic normalization, punctuation restoration, spelling correction, and grammar cleanup
- Dual-speaker attribution with explicit markers, alternating-line fallback, clustering fallback, and anchor-ready plumbing
- Emotion inference mapped into live Chatterbox controls such as `cfg_weight`, `exaggeration`, and `temperature`, with per-speaker emotion intensity scaling
- Per-speaker language, pause-after-turn, and voice tuning controls, plus a clearly heuristic naturalness control
- Incremental stem caching keyed by repaired text, speaker, model variant, language, Chatterbox parameters, reference hash, and Chatterbox version
- Render timing audit logs under `logs/render_timings.json` for investigating dead time between speaker segments
- FLAC export with metadata tags plus render plans and correction logs

## Install

Exact Linux Mint 22.1 install flow:

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv ffmpeg \
  libasound2t64 libdbus-1-3 libegl1 libfontconfig1 libglib2.0-0t64 \
  libnss3 libopengl0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 \
  libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
  libxcb-shape0 libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0

git clone git@github.com:kingcinder/The_Oracle_TTS.git
cd The_Oracle_TTS
./bootstrap_oracle_tts.sh
```

`./bootstrap_oracle_tts.sh` creates or reuses `.venv`, installs the project plus `.[ml]`, installs CPU PyTorch and the Chatterbox runtime bundle explicitly, installs a managed `~/.local/bin/dualvoice` wrapper, checks Qt/XCB runtime dependencies, verifies `from chatterbox.tts import ChatterboxTTS`, warms the CPU model with `ChatterboxTTS.from_pretrained(device="cpu")`, and prints exact PASS/FAIL next steps.

Fresh-shell CLI verification:

```bash
dualvoice --help
```

Full diagnostics:

```bash
./doctor_oracle_tts.sh
```

Launch the GUI:

```bash
./run_oracle_tts.sh
```

If `dualvoice` is not found in a fresh shell after bootstrap, add `~/.local/bin` to `PATH`, open a new shell, and retry:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Optional model prefetch:

```bash
./.venv/bin/python scripts/download_models.py --variant all --device cpu
```

Deterministic repo smoke render:

```bash
./.venv/bin/python scripts/smoke_render.py
```

This smoke path is intentionally deterministic and uses a patched in-repo test engine so you can verify ingest, repair, caching, assembly, and FLAC export without depending on live Chatterbox generation. The real Chatterbox model smoke readiness is reported by `./doctor_oracle_tts.sh`.

## Run

GUI:

```bash
./run_oracle_tts.sh
```

Direct CLI launch after bootstrap:

```bash
dualvoice gui
```

CLI:

```bash
dualvoice render \
  --input examples/demo_dialogue.md \
  --outdir output \
  --speakerA-ref /path/to/speaker_a.wav \
  --speakerB-ref /path/to/speaker_b.wav \
  --model-variant standard \
  --cfg-weight 0.5 \
  --exaggeration 0.5
```

Use `--model-variant multilingual --language es` for multilingual rendering. `turbo` is available, but the default recommended path remains `standard`.

Saved project workflow:

```bash
dualvoice render \
  --input examples/demo_dialogue.md \
  --outdir output \
  --speakerA-ref /path/to/speaker_a.wav \
  --speakerB-ref /path/to/speaker_b.wav \
  --save-project output/oracle_project.json

dualvoice render --project output/oracle_project.json
```

The desktop app also supports `File > New Project`, `Open Project`, `Save Project`, and `Save Project As` for round-tripping long review/edit sessions without losing repaired text, speaker overrides, emotion edits, or Chatterbox voice settings.

Recurring setup workflow:

```bash
# In the GUI:
# Settings > Save Settings...
# Settings > Save Current as Template...
# Settings > Load Template
```

Templates and recent custom reference clips are stored in the user config directory under `~/.config/dualvoice_studio/` unless `XDG_CONFIG_HOME` overrides it.

## Output Layout

Each render creates:

- final tagged FLAC in the selected output directory
- `render_plan.json`
- `logs/corrections.json`
- `logs/corrections.diff`
- per-speaker profile JSON under `profiles/`
- cached utterance stems in `cache/utterances/`
- normalized reference clips in `cache/references/`
- optional exported stems in `stems/`

## Hardware Notes

- Linux and Windows are both supported at the code level.
- CPU-only mode works, but first-load and long-form renders are slow.
- The current verified runtime path in this repository is CPU.
- Vulkan GPU mode is surfaced in the GUI only as an availability check; it is not claimed as a working execution path unless the installed runtime is explicitly verified.

## Product Notes

- Chatterbox standard is the default quality-first backend.
- Multilingual mode requires the multilingual Chatterbox variant and a real language code.
- Turbo is lower-latency and optional; it is not the default backend.
- Chatterbox’s Perth watermark is retained in output audio.
- Hybrid CPU+GPU work splitting is not implemented because the current Chatterbox runtime in this repo does not expose a verified, defensible multi-device execution path.

## Demo

Use [examples/demo_dialogue.md](/home/oem/Documents/The_Oracle_TTS/examples/demo_dialogue.md) with two short reference clips to validate the full pipeline before moving to longer material.

## Licensing

This project is publicly visible for inspection and evaluation, but it is not open-source.
All rights are reserved.

Commercial licensing and leasing inquiries: codysa90@gmail.com
