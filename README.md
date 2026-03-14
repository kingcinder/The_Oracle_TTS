# DualVoice Studio

DualVoice Studio is a local PySide6 desktop app and CLI for turning a `.txt` or `.md` two-person dialogue into a single FLAC render with Chatterbox. The product is now Chatterbox-exclusive: the standard Chatterbox model is the default quality-first backend, with multilingual and turbo variants available inside the same engine family.

Chatterbox outputs include built-in Perth watermarking by design. This project does not remove or hide that.

## Features

- Chatterbox-only render path with `standard`, `multilingual`, and `turbo` variants
- Voice cloning via per-speaker reference clips passed through `audio_prompt_path`
- GUI review table for `[index | speaker | original text | repaired text | emotion | duration | preview]`
- Automatic normalization, punctuation restoration, spelling correction, and grammar cleanup
- Dual-speaker attribution with explicit markers, alternating-line fallback, clustering fallback, and anchor-ready plumbing
- Emotion inference mapped into live Chatterbox controls such as `cfg_weight`, `exaggeration`, and `temperature`
- Incremental stem caching keyed by repaired text, speaker, model variant, language, Chatterbox parameters, reference hash, and Chatterbox version
- FLAC export with metadata tags plus render plans and correction logs

## Install

Recommended engine setup:

```bash
./bootstrap_chatterbox_only.sh
source .engine-setup/chatterbox_env.sh
python scripts/doctor.py
```

If you want a project-local editable install in your own venv:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -e ".[ml,chatterbox]"
```

Optional model prefetch:

```bash
python scripts/download_models.py --variant all --device cpu
```

Deterministic repo smoke render:

```bash
python scripts/smoke_render.py
```

This smoke path is intentionally deterministic and uses a patched in-repo test engine so you can verify ingest, repair, caching, assembly, and FLAC export without depending on live Chatterbox generation. The real Chatterbox model smoke remains part of `./bootstrap_chatterbox_only.sh`.

## Run

GUI:

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
- A CUDA GPU is strongly recommended for production-length jobs.

## Product Notes

- Chatterbox standard is the default quality-first backend.
- Multilingual mode requires the multilingual Chatterbox variant and a real language code.
- Turbo is lower-latency and optional; it is not the default backend.
- Chatterbox’s Perth watermark is retained in output audio.

## Demo

Use [examples/demo_dialogue.md](/home/oem/Documents/The_Oracle_TTS/examples/demo_dialogue.md) with two short reference clips to validate the full pipeline before moving to longer material.

## Licensing

This project is publicly visible for inspection and evaluation, but it is not open-source.
All rights are reserved.

Commercial licensing and leasing inquiries: codysa90@gmail.com
