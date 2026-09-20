# Troubleshooting The Oracle

Symptoms, causes, and fixes, ordered the way a real problem usually
presents: start with the doctor.

## First stop: the doctor

```bash
./oracle doctor                # or: python scripts/manage_install.py doctor
python scripts/doctor.py --json    # machine-readable report
```

The doctor verifies capabilities by executing them (entrypoint, engine
import and model init, Qt offscreen, deterministic smoke render, backend
probes), not by checking that files exist. `--ci` runs the same gate
without interactive model init, and is what CI uses. Its output is
read-only and idempotent: running it twice gives byte-identical reports
(timings are measurements, not facts, and are not printed in the human
report).

## Entrypoint: "the-oracle is not installed in <old path>"

The managed launcher at `~/.local/bin/the-oracle` (Linux) points at the
checkout it was installed from. If the repo moved, re-run the install step
to rewrite it:

```bash
./oracle install
```

On Windows the Start Menu / desktop shortcuts are rebuilt the same way.

## Render fails or the GUI can't load the engine

- `./oracle doctor` says which layer is broken: venv, dependencies, engine
  import, or model init.
- If dependencies are stale after an upgrade, `./oracle update` reinstalls
  them from `pyproject.toml`.
- If a specific model is missing from the local Hugging Face cache, a
  networked machine fetches it on first use; an offline install must get it
  from its bundle (`update --offline-bundle <dir>`).

## "setup-vulkan" refuses on an offline install

The Vulkan GGUF model is not part of the offline bundle, and an offline
install must never attempt a network fetch. Run `the-oracle setup-vulkan`
on a networked machine (deleting the `.oracle_offline` marker in the
install root re-enables network fetches), or re-install without
`--offline-bundle`.

## First render is slow / grammar falls back to local fixes

The first grammar pass may need the LanguageTool server (hundreds of MB).
The render never stalls on it: it falls back to local fixes and warms the
download in the background for the next run. Offline installs skip that
download entirely (the local fallback still applies). Check
`ORACLE_LANGUAGE_TOOL_TIMEOUT` if you want to bound the load attempt
differently (seconds; default 25).

## Input file flagged as misformatted, or a fix regretted

- `the-oracle check-input <file>` lints without changing anything;
  `--json` gives a machine-readable report, `--fix` corrects in place.
- Every fix keeps a timestamped backup next to the file. Settings →
  *Restore most recent input-file backup* undoes the latest one from the
  GUI.
- Typos are *content*, not *formatting*: `check-input` is spelling-blind
  by design; render-time text repair handles spelling.

## Subtitles

`.srt`/`.vtt` files are auto-detected and converted to dialogue scripts in
the CLI and GUI; an existing conversion is reused rather than overwritten.
If a subtitle file is unreadable in one encoding, the converter falls back
through utf-8-sig then cp1252; a file with no valid cues is loaded as-is
with a warning.

## Audio: no sound / device errors

- The doctor probes the audio stack; `import sounddevice` failing usually
  means the PortAudio system library is missing (Linux:
  `libportaudio2`).
- The Recording Studio needs a working input device; the doctor's qt and
  audio checks distinguish "no device" from "stack broken".

## GUI won't start on a headless machine

The GUI needs a display or the offscreen platform for tests/smokes:
`QT_QPA_PLATFORM=offscreen` is what CI and the smoke harnesses use.

## Vulkan backend reports no device

`./oracle doctor` runs `audiocpp_cli` to enumerate devices. No device
usually means the driver/loader layer, not the app: `vulkaninfo` is the
next diagnostic. The binary and model are set up by `the-oracle
setup-vulkan`; the CI job `vulkan-smoke` shows a known-good path.

## Windows launcher quirks

The `.cmd`/`.ps1` launchers quote the install path (this repo's own path
contains a space and is part of the install-boundary test suite). If a
launcher misbehaves after moving the checkout, re-run
`.\oracle.ps1 install` to regenerate it.

## Still stuck

`python scripts/doctor.py --json > report.json` attached to a bug report
is the most useful single artifact; `JUNO_FIXES.log` documents what changed
and why across the whole campaign.
