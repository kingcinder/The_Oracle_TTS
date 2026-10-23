# The Oracle TTS — Quick Start

This guide covers a first local render from a source checkout. An isolated
managed install and CPU render passed, but the complete first-user/project
workflow and human audio review remain open; this guide does not mean the
candidate is ready for general distribution.

## 1. Check the prerequisites

- Linux x86_64 is the alpha qualification target. Windows is unsupported for
  the alpha.
- Python 3.11 or 3.12 and `ffmpeg` on `PATH` are required.
- The first render needs an internet connection to fetch the pinned Chatterbox
  model, unless you install from a prepared offline bundle. Model files and
  renders require several gigabytes of disk space.

## 2. Install and check the machine

From the release checkout:

```bash
./oracle install
./oracle doctor
```

`doctor` reports missing dependencies, model readiness, and backend capability.
The default Chatterbox path runs on CPU. CUDA requires a supported NVIDIA GPU
and a CUDA PyTorch runtime. Vulkan requires the separately configured
audio.cpp backend.

## 3. Render a dialogue

1. Start the desktop app with `./oracle start`.
2. Open a UTF-8 `.txt` or `.md` dialogue with speaker labels, such as:

   ```text
   Speaker A: The train arrives in five minutes.
   Speaker B: I will wait by the east entrance.
   ```

3. Review the detected utterances and map each speaker to a voice reference.
   The bundled generic references are available in the voice picker.
4. Select the Chatterbox model variant and inference backend supported by the
   machine. Start with the default CPU backend if unsure.
5. Start rendering. Wait for export completion before using the output file.

The rendered dialogue is exported as FLAC. Save the project before closing if
you need to reopen its settings and utterance review state later.

## 4. Get help

Run `./oracle doctor` first. For a report, use **Help → Report a Problem…** or
follow [Support](../SUPPORT.md). Review the generated package before sharing;
it is never uploaded by the application.

For data preservation during an update, see [Upgrading](../UPGRADING.md).
