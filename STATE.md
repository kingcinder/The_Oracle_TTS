# The Oracle — State (completeness manifest)

This file is the repo's self-designated completeness record. It is the
authoritative context for the Omega meta-skill loop (searched, never
rewritten by the loop).

## Done

- Chatterbox-only render pipeline: `standard`, `multilingual`, `turbo`
  variants on PyTorch (CPU) and an opt-in Vulkan backend via audio.cpp
  (AMD RDNA1-class GPUs; vendored RDNA1 device-lost fix).
- Cross-platform bootstrap/install/doctor/run/uninstall for Linux and
  Windows; managed launcher + desktop integration.
- Desktop GUI: review table, per-row preview, repair, live progress panel,
  profiles/templates, saved project manifests, Ctrl+hover help.
- CLI render flow with saved project manifests and deterministic smoke
  render path for repo-local verification.
- Batched Vulkan rendering (`--request-sequence`, bounded 32-request
  groups, live per-request progress) with truthful timing logs.
- Voice catalog: bundled generic voices (`Seashells/generic/` — 5
  English + 4 Chinese references, Apache-2.0 / CC BY 4.0 attributed,
  English-first in the picker), curated `Seashells/` defaults, recent
  custom clips.
- Voice blending: deterministic derived reference from two clips with a
  preference weight and mix/alternate/layer modes; persisted in projects
  and profiles; honored by both backends.
- Saved blend voices: **Save Blend As...** creates a named voice that
  appears under "Saved Blends" in the picker (catalog
  `Profiles/blend_voices.json`, derived clips `Profiles/.blends/`).
- Correction Mode **Verbatim (no changes)**: true passthrough of the
  source text (no spelling/grammar/punctuation edits).
- Fidelity fixes (commit `d77c278`): assembly no longer drops chunks of
  chunked utterances (parallel loader keyed by position, regression-tested);
  monologue renders no longer require/construct an unused speaker profile.
- Test suite: 454 passing tests on the `feature/voice-craft-and-
  recording-studio` tree (416 on `main`) including the new assembly/
  blend/monologue/pacing/parity/recorder coverage.
- Voice-craft + Recording Studio (branch
  `feature/voice-craft-and-recording-studio`, **uncommitted pending review**):
  punctuation-aware pacing (turn pause scaled by terminal punctuation;
  chunk seams breathe ~35%, the full turn pause applies only after an
  utterance's final chunk), timbre-locked emotion (per-line emotion moves
  emphasis/pause only; temperature/CFG stay per-speaker so the voice
  character is consistent), perceptual voice sliders (Voice Lock, Emotional
  Punch, Delivery Variety, Emotion Strength, Human Drift, Breath After This
  Speaker, Which Voice Wins), and the Custom Voice Recording Studio window
  (teleprompter fed from `Input/`, mic + supported-sample-rate pickers,
  auto-incrementing `Seashell_No_x.wav` saving into `Seashells/`, live
  level meter, auto-audition of each take, Listen again, Use for Speaker
  A/B quick-assign that refreshes the voice pickers on close).
- **CPU (PyTorch) ⇄ Vulkan parity** for the voice-craft decisions:
  `tests/test_backend_parity.py` plans a mixed-punctuation dialogue with
  `inference_backend=pytorch` vs `vulkan` and asserts per-utterance pauses
  and engine settings are identical on both, with timbre-lock holding (one
  temperature per speaker); the Vulkan/engine-path suites pass
  (84 tests in the batching/backend/synthesis/render-worker files). Pacing
  and emotion live in shared plan/assembly code, never in the engine call,
  so both inference backends behave identically by construction.

## Next

- Re-render `Input/What is, reality.txt` on the Vulkan backend with generic
  voices as live verification of the assembly fix (hardware-dependent).

## Noticed, not yet actioned

- Serpent-circle inventory scan updated (2026-09-08, in
  `~/.agents/skills/serpent-circle/`): the bloat scan and language
  histogram now honor `.gitignore` — untracked+gitignored residue
  (bytecode, `.venv/`, vendored clones) is treated as repo-declared
  retention, while tracked junk and non-ignored strays stay flagged. This
  lets the omega loop terminate **CONVERGED** instead of NO-PROGRESS on a
  fresh campaign; the change lives in the skill (outside this repo), so
  it is documented here rather than committed here.

## Deferred (intentional)

- **GUI native crash — root cause found & fixed (2026-09-08)**: kernel-log
  `python: segfault at 0 ip 0000000000000000` (execution through a
  null/corrupted function pointer → use-after-free signature). Two lifetime
  bugs in the Recording Studio (added by the voice-craft campaign) matched
  the signature exactly and were fixed:
  1. `_stop_playback` called `stop()` + `deleteLater()` on the audition
     `QMediaPlayer` from inside its own `mediaStatusChanged` handler —
     tearing the QtMultimedia FFmpeg backend down mid-emission is the
     canonical player use-after-free. Fixed: one persistent player per
     dialog (lazy, created on first audition, reused like MainWindow's
     preview player); EndOfMedia now defers the stop via a zero-timer and
     never deletes the player.
  2. The `RecordStudioWorker` QThread was `deleteLater`'d from the
     `captured`/`failed` slots (fired from `run()`'s final lines while the
     thread still exits) and `closeEvent` never joined it — destroying a
     live QThread is a hard Qt abort. Fixed: teardown moved to the
     `finished` handler; dialog close and MainWindow close now do a bounded
     `wait()` and refuse to close if the thread cannot stop.
  3. Same deferred-stop discipline applied to MainWindow's preview player:
     EndOfMedia defers the stop via a zero-timer (never stop/delete from
     inside `mediaStatusChanged`), one persistent player is reused across
     previews, and window close stops the player before teardown.
  Regression tests: `tests/test_recording_studio.py` (EndOfMedia handler
  safety, single-player reuse, finished-based worker teardown, close waits
  for worker, MainWindow preview-player deferral and close-stop). The pre-voice-craft Render-click crash report remains
  separate and still blocked-on-repro; repro launcher: `bash
  /tmp/gui_crash_catcher.sh`. Findings: `.serpent-circle/04-debug/root-causes.md`.
- **audio.cpp punctuation normalization is NOT patched**: its replacement
  table (`:`→`,`, `;`→`, `, dashes, quotes) exactly mirrors the installed
  Chatterbox Python reference `punc_norm`, so diverging would reduce
  model fidelity, not improve it. "Inflection at punctuation" is canonical
  model behavior, identical on both backends.
- `turbo` variant on Vulkan (PyTorch-only by design, rejected clearly).
- Additional GPU paths beyond Vulkan/CPU (CUDA/ROCm) — out of scope for
  RDNA1 hardware.
- `Seashells/` runtime caches and `Output/` renders stay gitignored;
  `.venv/` bytecode is documented retention (see `.omega/CHANGELIST.md`).
- Recording Studio extras (deferred): a waveform/trim editor for cropping a
  take before saving, punch-in/overdub re-recording from a chosen prompt
  line while keeping earlier audio, and per-Seashell metadata sidecars
  (recorded-in sample rate, speaker tag) surfaced in the voice picker.