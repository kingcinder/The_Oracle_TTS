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
- Test suite: 416 passing tests including the new assembly/blend/monologue
  coverage.

## Next

- Re-render `Input/What is, reality.txt` on the Vulkan backend with generic
  voices as live verification of the assembly fix (hardware-dependent).

## Deferred (intentional)

- **Open investigation — GUI closes instantly on Render click (native
  crash)**: kernel-log `general protection fault` in libQt6Widgets at a
  data-region offset (execution through a corrupted function pointer →
  use-after-free signature) in the GUI process, same second as the
  `render_click` event. Not reproduced programmatically on current or
  crash-era code; blocked-on-repro. Repro launcher: `bash
  /tmp/gui_crash_catcher.sh` → click Render → backtrace lands in
  `/tmp/gui_crash_backtrace.log`. Findings: `.serpent-circle/04-debug/root-causes.md`.
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