# Phase 6 Windows and GPU Progress — 1.5.0 RC

## Implemented

- The Windows CI matrix already runs the full six-theme certification and a
  separate ten-run loop with no retries. The workflow now also names the
  50-cycle real-window Ctrl-hover lifecycle regression explicitly.
- Windows packaging remains labeled `EXPERIMENTAL` in the unsigned Inno Setup
  recipe. CUDA and Vulkan remain opt-in/experimental paths; CPU is the default.

## Evidence

- The lifecycle regression exists as
  `test_real_main_window_churn_leaves_no_filters_behind` and executes 50 real
  build/destroy cycles while asserting no live filters remain.
- This session has not run Windows CI, the ten-run six-theme loop, Windows
  doctor/offline install, or matching CUDA/Vulkan hardware acceptance.

## Remaining or unverified

- Hosted Windows evidence must be linked from the exact release commit before
  Windows can be considered qualified. Until then the installer is not a
  supported release artifact.
- A Windows 11 installed-artifact acceptance and SmartScreen observation remain
  pending. GPU qualification requires matching hardware; otherwise the UI,
  docs, and EULA must retain experimental wording.

## Owner decisions needed

- Decide whether to qualify Windows and GPU after the hosted acceptance runs.
  The current default is to keep both experimental/unsupported until evidence
  meets the stated gates.
