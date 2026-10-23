# Real Audio Acceptance Record — 1.4.2 Candidate

## Isolated managed-install run — 2026-10-23

- Environment: isolated Linux x86_64 checkout at
  `/media/cody/Data/oracle-clean-install-YvsR8r/tree`, Python 3.12 managed
  `.venv`, CPU PyTorch 2.6.0; installation was provisioned with
  `scripts/manage_install.py bootstrap --pytorch-runtime cpu`. This is a
  disposable test tree, not the canonical checkout.
- Dependency consistency: `pip check` — **No broken requirements found**.
  The run exposed and corrected the project dependency bounds: Chatterbox
  0.1.7 requires NumPy `<2`, while the then-current `ml-dtypes` required NumPy
  `>=2`. The candidate now constrains `numpy>=1.26,<2` and `ml-dtypes<0.6`.
- Doctor: normal managed-install doctor exited 0; dependency pins 18/18,
  Chatterbox CPU initialization, Perth, Qt, and deterministic smoke passed.
  Optional Turbo is absent. This does not qualify CUDA or Vulkan production
  inference.
- Engine/corpus: standard Chatterbox, CPU, 12-utterance two-speaker dialogue,
  bundled `english_male_1.wav` and `english_reader_1.wav` references.
- Output: `/media/cody/Data/oracle-clean-install-YvsR8r/real-render/real_engine_smoke.flac`.
- Runtime: 249.356 seconds.
- Decode: mono FLAC, 24,000 Hz, 1,128,480 frames (47.02 seconds); readable by
  `ffprobe` and SoundFile. File size 1,076,521 bytes.
- Peak: 0.9800109863; zero samples at or above 0.9999.
- SHA-256: `1db8d6806f5c58a996b9e02f2dad43f1674ec00132dbbc965732c47a08d255da`.
- Dependency audit: `pip-audit --path` reported no known vulnerabilities in
  the auditable installed set; local project, Perth, and CPU PyTorch packages
  were skipped.

This demonstrates a real render from an isolated managed installation, but
the install was not a complete new-user walkthrough: restart, project save and
reopen, update, rollback, and uninstall remain unverified. The test tree had
an existing model cache available, so a cold model download was not measured.

## Automated evidence

- Engine: standard Chatterbox, CPU, `light` loudness preset; no Gradio dependency.
- Corpus: 12 utterances in a two-speaker dialogue, using the bundled
  `english_male_1.wav` and `english_reader_1.wav` references. The copied inputs
  byte-match the repository assets.
- Result: `/tmp/oracle-release-acceptance-no-gradio/real_engine_smoke.flac`
- Runtime: 243.06 seconds.
- Decode: mono FLAC, 24,000 Hz, 1,085,280 frames (45.22 seconds); readable by
  both `ffprobe` and SoundFile.
- Peak: 0.9800109863; no sample reached 0.9999. Active-frame RMS coefficient
  of variation: 1.008385.
- SHA-256: `f3ea3671c8ecc290e14fcc2a032f16fc058fcf09692963c591f62656034e6936`

## Acceptance status

**UNVERIFIED — human listening and intelligibility.** The automated harness
checks synthesis completion, valid decode, output shape, and non-stationary
energy. Its energy-variation check is only a rough proxy; it does not establish
that the words are intelligible, voices sound natural, speaker order is right,
or transitions are suitable for customers. No ASR model was available, and no
human listening review was recorded.

**UNVERIFIED — GPU production paths.** CUDA was unavailable on this CPU-only
PyTorch runtime. The installed Quadro K600 has 1 GiB VRAM, below the documented
4 GiB Chatterbox suitability floor. An AMD RX 5700 XT Vulkan/audio.cpp device
and native binary were detected, but no production render was qualified on it.

## Required next evidence

A human reviewer should listen to the linked output and record: intelligibility
of each utterance, correct speaker assignment/order, transitions and pauses,
unwanted clipping or silence, and whether the result is usable for the stated
alpha use case. Record reviewer, date, candidate commit, and any defects. Do not
replace this record with the automated envelope check.

The smoke output is a temporary acceptance artifact, not a release package or
a redistributable corpus. No model or voice reference is added to the release
by this record.
