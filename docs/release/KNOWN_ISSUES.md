# Known Issues — 1.4.2 Limited-Release Candidate

This is a candidate issue list, not a claim that the release has passed all
acceptance gates.

- **Human audio review is outstanding.** Automated CPU synthesis produced a
  valid 24 kHz mono FLAC, but intelligibility and audible quality have not been
  reviewed. See [Real Audio Acceptance](REAL_AUDIO_ACCEPTANCE.md).
- **Windows is unsupported for this alpha.** Its GUI certification and native
  crash investigation are incomplete.
- **GPU acceleration is not generally qualified.** CUDA needs a suitable
  NVIDIA card and matching PyTorch runtime. Vulkan/audio.cpp was detected on
  one AMD host, but real production rendering there remains unverified.
- **First model use requires a large download** unless a correctly prepared
  offline bundle is used. The provider receives ordinary connection metadata.
- **Japanese conversion is not included in the managed installation.**
  Japanese text normalization may therefore be limited. English and other
  supported multilingual paths were not exhaustively qualified in this pass.
- **The EULA is still a draft.** Commercial distribution is blocked pending
  owner and counsel review; no purchase or pricing terms are approved here.
- **Dependency review is scoped.** `pip-audit` found no known vulnerabilities
  in its auditable installed dependency set. Locally sourced or non-PyPI
  packages were skipped and are not thereby proven free of vulnerabilities.
- **Complete installation and update qualification are outstanding.** An
  isolated managed Linux installation and first CPU render passed, but cold
  model download, project save/reopen, update, rollback, and uninstall were
  not tested.
- **No hosted CI result exists for the modified working-tree candidate.** The
  latest remote result refers to an older commit.

Report a reproducible issue with the in-app **Help → Report a Problem…** flow
or follow [Support](../SUPPORT.md). Review diagnostics before sharing them.
