# System Requirements and Qualification

This describes the 1.4.2 limited-release candidate. A listed target is not
qualified until the install and real-render checks in the readiness report
have passed.

| Area | Candidate requirement | Evidence / status |
|---|---|---|
| Operating system | Linux x86_64 | Intended alpha target; isolated managed install and CPU render passed, but the complete new-user/project workflow is not qualified. |
| Windows | Not supported for the alpha | Installer scripts exist, but Windows GUI certification remains open and the current remote CI run failed. |
| Python | 3.11 or 3.12 | Enforced by the managed installer. |
| Media tools | `ffmpeg` on `PATH` | Local Linux doctor found `/usr/bin/ffmpeg`. |
| CPU inference | PyTorch Chatterbox CPU path | Standard Chatterbox initialized and produced a valid two-speaker FLAC from the isolated managed install. Human listening/intelligibility review remains outstanding. |
| NVIDIA GPU | CUDA PyTorch runtime; doctor documents a 4 GiB Chatterbox VRAM minimum | Not verified here: the installed runtime is CPU-only and the Quadro K600 has 1 GiB VRAM. |
| Vulkan GPU | Separately configured audio.cpp backend | Local doctor found a built backend and AMD Radeon RX 5700 XT Vulkan device; real production rendering is unverified. |
| RAM | Not established as a minimum | Measure peak use on qualified hardware before publishing a numeric minimum. |
| Disk | Several GB for model files, plus project and render space | Model set and first-run download can be multi-GB; a precise minimum has not been certified. |
| Network | Needed for first-run model downloads; not needed for a correctly provisioned offline install | Model download sends ordinary connection metadata to its provider; no telemetry is configured. |

For installation and a first render, see [Quick Start](QUICK_START.md). For
offline installation and updates, use the existing
[soft-launch runbook](../SOFT_LAUNCH.md) and
[upgrade guide](../UPGRADING.md).
