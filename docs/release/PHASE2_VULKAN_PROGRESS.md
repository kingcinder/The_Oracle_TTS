# Phase 2 Vulkan Progress — 1.5.0 RC

## Implemented

- `scripts/build_audio_cpp.sh` pins audio.cpp to
  `748c5e28f6a7228b8f38ad7142ca97d29584544b` and refuses a mismatched
  production checkout. The source build remains the documented advanced path.
- `scripts/package_vulkan_binary.py` packages the CI-built
  `audiocpp_cli` as an unsigned Linux x86_64 tarball and records the pinned
  commit, size, and SHA-256 in `audiocpp-vulkan-manifest.json`.
- `the_oracle.vulkan_prebuilt.install_from_manifest` verifies the archive hash,
  rejects path traversal and symlinks, and installs the executable only after
  verification. A changed archive is refused.
- `download_and_install` and `scripts/install_vulkan_prebuilt.py` fetch the
  static manifest before the archive, require the URL filename to be listed,
  and verify size plus SHA-256 before replacing the target binary.
- The non-PR Vulkan CI job now uploads the unsigned binary and manifest after
  its pinned build and no-skip smoke.

## Evidence

- `./.venv/bin/python -m pytest -q tests/test_vulkan_prebuilt.py` — **3 passed**.
- Existing pinned-build focused tests remain green in the prior Phase 2 run —
  **6 passed**.

## Remaining or unverified

- The hosted Vulkan job has not run on this commit in this session, so the
  real audio.cpp build, manifest upload, and checksum installation remain
  CI evidence rather than local evidence.
- The GUI does not yet fetch a remote prebuilt archive automatically; source
  build remains the active setup path. Model download progress and offline
  Vulkan model acceptance remain unverified.
- Windows Vulkan remains experimental pending Phase 6 hardware qualification.

## Owner decisions needed

- Supply the publication location and later signing keys for the unsigned
  binary/manifest. No production key or signing seed was created.
