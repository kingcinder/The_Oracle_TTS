# Phase 1 Distribution Progress — 1.5.0 RC

## Implemented

- Offline bundle wheel sets now produce per-platform, per-Python lock files under
  `locks/`, with `--require-hashes`, `--no-index`, and SHA-256 for every wheel.
- The bundle manifest records each lock file and wheel digest. A bundle may
  contain Linux and Windows lock sets for the same Python target.
- Offline managed installs use the lock file when present, then install the
  local application editable with `--no-deps`; malformed legacy bundles retain
  the previous path so their failure mode remains explicit.
- Linux unsigned build SBOM generation now uses pip-audit's CycloneDX JSON
  output instead of a pip-licenses inventory. The output remains included in
  the SHA-256 manifest.
- `scripts/build_frozen_linux.py` now provides a PyInstaller one-folder recipe,
  AppImage and amd64 `.deb` staging, explicit model/signing-material exclusion,
  desktop entry/icon staging, unsigned SHA-256 output, and a static update
  manifest with manual rollback instructions. The Windows recipe is present in
  `packaging/windows/` and is labeled EXPERIMENTAL.

## Evidence

- `/tmp/oracle-baseline-uV4W9R/bin/python -m pytest tests/test_distribution_locks.py tests/test_linux_release.py tests/test_offline_bundle.py -q`
  — **31 passed**.
- The lock test verifies `--require-hashes`, deterministic package ordering,
  normalized distribution names, and manifest SHA-256 values.
- The baseline fresh environment reported `pip check` clean and pip-audit
  `No known vulnerabilities found`.

## Remaining Phase 1 work

- The frozen recipe has not run in this checkout: `pyinstaller` and
  `appimagetool` are not installed locally. The CI build job and a clean
  Ubuntu 24.04 installed-artifact acceptance are still required.
- The current first-run flow accepts the EULA before opening the window, but
  activation is optional and model download ordering is not yet the required
  EULA → serial activation → model download sequence.
- Clean-machine installed-artifact acceptance, rollback, uninstall retention,
  GPU wheel download, and Windows experimental installer CI remain unverified
  or unimplemented.

## Phase 2 progress started

- `scripts/build_audio_cpp.sh` now names the exact validated upstream commit
  `748c5e28f6a7228b8f38ad7142ca97d29584544b`, fetches/checks out that commit,
  and refuses an invalid or mismatched checkout. The test harness must opt in
  to `AUDIOCPP_SKIP_PIN_VERIFY=1`; production builds do not.
- The existing notices already list audio.cpp and ggml attribution and their
  license files. Prebuilt Vulkan binaries and end-to-end checksum download
  verification remain unimplemented.
