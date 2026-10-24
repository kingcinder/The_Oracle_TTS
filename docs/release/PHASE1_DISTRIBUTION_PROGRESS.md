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

## Evidence

- `/tmp/oracle-baseline-uV4W9R/bin/python -m pytest tests/test_distribution_locks.py tests/test_linux_release.py tests/test_offline_bundle.py -q`
  — **31 passed**.
- The lock test verifies `--require-hashes`, deterministic package ordering,
  normalized distribution names, and manifest SHA-256 values.
- The baseline fresh environment reported `pip check` clean and pip-audit
  `No known vulnerabilities found`.

## Remaining Phase 1 work

- The repository has no PyInstaller/AppImage/deb build path; `dpkg-deb` exists,
  but PyInstaller, AppImage tooling, and a frozen runtime recipe are absent.
- The current first-run flow accepts the EULA before opening the window, but
  activation is optional and model download ordering is not yet the required
  EULA → serial activation → model download sequence.
- Clean-machine installed-artifact acceptance, rollback, uninstall retention,
  static update manifest, GPU wheel download, and Windows experimental installer
  CI remain unverified or unimplemented.
