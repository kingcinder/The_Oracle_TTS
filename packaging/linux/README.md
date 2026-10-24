# Linux frozen rehearsal build

Run on a clean Linux x86_64 builder:

```sh
python scripts/build_frozen_linux.py --outdir release_artifacts/linux-frozen
```

The builder requires PyInstaller, `dpkg-deb`, and `appimagetool`. It emits an
unsigned AppImage, an unsigned amd64 `.deb`, `checksums-<version>.sha256`, and
`update-manifest.json`. The update manifest is static and is intended for a
manual check; no background update or telemetry is implemented. Rollback is a
manual reinstall of a previously verified artifact.

The PyInstaller output contains the application runtime and CPU dependencies.
It does not contain model weights, license signing material, or user data.
The application must show the EULA first; model acquisition remains behind the
serial-number gate. The AppImage and package keep per-user settings and model
caches under `~/.config/the-oracle` and `~/.cache`; uninstall removes the
installed application tree and leaves those paths in place.

The owner supplies signing keys later. Signing, publication, and clean-machine
acceptance are release-phase steps and are not performed by this recipe.
