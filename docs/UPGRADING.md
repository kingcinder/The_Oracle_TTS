# Upgrading The Oracle

How to move an existing install to a new version, and what the upgrade
touches and preserves.

## The short version

From the repository root:

```bash
git pull            # or otherwise bring the checkout to the new version
./oracle update     # refresh dependencies and launchers, keep your data
./oracle doctor     # optional: verify the upgraded install end to end
```

On Windows:

```powershell
git pull
.\oracle.ps1 update
.\oracle.ps1 doctor
```

`update` reinstalls the Python dependencies from `pyproject.toml` (picking
up any changes), rebuilds the managed launchers, and keeps **all user data**:
`Input/`, `Seashells/`, `Profiles/`, `Output/`, and the app settings file.

## What you are running

- The version number lives in one place (`src/the_oracle/__init__.py`);
  `the-oracle --version` prints it.
- What changed between your version and the new one is in
  [`CHANGELOG.md`](../CHANGELOG.md).

## Switching the compute runtime

The PyTorch runtime choice (CPU vs CUDA wheels) is an install-time decision
that `update` can change in place:

```bash
./oracle update --pytorch-runtime cuda   # switch an existing install to CUDA
./oracle update --pytorch-runtime cpu    # ...and back
```

CUDA requires a compatible NVIDIA GPU and driver; the doctor reports what
the machine actually supports. The Vulkan (audio.cpp) backend is separate:
set it up once with `the-oracle setup-vulkan` on a networked machine.

## Offline installs

An offline install (created from a bundle with
`--offline-bundle`) keeps working through `update` the same way: run
`update --offline-bundle <dir>` with the bundle it was installed from, and
the update reinstalls from the bundled wheels without touching the network.

The `.oracle_offline` marker in the install root is what forces every model
load to resolve locally. Deleting it re-enables network model fetches (for
example, to run `the-oracle setup-vulkan`); the offline installers and
launchers re-establish the environment on the next start.

## If `update` reports no existing install

`update` on a checkout without a `.venv` forwards to a full `install`
automatically — the same command is safe on both a fresh clone and an
existing install.

## What upgrade does not do

- It never touches your renders, recordings, reference clips, or dialogue
  files.
- It does not change the input-format backups the transformer/fix flows
  keep next to your files.
- It does not push anything anywhere; this project has no telemetry.

See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) when something goes wrong
during or after an upgrade — `doctor` is the first stop and its `--json`
report is machine-readable.
