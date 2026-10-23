# The Oracle 1.4.1 — Commercial Alpha Soft Launch

Status: **paid alpha release candidate**. Linux is supported; Windows is
**unsupported** for this alpha. The commercial gate (`python scripts/soft_launch.py --check`)
is the authority on whether a tree may be distributed. A red gate means the
tree is not approved for commercial distribution, even if the app installs and
runs.

## What this alpha is

A narrow, early-access release for users who want local, offline dialogue
rendering and are willing to report problems. It is not a general-availability
release: the Windows GUI crash certification item is still open and the EULA
is awaiting the owner's governing-law decision and counsel review. The three
strong-copyleft runtime dependencies from the 1.4.0 candidate are gone.

Community remains everything that shipped before 1.4.0. A missing or expired
license never locks a workflow; it reports the community edition and keeps
working.

## System requirements

- Linux x86_64 (supported) or Windows x86_64 (beta).
- Python 3.11 or 3.12.
- `ffmpeg` on `PATH`.
- Git, to fetch the release branch and updates.
- Disk space for the pinned models (several GB) and renders.
- Optional: an NVIDIA GPU with a suitable PyTorch/CUDA runtime, or a
  Vulkan-capable GPU for the opt-in audio.cpp backend.

## Install

```bash
git clone --branch release/1.4.0 https://github.com/kingcinder/The_Oracle_TTS.git
cd The_Oracle_TTS
./oracle install
./oracle start
```

On Windows:

```powershell
git clone --branch release/1.4.0 https://github.com/kingcinder/The_Oracle_TTS.git
cd The_Oracle_TTS
.\oracle.ps1 install
.\oracle.ps1 start
```

`./oracle install` creates the venv, installs the pinned runtime, registers the
launcher, and runs the doctor. `./oracle doctor` is the first stop for any
problem. The doctor's `--json` report is machine-readable.

## Purchase and activation

The soft launch is sold directly while the storefront is being set up:

1. Email **codysa90@gmail.com** with the subject **"Oracle alpha license"** and
   the name you want recorded on the license (optional).
2. You receive a single `ORACLE1.…` token. The token is the license; there is no
   account and no activation server.
3. Activate offline:

   ```bash
   the-oracle activate "<token>"
   the-oracle license-status
   ```

   Or use **Help → Activate License…** in the GUI.

The license file contains no personal data unless you typed a licensee name.
Activation never contacts a server. Removing the local token file moves the
seat; support can re-issue a token from the purchase record.

The vendor mints tokens with `scripts/license_sign.py mint --key-id k2`. The
signing seed never leaves the owner's secret manager. See
[LICENSING_OPS.md](LICENSING_OPS.md) for the custody and rotation procedure.

## Support

Compose a bug report, package it with the redacted diagnostics, and attach the
zip to the issue or email you send:

```bash
the-oracle bug-report new --title "..." --summary "..." --step "..."
the-oracle bug-report show <report-id>       # read exactly what will be shared
the-oracle bug-report package <report-id>    # Output/oracle-report-<id>.zip
the-oracle bug-report issue <report-id>      # prefilled issue + email links
```

In the GUI this is **Help → Report a Problem…**. Drafts stay in
`support_reports/` (newest 25); nothing is uploaded, and deleting one is
explicit. The full workflow is in [SUPPORT.md](SUPPORT.md).

For a bundle without a written report, `the-oracle support-bundle` still works:

```bash
the-oracle support-bundle
```

The bundle is written under `Output/` and is never uploaded. It contains the
app version and platform, whether `ORACLE_*` variables are present (not their
values), the license state (never the token or licensee), redacted settings,
the newest redacted logs, and a doctor report. Crash reports are included only
with `--include-crash-reports` **and** when local crash-reporting consent is
on.

The bundle never includes dialogue files, manuscripts, rendered audio, voice
recordings, reference clips, license tokens, or home-directory paths. The
report package adds `report.md`, `report.json`, and `diagnostics.json` to the
same bundle. Read the zip before sending it; redaction is best-effort.

For a render problem, include the project's `render_plan.json` and the exact
input text only if you are comfortable sharing them. They are never collected
automatically.

## Privacy

There is no telemetry, no analytics, no crash upload, and no phone-home. See
[PRIVACY.md](../PRIVACY.md). The licensing and crash packages are pinned to
import no network module at all.

## Updates

Updates are manual for the alpha:

```bash
git pull
./oracle update
./oracle doctor
```

`./oracle update` keeps `Input/`, `Seashells/`, `Profiles/`, `Output/`, and
your settings. The [CHANGELOG](../CHANGELOG.md) is the source of truth for what
changed.

## Known limitations

- **Windows is unsupported in this alpha.** The six-theme GUI certification and the Windows
  certifier crash investigation (native exception 0xC0000005) are not closed.
  Paid seats should expect Linux first; Windows support is best-effort during
  the alpha.
- **First run downloads models.** The first render may need a multi-GB model
  download; the doctor announces it before it starts. Offline installs use the
  bundle instead.
- **Strong-copyleft dependencies are removed.** FLAC tagging is native,
  grammar repair uses the local fixes only, and Japanese token conversion is
  skipped by `chatterbox-tts` when `pykakasi` is absent. English and
  non-Japanese multilingual rendering are unchanged. There is no
  strong-copyleft review item left in
  [`scripts/license_review.json`](../scripts/license_review.json).
- **The EULA is a draft.** `EULA-DRAFT.md` still needs a governing-law/venue
  section and counsel review. The gate refuses distribution until it is
  renamed to `EULA.md` and finalized.
- **Studio features are additive.** `render-report` is the first Studio-gated
  command; community keeps every workflow that existed before it.

## Owner go-live checklist

Run the gate, then clear each blocker:

```bash
.venv/bin/python scripts/soft_launch.py --check
```

| Blocker | Owner action |
|---|---|
| `eula` | Complete the governing-law/venue section, rename `EULA-DRAFT.md` to `EULA.md`, remove the DRAFT marker, and have counsel review it. |
| `third_party_notices` | Regenerate with `python scripts/generate_third_party_notices.py --check` and commit the diff. |
| `voice_assets` | Keep `Seashells/ASSETS.md` truthful; every shipped `.wav` needs a source and license. |
| `version` | Keep `__version__`, the README banner, and `CHANGELOG.md` in agreement (`scripts/release.py --check`). |
| `checksums` | Build with `scripts/release.py` on a clean tree and commit `release_checksums/checksums-<version>.sha256`. |
| `windows_scope` | Keep this runbook's Linux-supported / Windows-experimental statement. |
| `support_bundle` | Keep `the_oracle.support_bundle` importable. |

Warnings (weak-copyleft LGPL/MPL packages such as PySide6, certifi, `orjson`,
`soxr`, and `tqdm`) do not block the gate, but their notice and
replacement/source obligations must be honored before distribution. The LGPL
obligation for PySide6/Qt is spelled out in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

A green gate means the tracked tree has no known commercial blocker. It does
not replace counsel's review of the EULA or the dependency decisions.
