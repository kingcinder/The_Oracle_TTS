# Windows 11 debugging VM for The Oracle

A local QEMU/KVM Windows 11 Pro box, provisioned from
`WIN11.PRO.25H2.U5-2.X64.(WPE).ISO`, used to debug and optimise the Windows
port. Everything lives outside the repo:

| Path | What it is |
| --- | --- |
| `/media/cody/Ark/oracle-win11-vm/` | VM disk, ISOs, screenshots |
| `~/.oracle-win11-vm/run-vm.sh` | start / stop / screenshot / monitor / ssh |
| `~/.oracle-win11-vm/credentials.txt` | local admin password (RDP) |
| `~/.ssh/oracle-win11-vm` | SSH key that gets installed in the guest |

Guest: Windows 11 Pro (compact image), 8 vCPU, 8 GB RAM, 64 GB disk,
virtual TPM 2.0, Secure Boot, SSH on **127.0.0.1:2222** (RDP is forwarded on
3389). The `oracle` account is a local Administrator and auto-logs-in.

## Day-to-day

```bash
~/.oracle-win11-vm/run-vm.sh start run     # boot the installed disk
~/.oracle-win11-vm/run-vm.sh status        # pid, disk, SSH port
~/.oracle-win11-vm/run-vm.sh ssh           # interactive PowerShell as oracle
~/.oracle-win11-vm/run-vm.sh screenshot x  # screenshots/x.png via QEMU monitor
~/.oracle-win11-vm/run-vm.sh stop          # ACPI shutdown, then kill
```

Non-interactive use:

```bash
ssh -i ~/.ssh/oracle-win11-vm -p 2222 \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    oracle@127.0.0.1 'python --version'
```

The default SSH shell is Windows PowerShell 5.1. Long installs should be run
as a background host process (`ssh ... > /tmp/job.log 2>&1 &`) — a remote
process started with `Start-Process` dies when the SSH session closes.

## What the guest already has

`C:\Windows\Setup\Scripts\setup-oracle.ps1` runs at first logon (payload
comes from the `ORACLEUNATTEND` ISO, log at `C:\oracle-provision.log`):

- Python 3.12 (all-users, on PATH), Git for Windows, ffmpeg `C:\tools\ffmpeg`,
  VC++ 2015-2022 runtime
- Win32-OpenSSH server, service `sshd` = Automatic, key auth for `oracle`,
  PowerShell as the default shell
- Long paths enabled, high-performance power plan, no sleep/hibernate,
  Remote Desktop enabled, Defender exclusions for build/model/pip paths

The repo is staged at `C:\Oracle` with a `.venv`:

```powershell
cd C:\Oracle
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\doctor.py --json
```

The venv carries the full declared runtime, ML and dev set, installed with the
installer a Windows user would run:

```powershell
.\.venv\Scripts\python.exe scripts\manage_install.py bootstrap --include-dev --pytorch-runtime cpu --skip-doctor
```

That yields `torch 2.6.0+cpu`, `torchaudio 2.6.0`, `transformers 5.20.0`,
`chatterbox-tts 0.1.7` (metadata patched to the audited pins by the installer),
`resemble-perth` at the pinned commit, `librosa 0.11.0`, plus the dev tools
(`pytest`, `pytest-xdist`, `pip-audit`, `detect-secrets`, `pip-licenses`).

Model weights are **seeded into the HF hub cache on the host and copied in**, not
downloaded from the guest. Measured on this link: guest PyPI/HF downloads ran at
30-340 KB/s (and stalled outright for minutes at a time), while a 4-way
parallel-range host download reached ~1.5 MB/s and `scp` into the guest measured
74 MB/s (1 GB in 13.6 s).

| Repo (pinned revision) | Size | Cache dir under `~/.cache/huggingface/hub` |
| --- | --- | --- |
| `ResembleAI/chatterbox` @ `5bb1f6ee` (standard + multilingual files) | 6.4 GB | `models--ResembleAI--chatterbox` |
| `oliverguhr/fullstop-punctuation-multilang-large` @ `345e80ad` | 2.2 GB | `models--oliverguhr--fullstop-punctuation-multilang-large` |

The cache is a plain `blobs/<sha256>` + `snapshots/<rev>/<file>` + `refs/main`
tree, so it can be rebuilt from flat files (a `/` in a repo path is stored as
`__`). VM tooling, kept outside the repo next to `run-vm.sh`:

```bash
# host: pull a pinned repo's files (validates HTTP 206 + every part's size)
python3 ~/.oracle-win11-vm/hf-pull.py <repo_id> <revision> /tmp/pull '*.json' 'model.safetensors'
# guest: rebuild the hub cache from the copied files
python C:\Users\oracle\seed_hf_cache.py C:\Users\oracle\hfseed <repo_id> <revision>
```

Verified with `huggingface_hub` 1.21.0: `snapshot_download(...
local_files_only=True)` and every `from_pretrained` resolve out of the seeded
tree, and the real-engine smoke renders with `HF_HUB_OFFLINE=1` set.

## Debug loop

1. Edit on the host, then push just what changed:

   ```bash
   rsync -az --delete --exclude .venv --exclude .git --exclude Output \
       -e 'ssh -i ~/.ssh/oracle-win11-vm -p 2222 -o StrictHostKeyChecking=no' \
       "src/" oracle@127.0.0.1:C:/Oracle/src/
   ```

2. Run the failing test with `-x -vv` over SSH.
3. GUI problems: `run-vm.sh screenshot` catches the current framebuffer; for
   interactive poking install FreeRDP on the host and run
   `xfreerdp /v:127.0.0.1:3389 /u:oracle /p:<password from ~/.oracle-win11-vm/credentials.txt>`.
4. Crash/exit codes: `Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000,1001} -MaxEvents 5`
   and `%LOCALAPPDATA%\CrashDumps` (set
   `HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps` if you
   want full dumps).
5. Timing: `.\.venv\Scripts\python.exe -m cProfile -o prof.out -m pytest -q tests/...`
   then `python -c "import pstats;pstats.Stats('prof.out').sort_stats('cumtime').print_stats(25)"`.

## How the install was produced (only needed to rebuild)

1. `bootmedia/` holds a vanilla WinPE (`boot.wim` image 1 from the ISO) plus
   `startnet.cmd` that finds the install media, partitions GPT, runs
   `dism /Apply-Image /Index:2`, writes boot files with `bcdboot`, drops the
   answer file at `C:\Windows\Panther\unattend.xml`, and shuts down.
   `xorriso` packs it into `boot.iso`.
2. `answer/` (autounattend + provisioning script + installers) is packed into
   `answer.iso` with volume label `ORACLEUNATTEND`.
3. `run-vm.sh start install` boots `boot.iso` once; after the guest shuts
   down, `run-vm.sh start run` boots the disk and Windows finishes
   specialise/OOBE and provisons itself.

Rebuild the boot media after editing `startnet.cmd`:

```bash
printf 'add %s /Windows/System32/startnet.cmd\n' ~/.oracle-win11-vm/startnet.cmd |
  wimlib-imagex update /media/cody/Ark/oracle-win11-vm/bootmedia/sources/boot.wim 1
xorriso -as mkisofs -iso-level 3 -J -joliet-long -V ORACLEWINPE \
  -b boot/etfsboot.com -no-emul-boot -boot-load-size 8 -boot-info-table \
  -eltorito-alt-boot -e efi/microsoft/boot/efisys_noprompt.bin -no-emul-boot \
  -isohybrid-gpt-basdat \
  -o /media/cody/Ark/oracle-win11-vm/boot.iso /media/cody/Ark/oracle-win11-vm/bootmedia
```

## Gotchas

- The ISO's own `setup.exe` is a Ghost Spectre launcher that hangs in a plain
  WinPE ("Setup is starting" forever). Do not try to drive Windows Setup from
  it — the DISM path above is the working one.
- Windows' execution policy is `Restricted` on this image. Run scripts with
  `powershell -ExecutionPolicy Bypass -File ...` or
  `Set-ExecutionPolicy Bypass -Scope Process -Force` in the session.
- A non-elevated SSH session cannot write `C:\` or install services. Provision
  from an elevated shell (Win+X, then `A`) or run the script with an
  administrator token.
- Hot-swapping `answer.iso` with the QEMU monitor leaves Windows showing a
  stale/empty CD volume until the media is ejected and re-inserted
  (`eject cd2`, `change cd2 <path>`); re-inserting is enough, a reboot is not
  required.
- The VM disk is `win11.qcow2`. Internal QEMU snapshots do **not** work here
  (`savevm` refuses because the OVMF `pflash1` vars drive isn't snapshot-able),
  so take a file-level copy while the VM is stopped:
  `run-vm.sh stop && cp --sparse=always win11.qcow2 win11-baseline.qcow2`.
  To roll back, stop the VM and copy the baseline back. The copy is ~11 GB and
  takes several minutes on the external drive; run it in the background.

## Measured Windows baseline (2026-10-20, full ML stack)

`pytest -q` on the guest, Python 3.12.10, chatterbox/perth/transformers installed
and every pinned model seeded in the HF cache:

```
8 failed, 1987 passed, 47 skipped in 855.96s (0:14:15)
```

The earlier run on the same guest, before the ML stack was installed, was
`17 failed, 1978 passed, 47 skipped in 335.81s`. The nine that went green are
the ML ones: `test_synthesis_task` seed pair, `test_backend_parity`'s pause
parity, `test_app_gui_srt::test_render_child_environment_uses_the_managed_runtime`,
`test_offline_guarantee::test_seeded_cache_resolves_every_loader_without_network`,
the three `test_doctor_dependency_pins` cases, and
`test_release::test_build_snippet_binds_outdir_before_setuptools_mutates_argv`
(it needed the build tooling the dev extra supplies).

`scripts/doctor.py` (full run, model init included): `python`, `ffmpeg`,
`chatterbox_import`, `chatterbox_init` (PASS, 15.96 s, 24000 Hz), `perth`, `qt`,
`voice_sources`, `deterministic_smoke`, `real_engine_smoke`, `dependency_pins`,
`input_subtitles`, `preview_writers`, `release_metadata`, `licensing`,
`crash_reports`, `resolved_records` all **PASS**. `entrypoint` FAIL is just the
launcher directory missing from `PATH` in the SSH session; `turbo`,
`cuda_backend` and `vulkan_backend` are optional/unavailable here.

Real-engine smoke, `HF_HUB_OFFLINE=1`, CPU, no network:

```
python scripts\real_engine_smoke.py --device cpu --json
runtime_seconds 131.8 | build\real_engine_smoke\real_engine_smoke.flac
FLAC 24000 Hz, 129600 frames (5.4 s), peak 0.32
```

Remaining failures, all real Windows-port issues in the tests themselves:

| Cause | Tests |
| --- | --- |
| Throwaway Qt config root is not reaped on Windows file locks | `test_gui_theme_certification::test_design_token_certification_runs_locally_and_passes`, `::test_the_dialog_aspects_certify_the_studio_and_the_wizard` |
| POSIX shell assumed inside tests (`test -f a.py`) | `test_commit_slices::test_verify_success_continues_to_the_journal` |
| CRLF changes byte counts in log summaries | `test_support_diagnostics::test_log_summary_has_names_and_sizes_only` (34 vs 32 bytes) |
| Path separator leaks into user-facing text | `test_soft_launch_gate::test_voice_asset_problems_accepts_owner_owned_inventory` (`Seashells\one.wav`) |
| `audio.cpp` tree not staged in the guest, so notices/Vulkan checks differ | `test_third_party_notices::*` (2), `test_vulkan_setup::test_run_vulkan_setup_download_failure` |

## Air-gap proof (2026-10-21, PASS)

Windows 11 guest booted with `-nic none` (no adapter, DNS dead), installing a
fresh `E:\OracleOffline` from the offline bundle with an empty `E:\hfhub` cache,
then rendering through the managed launcher. The proof script lives outside the
repo (`~/.oracle-win11-vm/airgap-proof.ps1`) and writes
`E:\airgap\airgap-result.json`:

```
result            PASS
install.bat       0  (1218.7 s, 116 windows wheels, offline)
seeded models     4 @ pins, refs/main byte-exact
render            0  (490 s via the-oracle.cmd, HF_HUB_OFFLINE set by the marker)
flac              91246 bytes, 24000 Hz, 4.14 s, peak 1.0
network adapters  0 at start and at end
```

The doctor inside the install reported `PASS Chatterbox CPU init:
from_pretrained(device="cpu") in 94.951s` — that line is what caught the seed
bug below.

**Found and fixed:** `manage_install.seed_offline_models` wrote `refs/main` as
`sha + "\n"`. huggingface_hub treats the file's raw contents as the revision
(snapshot directory name), so every hub-level loader missed the seeded cache
offline and the doctor's model check failed, making a working install exit 1.
The ref is now written byte-exact, enforced by
`test_seed_offline_models_points_refs_main_at_pins` and the install-boundary
test.

## Not yet set up

- `audio.cpp` (3.3 GB with build artefacts) is not copied to `C:\Oracle`; the
  Vulkan backend and its tests need it plus Visual Studio Build Tools.
- `resemble-perth` is installed from its Git URL, which needs `git` on the
  guest — fine here, but worth remembering for an air-gapped rebuild (the
  offline bundle's `resemble-perth==1.1.0` wheel is the alternative).
- The turbo variant (`ResembleAI/chatterbox-turbo`) is not seeded; `turbo` stays
  FAIL in the doctor by design until `scripts/download_models.py --variant turbo`
  runs (its checkpoint is ~2 GB more).
- `PyYAML` is imported by `tests/test_doctor_idempotence.py` and
  `tests/test_windows_install_smoke.py` but is not declared in `pyproject.toml`;
  `manage_install` happens to install it (`PyYAML>=6.0` in
  `CHATTERBOX_ENGINE_PACKAGES`), so a clean `pip install -e ".[dev]"` alone still
  fails those two at collection. Logged, not fixed (out of scope for the VM).

## Gotchas found while installing the ML stack

- **`pip install -e ".[ml,chatterbox,dev]"` cannot resolve.** `the-oracle`'s `ml`
  extra pins `transformers==5.20.0` while `chatterbox-tts 0.1.7` pins
  `transformers==5.2.0`, so the resolver aborts. This is why
  `manage_install.install_dependencies` installs `.[ml,dev]`, then the engine
  packages, then `chatterbox-tts --no-deps` and finally patches its metadata
  (`scripts/patch_chatterbox_metadata.py`). Use the installer, not a hand-rolled
  extras install.
- **pip stalls silently on this network.** Downloads sit at 0 bytes for tens of
  minutes with no error (`pip` has no default read timeout). Set
  `PIP_DEFAULT_TIMEOUT=20` and `PIP_RETRIES=20` for guest installs, and pull the
  multi-hundred-MB wheels (torch, llvmlite, numba) on the host instead —
  `--find-links C:\wheels` (via `PIP_FIND_LINKS`) makes pip prefer them.
- **Hugging Face range requests need a 206 check.** A CDN edge that answers 200
  to a `Range` request silently appends a whole file into a chunk; a 4-way
  parallel download then merges into a file that is 30% too large and only fails
  at `torch.load`. Validate `response.status == 206` and the byte count of every
  part before merging.
- `HF_HUB_ENABLE_HF_TRANSFER` is deprecated in `huggingface_hub` 1.x; Xet is the
  transfer path now, and on this network it was slower than plain ranged
  `curl`/`urllib`.
