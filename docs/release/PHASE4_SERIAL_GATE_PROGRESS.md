# Phase 4 — Serial-number gate progress

Date: 2026-10-23

## What changed

- Replaced the edition entitlement model at the product boundary with
  licensed/unlicensed states. A valid serial number enables functional
  features; an unlicensed install remains visible but inert.
- Added core guards at pipeline analysis, render, and preview entry points;
  GUI project, ingest, recording, voice-processing, model-download, and
  preview entry points; the Recording Studio capture path; and every CLI
  subcommand dispatch.
- Kept the unlicensed allow-list to activation, license status, doctor,
  install/uninstall entry points, `--help`, `--version`, help links, About,
  privacy/help, and quit.
- Rewrote stale edition language in the GUI, CLI, README, privacy policy,
  soft-launch runbook, licensing design/operations/threat-model docs, EULA
  drafts, and counsel packet.
- Added explicit CLI and GUI action classification tests and refusal tests for
  absent, malformed, unknown-key, tampered, expired, and machine-mismatched
  serial-number states.

## Evidence

- `PYTHONPATH=src /tmp/oracle-baseline-uV4W9R/bin/python -m pytest tests/test_eula_acceptance.py tests/test_serial_gate_product.py tests/test_licensing_policy_enforcement.py tests/test_render_report.py tests/test_license_state_render.py tests/test_gui_license.py tests/test_doctor_licensing.py tests/test_cli_bug_report.py tests/test_support_diagnostics.py tests/test_gui_crash.py -q`
  → **99 passed**.
- `git diff --check` → passed.
- `cmp -s EULA-DRAFT.md src/the_oracle/EULA-DRAFT.md` → identical.
- `scripts/soft_launch.py --check --json` → blocked by 7 existing blockers:
  draft EULA and unresolved owner/counsel dependency decisions for
  `chatterbox-tts`, `deepmultilingualpunctuation`, `torch`, `torchaudio`,
  `torchvision`, and `transformers`; one weak-copyleft warning remains.
- `rg -n -i '(community|trial)'` over the Phase 4 code/UI and named release
  documents → no matches.
- `PYTHONPATH=src /tmp/oracle-baseline-uV4W9R/bin/python -m pytest tests/test_serial_vendor_workflow.py -q`
  → **1 passed** using throwaway `k2`/`k3` test keys only; no production seed
  was read or written.

## Remaining or unverified

- Phase 1 frozen Linux artifacts, first-run model workflow, update manifest,
  clean installed-artifact acceptance, and Windows recipe remain incomplete.
- Phase 2 prebuilt Vulkan binaries and end-to-end checksum download remain
  incomplete.
- The production vendor workflow still needs an owner-run confirmation using
  the production seed held outside the repository; the throwaway lifecycle is
  covered above.
- Full-suite compatibility after the gate change, Windows behavior, and
  installed-artifact GUI behavior are not yet verified.

## Owner decisions needed

- Counsel approval of the EULA and dependency license decisions remains an
  OWNER STOP before distribution.
- Proposed serial defaults: one named licensee, own machines, no machine
  binding, current-major-version scope, and non-transferable.
- No serials for 1.3.x/1.4.x users by default.
