# Final Release Readiness — 1.5.0 RC

**Assessment date:** 2026-10-24
**Branch:** `release/1.5.0-rc`
**Artifact source commit:** `c3ad502`
**Distribution:** Not authorized or performed.

## Status

**ENGINEERING STATUS: NOT READY for customer release.** Distribution, serial
gating, pinned Vulkan packaging, update-manifest verification, and Windows
experimental recipes are implemented and focused-tested. Frozen artifact
builds, clean-machine acceptance, hosted CI, and human audio review remain
outstanding.

**COMMERCIAL STATUS: BLOCKED.** `EULA-DRAFT.md` remains a draft requiring
owner and qualified-counsel approval. The soft-launch gate reports that as its
sole blocker; LGPL replacement/source obligations remain a warning.

## Gate results

| Gate | Status | Evidence and remaining work |
|---|---|---|
| A — Core functionality | **UNVERIFIED** | Standard Chatterbox CPU inference produced a decodable 24 kHz two-speaker FLAC. Human intelligibility, voice assignment, and audible quality review remain outstanding. Save/reopen GUI walkthrough is also unverified. |
| B — Installation | **PARTIAL / UNVERIFIED** | Frozen Linux recipes, first-run EULA→activation ordering, static update verification, and model size/free-space preflight are implemented. PyInstaller/AppImage tooling is absent locally; installed-artifact, cold download, resume, rollback, and uninstall acceptance remain untested. |
| C — Quality | **PARTIAL** | Focused current checks include 40 passed for serial, notices, update, frozen packaging, and Vulkan prebuilt paths; the 50-cycle lifecycle test passed locally. Full current suite counts are not recorded for `c3ad502`; no hosted CI result exists for this candidate. |
| D — Privacy and security | **PARTIAL** | No telemetry/upload path was introduced. `pip-audit` found no known vulnerabilities in its auditable runtime sets; local/non-PyPI components were skipped. `detect-secrets` reported 0 findings in both built artifacts. This does not prove the application secure. |
| E — Distribution | **PARTIAL for local artifact preparation** | Source sdist/wheel checksums are recorded. PyInstaller/AppImage/.deb and unsigned Vulkan artifact recipes plus CI upload steps are committed but not locally built. No publication occurred. |
| F — Commercial approval | **BLOCKED** | Draft EULA, counsel review, owner offer/pricing and voice-rights confirmation, LGPL compliance decisions, and publication authorization remain absent. |

## Verification record

- `.venv/bin/python scripts/release.py` — **2,089 passed, 0 skipped, 6 warnings, 0 failed** in 392.34 seconds; built the sdist and wheel after tests.
- `.venv/bin/python scripts/certify_suite_counts.py --availability` — pass for `b03a667`, inheriting the fresh-worktree measurement for `65fb0b5`: **2,086 passed, 3 skipped, 0 failed, 0 errors**. The fresh tree had no ignored native build; its Vulkan device smoke was skipped. The skip-ID ledger is incomplete because `tests/test_skip_audit.py` clears it mid-run; the skip count is from pytest's summary.
- `.venv/bin/python scripts/release.py --check` — pass; version 1.4.2, banners, and changelog agree.
- `.venv/bin/python scripts/generate_third_party_notices.py --check` — pass, 115 components.
- `.venv/bin/python -m pip check` — no broken requirements.
- `.venv/bin/python -m pip_audit` and isolated-runtime `pip-audit --path` — no known vulnerabilities in their auditable installed sets; local project, Perth, and CPU PyTorch were skipped.
- Isolated managed Linux x86_64 installation: NumPy 1.26.4, `ml-dtypes` 0.5.4, doctor exit 0 with 18/18 runtime pins; CPU Chatterbox initialized and a real render completed. The tree had a model cache; cold download and full project workflow were not tested.
- `.venv/bin/python scripts/soft_launch.py --check --json` — one blocker: `EULA-DRAFT.md`; LGPL obligations remain a warning.
- `.venv/bin/python scripts/generate_third_party_notices.py --check` — pass,
  115 components; the earlier missing `antlr4-python3-runtime` entry is now
  present.
- `./.venv/bin/python -m pytest -q tests/test_vulkan_prebuilt.py tests/test_frozen_linux_packaging.py tests/test_update_manifest.py tests/test_serial_gate_product.py tests/test_gui_crash.py::test_first_run_gui_order_is_eula_then_activation_then_window tests/test_third_party_notices.py` — **40 passed**.
- `./.venv/bin/python -m pytest -q tests/test_gui_tooltips_lifetime.py::test_real_main_window_churn_leaves_no_filters_behind` — **1 passed** in 6.59 seconds; hosted Windows evidence is still required.
- `gitleaks git --log-opts='--all' --redact --no-banner` — exit 0, no leaks in 790 commits. The Phase 5 recovery bundle is outside the repository and its SHA-256 is recorded in `PHASE5_HISTORY_AUDIT.md`.
- GUI usability was not exercised because no desktop windows were available. CUDA and Vulkan production inference were not tested.
- Real output, runtime, decode, peak, and SHA-256 are in [Real Audio Acceptance](REAL_AUDIO_ACCEPTANCE.md); human listening remains pending.

## Supported platform

No platform is fully qualified for distribution. Linux x86_64 is the intended
target, but frozen installed-artifact acceptance is open. Windows is
experimental/unsupported pending hosted evidence. CUDA and Vulkan production
rendering are unverified.

## Artifacts

The hashes below are the prior 1.4.2 source rehearsal artifacts. The 1.5.0 RC
frozen artifacts are not yet built locally; hosted CI is intended to produce
them from the post-change commit.

- `release_artifacts/the_oracle-1.4.2.tar.gz` — SHA-256
  `f6567a64dd3b003853350ddea5e4ec9524b1c7d0035d16a7f80bc54276d9eeb7`
- `release_artifacts/the_oracle-1.4.2-py3-none-any.whl` — SHA-256
  `c2ba4067537fc39b6a73f118a620078d9008ddce311c837080046b3e8a2667d6`
- `release_artifacts/checksums-1.4.2.sha256`
- `release_checksums/checksums-1.4.2.sha256`

The initial local 1.4.2 source archive included development tests and was
superseded. The final archive contains zero files under `tests/`; only the
corrected build and its checksum are intended for evaluation.

## Owner actions

1. Approve the EULA with qualified counsel, including governing law and venue.
2. Confirm voice-asset rights, third-party obligations, and the final offer,
   pricing, and support terms.
3. Have a human reviewer assess intelligibility, speaker order, and usability
   of the exact CPU output and record a dated verdict.
4. Complete the full install, first-user, project save/reopen, update, rollback,
   and uninstall walkthrough on the intended supported system.
5. Review and explicitly authorize any later publication after all gates pass.
6. Send the exact message `push the scrub` before any history rewrite force-push.

## Recommended next action

Have an independent listener review the real CPU output while the owner and
counsel resolve the EULA and commercial terms. Keep the 1.4.2 packages local
until those approvals and remaining engineering gates are complete.

## Deferred roadmap

- Qualify Windows only after GUI/native crash certification.
- Measure RAM, disk, and GPU minimums on clean supported hardware.
- Qualify CUDA and Vulkan production paths on matching hardware.
- Consider additional Studio-only tools after customer-value evidence and
  owner approval.
