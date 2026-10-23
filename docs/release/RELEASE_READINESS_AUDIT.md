# Release Readiness Audit — 2026-10-23

Status: **in progress; not a release approval**. Evidence in this audit is tied to local HEAD `8f08151` (`1.4.1`) unless a different revision is named. The local checkout is the canonical source. The two pre-existing unstaged changes in `pyproject.toml` and `tests/conftest.py` are preserved.

## Findings

### RCI-001 — Candidate changes do not yet have a certified suite baseline

- **Severity:** P0 — release blocker
- **Component:** CI suite-count gate
- **User consequence:** CI cannot establish that the current release tree has a known, stable test-suite count.
- **Evidence:** The initial availability check exited 2 for `8f08151cd161`; its baseline was then recorded. The packaging-corrected candidate commit `65fb0b5bd5ac` measured **2,086 passed, 3 skipped, 0 failed, 0 errors** in a fresh detached worktree. Baseline-only commit `b03a667f9390` inherits that record. The local release run passed **2,089 tests, 0 skipped, 0 failed** with its native Vulkan build available. The workflow run `39913481331` failed its “Suite counts recorded or inheritable” job at `b21caa452736`.
- **Reproduction:** `.venv/bin/python scripts/certify_suite_counts.py HEAD --availability`.
- **Correction:** Record suite counts for the candidate commit and use the existing baseline-only inheritance mechanism for its follow-up metadata commit.
- **Validation:** `scripts/certify_suite_counts.py --availability` passes for `b03a667f9390`, inheriting the measured `65fb0b5bd5ac` record. The certified fresh tree skipped the device-gated Vulkan smoke; the local release run had the native build and passed it.
- **Release blocker:** No for local suite-count qualification. Hosted CI for the local candidate remains open under RCI-002.

### RCI-002 — Remote release branch and CI do not qualify local candidate

- **Severity:** P0 — release blocker
- **Component:** Source/CI parity
- **User consequence:** The source at the remote release branch has not been shown to match or pass checks for the local candidate.
- **Evidence:** Local `release/1.4.0` is `8f08151`; `git ls-remote origin refs/heads/release/1.4.0` returned `b21caa452736`. GitHub run `39913481331` is a failure on `b21caa452736`; `gh run list` shows no run for `8f08151`.
- **Reproduction:** `git rev-parse HEAD`, `git ls-remote origin refs/heads/release/1.4.0`, and `gh run list --repo kingcinder/The_Oracle_TTS --branch release/1.4.0`.
- **Correction:** Reconcile and qualify the exact candidate revision through CI before distribution. No push or publication is part of this work.
- **Validation:** A successful CI run must name the final candidate SHA and required matrix jobs.
- **Release blocker:** Yes.

### RLIC-001 — Product-key gate removed all Community CLI access

- **Severity:** P0 — existing customers would lose shipped workflows
- **Component:** Licensing policy and CLI dispatch
- **User consequence:** An install without a key could not launch `voices` or other existing commands, despite the approved rollout design defining Community as the default and preserving current workflows.
- **Evidence:** On local HEAD `8f08151`, `.venv/bin/python -m the_oracle.cli voices` exited 1 and printed the activation/purchase message. The licensing policy and `cli.main()` applied a product-key check globally. The regression test `test_community_can_use_existing_cli_commands_without_activation` reproduced this before the fix.
- **Correction:** Removed the global activation check; normal commands remain available in Community. Kept the new render report behind its Studio entitlement. No token format, signing key, or issuance policy was changed.
- **Validation:** The regression passes. Full licensing, CLI, report, GUI, and integration coverage passed in the complete suite (**2,087 passed, 0 skipped, 0 failed**). `the-oracle voices` also exits 0 without an activated key.
- **Release blocker:** Corrected locally; still requires final suite and candidate qualification.

### RAUD-001 — Speech quality still needs listening review

- **Severity:** P0 — release blocker until real output is reviewed
- **Component:** Real-engine smoke and voice conditioning
- **User consequence:** The prior harness supplied pure tones as speaker references and accepted any nonzero output bytes; it could pass without demonstrating realistic voice conditioning or valid speech output. A real render proves the engine completed and emitted a valid file, but not that its speech is intelligible or pleasant.
- **Evidence:** `src/the_oracle/real_engine_smoke.py` generated 1-second 220/330 Hz sine-wave references. `_validate_smoke_output` checked only nonempty samples, positive peak, and nonempty file. Two regression tests failed against that behavior. They now require attributed bundled speech references, a 12-utterance two-speaker corpus, speech-like signal variation, and a decodable FLAC.
- **Reproduction:** `.venv/bin/python -m pytest tests/test_review_fixes_text.py -q`; initial red run failed both new reference and stationary-tone checks. Green result is recorded in progress.
- **Correction:** Use real bundled English references and validate exported FLAC structure/sample content. Finish a real render, inspect its metadata and signal, and perform the required listening review.
- **Validation:** Standard Chatterbox CPU render completed in 243.06 seconds after removing Gradio and upgrading Pillow to 13.0.0. `ffprobe` and SoundFile decoded mono FLAC at 24 kHz, 1,085,280 frames / 45.22 s; peak 0.980011, zero samples at/above 0.9999, active-frame RMS CV 1.008. Output SHA-256: `f3ea3671c8ecc290e14fcc2a032f16fc058fcf09692963c591f62656034e6936`. The copied bundled references byte-match their source files. Human intelligibility/timbre review is not available in this run.
- **Release blocker:** Yes, pending completed render and review.

### RCOM-001 — Commercial gate correctly blocks the draft EULA

- **Severity:** P0 — commercial distribution blocker
- **Component:** EULA and commercial approval
- **User consequence:** Users cannot receive approved license terms for a paid alpha.
- **Evidence:** `.venv/bin/python scripts/soft_launch.py --check --json` reports the sole blocker as `EULA-DRAFT.md is still marked as a draft`; the file itself says it is not approved and requires counsel review of governing law/venue.
- **Reproduction:** `.venv/bin/python scripts/soft_launch.py --check --json`.
- **Correction:** Owner and qualified counsel must finalize and approve terms. Engineering will preserve the gate and will not rename or rewrite the draft as if it were approved.
- **Validation:** The gate must report the actual approved final EULA and all relevant legal decisions.
- **Release blocker:** Yes; owner action required.

### RAUD-002 — Loudness normalization could hard-clip peaks

- **Severity:** P1 — limited-release audio-quality requirement
- **Component:** Audio assembly and export
- **User consequence:** The default `light`/`medium` loudness normalization could drive a high-crest-factor signal above full scale and then hard-clip it.
- **Evidence:** `normalize_loudness()` previously limited gain by a fixed maximum and target RMS, then used `np.clip`; a regression fixture with one transient reached exactly `1.0` and failed the new no-hard-clipping assertion. The gain is now also capped by the input peak with 2% headroom.
- **Reproduction:** `.venv/bin/python -m pytest tests/test_audio_assemble.py::test_loudness_normalization_reduces_gain_before_hard_clipping -q` failed before the correction. The focused audio-assembly suite now passes.
- **Correction:** Bound normalization gain to `0.98 / peak` and use the product's default `light` preset in the real-engine acceptance render.
- **Validation:** Default-preset standard CPU render measured peak **0.980011** and zero samples at/above 0.9999. The transient regression and full test suite pass.
- **Release blocker:** Pending real-render verification; user-visible clipping was not measured in a published customer render.

### RSEC-001 — Pillow security fixes required removing unused Gradio

- **Severity:** Closed for the managed runtime dependency graph
- **Component:** Python dependency environment
- **User consequence:** Earlier installs included a Pillow release with known critical/high image-parser defects.
- **Evidence:** Initial audit reported 42 advisory records. The managed install now uses Pillow 13.0.0, Brotli 1.2.1, certifi 2026.10.23, and virtualenv 21.14.13. Pillow 13's official [release notes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html) describe critical/high security fixes. Chatterbox's source imports no Gradio API, and the Oracle uses its Qt UI, so the installer no longer installs Gradio and the metadata patch removes its unused requirement. The rendered CPU smoke passed with Gradio absent. `resemble-perth`, the local project, and CPU PyTorch packages remain outside `pip-audit`'s configured PyPI audit scope.
- **Reproduction:** `.venv/bin/python -m pip_audit`; managed installation commands are pinned in `tests/test_install_boundary.py`.
- **Correction:** Removed the unused Gradio dependency from the managed online/offline install and patched Chatterbox metadata; raised Pillow's minimum to 13.0.0. Updated Brotli, certifi, and virtualenv to fixed versions and regenerated third-party notices.
- **Validation:** `pip-audit` reports no known vulnerabilities, `.venv/bin/python -m pip check` passes, and a real CPU render succeeds with Gradio absent.
- **Release blocker:** No for the audited managed dependency graph; retain the non-PyPI audit limitation in release notes.

### RDOC-001 — Platform support boundary is aligned, Linux qualification remains open

- **Severity:** Closed for wording; Linux qualification remains a separate release blocker
- **Component:** `docs/SOFT_LAUNCH.md` and system requirements
- **User consequence:** Documentation previously gave conflicting platform support statements.
- **Evidence:** README, `docs/SOFT_LAUNCH.md`, Quick Start, and System Requirements now consistently identify Linux x86_64 as the alpha qualification target and Windows as unsupported.
- **Correction:** Reconciled those customer-facing statements without claiming the Linux candidate is qualified.
- **Validation:** `rg` review of customer platform statements found no beta/best-effort Windows claims in these entry points.
- **Release blocker:** Wording issue closed. Linux install and real-render qualification remain open under RCI/RPKG findings.

### RPKG-001 — Existing 1.4.1 artifacts are prior outputs, not this candidate

- **Severity:** P0 — release blocker for candidate distribution
- **Component:** Packaging and artifact provenance
- **User consequence:** Existing checksums do not prove the current local source can recreate the package.
- **Evidence:** Existing 1.4.1 outputs remain separately checksummed. The release builder produced the 1.4.2 wheel and sdist from clean commit `b03a667f93900ee7ea9704f50719e328604d85e1`; the tracked manifest is `release_checksums/checksums-1.4.2.sha256`.
- **Reproduction:** `.venv/bin/python scripts/release.py --skip-tests`.
- **Correction:** The source version is 1.4.2 to avoid colliding with the tracked 1.4.1 checksums. The release script built both formats after a clean-tree test run.
- **Validation:** `sha256sum -c` passes for both archives. Both report version 1.4.2 and include notices/resources. `detect-secrets` found zero items; the corrected sdist contains zero `tests/` entries. Nothing was published.
- **Release blocker:** No for local artifact preparation; hosted CI and human/commercial gates remain separate blockers.

### RPKG-002 — Source archive included the development test suite

- **Severity:** P1 — distribution hygiene requirement
- **Component:** Source distribution manifest
- **User consequence:** The first local 1.4.2 source archive contained 163 test files that users do not need to install or run the application.
- **Evidence:** `tar -tzf release_artifacts/the_oracle-1.4.2.tar.gz | rg '^the_oracle-1.4.2/tests/' | wc -l` returned 163. No `tests` directive existed in `MANIFEST.in`; setuptools included the tracked tree by default. The wheel contained no test suite.
- **Correction:** Add `prune tests` to `MANIFEST.in` and a regression in `tests/test_release_hygiene.py` that requires the exclusion.
- **Validation:** Regression was observed failing before the manifest change. After the change, `pytest tests/test_release_hygiene.py -q` passed (6 tests); the final clean release build contains zero `tests/` entries and its checksum verifies.
- **Release blocker:** No; corrected archive was rebuilt and inspected.

### RINST-001 — Managed install selected mutually incompatible NumPy requirements

- **Severity:** P0 — clean-install blocker, corrected in the working tree
- **Component:** Python dependency resolution for managed Chatterbox installs
- **User consequence:** A fresh install could provision NumPy 2.4.6 even though Chatterbox 0.1.7 requires NumPy `<2`; after downgrading NumPy, `ml-dtypes 0.6.0` still required NumPy `>=2`. `pip check` therefore reported broken requirements and the environment was not internally consistent.
- **Evidence:** Reproduced in the isolated managed install at `/media/cody/Data/oracle-clean-install-YvsR8r/tree`. `pip check` first reported the Chatterbox/NumPy conflict, then the `ml-dtypes`/NumPy conflict. Dry-run resolution selected `ml-dtypes 0.5.4` for the compatible closure.
- **Correction:** Constrain the project to `numpy>=1.26,<2`, pin the managed Chatterbox runtime to `ml-dtypes<0.6`, and add `tests/test_release_dependency_constraints.py`.
- **Validation:** Reinstalled in the isolated tree: NumPy 1.26.4, ml-dtypes 0.5.4, `pip check` clean, and normal doctor exited 0. Targeted install/offline/metadata regression set: 40 passed. The corrected full working-tree suite passed **2,089 tests**, and the clean release build repeated that result.
- **Release blocker:** No for managed dependency consistency; cold download and first-user workflow remain unverified.

## Prioritized implementation plan

1. Complete the suite-count record and repair any current CI regressions; re-run target-specific notice and Windows tests.
2. Complete real CPU speech acceptance using bundled spoken references; record objective audio checks and manual listening results. Test other backends only where actual hardware/runtime support exists.
3. Resolve dependency advisories by severity, reachability, and supported-platform evidence; fix any high/critical supported-path issue.
4. Review cancellation, partial output, failure recovery, project persistence, and export behavior against current code and tests; fix reproduced P0 defects.
5. Qualify clean Linux installation, update, rollback, export, and support paths; identify Windows as unsupported unless current evidence closes the Windows certification gaps.
6. Resolve documentation conflicts, complete customer guidance, and prepare the offer/tester materials without inventing pricing or legal terms.
7. Generate final readiness evidence and reproducible artifacts. Keep commercial status blocked until owner/counsel approval.

## Evidence limits

- The failed GitHub run's job metadata is available, but detailed hosted logs were not returned by `gh run view --log`; its failure summary is not used to infer unobserved root causes.
- No manual listening verdict is recorded yet.
- This audit does not authorize a release or publication.
