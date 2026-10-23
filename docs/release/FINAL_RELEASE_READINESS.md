# Final Release Readiness — 1.4.2 Candidate

**Assessment date:** 2026-10-23
**Source checkout:** `/media/cody/Data/DOCUMENTS_PROJECT_FILES/the oracle tts`
**Branch / HEAD:** `release/1.4.0` / local 1.4.2 candidate revision not yet committed
**Distribution:** Not authorized or performed.

## Status

**ENGINEERING STATUS: NOT READY for customer limited release.** The managed
Linux x86_64 installer now provisions a consistent runtime, doctor succeeds,
and real CPU inference produced a decodable FLAC. The full test suite passed
after fixing the dependency constraints. The primary success condition still
lacks human intelligibility review, a complete first-user/project workflow,
update/rollback/uninstall qualification, final-candidate hosted CI, and
versioned candidate artifact verification.

**COMMERCIAL STATUS: BLOCKED.** `EULA-DRAFT.md` requires owner and counsel
approval. Product positioning, pricing, and publication authorization remain
owner decisions.

## Gate results

| Gate | Status | Evidence and remaining work |
|---|---|---|
| A — Core functionality | **UNVERIFIED** | Real standard CPU inference succeeds and output is valid FLAC. Human listening, save/reopen customer walkthrough, and broader failure recovery confirmation remain outstanding. |
| B — Installation | **PARTIAL / UNVERIFIED** | Isolated managed Linux install, `pip check`, doctor, and first CPU render pass. Cold model download, restart/project save and reopen, update, rollback, uninstall, and complete customer walkthrough remain untested. |
| C — Quality | **UNVERIFIED** | Working-tree pytest after dependency correction: 2,088 passed, 0 skipped, 6 warnings. The test run preceded the version-only 1.4.2 cut; release tooling must rerun on the clean candidate. No hosted CI result for this candidate; GUI usability and human audio review are incomplete. |
| D — Privacy and security | **UNVERIFIED** | Support packages are user-initiated and redacted, with tests covering their privacy boundary; no telemetry/upload path was introduced. `pip-audit` reports zero findings in its auditable installed set. Non-PyPI components are excluded, and the 1.4.2 artifact secret scan is pending. |
| E — Distribution | **BLOCKED** | Versioned 1.4.2 artifacts have not yet been built from a clean candidate revision. Existing 1.4.1 artifacts/checksums are not evidence for this candidate. |
| F — Commercial approval | **BLOCKED** | Draft EULA, counsel review, owner offer/pricing decision, voice-rights confirmation, and publication authorization are absent. |

## Verification record

- `.venv/bin/python -m pytest -q` — **2,088 passed, 0 skipped, 6 warnings, 0 failed** in 391.57 seconds after the NumPy/`ml-dtypes` correction and matching doctor-pin expectation. Warnings were multiprocessing `fork()` deprecation notices. This preceded the version-only 1.4.2 metadata cut; the clean candidate release command must repeat the suite.
- `.venv/bin/python scripts/doctor.py --ci` — exit 0; CPU Chatterbox initialization, Perth watermarking, Qt, deterministic smoke, and dependency pins passed. Optional Turbo checkpoint absent; this host has CPU-only PyTorch, a 1 GiB Quadro K600, and an AMD Vulkan device whose production inference was not tested.
- Canonical `.venv/bin/python -m pip check` — no broken requirements found.
- `.venv/bin/python -m pip_audit` and isolated-runtime `pip-audit --path` — no known vulnerabilities in their auditable installed sets; local/non-PyPI components were skipped.
- `.venv/bin/python scripts/generate_third_party_notices.py --check` — pass, 115 components.
- Isolated managed install: `scripts/manage_install.py bootstrap --pytorch-runtime cpu`; `pip check` reports no broken requirements, normal doctor exits 0 with 18/18 runtime pins. The isolated install then produced the real CPU output recorded in [Real Audio Acceptance](REAL_AUDIO_ACCEPTANCE.md). The test tree had a model cache; it was not a cold download or complete project workflow.
- `.venv/bin/python scripts/release.py --check` — pass for version 1.4.2 and dated changelog. Candidate source is not yet committed and no package has been built.
- `.venv/bin/python scripts/soft_launch.py --check --json` — blocked by the draft EULA; weak-copyleft obligations remain a warning.
- `.venv/bin/python -m pytest tests/test_audio_assemble.py tests/test_cache_miss_is_not_corruption.py tests/test_project_manifest.py tests/test_render_subprocess_arming.py tests/test_render_worker_integration.py tests/test_stem_cache_write_path.py tests/test_sweep_cache.py -q` — **74 passed** for targeted audio, project, cache, and render worker behavior. This is not a physical interruption/power-loss qualification.
- `.venv/bin/python -m pytest tests/test_support_bundle.py tests/test_support_diagnostics.py tests/test_support_offline.py tests/test_support_package.py tests/test_support_report.py tests/test_support_store.py tests/test_cli_bug_report.py tests/test_gui_problem_report.py -q` — **76 passed** for support report, redaction, bundle, CLI, and GUI code paths.
- `.venv/bin/python scripts/fresh_clone_acceptance.py --working-tree --only doctor,wrappers --keep` — pass: the archived candidate and working tree both report doctor ready; all 9 launcher rows dispatch and expected tree differences are documented. The run shares the existing interpreter and is not a clean installation.
- GUI practical usability was not exercised: the desktop inventory returned no open applications or windows. Unit coverage does not establish a new user's ability to complete the GUI workflow.
- Real output is recorded in [Real Audio Acceptance](REAL_AUDIO_ACCEPTANCE.md); it is valid FLAC but not human-reviewed.

## Supported platform

No platform is fully qualified for distribution by this pass. Linux x86_64 is
the intended alpha qualification target and has passed an isolated managed
install plus one CPU render, pending the full first-user and project workflow.
Windows is unsupported. CUDA and Vulkan production rendering remain
unverified in this environment.

## Owner actions

1. Approve the EULA with qualified counsel, including governing law and venue.
2. Confirm voice asset rights, third-party source/replacement obligations, and
   the final offer/pricing/support terms.
3. Review the real audio output for intelligibility and usability.
4. Review final versioned artifacts and authorize any later publication only
   after the engineering and commercial gates pass; no production credential
   changes or publication occurred.

## Recommended next action

Have an independent listener review the real CPU output and record a dated
intelligibility and speaker-order verdict while the owner resolves the EULA
and offer decisions. In parallel, build and inspect the 1.4.2 artifacts from
the clean local candidate revision; do not publish them.

## Deferred roadmap

- Qualify and support Windows only after its GUI/native crash certification.
- Measure minimum RAM, disk, and GPU requirements on clean supported hardware.
- Qualify production CUDA and Vulkan paths on matching hardware.
- Consider additional Studio-only reporting or production tools after evidence
  of customer value and owner approval.
- Improve automated release provenance so candidate suite counts, artifacts,
  and checksums are bound to one clean source revision.
