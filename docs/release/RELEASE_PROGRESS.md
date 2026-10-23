# Release Progress — 2026-10-23

## Current candidate

- Canonical checkout: `/media/cody/Data/DOCUMENTS_PROJECT_FILES/the oracle tts`
- Branch / pre-work HEAD: `release/1.4.0` / `8f08151` (`1.4.1`); current source is being prepared as version `1.4.2` to avoid colliding with the already checksummed 1.4.1 artifacts.
- Remote `origin/release/1.4.0`: `b21caa4`; latest listed hosted run `39913481331` failed on that older SHA. No hosted run for local HEAD was found.
- Pre-existing unstaged edits retained: `pyproject.toml`, `tests/conftest.py` (xdist real-repository isolation).
- A local release-readiness audit and execution plan are in `docs/release/RELEASE_READINESS_AUDIT.md` and `docs/superpowers/plans/2026-10-23-oracle-limited-release.md`.

## Completed evidence

- Focused recorded CI regressions on current checkout: `.venv/bin/python -m pytest tests/test_third_party_notices.py tests/test_soft_launch_gate.py::test_voice_asset_problems_accepts_owner_owned_inventory tests/test_support_diagnostics.py::test_log_summary_has_names_and_sizes_only -q` — **15 passed**.
- `.venv/bin/python scripts/generate_third_party_notices.py --check` — **pass, 136 components** on this Linux environment.
- New real-audio regressions were observed failing against the old harness, then `.venv/bin/python -m pytest tests/test_review_fixes_text.py -q` — **35 passed** after replacing generated tones with bundled attributed speech references, expanding to 12 two-speaker utterances, requiring decoded FLAC, and rejecting stationary-tone output.
- `compileall` for `src/the_oracle/real_engine_smoke.py` and `git diff --check` — **pass**.
- `.venv/bin/python scripts/doctor.py --ci` — **pass**; managed dependencies, CPU Chatterbox initialization (83.189 seconds), Perth watermarking, Qt, deterministic smoke, and Vulkan build/device reported ready. CUDA is unsuitable on the Quadro K600 (1 GiB; 4 GiB minimum). AMD Radeon RX 5700 XT Vulkan device detected for audio.cpp.
- `.venv/bin/python scripts/soft_launch.py --check --json` — **blocked** only by the explicitly draft EULA; 136 components, asset inventory, production verify key, prior 1.4.1 checksums/artifacts, and support workflow checks passed. LGPL source/replacement obligations remain a warning.
- Initial `.venv/bin/python -m pip_audit` reported **42 advisories** across Brotli, certifi, Pillow, and virtualenv. After resolving the Gradio/Pillow constraint and updating managed dependencies, the current audit reports zero known vulnerabilities in the auditable installed set. It cannot audit the local package, pinned Perth source, or CPU PyTorch packages through PyPI.
- The Community licensing regression from local HEAD is fixed: ordinary CLI commands no longer require activation; `render-report` alone checks the `studio_features` entitlement. `voices` now runs successfully with no key, and focused licensing/report/GUI tests passed (**26 passed** at the time recorded).
- Added a Community edition status to `license-status` and About. Preserved the EULA acceptance flow and draft legal gate.
- Default loudness normalization now limits gain to 0.98 peak headroom. The initial transient fixture failed before the fix; `tests/test_audio_assemble.py` passed after it.
- Dependency follow-up: removed Gradio from the managed install and Chatterbox's patched installed metadata because Oracle does not use its demo UI. Pillow 13.0.0 now resolves. Brotli 1.2.1, certifi 2026.10.23, and virtualenv 21.14.13 are installed.
- `.venv/bin/python -m pip check` — **pass**. `.venv/bin/python -m pip_audit` — **pass, 0 known vulnerabilities** in the audited dependency set. `.venv/bin/python scripts/generate_third_party_notices.py --check` — **pass, 115 components** after regeneration.
- Full current-working-tree suite before the isolated dependency follow-up: `.venv/bin/python -m pytest -q` — **2,087 passed, 0 skipped, 6 warnings, 0 failed** in 411.07 seconds. This result is superseded below.
- Re-ran `.venv/bin/python scripts/doctor.py --ci` after removing Gradio and upgrading Pillow — **exit 0**. Managed dependency pins, Chatterbox CPU initialization (21.453 seconds), Perth, Qt, deterministic smoke, audio.cpp Vulkan device detection, and release metadata passed. Optional Turbo checkpoint is not cached; NVIDIA CUDA is unsuitable on the installed 1 GiB Quadro K600. Doctor's real-engine smoke check looks for its default build path; the separate real render is under `/tmp/oracle-release-acceptance-no-gradio`.
- `.venv/bin/python -m pip install --dry-run .[ml]` resolved the managed project's declared base/ML dependencies. These are resolver/metadata checks, not a clean installation.
- Failure-recovery and persistence targeted checks: `.venv/bin/python -m pytest tests/test_audio_assemble.py tests/test_cache_miss_is_not_corruption.py tests/test_project_manifest.py tests/test_render_subprocess_arming.py tests/test_render_worker_integration.py tests/test_stem_cache_write_path.py tests/test_sweep_cache.py -q` — **74 passed**. These cover project manifest round-trip, cache write/recovery, per-utterance partial failure, and render subprocess behavior; they do not prove data survives an actual power loss or customer cancellation session.
- Support/privacy targeted checks: `.venv/bin/python -m pytest tests/test_support_bundle.py tests/test_support_diagnostics.py tests/test_support_offline.py tests/test_support_package.py tests/test_support_report.py tests/test_support_store.py tests/test_cli_bug_report.py tests/test_gui_problem_report.py -q` — **76 passed**. Source review confirms support bundles are local and user-initiated; user must inspect report/package before sending.
- `.venv/bin/python scripts/fresh_clone_acceptance.py --working-tree --only doctor,wrappers --keep` — **pass**. The archived candidate and working tree both report doctor ready; all 9 wrapper rows dispatch correctly and all 22 expected environment-only deltas are registered. It used the existing `.venv` interpreter, so it does not prove clean dependency installation.
- `.venv/bin/python scripts/release.py --skip-tests` — refuses because the source tree is dirty; it also reports the 1.4.1 post-cut changelog date as a by-design state. No artifact was rebuilt.

## Completed real-engine evidence

- Isolated managed Linux x86_64 installation: `scripts/manage_install.py
  bootstrap --pytorch-runtime cpu` completed in a disposable checkout at
  `/media/cody/Data/oracle-clean-install-YvsR8r/tree`. The first install
  exposed an incompatible dependency closure (`numpy 2.4.6` against Chatterbox
  0.1.7's `<2` requirement, then `ml-dtypes 0.6.0` requiring NumPy `>=2`).
  Fixed the declared NumPy range and capped `ml-dtypes<0.6`; added a regression
  test. Reinstalled the managed dependencies: `pip check` is clean, and
  normal `doctor.py` exits 0 with 18/18 runtime pins, CPU Chatterbox,
  watermarking, Qt, and deterministic smoke ready. Focused regression run:
  `pytest tests/test_install_boundary.py tests/test_chatterbox_metadata_patch.py
  tests/test_offline_bundle.py tests/test_release_dependency_constraints.py
  -q` — **40 passed**.
- Real CPU inference through that isolated managed install completed in
  **249.356 seconds**. Output:
  `/media/cody/Data/oracle-clean-install-YvsR8r/real-render/real_engine_smoke.flac`,
  mono 24 kHz FLAC, 47.02 seconds, 1,076,521 bytes, peak 0.980011, no samples
  at/above 0.9999; SHA-256
  `1db8d6806f5c58a996b9e02f2dad43f1674ec00132dbbc965732c47a08d255da`.
  `ffprobe` and SoundFile both decoded it. This isolated tree had a model cache;
  cold download, restart/project workflow, update, rollback, and uninstall were
  not tested. Human listening remains unverified.
- `pip-audit --path` on the isolated runtime found no known vulnerabilities in
  its auditable set; local project, Perth, and CPU PyTorch packages were
  skipped. This is not a complete security audit.
- The first full run after the dependency constraint fix found one stale
  expected pin string in `tests/test_doctor_dependency_pins.py`; the focused
  test passed after updating that expectation to `numpy>=1.26,<2`.
- The subsequent full suite passed: `.venv/bin/python -m pytest -q` —
  **2,088 passed, 0 skipped, 6 warnings, 0 failed** in 391.57 seconds.
  Warnings are Python multiprocessing `fork()` deprecation notices. The
  candidate version/changelog metadata was then advanced to 1.4.2; the
  release builder must rerun tests against the clean candidate commit.
- Candidate source committed locally as `06e6881462b87c8e0d8b2ba601c8abe38273e6b3`
  (`fix(release): prepare 1.4.2 engineering candidate`).
- `.venv/bin/python scripts/certify_suite_counts.py --record` on that commit
  used a fresh detached worktree and recorded **2,085 passed, 3 skipped, 0
  failed, 0 errors**. The one listed skip is the Vulkan device smoke; the
  suite's skip ledger was incomplete because `tests/test_skip_audit.py`
  clears it during the run. This count is distinct from the working-tree run,
  where the local native build made the Vulkan test pass.

- Clean committed-base suite-count certification for `8f08151`: **2077 passed, 3 skipped, 0 failed, 0 errors**. Availability check passed. This measures committed base, not the modified candidate tree.
- Real standard Chatterbox CPU inference with the 12-utterance two-speaker corpus and bundled speech references completed with default `light` loudness in **213.62 seconds**. Output root: `/tmp/oracle-release-acceptance-default`.
- `ffprobe` and SoundFile both decoded the output as mono FLAC at 24 kHz, 1,060,320 frames / 44.18 seconds. Peak **0.980011**, RMS **0.079409**, zero samples at/above 0.9999, 46 100 ms frames below -60 dBFS, active-frame RMS coefficient of variation **1.109**. FLAC SHA-256: `2688bc3049c2733e3c6f25d8e5b4cc8758b6cbf0aae2815d319555aaa366d9e2`.
- The reference WAV files copied into the acceptance inputs byte-match `Seashells/generic/english_male_1.wav` and `english_reader_1.wav`. No ASR model is cached; human listening/intelligibility review remains unverified.
- Repeated the standard CPU render after removing Gradio and upgrading Pillow. It completed in **243.06 seconds**, producing mono FLAC at 24 kHz, 1,085,280 frames / 45.22 seconds; peak **0.980011**, zero samples at/above 0.9999. SHA-256: `f3ea3671c8ecc290e14fcc2a032f16fc058fcf09692963c591f62656034e6936`.

## Customer, support, and product review

- Added a customer-facing Quick Start, system requirements, candidate release notes, and known-issues list. README now identifies the candidate as not approved for sale and links the readiness evidence.
- Support source review: reports are stored locally; the UI/CLI package is user-initiated and never uploads. Bundle content excludes dialogue, manuscripts, rendered audio, voice recordings, reference clips, license tokens, and home-directory paths by default. Crash records require explicit inclusion and local consent. User-authored report text is redacted as a safety net, but the preview still needs human review before sharing. Model and dependency downloads make ordinary connections to external providers; no telemetry is configured.
- Added a 5–10 tester scenario/feedback template and owner-only product offer recommendation. Founder-supported positioning is a recommendation, not an approved offer. Only `render-report` was verified as Studio-gated in the current policy.
- Added a real-audio acceptance record. It clearly leaves listening/intelligibility unverified; automated envelope variation is not speech proof.

## Remaining blockers and next actions

1. Establish a certified suite-count baseline for the final committed candidate; the full working-tree run is not keyed to a source commit.
2. Obtain human listening/intelligibility review for the real output.
3. Complete cold model-download, project save/reopen, update/rollback/uninstall, and first-user workflow acceptance; the isolated managed install and first real render passed, but these steps remain untested.
4. Certify suite counts for the final candidate commit, run hosted CI when an authorized route is available, and build/inspect versioned artifacts from a clean candidate with non-conflicting checksum provenance.
5. Have the owner and qualified counsel approve the EULA, third-party obligations, voice rights, and offer before any commercial distribution.

## Owner-controlled actions

- Owner/counsel must approve final EULA terms and governing law/venue.
- Owner must confirm final commercial positioning/pricing and authorize any publication.
- A human listener must review and record the exact CPU smoke output's intelligibility and speaker ordering.
- No production signing credential was generated or modified in this run.
