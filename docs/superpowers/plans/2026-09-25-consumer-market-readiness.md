# Consumer-Market Readiness — Implementation Plan

> **For the executing agent:** one unit per bounded session; inside a unit,
> one step at a time in order. A step is complete only when its tests are
> green, the full suite has been re-run, and `STATE.md` (plus
> `JUNO_FIXES.log` for fixes) records what happened. Steps use
> RED → GREEN → suite → log → commit. Tick the checkboxes as they land.
> Design inputs: `docs/superpowers/specs/2026-09-25-consumer-market-readiness-design.md`,
> `docs/CRASH_TELEMETRY_DESIGN.md`, `docs/LICENSING_DESIGN.md`.

**Goal:** take the feature-complete V1.3.0 suite to a state fit for sale:
offline licensing, local-first crash/privacy, then measurement-driven
refinement and a release-day climb — without gating anything that ships today.

**Architecture:** two new small, Qt-free, lazily-imported packages under
`src/the_oracle/` (`crash/`, `licensing/`), their CLI/doctor/GUI surfaces
added at the edges (existing CLI wiring, one check per concern in
`scripts/doctor.py`, injected-Qt dialogs per the `gui_ingest.py` pattern),
plus targeted edits to `utils/logging.py`, `README.md`, `docs/`, and the
release machinery. No new processes, no network code, no new persistence
locations (repo-local only, gitignored).

**Tech Stack:** Python 3.12 venv (`requires-python >=3.11,<3.13`), stdlib
first, PySide6 for the GUI slices, `pynacl==1.5.0` (U2 only), pytest 8.3.5
with the repo's `slow` marker, existing scripts (`doctor.py`, `release.py`,
`build_offline_bundle.py`, `fresh_clone_acceptance.py`,
`doctor_idempotence.py`).

## Global Constraints

- **Baseline:** `258f207` → `./.venv/bin/python -m pytest -q` = **1250
  passed, 0 failed, 0 skipped** in ~4 min. Never regress; CI treats skips as
  failures (`ORACLE_FAIL_ON_SKIP=1`), so land no new skips.
- **Offline truth:** no new network I/O paths. The offline-guarantee pins in
  `tests/test_offline_guarantee.py` are extended, never relaxed.
- **Doctor pins:** read-only, idempotent, history-independent — every new
  check must keep `test_doctor_read_only.py`, `test_doctor_idempotence.py`,
  `test_doctor_report_is_history_independent.py` green unchanged.
- **Dependencies:** pin exactly (`==`) and only what the plan names;
  `dependency_pins` in the doctor covers whatever `pyproject.toml` declares.
- **Persistence:** repo-local and gitignored, matching `app_paths.py`'s
  convention (there is no XDG user dir in this suite).
- **Scope:** adjacent findings go to STATE "Noticed, not yet actioned" —
  not fixed in passing. One unit per session.
- **Logging:** every fix gets a `JUNO_FIXES.log` entry
  (`<date> | <file> | <what was wrong> | <what changed> | <tests>`); every
  unit updates `STATE.md`. Local commits only — **never push**.
- **Commands:** RED/GREEN runs use
  `./.venv/bin/python -m pytest <file> -q`; suite runs use
  `./.venv/bin/python -m pytest -q`.

---

## Unit 1 — Diagnostics core (crash / logging / privacy)

Contract: `docs/CRASH_TELEMETRY_DESIGN.md` §3–§11.
Exit: rollout steps 1–3 and 5 landed; §8 pins mutation-proven;
`PRIVACY.md` is born true.

### U1.1 — Log rotation + repo-local default log file (§11 step 1, ships alone)

- [x] RED: add `tests/test_logging_rotation.py` with four tests —
      `test_rotates_at_cap` (monkeypatch `LOG_MAX_BYTES`/`LOG_BACKUP_COUNT`
      small, write past the cap, assert `oracle.log.1` exists and the live
      file is ≤ cap), `test_backups_are_bounded` (past `cap × (backups+2)`,
      assert no `oracle.log.4` ever appears),
      `test_reconfigure_is_fd_clean_with_rotation` (mirror
      `tests/test_review_fixes_concurrency.py::TestLoggingReconfiguration`
      style: one file handler after repeated configure; closed streams on the
      detached ones), `test_default_log_file_is_repo_local`
      (`default_log_file()` lands under the repo root's `logs/`, dir created
      on demand).
- [x] Verify RED: `./.venv/bin/python -m pytest tests/test_logging_rotation.py -q`
      — failed for the right reason: `ImportError: cannot import name
      'default_log_file' from 'the_oracle.utils.logging'`.
- [x] GREEN: in `src/the_oracle/utils/logging.py` add
      `LOG_MAX_BYTES = 5 * 1024 * 1024`, `LOG_BACKUP_COUNT = 3`, and
      `default_log_file() -> Path` computing the repo root the way
      `cli.py` does (`Path(__file__).resolve().parents[3] / "logs" / "oracle.log"`);
      swap `logging.FileHandler` for `logging.handlers.RotatingFileHandler`
      read from those module constants at call time. Public signature,
      FD-clean reconfigure comment, and existing callers unchanged.
- [x] `.gitignore`: add the `logs/` runtime dir beside the other runtime
      sections.
- [x] Suite: `./.venv/bin/python -m pytest -q` — **1315 passed / 0 failed**
      (1250 baseline + 4 rotation + the concurrent licensing unit's 61).
- [x] `JUNO_FIXES.log` entry; STATE "Next" pointer; commit
      (`Cap log files with rotation and add the repo-local default path`).

### U1.2 — `crash/` core (§11 step 2)

TDD order inside this unit (one RED/GREEN cycle each, tests as promised in
CRASH §10):

- [x] RED/GREEN `crash/sanitize.py` + `tests/test_crash_sanitize.py` — every
      §4 rule; mutation: delete the home-prefix rule → the net must fail;
      property probe: no output may contain the input path prefix.
      **Landed 2026-09-25** (commit `45eea1b`): 8 tests; M-SANITIZE caught
      live, reverted sha256-identical.
- [x] RED/GREEN `crash/consent.py` + `tests/test_crash_consent.py` —
      fail-closed reader (mutation: flip the unreadable-file branch → the net
      must fail); opt-out purge; fire-time check (revoking mid-session
      suppresses the next event).
      **Landed 2026-09-25** (commit `45eea1b`): 8 tests; M-CONSENT
      (unreadable file forging consent) caught by 7 tests.
- [x] RED/GREEN `crash/record.py` + `crash/bundle.py` +
      `tests/test_crash_bundle.py` — typed record → JSON; atomic write; 20
      reports / 32 KiB caps (oldest dropped first).
      **Landed 2026-09-25** (commit `45eea1b`): `crash/record.py` +
      `crash/bundle.py`; the tests live in `tests/test_crash_handlers.py`
      (`test_record_schema_shape`, `test_cap_enforced_by_bundled_writes`) —
      no separate `test_crash_bundle.py`.
- [x] RED/GREEN `crash/handlers.py` + `tests/test_crash_handlers.py` —
      `sys.excepthook` + threading hook + Qt message handler + faulthandler
      into the crash dir; synthetic exception writes a record and does not
      re-raise; double-install idempotent; a crashing handler writes nothing
      and does not recurse.
      **Landed 2026-09-25** (commit `45eea1b`): sys/threading excepthooks and
      consent-gated faulthandler landed; the **Qt message handler was
      deferred to the U3 GUI slice** (recorded 2026-09-25: `app_gui.py` was
      the concurrent engine thread's in-flight surface).
- [x] Wire `install_crash_handlers()` once in `cli.main()` and once in
      `MainWindow.__init__` (after paths exist); `.gitignore` the
      `crash_reports/` dir.
      **Landed 2026-09-25** (commit `45eea1b`): `crash_handlers.install()` is
      called once in `cli.main()` (the function is named `install`, not
      `install_crash_handlers`); `crash_reports/` is gitignored. **The
      `MainWindow.__init__` wiring was deferred to U3** with the Qt message
      handler, for the same recorded reason.
- [x] Suite green; JUNO/STATE; commit.
      **Landed 2026-09-25** (commit `45eea1b`): full suite **1354 passed / 0
      failed**; `JUNO_FIXES.log` + `STATE.md` updated.

### U1.3 — CLI privacy surface + doctor `crash_reports` (§11 step 3)

- [x] RED: `tests/test_doctor_crash.py` (consent-off is `ok=true`
      informational; consent-on shows count/oldest/newest/size/cap state;
      crash-present points at the newest report + one exception line;
      write-permission failure is `ok=false` with a local remedy; remedy
      strings offline-safe) + CLI tests in `tests/test_cli.py` style for
      `privacy-status` / `privacy-opt-in` / `privacy-opt-out [--purge]`.
      **Landed 2026-09-25** (commit `f2860a2`): the CLI tests live in
      `tests/test_crash_consent_cli.py` (5), not `tests/test_cli.py`;
      `tests/test_doctor_crash.py` (8) carries the check's cases and its own
      read-only/idempotence pin.
- [x] GREEN: command wiring in `cli.py`; `crash_reports` check in
      `scripts/doctor.py` following the existing check shape.
      **Landed 2026-09-25**: opted-out stays valid and is excluded from
      `overall_ready`; writability via `os.access` (the read-only pin forbids
      write probes); native dumps surface to `next_steps`; `--purge` is an
      explicit flag, not a prompt (the CLI refuses interactive prompts on
      non-TTY).
- [x] Suite green including all three doctor pins; JUNO/STATE; commit.
      **Landed 2026-09-25** (commit `f2860a2`): full suite **1369 passed / 0
      failed**; M-CLI-CONSENT and M-DOC-CRASH caught live, reversions
      sha256-identical; the four §12 decisions confirmed and recorded.

### U1.4 — Offline-guarantee pins + `PRIVACY.md` + cross-links (§11 step 5)

- [x] RED: extend `tests/test_offline_guarantee.py` with CRASH §8 pins —
      `socket.socket` patched to raise while a synthetic crash is captured;
      `the_oracle.crash` imports no torch/huggingface/network modules;
      consent writes only the local flag; doctor remedies are
      network-free strings. Mutation-prove each.
      **Landed 2026-09-28**: 4 new pins (network-free capture, no
      heavyweight imports via module-delta, no network imports in `crash/`
      via source scan, consent writes only the local flag); M3 (smuggled
      `import socket`) and M4 (smuggled `import torch`) caught live with
      byte-identical reverts. The doctor-remedy string pin already existed
      (`test_doctor_crash.py` + the licensing offline remedy pins).
- [x] GREEN: fix whatever the pins expose (they should be green by
      construction; any failure here is a real find, not a test to soften).
      **Green by construction** — no fixes needed.
- [x] Write `PRIVACY.md` from CRASH §9's skeleton, each claim footnoted to
      the test that pins it; add the TROUBLESHOOTING crash-review flow;
      README link, CHANGELOG entry.
      **Landed 2026-09-28**: all six §9 sections, every commitment
      footnoted to its pinning test; TROUBLESHOOTING crash-review flow;
      README link; `[Unreleased]` CHANGELOG entries.
- [x] Suite green; JUNO/STATE; commit.
      **Landed 2026-09-28** (JUNO entry in the record commit; the STATE
      "Next" pointer was deferred — STATE.md carried a concurrent actor's
      in-flight edit at slice time).

---

## Unit 2 — Licensing core

Contract: `docs/LICENSING_DESIGN.md` §3–§8. Exit: rollout steps 1–5 landed;
nothing that ships today is gated.

### U2.1 — Crypto decision executed + dependency pin (§8 step 1)

> **Reconciled 2026-09-27 — closed as Option B, not as written.** The unit
> used the vendored verify-only pure-Python Ed25519
> (`src/the_oracle/_ed25519.py`) and recorded the decision in
> `docs/LICENSING_DESIGN.md` §3 / §8 step 1 / §9.1 (DECIDED 2026-09-25).
> No dependency was added — `pyproject.toml` carries no `pynacl`
> (grep-verified) — so the offline bundle is unchanged and `dependency_pins`
> gained nothing to police. The two `pynacl` boxes below were therefore not
> executed. The campaign spec's **D1** record still reads Option A; repointing
> it is **pending the owner's ratification**, and no code is reverted or
> re-added without that call.

- [ ] Add `pynacl==1.5.0` to `dependencies` in `pyproject.toml`.
      **Not executed — superseded by Option B** (note above).
- [ ] Prove the offline-bundle path: extend
      `tests/test_offline_bundle.py` to assert `pynacl` appears in the base
      requirement set, then run the bundle wheel-download path for
      windows + linux (`--only-binary :all:`, the bundle's own
      `--python-version`) and record that `PyNaCl` + `cffi` wheels land for
      both. **If this probe fails: stop the unit and switch to Option B
      (vendored pure-Python verify) per LICENSING §3 — that switch is a
      design change, so it gets its own record before code continues.**
      **Not executed — moot under Option B** (no new dependency; the offline
      bundle is unchanged). The switch the clause describes is the one that
      landed, with the §3 record.
- [x] Suite green (doctor `dependency_pins` now covers pynacl); JUNO/STATE;
      commit.
      **Landed 2026-09-25** (commit `6023a72`): full suite **1316 passed / 0
      failed**; `dependency_pins` gained nothing new to police — no
      dependency was added.

### U2.2 — `licensing/` core (§8 step 2)

- [x] RED/GREEN `tokens.py` + `keys.py` + `tests/test_licensing_tokens.py` —
      `ORACLE1.` envelope, canonical JSON byte-stability across key-order
      shuffles, mint→verify roundtrip, tampered payload / wrong key /
      unknown key_id refused as typed statuses, `exp` boundary with a frozen
      clock, key-rotation test (§7).
      **Landed 2026-09-25** (commit `6023a72`): 11 token tests; the rotation
      pin lives in `tests/test_licensing_keys.py`
      (`test_key_rotation_old_key_keeps_verifying_until_dropped`) and the
      vendored Ed25519 has its own file, `tests/test_licensing_ed25519.py`
      (RFC 8032 §7.1 vectors).
- [x] RED/GREEN `policy.py` + `machine.py` + `store.py` +
      `tests/test_licensing_store.py` — atomic write (no partial token on
      simulated crash), corrupted file → typed error not a crash, editions
      mapping with **community = today's full set**, downgrade semantics
      (expired trial → community with notice), machine fingerprint SHA-256
      only.
      **Completed 2026-09-28**: the open enforcement gap is closed by
      `tests/test_licensing_policy_enforcement.py` — community grants the
      full shipped set, unknown/absent editions degrade to community,
      `require()` raises the typed `LicenseRequired` only outside the
      edition's entitlements, and the degrade-never-lock rule is pinned.
      Mutations M1 (gate disabled) and M2 (expired keeps its edition)
      caught live, byte-identical reverts. Debug verdict: the production
      code was correct; the gap was coverage, not behavior.
- [x] Suite green; JUNO/STATE; commit.
      **Landed 2026-09-25** (commit `6023a72`).

### U2.3 — CLI surface + vendor signer (§8 step 3)

- [x] RED: CLI tests for `the-oracle activate <token>`,
      `license-status`, `machine-id` (community when absent, typed errors).
      **Completed 2026-09-28**: `license-status` now pinned in
      `tests/test_licensing_policy_enforcement.py` (community-when-absent,
      valid-token report, corrupted-store fails closed with exit 1).
      `activate`/`machine-id` were already pinned in
      `tests/test_activation_flow.py`.
- [x] GREEN: `cli.py` wiring; `scripts/license_sign.py` reading
      `ORACLE_LICENSE_SIGNING_KEY` (vendor-only; tests mint with an
      ephemeral test key — no real key material ever in the repo).
      **Landed 2026-09-25**: `tests/test_license_sign.py` (5) includes the
      pin that `scripts/` never enters the wheel.
- [x] Suite green; JUNO/STATE; commit.
      **Landed 2026-09-25** (commit `6023a72`).

### U2.4 — Doctor check + offline pins (§8 steps 4–5)

- [x] RED: `tests/test_doctor_licensing.py` — §5 states (no token →
      `ok=true` informational; valid token details; each failure state with
      its inline remedy); read-only + idempotent + history-independent with
      a license present; signature and exp checks mutation-proven.
      **Landed 2026-09-25** (commit `6023a72`): 9 tests; M1 (skipped
      signature) and M2 (expiry boundary) caught live.
- [x] RED: `tests/test_licensing_offline.py` — §6 pins (activation with
      `socket.socket` patched to raise; no heavyweight imports; remedies
      offline-safe).
      **Landed 2026-09-25**: 5 tests; M6 (network import in the package)
      caught live.
- [x] GREEN: `licensing` check in `scripts/doctor.py`.
      **Landed 2026-09-25**: unlicensed = `ok=true` and deliberately excluded
      from `overall_ready`; every remedy offline-safe; M5 (doctor failing an
      unlicensed install) caught live.
- [x] Suite green; JUNO/STATE; commit.
      **Landed 2026-09-25** (commit `6023a72`): full suite **1316 passed / 0
      failed**.
- [x] **Still open — write `docs/LICENSING_OPS.md`** (vendor guide: key custody,
      mint, rotate, revoke-by-rotation). Verified absent 2026-09-27 (no file,
      no commit history); split out during the reconciliation because the rest
      of the box landed.
      **Landed 2026-09-28**: grounded in the real `license_sign.py` surface
      (keygen/mint flags verified live, verify-snippet exercised against a
      real token); covers custody, minting, pre-delivery verification,
      rotation, revoke-by-rotation, and the customer-facing experience.

---

## Unit 3 — GUI surfaces (crash consent/review + license activation)

Both slices follow the `gui_ingest.py` injected-Qt pattern and get
patch-surface manifest entries (`scripts/patch_surface_manifest.json` +
its test).

- [x] `gui_crash.py`: first-run consent dialog (enable local reports / open
      privacy policy / not now = no, never asked again this install;
      enabling available later from Help) + post-crash next-session review
      ("review / share / delete", share = open sanitized JSON then copy or
      save; no upload code). Tests `tests/test_gui_crash.py` in the
      `tests/test_gui_settings.py` style.
      **Landed 2026-09-28** (commit `4db0852`): all three consent outcomes per
      contract; review = open-with-OS-viewer / delete / keep; startup branch
      D8 (review → first-run ask → nothing); the U1.2-deferred Qt message
      handler wired here (fatal/critical only). 16 tests with click-through
      fakes.
- [x] `gui_license.py`: activation dialog (paste token, typed errors) +
      About panel showing edition/licensee/key_id/expiry; tests
      `tests/test_gui_license.py`.
      **Landed 2026-09-28** (commit `4db0852`): activation surfaces the same
      typed states as CLI/doctor; a bad token writes nothing; About reads
      `current_license()` (one source of truth). 5 tests; the tests caught
      one real bug in the new dialog (DialogCode on an injected factory).
- [x] Manifest + ownership tests green; suite green; JUNO/STATE; commit.
      **Landed 2026-09-28** (commit `4db0852`): the new modules need no
      manifest entry (they are leaf presentation modules, not moved app_gui
      names — the import-direction net stays green); full suite 1444/0. The
      MainWindow wiring hunks were index-split from a concurrent actor's
      in-flight gui_recording extraction in the same file (4 hunks mine,
      3 theirs, zero mixed, leak-checked). The STATE 'Next' pointer update
      rode the concurrent actor's own STATE reorganization.

---

## Unit 4 — Refinement (measurement-first)

No change in this unit lands without a recorded finding attached to it.

- [ ] **U4.1 Architecture deep-dive** — run `/improve-codebase-architecture`
      (analysis only; HTML report lands in the OS temp dir), distill its
      findings into a tracked digest (`docs/ARCHITECTURE_REVIEW-2026-09-25.md`),
      promote at most three deepening items into this unit; log the rest in
      STATE "Noticed".
- [ ] **U4.2 Stability / segfault** — with U1's faulthandler records live,
      recreate `/tmp/gui_crash_catcher.sh` if absent and drive the candidate
      state (render-after-render, preview-active) from
      `.serpent-circle/04-debug/root-causes.md`; fix the root cause with a
      regression test, or document a data-backed workaround. Exit probe:
      the scripted smoke path runs repeatedly with zero fatal records.
- [ ] **U4.3 Pipeline efficiency** — profile a representative render
      (`cProfile` over `scripts/smoke_render.py`), record wall/CPU per stage
      in `docs/PERFORMANCE_BASELINE.md`, land the top bounded win(s) with
      before/after numbers; no behavior change; suite green after each.
- [ ] **U4.4 UX / accessibility audit** — run the `accessibility` skill
      checklist over the main windows and the two new dialogs
      (names/roles, keyboard-only path, focus order), plus scripted persona
      walkthroughs (first render, multi-speaker, recording studio,
      settings); fix only labeled findings; capture the audit in
      `docs/UX_AUDIT-2026-09-25.md`.
- [ ] **U4.5 Hardware-claim reconciliation** — find the code's actual
      hardware floor, compare with the 3 GB claim, probe where feasible, and
      present the decision; do not silently change code or copy.
- [ ] **U4.6 Noticed-item promotion** — up to three STATE "Noticed" items
      with explicit approval (candidates: `.ps1` bits, `pkg_resources`
      warning when its pinned dependency is next touched).

---

## Unit 5 — Release / sale readiness

- [ ] **U5.1 Docs final pass** — README feature/hardware summary matches
      reality; TROUBLESHOOTING carries activation + crash flows; UPGRADING
      notes license continuity; PRIVACY.md links verified.
- [ ] **U5.2 Commercial surface** — vendor ops guide referenced from
      README's distribution section; EULA / delivery copy is an owner
      decision (ask, do not draft legal text unilaterally).
- [ ] **U5.3 Release gate** — version-bump decision → `release.py
      --sync-changelog` / `--sync-banners` / `--check`; build sdist+wheel;
      tracked `release_checksums/`; `scripts/fresh_clone_acceptance.py`;
      `scripts/doctor_idempotence.py`; full suite; offline bundle for both
      platforms with the pynacl/`cffi` presence check.
- [ ] **U5.4 Ship** — execute the release checklist locally (commits only,
      **no push**), update STATE with the final acceptance evidence.

---

## Progress log

| Date | Unit | Outcome | Evidence |
|------|------|---------|----------|
| 2026-09-25 | U1.1 | Landed: rotation 5 MiB × 3 + repo-local `default_log_file()`; `logs/` gitignored | `tests/test_logging_rotation.py` 4 pass; shape suites combined 33 pass; full suite 1315 pass |
| 2026-09-25 | U1.2 | Landed: crash core (fail-closed consent, sanitizer, record/bundle caps, excepthooks, consent-gated faulthandler); Qt message handler + `MainWindow` wiring deferred to U3 by recorded decision | commit `45eea1b`; `tests/test_crash_sanitize.py` (8) + `test_crash_consent.py` (8) + `test_crash_handlers.py` (10); M-CONSENT/M-SANITIZE caught; full suite 1354/0 |
| 2026-09-25 | U1.3 | Landed: privacy CLI + doctor `crash_reports`; the four §12 decisions confirmed | commit `f2860a2`; `tests/test_crash_consent_cli.py` (5) + `tests/test_doctor_crash.py` (8); full suite 1369/0 |
| 2026-09-25 | U2.1 | Closed as Option B (vendored Ed25519) instead of the written `pynacl` pin; no dependency added; spec D1 repoint pending owner ratification | `docs/LICENSING_DESIGN.md` §3/§8/§9.1; commit `6023a72`; `pyproject.toml` carries no `pynacl` |
| 2026-09-25 | U2.2 | Landed: tokens/keys/Ed25519/store/machine/policy; **open: direct policy tests — none found** | commit `6023a72`; `tests/test_licensing_{tokens,keys,ed25519,store}.py` |
| 2026-09-25 | U2.3 | Landed: `activate`/`machine-id` CLI + vendor signer; **open: `license-status` has no test** | commit `6023a72`; `tests/test_activation_flow.py` (6) + `tests/test_license_sign.py` (5) |
| 2026-09-25 | U2.4 | Landed: doctor check + offline pins; **open: `docs/LICENSING_OPS.md` absent** | commit `6023a72`; `tests/test_doctor_licensing.py` (9) + `tests/test_licensing_offline.py` (5); full suite 1316/0 |
| 2026-09-27 | Recon | Checkboxes and log reconciled with landed work; open sub-items split out, not fixed; D1 ratification awaits the owner | this file; 13 crash/licensing test files re-ran green: **101 passed** |

*Reconciled 2026-09-27 against `git log`, `STATE.md`, and the test files on disk; the 13 crash/licensing test files re-ran green (**101 passed**). No production code changed in this reconciliation.*
