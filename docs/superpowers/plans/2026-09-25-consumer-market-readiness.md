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

- [ ] RED: add `tests/test_logging_rotation.py` with four tests —
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
- [ ] Verify RED: `./.venv/bin/python -m pytest tests/test_logging_rotation.py -q`
      — fails because the constants/helper do not exist yet.
- [ ] GREEN: in `src/the_oracle/utils/logging.py` add
      `LOG_MAX_BYTES = 5 * 1024 * 1024`, `LOG_BACKUP_COUNT = 3`, and
      `default_log_file() -> Path` computing the repo root the way
      `cli.py` does (`Path(__file__).resolve().parents[3] / "logs" / "oracle.log"`);
      swap `logging.FileHandler` for `logging.handlers.RotatingFileHandler`
      read from those module constants at call time. Public signature,
      FD-clean reconfigure comment, and existing callers unchanged.
- [ ] `.gitignore`: add the `logs/` runtime dir beside the other runtime
      sections.
- [ ] Suite: `./.venv/bin/python -m pytest -q` — 1250 + new tests, green.
- [ ] `JUNO_FIXES.log` entry; STATE "Next" pointer; commit
      (`Cap log files with rotation and add the repo-local default path`).

### U1.2 — `crash/` core (§11 step 2)

TDD order inside this unit (one RED/GREEN cycle each, tests as promised in
CRASH §10):

- [ ] RED/GREEN `crash/sanitize.py` + `tests/test_crash_sanitize.py` — every
      §4 rule; mutation: delete the home-prefix rule → the net must fail;
      property probe: no output may contain the input path prefix.
- [ ] RED/GREEN `crash/consent.py` + `tests/test_crash_consent.py` —
      fail-closed reader (mutation: flip the unreadable-file branch → the net
      must fail); opt-out purge; fire-time check (revoking mid-session
      suppresses the next event).
- [ ] RED/GREEN `crash/record.py` + `crash/bundle.py` +
      `tests/test_crash_bundle.py` — typed record → JSON; atomic write; 20
      reports / 32 KiB caps (oldest dropped first).
- [ ] RED/GREEN `crash/handlers.py` + `tests/test_crash_handlers.py` —
      `sys.excepthook` + threading hook + Qt message handler + faulthandler
      into the crash dir; synthetic exception writes a record and does not
      re-raise; double-install idempotent; a crashing handler writes nothing
      and does not recurse.
- [ ] Wire `install_crash_handlers()` once in `cli.main()` and once in
      `MainWindow.__init__` (after paths exist); `.gitignore` the
      `crash_reports/` dir.
- [ ] Suite green; JUNO/STATE; commit.

### U1.3 — CLI privacy surface + doctor `crash_reports` (§11 step 3)

- [ ] RED: `tests/test_doctor_crash.py` (consent-off is `ok=true`
      informational; consent-on shows count/oldest/newest/size/cap state;
      crash-present points at the newest report + one exception line;
      write-permission failure is `ok=false` with a local remedy; remedy
      strings offline-safe) + CLI tests in `tests/test_cli.py` style for
      `privacy-status` / `privacy-opt-in` / `privacy-opt-out [--purge]`.
- [ ] GREEN: command wiring in `cli.py`; `crash_reports` check in
      `scripts/doctor.py` following the existing check shape.
- [ ] Suite green including all three doctor pins; JUNO/STATE; commit.

### U1.4 — Offline-guarantee pins + `PRIVACY.md` + cross-links (§11 step 5)

- [ ] RED: extend `tests/test_offline_guarantee.py` with CRASH §8 pins —
      `socket.socket` patched to raise while a synthetic crash is captured;
      `the_oracle.crash` imports no torch/huggingface/network modules;
      consent writes only the local flag; doctor remedies are
      network-free strings. Mutation-prove each.
- [ ] GREEN: fix whatever the pins expose (they should be green by
      construction; any failure here is a real find, not a test to soften).
- [ ] Write `PRIVACY.md` from CRASH §9's skeleton, each claim footnoted to
      the test that pins it; add the TROUBLESHOOTING crash-review flow;
      README link, CHANGELOG entry.
- [ ] Suite green; JUNO/STATE; commit.

---

## Unit 2 — Licensing core

Contract: `docs/LICENSING_DESIGN.md` §3–§8. Exit: rollout steps 1–5 landed;
nothing that ships today is gated.

### U2.1 — Crypto decision executed + dependency pin (§8 step 1)

- [ ] Add `pynacl==1.5.0` to `dependencies` in `pyproject.toml`.
- [ ] Prove the offline-bundle path: extend
      `tests/test_offline_bundle.py` to assert `pynacl` appears in the base
      requirement set, then run the bundle wheel-download path for
      windows + linux (`--only-binary :all:`, the bundle's own
      `--python-version`) and record that `PyNaCl` + `cffi` wheels land for
      both. **If this probe fails: stop the unit and switch to Option B
      (vendored pure-Python verify) per LICENSING §3 — that switch is a
      design change, so it gets its own record before code continues.**
- [ ] Suite green (doctor `dependency_pins` now covers pynacl); JUNO/STATE;
      commit.

### U2.2 — `licensing/` core (§8 step 2)

- [ ] RED/GREEN `tokens.py` + `keys.py` + `tests/test_licensing_tokens.py` —
      `ORACLE1.` envelope, canonical JSON byte-stability across key-order
      shuffles, mint→verify roundtrip, tampered payload / wrong key /
      unknown key_id refused as typed statuses, `exp` boundary with a frozen
      clock, key-rotation test (§7).
- [ ] RED/GREEN `policy.py` + `machine.py` + `store.py` +
      `tests/test_licensing_store.py` — atomic write (no partial token on
      simulated crash), corrupted file → typed error not a crash, editions
      mapping with **community = today's full set**, downgrade semantics
      (expired trial → community with notice), machine fingerprint SHA-256
      only.
- [ ] Suite green; JUNO/STATE; commit.

### U2.3 — CLI surface + vendor signer (§8 step 3)

- [ ] RED: CLI tests for `the-oracle activate <token>`,
      `license-status`, `machine-id` (community when absent, typed errors).
- [ ] GREEN: `cli.py` wiring; `scripts/license_sign.py` reading
      `ORACLE_LICENSE_SIGNING_KEY` (vendor-only; tests mint with an
      ephemeral test key — no real key material ever in the repo).
- [ ] Suite green; JUNO/STATE; commit.

### U2.4 — Doctor check + offline pins (§8 steps 4–5)

- [ ] RED: `tests/test_doctor_licensing.py` — §5 states (no token →
      `ok=true` informational; valid token details; each failure state with
      its inline remedy); read-only + idempotent + history-independent with
      a license present; signature and exp checks mutation-proven.
- [ ] RED: `tests/test_licensing_offline.py` — §6 pins (activation with
      `socket.socket` patched to raise; no heavyweight imports; remedies
      offline-safe).
- [ ] GREEN: `licensing` check in `scripts/doctor.py`.
- [ ] Suite green; JUNO/STATE; commit; write `docs/LICENSING_OPS.md`
      (vendor guide: key custody, mint, rotate, revoke-by-rotation).

---

## Unit 3 — GUI surfaces (crash consent/review + license activation)

Both slices follow the `gui_ingest.py` injected-Qt pattern and get
patch-surface manifest entries (`scripts/patch_surface_manifest.json` +
its test).

- [ ] `gui_crash.py`: first-run consent dialog (enable local reports / open
      privacy policy / not now = no, never asked again this install;
      enabling available later from Help) + post-crash next-session review
      ("review / share / delete", share = open sanitized JSON then copy or
      save; no upload code). Tests `tests/test_gui_crash.py` in the
      `tests/test_gui_settings.py` style.
- [ ] `gui_license.py`: activation dialog (paste token, typed errors) +
      About panel showing edition/licensee/key_id/expiry; tests
      `tests/test_gui_license.py`.
- [ ] Manifest + ownership tests green; suite green; JUNO/STATE; commit.

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
| 2026-09-25 | U1.1 | — | — |
