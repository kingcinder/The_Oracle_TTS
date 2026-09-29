# Consumer-Market Readiness — Campaign Design & Decisions

Status: **decisions resolved, ready to execute** (2026-09-25). This is the
campaign-level companion to the two committed unit contracts —
`docs/CRASH_TELEMETRY_DESIGN.md` and `docs/LICENSING_DESIGN.md`. Those remain
authoritative for unit internals; this document records the decomposition,
the resolutions of their open decisions, and the acceptance bar for
"fit to sell". Execution plan:
`docs/superpowers/plans/2026-09-25-consumer-market-readiness.md`.

## 1. Where the product stands (verified 2026-09-25)

- Tree `258f207` on `main`, clean; full suite **1250 passed, 0 failed,
  0 skipped** in 243.8s; 16 warnings, all dependency-side (logged in STATE).
- Every major feature ships already: custom synthesis, multi-party
  conversation, recording studio, CPU / CUDA / Vulkan paths, offline install
  mode, doctor with pinned read-only + idempotence + history-independence
  tests, release metadata single-sourced with tracked checksums.
- Missing for sale: the two scoped units (licensing, crash/privacy), a GUI
  pass over the new surfaces, measurement-driven refinement, and the
  release-day climb.

## 2. Unit decomposition (each unit is one bounded session)

| # | Unit | Deliverable | Contract | Exit gate |
|---|------|-------------|----------|-----------|
| U1 | Diagnostics core | Log rotation, `crash/` package, CLI privacy surface, doctor `crash_reports`, offline pins, `PRIVACY.md` | CRASH §3–§11 | §11 steps 1–3 + 5 landed; §8 pins mutation-proven; suite green |
| U2 | Licensing core | `licensing/` package, CLI `activate`/`license-status`/`machine-id`, `scripts/license_sign.py`, doctor `licensing`, offline pins | LICENSING §3–§8 | §8 steps 1–5 landed; community = today's feature set (nothing gated) |
| U3 | GUI surfaces | `gui_crash.py` (first-run consent, post-crash review), `gui_license.py` (activation + About), patch-surface manifest entries | CRASH §6, LICENSING §2 | injected-Qt pattern honored; manifest test green |
| U4 | Refinement (measurement-first) | Architecture deep-dive digest, segfault repro under the new faulthandler path, profiling baseline + bounded perf wins, accessibility/UX audit, hardware-claim reconciliation | this doc §4 | every change has a before/after probe; no speculative polish |
| U5 | Release / sale readiness | Docs final pass, vendor ops guide, release gate run (`release.py`, checksums, fresh-clone acceptance, offline bundle) | existing release machinery | all gates green on the release commit |

**Ordering rationale:** U1 first — it is the smallest standalone slice
(logging rotation ships alone), and its crash records are the data path the
blocked-on-repro GUI segfault needs. U2 next — the crypto probe resolves the
one blocking dependency before any licensing code. U3 groups both GUI slices
so the injected-Qt / patch-surface discipline is exercised in one session.
U4 refines against real artifacts (including the new dialogs). U5 last: the
docs it ships must not describe behavior that has not landed.

## 3. Decisions resolved (defaulting to the contracts' recommendations)

**D1 — Crypto: ~~Option A, `pynacl`~~ → RATIFIED 2026-09-28: Option B.**
Owner ratified the fallback against this clause's own escape hatch ("Option B
with its own record"): the built licensing unit uses the vendored pure-Python
Ed25519 verifier, DECIDED in `docs/LICENSING_DESIGN.md` §3 (61
mutation-proven tests; no new dependency; the offline bundle carries no crypto
wheels, making the U2.1 `--only-binary` pynacl probe moot). The Option A
evidence below is retained as the record of why a compiled dependency was
acceptable-but-unnecessary. Original decision text: **The blocking
§3 caveat is resolved with repo evidence: `scripts/build_offline_bundle.py`
derives `pypi_reqs` from `project_requirements()` (pyproject base
`dependencies` + `ml` extra) and downloads them per platform with
`--platform win_amd64` / host tags and `--only-binary :all:`
(lines 72, 107–117), so a base dependency rides into both bundle platforms
with no bundler edits. PyNaCl publishes `cp36-abi3` wheels for `win_amd64`
and manylinux — one wheel covers every Python in `>=3.11,<3.13` — and the
base set already carries compiled wheels (PySide6, numpy, scipy,
scikit-learn, soundfile) through this same path. The decision is *proved* at
U2.1 with a `--only-binary` download probe for both platforms; if the probe
fails, the documented fallback is Option B (vendored pure-Python verify,
LICENSING §3) and the plan stops for that switch.

**D2 — No machine-locked seats in v1** (LICENSING §3 recommendation):
`machine.py` ships; `machine_hash` stays unpopulated; the field makes opt-in
a vendor-side token change later with zero client releases.

**D3 — Edition names `community` / `studio` / `trial` stand as code names**
(LICENSING §9.3): renaming later is one table in `policy.py` plus docs, so
the sales-facing final names are a copy decision, not an architecture one.

**D4 — Trial tokens mint vendor-side only** via
`scripts/license_sign.py --edition trial --days N` (LICENSING §9.4). No
in-app minting, no expiry logic beyond the token's `exp`.

**D5 — No built-in crash transport in v1** (CRASH §12.1): the user reviews
and shares a sanitized report themselves; an uploader, if ever, is additive.

**D6 — Caps as designed** (CRASH §12.2): 20 crash reports, 32 KiB per
report, logs 5 MiB × 3 backups.

**D7 — Consent lives in its own fail-closed `consent.json`** (CRASH §12.3):
unreadable/missing/malformed = opted out; no path can default to on.

**D8 — Crash notice at next session, not modal at crash time** (CRASH
§12.4).

**D9 — Refinement is measurement-first**: U4 lands no "polish" without a
recorded finding (profile, audit checklist, or crash record) attached to it.

**D10 — Unit order U1 → U2 → U3 → U4 → U5** as in §2; within every unit the
rollout order of its contract is binding.

## 4. Acceptance bar for "ready for sale"

A release commit is sale-ready when all of these are true and evidenced:

1. Full suite green, zero skips under `ORACLE_FAIL_ON_SKIP=1`, including the
   doctor's three pinned guarantees with every new check added.
2. Offline guarantee holds with new pins: activation and crash capture
   perform zero network I/O; no heavyweight imports; no network remedy
   strings anywhere.
3. Licensing: mint → store → verify roundtrip; tampered/expired/unknown-key
   tokens refused as typed statuses; community edition gating nothing that
   ships today.
4. Crash/privacy: fail-closed consent proven by mutation; sanitizer's
   drop-what-it-cannot-classify proven by mutation; `PRIVACY.md` statements
   each footnoted to the test that pins them.
5. Stability: the scripted GUI smoke run completes with zero fatal records;
   the blocked-on-repro segfault either has a root-cause fix with a
   regression test or a documented, data-backed workaround.
6. Performance: a recorded baseline exists for the render path; every landed
   optimization has before/after numbers; no regression exceeds the noise
   floor.
7. Release gate: `scripts/release.py --check` passes (version +
   CHANGELOG dated), tracked checksums updated, fresh-clone acceptance and
   doctor idempotence scripts pass, offline bundle builds for both
   platforms (crypto wheels no longer apply — ratified Option B adds no
   dependency).
8. The hardware claim ("min. 3 GB free VRAM/DRAM") has been reconciled with
   the code's actual floor and appears identically in README and docs.
   (RATIFIED 2026-09-28: the numeric claim is dropped — no sale copy states a
   GB figure; copy mirrors README's enforced framing — optional NVIDIA 4+ GiB
   VRAM for CUDA, CPU/DRAM guaranteed fallback; DRAM guidance stays a
   recommendation until U4.3 measures the CPU memory profile.)

## 5. Skill mapping (and honest substitutions)

- `/brainstorming` → this doc; the two unit contracts were already the
  designs, so no third design doc was invented.
- `/writing-plans` → the plan document named in the header.
- `/executing-plans`, `/test`, `/test-driven-development` → the execution
  discipline in the plan: RED before GREEN, mutation-proven pins, full suite
  after every slice, `STATE.md` + `JUNO_FIXES.log` on every fix.
- `/improve-codebase-architecture` → U4.1, where it can critique the new
  packages too; its report lives in the OS temp dir, so the plan distills
  findings into a tracked digest before acting.
- `/python-performance-optimization` → U4.3, measurement-first.
- `/ship-it` → U5.4, local commits only; **nothing is pushed** (repo rule).
- `/reenvision`, `/experience` are **not installed** in
  `/home/cody/.agents/skills/`; substituted by U4.1 (architecture report) and
  U4.4 (accessibility skill + scripted persona walkthroughs). Flagged, not
  silently skipped.

## 6. Logged, not actioned (scope discipline)

- 3 GB sales claim vs. the repository's documented hardware floor — owned by
  U4.5; nothing is edited until that evidence pass happens (see STATE
  "Noticed").
- `.ps1` executable-bit inconsistency, `pkg_resources` deprecation warnings,
  first real run of the `vulkan-smoke` CI job, live CUDA certification — all
  remain STATE "Noticed" items, unchanged by this campaign unless U4.6
  promotes one with explicit approval.
