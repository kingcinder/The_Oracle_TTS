# Security audit — 1.4.0 release work

Date: 2026-10-13 · Branch: `release/1.4.0` · Host: Ubuntu 24.04, Python 3.12.3
Scope: the license/verify path, the vendored Ed25519 primitive, the installed
dependency set, and a secrets sweep of the tracked tree.

Everything below was run in this session; the commands are given so they can be
re-run. Where a finding is documented rather than fixed, the reason is stated —
nothing here is silently accepted.

## 1. Published test-vector key in the production registry — FIXED

`src/the_oracle/licensing/keys.py` shipped exactly one entry, `k1`, whose public
key was RFC 8032 §7.1 TEST 1's published vector. Its seed is printed in the RFC
and was hard-coded throughout this repository's tests, so **any reader of the
spec could mint a token that every shipped install accepted as a genuine
licence.**

Fix: `k1` removed; the registry is deliberately **empty until the owner
registers the production key** (see the module docstring for the three-step
custody procedure). An empty registry fails closed — every token reports
`unknown_key` and the install runs as `community` — and:

- `tests/test_licensing_registry_hygiene.py` fails if any registered key is a
  known RFC 8032 vector key **or** if the vector's hex appears in `keys.py` at
  all, with a vacuity probe that re-planted the historical defect and confirmed
  the scan goes red on it;
- `scripts/release.py::licensing_key_problems` refuses to build artifacts when
  the registry is empty or holds a published vector key, reading the tree being
  packaged rather than the running installation (both branches tested);
- every licensing test now injects a **test-only** registry
  (`tests/licensing_fixtures.py`, key id `ktest`) via
  `pytest.mark.usefixtures("test_license_registry")`, so no test depends on the
  shipped table trusting a test key.

Owner action still required: `python scripts/license_sign.py keygen` on your own
machine, keep `signing_seed_hex` in your secret manager, register only
`verify_key_hex` as a new `VerifyKey(...)` in `keys.py`. Until then the release
gate is red by design and no customer can activate.

## 2. Ed25519 verification accepted a universal forgery — FIXED

Found by the malformed-input work in this phase, not by a report. With a public
key that decodes to the **identity point** (`01` followed by 31 zero bytes,
a valid curve point), `[h]A` is the neutral element for every `h`, so the
verification equation collapses to `R == [s]B`. The pair `(R = [s]B, S = s)`
therefore verifies **for any message**: a universal forgery from a 32-byte
public key. `tests/test_licensing_ed25519.py::test_identity_public_key_must_not_verify_a_forged_signature`
failed before the fix and passes after it.

Fix in `src/the_oracle/_ed25519.py::verify`:

- small-order public keys **and** R are refused (`[8]P == neutral`), which kills
  the collapse above;
- non-canonical point encodings (`y >= p`) are refused at decode, per RFC 8032
  §5.1.3, so one point cannot have two encodings;
- existing hardening kept: non-canonical S (`S >= L`) and off-curve points.

Coverage added alongside: the full RFC 8032 §7.1 set (TEST 1, 2, 3, 1024,
SHA(abc) — TEST 1024's message transcribed from the RFC and length-asserted at
1023 bytes), per-vector one-bit mutation rejection over R/S/message/key,
non-canonical S per vector, small-order keys in four encodings, wrong and empty
lengths, and a totality check that `verify` never raises on malformed input.

Decision on the primitive (Phase 1 item 4): **keep the vendored pure-Python
Ed25519.** No new dependency, nothing compiled to bundle, fully auditable, and
verification runs once per activation. The pynacl swap path is contained to this
one module plus `keys.py`'s import if that ever changes.

## 3. Dependency vulnerability audit — 103 advisories, 9 packages, PARTLY DOCUMENTED

Command: `.venv/bin/python -m pip_audit --progress-spinner off` (pip-audit 2.10.1),
JSON summarized below. Exit 1 (advisories found).

| Package | Advisories | Lowest fixing version | In the shipped runtime? | Decision |
|---|---|---|---|---|
| transformers | 43 | 4.48.0 | Yes (via `chatterbox-tts`) | **Follow-up, owner-visible.** Bumping 4.46.3 crosses many ML API changes; Chatterbox 0.1.6 pins its own range. Needs a proof-of-failure pass, not a blind bump. |
| pillow | 25 | 12.3.0 | Yes (via `diffusers`/`gradio`) | Documented. |
| starlette | 10 | 1.0.1 | Yes (via `gradio`) | Documented; no HTTP server is exposed by The Oracle, so the affected surface is not reachable. |
| virtualenv | 8 | 21.7.11 | Dev toolchain only | Documented; used by the installer's `venv` step, not by the app. |
| diffusers | 5 | 0.38.0 | Yes (via `chatterbox-tts`) | Documented. |
| gradio | 5 | 6.15.0 | Yes (via `chatterbox-tts`) | Documented; the app never launches gradio's server. |
| urllib3 | 3 | 2.8.0 | Yes (transitive) | Documented; low-risk bump candidate for the 1.4.0 lock file (Phase 4.4). |
| pytest | 2 | 9.0.3 | Dev only | Documented; the release build does not ship pytest. |
| setuptools | 2 | 83.0.0 | Build time only | Documented. |

Honest reading: almost all of these arrive transitively through
`chatterbox-tts`, and the highest-count entry (`transformers`) is a
deserialization/model-loading surface in an app that only ever loads pinned
local checkpoints. That lowers exploitability; it does not make them disappear.
They are recorded here, and the lock-file + SBOM work (Phase 4) is where the
pinned floors get applied and re-audited. **Do not claim a clean audit for
1.4.0**: the release-gating subsection is the licensing key custody, which is
fixed; the transitive advisories remain open with stated reasons.

Also skipped by the audit (not a clean bill of health): `the-oracle` (local),
`torch`, `torchaudio`, `torchvision` (local `+cpu` builds are not on PyPI, so
they could not be matched against advisories).

## 4. Secrets sweep — CLEAN, 59 findings all vetted public constants

Commands:
```
.venv/bin/python -m detect_secrets scan --exclude-files '\.venv/|/build/|audio\.cpp/|Output/|Seashells/|release_artifacts/|\.git/' > .secrets.baseline
# then a re-scan compared against the baseline: NEW (unvetted): none
```
`.secrets.baseline` records all 59 findings, each annotated
`is_secret: false` with the reason in `vetted_reasons`. They are exclusively:
RFC 8032 public test vectors (the primitive's tests, and the two the release gate
forbids), the test-only registry key fixture, public Hugging Face commit SHAs in
`models/pins.py`, and public git commit SHAs in the suite-count record and its
tests. **No API key, token, private key, password, or personal credential is
present in the tracked tree.** No `gitleaks`/`trufflehog` binary exists on this
host, so `detect-secrets` stands in; its baseline makes the sweep repeatable.

Not covered by this sweep: git history. The repository has effectively one
commit era, and Phase 3 owns the history decision (clean orphan publish). A
tree scan is not a history scan and this file does not claim to be one.

## 5. Reproduce

```
.venv/bin/python -m pytest tests/test_licensing_registry_hygiene.py tests/test_licensing_ed25519.py tests/test_licensing_keys.py \
    tests/test_licensing_tokens.py tests/test_license_sign.py tests/test_activation_flow.py tests/test_doctor_licensing.py \
    tests/test_gui_license.py tests/test_licensing_offline.py tests/test_licensing_policy_enforcement.py tests/test_licensing_store.py \
    tests/test_release.py -q          # 148 passed
.venv/bin/python scripts/release.py --check                    # exit 0 (version invariant)
.venv/bin/python scripts/release.py --skip-tests               # exit 1, BLOCKED: no production verify key
```
