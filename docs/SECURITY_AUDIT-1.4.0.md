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

## 3. Dependency vulnerability audit — current scan clear, bounded exclusions

Rechecked 2026-10-16 with `.venv/bin/python -m pip_audit --progress-spinner off`
and `.venv/bin/python -m pip check`. The PyPI advisory scan reports no known
vulnerabilities and the dependency consistency check passes. The tested runtime
uses Transformers 5.20.0, Diffusers 0.38.0, Gradio 6.16.0, Starlette 1.3.1,
Hugging Face Hub 1.33.0, pytest 9.0.3, setuptools 83.0.0, virtualenv 21.7.13,
safetensors 0.8.0, and Chatterbox 0.1.7. The install step applies the
audited dependency overrides to Chatterbox's installed metadata and refreshes
its RECORD hashes; its bundled Python source remains the upstream wheel.

Setuptools 83 removes `pkg_resources`, which the PyPI Perth 1.0.1 wheel still
imports. The installer now uses a pinned upstream Perth commit (version 1.1.0),
whose resource loading uses `importlib.resources`. The offline builder converts
that exact commit to a wheel and the offline installer consumes the wheel
without VCS or build-network access.

Audit limits: pip-audit cannot match the local Oracle package, CPU-only
PyTorch wheels, or Git-sourced Perth against PyPI advisories. They remain
explicitly unaudited rather than being counted as clean.

## 4. Secrets sweep — 71 vetted findings, no unreviewed tree findings

Commands:
```
.venv/bin/python -m detect_secrets scan --exclude-files '\.venv/|/build/|audio\.cpp/|Output/|Seashells/|release_artifacts/|\.git/|\.secrets\.baseline'
# Compare each (path, hashed_secret) with .secrets.baseline; fail on any new pair.
```
The 2026-10-13 re-scan found 71 entries. Twelve were absent from the old
baseline: four public Git commit SHAs in the suite-count record and eight
public RFC 8032 test-vector values in test-only fixtures and regression tests.
They are recorded with `is_secret: false`. The findings are public test
vectors, test-only registry material, pinned public Hugging Face commit SHAs,
and public Git commit SHAs. **No API key, token, production private key,
password, or personal credential was found in the tracked tree.** The host's
gitleaks binary was used for the history scan below; detect-secrets remains the
repeatable tree-scan fallback.

The rewritten release history was scanned separately with gitleaks after the
history scrub. The scan covered 780 commits reachable from the local release
checkout's refs and reported no leaks. The scrub verification also searched all
reachable path names and the current tree for the removed personal files; both
counts were zero. The recovery branch
`pre-history-scrub-release-2026-10-14` preserves the pre-rewrite history for
rollback.

Commands and results:
```
/home/cody/redteam-tools/bin/gitleaks detect --source . --no-banner --redact \
  --config .gitleaks.toml --report-format json \
  --report-path /tmp/oracle-gitleaks-final.json
# 780 commits scanned; no leaks found; exit 0

git log --all --name-only --format='' | rg -c \
  "Cody's Seashell|fransisco|Seashell_No_[12]|Input/fransisco help|JUNO_FIXES|STATE.md|docs/superpowers"
# 0
```

## 5. Reproduce

```
.venv/bin/python -m pytest tests/test_licensing_registry_hygiene.py tests/test_licensing_ed25519.py tests/test_licensing_keys.py \
    tests/test_licensing_tokens.py tests/test_license_sign.py tests/test_activation_flow.py tests/test_doctor_licensing.py \
    tests/test_gui_license.py tests/test_licensing_offline.py tests/test_licensing_policy_enforcement.py tests/test_licensing_store.py \
    tests/test_release.py -q          # 148 passed
.venv/bin/python scripts/release.py --check                    # exit 0 (version invariant)
.venv/bin/python scripts/release.py --skip-tests               # exit 1, BLOCKED: no production verify key
```
