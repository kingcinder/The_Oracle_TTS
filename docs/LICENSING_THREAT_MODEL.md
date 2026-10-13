# Licensing threat model — what offline licensing actually defends

Status: **written 2026-10-13** for the 1.4.0 release work (Phase 1, items 1/4/6).
It documents the honest boundary of `the_oracle.licensing` so that no marketing
claim, support script, or EULA draft promises more than the design delivers.
`docs/LICENSING_DESIGN.md` remains the design contract; this file is the threat
model and the residual-risk record.

## 1. What the scheme is

The client embeds **public verify keys only** and verifies a signed token
locally. The signing seed exists solely on the vendor's machine, read from
`ORACLE_LICENSE_SIGNING_KEY` by `scripts/license_sign.py`, which is not packaged
into the wheel. Verification is a pure function of (token bytes, embedded key
registry, wall clock) — no network, ever.

Token: `ORACLE1.<base64url(canonical-json payload)>.<base64url(signature)>`,
Ed25519 over the canonical bytes. Failures are typed (`malformed`,
`unknown_key`, `bad_signature`, `expired`, `machine_mismatch`, `store_error`,
`no_token`) and never raise at the call site.

## 2. Adversaries, and what each one can do

| Adversary | Capability they have | What the scheme does about it |
|---|---|---|
| **Honest customer who shares a token** | A valid token and a copy of the install | Nothing beyond deterrence. A token is bearer credential; the machine-binding field exists but is **off by default in v1**. Accepted risk: support friction of machine-locking exceeds the revenue it protects at this scale. |
| **Reader of the source** | The full client, including `keys.py` | Asymmetric design is the whole defence: extracting a *public* key buys nothing. The historical `k1` defect violated exactly this — it embedded a **published test-vector** key whose seed is in RFC 8032, so any reader could mint valid tokens. Removed 2026-10-13; `tests/test_licensing_registry_hygiene.py` and `scripts/release.py` both refuse to let one return. |
| **Cracker who patches one Python file** | The ability to make `verify_token` return `ok=True` | **Not defended.** Client-side checks in readable Python are removable. See §3. |
| **Reader of the published test vectors** | RFC 8032 §7.1 seeds | The registry must never hold such a key; tests inject their own registry (`tests/licensing_fixtures.py`) so no test can require the shipped table to trust a public key. |
| **Attacker who can write the token file** | Local filesystem access | Not defended by design: whoever can write the store can also patch the client. The store's atomic write protects against *torn* files, not against a hostile writer. |

## 3. Non-goals (do not claim these)

- **Not DRM.** No obfuscation, no tamper detection, no anti-debugging, no
  integrity attestation of the client.
- **Not a network licence service.** No activation server, no phone-home, no
  revocation list, no analytics. Deliberate: it is what lets `PRIVACY.md` say
  activation never contacts a server, and what keeps the offline guarantee
  (`.oracle_offline`) true.
- **Not uncopyable.** A copied install without a token runs as `community`.

The realistic value, in order, is: a clean offline purchase path; casual-copy
friction; an edition the doctor and About panel can report; and entitlement
scaffolding so a future paid feature needs no re-architecture.

## 4. Clock rollback

`exp` is compared against `time.time()` at verification. A user who moves the
system clock backwards can extend a trial or an expired token's life, and one
who sets it forwards can expire a valid licence early.

- **Accepted:** the forward case is *user-hostile to no one but its owner*, and
  the schema, not the clock, is the contract with the customer.
- **Not mitigated:** no "last seen time" is persisted, because persisting one
  creates a write on every verification, breaks the doctor's read-only and
  idempotence pins, and can itself be rolled back.
- **Consequence for trials:** `trial` is a courtesy edition, not a security
  boundary. A determined trial user can extend one by changing the clock. The
  entitlement gate it drives is a purchase path with friction, not an
  enforcement mechanism.

## 5. Machine binding (opt-in, currently unused)

`payload.machine_hash` is an optional 32-hex SHA-256 prefix of the OS machine
id. The client ships the field and `the_oracle.licensing.machine` so turning it
on is a **vendor-side token field**, with no client release.

- The raw machine id is never stored, logged, or transmitted; only the hash
  prefix, and only inside the token.
- **Fail open:** if the local fingerprint cannot be read, a machine-locked token
  still verifies. Locking a paying customer out over an unreadable registry key
  is worse than the piracy it prevents.
- Privacy: see `PRIVACY.md` — "no personal data unless you typed it in" holds
  because the fingerprint is a hash and the licensee field is optional and
  customer-chosen.

## 6. Integrity of the receipt side

- `scripts/license_sign.py` refuses to mint without an explicit `--key-id` (a
  defaulted id is how every token once got signed under the compromised key) and
  refuses without a seed in the environment.
- Rotation: register the new key, keep the old entry for one release cycle,
  then drop it. A token whose key id matches no entry is `unknown_key`, never a
  crash — proven by `tests/test_licensing_keys.py`.
- Distribution: `scripts/release.py` refuses to build artifacts when the
  registry is *empty* (no customer could activate) or holds a published vector
  key (anyone could forge). Both branches are tested.

## 7. Residual risks the owner owns

1. **Key custody.** The production seed must live in a secret manager and never
   in the repo, a log, a test, or a support transcript. Losing it means
   rotating keys, not losing the product; leaking it means rotating keys *and*
   invalidating everything minted under it.
2. **No revocation.** A leaked token cannot be revoked without dropping the key
   id it was minted under (a rotation), which also invalidates legitimate
   tokens under that id.
3. **Edition naming** (`community` / `studio` / `trial`) is a sales decision
   still open in `docs/LICENSING_DESIGN.md` §9.
4. **The first Studio-gated feature** is the owner's choice; see
   `docs/LICENSING_OPS.md` for the current placeholder gate and how to move it.
