# License Operations — Vendor Guide

Audience: the vendor operator (you) minting and managing Oracle licenses.
This is the operational half of `docs/LICENSING_DESIGN.md` (design §3–§8);
everything here is grounded in the shipped tooling, not aspirations.

The client never contacts a server: activation, verification, and status
are pure functions of the stored token and the embedded verify keys. That
is a design constraint (LICENSING §6) — every operational procedure below
works fully offline on the vendor side too.

## 1. Key custody (do this once, guard it forever)

The signing seed is the **only secret** in the whole system. It never
touches the repo, the wheel, or any shipped artifact (`scripts/` is not
packaged; `tests/test_license_sign.py` pins the wheel exclusion).

Generate the signing seed and its matching verify key:

```bash
python scripts/license_sign.py keygen
```

- Store the printed `seed_hex` in a secret manager (or offline media).
  **Never commit it, never put it in a shell dotfile.**
- `ORACLE_LICENSE_SIGNING_KEY` (64-hex, 32 bytes) is how the seed enters
  the minting environment at mint time — export it from the secret
  manager per-session, per-command.
- Record the `verify_key_hex` in `src/the_oracle/licensing/keys.py` under
  a key id (`k1` is the commissioning key). The verify key is *public*
  material: it ships inside the client and its exposure forges nothing.

The registry's commissioning key is pinned to the RFC 8032 §7.1 test
vector (`tests/test_licensing_keys.py`) — registry corruption therefore
fails a spec pin rather than silently trusting a new key.

## 2. Minting a license

```bash
ORACLE_LICENSE_SIGNING_KEY=<seed_hex> python scripts/license_sign.py mint \
  --edition studio \
  --licensee "customer name or email (their choice)" \
  [--key-id k1] \
  [--expires 2027-01-31T00:00:00Z] \
  [--machine-hash <32-hex from `the-oracle machine-id`>]
```

- Omit `--expires` for a **perpetual** license (`exp: null`).
- `--edition trial` tokens behave as studio until `exp` passes, then
  degrade to community (LICENSING §4 — degrade, never lock).
- `--machine-hash` is optional (v1 licenses are not machine-locked, D2);
  if used, take the hash from the *customer's* run of
  `the-oracle machine-id` — the client stores and reports the hash only,
  never the raw machine id.
- The output is a JSON document with the token and its payload. **Deliver
  only the `token` string** to the customer; activation is
  `the-oracle activate <token>` (or pipe it via stdin).
- Mint with an ephemeral seed for tests, never the production seed — the
  repo's own tests do exactly that.

## 3. Verification before delivery (always)

```bash
python -c "import sys; sys.path.insert(0, 'src'); \
from the_oracle.licensing import tokens; \
print(tokens.verify_token(open('token.txt').read().strip()))"
```

(or activate it with `the-oracle activate <token>` in a scratch checkout
and read `the-oracle license-status` there.)

A token must verify with state `valid` and the intended edition before it
goes to a customer. Any tampering — one flipped payload byte — fails as
the typed `bad_signature` status, not an exception at the call site.

## 4. Key rotation (planned change of key, zero customer disruption)

Rotation is embedding a new verify key and phasing out the old one
(LICENSING §3):

1. `license_sign.py keygen` → new `seed_hex` + `verify_key_hex`.
2. Add the new verify key to `keys.py` under a **new key id** (`k2`).
   Keep the old entry — both verify during the overlap.
3. Mint new licenses with `--key-id k2`.
4. After one release cycle (every deployed client carries `k2`), remove
   the old key entry from `keys.py`.

The rotation pin (`test_key_rotation_old_key_keeps_verifying_until_dropped`)
proves the overlap contract: a token minted under `k1` still verifies while
`k1` is registered, and reports the typed unknown-key state once dropped.

## 5. Revocation (no server — so do it by rotation)

There is no revocation endpoint by design (LICENSING §1 non-goals).
Revoking a leaked or abused license means **rotation plus expiry**:

- Rotate to a new key id (§4) and drop the old one in the next release:
  every token signed under the dropped key reports unknown-key and the
  install degrades to community.
- For time-boxed abuse, mint replacements with `--expires` so the seat
  self-degrades at the expiry boundary instead of requiring a client
  release.

## 6. What the customer experiences (so support can say it precisely)

- Activation: paste the token (or `the-oracle activate <token>`), fully
  offline; a failure is a typed status with an inline remedy, and a bad
  token writes nothing.
- `the-oracle license-status` prints the edition, licensee, key id, expiry
  (if any), and machine-locked state as JSON.
- `the-oracle machine-id` prints the SHA-256 hash of the machine id —
  safe to share; the raw id is never stored, logged, or sent anywhere.
- An expired or removed license degrades to community — everything that
  ships today keeps working (LICENSING §4). Nothing is ever locked away.
- Removing the token file moves the seat; there is no server to tell.

## 7. Sales-side decisions still open (owner: you)

- Final edition names (`community` / `studio` / `trial` are code
  placeholders — renaming is one table in `policy.py` plus docs, LICENSING §9.3).
- Whether trial tokens are minted per purchase or ship as a generic eval
  token (affects only `license_sign.py` UX, LICENSING §9.4).
