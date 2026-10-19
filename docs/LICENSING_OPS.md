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

- Store the printed `signing_seed_hex` in a secret manager (or offline
  media). **Never commit it, never put it in a shell dotfile.** Run
  `keygen` on your own machine: its output is the only copy of the seed,
  and a seed that passes through a chat transcript, an issue tracker, or
  a CI log is compromised.
- `ORACLE_LICENSE_SIGNING_KEY` (64-hex, 32 bytes) is how the seed enters
  the minting environment at mint time — export it from the secret
  manager per-session, per-command.
- Record ONLY `verify_key_hex` in `src/the_oracle/licensing/keys.py` under
  a key id — the two steps above this one are the whole procedure:

  ```python
  KEYS: tuple[VerifyKey, ...] = (
      VerifyKey(key_id="k2", public_key=bytes.fromhex("<verify_key_hex>")),
  )
  ```

  The verify key is *public* material: it ships inside the client and its
  exposure forges nothing.

**The registry must never hold a published key.** Until 2026-10-13 the
commissioning key `k1` *was* the RFC 8032 §7.1 TEST 1 public key, whose
seed is printed in the spec and was hard-coded in this repository's own
tests — any reader could mint a licence every shipped install accepted.
It is gone. The production key **`k2` was registered on 2026-10-19**; its
signing seed was generated on the owner's machine and lives only in the
owner's secret manager (the provisioning run wrote it to
`~/.config/the-oracle/license-signing-seed`, mode `0600`; move it to your
secret manager and delete the local copy). Only the public verify key is
in `keys.py`. Two nets keep a forgeable key out:

- `tests/test_licensing_registry_hygiene.py` fails if any registered key
  is a published RFC 8032 vector key, or if one of the vectors' hex
  appears in `keys.py` at all (with a vacuity probe that re-plants the
  historical defect and requires the scan to go red);
- `scripts/release.py` **refuses to build release artifacts** when the
  registry is empty or holds a published vector key.

The registry now carries `k2`, so the licensing branch of the release
gate is open. The commercial gate still refuses distribution until the
legal and dependency-license owner actions in `docs/SOFT_LAUNCH.md` are
recorded. See §5 for the gated feature.

## 2. Minting a license

The soft launch sells directly: the customer emails **codysa90@gmail.com**
with the subject **"Oracle alpha license"**, and the owner mints a `k2` token
with the procedure below. Deliver only the `token` string; never include the
signing seed or the minting environment in the reply.

```bash
ORACLE_LICENSE_SIGNING_KEY=<seed_hex> python scripts/license_sign.py mint \
  --edition studio \
  --licensee "customer name or email (their choice)" \
  --key-id k2 \  # required: a defaulted key id is how k1 got used
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

## 8. The first gated feature (`render-report`) — and how to move it

Until 1.4.0 nothing was gated: `studio_features` existed in `policy.py`
and no code ever called `require(...)` for it, so the app advertised a
licensing system that enforced nothing. Phase 1 of the 1.4.0 release work
closed that with the **least destructive** option available, because the
other branch of the requirement — "or remove the license UI and the studio
edition" — destroys work that already shipped.

**The gated capability:** `the-oracle render-report PROJECT_DIR` — a
machine-readable technical conformance report for a finished render
(peak/RMS level, clipping, silence ratio, sample rate, channels, frames,
utterance and speaker counts, and whether the plan's declared durations
agree with the audio). It is **additive**: nothing that shipped before it
is gated, so community keeps every existing workflow, which is what
`LICENSING_DESIGN.md` §4 requires.

- Implementation: `src/the_oracle/render_report.py` (measurement) and
  `handle_render_report` in `cli.py` (the gate).
- The entitlement is one constant, `render_report.STUDIO_ENTITLEMENT`, and
  exactly one `policy.require(...)` call site — pinned by
  `tests/test_render_report.py::test_the_gate_is_one_call_site` so moving
  the gate is a one-line change.
- Behaviour under each license state is pinned end to end in that file:
  community refuses with the offline purchase path and writes nothing;
  studio produces the report; an **expired** studio token degrades to
  community, refuses, and leaves the render byte-identical.
- The command never writes to the project it measures (read-only, pinned).

**To move or change the gate** (owner decision, still open — the feature
was chosen as a defensible default, not as final product strategy):

1. If a different capability should be Studio, change the single
   `require(status, ...)` line in `handle_render_report` to that
   capability's own entitlement, and add its own entitlement name to
   `policy.EDITIONS["studio"]` (community must keep everything that
   shipped before).
2. If the licence system should not ship at all, the opposite branch is:
   delete `gui_license.py`'s activation surface + the Help-menu entries +
   the `license-status`/`activate`/`machine-id` commands, drop the `studio`
   and `trial` editions from `policy.EDITIONS`, and remove
   `licensing_key_problems` from `scripts/release.py`. That is a
   destructive change to shipped work: confirm with the owner first.
3. Either way, the honest threat model is in
   `docs/LICENSING_THREAT_MODEL.md`; do not promise more than it says.
