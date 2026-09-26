# Licensing & Anti-Piracy — Feature Scope

Status: **steps 1–5 implemented** (2026-09-25): the crypto decision, core
package, vendor script, CLI surface, doctor check, and offline pins have
landed with mutation-proven tests (61 new; six mutations M1–M6 caught live).
Remaining: step 6 (GUI activation slice) and step 7 (privacy policy unit).
This document remains the design contract.

---

## 1. Goals and honest threat model

The Oracle is distributed as Python source plus a built wheel. Any
client-side license check in readable Python can be bypassed by a determined
user editing one file. **This design does not pretend otherwise.** What it
delivers, in order of value:

1. **A clean purchase path** — activation is paste-a-token, works fully
   offline, and says so in the privacy policy truthfully.
2. **Casual-copy friction** — a copied install without its token runs as the
   free/community edition, which is a real deterrent for the target audience
   (home studios), not for crackers.
3. **An audit trail** — every install reports its license state through the
   doctor, and support can identify a customer's edition from the About
   panel.
4. **A foundation for paid tiers** — entitlement gates exist from day one so
   future paid features bolt on without re-architecture.

Non-goals: DRM, obfuscation, tamper-resistance, network license servers,
phone-home validation. A phone-home check would contradict the suite's
offline guarantee and its privacy posture — rejected up front.

## 2. Module layout

New package, following the existing small-module conventions (Qt-free policy
code, app_paths-managed persistence, cli.py owns command wiring):

```
src/the_oracle/licensing/
├── __init__.py     # public surface: current_license(), verify_token(), LicenseStatus
├── keys.py         # embedded verify keys + key-id registry (ONLY place key material lives)
├── tokens.py       # token parse/verify over a versioned, canonical-payload envelope
├── machine.py      # machine fingerprint: SHA-256 of OS machine id — hash only, never raw
├── store.py        # token persistence, repo-local (no XDG user dir exists); atomic write
└── policy.py       # editions → entitlements; grace/downgrade semantics; single source
```

Release-side (repo-only; `scripts/` is not packaged into the wheel):

```
scripts/license_sign.py   # vendor-only minting CLI; private key NEVER in the repo
```

The signing key enters the release process through an environment variable
(`ORACLE_LICENSE_SIGNING_KEY`); losing it means key rotation (supported, see
below), not disaster. A GUI slice later adds a small `gui_license.py`
(activation + About panel) following the `gui_ingest.py` injected-Qt pattern —
out of scope for the first slice.

Import discipline: `the_oracle.licensing` imports stdlib plus the vendored
`the_oracle._ed25519` only — no third-party crypto dependency exists (decision
§3). It must never import huggingface_hub, torch, or anything heavy, and it
must never run at import time — `cli.main()` and `MainWindow` call
`current_license()` lazily at feature gates, so GUI launch stays fast and an
unlicensed install never blocks startup.

## 3. Key and token model

**Crypto choice — DECIDED (2026-09-25): Option B, vendored pure-Python
Ed25519** (`the_oracle/_ed25519.py`, RFC 8032, pinned to the spec's §7.1 test
vectors; verify in the client, sign only in the vendor script). Rationale: no
new dependency for offline installs, nothing compiled to bundle, fully
auditable; verification runs once per activation and once per doctor run, so
the speed cost is irrelevant. The pynacl swap path is contained to this one
module plus `keys.py`'s import. (Option A remains viable if a future need —
high-volume revocation lists, say — makes C-speed verification real.)

An HMAC-SHA256 stdlib-only scheme was considered and rejected: the verify
key must be embedded in the client, so key extraction forges licenses. With
asymmetric signing, extraction buys the attacker nothing.

**Token format** (`tokens.py` owns; versioned envelope):

```
ORACLE1.<base64url(payload)>.<base64url(signature)>
```

Payload is canonical JSON (sorted keys, compact separators, UTF-8) so the
signature is byte-stable across platforms:

```json
{
  "v": 1,
  "key_id": "k1",
  "lic_id": "UUID",
  "edition": "studio",
  "licensee": "optional name or email — the customer's own choice",
  "machine_hash": "optional sha256 — present only for machine-locked seats",
  "iat": 1780000000,
  "exp": null
}
```

`exp: null` is a perpetual license. Verification order: envelope shape →
key_id known → signature over canonical payload → `exp` → machine binding →
edition policy. Any failure yields a typed `LicenseStatus`, never an
exception at the call site.

**Key rotation:** `keys.py` holds an ordered registry of `(key_id, verify_key)`.
Tokens carry their key_id. Rotation = embed the new key, keep the old one
verifying for one release cycle, then drop it. A token failing *all* known
keys is "invalid", not "needs rotation".

**Machine binding — decision point.** V1 recommendation: **not
machine-locked** (per-seat honor system; `machine_hash` field exists but is
unpopulated). Machine-locked tokens require the buyer to run
`the-oracle machine-id` and include the hash at purchase, which adds
support friction for negligible deterrent gain in v1. `machine.py` ships in
v1 (the fingerprint is one stdlib call per OS: `/etc/machine-id` /
Windows `MachineGuid` / macOS `IOPlatformUUID` — hashed, raw value never
stored or logged) so opting in later is a vendor-side token field, zero
client releases. Store only the hash: this is a deliberate privacy
decision, and it is what lets the privacy policy say the license contains
no personal data unless the customer typed some in.

## 4. Editions and entitlements

`policy.py` maps editions to entitlements — the single source the GUI,
CLI, and doctor all read:

| Edition    | Who gets it                       | Entitlements                                    |
|------------|-----------------------------------|-------------------------------------------------|
| community  | default when no token is present  | **everything that ships today**                 |
| studio     | purchasers                        | everything in community + future paid features  |
| trial      | evaluation tokens, `exp`-bounded  | studio features until `exp` + grace, then community |

The critical rollout rule: **community = the current feature set.** Nobody's
existing workflow is gated at any point in this unit. Entitlement gates are
wired and tested from the start, but the first gated feature is a *future*
one. Downgrade semantics: an expired trial/studio token degrades to
community entitlements with a visible notice — renders in flight finish,
no data is ever locked away.

Enforcement surface is deliberately soft: `policy.require(entitlement)`
raises a typed `LicenseRequired` that the CLI and GUI catch and render as a
clean purchase-path message. There is no enforcement deeper than feature
boundaries, by design (see §1).

## 5. What the doctor must say

New `licensing` check in `scripts/doctor.py`, under the same constraints as
every other check (read-only, idempotent, history-independent — pinned by
`test_doctor_read_only.py`, `test_doctor_idempotence.py`,
`test_doctor_report_is_history_independent.py`):

- **No token installed** → `ok=true`, informational: "not activated —
  community edition". Unlicensed is a *valid state*; the doctor never fails
  an install for lacking a license and never suggests purchasing as a
  "problem".
- **Valid token** → `ok=true`, informational: edition, licensee (as the
  customer entered it), key_id, expiry if any, machine-locked yes/no.
- **Failure states** → `ok=false` with the remedy inline: malformed token
  (`re-enter the activation token`), signature invalid or unknown key_id
  (`token is not a genuine Oracle license`), unreadable store file
  (permissions fix), expired (renewal contact — **never** "go online to
  verify"; see §6).
- Verification is a pure function of (token bytes, embedded keys, wall
  clock). The clock is the only external input; a token that crosses its
  `exp` between runs legitimately changes the report, the same way any
  time-derived check does. Nothing else in the check touches mutable state,
  which is what the history-independence pin actually requires.

## 6. What the offline guarantee must say

The licensing unit is **fully offline by construction** — the offline
guarantee test file (`tests/test_offline_guarantee.py`) gains these pins:

1. **Activation and verification perform zero network I/O**: with
   `socket.socket` monkeypatched to raise, mint→store→verify→`current_license()`
   succeeds end to end.
2. **No heavyweight imports**: importing `the_oracle.licensing` never imports
   huggingface_hub or any ML dependency — same import-time discipline the
   offline policy already enforces elsewhere (`offline.py` owns the marker;
   licensing reads it, never writes it, never triggers a download).
3. **Doctor licensing check runs on offline installs** and its failure
   remedies never instruct a network action — an expired subscription's
   remedy is a renewal contact, stated as such.
4. **Deactivation is local**: removing the token file moves the seat; there
   is no server to tell. (Machine-locked re-issue is a vendor email action,
   out of client scope.)

The privacy policy (separate unit) can then truthfully state: *activation
never contacts a server; the license file contains no personal data unless
you typed it in; the machine fingerprint, when used, is stored only as a
hash and never leaves the machine.* That sentence is a design constraint on
this unit, not a later promise to retrofit.

## 7. Test plan (promised, per existing discipline)

- `tests/test_licensing_tokens.py` — mint→verify roundtrip; tampered payload
  rejected; wrong-key token rejected; unknown key_id rejected; `exp`
  boundary (before/after, with a frozen clock); canonical-JSON
  byte-stability across key-order shuffles.
- `tests/test_licensing_store.py` — atomic write (no partial token on
  simulated crash), corrupted file → typed error status, not a crash.
- `tests/test_licensing_offline.py` — the §6 pins (1) and (2), in the same
  mutation-proven style as the existing guarantee tests.
- `tests/test_doctor_licensing.py` — read-only + idempotent with a license
  present; unlicensed install reports `ok=true`; each §5 failure state
  renders its remedy. Mutations: flip the signature check and the exp
  check, confirm the net catches both.
- Rotation test: mint under key 1, add key 2, verify token still passes;
  drop key 1, same token now reports unknown-key.

## 8. Rollout order (each step its own bounded unit)

1. ~~Crypto decision + dependency pin~~ **DONE 2026-09-25** — decided §3
   Option B (vendored pure-Python Ed25519); no dependency was added, so the
   offline bundle is unchanged and `dependency_pins` gained nothing new to
   police.
2. **`licensing/` core** — tokens, keys, policy, store + the §7 unit tests.
3. **CLI surface** — `the-oracle activate <token>`, `the-oracle
   license-status`, `the-oracle machine-id`; `scripts/license_sign.py`
   (vendor-only) + its roundtrip test.
4. **Doctor check** — §5 + doctor tests.
5. **Offline-guarantee pins** — §6 tests, mutation-proven.
6. **GUI slice** — `gui_license.py` (activation dialog + About panel),
   separate session, following the injected-Qt patch-surface pattern.
7. **Privacy policy unit** — consumes §6's final sentence; separate scope.

## 9. Open decisions (owner: you)

1. ~~Crypto~~ **DECIDED 2026-09-25**: vendored pure-Python Ed25519 (see §3).
2. ~~Machine-locked seats in v1~~ **resolved by implementation**: tokens are
   not machine-locked in v1 (the `machine_hash` field and `machine.py` ship,
   so opting in later is a vendor-side token field — zero client releases).
3. Edition naming: `community` / `studio` / `trial` are placeholders —
   final names are a sales decision, not an engineering one.
4. Whether trial tokens mint at purchase time or ship as a generic
   eval token — affects only `license_sign.py` UX (which already takes
   `--expires`), nothing structural.
