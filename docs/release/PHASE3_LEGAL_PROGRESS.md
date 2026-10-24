# Phase 3 Legal and License Progress — 1.5.0 RC

## Changed

- Updated `EULA-DRAFT.md` and `src/the_oracle/EULA-DRAFT.md` to reflect the
  owner's one-time serial-number product, no subscription, no Community/trial
  edition, inert unlicensed state, offline activation, no telemetry, and final
  sales policy.
- Marked proposed seats, machine binding, version scope, and transfer terms as
  proposals rather than approved legal terms.
- Added `docs/legal/EULA_COUNSEL_PACKET.md` with the exact counsel questions,
  plain-language product terms, and owner decisions.
- Added `docs/legal/LICENSE_DECISIONS_PROPOSED.md` from installed package
  metadata and exact pinned model-card revisions. No decisions were written to
  `scripts/license_review.json`.

## Evidence

- `cmp -s EULA-DRAFT.md src/the_oracle/EULA-DRAFT.md` — exit 0.
- `/tmp/oracle-baseline-uV4W9R/bin/python -m pytest tests/test_eula_acceptance.py -q`
  — **8 passed**.
- Managed-environment metadata confirmed the listed package versions/licenses;
  pinned Hugging Face model cards each reported `license: mit` at their pinned
  commit.

## Remaining / unverified

- Counsel has not approved the draft, so it must remain named `EULA-DRAFT.md`.
- `scripts/license_review.json` still intentionally has no approvals; the
  soft-launch gate remains blocked.
- Frozen-artifact LGPL replacement/relink testing is not complete.
- The whole-program serial-number gate has not yet been implemented.
