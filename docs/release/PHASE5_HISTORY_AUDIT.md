# Phase 5 History Audit — prepared, not pushed

Audit date: 2026-10-24 UTC

## Recovery anchor

- Full-ref bundle: `/tmp/the-oracle-pre-scrub-20261024T014535Z.bundle`
- SHA-256: `7f2568b884c0f777ebafe5111b63ec700b9195004f73856f42bb9e868ffe9b78`
- The bundle was created outside the repository with `git bundle create ... --all`
  after fetching the four fetchable pull heads. No history rewrite or remote
  mutation has been performed.

## Refs inspected

The local audit covered 18 refs: `release/1.4.0`, `release/1.5.0-rc`, both
origin branches, `alpha-0.1`, `v0.2.0`, five `archive/*` tags, four Codex
turn-diff refs, and fetched `origin/pr/2` through `origin/pr/5` (the remote
names are local tracking refs for `refs/pull/*/head`).

## Findings

- `gitleaks git --log-opts='--all' --redact --no-banner` scanned **790 commits**,
  about **10.46 MB**, and returned **no leaks** (exit 0). This is evidence for
  the current scan only; it does not authorize skipping the post-rewrite scan.
- Historical scrub candidates exist: `STATE.md`, `JUNO_FIXES.log`,
  `RELEASE-CHECKLIST-V1.3.2.md`, `docs/superpowers/**`, and the historical
  crash/capture scripts and reports. Crash source/tests currently present in
  the head are product code and are not automatically scrub targets.
- Historical `/home/cody` strings occur in `docs/SECURITY_AUDIT-1.4.0.md` and
  the Phase 0 evidence, plus intentional `/home/test-user` test fixtures. The
  owner must decide whether all personal path mentions are scrubbed while
  retaining author identities and the published contact email.
- No path named `license-signing-seed`, `ORACLE_LICENSE_SIGNING_KEY`, or an
  equivalent production seed file was found in the object path inventory; the
  gitleaks scan found no secret. This does not prove a secret was never present
  under a different name.
- Three blobs are at least 10 MiB, all current voice/audio assets:
  `Seashells/seashell fransisco.wav`,
  `Seashells/seashell fransisco_TTS_clean_v5_aggressive_targeted_cleanup.wav`,
  and `Output/stream_of_consciousness_dialogue_with_typos.flac`. They are
  present in the current tree and are excluded from the scrub by the owner’s
  instruction to preserve Seashells recordings.
- Current repository object storage at audit time: `git count-objects -vH`
  reported 82.43 MiB packed and `.git` occupied 161M. A post-rewrite size
  comparison is still required.

## OWNER STOP

The rewrite and force-push are deliberately pending the exact owner message:
**`push the scrub`**. After that message, run the scrub on a fresh mirror,
verify every ref, then use commands equivalent to:

```sh
git push --force --all origin
git push --force --tags origin
git push origin --delete \
  archive/codex-cross-platform-install-refactor \
  archive/codex-hardening-fix \
  archive/ingest-refactor-hints \
  archive/serpent-circle-renovation \
  archive/voice-craft-and-recording-studio
```

Do not run those commands until the exact message arrives. The remote remains
unchanged by this audit.
