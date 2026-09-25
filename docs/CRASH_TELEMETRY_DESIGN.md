# Crash Reporting, Error Logging & Privacy Policy — Feature Scope

Status: **scoped, not implemented** (2026-09-26). Companion to
`docs/LICENSING_DESIGN.md`. One sentence is the whole philosophy:

> **Nothing ever leaves this machine unless the user deliberately makes it
> happen, and the suite must keep working when it can't.**

That sentence is a design constraint enforced by tests, not a promise
retrofitted into a policy document later.

---

## 1. Goals and honest threat model

The suite already makes one privacy-adjacent guarantee today: all network
I/O is model/grammar downloads, user-triggered, and the offline install mode
turns even those off. This unit extends the same posture to diagnostics:

1. **Diagnostics that survive disasters.** A native GUI segfault currently
   vanishes (the STATE.md watch item is blocked-on-repro with an external
   `/tmp/gui_crash_catcher.sh`). Crashes and errors should leave a durable,
   size-capped, privacy-clean record on the user's own disk.
2. **A support path that respects the user.** When something breaks, the
   user decides what — if anything — leaves the machine. The suite composes
   a shareable report; the user sends it. No built-in transport.
3. **A privacy policy that is trivially true.** Because nothing phones home
   and nothing is collected automatically, `PRIVACY.md` states facts the
   code provably upholds (offline-guarantee tests pin them), not intentions.
4. **Better logs by default.** Today `utils/logging.py` appends forever to
   whatever file the CLI passes; nothing rotates, nothing caps. That's a
   disk-growth bug waiting for a long-lived GUI install.

Non-goals: analytics, usage metrics, automatic upload, crash-rate
dashboards, any background process, any scheduled task.

## 2. Existing substrate (what this builds on)

- `utils/logging.py` — `configure_logging(log_file, level)` on the root
  logger, FD-clean reconfigure, used by `cli.py`, `render_subprocess.py`,
  and tests. Retention work extends this; the public signature stays.
- `app_paths.py` — repo-local dirs (`Input/`, `Output/`, `Profiles/`,
  `Seashells/`). **There is no XDG user-data dir**; persistence is
  repo-local (`gui_settings.py` owns `app_settings.json`). Crash artifacts
  therefore live in a gitignored repo-local dir too, consistent with how
  everything else already behaves.
- No crash machinery exists today: zero `excepthook`, zero
  `faulthandler`, zero Qt message handler. This unit fills that hole.
- `.gitignore` already excludes runtime artifact dirs; one new entry covers
  the crash/log dirs.

## 3. Architecture

New package, Qt-free core, lazy, never at import time:

```
src/the_oracle/crash/
├── __init__.py   # install_crash_handlers(), current_crash_consent() public surface
├── handlers.py   # sys.excepthook + threading.excepthook + Qt msg handler + faulthandler
├── record.py     # crash-record schema builder (typed dataclass → JSON)
├── sanitize.py   # redaction pass over every string field (the privacy core)
├── bundle.py     # atomic write, one file per event, directory capped
└── consent.py    # consent flag reader/writer; READ FAILURE = OPTED OUT
```

Wiring: `install_crash_handlers()` is called once from `cli.main()` and
once from `MainWindow.__init__` (after paths exist). Handlers are
fail-safe: every handler body is wrapped so a crash *in the crash handler*
writes nothing rather than recursing, and no handler ever re-raises into
the Qt event loop (that's the failure mode that turns a hang into a
hang-with-modal).

`faulthandler.enable(file=...)` is the piece that finally gives the
blocked-on-repro segfault watch item a path forward: native crashes get a
Python-side stack dump into `crash_reports/` where the external `/tmp`
catcher used to lose them.

### Consent store — deliberately NOT app_settings.json

One small file, `consent.json`, atomic write, schema-validated, sitting
next to the crash dir. Rationale for not reusing the settings schema:
crash capture must decide consent *before* settings load (a segfault
during startup must know the answer), and settings corruption must never
be able to flip consent on. Fail direction is fixed: **unreadable,
missing, or malformed consent file = opted out.** No default-on path
exists anywhere in the code.

## 4. What a crash record contains — and never contains

This table is the unit's core contract; `sanitize.py` and `record.py` are
its enforcement, and the tests mutate them to prove it.

| Field                | Contains                                            | Never contains |
|----------------------|-----------------------------------------------------|----------------|
| `timestamp`          | UTC ISO-8601                                        | — |
| `app_version`        | `the_oracle.__version__`                            | — |
| `platform`           | OS, Python version, Qt version                      | MAC address, hostname, serials |
| `gpu_backend`        | selected backend name, Vulkan device *name string*  | full `vulkaninfo` dump |
| `exception`          | type name + message (sanitized)                     | stack frames with file *contents* |
| `traceback`          | sanitized frames: function + line number only       | local variable values |
| `paths`              | path *kinds* only: `"Seashells/voice.wav"` → `<voice>` | absolute paths, usernames |
| `log_tail`           | last ≤50 log lines, sanitized                       | Input/Output text, cast names, transcript content |
| `edition`            | license edition string (joins with licensing unit)  | licensee name/email, token, machine hash |

Sanitizer rules (deterministic, ordered): absolute paths → kind label by
allowlist (`Input`→`<input>`, `Output`→`<output>`, `Seashells`→`<voice>`,
`Profiles`→`<profile>`, else `<path>`); anything matching the OS user-home
prefix → `~`; log lines matching cast-name patterns or longer than 200
chars → truncated to kind. Unknown stays out: the sanitizer *drops* rather
than passes through when it can't classify. One exported function,
`sanitize_text(text) -> str`, so tests can pin it in isolation.

## 5. Log retention and rotation

`utils/logging.py` gains rotation with the same public signature:
`configure_logging` internally uses `RotatingFileHandler` (5 MiB, 3
backups) instead of the unbounded `FileHandler`, and a new
`default_log_file()` helper returns the repo-local `logs/oracle.log`
(gitignored). Old behavior (explicit `log_file=`) still works; existing
callers change nothing. FD-clean reconfigure semantics — the part the
current file pinns in comments — are preserved and re-tested with rotation
in play.

## 6. Consent UX (the explicit opt-in path)

- **GUI first-run slice** (`gui_crash.py`, later; injected-Qt pattern like
  `gui_ingest.py`): one dialog, plain language, three controls —
  *enable local crash reports* (on-device only, never uploaded), *open
  privacy policy*, *not now* (= no). "Not now" never asks again this
  install; enabling is available anytime from the Help menu.
- **Crash prompt:** after a captured crash, the *next* GUI session shows a
  short notice: "A crash report was saved locally. Review / share / delete."
  Sharing opens the sanitized JSON for inspection, then offers to copy it
  to the clipboard or save it wherever the user chooses. **The suite has no
  upload code at all in v1** — the user attaches the report to an email or
  support ticket themselves. This is what makes the privacy policy a
  statement of architecture rather than intent.
- **CLI:** `the-oracle privacy-status`, `the-oracle privacy-opt-in`,
  `the-oracle privacy-opt-out [--purge]`. `--purge` deletes the crash dir
  contents (and is the revocation story: opting out can immediately purge,
  defaulting to yes with a confirmation).
- Consent is per-install, revocable anytime, and revocation takes effect on
  the next event, not at next launch (handlers check the flag at fire
  time, not install time).

## 7. What the doctor must say

New `crash_reports` check, same constraints as every other check
(read-only, idempotent, history-independent — the three doctor pins):

- **Consent off (the default)** → `ok=true`, informational: "local crash
  reporting disabled — enable with `the-oracle privacy-opt-in`". Disabled
  is a valid state; the doctor never fails for it.
- **Consent on** → report dir stats: count, oldest, newest, total size, cap
  state ("at cap — oldest will be dropped"); each explained inline.
- **Crash present** → pointer to the newest report file and the review
  command, plus one line of its exception type.
- **Write-permission failure** on the crash dir → `ok=false` with the local
  remedy. All remedy strings are offline-safe by review pin: none may
  instruct a network action (same rule the licensing unit adopts).
- This check is also where the segfault watch item surfaces: if
  faulthandler has captured a native-crash record, the doctor names it —
  turning "blocked on repro" into "reproducible with data".

## 8. Offline-guarantee pins this unit must land

In `tests/test_offline_guarantee.py`, same mutation-proven style:

1. **Capture is network-free:** with `socket.socket` patched to raise,
   raise a synthetic exception through the installed handlers → record
   written, no error escapes, suite unaffected.
2. **No heavyweight imports:** importing `the_oracle.crash` never imports
   torch/huggingface_hub/ML deps; `install_crash_handlers()` adds
   milliseconds, not model loads. GUI startup latency is pinned by an
   import-time test.
3. **Consent never triggers I/O beyond the local flag:** opting in writes
   one file; it never schedules, downloads, or contacts anything (the
   no-transport architecture makes this trivially testable: grep-pin that
   `crash/` imports no network modules at all).
4. **All doctor remedies offline-safe:** string-level pin on the remedy
   table.

## 9. PRIVACY.md skeleton (repo root, beside README)

Sections, each one sentence of commitment + one of mechanism:

1. **What we collect: nothing, automatically.** No telemetry, no metrics,
   no phone-home. (Mechanism: the suite's network code is model downloads
   only, and offline mode disables even those — pinned by tests.)
2. **Crash reports stay on your disk** until you choose to share one, and
   you can review exactly what's in one before sharing (§4 table as user
   prose). Deleting them is one command.
3. **Your manuscripts and audio are yours.** They are never included in any
   report; the sanitizer drops rather than passes through what it can't
   classify.
4. **Activation, if you license the suite:** never contacts a server (the
   licensing unit's §6 sentence, cross-referenced); the license file
   contains no personal data unless you typed it in; machine fingerprints,
   where used, are stored only as hashes and never leave the machine.
5. **Model downloads:** when you fetch voices/grammar tools, the download
   goes to the model cache on your disk; we don't see it. Offline installs
   never download at all.
6. **Logs:** written to your disk, rotated, never uploaded; location
   documented; deletion documented.

## 10. Test plan (promised, mutation-proven)

- `tests/test_crash_sanitize.py` — every §4 rule; **mutation: delete the
  home-prefix rule → net must fail**; property-style probe: no output
  string may contain the input path prefix.
- `tests/test_crash_handlers.py` — install → synthetic exception → record
  exists, handler doesn't re-raise, double-install is idempotent, handler
  crash writes nothing and doesn't recurse.
- `tests/test_crash_consent.py` — **mutation: flip the fail direction of
  the unreadable-file branch → net must fail** (this pin is the
  privacy-critical one); opt-out purge; fire-time check (revoke mid-session
  suppresses the next event).
- `tests/test_logging_rotation.py` — cap reached → rollover, no growth past
  3 backups, FD-clean reconfigure still holds.
- `tests/test_doctor_crash.py` — the §7 states, idempotent + read-only pins.
- §8's offline pins live in the existing guarantee file.

## 11. Rollout order (each step its own bounded unit)

1. **Logging rotation** (`utils/logging.py` + `default_log_file()`) —
   smallest, standalone, ships alone.
2. **Crash core** — `crash/` handlers/record/sanitize/bundle + consent
   fail-closed reader + §10 tests. **Immediately improves the segfault
   watch item:** next native repro leaves an actionable record in-repo.
3. **CLI consent surface** — `privacy-status/opt-in/opt-out` + doctor
   `crash_reports` check.
4. **GUI slice** — first-run consent dialog + post-crash review/share
   prompt (`gui_crash.py`, injected-Qt pattern, patch-surface manifest
   entry).
5. **PRIVACY.md + offline pins** — the doc lands only after its claims are
   pinned by tests, so it is born true.
6. **Cross-link docs** — TROUBLESHOOTING gains the crash-report review
   flow; README links PRIVACY.md; CHANGELOG entries.

## 12. Open decisions (owner: you)

1. **No built-in transport in v1 — confirm.** Recommended: yes, none. A
   future opt-in uploader is additive; building it now would weaken the
   policy's core sentence on day one.
2. **Caps: 20 reports / 32 KiB / 5 MiB logs — confirm** or retune.
3. **Consent file separate from `app_settings.json` — confirm** (rationale
   in §3).
4. **Crash prompt placement:** next-session notice (recommended) vs
   immediate post-crash dialog. Immediate dialogs during crash recovery
   are how a bad day gets worse; next-session is calmer and equally
   honest.
