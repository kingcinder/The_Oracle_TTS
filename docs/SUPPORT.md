# Support: diagnose, report, package, share

The Oracle's support path is local-first, like everything else in the suite:
the app collects the evidence, lets you read it, and then waits for you to
send it. There is no upload code, no telemetry, and no background reporting.

Three tools cover a support case:

| Tool | Answers | Writes |
|------|---------|--------|
| `the-oracle bug-report diagnose` | "what does this machine look like?" | nothing |
| `the-oracle bug-report new` | "what happened, in your words?" | one local draft |
| `the-oracle bug-report package` | "what do I attach to the ticket?" | one zip under `Output/` |

The GUI equivalents live under **Help → Report a Problem…** (compose and save),
**Help → Privacy & Crash Reports…** (consent and crash review), and the
existing `support-bundle` command for a bundle without a written report.

## 1. Start with the doctor

```bash
./oracle doctor
python scripts/doctor.py --json     # machine-readable
```

The doctor verifies capabilities by executing them (entrypoint, engine import,
Qt offscreen, deterministic smoke render, backend probes). If it names a
layer, fix that first: most "the app is broken" reports are a stale venv, a
missing `ffmpeg`, or a model that was never fetched.

`the-oracle bug-report diagnose` is the next step: it produces the same kind
of environment snapshot a report would attach, without creating a report.

## 2. Compose a report

```bash
the-oracle bug-report new \
  --title "Render stops at the last line" \
  --summary "The progress bar freezes at 92 percent and never finishes." \
  --step "Open a two-speaker script" \
  --step "Click Render" \
  --step "Wait for the last line" \
  --expected "A finished FLAC" \
  --actual "The render never finishes" \
  --area render \
  --severity high \
  --contact you@example.com
```

Useful flags:

- `--crash` (or `--crash newest`) references the newest local crash record.
  `--crash <record.json>` names a specific one from `crash_reports/`.
- `--no-diagnostics` leaves the diagnostics snapshot out of the report.
- `--no-doctor` skips the doctor subprocess; the rest of the snapshot is still
  collected.
- `--json` is the machine-readable form for scripts and tests.

Titles are 8–160 characters; steps are capped at 20 entries of 500 characters
each. Invalid input exits `2` and writes nothing.

Drafts live in `support_reports/` next to the other runtime folders. The
store keeps the newest 25 and drops the oldest beyond that. `bug-report list`
shows what is on disk, `bug-report show <id>` prints the exact Markdown that
would be shared, and `bug-report delete <id> --yes` removes one (the CLI never
prompts, so `--yes` is required).

## 3. Package it

```bash
the-oracle bug-report package <id>                 # -> Output/oracle-report-<id>.zip
the-oracle bug-report package <id> --include-crash-reports
```

The zip contains:

- `report.md` — the reviewed, shareable report;
- `report.json` — the same report as data (also what `show --json` prints);
- `diagnostics.json` — the full snapshot (version, platform, memory/disk,
  engine availability, settings summary, doctor verdict);
- the existing support bundle's entries: `environment.json`, `license.json`,
  redacted `settings.json`, the newest redacted logs, and `doctor.json`;
- `manifest.json` — what was included, what was skipped and why, and the
  attachment list.

Crash records are included **only** when `--include-crash-reports` is passed
and local crash-reporting consent is on. The report may *reference* a crash
record without that consent; referencing is not copying.

## 4. Share it

```bash
the-oracle bug-report issue <id>
```

prints two prefilled links and opens nothing:

- a GitHub issue in `kingcinder/The_Oracle_TTS` with the title, report body,
  and `bug` label already filled in;
- a `mailto:` link to the support address with the same information.

In the GUI, **Copy Issue Link** puts the issue URL on the clipboard and
**Email Support** hands the `mailto:` link to your mail client. Both are
button presses you make; the support package itself contains no transport
code.

Attach the zip from step 3 to whichever channel you chose. The issue link
cannot carry attachments, so the package is always attached separately.

## 5. What is collected — and what never is

Collected (redacted):

- app version, edition, install shape, Python and Qt versions;
- OS, release, machine architecture, CPU count, total/available memory, free
  disk at the install;
- engine availability and versions (Chatterbox, torch, numpy, scipy,
  sounddevice), GPU/Vulkan configuration *presence*;
- model-cache presence (not model contents);
- `ffmpeg` presence and version line;
- a whitelist of UI settings (theme, backend, model variant);
- log file *names and sizes*, and the newest log contents inside the bundle
  (redacted, capped);
- crash-report consent state and record count;
- environment-variable **names** that affect the app, never their values;
- the doctor's verdict and failing check names.

Never collected:

- dialogue files, manuscripts, rendered audio, voice recordings, reference
  clips;
- license tokens, licensee names, machine fingerprints;
- home-directory or absolute paths, usernames;
- environment-variable values;
- arbitrary settings keys (recent-file lists are excluded by the whitelist and
  redacted in the bundle's own `settings.json`).

User-written report text passes through the same redaction as the bundle
(home prefix → `~`, license tokens and common API-key shapes → placeholders,
emails → `[redacted email]`). The contact field is the exception: it is stored
so support can reply, and it is not printed in the Markdown body. Redaction is
best-effort; `bug-report show` exists so you can read the exact text before
sharing it.

## 6. For maintainers: working a report

1. Read `report.md` first; it names the area, severity, and reproduction
   steps.
2. Read `diagnostics.json` for the environment, then `doctor.json` for the
   capability verdict. `manifest.json` says what was deliberately omitted.
3. Reproduce with the report's steps. If the report references a crash record,
   `the-oracle privacy-status` on the user's machine names the file; ask for
   the package (not the raw record) if it was not attached.
4. Ask for `render_plan.json` or the input text **only** when needed, and say
   why — the suite never collects them.
5. When fixed, add a regression test that encodes the reported steps. A bug
   report that produces no test is a bug report that can return.

## 7. Related documents

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — symptom → cause → fix.
- [SOFT_LAUNCH.md](SOFT_LAUNCH.md) — alpha scope and purchase/activation.
- [PRIVACY.md](../PRIVACY.md) — the standing data policy and its test pins.
- [CRASH_TELEMETRY_DESIGN.md](CRASH_TELEMETRY_DESIGN.md) — crash unit design.
