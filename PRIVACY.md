# Privacy

What The Oracle collects, what stays on your disk, and what leaves your
machine. Every commitment below is pinned by a test in the suite — the
policy is born true and the tests keep it true.

## What we collect: nothing, automatically

The Oracle has no telemetry, no metrics, and no phone-home. There is no
analytics service, no crash upload, and no account system.

*Mechanism:* the only network code in the suite is model downloading, and
offline installs disable even that; the diagnostics packages
(`the_oracle.crash`, `the_oracle.licensing`) are source-pinned to import no
network module at all (`tests/test_offline_guarantee.py`).

## Crash reports stay on your disk until you choose to share one

When local crash reporting is enabled (off by default), a crash writes a
sanitized report into this install's `crash_reports/` folder — capped at 20
reports of at most 32 KiB each. Nothing is ever uploaded: there is no
upload code to review or distrust. You can read a report as plain JSON
before deciding to share it, and delete everything with
`the-oracle privacy-opt-out --purge` (or manage consent with
`privacy-status` / `privacy-opt-in` / `privacy-opt-out`).

*Mechanism:* fail-closed consent stored separately from your settings —
an unreadable or missing consent file means **opted out**; capture is
consent-gated at fire time (`tests/test_crash_consent.py`,
`tests/test_crash_handlers.py`); the doctor's `crash_reports` check is
read-only (`tests/test_doctor_crash.py`).

## Your manuscripts and audio are yours

Project files, input text, reference clips, and recordings are never
included in any crash report. The sanitizer's core rule is **drop what it
cannot classify**: paths are reduced to kinds, your home directory becomes
`~`, long content is truncated, and stack frames carry locations — not
your words.

*Mechanism:* the sanitizer contract and its property probe
("no output may contain the input path prefix")
(`tests/test_crash_sanitize.py`).

## Activation, if you license the suite

Activation never contacts a server — it is a paste-a-token flow that
works fully offline. The license file contains no personal data unless you
typed it into the licensee field. Machine fingerprints, where a vendor
opts to use them, are stored only as SHA-256 hashes and never leave your
machine.

*Mechanism:* verification is a pure function of the stored token and
embedded keys (`tests/test_licensing_offline.py`); the fingerprint is
hash-only with a fail-open skip when unavailable
(`tests/test_licensing_keys.py`).

## Model downloads: yours, local, and skippable

When *you* fetch a voice model or grammar tool, the download goes to the
model cache on your disk; nobody sees it. Offline installs (the
`.oracle_offline` marker) never download at all — model resolution is
pinned to the local cache with outbound connections forbidden.

*Mechanism:* `tests/test_offline_guarantee.py` — every model-resolving
entry point inherits offline mode, and resolution is proven cache-only
under a blocked socket.

## Logs: on your disk, rotated, never uploaded

Application logs live in this install's `logs/` folder, capped at 5 MiB ×
3 backups, and are never transmitted anywhere. Delete the folder at any
time; it is recreated on demand and git-ignored.

*Mechanism:* the rotation contract and repo-local default path
(`tests/test_logging_rotation.py`).

## Questions or a report to share

Every crash report names the exception and where it happened, in plain
language. Share a report only through a channel you choose (for example,
email to the vendor); The Oracle itself will never send one.
