# Phase 0 Baseline — 1.5.0 RC

Date: 2026-10-23
Branch: `release/1.5.0-rc`
Base commit: `ef5abbe5b204` (`release/1.4.0`)

## Environment

- Checkout status before baseline: clean.
- Repository size: `9.6G` including ignored local runtime/build data.
- Interpreter: Python 3.12.3.
- Fresh environment: `/tmp/oracle-baseline-uV4W9R`.
- Installation: `python3.12 -m venv ...` followed by `python -m pip install --upgrade pip` and `python -m pip install -e '.[dev]'`.
- The repository `.venv` was not used because its pip launcher referenced the stale path `/home/cody/Documents/the oracle tts/.venv/bin/python`.

## Required commands

| Command | Result |
| --- | --- |
| `/tmp/oracle-baseline-uV4W9R/bin/python scripts/release.py --check` | Pass: `single-source version invariant holds` |
| `/tmp/oracle-baseline-uV4W9R/bin/python scripts/certify_suite_counts.py --availability` | Fail: no baseline recorded for `ef5abbe5b204` |
| `/tmp/oracle-baseline-uV4W9R/bin/python scripts/generate_third_party_notices.py --check` | Fail: stale notices; missing `cloudpickle` |
| `/tmp/oracle-baseline-uV4W9R/bin/python scripts/soft_launch.py --check --json` | Fail: 8 blockers, 1 warning |
| `/tmp/oracle-baseline-uV4W9R/bin/python -m pip check` | Pass: `No broken requirements found.` |
| `/tmp/oracle-baseline-uV4W9R/bin/python -m pip_audit` | Pass: `No known vulnerabilities found`; local `the-oracle` was not auditable on PyPI |
| `/tmp/oracle-baseline-uV4W9R/bin/python scripts/release.py` | Fail: 2,078 passed, 11 failed, 6 warnings in 282.41s |

## Explanation of baseline failures

The documented `.[dev]` editable install installs development tooling and the
base package, but not the optional real-engine extras. The failing tests that
import `chatterbox`, `torch`, `transformers`, `deepmultilingualpunctuation`, or
`setuptools` therefore fail in the fresh environment. The remaining failures
are the known stale third-party notice (`cloudpickle`) and a backend-parity
assertion. This is evidence that the documented fresh-install path does not
yet provide a complete test/runtime environment; it is not a release pass.

The soft-launch blockers were the draft EULA and missing/unknown dependency
license decisions for `chatterbox-tts`, `deepmultilingualpunctuation`, `torch`,
`torchaudio`, `torchvision`, and `transformers`. The weak-copyleft notice
warning also remains.

## Phase 0 exit assessment

Baseline evidence is recorded. Phase 0 is **not green**: the release suite,
suite-count availability, third-party notice check, and soft-launch authority
remain unresolved. No behavior change was made in this phase.
