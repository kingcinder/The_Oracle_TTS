#!/usr/bin/env python3
"""The certifier family's throwaway config root: one owner for the whole policy.

``certify_gui.py`` and ``certify_theme_switch.py`` both need the same three
things before they touch Qt or ``the_oracle``, and each used to implement them
itself — the mkdtemp, the two environment redirects, and a reap loop with the
same five attempts, the same pause and the same warning, byte-for-byte. This
module is that preamble once, so the policy has one place to be right and one
place to be tested.

What the policy is, and why each part of it exists:

* **A throwaway root**, under the system temp directory. Machine state is
  isolated rather than backed up: while the redirect is in place, the per-theme
  settings rewrites, the window timing logs and the crash-consent reads all
  resolve inside it, so even a process that dies mid-run cannot have touched the
  developer's real state. (The scripts measured this the hard way: before the
  redirect, a sweep rewrote the real settings file once per theme — backed up and
  restored in ``finally``, so a SIGSEGV mid-run lost the user's theme, layout and
  last input — and wrote ``Output/logs/gui_startup_timing.json`` and
  ``gui_prewarm_timing.json`` into the repo.)

* **BOTH ``XDG_CONFIG_HOME`` and ``APPDATA``**, because
  ``platform_support.user_config_root`` reads the first on POSIX and the second
  on Windows: redirecting one alone leaves the other platform's real config
  readable and writable.

* **The reap retries** (``REAP_ATTEMPTS`` × ``REAP_PAUSE_S``) rather than
  removing once: the Qt media stack's PulseAudio client writes its own
  ``pulse/cookie`` under ``$XDG_CONFIG_HOME`` while the client library loads, so
  a single ``rmtree`` can lose that race. A clean run exits on the first attempt.
  A root that survives every attempt is REPORTED on stderr, never swallowed by
  ``ignore_errors``: directories left in /tmp are a defect, not a courtesy.

* **The reap covers every exit path**, not just the sweep's ``finally``: this
  module registers its own ``atexit`` hook, which is what covers argparse's
  ``--help``/``SystemExit``, a usage error, and anything that dies before the
  scripts reach their teardown. Measured 2026-10-10: those uncovered exits were
  exactly where the harnesses' /tmp litter came from, one root per run.

Ordering, and the one rule callers cannot delegate: **the redirect has to happen
before the first ``the_oracle`` import.** Nothing here imports the package (this
module is deliberately Qt-free and ``the_oracle``-free — it must be importable
before the config root is known, and it must stay importable at interpreter exit,
when Qt's C++ application may already be gone), so the ordering holds only if the
caller imports THIS module and calls ``isolated_config_root`` above its own
package imports. ``tests/test_certifier_isolation.py`` pins that the module stays
package-free; ``tests/test_certifier_hermeticity.py`` pins the call's position in
the certifier that has the rule.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

#: The variables ``platform_support.user_config_root`` reads: ``XDG_CONFIG_HOME``
#: on POSIX, ``APPDATA`` on Windows. Both are redirected — one alone leaves the
#: other platform's real config root live.
THROWAWAY_ENV_VARS = ("XDG_CONFIG_HOME", "APPDATA")

#: How many times a reap is attempted before it is reported as a survivor.
REAP_ATTEMPTS = 5

#: Seconds to wait between attempts. The write being raced is a load-time write
#: by another library, so the wait is short and the count is what buys certainty.
REAP_PAUSE_S = 0.1


def isolated_config_root(
    prefix: str,
    *,
    attempts: int = REAP_ATTEMPTS,
    pause: float = REAP_PAUSE_S,
) -> str:
    """Create a throwaway config root, redirect the config environment into it.

    Returns the root as a ``str`` so callers can hand it straight to the
    functions that take a config root (``crash.handlers.install(root=...)``,
    ``app_gui.ensure_repo_default_paths``, the dialogs). The root is reaped on
    every exit path: this call registers the ``atexit`` hook and the caller reaps
    at its own teardown point, which between them cover both.
    """
    root = tempfile.mkdtemp(prefix=prefix)
    for name in THROWAWAY_ENV_VARS:
        os.environ[name] = root
    atexit.register(reap_isolated_root, root, attempts=attempts, pause=pause)
    return root


def reap_isolated_root(
    root: str | os.PathLike[str],
    *,
    attempts: int = REAP_ATTEMPTS,
    pause: float = REAP_PAUSE_S,
) -> bool:
    """Remove ``root``, retrying; report a survivor. True when it is gone.

    Deliberately Qt-free: at interpreter exit the C++ application may already be
    gone, and reaching for it there raises instead of helping. An already-absent
    root is success, not an error — a run that reaps at its teardown and again at
    interpreter exit must not warn about the second attempt.
    """
    path = Path(root)
    for attempt in range(attempts):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return True
        if attempt + 1 < attempts:
            time.sleep(pause)
    print(
        f"warning: {path} survived {attempts} reap attempts; remove it by hand",
        file=sys.stderr,
    )
    return False
