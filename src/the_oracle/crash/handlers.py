"""Crash handlers: sys.excepthook, threading.excepthook, faulthandler.

Fail-safe rules (CRASH_TELEMETRY_DESIGN §3): every handler body is wrapped so
a crash *inside the crash handler* writes nothing and never re-raises into
the Qt event loop (the failure mode that turns a hang into a hang-with-
modal). Consent is read at fire time — mid-session revocation takes effect
on the next event, not at next launch.

faulthandler is the piece that finally gives the GUI native-crash watch item
(STATE.md, 2026-09-08) a data path: a native segfault leaves the Python-side
stack in ``crash_reports/native-crash.txt`` where the external /tmp catcher
used to lose it. The faulthandler file is enabled only while consent is
on at enable time, and disarmed again by ``disable_faulthandler_catch`` on
opt-out (the CLI's opt-out path calls it). Faulthandler itself is C-level
and cannot be wrapped, so enable-time consent is the honest contract here —
unlike the record path above, which checks consent at fire time.

The Qt message handler is deliberately NOT installed here: MainWindow is the
in-flight surface of a concurrent extraction, and the design routes that
wiring through the GUI slice (design step 4), not the CLI-reachable core.
"""

from __future__ import annotations

import faulthandler
import sys
import threading
import traceback
from pathlib import Path

from the_oracle.crash import bundle, consent
from the_oracle.crash.record import build_record

_STATE: dict[str, object] = {}
_NATIVE_FILENAME = "native-crash.txt"


def _capture(
    *,
    exception_type: str,
    exception_message: str,
    traceback_frames: list[tuple[str, str, int]] | None,
    thread_name: str | None,
) -> Path | None:
    """One consent-checked capture. Never raises — that is the contract."""
    try:
        root = _root()
        if not consent.read_consent(root):
            return None
        log_tail: list[str] = []
        try:
            log_path = _log_file()
            if log_path.exists():
                log_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
        except OSError:
            log_tail = []
        record = build_record(
            exception_type=exception_type,
            exception_message=exception_message,
            traceback_frames=traceback_frames,
            log_tail=log_tail,
            thread_name=thread_name,
            edition=_edition(),
        )
        return bundle.write_record(root, record)
    except Exception:  # noqa: BLE001 - a failing crash handler must stay silent
        return None


def _sys_excepthook(exc_type, exc_value, exc_tb) -> None:  # noqa: ANN001
    """Chain to the previous hook first: Oracle capture must never suppress
    the interpreter's own reporting."""
    previous = _STATE.get("previous_sys_hook")
    if callable(previous):
        try:
            previous(exc_type, exc_value, exc_tb)
        except Exception:  # noqa: BLE001
            pass
    frames = [
        (frame.filename, frame.name, frame.lineno)
        for frame in traceback.extract_tb(exc_tb)
    ]
    _capture(
        exception_type=exc_type.__name__ if exc_type is not None else "Exception",
        exception_message=str(exc_value),
        traceback_frames=frames,
        thread_name=None,
    )


def _threading_excepthook(args: threading.ExceptHookArgs) -> None:
    previous = _STATE.get("previous_threading_hook")
    if callable(previous):
        try:
            previous(args)
        except Exception:  # noqa: BLE001
            pass
    frames = [
        (frame.filename, frame.name, frame.lineno)
        for frame in traceback.extract_tb(args.exc_traceback)
    ]
    _capture(
        exception_type=args.exc_type.__name__ if args.exc_type is not None else "Exception",
        exception_message=str(args.exc_value),
        traceback_frames=frames,
        thread_name=args.thread.name if args.thread is not None else None,
    )


def install(root: str | Path | None = None, *, log_file: Path | None = None) -> None:
    """Install both exception hooks (idempotent: the previous hooks are
    remembered once; re-invocation does not stack)."""
    _STATE.setdefault("previous_sys_hook", sys.excepthook)
    _STATE.setdefault("previous_threading_hook", threading.excepthook)
    _STATE.setdefault("root_override", root)
    _STATE.setdefault("log_file_override", log_file)
    sys.excepthook = _sys_excepthook
    threading.excepthook = _threading_excepthook


def _root() -> Path:
    override = _STATE.get("root_override")
    if override is not None:
        return Path(override)
    from the_oracle.offline import repo_root

    return repo_root()


def _log_file() -> Path:
    override = _STATE.get("log_file_override")
    if override is not None:
        return Path(override)
    from the_oracle.utils.logging import default_log_file

    return default_log_file()


def _edition() -> str | None:
    try:
        from the_oracle.licensing import current_license

        return current_license().edition
    except Exception:  # noqa: BLE001 - diagnostics never block diagnostics
        return None


def enable_faulthandler_catch(root: str | Path) -> bool:
    """Arm faulthandler for native crashes — only while consent is on.

    Returns False (and creates nothing) when consent is off: an opted-out
    install gets no capture machinery, not silent capture. The dump file
    lives inside crash_reports/ so ``clear_records``-style purging and the
    doctor's cap reporting cover it; opt-out disarms via
    ``disable_faulthandler_catch``."""
    if not consent.read_consent(root):
        return False
    try:
        dump_path = Path(root) / bundle.crash_dir(root).name / _NATIVE_FILENAME
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(dump_path, "a", encoding="utf-8")
        faulthandler.enable(file=handle)
        _STATE["native_dump_handle"] = handle
        _STATE["native_dump_path"] = dump_path
        return True
    except Exception:  # noqa: BLE001
        return False


def disable_faulthandler_catch() -> None:
    handle = _STATE.pop("native_dump_handle", None)
    if handle is not None:
        try:
            faulthandler.disable()
        except Exception:  # noqa: BLE001
            pass
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass


def current_handlers_installed() -> bool:
    return sys.excepthook is _sys_excepthook and threading.excepthook is _threading_excepthook
