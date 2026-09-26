"""Logging utilities for CLI, GUI, and render jobs."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Retention (CRASH_TELEMETRY_DESIGN §5): the unbounded FileHandler this module
# used to create meant a long-lived GUI install appended to one file forever.
# Tests shrink these via monkeypatch; configure_logging reads them at call time.
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


# File handlers created by configure_logging(). logging.basicConfig(force=True)
# detaches old handlers from the root logger but never closes them, so every
# reconfigure leaked the log file's descriptor; closing the tracked handlers
# first keeps repeated configuration (CLI runs, GUI restarts, tests) FD-clean.
_created_file_handlers: list[logging.Handler] = []


def default_log_file() -> Path:
    """The repo-local rotating log, matching app_paths' repo-local convention.

    The repo root is resolved the way cli.py resolves it; one level deeper here
    because this module lives in utils/.
    """
    log_path = Path(__file__).resolve().parents[3] / "logs" / "oracle.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def configure_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    for handler in _created_file_handlers:
        try:
            handler.close()
        except Exception:
            pass
    _created_file_handlers.clear()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        _created_file_handlers.append(file_handler)
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
