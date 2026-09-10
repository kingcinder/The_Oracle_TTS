"""Logging utilities for CLI, GUI, and render jobs."""

from __future__ import annotations

import logging
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


# File handlers created by configure_logging(). logging.basicConfig(force=True)
# detaches old handlers from the root logger but never closes them, so every
# reconfigure leaked the log file's descriptor; closing the tracked handlers
# first keeps repeated configuration (CLI runs, GUI restarts, tests) FD-clean.
_created_file_handlers: list[logging.Handler] = []


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
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        _created_file_handlers.append(file_handler)
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
