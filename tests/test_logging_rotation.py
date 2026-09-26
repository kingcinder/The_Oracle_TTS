"""Log rotation and the repo-local default log path (CRASH_TELEMETRY_DESIGN §5).

The design contract promises: `configure_logging` keeps its public signature
and FD-clean reconfigure semantics while capping growth (5 MiB x 3 backups in
production), and a new `default_log_file()` helper returns the repo-local
rotating log path. These tests shrink the caps via the module constants so the
rotation cycle is exercised without writing megabytes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import the_oracle.utils.logging as oracle_logging
from the_oracle.utils.logging import configure_logging, default_log_file


def _flush_root_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


class TestLogRotation:
    def test_rotates_at_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(oracle_logging, "LOG_MAX_BYTES", 2_000)
        monkeypatch.setattr(oracle_logging, "LOG_BACKUP_COUNT", 3)
        log_file = tmp_path / "oracle.log"
        configure_logging(log_file)

        logger = logging.getLogger("oracle-rotation")
        for _ in range(200):
            logger.info("x" * 100)
        _flush_root_handlers()

        assert (tmp_path / "oracle.log.1").exists(), (
            "a rollover backup should exist once the cap is passed"
        )
        # One record of slack is expected: rollover triggers before the write
        # that would exceed the cap, so the live file stays at/below cap size.
        assert log_file.stat().st_size <= 2_200, "the live log must not grow unbounded"

    def test_backups_are_bounded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(oracle_logging, "LOG_MAX_BYTES", 2_000)
        monkeypatch.setattr(oracle_logging, "LOG_BACKUP_COUNT", 3)
        log_file = tmp_path / "oracle.log"
        configure_logging(log_file)

        logger = logging.getLogger("oracle-rotation-bounded")
        for _ in range(600):
            logger.info("y" * 100)
        _flush_root_handlers()

        assert (tmp_path / "oracle.log.3").exists(), "rotation should keep the configured backups"
        assert not (tmp_path / "oracle.log.4").exists(), (
            "rotation must keep at most LOG_BACKUP_COUNT backups"
        )

    def test_reconfigure_is_fd_clean_with_rotation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(oracle_logging, "LOG_MAX_BYTES", 2_000)
        monkeypatch.setattr(oracle_logging, "LOG_BACKUP_COUNT", 3)
        log_a = tmp_path / "a.log"
        log_b = tmp_path / "b.log"
        configure_logging(log_a)
        first_handlers = list(logging.getLogger().handlers)
        configure_logging(log_b)
        configure_logging(log_b)

        file_handlers = [
            h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == str(log_b)
        old_file_handlers = [
            h for h in first_handlers if isinstance(h, logging.FileHandler)
        ]
        assert old_file_handlers, "expected a file handler from the first configure"
        for handler in old_file_handlers:
            stream = handler.stream
            assert stream is None or stream.closed, (
                "detached handler must be closed (FD leak)"
            )


class TestDefaultLogFile:
    def test_default_log_file_is_repo_local(self):
        path = default_log_file()

        assert path.name == "oracle.log"
        assert path.parent.name == "logs"
        assert path.parent.is_dir(), "the logs/ directory is created on demand"
        repo_root = Path(oracle_logging.__file__).resolve().parents[3]
        assert path == repo_root / "logs" / "oracle.log"
