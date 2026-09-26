"""The privacy CLI commands (the-oracle privacy-status / opt-in / opt-out).

The opt-in is the only write path that can enable capture; opt-out takes
effect immediately (fire-time consent) and --purge is an explicit flag, not
a prompt (the CLI's non-TTY no-prompts precedent — the confirm dialog
belongs to the GUI slice, per the design doc's decision record).

**Mutation contract (M-CLI-CONSENT):** making handle_privacy_opt_in skip the
consent write must fail test_opt_in_enables_capture_and_arms_faulthandler;
making opt-out skip the faulthandler disarm must fail
test_opt_out_disarms_the_native_catch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import cli, crash  # noqa: E402
from the_oracle.crash import handlers as crash_handlers  # noqa: E402
from the_oracle.crash import bundle  # noqa: E402


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch) -> Path:
    from the_oracle import offline

    monkeypatch.setattr(offline, "repo_root", lambda: tmp_path)
    return tmp_path


def _args(command: str, *extra: str):
    argv = [command, *extra]
    return cli.build_parser().parse_args(argv)


def test_status_defaults_to_opted_out(sandbox: Path, capsys) -> None:
    assert cli.handle_privacy_status() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["consent"] is False
    assert payload["records"] == 0
    assert payload["native_dump_present"] is False
    assert "nothing is ever uploaded" in payload["detail"]


def test_opt_in_enables_capture_and_arms_faulthandler(sandbox: Path, capsys) -> None:
    assert cli.handle_privacy_opt_in() == 0
    assert crash.read_consent(sandbox) is True
    assert crash_handlers._STATE.get("native_dump_handle") is not None
    crash_handlers.disable_faulthandler_catch()


def test_opt_out_disarms_the_native_catch(sandbox: Path, capsys) -> None:
    cli.handle_privacy_opt_in()
    assert crash_handlers._STATE.get("native_dump_handle") is not None
    assert cli.handle_privacy_opt_out(_args("privacy-opt-out")) == 0
    assert crash_handlers._STATE.get("native_dump_handle") is None
    assert crash.read_consent(sandbox) is False
    output = capsys.readouterr().out
    assert "kept" in output  # reports preserved without --purge
    assert "--purge" in output  # and the user is told how


def test_opt_out_purge_deletes_every_report(sandbox: Path, capsys) -> None:
    crash.write_consent(sandbox, True)
    bundle.write_record(sandbox, {"exception": {"type": "T", "message": "m"}, "log_tail": []})
    bundle.write_record(sandbox, {"exception": {"type": "T", "message": "m2"}, "log_tail": []})
    assert len(crash.list_records(sandbox)) == 2
    assert cli.handle_privacy_opt_out(_args("privacy-opt-out", "--purge")) == 0
    assert len(crash.list_records(sandbox)) == 0
    assert "Deleted 2" in capsys.readouterr().out


def test_status_reports_records_after_capture(sandbox: Path, capsys) -> None:
    crash.write_consent(sandbox, True)
    bundle.write_record(sandbox, {"exception": {"type": "ValueError", "message": "x"}, "log_tail": []})
    assert cli.handle_privacy_status() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["consent"] is True
    assert payload["records"] == 1
    assert payload["newest"] is not None
