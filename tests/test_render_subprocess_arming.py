"""The render child arms native-crash capture like every other launch path.

``render_subprocess.main`` is the spawned clean interpreter the GUI delegates
renders to (``python -m the_oracle.render_subprocess``) — precisely so native
model code runs OUT of the Qt process. That made it the last unarmed process
class: the parent's ``cli.main`` arm lives in a different process and a
faulthandler cannot see across a process boundary, so a native segfault
during the child's own pipeline init or render previously left no dump
anywhere (the child's ``run_job`` catches ``BaseException`` for ordinary
Python failures; a SIGSEGV cannot reach it).

The contract pinned here mirrors the cli.main launch-path pin: the child
arms BEFORE any job work, fail-closed without consent (no handle, not even
the crash_reports/ directory), and the arm call is pinned in ``main`` ahead
of the preview/render dispatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import crash, render_subprocess  # noqa: E402
from the_oracle.crash import handlers as crash_handlers  # noqa: E402


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch) -> Path:
    from the_oracle import offline

    monkeypatch.setattr(offline, "repo_root", lambda: tmp_path)
    # Fresh handler state: sibling crash tests install targeted handlers and
    # leave root_override/native handles behind, and _root() prefers the
    # override — a stale one would silently point this test's arm at another
    # root while the assertion failed only in full-suite order.
    crash_handlers._STATE.clear()
    yield tmp_path
    crash_handlers.disable_faulthandler_catch()


def _run_child(sandbox: Path) -> tuple[int, Path]:
    """Invoke the real child main with a missing job file: the job path is
    read inside run_job, so this exercises the full launch sequence — arming,
    then job dispatch, then the child's own BaseException result protocol."""
    result = sandbox / "out" / "result.json"
    exit_code = render_subprocess.main(["--job", str(sandbox / "missing-job.json"), "--result", str(result)])
    return exit_code, result


def test_render_child_arms_on_a_consented_launch(sandbox: Path) -> None:
    crash.write_consent(sandbox, True)
    exit_code, result = _run_child(sandbox)
    assert exit_code == 1  # the missing-job failure is the child's own protocol
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["ok"] is False and "FileNotFoundError" in payload["error"], (
        "the job must actually have run — an arm that never reaches dispatch proves nothing"
    )
    assert crash_handlers._STATE.get("native_dump_handle") is not None, (
        "the render child must come up armed on a consented launch — the parent's "
        "arming cannot see across the process boundary"
    )
    dump = crash_handlers._STATE["native_dump_path"]
    assert str(sandbox) in str(dump) and "native-crash.txt" in str(dump), (
        "the dump must land in THIS checkout's crash_reports/, which the doctor caps and purges"
    )


def test_render_child_fails_closed_without_consent(sandbox: Path) -> None:
    exit_code, _ = _run_child(sandbox)  # no consent file: opted-out by default
    assert exit_code == 1
    assert crash_handlers._STATE.get("native_dump_handle") is None
    assert not (sandbox / "crash_reports").exists(), (
        "a consent-off child creates no capture machinery — the enable-time contract"
    )


def test_render_child_arming_is_pinned_in_main() -> None:
    """One-owner pin with vacuity: the arm call must sit in render_subprocess.main
    BEFORE the preview/render dispatch (a segfault during pipeline init is the
    exact class being covered). A scan that sees neither string proves nothing."""
    source = Path(render_subprocess.__file__).read_text(encoding="utf-8")
    assert "crash_handlers.arm_native_capture()" in source
    assert "return run_preview_job(" in source  # vacuity: the scan sees real code
    assert source.index("crash_handlers.arm_native_capture()") < source.index(
        "return run_preview_job("
    ), "the child must arm before any job work — an after-dispatch arm covers nothing"
