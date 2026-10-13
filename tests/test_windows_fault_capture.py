"""Pin the Windows certifier native-fault capture job in ci.yml.

The Windows certification steps die on the runner with rc=3221225477
(0xC0000005, access violation) and EMPTY stdout and stderr, so the job pinned
here is the only place CI can produce the faulting frame at all. Every pin
below is a way that evidence could silently stop existing again:

* without ``PYTHONFAULTHANDLER``, the process dies with no Python stack
  anywhere (the suite already shows stdout/stderr empty on this fault);
* without ``-u``, the certifier's progress buffer dies with it, so the log
  cannot say which aspect the fault happened in;
* with ``--aspects`` the run would no longer be the byte-for-byte repro the
  suite's hermeticity test observed failing at 3221225477;
* without ``if: always()`` on the upload, the crash step's own failure would
  skip the artifact — exactly how the chrome certification step got skipped;
* without the manual gate, every push would go red on a hunt job and say
  nothing new;
* writing a consent file here would manufacture a consent decision, which is
  the one thing the crash-capture design forbids (consent is absent in CI
  because crash_consent.json is gitignored, and the capture must run in that
  state).
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
JOB = "windows-certifier-fault-capture"


def _job() -> dict:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    assert JOB in jobs, (
        f"the Windows fault capture job is gone from ci.yml; without it the "
        f"0xC0000005 certifier crash has no evidence path on Windows: {sorted(jobs)}"
    )
    return jobs[JOB]


def _steps(job: dict) -> list[dict]:
    return job["steps"]


def _step_containing(job: dict, needle: str) -> dict:
    for step in _steps(job):
        if needle in str(step.get("run", "")):
            return step
    raise AssertionError(f"no step in {JOB} runs {needle!r}")


def test_job_is_windows_and_manual_only() -> None:
    """The hunt runs on the failing OS, and only when dispatched by hand."""
    job = _job()
    assert job["runs-on"] == "windows-latest"
    assert "workflow_dispatch" in str(job.get("if", "")), (
        "the capture job must stay dispatch-only: it is a crash hunt that is "
        "expected to be red until the fault is fixed"
    )


def test_capture_step_arms_python_faulthandler_on_stderr() -> None:
    """PYTHONFAULTHANDLER is the only consent-free way to get the frame.

    The certifier's own dump needs consent, and consent is absent in a fresh
    checkout (crash_consent.json is gitignored), so Python's stderr handler
    must be the one that prints the faulting stack.
    """
    step = _step_containing(_job(), "scripts/certify_gui.py")
    env = step.get("env", {})
    assert env.get("PYTHONFAULTHANDLER") == "1", (
        "without PYTHONFAULTHANDLER the faulting frame is never written: the "
        "suite observes stdout and stderr both empty on 0xC0000005"
    )
    assert env.get("QT_QPA_PLATFORM") == "offscreen", (
        "the crash was observed under the offscreen platform; running any "
        "other platform would capture a different (or no) fault"
    )


def test_capture_runs_the_hermeticity_repro_unbuffered_and_complete() -> None:
    """Byte-for-byte the run the suite watched die at 3221225477.

    ``-u`` because a process killed by an unhandled exception loses whatever
    sat in its stdout buffer — which is why the suite's detail block showed an
    empty stdout tail for a sweep that prints per aspect.
    """
    step = _step_containing(_job(), "scripts/certify_gui.py")
    run = str(step["run"])
    assert "-u" in run, "the certifier must run unbuffered or the aspect progress dies with it"
    assert "--aspects" not in run, (
        "the repro that crashed in the suite is the FULL sweep (no --aspects); "
        "scoping it would change the fault we are hunting"
    )


def test_dump_is_uploaded_even_when_the_crash_step_fails() -> None:
    """A crash step exits nonzero; the artifact must still ship.

    This is the lesson of the skipped chrome certification: anything after a
    failing step never runs unless it is explicitly unconditional.
    """
    job = _job()
    uploads = [
        step for step in _steps(job) if "upload-artifact" in str(step.get("uses", ""))
    ]
    assert uploads, f"{JOB} uploads no artifact, so the frame lives only in a log that expires"
    upload = uploads[0]
    assert str(upload.get("if", "")).strip() == "always()", (
        "the artifact upload must run even when the capture step crashes"
    )
    assert "certifier-fault.txt" in str(upload.get("with", {}).get("path", ""))


def test_capture_step_timeout_is_set() -> None:
    """The full sweep must be bounded: a hung certifier must not eat the job."""
    step = _step_containing(_job(), "scripts/certify_gui.py")
    assert "timeout-minutes" in step, "the certifier step needs its own timeout"


def test_the_capture_never_manufactures_consent() -> None:
    """Privacy pin: capture with the checkout's own (absent) consent state.

    ``crash_consent.json`` is gitignored, so CI is consent-off by construction.
    A job that wrote one would be creating a consent decision out of nothing —
    the exact failure branch consent.read_consent() is built to refuse.
    """
    job = _job()
    for step in _steps(job):
        run = str(step.get("run", ""))
        assert "crash_consent" not in run, (
            f"step {step.get('name')!r} writes or reads a consent file; the "
            f"capture must run in CI's own consent-off state"
        )


def test_bootstrap_runs_before_the_certifier() -> None:
    """The capture invokes ``.venv\\Scripts\\python.exe``; the venv is bootstrap's."""
    job = _job()
    names = [str(step.get("name", "")) for step in _steps(job)]
    bootstrap = next(
        (i for i, name in enumerate(names) if "Bootstrap" in name), None
    )
    capture = next(
        (
            i
            for i, step in enumerate(_steps(job))
            if "scripts/certify_gui.py" in str(step.get("run", ""))
        ),
        None,
    )
    assert bootstrap is not None, f"{JOB} never bootstraps the venv it runs the certifier with"
    assert capture is not None, f"{JOB} never runs the certifier"
    assert bootstrap < capture, "the certifier must run after the bootstrap that creates .venv"
