"""The Studio render report: the first real entitlement gate, and its degrade path.

Phase 1 item 5 of the 1.4.0 release work. The requirement was "gate at least one
real feature behind the studio entitlement, or stop advertising a licensing
system that enforces nothing". This is the first branch, chosen as the *least
destructive* default: the capability is NEW in 1.4.0, so community loses no
workflow, and the gate is one call site (``render_report.STUDIO_ENTITLEMENT``).

What these tests pin, in both directions:

1. **Unlicensed (community)** — the command refuses with the offline purchase
   path, exits 1, prints nothing to stdout, and **writes nothing**: no report
   file, and the project directory is byte-identical afterwards.
2. **Licensed (studio)** — the same command produces a report whose numbers
   match the real artifacts (sample rate/channels/frames read back from the
   file, peak/clipping measured from samples, utterance and speaker counts from
   ``render_plan.json``).
3. **Expired studio** — degrades to community: refuses, and still destroys
   nothing (the render is untouched).

**Mutation contract:** removing ``policy.require`` from ``handle_render_report``
must fail ``test_community_install_refuses_and_writes_nothing``; making the
degrade path write the report anyway must fail
``test_expired_license_degrades_without_destroying_work``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests.licensing_fixtures import TEST_SEED, payload  # noqa: E402
from the_oracle import cli  # noqa: E402
from the_oracle.licensing import store, tokens  # noqa: E402

pytestmark = pytest.mark.usefixtures("test_license_registry")


def _make_render(project_dir: Path, *, sample_rate: int = 8000, channels: int = 1) -> Path:
    """A real FLAC plus a real render_plan.json, shaped like a pipeline output."""
    project_dir.mkdir(parents=True, exist_ok=True)
    frames = sample_rate  # one second
    tone = 0.25 * np.sin(2 * np.pi * 220 * np.arange(frames) / sample_rate)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    flac = project_dir / "take.flac"
    soundfile.write(str(flac), data, sample_rate, format="FLAC")
    plan = {
        "engine": "chatterbox",
        "utterances": [
            {
                "index": 0,
                "speaker": "A",
                "status": "ok",
                "duration_seconds": 0.5,
                "original_text": "One.",
            },
            {
                "index": 1,
                "speaker": "B",
                "status": "ok",
                "duration_seconds": 0.5,
                "original_text": "Two.",
            },
        ],
    }
    (project_dir / "render_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return flac


def _activate_studio(root: Path) -> None:
    token = tokens.mint(payload(), TEST_SEED)
    result = store.save_token(root, token, verify_before_save=lambda t: tokens.verify_token(t))
    assert result.state == "saved", result


def _write_expired_token_directly(root: Path) -> None:
    """Store an already-expired token the way a real one ages into that state.

    ``save_token`` correctly refuses a token that does not verify, so it cannot
    produce an expired-by-the-clock store: that state arises from a token that
    was valid at activation and whose ``exp`` has since passed. Writing the
    store file directly is the shape of that install (the same technique
    tests/test_doctor_licensing.py uses for the identical reason).
    """
    token = tokens.mint(payload(exp=int(time.time()) - 10), TEST_SEED)
    store.store_path(root).write_text(json.dumps({"token": token}), encoding="utf-8")


def _isolate(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Point the install root (and therefore the license store) at scratch.

    ``current_license()`` resolves the running install's root through
    ``the_oracle.offline.repo_root`` (imported inside the function), so patching
    that one seam isolates the whole licensing stack for the command.
    """
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("the_oracle.offline.repo_root", lambda: root)
    return root


def _run(project_dir: Path, *extra: str) -> int:
    return cli.main(["render-report", str(project_dir), *extra])


# --- community: refuses, and writes nothing ----------------------------------


def test_community_install_refuses_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlicensed: exit 1, purchase path on stderr, empty stdout, no file."""
    project = tmp_path / "project"
    _make_render(project)
    before = {path.name: path.read_bytes() for path in project.iterdir()}

    # No token anywhere: the install IS community (fail-closed default).
    _isolate(monkeypatch, tmp_path / "no-token-here")
    status = _run(project, "--json", "--out", str(tmp_path / "report.json"))

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out.strip() == "", "a refused report must not print a document"
    assert "activate" in captured.err  # the offline purchase path
    assert not (tmp_path / "report.json").exists(), "a refused report must write nothing"
    assert {path.name: path.read_bytes() for path in project.iterdir()} == before, (
        "the project directory must be untouched"
    )


def test_community_message_never_asks_for_a_network_action(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    _make_render(project)
    _isolate(monkeypatch, tmp_path / "no-token-here")
    assert _run(project) == 1
    err = capsys.readouterr().err.lower()
    for forbidden in ("go online", "connect to", "download", "sign in"):
        assert forbidden not in err, forbidden


# --- studio: it works, and the numbers are real ------------------------------


def test_studio_install_produces_a_report_matching_the_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    flac = _make_render(project, sample_rate=8000, channels=1)
    root = _isolate(monkeypatch, tmp_path / "install")
    _activate_studio(root)

    out = tmp_path / "report.json"
    assert _run(project, "--json", "--out", str(out)) == 0

    report = json.loads(capsys.readouterr().out)
    info = soundfile.info(str(flac))
    assert report["output"]["file"] == flac.name
    assert report["output"]["sample_rate"] == info.samplerate == 8000
    assert report["output"]["channels"] == info.channels == 1
    assert report["output"]["frames"] == info.frames
    assert report["output"]["duration_seconds"] == pytest.approx(1.0, abs=0.01)
    # A 0.25-amplitude sine: peak ~= -12 dBFS, no clipping.
    assert -12.5 < report["loudness"]["peak_dbfs"] < -11.5
    assert report["loudness"]["clipped_samples"] == 0
    # Counts come from the plan that was actually written.
    assert report["plan"]["utterances"] == 2
    assert report["plan"]["speakers"] == {"A": 1, "B": 1}
    assert report["plan"]["speaker_count"] == 2
    # The report file is the same document that was printed.
    assert json.loads(out.read_text(encoding="utf-8")) == report


def test_a_clipping_render_is_reported_as_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report has to be able to say something negative, or it certifies nothing."""
    project = tmp_path / "project"
    project.mkdir()
    frames = 1000
    loud = np.ones(frames, dtype="float32")  # full scale: clipped by definition
    soundfile.write(str(project / "take.flac"), loud, 8000, format="FLAC")
    (project / "render_plan.json").write_text(json.dumps({"utterances": []}), encoding="utf-8")
    root = _isolate(monkeypatch, tmp_path / "install")
    _activate_studio(root)

    assert _run(project, "--json") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["loudness"]["clipped_samples"] > 0
    assert any("clip" in warning for warning in report["warnings"])


def test_detector_availability_is_reported_not_assumed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The watermark line must not claim a detection nobody ran."""
    project = tmp_path / "project"
    _make_render(project)
    root = _isolate(monkeypatch, tmp_path / "install")
    _activate_studio(root)

    assert _run(project, "--json") == 0
    watermark = json.loads(capsys.readouterr().out)["watermark"]
    assert watermark["expected"] is True
    assert "detector_available" in watermark
    assert "does not run detection" in watermark["note"]


# --- expired: degrades, destroys nothing -------------------------------------


def test_expired_license_degrades_without_destroying_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Phase 1 promise, at the feature boundary: expired means community."""
    project = tmp_path / "project"
    _make_render(project)
    before = {path.name: path.read_bytes() for path in project.iterdir()}
    root = _isolate(monkeypatch, tmp_path / "install")
    _write_expired_token_directly(root)

    out = tmp_path / "report.json"
    assert _run(project, "--out", str(out)) == 1

    captured = capsys.readouterr()
    assert "expired" in captured.err
    assert not out.exists()
    assert {path.name: path.read_bytes() for path in project.iterdir()} == before


# --- the report itself -------------------------------------------------------


def test_missing_render_is_a_typed_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    root = _isolate(monkeypatch, tmp_path / "install")
    _activate_studio(root)

    assert _run(empty) == 2
    err = capsys.readouterr().err
    assert "no rendered .flac" in err
    assert "Traceback" not in err


def test_two_renders_in_one_folder_is_refused_rather_than_guessed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    _make_render(project)
    soundfile.write(str(project / "second.flac"), np.zeros(10, dtype="float32"), 8000, format="FLAC")
    root = _isolate(monkeypatch, tmp_path / "install")
    _activate_studio(root)

    assert _run(project) == 2
    assert "covers exactly one render" in capsys.readouterr().err


def test_the_report_is_read_only(tmp_path: Path) -> None:
    """No fixture, no CLI: build_report must not touch the project at all."""
    import the_oracle.render_report as render_report

    project = tmp_path / "project"
    _make_render(project)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in project.iterdir()}
    render_report.build_report(project)
    after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in project.iterdir()}
    assert after == before


def test_the_gate_is_one_call_site() -> None:
    """The product-key gate stays a single call site in the CLI entry point."""
    cli_source = Path(cli.__file__).read_text(encoding="utf-8")
    assert cli_source.count("require(current_license())") == 1, (
        "the product-key gate must stay a single call site so it cannot be "
        "bypassed by a copy that forgets the check"
    )
