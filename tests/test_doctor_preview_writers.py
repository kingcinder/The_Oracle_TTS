"""Tests for the doctor's preview-cache audit (scripts/doctor.py).

The writer-manifest net (``tests/test_stem_cache_write_path.py``) pins the
SOURCE side of preview ownership: ``OraclePipeline.render_preview`` alone
writes preview audio, naming every file through ``ProjectCache.preview_path``
(``preview_<alnum-speaker>_<index>.wav`` inside the project's ``previews/``
directory). The doctor's check is the DISK side: it scans every project
``previews/`` directory under the repo (disposable ``build/`` excluded) and
classifies each file against ``ProjectCache.is_gated_preview_name`` — a file
that fails the predicate is runtime evidence of a writer outside the gated
owner, or a renamed scheme the recognizer no longer matches.

The tests pin the classification (including the >= 10000 index case, where
the gated writer itself pads to five digits), the read-only contract, the
human-report rendering, the next-steps wiring, and — most load-bearing — the
identity between the writer and its recognizer: everything
``ProjectCache.preview_path`` produces must satisfy ``is_gated_preview_name``,
or the audit flags the gated owner's own output.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.test_doctor_input_subtitles import (
    _full_report,
    _load_doctor,
    _minimal_report_for_next_steps,
)

DOCTOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "doctor.py"


# --- classification ----------------------------------------------------------


def test_classifies_gated_and_foreign_preview_files(tmp_path: Path) -> None:
    doctor = _load_doctor()
    # A gated project's previews/ dir (nested, proving the scan is recursive):
    gated = tmp_path / "Output" / "my_project" / "previews"
    gated.mkdir(parents=True)
    (gated / "preview_A_0000.wav").write_bytes(b"")
    (gated / "preview_Julia_0042.wav").write_bytes(b"")
    # Foreign shapes: a stem written into the wrong directory, a stray text
    # file, and a hand-named file missing the scheme's shape.
    (gated / "ab12cd34.wav").write_bytes(b"")
    (gated / "notes.txt").write_text("scratch", encoding="utf-8")
    # The writer's sanitizer keeps non-ASCII alnum (preview_path('Zoë')) —
    # the recognizer must accept the writer's own output for any alphabet.
    (gated / "preview_Zoë_0000.wav").write_bytes(b"")
    (gated / "preview_speaker one_0000.wav").write_bytes(b"")
    # An unrelated previews/ directory that is empty: scanned, zero files.
    (tmp_path / "other" / "previews").mkdir(parents=True)

    status = doctor._preview_writers_status(tmp_path)

    assert status["dirs_scanned"] == 2
    assert status["scanned"] == 6
    assert [entry["name"] for entry in status["foreign"]] == [
        "ab12cd34.wav",
        "notes.txt",
        "preview_speaker one_0000.wav",
    ]
    assert status["ok"] is True, "a foreign file is informational, not a broken install"


def test_build_directories_are_excluded_from_the_scan(tmp_path: Path) -> None:
    """Disposable smoke-harness project trees live under build/; their naming
    drift is meaningless, so they must not reach the report."""
    doctor = _load_doctor()
    smoke = tmp_path / "build" / "doctor_smoke" / "render_project" / "previews"
    smoke.mkdir(parents=True)
    (smoke / "whatever.wav").write_bytes(b"")

    status = doctor._preview_writers_status(tmp_path)

    assert status["dirs_scanned"] == 0
    assert status["scanned"] == 0
    assert status["foreign"] == []
    assert status["ok"] is True


def test_five_digit_index_is_still_the_gated_writer(tmp_path: Path) -> None:
    """:04d pads to AT LEAST four digits, so preview_path at index >= 10000
    emits five — the recognizer must accept the gated writer's own output."""
    doctor = _load_doctor()
    previews = tmp_path / "proj" / "previews"
    previews.mkdir(parents=True)
    (previews / "preview_Narrator_10000.wav").write_bytes(b"")

    status = doctor._preview_writers_status(tmp_path)

    assert status["foreign"] == []


def test_unreadable_previews_dir_fails_the_check(tmp_path: Path) -> None:
    doctor = _load_doctor()
    previews = tmp_path / "proj" / "previews"
    previews.mkdir(parents=True)
    (previews / "preview_A_0000.wav").write_bytes(b"")
    # Make the directory unreadable AFTER planting the file.
    previews.chmod(0o000)
    try:
        status = doctor._preview_writers_status(tmp_path)
    finally:
        previews.chmod(0o755)
    assert status["ok"] is False
    assert status["unreadable"], status
    assert "previews" in status["error"]


# --- read-only contract --------------------------------------------------------


def test_no_previews_dirs_reported_and_none_created(tmp_path: Path) -> None:
    doctor = _load_doctor()
    before = sorted(p.name for p in tmp_path.iterdir())

    status = doctor._preview_writers_status(tmp_path)

    assert status["dirs_scanned"] == 0 and status["scanned"] == 0
    assert status["foreign"] == [] and status["ok"] is True
    assert sorted(p.name for p in tmp_path.iterdir()) == before, (
        "the audit must not create anything (ProjectCache eagerly mkdirs; "
        "this check must not)"
    )


# --- writer/recognizer identity ------------------------------------------------


def test_every_preview_path_output_satisfies_the_recognizer() -> None:
    """THE load-bearing pin: the recognizer mirrors ProjectCache.preview_path.

    If the naming scheme ever changes in the writer, this fails here — before
    the doctor's disk audit starts flagging the gated owner's own files.
    """
    from the_oracle.models.cache import ProjectCache

    cache = ProjectCache.__new__(ProjectCache)  # no mkdir side effects
    cache.project_dir = Path("/unused")
    cache.preview_dir = Path("/unused/previews")
    for speaker in ("A", "Narrator", "Julia", "", "!!spaced speaker!!", "Zoë"):
        for index in (0, 1, 42, 9999, 10000, 123456):
            name = cache.preview_path(speaker, index).name
            assert ProjectCache.is_gated_preview_name(name), (
                f"preview_path({speaker!r}, {index}) -> {name!r} no longer "
                "matches the gated naming scheme the doctor audits against"
            )


def test_recognizer_rejects_near_miss_shapes() -> None:
    """Shapes a foreign writer could plausibly produce must not pass."""
    from the_oracle.models.cache import ProjectCache

    for name in (
        "ab12cd34.wav",  # a stem hash
        "preview_.wav",  # no speaker/index
        "preview_A.wav",  # no index
        "preview_A_0.wav",  # index below the padded floor
        "preview_A_0000.txt",  # wrong extension
        "PREVIEW_A_0000.wav",  # wrong prefix casing
        "preview_A_0000_final.wav",  # suffix junk
    ):
        assert not ProjectCache.is_gated_preview_name(name), name


# --- human report --------------------------------------------------------------


def test_human_report_warns_and_details_foreign_files(capsys) -> None:
    doctor = _load_doctor()
    report = _full_report(
        {"ok": True, "input_dir": "/repo/Input", "exists": False, "scanned": 0, "utf8_count": 0, "fallback": [], "blocked": [], "unreadable": []}
    )
    report["preview_writers"] = {
        "ok": True,
        "scanned": 3,
        "dirs_scanned": 1,
        "foreign": [{"path": "Output/proj/previews/ab12cd34.wav", "name": "ab12cd34.wav"}],
        "unreadable": [],
        "error": "",
    }
    doctor._print_human_report(report)
    out = capsys.readouterr().out
    assert "WARN Preview cache audit:" in out
    assert "ab12cd34.wav" in out
    assert "gated preview owner" in out


def test_human_report_passes_when_all_files_are_gated(capsys) -> None:
    doctor = _load_doctor()
    report = _full_report(
        {"ok": True, "input_dir": "/repo/Input", "exists": False, "scanned": 0, "utf8_count": 0, "fallback": [], "blocked": [], "unreadable": []}
    )
    report["preview_writers"] = {"ok": True, "scanned": 2, "dirs_scanned": 1, "foreign": [], "unreadable": [], "error": ""}
    doctor._print_human_report(report)
    out = capsys.readouterr().out
    assert "PASS Preview cache audit: 1 previews/ dir(s) scanned, 2 file(s)" in out


def test_human_report_skips_when_no_dirs_and_handles_older_reports(capsys) -> None:
    doctor = _load_doctor()
    report = _full_report(
        {"ok": True, "input_dir": "/repo/Input", "exists": False, "scanned": 0, "utf8_count": 0, "fallback": [], "blocked": [], "unreadable": []}
    )
    report["preview_writers"] = {"ok": True, "scanned": 0, "dirs_scanned": 0, "foreign": [], "unreadable": [], "error": ""}
    doctor._print_human_report(report)
    assert "SKIP Preview cache audit" in capsys.readouterr().out
    # A report from before this check existed (a full report simply lacking
    # the key): render nothing, no crash.
    doctor._print_human_report(
        _full_report({"ok": True, "input_dir": "/repo/Input", "exists": False, "scanned": 0, "utf8_count": 0, "fallback": [], "blocked": [], "unreadable": []})
    )
    assert "Preview cache audit" not in capsys.readouterr().out


# --- next steps + wiring ---------------------------------------------------------


def test_next_steps_flag_foreign_preview_files() -> None:
    doctor = _load_doctor()
    report = _minimal_report_for_next_steps({"blocked": [], "unreadable": []})
    report["preview_writers"] = {"foreign": [{"path": "Output/proj/previews/stray.wav"}]}
    steps = doctor._build_next_steps(report, ci_mode=False)
    assert any("stray.wav" in step and "gated preview owner" in step for step in steps)


def test_next_steps_quiet_when_no_foreign_files() -> None:
    doctor = _load_doctor()
    report = _minimal_report_for_next_steps({"blocked": [], "unreadable": []})
    report["preview_writers"] = {"foreign": []}
    assert not any("Preview cache" in step for step in doctor._build_next_steps(report, ci_mode=False))


def test_check_is_wired_into_the_report() -> None:
    """Vacuity guard: the check must actually run in run()'s report."""
    doctor_src = DOCTOR_PATH.read_text(encoding="utf-8")
    assert '"preview_writers": _preview_writers_status(repo_root),' in doctor_src
    assert doctor_src.count("_preview_writers_status(") >= 2  # definition + call site
