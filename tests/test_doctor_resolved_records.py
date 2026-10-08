"""Tests for the doctor's stale-resolved-records check (scripts/doctor.py).

A RESOLVED marker claims that specific landed work closed the entry, and the
records-hygiene doctrine requires that claim to cite its holding commit. This
runtime check surfaces the failure mode the suite-side record net cannot see
on its own: a citation that still RESOLVES in git but is not an ancestor of
HEAD (rewritten history, a rebased-away branch) — the record reads green
while the evidence HEAD shows is gone.

Division of labor pinned here: unresolvable citations are INFORMATIONAL at
runtime (a shallow clone cannot verify them; the suite-side net owns
strictness there), while resolvable-but-not-ancestor is the finding and
flips ok to False. Ancestry probes are read-only git commands against a
disposable tmp repo, so these tests stay hermetic.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.test_doctor_input_subtitles import (
    _full_report,
    _load_doctor,
    _minimal_report_for_next_steps,
)

INPUT_OK = {
    "ok": True,
    "input_dir": "/repo/Input",
    "exists": True,
    "scanned": 1,
    "utf8_count": 1,
    "fallback": [],
    "mixed": [],
    "blocked": [],
    "unreadable": [],
    "error": "",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo_with_side_branch(tmp_path: Path) -> tuple[Path, str, str]:
    """A disposable git repo whose HEAD history does not contain one resolvable
    commit: `ancestor` is HEAD's own hash, `orphan` lives on a side branch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "STATE.md").write_text("seed record\n", encoding="utf-8")
    _git(repo, "add", "STATE.md")
    _git(repo, "commit", "-qm", "ancestor commit")
    ancestor = _git(repo, "rev-parse", "HEAD")
    default_branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _git(repo, "checkout", "-qb", "side")
    (repo / "SIDE.md").write_text("side work\n", encoding="utf-8")
    _git(repo, "add", "SIDE.md")
    _git(repo, "commit", "-qm", "side commit")
    orphan = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", default_branch)
    return repo, ancestor, orphan


def _write_resolved_records(repo: Path, body: str) -> None:
    (repo / "STATE.md").write_text(body, encoding="utf-8")


def test_stale_citation_is_flagged_and_ancestor_citation_is_not(tmp_path: Path) -> None:
    """The core verdict split: a commit on a discarded branch is stale (ok
    flips False, entry names file/line/token); HEAD's own hash keeps ok True.
    Vacuity guards: the scan must actually have read both paragraphs."""
    doctor = _load_doctor()
    repo, ancestor, orphan = _repo_with_side_branch(tmp_path)
    _write_resolved_records(
        repo,
        "old entry RESOLVED (commit " + ancestor + ") done.\n\n"
        "fresh entry RESOLVED (commit " + orphan + ") done.\n",
    )

    status = doctor._resolved_records_status(repo)

    assert status["skipped"] is False
    assert status["scanned"] == 2  # vacuity: both RESOLVED paragraphs were read
    assert status["stale"] == [
        {"file": "STATE.md", "line": "3", "token": orphan}
    ]
    assert status["ok"] is False
    assert "1 stale" in status["detail"]


def test_ancestor_only_report_is_ok(tmp_path: Path) -> None:
    """Citing HEAD's own hash — the normal, healthy shape — stays green, and
    the unresolvable list stays empty."""
    doctor = _load_doctor()
    repo, ancestor, _orphan = _repo_with_side_branch(tmp_path)
    _write_resolved_records(
        repo,
        "entry RESOLVED (commit " + ancestor + ") done.\n\n"
        "second entry RESOLVED (commit " + ancestor[:12] + ") also done.\n",
    )

    status = doctor._resolved_records_status(repo)

    assert status["ok"] is True
    assert status["scanned"] == 2  # vacuity: both paragraphs were read
    assert status["stale"] == []
    assert status["unverifiable"] == []


def test_unresolvable_citation_is_informational_not_a_finding(tmp_path: Path) -> None:
    """A hash that does not resolve at all (shallow clone, typo) is reported
    informationally: ok stays True, because the suite-side record net owns
    unresolvable-citation strictness and the doctor must not double-fail a
    tree it cannot fully verify."""
    doctor = _load_doctor()
    repo, _ancestor, _orphan = _repo_with_side_branch(tmp_path)
    _write_resolved_records(
        repo,
        "entry RESOLVED (commit 0badc0de) done.\n",
    )

    status = doctor._resolved_records_status(repo)

    assert status["ok"] is True
    assert status["stale"] == []
    assert status["unverifiable"] == [
        {"file": "STATE.md", "line": "1", "token": "0badc0de"}
    ]


def test_no_git_repo_skips_cleanly(tmp_path: Path) -> None:
    """Without .git, ancestry cannot be verified at all: skip with ok True
    rather than failing a tree for evidence the doctor cannot gather."""
    doctor = _load_doctor()
    bare = tmp_path / "no-git"
    bare.mkdir()
    (bare / "STATE.md").write_text("entry RESOLVED (commit 0badc0de)\n", encoding="utf-8")

    status = doctor._resolved_records_status(bare)

    assert status["skipped"] is True
    assert status["ok"] is True
    assert status["scanned"] == 0


def test_line_number_is_the_first_line_of_the_resolved_paragraph(tmp_path: Path) -> None:
    """The reported line anchors the paragraph, not the citation's offset
    within it — a reader must be able to jump straight to the entry."""
    doctor = _load_doctor()
    repo, _ancestor, orphan = _repo_with_side_branch(tmp_path)
    body = (
        "intro paragraph one.\n"
        "still the same paragraph, line two.\n\n"
        "mid report RESOLVED (commit " + orphan + ") done.\n"
    )
    _write_resolved_records(repo, body)

    status = doctor._resolved_records_status(repo)

    assert status["stale"][0]["line"] == "4"


def test_docs_records_are_scanned_and_every_occurrence_attributed(tmp_path: Path) -> None:
    """The every-occurrence idiom, runtime edition: one stale token cited in
    TWO records (root + docs/) is attributed in EACH — no first-record-only
    shortcut may drop the second attribution."""
    doctor = _load_doctor()
    repo, _ancestor, orphan = _repo_with_side_branch(tmp_path)
    findings = repo / "docs" / "superpowers" / "findings"
    findings.mkdir(parents=True)
    _write_resolved_records(repo, "root entry RESOLVED (commit " + orphan + ")\n")
    (findings / "2026-10-08-evidence.md").write_text(
        "docs entry RESOLVED (commit " + orphan + ")\n", encoding="utf-8"
    )

    status = doctor._resolved_records_status(repo)

    assert status["ok"] is False
    assert sorted((e["file"], e["token"]) for e in status["stale"]) == [
        ("STATE.md", orphan),
        ("docs/superpowers/findings/2026-10-08-evidence.md", orphan),
    ]


def test_human_report_fails_and_names_each_stale_entry(capsys) -> None:
    """The human report renders FAIL with the detail and one line per stale
    entry, so the operator sees file, line, and token without reading JSON."""
    doctor = _load_doctor()
    report = _full_report(INPUT_OK)
    report["resolved_records"] = {
        "ok": False,
        "skipped": False,
        "scanned": 2,
        "stale": [{"file": "STATE.md", "line": "12", "token": "0badc0de"}],
        "unverifiable": [],
        "detail": "2 RESOLVED entry(ies) checked; 1 stale",
    }

    doctor._print_human_report(report)

    out = capsys.readouterr().out
    assert "FAIL Resolved-record ancestry: 2 RESOLVED entry(ies) checked; 1 stale" in out
    assert "      STATE.md:12: cites 0badc0de — resolves in git but is NOT an ancestor of HEAD" in out


def test_human_report_passes_and_stays_quiet_when_clean(capsys) -> None:
    """No stale entries: PASS with the detail, and no per-entry lines. A
    rendered FAIL line here would mean the report block ignores `ok`."""
    doctor = _load_doctor()
    report = _full_report(INPUT_OK)
    report["resolved_records"] = {
        "ok": True,
        "skipped": False,
        "scanned": 3,
        "stale": [],
        "unverifiable": [{"file": "STATE.md", "line": "7", "token": "0badc0de"}],
        "detail": "3 RESOLVED entry(ies) checked",
    }

    doctor._print_human_report(report)

    out = capsys.readouterr().out
    assert "PASS Resolved-record ancestry: 3 RESOLVED entry(ies) checked" in out
    assert "NOT an ancestor" not in out


def test_next_steps_names_every_stale_occurrence() -> None:
    """Two stale records → two distinct next-steps, each naming file, line,
    and token with the re-verify remediation. A first-match-only regression
    would truncate the operator's remediation list; the vacuity probe (no
    stale → no step) proves the builder reads this exact report key."""
    doctor = _load_doctor()
    report = _minimal_report_for_next_steps({"ok": True, "problems": [], "error": "", "version": "9.9.9"})
    report["resolved_records"] = {
        "ok": False,
        "skipped": False,
        "scanned": 2,
        "stale": [
            {"file": "STATE.md", "line": "12", "token": "0badc0de"},
            {"file": "JUNO_FIXES.log", "line": "351", "token": "1badc0de"},
        ],
        "unverifiable": [],
        "detail": "2 RESOLVED entry(ies) checked; 2 stale",
    }

    steps = doctor._build_next_steps(report, ci_mode=True)

    stale_steps = [s for s in steps if "Stale-resolved record" in s]
    assert len(stale_steps) == 2
    assert any("STATE.md:12 cites 0badc0de" in s for s in stale_steps)
    assert any("JUNO_FIXES.log:351 cites 1badc0de" in s for s in stale_steps)
    assert any("not an ancestor of HEAD" in s for s in stale_steps)

    quiet = _minimal_report_for_next_steps({"ok": True, "problems": [], "error": "", "version": "9.9.9"})
    quiet["resolved_records"] = {"ok": True, "skipped": False, "scanned": 5, "stale": [], "unverifiable": []}
    assert not [s for s in doctor._build_next_steps(quiet, ci_mode=True) if "Stale-resolved record" in s]
