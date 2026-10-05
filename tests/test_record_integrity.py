"""Record-integrity net for the campaign's evidence trail.

The records — ``STATE.md``, ``JUNO_FIXES.log``, and every tracked root or
``docs/`` prose file discovered carrying citations — cite commit hashes as
the evidence behind every record claim (``(commit 317d2f4)``,
``a320ab2 -> 96f5081``, `` `5d26f1a` ``). A citation that no longer resolves in git — rewritten
history, a typo, or a hash recorded from another machine — makes the record
misleading at exactly the moment someone trusts it to verify a claim. This net
runs the whole trail through git on every pass.

Environmental honesty: the net needs this repository's git history. In a
source archive (no ``.git``) or a shallow clone (history past the boundary is
simply absent) it skips with a visible reason — neither failing for something
the tree cannot know, nor passing blind.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
#: The minimum record set the net must always discover. Discovery (below)
#: sweeps new records in automatically, but if the discovered set ever
#: drops below this floor — a record deleted from the repo, a record
#: emptied of citations, or the discovery rule broken — the net fails
#: until the shrink is acknowledged here. Removal is a floor edit, not an
#: absorption.
RECORD_SET_FLOOR = frozenset(
    {
        "STATE.md",
        "JUNO_FIXES.log",
        "RELEASE-CHECKLIST-V1.3.2.md",
        "docs/superpowers/findings/2026-09-28-crash-evidence.md",
        "docs/superpowers/plans/2026-09-25-consumer-market-readiness.md",
        # (the 2026-09-28 crash-eradication plan is NOT floored: its only
        # hash-shaped tokens were prose — a date and a placeholder stem in a
        # synthetic filename — so it carries no citations to verify)
        "docs/superpowers/specs/2026-09-06-voice-craft-and-recording-studio-design.md",
        "docs/superpowers/specs/2026-09-25-consumer-market-readiness-design.md",
    }
)

#: Hash-shaped tokens that are not commit citations. Classification is
#: deliberate and additive: prose that happens to spell itself in hex joins
#: this list with a comment, and an unclassifiable token fails the net loudly
#: (with this list named in the message) rather than being absorbed.
_NON_HASH_TOKENS = frozenset(
    {
        "ed25519",  # a signing-algorithm name (mixed case never matches; lowercase does)
        "1697156",  # a PID in a kernel-log segfault quote (python[1697156]: …) — 7 digits slips the >8 all-digit rule
        "3724240",  # core-dump ProcStatus process ids (Pid=/PPid=/NSpgid=/NSsid=), crash-evidence findings
        "3724241",  # …same quote
        "3724243",  # …same quote
        "20260928",  # the date inside a synthetic crash-record filename (crash-20260928-…json), campaign plan
        "aaaaaaaa",  # the placeholder stem of that same synthetic filename
    }
)


def _cited_hashes(text: str) -> list[str]:
    """Extract commit-hash citations from record prose.

    Excluded by rule, each proven form-by-form below: decimal numbers long
    enough to look hex-ish (the CI run id ``35039888506``), null register
    dumps (``ip 0000000000000000``, ``0000000``), and mixed-case words the
    lowercase-only match never sees (``Ed25519``). All-digit tokens of hash
    length stay in — ``3808064`` and ``7336349`` are real commits.
    """
    hashes: list[str] = []
    for token in re.findall(r"\b[0-9a-f]{7,40}\b", text):
        if token.isdigit() and len(token) > 8:
            continue
        if set(token) == {"0"}:
            continue
        if token in _NON_HASH_TOKENS:
            continue
        hashes.append(token)
    return hashes


def _tracked_record_files(repo: Path = REPO_ROOT) -> tuple[str, ...]:
    """Discover the record set: core records + every tracked cited record.

    A record is tracked prose at the repo root (``*.md``/``*.log``) or a
    ``docs/**.md`` findings/spec/plan document that carries at least one
    hash-shaped token — citing commits is what makes a file this net's
    business, so a new record is swept in the moment it is committed, with
    no list to edit. Untracked files are invisible by construction, and a
    tracked file missing from the worktree cannot be scanned — discovery
    drops it, and the floor (not a ``FileNotFoundError``) reports it.
    """
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    candidates = [
        path
        for path in sorted(tracked)
        if (
            ("/" not in path and path.endswith((".md", ".log")))
            or (path.startswith("docs/") and path.endswith(".md"))
        )
    ]
    return tuple(
        path
        for path in candidates
        if (repo / path).is_file()
        and _cited_hashes((repo / path).read_text(encoding="utf-8"))
    )


def _record_set_floor_missing(
    discovered: set[str], floor: frozenset[str] = RECORD_SET_FLOOR
) -> list[str]:
    """Sorted floor entries the discovery set no longer contains."""
    return sorted(floor - discovered)


def _resolves(token: str) -> bool:
    """True when git resolves ``token`` to a commit in this repository."""
    probe = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "rev-parse",
            "--verify",
            "--quiet",
            f"{token}^{{commit}}",
        ],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _broken_citation_lines(lines: list[str], tokens: set[str]) -> list[str]:
    """Attribute each broken token to EVERY record line carrying it.

    A first-match-only report would name one line and hide the rest — the
    record must be fixed everywhere a stale hash is cited, so the return
    is one ``<line>: <token>`` entry per line, in token-major order.
    """
    attributions: list[str] = []
    for token in sorted(tokens):
        if _resolves(token):
            continue
        attributions.extend(
            f"{i + 1}: {token}" for i, row in enumerate(lines) if token in row
        )
    return attributions


def _history_unavailable_reason() -> str | None:
    """``None`` when the net can run; otherwise why this tree cannot."""
    if not (REPO_ROOT / ".git").exists():
        return (
            "this tree is a source archive without .git; commit citations "
            "cannot be verified here (supported tree — run the net in a clone)"
        )
    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--is-shallow-repository"],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0 and probe.stdout.strip() == "true":
        return (
            "this clone is shallow; cited commits may predate the history "
            "boundary, which is indistinguishable from a stale citation — "
            "run with full history (git fetch --unshallow)"
        )
    return None


def test_every_commit_hash_cited_in_the_records_resolves() -> None:
    """Every hash the records cite must still exist in git.

    The failure message names the file and EVERY line carrying each broken
    citation (a hash cited on several lines is broken on all of them), and
    an unclassifiable hash-shaped token is a failure too (see
    ``_NON_HASH_TOKENS``) — a citation that cannot be checked must never
    pass silently.

    The record set is discovered, not listed: root ``*.md``/``*.log`` and
    tracked ``docs/**.md`` files carrying citations join automatically, and
    ``RECORD_SET_FLOOR`` fails the net if the discovered set ever shrinks.
    """
    reason = _history_unavailable_reason()
    if reason:
        pytest.skip("record-integrity net: " + reason)

    record_files = _tracked_record_files()
    missing = _record_set_floor_missing(set(record_files))
    assert not missing, (
        "the record set shrank below its floor — these files are no longer "
        "discovered (deleted from the repo or the worktree, emptied of "
        "citations, or the discovery rule broke). A record's removal is a "
        "deliberate act that must be acknowledged in RECORD_SET_FLOOR: "
        + ", ".join(missing)
    )

    per_file: dict[str, list[str]] = {}
    for name in record_files:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        per_file[name] = _cited_hashes(text)

    # Vacuity: the extractor must demonstrably see the trail in both records.
    # If these trip, the scan went blind — fix the extractor (or deliberate
    # prose classification), not these floors.
    assert len(per_file["STATE.md"]) >= 5, (
        f"only {len(per_file['STATE.md'])} hash citations seen in STATE.md; "
        "the extractor went blind — fix the scanner, not this assertion."
    )
    assert len(per_file["JUNO_FIXES.log"]) >= 10, (
        f"only {len(per_file['JUNO_FIXES.log'])} hash citations seen in "
        "JUNO_FIXES.log; the extractor went blind — fix the scanner, not "
        "this assertion."
    )

    broken: list[str] = []
    for name in record_files:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        for attribution in _broken_citation_lines(
            text.splitlines(), set(per_file[name])
        ):
            broken.append(f"{name}:{attribution}")

    assert broken == [], (
        "these commit hashes are cited in the records but do not resolve in "
        "git — the evidence trail is broken (rewritten history, a typo, or a "
        "hash from another machine). Fix the citation, or if the token is "
        "prose rather than a hash, classify it in _NON_HASH_TOKENS:\n  "
        + "\n  ".join(broken)
    )


def test_a_broken_citation_is_attributed_to_every_line_carrying_it() -> None:
    """Synthetic pin: one stale hash cited on two lines is reported twice.

    A first-match-only attribution would let a hash "fixed" on one line
    stay broken on another while the net keeps pointing at a single spot.
    The stale tokens here are pure fiction; the pin never touches git
    resolution of real history (the probe itself is pinned form-by-form
    in ``test_the_resolution_probe_can_pass_and_can_fail``).
    """
    lines = [
        "2026-09-28 | files (commit deadbeef) | stale | red.",
        "2026-09-28 | other (commit deadbe1) | stale pair | red.",
        "2026-09-28 | again (commit deadbeef) | same stale hash | red.",
    ]
    attributions = _broken_citation_lines(lines, {"deadbeef", "deadbe1"})
    # Token-major order, and BOTH lines of the twice-cited hash are named —
    # the non-adjacent pair is the case a first-match report would miss.
    assert attributions == [
        "2: deadbe1",
        "1: deadbeef",
        "3: deadbeef",
    ], attributions


def test_the_hash_extractor_reads_every_citation_shape_form_by_form() -> None:
    """Form-by-form proofs the extractor reads what the net claims.

    The live records only exercise a few citation spellings, so a new
    spelling (or a new hex-shaped piece of prose) could be missed or
    misread with every pass green. These forms run in every CI pass.
    """
    forms: dict[str, list[str]] = {
        "(commit 317d2f4) | the entry": ["317d2f4"],
        "fast-forwarded local main a320ab2 -> 96f5081.": ["a320ab2", "96f5081"],
        "the thread-cluster slice `5d26f1a` is the follow-on": ["5d26f1a"],
        "(b94ec88) reset the sidebar": ["b94ec88"],
        "full hash 2db5360a1b2c3d4e5f60718293a4b5c6d7e8f90a cited": [
            "2db5360a1b2c3d4e5f60718293a4b5c6d7e8f90a"
        ],
        "3808064 and 7336349 are all-digit commits": ["3808064", "7336349"],
        "0000000 is the null object": [],
        "run 35039888506 on `main` is green": [],
        "`python: segfault at 0 ip 0000000000000000`": [],
        "python[1697156]: segfault at 0 ip 0 … error 14": [],  # a process id, not a commit — see _NON_HASH_TOKENS
        "dated 2026-09-27 with 1394 passed / 0 failed.": [],
        "signing uses Ed25519 and ed25519 keys": [],
    }
    for text, expected in forms.items():
        assert _cited_hashes(text) == expected, f"extractor misreads {text!r}"


def test_the_resolution_probe_can_pass_and_can_fail() -> None:
    """The probe must be able to say both answers, or the net is theater."""
    reason = _history_unavailable_reason()
    if reason:
        pytest.skip("record-integrity net: " + reason)

    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert _resolves(head), "the probe cannot even resolve HEAD"
    assert not _resolves("0000000deadbeef"), (
        "the probe claims a nonexistent hash resolves; it cannot fail"
    )


# --- record discovery + shrink floor (disposable repo) ----------------------------


def _tmp_record_repo(tmp_path: Path) -> Path:
    """A disposable tracked tree: two core records, one cited docs record,
    one zero-citation docs file, and one cited-but-untracked decoy."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
    (repo / "STATE.md").write_text("records (commit 0badc0de)\n", encoding="utf-8")
    (repo / "JUNO_FIXES.log").write_text(
        "2026-10-05 | x (commit 0badc0de) | d | green.\n", encoding="utf-8"
    )
    findings = repo / "docs" / "superpowers" / "findings"
    findings.mkdir(parents=True)
    (findings / "2026-10-05-evidence.md").write_text(
        "evidence cites (commit 0badc0de)\n", encoding="utf-8"
    )
    (findings / "2026-10-05-empty.md").write_text(
        "no citations here\n", encoding="utf-8"
    )
    (repo / "decoy.md").write_text(
        "untracked decoy (commit 0badc0de)\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "add", "STATE.md", "JUNO_FIXES.log", "docs"],
        cwd=str(repo),
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", "records"], cwd=str(repo), check=True)
    return repo


def test_discovery_sweeps_cited_records_and_skips_untracked_and_empty(
    tmp_path: Path,
) -> None:
    """A new cited record is discovered with no list edit; untracked files
    are invisible and a zero-citation file is not this net's business."""
    repo = _tmp_record_repo(tmp_path)
    assert set(_tracked_record_files(repo)) == {
        "STATE.md",
        "JUNO_FIXES.log",
        "docs/superpowers/findings/2026-10-05-evidence.md",
    }


def test_the_floor_fails_when_a_record_file_disappears(tmp_path: Path) -> None:
    """A record deleted from the worktree (still tracked) can no longer be
    scanned, so discovery drops it — and the floor, not a crash, must name
    the shrink."""
    repo = _tmp_record_repo(tmp_path)
    discovered = set(_tracked_record_files(repo))
    (repo / "docs" / "superpowers" / "findings" / "2026-10-05-evidence.md").unlink()
    shrunk = set(_tracked_record_files(repo))
    assert shrunk < discovered, "the shrink scenario must actually shrink"
    assert _record_set_floor_missing(shrunk, floor=frozenset(discovered)) == [
        "docs/superpowers/findings/2026-10-05-evidence.md"
    ]


def test_the_floor_fails_when_a_record_is_emptied_of_citations(tmp_path: Path) -> None:
    """The subtler shrink: the file still exists but no longer carries a
    trail — discovery drops it, and the floor must fail just as loudly."""
    repo = _tmp_record_repo(tmp_path)
    (repo / "STATE.md").write_text("no citations remain\n", encoding="utf-8")
    shrunk = set(_tracked_record_files(repo))
    assert "STATE.md" not in shrunk
    assert _record_set_floor_missing(
        shrunk, floor=frozenset({"STATE.md", "JUNO_FIXES.log"})
    ) == ["STATE.md"]
