"""Record-integrity net for the campaign's evidence trail.

``STATE.md`` and ``JUNO_FIXES.log`` cite commit hashes as the evidence behind
every record claim (``(commit 317d2f4)``, ``a320ab2 -> 96f5081``,
`` `5d26f1a` ``). A citation that no longer resolves in git — rewritten
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
RECORD_FILES = ("STATE.md", "JUNO_FIXES.log")

#: Hash-shaped tokens that are not commit citations. Classification is
#: deliberate and additive: prose that happens to spell itself in hex joins
#: this list with a comment, and an unclassifiable token fails the net loudly
#: (with this list named in the message) rather than being absorbed.
_NON_HASH_TOKENS = frozenset({"ed25519"})


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

    The failure message names file and line per broken citation, and an
    unclassifiable hash-shaped token is a failure too (see ``_NON_HASH_TOKENS``)
    — a citation that cannot be checked must never pass silently.
    """
    reason = _history_unavailable_reason()
    if reason:
        pytest.skip("record-integrity net: " + reason)

    per_file: dict[str, list[str]] = {}
    for name in RECORD_FILES:
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
    for name in RECORD_FILES:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        lines = text.splitlines()
        for token in sorted(set(per_file[name])):
            if _resolves(token):
                continue
            line = next(
                (i + 1 for i, row in enumerate(lines) if token in row), "?"
            )
            broken.append(f"{name}:{line}: {token}")

    assert broken == [], (
        "these commit hashes are cited in the records but do not resolve in "
        "git — the evidence trail is broken (rewritten history, a typo, or a "
        "hash from another machine). Fix the citation, or if the token is "
        "prose rather than a hash, classify it in _NON_HASH_TOKENS:\n  "
        + "\n  ".join(broken)
    )


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
