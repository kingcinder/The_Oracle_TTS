"""The sanitizer — every redaction rule, plus the property probe.

**Mutation contract (M-SANITIZE):** deleting rule 1 (home prefix → ~) from
sanitize_text must fail test_property_no_output_contains_the_home_prefix.
Deleting a path-kind rule must fail its dedicated test.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle.crash import sanitize  # noqa: E402

HOME = "/home/cody"


def test_home_prefix_becomes_tilde() -> None:
    out = sanitize.sanitize_text(f"{HOME}/Documents/the oracle tts/crash_reports/x.json", home=HOME)
    # The two-stage collapse (~, then <path>) keeps nothing structured; the
    # contract is: tilde form, and no home prefix or username anywhere.
    assert out.startswith("~")
    assert "/home" not in out
    assert "cody" not in out


def test_input_output_seashells_profiles_become_kinds() -> None:
    assert sanitize.sanitize_text("rendered Input/scene.txt to Output/scene.flac", home=HOME) == \
        "rendered <input>scene.txt to <output>scene.flac"
    assert sanitize.sanitize_text("voice Seashells/hero.wav loaded", home=HOME) == \
        "voice <voice>hero.wav loaded"
    assert sanitize.sanitize_text("profile Profiles/default.json saved", home=HOME) == \
        "profile <profile>default.json saved"


def test_remaining_absolute_paths_collapse() -> None:
    out = sanitize.sanitize_text("failed to read /opt/secret/config.ini", home=HOME)
    assert "/opt/secret" not in out
    assert "<path>" in out


def test_property_no_output_contains_the_home_prefix() -> None:
    """**M-SANITIZE target:** the net fails if the home rule disappears."""
    samples = [
        f"{HOME}/x.wav",
        f"file://{HOME}/Input/a.txt",
        f"trace: {HOME}/.venv/lib/mod.py:12",
        f"{HOME}/Documents/the oracle tts/Output/b.flac",
        "no path here",
    ]
    for text in samples:
        out = sanitize.sanitize_text(text, home=HOME)
        assert HOME not in out, (text, out)
        assert "cody" not in out, (text, out)


def test_long_lines_are_truncated() -> None:
    long_text = "transcript-like filler " * 30
    out = sanitize.sanitize_text(long_text, home=HOME)
    assert len(out) <= sanitize.MAX_FIELD_CHARS


def test_log_tail_cap_and_order() -> None:
    lines = [f"line {i}" for i in range(80)]
    tail = sanitize.sanitize_log_tail(lines, home=HOME)
    assert len(tail) == sanitize.MAX_LOG_TAIL_LINES
    assert tail[-1] == "line 79"
    assert tail[0] == "line 30"


def test_frames_carry_location_not_content() -> None:
    frame = sanitize.sanitize_frame(f"{HOME}/src/the_oracle/pipeline.py", "render_one", 412, home=HOME)
    assert frame.endswith(":412: render_one")
    assert "src" not in frame or "<path>" in frame or frame.startswith("~")
    # A frame never carries source text or variable values by construction.


def test_empty_input_stays_empty() -> None:
    assert sanitize.sanitize_text("") == ""
    assert sanitize.sanitize_log_tail([]) == []
