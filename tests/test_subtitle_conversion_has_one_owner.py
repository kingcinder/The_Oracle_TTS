"""The convert-not-overwrite subtitle policy has one owner.

The CLI render path and the GUI file picker each used to carry their own copy
of "decode, detect, convert, reuse-or-fail" -- and the copies had drifted: the
GUI's decode was strict, so a CP1252-encoded subtitle was blocked in the GUI
while the same file converted fine from the CLI. ``srt_ingest.
ensure_subtitle_script`` now owns the policy and both render paths are
presentation wrappers around it.

Like the naming/wording ownership tests, the load-bearing assertions are
source scans: a policy with one owner is one no other module re-implements.
A future render path that converts subtitles by hand fails here instead of
drifting again.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "the_oracle"

POLICY_OWNER = SRC / "srt_ingest.py"
REPORT_OWNER = SRC / "ingest_transformer.py"


def _code_lines(path: Path) -> list[str]:
    """Non-comment source lines, so a comment cannot count as a second owner."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        code = line.split("#", 1)[0]
        if code.strip():
            lines.append(code)
    return lines


def _modules_containing(pattern: str, *, owner: Path) -> list[str]:
    """Every module outside ``owner`` whose *code* matches the pattern."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path == owner or "__pycache__" in path.parts:
            continue
        for code in _code_lines(path):
            if re.search(pattern, code):
                found.append(f"{path.relative_to(REPO_ROOT)}: {code.strip()}")
                break
    return found


def _function_source(path: Path, name: str) -> str:
    """The text of ``def name`` up to the next ``def`` at the same indent."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if re.match(rf"\s*def {name}\(", line)), None
    )
    assert start is not None, f"{name} not found in {path}"
    indent = len(lines[start]) - len(lines[start].lstrip())
    for end in range(start + 1, len(lines)):
        stripped = lines[end].lstrip()
        if stripped.startswith("def ") and len(lines[end]) - len(stripped) == indent:
            return "\n".join(lines[start:end])
    return "\n".join(lines[start:])


# --- the conversion policy ------------------------------------------------------


def test_only_the_owner_converts_subtitles() -> None:
    """``convert_srt_file`` writes dialogue scripts; only srt_ingest may call it.

    Every render path that needs a conversion must go through
    ``ensure_subtitle_script`` -- that is what keeps the GUI and the CLI from
    converting (or reusing, or failing to decode) differently.
    """
    assert _modules_containing(r"convert_srt_file\(", owner=POLICY_OWNER) == []


def test_both_render_paths_delegate_to_the_shared_core() -> None:
    """The CLI wrapper and the GUI wrapper call the same policy function."""
    cli_wrapper = _function_source(SRC / "cli.py", "_maybe_convert_srt")
    gui_wrapper = _function_source(SRC / "app_gui.py", "_convert_subtitle_input")

    assert "ensure_subtitle_script" in cli_wrapper
    assert "ensure_subtitle_script" in gui_wrapper
    # Neither wrapper re-implements the policy's steps by hand.
    for wrapper_name, wrapper in (("_maybe_convert_srt", cli_wrapper), ("_convert_subtitle_input", gui_wrapper)):
        assert "looks_like_srt" not in wrapper, f"{wrapper_name} detects subtitles itself"
        assert "decode(" not in wrapper, f"{wrapper_name} decodes the file itself"


def test_the_policy_scan_can_actually_see_its_literals() -> None:
    """Guards the guard: a scan that matched nothing would pass forever."""
    owner_code = "\n".join(_code_lines(POLICY_OWNER))

    assert "convert_srt_file(" in owner_code
    assert "def ensure_subtitle_script(" in owner_code


# --- the speaker-ref report data -------------------------------------------------


def test_the_report_data_is_composed_in_one_place() -> None:
    """``suggest_speaker_refs`` feeds every surface through the transformer.

    The report used to be a CLI private the GUI imported across module
    boundaries; now the data lives with the transformer and both surfaces
    consume it, so the cast a CLI report names is the cast the GUI names.
    """
    assert _modules_containing(r"suggest_speaker_refs\(", owner=REPORT_OWNER) == []
    assert _modules_containing(r"suggest_rejected_label_refs\(", owner=REPORT_OWNER) == []


def test_the_report_scan_can_actually_see_its_literals() -> None:
    report_code = "\n".join(_code_lines(REPORT_OWNER))

    assert "suggest_speaker_refs(" in report_code
    assert "suggest_rejected_label_refs(" in report_code
