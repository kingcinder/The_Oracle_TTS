"""Two policies that used to be spelled out at every call site have one owner each.

"Naming" is where a subtitle's companion files go, and "wording" is the advice
the CLI and the GUI both show for voicing a cast. Both were re-implemented per
call site and both had already drifted: the converted-script suffix was written
out four times and two copies silently dropped the fallback for non-subtitle
inputs, and the GUI told the user to "edit the label in the file" where the CLI
went on to "(e.g. to a name or 'Speaker X')".

So these tests are about ownership rather than behaviour. Rendered output is
pinned in the modules' own terms, but the load-bearing assertions are the source
scans: a policy with one owner is one that no other module re-spells. A rename
that moves the constant but leaves a copy behind fails here, which is the only
way this stays true a year from now.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "the_oracle"

NAMING_OWNER = SRC / "subtitle_targets.py"
WORDING_OWNER = SRC / "speaker_ref_hints.py"


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """Non-comment source lines, so a comment cannot count as a second owner.

    Comments *refer* to these literals on purpose (the transformer's dataclass
    documents the flag shape it now delegates), which is documentation rather
    than a competing implementation.
    """
    lines = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        code = line.split("#", 1)[0]
        if code.strip():
            lines.append((number, code))
    return lines


def _modules_spelling(literal_pattern: str, *, owner: Path) -> list[str]:
    """Every module outside ``owner`` whose *code* contains the pattern."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path == owner or "__pycache__" in path.parts:
            continue
        for number, code in _code_lines(path):
            if re.search(literal_pattern, code):
                rel = path.relative_to(REPO_ROOT)
                found.append(f"{rel}:{number}: {code.strip()}")
                break
    return found


def _function_source(path: Path, name: str) -> str:
    """The text of ``def name`` up to the next `def` line at the same indent."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(rf"\s*def {name}\(", line)), None)
    assert start is not None, f"{name} not found in {path}"
    indent = len(lines[start]) - len(lines[start].lstrip())
    for end in range(start + 1, len(lines)):
        stripped = lines[end].lstrip()
        if stripped.startswith("def ") and len(lines[end]) - len(stripped) == indent:
            return "\n".join(lines[start:end])
    return "\n".join(lines[start:])


# --- the naming policy ----------------------------------------------------------


@pytest.mark.parametrize("pattern", [r'"\.srt\.txt"', r'"\.vtt\.txt"', r"'\.srt\.txt'"])
def test_the_converted_script_name_is_built_in_one_place(pattern: str) -> None:
    assert _modules_spelling(pattern, owner=NAMING_OWNER) == []


def test_the_subtitle_sidecar_name_is_built_in_one_place() -> None:
    """``with_suffix(".srt")`` is the other half of the naming, so it is owned too."""
    assert _modules_spelling(r'with_suffix\(\s*["\']\.srt["\']\s*\)', owner=NAMING_OWNER) == []


def test_the_naming_scan_can_actually_see_its_literals() -> None:
    """Guards the guard: a scan that matched nothing would pass forever."""
    owner_code = "\n".join(code for _, code in _code_lines(NAMING_OWNER))

    assert '".srt.txt"' in owner_code
    assert '".vtt.txt"' in owner_code


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("movie.srt", "movie.srt.txt"),
        ("episode.vtt", "episode.vtt.txt"),
        ("EPISODE.VTT", "EPISODE.vtt.txt"),
        ("a.b.srt", "a.b.srt.txt"),
        ("notes.md", "notes.txt"),
    ],
)
def test_converted_script_names_are_unchanged(name: str, expected: str) -> None:
    """The names the rest of the project and its tests already depend on."""
    from the_oracle.subtitle_targets import converted_script_target

    assert converted_script_target(name).name == expected


def test_sidecar_subtitles_replace_the_render_suffix() -> None:
    from the_oracle.subtitle_targets import subtitle_sidecar_target

    assert subtitle_sidecar_target("out/episode.flac").name == "episode.srt"
    assert subtitle_sidecar_target("out/part 1.wav").name == "part 1.srt"


# --- the wording policy ---------------------------------------------------------


@pytest.mark.parametrize(
    "pattern",
    [r'"--speakerA-ref PATH"', r'"--speakerB-ref PATH"', r"'--speakerA-ref PATH'"],
)
def test_the_speaker_flag_forms_are_built_in_one_place(pattern: str) -> None:
    """These were written out twice inside the transformer alone."""
    assert _modules_spelling(pattern, owner=WORDING_OWNER) == []


def test_the_wording_scan_can_actually_see_its_literals() -> None:
    owner_code = "\n".join(code for _, code in _code_lines(WORDING_OWNER))

    assert '"--speakerA-ref PATH"' in owner_code


@pytest.mark.parametrize("surface", ["cli.py", "app_gui.py"])
def test_no_surface_spells_the_hint_wording_out(surface: str) -> None:
    """Both display surfaces must render the owner's sentences, not their own.

    Checked by source rather than by rendering, because the point is that the
    text is not *written* here -- rendering a copy would still pass an
    output-only test on the day it was written.
    """
    text = (SRC / surface).read_text(encoding="utf-8")

    assert "is not accepted as a speaker label" not in text, (
        f"{surface} restates the rejected-label advice; it should render "
        "the_oracle.speaker_ref_hints instead"
    )
    assert "Speaker voices to provide" not in text, (
        f"{surface} restates the hint header; it should render "
        "the_oracle.speaker_ref_hints instead"
    )


def test_each_surface_calls_the_owner() -> None:
    cli_hints = _function_source(SRC / "cli.py", "_print_speaker_ref_hints")
    gui_hints = _function_source(SRC / "app_gui.py", "_speaker_ref_hint_lines")

    assert "hint_lines(" in cli_hints
    assert "sentence_lines(" in gui_hints


def test_the_two_surfaces_produce_the_same_sentences() -> None:
    """The drift that existed: one surface stopped short of the other's advice."""
    from the_oracle.speaker_ref_hints import hint_lines, sentence_lines

    refs = [{"speaker": "winston", "voice_key": "A", "flag": "--speakerA-ref PATH", "new": False}]
    rejected = ["Chapter", {"label": "Assistant", "rename_flag": "--speaker-ref C=PATH"}]

    cli_lines = hint_lines(refs, rejected)
    gui_lines = sentence_lines(refs, rejected, bullet="  \u2022 ")

    # Identical apart from the leading bullet, which is the one thing a surface
    # is allowed to choose.
    assert [line.lstrip("  -!\u2022") for line in cli_lines] == [
        line.lstrip("  -\u2022") for line in gui_lines
    ]


def test_the_advice_for_a_rejected_label_is_identical_on_both_surfaces() -> None:
    """The exact sentence that used to differ."""
    from the_oracle.speaker_ref_hints import hint_lines, sentence_lines

    cli = [line for line in hint_lines([], ["Chapter"]) if "Chapter" in line]
    gui = [line for line in sentence_lines([], ["Chapter"], bullet="  \u2022 ") if "Chapter" in line]

    assert cli and gui
    assert cli[0].lstrip("  -!") == gui[0].lstrip("  \u2022")
    assert "Speaker X" in cli[0], "the fuller of the two versions is the one kept"


def test_additional_voices_are_the_ones_needing_a_new_flag() -> None:
    from the_oracle.speaker_ref_hints import is_additional_voice, voice_flag

    assert voice_flag("A") == "--speakerA-ref PATH"
    assert voice_flag("B") == "--speakerB-ref PATH"
    assert voice_flag("C") == "--speaker-ref C=PATH"
    assert not is_additional_voice("A")
    assert not is_additional_voice("B")
    assert is_additional_voice("C")


def test_the_owner_only_says_nothing_when_there_is_nothing_to_say() -> None:
    from the_oracle.speaker_ref_hints import hint_lines

    assert hint_lines([], []) == []
    # The header is dropped with no references, so a panel showing only warnings
    # does not open with a promise it does not keep.
    assert hint_lines([], ["Chapter"]) == [
        "  ! 'Chapter' is not accepted as a speaker label; no reference audio can "
        "attribute it \u2014 edit the label in the file (e.g. to a name or 'Speaker X')."
    ]
