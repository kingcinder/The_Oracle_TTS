"""How a speaker-ref suggestion is spelled and worded -- in exactly one place.

Three surfaces show the same advice: the CLI prints it to stderr, the GUI's
format-health popup lists it, and the transformer computes the flags the advice
is built from. Every one of them used to spell the wording out itself, and they
had already drifted: the CLI told the user to edit a rejected label "to a name or
'Speaker X'", while the GUI stopped at "edit the label in the file", so the same
suggestion told two different stories depending on where it appeared. The flag
forms were written out twice inside the transformer alone.

So this module owns three things:

* the **flag form** for a voice key (``--speakerA-ref PATH`` for A/B, the
  ``KEY=PATH`` form for the rest) and which keys count as additional voices;
* the **sentences**, as data rather than as a rendered string, so a surface picks
  its own bullet without restating the words;
* the **tolerance for the shapes** the advice arrives in -- the CLI's report
  dicts and bare label strings both come through here, so no surface has to
  re-implement that unwrapping.

Bullets are arguments rather than constants because the surfaces genuinely
differ (the CLI uses ``- `` and ``! `` so a warning reads as a warning in a
terminal, the GUI uses one ``•`` for both), and that difference is the *only*
thing a surface is allowed to change.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

#: Printed once above the list, on every surface.
HEADER = "Speaker voices to provide (in first-appearance order):"

#: One bullet per reference suggestion, and one per warning.
DEFAULT_REFERENCE_BULLET = "  - "
DEFAULT_WARNING_BULLET = "  ! "

#: The two voices that have dedicated flags instead of the ``KEY=PATH`` form.
_DEDICATED_KEY_FLAGS = {"A": "--speakerA-ref PATH", "B": "--speakerB-ref PATH"}


@dataclass(frozen=True)
class ReferenceHint:
    """One voice to provide, and the exact flag that provides it."""

    speaker: str
    voice_key: str
    flag: str
    is_new: bool

    @property
    def marker(self) -> str:
        """``add`` for a flag the user must add, ``use`` for one they may reuse."""
        return "add" if self.is_new else "use"


def voice_flag(voice_key: str) -> str:
    """The exact flag that voices ``voice_key``.

    A and B have dedicated ``--speakerA-ref``/``--speakerB-ref`` flags; C and
    beyond use the ``--speaker-ref KEY=PATH`` form. Both spellings are what the
    CLI actually accepts, so the suggestion is always runnable as printed.
    """
    return _DEDICATED_KEY_FLAGS.get(voice_key, f"--speaker-ref {voice_key}=PATH")


def is_additional_voice(voice_key: str) -> bool:
    """Whether voicing ``voice_key`` needs a flag the user must add."""
    return voice_key not in _DEDICATED_KEY_FLAGS


def reference_hints(entries: Iterable[Any]) -> list[ReferenceHint]:
    """Normalise reference suggestions into :class:`ReferenceHint`.

    Accepts the CLI report's dicts (``speaker``/``voice_key``/``flag``/``new``)
    and objects that already are hints, because both cross this boundary.
    """
    hints: list[ReferenceHint] = []
    for entry in entries:
        if isinstance(entry, ReferenceHint):
            hints.append(entry)
        elif isinstance(entry, Mapping):
            hints.append(
                ReferenceHint(
                    speaker=str(entry["speaker"]),
                    voice_key=str(entry["voice_key"]),
                    flag=str(entry["flag"]),
                    is_new=bool(entry.get("new")),
                )
            )
        else:  # an object with the same attributes, e.g. the transformer's suggestion
            hints.append(
                ReferenceHint(
                    speaker=str(entry.speaker),
                    voice_key=str(entry.voice_key),
                    flag=str(entry.flag),
                    is_new=bool(entry.is_new),
                )
            )
    return hints


def rejected_hints(entries: Iterable[Any]) -> list[tuple[str, str | None]]:
    """Normalise rejected labels into ``(label, rename_flag or None)`` pairs.

    A rejected label reads as narration whoever voices it, so the advice is to
    rename it; ``rename_flag`` is the flag that then voices it, when the engine
    can work one out.
    """
    hints: list[tuple[str, str | None]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            hints.append((str(entry["label"]), entry.get("rename_flag")))
        else:
            hints.append((str(entry), None))
    return hints


def reference_sentence(hint: ReferenceHint) -> str:
    """``winston -> voice A: use --speakerA-ref PATH``."""
    return f"{hint.speaker} -> voice {hint.voice_key}: {hint.marker} {hint.flag}"


def rename_sentence(label: str, rename_flag: str) -> str:
    """Advice for a rejected label the engine can voice once it is renamed."""
    return (
        f"'{label}' is not accepted as a speaker label; no reference audio can "
        "attribute it as-is \u2014 rename the label in the file (e.g. to a name), "
        f"then provide {rename_flag}"
    )


def rejected_sentence(label: str) -> str:
    """Advice for a rejected label with no workable rename."""
    return (
        f"'{label}' is not accepted as a speaker label; no reference audio can "
        "attribute it \u2014 edit the label in the file (e.g. to a name or 'Speaker X')."
    )


def hint_lines(
    references: Iterable[Any],
    rejected: Iterable[Any] = (),
    *,
    reference_bullet: str = DEFAULT_REFERENCE_BULLET,
    warning_bullet: str = DEFAULT_WARNING_BULLET,
    header: str = HEADER,
    include_header: bool = True,
) -> list[str]:
    """The full advice, one line per entry, ready to print or display.

    Returns lines rather than a string so the CLI can print them to stderr and
    the GUI can put them in a list without either re-wording anything.
    """
    hints = reference_hints(references)
    warnings = rejected_hints(rejected)
    lines: list[str] = []
    if hints and include_header:
        lines.append(header)
    lines.extend(f"{reference_bullet}{reference_sentence(hint)}" for hint in hints)
    for label, rename_flag in warnings:
        sentence = rename_sentence(label, rename_flag) if rename_flag else rejected_sentence(label)
        lines.append(f"{warning_bullet}{sentence}")
    return lines


def sentence_lines(
    references: Iterable[Any],
    rejected: Iterable[Any] = (),
    *,
    bullet: str = DEFAULT_REFERENCE_BULLET,
) -> list[str]:
    """The advice with one bullet style throughout.

    For a surface that has a single list style -- the GUI's popup -- rather than
    the CLI's split between suggestions and warnings.
    """
    return hint_lines(references, rejected, reference_bullet=bullet, warning_bullet=bullet)


__all__ = [
    "HEADER",
    "DEFAULT_REFERENCE_BULLET",
    "DEFAULT_WARNING_BULLET",
    "ReferenceHint",
    "voice_flag",
    "is_additional_voice",
    "reference_hints",
    "rejected_hints",
    "reference_sentence",
    "rename_sentence",
    "rejected_sentence",
    "hint_lines",
    "sentence_lines",
]
