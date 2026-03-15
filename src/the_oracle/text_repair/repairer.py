"""Orchestrates normalization, punctuation, spelling, and grammar repair."""

from __future__ import annotations

from dataclasses import dataclass

from the_oracle.models.project import CorrectionRecord
from the_oracle.text_repair.grammar import GrammarCorrector
from the_oracle.text_repair.normalization import normalize_text
from the_oracle.text_repair.punctuation import PunctuationRestorer
from the_oracle.text_repair.spelling import SpellCorrector


@dataclass(slots=True)
class RepairResult:
    text: str
    corrections: list[CorrectionRecord]


class TextRepairPipeline:
    def __init__(self) -> None:
        self.punctuator = PunctuationRestorer()
        self.speller = SpellCorrector()
        self.grammar = GrammarCorrector()

    def repair(self, text: str, mode: str = "conservative") -> RepairResult:
        aggressive = mode == "aggressive"
        corrections: list[CorrectionRecord] = []

        current = text
        normalized = normalize_text(current)
        if normalized != current:
            corrections.append(CorrectionRecord(stage="normalize", before=current, after=normalized))
            current = normalized

        punctuated = self.punctuator.restore(current)
        if punctuated != current:
            corrections.append(CorrectionRecord(stage="punctuation", before=current, after=punctuated))
            current = punctuated

        spelled = self.speller.correct(current, aggressive=aggressive)
        if spelled != current:
            corrections.append(CorrectionRecord(stage="spelling", before=current, after=spelled))
            current = spelled

        corrected = self.grammar.correct(current, aggressive=aggressive)
        if corrected != current:
            corrections.append(CorrectionRecord(stage="grammar", before=current, after=corrected))
            current = corrected

        return RepairResult(text=current, corrections=corrections)
