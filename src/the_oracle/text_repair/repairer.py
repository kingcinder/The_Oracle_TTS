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

    def repair(self, text: str, mode: str = "moderate") -> RepairResult:
        normalized_mode = (mode or "moderate").strip().lower()
        aggressive = normalized_mode == "aggressive"
        corrections: list[CorrectionRecord] = []

        current = text
        if normalized_mode == "off":
            return RepairResult(text=text, corrections=[])

        normalized = normalize_text(current)
        if normalized != current:
            corrections.append(CorrectionRecord(stage="normalize", before=current, after=normalized))
            current = normalized

        if normalized_mode in {"aggressive", "moderate"}:
            punctuated = self.punctuator.restore(current)
            if punctuated != current:
                corrections.append(CorrectionRecord(stage="punctuation", before=current, after=punctuated))
                current = punctuated

        if normalized_mode in {"aggressive", "moderate", "mild"}:
            spelled = self.speller.correct(current, aggressive=aggressive)
            if spelled != current:
                corrections.append(CorrectionRecord(stage="spelling", before=current, after=spelled))
                current = spelled

        if normalized_mode in {"aggressive", "moderate"}:
            corrected = self.grammar.correct(current, aggressive=aggressive)
            if corrected != current:
                corrections.append(CorrectionRecord(stage="grammar", before=current, after=corrected))
                current = corrected

        return RepairResult(text=current, corrections=corrections)
