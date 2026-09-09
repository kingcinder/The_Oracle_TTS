"""Tests for the word~word pain-point markers in input scripts.

A '~' glued between two words marks a junction where the TTS engine hung up or
mispronounced. The marker is an annotation for the review table: it must be
stripped from synthesis text in every correction mode (including Verbatim),
while unattached '~' characters are read verbatim.
"""

from __future__ import annotations

from the_oracle.models.project import Utterance, strip_pain_point_markers


def test_strip_removes_marker_between_words() -> None:
    assert strip_pain_point_markers("syncronized~lockstep") == "syncronized lockstep"
    assert strip_pain_point_markers("atmospheric~soup from") == "atmospheric soup from"
    assert strip_pain_point_markers("just what~speakers your") == "just what speakers your"


def test_strip_leaves_attached_tilde_alone() -> None:
    # A '~' at a word edge is genuine content (e.g. an approximation "~42"),
    # not a marker between two words.
    assert strip_pain_point_markers("about ~42 units") == "about ~42 units"
    assert strip_pain_point_markers("42~ units") == "42~ units"


def test_strip_is_idempotent_and_safe_on_plain_text() -> None:
    assert strip_pain_point_markers("plain narration line") == "plain narration line"
    once = strip_pain_point_markers("a~b and ~x")
    assert strip_pain_point_markers(once) == once


def test_utterance_text_for_tts_strips_markers_in_verbatim_mode() -> None:
    utterance = Utterance(
        index=0,
        original_text="syncronized~lockstep with one another",
        repaired_text="syncronized~lockstep with one another",  # Verbatim passthrough
    )
    assert utterance.text_for_tts() == "syncronized lockstep with one another"
    # The review table keeps the author's annotation.
    assert utterance.repaired_text == "syncronized~lockstep with one another"


def test_utterance_text_for_tts_falls_back_to_original_and_strips() -> None:
    utterance = Utterance(
        index=0,
        original_text="atmospheric~soup",
        repaired_text="",
    )
    assert utterance.text_for_tts() == "atmospheric soup"
