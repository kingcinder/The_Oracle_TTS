from __future__ import annotations

from the_oracle.utils.pacing import chunk_seam_pause_ms, pause_for_utterance


def test_period_keeps_base_pause():
    assert pause_for_utterance("This is a calm line.", 180) == 180


def test_question_and_exclamation_lengthen():
    assert pause_for_utterance("Wait, really?", 180) == int(180 * 1.25)
    assert pause_for_utterance("Stop!", 180) == int(180 * 1.3)


def test_ellipsis_breathes_longest():
    assert pause_for_utterance("And then…", 180) == int(180 * 1.6)
    assert pause_for_utterance("And then...", 180) == int(180 * 1.6)


def test_clause_or_no_terminal_shortens():
    assert pause_for_utterance("He paused,", 180) == 126
    assert pause_for_utterance("and kept going", 180) == 126


def test_quotes_after_terminal_still_scaled():
    assert pause_for_utterance('No!"', 180) == int(180 * 1.3)
    assert pause_for_utterance('(Really?)', 180) == int(180 * 1.25)


def test_clamped_to_domain():
    assert 0 <= pause_for_utterance("…", 2000) <= 2000
    assert pause_for_utterance("…", 2000) == 2000


def test_chunk_seam_is_small_breath():
    assert chunk_seam_pause_ms(180) == 63
    assert chunk_seam_pause_ms(0) == 40
    assert chunk_seam_pause_ms(2000) == 700
