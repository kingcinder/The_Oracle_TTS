from dualvoice_studio.voice_catalog import default_voice_choices


def test_default_voice_choices_are_capped_and_real_paths() -> None:
    choices = default_voice_choices("/home/oem/Documents/The_Oracle_TTS")

    assert len(choices) <= 10
    for choice in choices:
        assert choice.path
