from pathlib import Path

from the_oracle.voice_catalog import default_voice_choices, voice_catalog_audit

def test_default_voice_choices_are_capped_and_real_paths() -> None:
    choices = default_voice_choices("/home/oem/Documents/The_Oracle_TTS")

    assert len(choices) <= 10
    for choice in choices:
        assert choice.path


def test_default_voice_choices_prefer_repo_local_seashells(tmp_path: Path) -> None:
    seashell = tmp_path / "Seashells" / "oracle.wav"
    fallback = tmp_path / "build" / "smoke_render" / "fallback.wav"
    seashell.parent.mkdir(parents=True, exist_ok=True)
    fallback.parent.mkdir(parents=True, exist_ok=True)
    seashell.write_bytes(b"RIFF")
    fallback.write_bytes(b"RIFF")

    choices = default_voice_choices(tmp_path)

    assert choices
    assert Path(choices[0].path) == seashell.resolve()


def test_voice_catalog_audit_reports_fallback_sources_and_no_low_risk_mixing(tmp_path: Path) -> None:
    fallback = tmp_path / "build" / "smoke_render" / "fallback.wav"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_bytes(b"RIFF")

    audit = voice_catalog_audit(tmp_path)

    assert audit["primary_source"] == "build_fallbacks"
    assert audit["fallback_clip_count"] == 1
    assert audit["seashell_clip_count"] == 0
    assert audit["voice_mixing_low_risk"] is False
