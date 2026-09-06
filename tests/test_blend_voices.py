"""Tests for the named blend-voice catalog (saved voices in the picker)."""

from pathlib import Path

from the_oracle.smoke import _write_reference
from the_oracle.voice_catalog import (
    blend_voice_choices,
    default_voice_choices,
    remove_blend_voice,
    save_blend_voice,
)


def _two_clips(tmp_path: Path) -> tuple[Path, Path]:
    clip_a = _write_reference(tmp_path / "voice_a.wav", 220.0)
    clip_b = _write_reference(tmp_path / "voice_b.wav", 440.0)
    return clip_a, clip_b


def test_save_blend_voice_writes_catalog_and_derived_clip(tmp_path: Path) -> None:
    profiles = tmp_path / "Profiles"
    clip_a, clip_b = _two_clips(tmp_path)

    choice = save_blend_voice(profiles, "Warm Narrator", clip_a, clip_b, weight=0.7, mode="mix")

    assert choice.kind == "blend"
    assert choice.label == "Warm Narrator"
    derived = Path(choice.path)
    assert derived.exists()
    assert derived.parent == profiles / ".blends"
    assert (profiles / "blend_voices.json").exists()


def test_save_blend_voice_is_deterministic_and_upserts_by_name(tmp_path: Path) -> None:
    profiles = tmp_path / "Profiles"
    clip_a, clip_b = _two_clips(tmp_path)

    first = save_blend_voice(profiles, "Narrator", clip_a, clip_b, weight=0.5, mode="mix")
    again = save_blend_voice(profiles, "Narrator", clip_a, clip_b, weight=0.5, mode="mix")
    assert first.path == again.path  # same inputs -> same derived clip

    # Same name with different inputs replaces the entry (upsert).
    clip_c = _write_reference(tmp_path / "voice_c.wav", 660.0)
    replaced = save_blend_voice(profiles, "Narrator", clip_a, clip_c, weight=0.5, mode="mix")
    choices = blend_voice_choices(profiles)
    assert len(choices) == 1
    assert choices[0].path == replaced.path
    assert choices[0].label == "Narrator"


def test_blend_voice_choices_skip_entries_with_missing_sources(tmp_path: Path) -> None:
    profiles = tmp_path / "Profiles"
    clip_a, clip_b = _two_clips(tmp_path)
    save_blend_voice(profiles, "Live", clip_a, clip_b)
    clip_a.unlink()

    choices = blend_voice_choices(profiles)
    assert choices == []  # source gone -> skipped, never fatal


def test_remove_blend_voice(tmp_path: Path) -> None:
    profiles = tmp_path / "Profiles"
    clip_a, clip_b = _two_clips(tmp_path)
    save_blend_voice(profiles, "Doomed", clip_a, clip_b)

    assert remove_blend_voice(profiles, "Doomed") is True
    assert remove_blend_voice(profiles, "Doomed") is False
    assert blend_voice_choices(profiles) == []


def test_corrupt_catalog_is_tolerated(tmp_path: Path) -> None:
    profiles = tmp_path / "Profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "blend_voices.json").write_text("{not json", encoding="utf-8")

    assert blend_voice_choices(profiles) == []
    clip_a, clip_b = _two_clips(tmp_path)
    choice = save_blend_voice(profiles, "Recovered", clip_a, clip_b)
    assert Path(choice.path).exists()


def test_default_voice_choices_append_saved_blends_after_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    seashell = repo / "Seashells" / "oracle.wav"
    seashell.parent.mkdir(parents=True, exist_ok=True)
    seashell.write_bytes(b"RIFF")
    clip_a = _write_reference(repo / "voice_a.wav", 220.0)
    clip_b = _write_reference(repo / "voice_b.wav", 440.0)
    save_blend_voice(repo / "Profiles", "Saved Blend One", clip_a, clip_b)

    choices = default_voice_choices(repo)
    assert choices[0].path == str(seashell.resolve())
    assert choices[-1].kind == "blend"
    assert choices[-1].label == "Saved Blend One"
    assert len(choices) <= 10