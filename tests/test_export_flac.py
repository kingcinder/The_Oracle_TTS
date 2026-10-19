from pathlib import Path

import numpy as np
import soundfile as sf

from the_oracle.audio.export_flac import next_available_output_path, write_flac
from the_oracle.audio.flac_tags import read_vorbis_comments, write_vorbis_comments


def test_next_available_output_path_uses_desktop_suffixes(tmp_path: Path) -> None:
    base = tmp_path / "name.flac"
    base.touch()
    (tmp_path / "name (1).flac").touch()

    candidate = next_available_output_path(base)

    assert candidate == tmp_path / "name (2).flac"


def test_write_flac_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    audio = np.linspace(-0.1, 0.1, num=2400, dtype=np.float32)
    first = write_flac(tmp_path / "oracle.flac", audio, 24000, {"title": "First"})
    second = write_flac(tmp_path / "oracle.flac", audio, 24000, {"title": "Second"})

    assert first == tmp_path / "oracle.flac"
    assert second == tmp_path / "oracle (1).flac"
    assert first.exists()
    assert second.exists()


def test_write_flac_writes_standard_vorbis_comments(tmp_path: Path) -> None:
    audio = np.linspace(-0.1, 0.1, num=2400, dtype=np.float32)
    destination = write_flac(
        tmp_path / "tagged.flac",
        audio,
        24000,
        {"title": "Tagged", "artist": "The Oracle", "comment": "Unicode ✓", "empty": ""},
    )

    tags = read_vorbis_comments(destination)
    assert tags["TITLE"] == ["Tagged"]
    assert tags["ARTIST"] == ["The Oracle"]
    assert tags["COMMENT"] == ["Unicode ✓"]
    assert "EMPTY" not in tags
    # The tag rewrite must leave a valid FLAC stream, not just parseable tags.
    decoded, sample_rate = sf.read(str(destination))
    assert decoded.shape == audio.shape
    assert sample_rate == 24000


def test_retagging_preserves_unrelated_comments(tmp_path: Path) -> None:
    destination = tmp_path / "retag.flac"
    sf.write(str(destination), np.zeros(240, dtype=np.float32), 24000, format="FLAC")

    write_vorbis_comments(destination, {"title": "First", "artist": "The Oracle"})
    write_vorbis_comments(destination, {"album": "Alpha"})

    tags = read_vorbis_comments(destination)
    assert tags["TITLE"] == ["First"]
    assert tags["ARTIST"] == ["The Oracle"]
    assert tags["ALBUM"] == ["Alpha"]
