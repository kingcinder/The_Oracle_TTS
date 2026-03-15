from pathlib import Path

import numpy as np

from the_oracle.audio.export_flac import next_available_output_path, write_flac


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
