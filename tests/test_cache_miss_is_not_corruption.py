"""A missing cached stem is a cache MISS, not corruption (plan Task 7).

Root cause captured from live evidence (2026-09-28 crash-hunt Vulkan run and
the 2026-09-27 render_child.log): the batched read path probes
``_load_servable_stem`` without an ``exists()`` check, so an ordinary cache
miss (e.g. a Vulkan run reading a cache populated by PyTorch — the chunk hash
keys on ``inference_backend``) hit ``sf.read`` on a nonexistent file, which
libsndfile reports as ``System error`` (ENOENT), and ``_load_cached_stem``
logged "unreadable ... deleting and re-synthesizing" for a file that was
never there. The re-synthesis was correct; the warning was a lie, and it
looked like wholesale cache corruption in the logs.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from the_oracle.pipeline import _load_cached_stem, _load_servable_stem
from the_oracle.audio.assemble import save_wav


def test_missing_stem_is_silent_cache_miss(tmp_path: Path, caplog):
    """Reading a stem that was never written must not log the corrupt warning."""
    missing = tmp_path / "never_written.wav"
    with caplog.at_level(logging.WARNING, logger="the_oracle.pipeline"):
        assert _load_cached_stem(missing) is None
        assert _load_servable_stem(missing) is None
    assert "unreadable" not in caplog.text, (
        "a missing cache entry must not be reported as unreadable/corrupt:\n"
        f"{caplog.text}"
    )


def test_missing_stem_does_not_attempt_unlink(tmp_path: Path, monkeypatch):
    """The miss path must not pretend to delete a file it never found."""
    from the_oracle import pipeline

    missing = tmp_path / "never_written.wav"
    attempted = []
    real_unlink = Path.unlink

    def spy(self, *args, **kwargs):
        attempted.append(str(self))
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", spy)
    assert _load_cached_stem(missing) is None
    assert attempted == [], f"unlink attempted on a missing file: {attempted}"


def test_corrupt_stem_still_warns_and_is_deleted(tmp_path: Path, caplog):
    """The real corruption path must keep its warning (vacuity guard)."""
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not a wav at all")
    with caplog.at_level(logging.WARNING, logger="the_oracle.pipeline"):
        assert _load_cached_stem(corrupt) is None
    assert "unreadable" in caplog.text
    assert not corrupt.exists(), "the corrupt entry must still be deleted"


def test_existing_valid_stem_still_loads(tmp_path: Path):
    """The happy path must be untouched by the fix."""
    good = tmp_path / "good.wav"
    audio = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
    save_wav(good, audio, 24000)
    loaded = _load_servable_stem(good)
    assert loaded is not None
    samples, rate = loaded
    assert rate == 24000
    assert samples.shape == audio.shape
