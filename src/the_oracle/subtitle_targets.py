"""Where a subtitle's companion files go -- named in exactly one place.

Two naming policies live here, and both used to be spelled out at each call
site, which is how they drifted:

* the **converted script** written beside a ``.srt``/``.vtt`` input, so the
  subtitle file itself is never modified (``movie.srt`` -> ``movie.srt.txt``);
* the **sidecar subtitles** written beside a render (``episode.flac`` ->
  ``episode.srt``).

The first was written out four times -- in ``cli``, ``srt_ingest`` and twice in
``ingest_transformer`` -- and the copies disagreed: two of them dropped the
``.txt`` fallback for non-subtitle inputs. Those two call sites are guarded to
subtitle extensions, so the difference was unreachable, but unreachable
divergence is still divergence: the next caller that forgets the guard gets a
``.vtt.txt`` sibling for a ``.md`` file.

Neither function touches the filesystem, so both are safe to call from the
picker, the batch fixer and the renderer alike.
"""

from __future__ import annotations

from pathlib import Path

#: What the converted script is named, per input suffix. ``with_suffix`` cannot
#: build a compound name like ``.vtt.txt`` (it replaces the whole suffix), so
#: targets are assembled from the stem instead.
CONVERTED_SCRIPT_SUFFIXES: dict[str, str] = {
    ".srt": ".srt.txt",
    ".vtt": ".vtt.txt",
}

#: Used for an input that is not a subtitle but is still being converted.
PLAIN_SCRIPT_SUFFIX = ".txt"

#: The render's sidecar subtitle name.
SIDECAR_SUFFIX = ".srt"


def converted_script_target(file_path: str | Path) -> Path:
    """The sibling script a converted input writes its dialogue to.

    ``movie.srt`` -> ``movie.srt.txt``, ``clip.vtt`` -> ``clip.vtt.txt``, and
    anything else -> ``<stem>.txt``. The suffix is appended to the *stem* so the
    original extension stays visible in the name: the converted script sits
    beside its source, and which source it came from is obvious from the name.
    """
    path = Path(file_path)
    tail = CONVERTED_SCRIPT_SUFFIXES.get(path.suffix.lower(), PLAIN_SCRIPT_SUFFIX)
    return path.with_name(path.stem + tail)


def subtitle_sidecar_target(output_path: str | Path) -> Path:
    """Where a render's subtitles are written: its own name with ``.srt``.

    ``with_suffix`` is right here (unlike above) because the sidecar replaces the
    render's single suffix rather than extending it.
    """
    return Path(output_path).with_suffix(SIDECAR_SUFFIX)
