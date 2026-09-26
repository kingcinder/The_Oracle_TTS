"""The cache sweep: maintenance counterpart of the servable-stem gate.

The read-side gate (:func:`the_oracle.pipeline._load_servable_stem`) deletes a
degenerate entry when a render reads it. But entries sit in project caches
between renders, and an operator may want the cache clean *now*:
``sweep_project_cache`` (and the ``sweep-cache`` CLI command) classifies every
stem-cache entry with the gate's taxonomy, reports exactly what it purges and
why, and purges only under ``apply`` — the next render re-synthesizes exactly
the purged hashes, so the report's purge list IS the re-synthesis set.

The subtle policy pinned here: the gate's ``allow_silence`` exemption cannot
be used by a sweep, because content alone cannot distinguish a sanctioned
pause stem from a degenerate spoken entry (both near-silent). The sweep
therefore decides pause-vs-degenerate by the canonical pause shape —
``_write_pause_only_stem`` banks max(1, int(rate * 0.5)) literal zeros at one
rate — so canonical-shaped silence is kept and *any other* silence is purged
as degenerate rather than silently served.

Everything is offline; the one end-to-end render is marked slow (the
deterministic smoke engine, no models).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from the_oracle.audio.assemble import save_wav
from the_oracle.cli import build_parser, handle_sweep_cache
from the_oracle.models.cache import ProjectCache
from the_oracle.pipeline import _write_pause_only_stem, sweep_project_cache


RATE = 24000


def _plant_cache(stem_dir: Path) -> None:
    """One cache holding one entry of every verdict class."""
    stem_dir.mkdir(parents=True, exist_ok=True)
    save_wav(stem_dir / "abc123.wav", np.linspace(-0.5, 0.5, RATE, dtype=np.float32), RATE)  # speech
    _write_pause_only_stem(stem_dir / "pause01.wav", RATE)  # canonical pause
    save_wav(stem_dir / "dead999.wav", np.zeros(28800, dtype=np.float32), RATE)  # degenerate spoken
    (stem_dir / "corrupt1.wav").write_bytes(b"not a wav at all")  # unreadable
    save_wav(stem_dir / "wrongrate.wav", np.zeros(8000, dtype=np.float32), 16000)  # odd rate


class TestSweepVerdicts:
    def test_every_verdict_class_is_reported_exactly(self, tmp_path: Path) -> None:
        cache = ProjectCache(tmp_path / "proj")
        _plant_cache(cache.stem_cache_dir)

        report = sweep_project_cache(tmp_path / "proj", expected_sample_rate=RATE, apply=True)

        assert report["scanned"] == 5
        assert report["kept"] == 1, "the speech-shaped entry stays"
        assert report["kept_pause"] == 1, "the canonical pause stem stays"
        assert report["purged"] == 3
        assert report["purge_reasons"] == {"unreadable": 1, "degenerate": 1, "sample_rate": 1}
        assert sorted(report["purged_hashes"]) == ["corrupt1", "dead999", "wrongrate"]
        assert [entry["hash"] for entry in report["purged_detail"]] == report["purged_hashes"] or sorted(
            entry["hash"] for entry in report["purged_detail"]
        ) == sorted(report["purged_hashes"])
        assert report["errors"] == []
        # Exactly the purged files are gone; the kept ones remain on disk.
        remaining = {path.stem for path in cache.stem_cache_dir.glob("*.wav")}
        assert remaining == {"abc123", "pause01"}

    def test_dry_run_reports_identically_and_deletes_nothing(self, tmp_path: Path) -> None:
        cache = ProjectCache(tmp_path / "proj")
        _plant_cache(cache.stem_cache_dir)
        before = {path.name: path.read_bytes() for path in cache.stem_cache_dir.glob("*.wav")}

        report = sweep_project_cache(tmp_path / "proj", expected_sample_rate=RATE, apply=False)

        assert report["apply"] is False
        assert report["purged"] == 3 and report["purged_hashes"] == ["corrupt1", "dead999", "wrongrate"]
        after = {path.name: path.read_bytes() for path in cache.stem_cache_dir.glob("*.wav")}
        assert after == before, "a dry run must not touch a single file"

    def test_missing_cache_directory_is_a_clean_noop_with_warning(self, tmp_path: Path) -> None:
        report = sweep_project_cache(tmp_path / "proj", apply=True)

        assert report["scanned"] == 0 and report["purged"] == 0
        assert "nothing to sweep" in report["cache_missing_warning"]


class TestPauseAmbiguity:
    """The sweep's pause decision is canonical *shape*, not "is silent"."""

    def test_silence_at_canonical_shape_is_kept(self, tmp_path: Path) -> None:
        cache = ProjectCache(tmp_path / "proj")
        cache.stem_cache_dir.mkdir(parents=True, exist_ok=True)
        _write_pause_only_stem(cache.stem_cache_dir / "p.wav", RATE)

        report = sweep_project_cache(tmp_path / "proj", apply=True)

        assert report["kept_pause"] == 1 and report["purged"] == 0

    def test_silence_at_wrong_length_is_purged_as_degenerate(self, tmp_path: Path) -> None:
        """Silence at the right rate but a non-canonical length: a degenerate
        spoken entry, not a pause stem — the exact case treating all silence
        as exempt would silently serve."""
        cache = ProjectCache(tmp_path / "proj")
        cache.stem_cache_dir.mkdir(parents=True, exist_ok=True)
        save_wav(cache.stem_cache_dir / "long_silence.wav", np.zeros(RATE, dtype=np.float32), RATE)  # 1.0s

        report = sweep_project_cache(tmp_path / "proj", apply=True)

        assert report["purged"] == 1
        assert report["purge_reasons"]["degenerate"] == 1

    def test_canonical_silence_at_wrong_rate_is_purged_when_rate_is_expected(self, tmp_path: Path) -> None:
        """Two dry-run sweeps over the same untouched cache: with an expected
        rate the 16 kHz pause stem is purged as odd-rate; without one it is
        recognized by its canonical shape. (Both dry runs — the first apply
        sweep would delete the file and make the second verdict vacuous.)"""
        cache = ProjectCache(tmp_path / "proj")
        cache.stem_cache_dir.mkdir(parents=True, exist_ok=True)
        _write_pause_only_stem(cache.stem_cache_dir / "p16.wav", 16000)

        report = sweep_project_cache(tmp_path / "proj", expected_sample_rate=RATE, apply=False)
        assert report["purge_reasons"]["sample_rate"] == 1

        report_no_rate = sweep_project_cache(tmp_path / "proj", apply=False)
        assert report_no_rate["kept_pause"] == 1


class TestCliSurface:
    def _args(self, *extra: str):
        return build_parser().parse_args(["sweep-cache", str(self.project), *extra])

    def test_human_report_names_hashes_reasons_and_resynthesis(self, tmp_path: Path, capsys) -> None:
        self.project = tmp_path / "proj"
        cache = ProjectCache(self.project)
        _plant_cache(cache.stem_cache_dir)

        exit_code = handle_sweep_cache(self._args("--sample-rate", str(RATE), "--apply"))

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "- dead999  (degenerate)" in out
        assert "- corrupt1  (unreadable)" in out
        assert "The next render will re-synthesize exactly these hashes." in out

    def test_default_is_a_dry_run_and_json_is_parseable(self, tmp_path: Path, capsys) -> None:
        self.project = tmp_path / "proj"
        cache = ProjectCache(self.project)
        _plant_cache(cache.stem_cache_dir)

        exit_code = handle_sweep_cache(self._args("--json", "--sample-rate", str(RATE)))

        report = json.loads(capsys.readouterr().out)
        assert exit_code == 0
        assert report["apply"] is False and report["purged"] == 3
        assert {path.name for path in cache.stem_cache_dir.glob("*.wav")} == {
            "abc123.wav",
            "pause01.wav",
            "dead999.wav",
            "corrupt1.wav",
            "wrongrate.wav",
        }, "without --apply the CLI must not delete anything"

    def test_missing_project_exits_1(self, tmp_path: Path, capsys) -> None:
        self.project = tmp_path / "nope"

        assert handle_sweep_cache(self._args()) == 1
        assert "not found" in capsys.readouterr().err

    def test_missing_cache_directory_exits_1_with_warning(self, tmp_path: Path, capsys) -> None:
        self.project = tmp_path / "proj"
        self.project.mkdir()

        assert handle_sweep_cache(self._args()) == 1
        assert "nothing to sweep" in capsys.readouterr().err


@pytest.mark.slow
def test_sweep_purge_list_is_exactly_the_resynthesis_set(tmp_path: Path) -> None:
    """The end-to-end promise, proven on a real deterministic render.

    Render a project, plant one degenerate entry, sweep it, re-render the
    same project: exactly the purged hash is re-synthesized (every other stem
    byte-identical), and the re-synthesized take matches the original bytes —
    the purge didn't just delete, the report predicted the future render's
    work exactly.
    """
    from tests.test_synthesis_retry_visibility import _seeded_determinism_render
    from the_oracle.smoke import _DeterministicChatterboxEngine

    plan_a, _events, _out, stems_a = _seeded_determinism_render(tmp_path, "sweep_e2e", _DeterministicChatterboxEngine)
    cache = ProjectCache(plan_a.output_dir)

    victim_hash = sorted(stems_a)[0]
    victim_path = cache.stem_cache_dir / f"{victim_hash}.wav"
    save_wav(victim_path, np.zeros(28800, dtype=np.float32), RATE)  # plant non-canonical silence

    report = sweep_project_cache(plan_a.output_dir, expected_sample_rate=RATE, apply=True)
    assert report["purged_hashes"] == [victim_hash], "exactly the planted entry is reported"

    _plan_b, _events, _out, stems_b = _seeded_determinism_render(tmp_path, "sweep_e2e", _DeterministicChatterboxEngine)

    assert victim_hash in stems_b, "the purged hash was re-synthesized"
    assert stems_b[victim_hash] == stems_a[victim_hash], (
        "the re-synthesized take matches the original bytes (determinism holds through the sweep)"
    )
    for name, payload in stems_a.items():
        if name != victim_hash:
            assert stems_b[name] == payload, f"untouched entry {name} must not be re-synthesized"


def test_stem_cache_dir_helper_cannot_drift_from_the_constructor_layout() -> None:
    """The sweep probes existence via the no-mkdir static helper; if that
    path ever drifted from ProjectCache.__init__'s layout, real caches would
    report "missing" and the drift would only surface in the slow e2e."""
    from the_oracle.models.cache import ProjectCache as _PC

    # The helper side never touches the filesystem; the constructor side
    # creates its layout under /tmp (disposable scratch), which is exactly
    # the comparison needed: same computed path, from the two code paths.
    probe = Path(tempfile.gettempdir()) / "sweep-drift-probe-disposable"
    expected = _PC.stem_cache_dir_for(probe)
    assert expected == _PC(probe).stem_cache_dir
    shutil.rmtree(probe, ignore_errors=True)
