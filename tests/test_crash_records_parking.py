"""Parking the crash-records store for unattended offscreen certifier runs.

The D8 startup branch (``gui_crash.maybe_run_startup_flow``) opens a MODAL
— the next-session review whenever records exist, the first-run consent ask
whenever no decision is recorded — and nothing in a headless sweep can
answer either dialog. The standalone certifiers
(``scripts/certify_gui_themes.py``, ``scripts/certify_section_chrome.py``)
build the REAL MainWindow, whose queued crash-flow timer reaches that
branch on every launch. Parking renames the whole store out of the D8
reader's sight (``list_records`` sees an empty root) instead of moving
records one by one, and the recovery sweep makes a mid-run crash
self-healing: a SIGSEGV skips every ``finally``, so the park lands in the
repo root where the NEXT parked run finds it and brings it home. Evidence
is at worst mislaid for one run, never lost — the merge keeps a colliding
record beside its namesake rather than overwriting either copy.

Mutation contracts: dropping the park (or the restore) from either
certifier fails its source-scan pin; removing the restore call fails the
raise test; removing the leftover-recovery glob fails the recovery test;
removing the never-overwrite branch fails the collision test.
"""

from __future__ import annotations

from pathlib import Path

from the_oracle.crash import bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
CERTIFIERS = (
    REPO_ROOT / "scripts" / "certify_gui_themes.py",
    REPO_ROOT / "scripts" / "certify_section_chrome.py",
)


def _seed_record(root: Path, name: str = "crash-20261010T120000-abcdef01.json", body: str = '{"x": 1}') -> Path:
    directory = root / "crash_reports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def test_park_hides_the_store_from_the_d8_reader(tmp_path: Path) -> None:
    record = _seed_record(tmp_path)
    assert len(bundle.list_records(tmp_path)) == 1

    parked = bundle.park_records(tmp_path)
    assert parked is not None and parked.is_dir()
    # The D8 review branch (gui_crash: `if bundle.list_records(root)`)
    # now sees an empty store, so it cannot open its modal.
    assert bundle.list_records(tmp_path) == []
    assert not bundle.crash_dir(tmp_path).exists()
    # The record itself moved, byte-identical — nothing was dropped.
    assert (parked / record.name).read_text(encoding="utf-8") == '{"x": 1}'

    bundle.restore_parked_records(tmp_path, parked)
    assert bundle.list_records(tmp_path) == [record]
    assert record.read_text(encoding="utf-8") == '{"x": 1}'
    assert not parked.exists()


def test_park_without_a_store_is_a_noop(tmp_path: Path) -> None:
    assert bundle.park_records(tmp_path) is None
    # restore(None) — the scripts' finally path with nothing parked.
    bundle.restore_parked_records(tmp_path, None)
    assert list(tmp_path.glob("crash_reports.certifier-park-*")) == []
    assert not bundle.crash_dir(tmp_path).exists()


def test_restore_brings_records_home_even_when_the_run_raises(tmp_path: Path) -> None:
    """The certifiers' shape: park, work, restore in ``finally`` — the
    cleanup must run on the exception path too."""
    record = _seed_record(tmp_path)
    parked = bundle.park_records(tmp_path)
    try:
        raise RuntimeError("sweep died mid-theme")
    except RuntimeError:
        pass
    finally:
        bundle.restore_parked_records(tmp_path, parked)
    assert bundle.list_records(tmp_path) == [record]


def test_a_crashed_predecessors_park_is_recovered_by_the_next_run(tmp_path: Path) -> None:
    """A sweep that dies (SIGSEGV skips every finally) leaves its park in
    the root; the next run's park_records merges it home before hiding the
    store again — evidence survives the crash of its protectors."""
    record = _seed_record(tmp_path)
    # Run 1 parks, then "crashes": no restore ever runs.
    bundle.park_records(tmp_path)
    assert bundle.list_records(tmp_path) == []

    # Run 2: recovery happens inside park_records itself.
    parked2 = bundle.park_records(tmp_path)
    assert parked2 is not None
    assert len(list(tmp_path.glob("crash_reports.certifier-park-*"))) == 1
    bundle.restore_parked_records(tmp_path, parked2)
    assert bundle.list_records(tmp_path) == [record]
    assert not list(tmp_path.glob("crash_reports.certifier-park-*"))


def test_recovery_never_loses_a_colliding_record(tmp_path: Path) -> None:
    """A name present on both sides is byte-compared: identical duplicates
    are dropped, differing copies survive side by side (``.parked``) —
    parking may hide evidence, never destroy it."""
    same_name = "crash-20261010T120000-aaaa0001.json"
    _seed_record(tmp_path, name=same_name, body='{"side": "store"}')
    # A crashed predecessor's park: one identical duplicate, one clashing
    # name with different bytes.
    leftover = tmp_path / "crash_reports.certifier-park-99999"
    leftover.mkdir()
    (leftover / same_name).write_text('{"side": "store"}', encoding="utf-8")
    (leftover / "crash-20261010T120000-aaaa0002.json").write_text('{"side": "park"}', encoding="utf-8")
    _seed_record(tmp_path, name="crash-20261010T120000-aaaa0002.json", body='{"side": "store"}')

    parked = bundle.park_records(tmp_path)
    bundle.restore_parked_records(tmp_path, parked)

    store = tmp_path / "crash_reports"
    # The identical duplicate collapsed to one copy.
    assert (store / same_name).read_text(encoding="utf-8") == '{"side": "store"}'
    # The clash: both bodies survive, the parked one under its own name.
    assert (store / "crash-20261010T120000-aaaa0002.json").read_text(encoding="utf-8") == '{"side": "store"}'
    assert (store / "crash-20261010T120000-aaaa0002.json.parked").read_text(encoding="utf-8") == '{"side": "park"}'
    assert not leftover.exists()


# --- the certifiers actually use it ------------------------------------------


def test_both_certifiers_park_and_restore_the_store() -> None:
    for script in CERTIFIERS:
        source = script.read_text(encoding="utf-8")
        assert "park_records(" in source, f"{script.name}: no park — the D8 review can block it"
        assert "restore_parked_records(" in source, f"{script.name}: parked but never restored"
        # Order: the store is hidden before the theme loop builds its first
        # window (the point where the crash-flow timer gets armed).
        assert source.index("park_records(") < source.index("for key in THEMES:"), (
            f"{script.name}: park must precede the theme loop, not follow it"
        )


def test_the_park_scan_can_actually_see_its_literals() -> None:
    """Vacuity guard: the scans above must match real call sites."""
    owner = (REPO_ROOT / "src" / "the_oracle" / "crash" / "bundle.py").read_text(encoding="utf-8")
    assert "def park_records(" in owner
    assert "def restore_parked_records(" in owner


def test_both_certifiers_stub_the_modal_d8_flow() -> None:
    """The park kills the records-review branch; the never-asked branch
    (first-run consent on a fresh checkout) still needs the per-window
    stub — records parked or not, that dialog would still block."""
    for script in CERTIFIERS:
        source = script.read_text(encoding="utf-8")
        assert "window._maybe_run_crash_startup_flow = " in source, (
            f"{script.name}: the first-run consent dialog can still block the sweep"
        )
