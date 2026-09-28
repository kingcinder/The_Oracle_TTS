"""Tests for scripts/perf_baseline.py (plan Task 8)."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "perf_baseline.py"
spec = importlib.util.spec_from_file_location("oracle_perf_baseline", SCRIPT)
pb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pb  # dataclasses resolve through sys.modules
spec.loader.exec_module(pb)


def test_top_n_orders_by_cumtime():
    rows = [{"name": "a", "cumtime": 1.0}, {"name": "b", "cumtime": 9.0}]
    assert pb.top_n(rows, 1)[0]["name"] == "b"
    # n larger than the row count returns everything, still ranked
    assert [row["name"] for row in pb.top_n(rows, 10)] == ["b", "a"]


def test_profile_render_returns_shape(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pb, "_run_smoke", lambda: None)
    result = pb.profile_render(tmp_path / "p.prof", tmp_path / "m.json")
    assert "top_functions" in result and "peak_memory_mb" in result
    assert isinstance(result["top_functions"], list)
    assert isinstance(result["peak_memory_mb"], float)
    # report is persisted for cross-run comparison
    assert (tmp_path / "m.json").exists()


def test_ranked_report_written(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pb, "_run_smoke", lambda: None)
    out = pb.profile_render(tmp_path / "p.prof", tmp_path / "m.json")
    report = json.loads((tmp_path / "m.json").read_text())
    assert report["top_functions"] == out["top_functions"]
    assert report["peak_memory_mb"] == out["peak_memory_mb"]
