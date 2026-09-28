"""Performance baseline: cProfile + tracemalloc over the deterministic smoke render.

Extends docs/PERFORMANCE_BASELINE.md's method (same subject: scripts/smoke_render.py
— the real pipeline on the deterministic engine) so numbers stay comparable
across campaigns. Emits a ranked hot-list (top functions by cumulative time)
and peak memory, written as JSON under build/perf/ for cross-run diffs.

Profiling only — no optimization happens here (measure first).
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import tracemalloc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = REPO_ROOT / "build" / "perf" / "baseline.prof"
DEFAULT_REPORT = REPO_ROOT / "build" / "perf" / "baseline.json"
TOP_N = 40


def _run_smoke() -> None:
    """Run the deterministic smoke render (real pipeline, no live model).

    The smoke script ends in ``raise SystemExit(main())``; under runpy that
    SystemExit would escape and kill the profiler before it writes a report
    (observed: exit 0 with no baseline.json), so it is absorbed here — the
    render's success/failure is not this tool's verdict to give.
    """
    import runpy

    sys.argv = ["smoke_render.py"]
    try:
        runpy.run_path(str(REPO_ROOT / "scripts" / "smoke_render.py"), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise


def top_n(rows: list[dict], n: int) -> list[dict]:
    """The n rows with the largest cumtime, descending."""
    return sorted(rows, key=lambda row: row["cumtime"], reverse=True)[:n]


def profile_render(profile_path: Path, report_path: Path) -> dict:
    """Profile one smoke render; write .prof and a ranked JSON report."""
    profile_path = Path(profile_path)
    report_path = Path(report_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        _run_smoke()
    finally:
        profiler.disable()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    profiler.dump_stats(str(profile_path))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    rows = [
        {"name": func, "cumtime": cumulative, "calls": ncalls}
        for (_filename, _lineno, func), (ncalls, _primitive, _selftime, cumulative, _callers) in stats.stats.items()
    ]
    rows.sort(key=lambda row: row["cumtime"], reverse=True)
    top = top_n(rows, TOP_N)

    report = {
        "top_functions": top,
        "peak_memory_mb": round(peak / (1024 * 1024), 2),
        "profile_path": str(profile_path),
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile the deterministic smoke render.")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)

    report = profile_render(Path(args.profile), Path(args.report))
    print(f"peak_memory_mb: {report['peak_memory_mb']}")
    print(f"top {len(report['top_functions'])} functions by cumulative time:")
    for row in report["top_functions"][:15]:
        print(f"  {row['cumtime']:8.3f}s  {row['calls']:>10} calls  {row['name']}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
