#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from the_oracle.smoke import run_deterministic_smoke_render, smoke_output_problem


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic smoke renders and verify their outputs.

    A render can return without raising yet leave no usable audio, so the
    exit code reflects the smoke module's output-verdict policy, not just
    whether an exception was raised: 0 only when every render's output file
    exists and is non-empty.
    """
    # Entry-point arm (2026-10-08 audit standard): this process drives the
    # deterministic engines and the FLAC export path natively. Consent-aware,
    # idempotent, fail-closed — same contract as cli.main and launch_gui.
    from the_oracle.crash import handlers as crash_handlers

    crash_handlers.install()
    crash_handlers.arm_native_capture()
    parser = argparse.ArgumentParser(description="Run deterministic smoke renders for txt and md dialogue inputs.")
    parser.add_argument("--output-root", type=Path, default=Path("build/smoke_render"))
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    results = [
        run_deterministic_smoke_render(args.output_root / "txt", source_format="txt"),
        run_deterministic_smoke_render(args.output_root / "md", source_format="md"),
    ]
    problems = [(result, smoke_output_problem(result)) for result in results]
    if args.as_json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        for result in results:
            print(f"Source format: {result.source_format}")
            print(f"Smoke render output: {result.output_path}")
            print(f"Project dir: {result.project_dir}")
            print(f"Stem count: {result.stem_count}")
            print(f"Cache reused on second pass: {result.cache_reused_on_second_pass}")
    failed = [(result, problem) for result, problem in problems if problem is not None]
    for result, problem in failed:
        print(f"FAIL ({result.source_format}): {problem}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
