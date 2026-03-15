#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from the_oracle.smoke import run_deterministic_smoke_render


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deterministic smoke renders for txt and md dialogue inputs.")
    parser.add_argument("--output-root", type=Path, default=Path("build/smoke_render"))
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    results = [
        run_deterministic_smoke_render(args.output_root / "txt", source_format="txt"),
        run_deterministic_smoke_render(args.output_root / "md", source_format="md"),
    ]
    if args.as_json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        for result in results:
            print(f"Source format: {result.source_format}")
            print(f"Smoke render output: {result.output_path}")
            print(f"Project dir: {result.project_dir}")
            print(f"Stem count: {result.stem_count}")
            print(f"Cache reused on second pass: {result.cache_reused_on_second_pass}")
