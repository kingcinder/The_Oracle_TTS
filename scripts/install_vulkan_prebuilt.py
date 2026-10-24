"""Download and install a checksum-verified unsigned Vulkan binary."""

from __future__ import annotations

import argparse
from pathlib import Path

from the_oracle.vulkan_prebuilt import download_and_install


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-url", required=True)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    installed = download_and_install(
        args.manifest_url,
        args.archive_url,
        args.destination,
        cache_dir=args.cache_dir,
    )
    print(f"Installed verified Vulkan binary: {installed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
