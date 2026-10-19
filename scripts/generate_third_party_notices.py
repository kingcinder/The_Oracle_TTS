#!/usr/bin/env python3
"""Generate and verify the release third-party notice inventory.

Usage:
    python scripts/generate_third_party_notices.py
        Write THIRD_PARTY_NOTICES.md from the current release tree.

    python scripts/generate_third_party_notices.py --check
        Fail if the committed file differs from what this tree would generate.

    python scripts/generate_third_party_notices.py --json
        Print the component inventory instead of the Markdown.

The inventory resolves the runtime dependency closure from ``pyproject.toml``
and the installed package metadata, then adds the bundled models, voice assets,
Qt binaries, and optional external tools. It performs no network I/O and adds
no dependency of its own.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import __version__  # noqa: E402
from the_oracle import third_party  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "THIRD_PARTY_NOTICES.md"


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _inventory(root: Path) -> third_party.Inventory:
    return third_party.build_inventory(root, version=__version__)


def _json_payload(inventory: third_party.Inventory) -> dict[str, object]:
    return {
        "version": inventory.version,
        "components": [
            {
                "name": component.name,
                "version": component.version,
                "license": component.license,
                "classification": component.classification,
                "kind": component.kind,
                "source": component.source,
                "notice": component.notice,
            }
            for component in inventory.components
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate or verify third-party notices.")
    parser.add_argument("--root", default=str(REPO_ROOT), help="Release tree root (default: this repository).")
    parser.add_argument("--out", default=None, help="Output path (default: <root>/THIRD_PARTY_NOTICES.md).")
    parser.add_argument("--check", action="store_true", help="Verify the committed file is current; do not write.")
    parser.add_argument("--json", action="store_true", help="Print the component inventory as JSON.")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    output = Path(args.out).resolve() if args.out else root / "THIRD_PARTY_NOTICES.md"
    version = __version__ if root == REPO_ROOT else _version_from_tree(root)
    inventory = third_party.build_inventory(root, version=version)
    rendered = third_party.render_notices(inventory)

    if args.json:
        print(json.dumps(_json_payload(inventory), indent=2))
        return 0

    problems = list(inventory.problems)
    if args.check:
        if not output.is_file():
            problems.append(f"{output} is missing; run this script without --check")
        else:
            committed = output.read_text(encoding="utf-8")
            if committed != rendered:
                problems.append(
                    f"{output} is stale; regenerate it with `python scripts/generate_third_party_notices.py`"
                )
        problems.extend(third_party.notice_problems(root, inventory=inventory))
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"Third-party notices are current ({len(inventory.components)} components).")
        return 0

    _write_atomically(output, rendered)
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    print(f"Wrote {output} ({len(inventory.components)} components).")
    return 0 if not problems else 1


def _version_from_tree(root: Path) -> str:
    init = root / "src" / "the_oracle" / "__init__.py"
    if not init.is_file():
        return "unknown"
    for line in init.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            _, _, value = line.partition("=")
            return value.strip().strip('"').strip("'")
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
