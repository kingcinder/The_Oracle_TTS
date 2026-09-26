#!/usr/bin/env python3
"""Vendor-only license minting tool — never shipped, never leaves the repo.

The signing seed never lives in the repository: it arrives through the
ORACLE_LICENSE_SIGNING_KEY environment variable (64 hex chars = 32 bytes),
which the operator generates once with ``license_sign.py keygen`` and keeps
in a secret manager. Losing it means key rotation (keys.py registry), not
disaster. This script is the only component in the project with signing
authority; ``scripts/`` is not packaged into the wheel, so it cannot ship by
accident — tests pin that.

Usage:
  ORACLE_LICENSE_SIGNING_KEY=<64 hex> python scripts/license_sign.py mint \
      --edition studio --licensee "name or email" [--key-id k1] \
      [--expires 2027-12-31T00:00:00Z] [--machine-hash <sha256-32hex>]
  python scripts/license_sign.py keygen
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import secrets
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import _ed25519  # noqa: E402
from the_oracle.licensing import tokens  # noqa: E402

ENV_VAR = "ORACLE_LICENSE_SIGNING_KEY"


def _load_seed() -> bytes:
    hex_seed = os.environ.get(ENV_VAR, "")
    if not hex_seed:
        print(
            f"error: {ENV_VAR} is not set. The signing seed never lives in the repo; "
            "export it from your secret manager, or generate one with `keygen`.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        seed = bytes.fromhex(hex_seed)
    except ValueError:
        print(f"error: {ENV_VAR} must be 64 hex characters.", file=sys.stderr)
        raise SystemExit(1)
    if len(seed) != 32:
        print(f"error: {ENV_VAR} must decode to exactly 32 bytes.", file=sys.stderr)
        raise SystemExit(1)
    return seed


def _parse_expires(value: str | None) -> int | None:
    if not value:
        return None
    try:
        moment = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        print(f"error: --expires must be an ISO-8601 timestamp, got {value!r}", file=sys.stderr)
        raise SystemExit(1)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return int(moment.timestamp())


def cmd_keygen(_args: argparse.Namespace) -> int:
    seed = secrets.token_bytes(32)
    print(
        json.dumps(
            {
                "note": "Store this OUTSIDE the repository. It is the only copy; losing it means key rotation.",
                "signing_seed_hex": seed.hex(),
                "verify_key_hex": _ed25519.public_key(seed).hex(),
                "next_step": "Add verify_key_hex to the_oracle/licensing/keys.py under a new key_id, then keep the old key for one release cycle.",
            },
            indent=2,
        )
    )
    return 0


def cmd_mint(args: argparse.Namespace) -> int:
    seed = _load_seed()
    if args.machine_hash and (
        len(args.machine_hash) != 32 or any(c not in "0123456789abcdef" for c in args.machine_hash.lower())
    ):
        print("error: --machine-hash must be 32 lowercase hex chars (SHA-256 prefix).", file=sys.stderr)
        return 1
    payload = {
        "v": 1,
        "key_id": args.key_id,
        "lic_id": str(uuid.uuid4()),
        "edition": args.edition,
        "licensee": args.licensee or "",
        "machine_hash": (args.machine_hash or "").lower(),
        "iat": int(time.time()),
        "exp": _parse_expires(args.expires),
    }
    token = tokens.mint(payload, seed)
    print(json.dumps({"token": token, "payload": payload}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vendor-only Oracle license minting.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("keygen", help="Generate a signing seed + its verify key. Never commit either output's seed.")

    mint_parser = sub.add_parser("mint", help="Mint one signed license token.")
    mint_parser.add_argument("--edition", required=True, choices=["community", "studio", "trial"])
    mint_parser.add_argument("--licensee", default="")
    mint_parser.add_argument("--key-id", default="k1")
    mint_parser.add_argument("--expires", default=None, help="ISO-8601 timestamp; omit for a perpetual license.")
    mint_parser.add_argument("--machine-hash", default=None, help="32-hex SHA-256 prefix from `the-oracle machine-id`.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "keygen":
        return cmd_keygen(args)
    if args.command == "mint":
        return cmd_mint(args)
    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
