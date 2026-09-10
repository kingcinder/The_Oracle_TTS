#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from the_oracle.tts_engines.chatterbox_engine import ChatterboxEngine


HF_MODELS = {
    "go_emotions": "SamLowe/roberta-base-go_emotions",
    "punctuation": "oliverguhr/fullstop-punctuation-multilang-large",
}

# Per-model Hugging Face commit-SHA pins for reproducible installs.
#
# Fill each value with the commit SHA from https://huggingface.co/<repo>/commits
# (pick the commit you want every install to use). A value of None means
# "unpinned: download whatever is newest", which is the historical behavior
# but makes installs non-reproducible: upstream model updates can silently
# change voice/emotion behavior between machines.
#
# NOTE: placeholder structure only — no real SHAs are known here. Do not
# invent SHA-like strings and treat them as real pins.
HF_MODEL_REVISIONS: dict[str, str | None] = {
    "go_emotions": None,  # TODO: pin from huggingface.co/SamLowe/roberta-base-go_emotions/commits
    "punctuation": None,  # TODO: pin from huggingface.co/oliverguhr/fullstop-punctuation-multilang-large/commits
}


def download_hf_model(repo_id: str, cache_dir: Path, revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(cache_dir / repo_id.replace("/", "_")),
        local_dir_use_symlinks=False,
    )


def warm_chatterbox(variant: str, device: str) -> None:
    ChatterboxEngine(variant=variant, device=device).ensure_model_ready()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download optional model artifacts for Chatterbox-only The Oracle.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".model_cache"))
    parser.add_argument("--variant", choices=["standard", "multilingual", "turbo", "all"], default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-helper-models", action="store_true")
    args = parser.parse_args(argv)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    if args.include_helper_models or args.variant == "all":
        for name, repo_id in HF_MODELS.items():
            revision = HF_MODEL_REVISIONS.get(name)
            try:
                download_hf_model(repo_id, args.cache_dir, revision=revision)
                print(f"Downloaded {name}: {repo_id}")
            except Exception as exc:
                failures.append(name)
                print(f"ERROR: Failed to download model {name} ({repo_id}): {exc}", file=sys.stderr)

    variants = ["standard", "multilingual", "turbo"] if args.variant == "all" else [args.variant]
    for variant in variants:
        try:
            warm_chatterbox(variant, device=args.device)
            print(f"Warmed Chatterbox variant: {variant}")
        except Exception as exc:
            failures.append(f"chatterbox-{variant}")
            print(f"ERROR: Failed to warm Chatterbox variant {variant}: {exc}", file=sys.stderr)
    if failures:
        print(
            f"ERROR: {len(failures)} model download(s) failed: {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
