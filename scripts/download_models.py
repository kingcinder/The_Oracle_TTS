#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from the_oracle.tts_engines.chatterbox_engine import ChatterboxEngine


HF_MODELS = {
    "go_emotions": "SamLowe/roberta-base-go_emotions",
    "punctuation": "oliverguhr/fullstop-punctuation-multilang-large",
}


def download_hf_model(repo_id: str, cache_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=str(cache_dir / repo_id.replace("/", "_")), local_dir_use_symlinks=False)


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
    if args.include_helper_models or args.variant == "all":
        for name, repo_id in HF_MODELS.items():
            try:
                download_hf_model(repo_id, args.cache_dir)
                print(f"Downloaded {name}: {repo_id}")
            except Exception as exc:
                print(f"Skipped {name}: {exc}")

    variants = ["standard", "multilingual", "turbo"] if args.variant == "all" else [args.variant]
    for variant in variants:
        try:
            warm_chatterbox(variant, device=args.device)
            print(f"Warmed Chatterbox variant: {variant}")
        except Exception as exc:
            print(f"Skipped Chatterbox variant {variant}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
