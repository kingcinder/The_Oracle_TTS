"""Pinned Hugging Face model revisions.

Every model The Oracle downloads is pinned to an exact commit SHA so installs
are reproducible: upstream model updates can never silently change voice,
emotion, or punctuation behavior between machines, and the offline bundle
builder / offline installer seed exactly these revisions.

To refresh a pin, take the newest commit SHA from
https://huggingface.co/<repo_id>/commits and update both the SHA here and the
offline bundle (scripts/build_offline_bundle.py).
"""

from __future__ import annotations

GO_EMOTIONS_REPO = "SamLowe/roberta-base-go_emotions"
PUNCTUATION_REPO = "oliverguhr/fullstop-punctuation-multilang-large"
CHATTERBOX_REPO = "ResembleAI/chatterbox"
TURBO_REPO_ID = "ResembleAI/chatterbox-turbo"

# Pinned 2026-09-14: newest commit on each repo at pin time.
MODEL_PINS: dict[str, str] = {
    GO_EMOTIONS_REPO: "d75048347613a25d77de8cf6412eaae9fa7b26be",
    PUNCTUATION_REPO: "345e80adc07e761d3a35feafd20f2f44a151f453",
    CHATTERBOX_REPO: "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
    TURBO_REPO_ID: "749d1c1a46eb10492095d68fbcf55691ccf137cd",
}

#: Helper-model file globs kept in the offline bundle (drops unused framework
#: variants such as the TF/ONNX copies to keep the bundle lean).
HELPER_MODEL_PATTERNS: dict[str, list[str]] = {
    GO_EMOTIONS_REPO: ["*.json", "*.txt", "*.safetensors"],
    PUNCTUATION_REPO: ["*.json", "*.model", "model.safetensors"],
}

#: Chatterbox engine file sets, matching the loaders in chatterbox-tts 0.1.6.
CHATTERBOX_STANDARD_PATTERNS = [
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
]
CHATTERBOX_MULTILINGUAL_PATTERNS = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]
TURBO_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model", "*.yaml"]


def pin_for(repo_id: str) -> str:
    """Return the pinned commit SHA for a model repo."""
    return MODEL_PINS[repo_id]
