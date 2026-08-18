#!/usr/bin/env bash
# Download the Chatterbox ggml model used by The Oracle's opt-in Vulkan
# backend (--inference-backend vulkan). Wraps audio.cpp's own model manager
# (tools/model_manager_v2.py) so the model lands in the expected location and
# prints the ORACLE_AUDIOCPP_MODEL export line to paste into your shell.
#
# Requirements: python3 and a clone of audio.cpp (see scripts/build_audio_cpp.sh).
#
# Examples:
#   ./scripts/download_audio_cpp_model.sh                            # chatterbox_q8_0 (default)
#   AUDIOCPP_MODEL_PACKAGE=chatterbox_f16 ./scripts/download_audio_cpp_model.sh
#   ./scripts/download_audio_cpp_model.sh --dry-run                  # show plan, no download
#   ./scripts/download_audio_cpp_model.sh --overwrite                # re-download existing install
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIOCPP_DIR="${AUDIOCPP_DIR:-$REPO_ROOT/audio.cpp}"
MODEL_MANAGER="$AUDIOCPP_DIR/tools/model_manager_v2.py"

# Package id from audio.cpp's model_specs/chatterbox.json (the spec marks
# Chatterbox Q8_0 GGUF as its default package). Override with
# AUDIOCPP_MODEL_PACKAGE (e.g. chatterbox_f16).
AUDIOCPP_MODEL_PACKAGE="${AUDIOCPP_MODEL_PACKAGE:-chatterbox_q8_0}"
# Where models are installed. Absolute, because the model manager resolves
# --models-root against the current working directory.
AUDIOCPP_MODELS_ROOT="${AUDIOCPP_MODELS_ROOT:-$AUDIOCPP_DIR/models}"

# Extra args are passed straight through to the model manager's install
# command (--dry-run, --check, --overwrite, --format, --precision...).
EXTRA_ARGS=("$@")

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to run audio.cpp's model manager." >&2
  exit 1
fi

if [[ ! -f "$MODEL_MANAGER" ]]; then
  echo "ERROR: audio.cpp model manager not found at: $MODEL_MANAGER" >&2
  echo "Clone audio.cpp first with ./scripts/build_audio_cpp.sh" >&2
  exit 1
fi

IS_PLAN_ONLY=0
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--dry-run" || "$arg" == "--check" ]]; then
    IS_PLAN_ONLY=1
  fi
  if [[ "$arg" == "--models-root" || "$arg" == "--models-root="* ]]; then
    echo "ERROR: do not pass --models-root to this script; it installs into" >&2
    echo "\$AUDIOCPP_MODELS_ROOT ($AUDIOCPP_MODELS_ROOT) and derives the printed" >&2
    echo "model path from that same root. Set AUDIOCPP_MODELS_ROOT instead." >&2
    exit 1
  fi
done

echo "[1/2] Installing '$AUDIOCPP_MODEL_PACKAGE' via audio.cpp's model manager"
python3 "$MODEL_MANAGER" install "$AUDIOCPP_MODEL_PACKAGE" \
  --models-root "$AUDIOCPP_MODELS_ROOT" \
  "${EXTRA_ARGS[@]}"

echo "[2/2] Resolving the installed model path"
# Derive the local model path from `info --json`: a single-file package
# (e.g. GGUF) resolves to that file; a multi-file package (e.g. safetensors)
# resolves to its target directory, which audio.cpp's --model accepts.
MODEL_PATH="$(python3 - "$MODEL_MANAGER" "$AUDIOCPP_MODEL_PACKAGE" "$AUDIOCPP_MODELS_ROOT" <<'PY'
import json
import subprocess
import sys

manager, package, models_root = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    info = json.loads(
        subprocess.run(
            [sys.executable, manager, "info", package, "--json"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
except Exception as exc:
    print(f"ERROR: could not resolve package {package} from audio.cpp specs: {exc}", file=sys.stderr)
    raise SystemExit(1)
target = info.get("target_directory", "")
files = info.get("files") or []
if not files:
    print(f"ERROR: package {package} declares no files in audio.cpp specs.", file=sys.stderr)
    raise SystemExit(1)
strip_prefix = info.get("strip_prefix", "").rstrip("/")
stripped: list[str] = []
for remote in files:
    path = remote
    if strip_prefix and path.startswith(strip_prefix + "/"):
        path = path[len(strip_prefix) + 1 :]
    stripped.append(path)
if len(stripped) == 1:
    print(f"{models_root}/{target}/{stripped[0]}")
else:
    print(f"{models_root}/{target}")
PY
)" || true
if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: failed to resolve the installed model path for '$AUDIOCPP_MODEL_PACKAGE'." >&2
  exit 1
fi

if [[ "$IS_PLAN_ONLY" -eq 1 ]]; then
  echo "Planned model path: $MODEL_PATH (not downloaded -- $* used)"
  exit 0
fi

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "ERROR: expected model at $MODEL_PATH but it is missing after install." >&2
  exit 1
fi

cat <<EOF

Chatterbox model installed: $MODEL_PATH

Point The Oracle at it and render on the Vulkan backend:

    export ORACLE_AUDIOCPP_MODEL="$MODEL_PATH"
    export ORACLE_AUDIOCPP_CLI="$AUDIOCPP_DIR/build/linux-vulkan-release/bin/audiocpp_cli"
    the-oracle render --input Input/cli_short.txt --outdir Output \\
      --speakerA-ref "Seashells/Cody's Seashell.wav" \\
      --speakerB-ref "Seashells/Cody's Seashell1.wav" \\
      --inference-backend vulkan

The model manager refuses to overwrite an existing install; pass --overwrite
to re-download, and use --dry-run to preview the download without fetching.
EOF
