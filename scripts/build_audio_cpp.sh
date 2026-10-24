#!/usr/bin/env bash
# Build the audio.cpp Vulkan CLI used by The Oracle's opt-in Vulkan backend
# (--inference-backend vulkan). Reuses audio.cpp's own Linux build helper so
# backend flags stay aligned with upstream, and applies The Oracle's vendored
# RDNA1 ggml fix (scripts/patch_audio_cpp_ggml.sh) before building.
#
# Requirements: GCC 13+, CMake, and the Vulkan SDK (for ENGINE_ENABLE_VULKAN).
# The build itself is heavy; run it once and point ORACLE_AUDIOCPP_CLI at the
# resulting binary.
#
# Pass --with-model to also fetch the Chatterbox model afterwards, so a single
# command builds the CLI and fetches the model. SKIP_MODEL_DOWNLOAD=1
# overrides and skips the download even then (e.g. CI that already has the
# model); set AUDIOCPP_MODEL_PACKAGE / AUDIOCPP_MODELS_ROOT to steer what and
# where it installs (same knobs as scripts/download_audio_cpp_model.sh).
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIOCPP_DIR="${AUDIOCPP_DIR:-$REPO_ROOT/audio.cpp}"
AUDIOCPP_REPO="${AUDIOCPP_REPO:-https://github.com/0xShug0/audio.cpp}"
# The vendored ggml patches were validated against this exact upstream tree.
# Do not silently build a moving branch or tag.
AUDIOCPP_PINNED_COMMIT="748c5e28f6a7228b8f38ad7142ca97d29584544b"

# Build only the chatterbox family by default (a full build is much slower).
# Override with AUDIOCPP_MODEL_SET=full (or AUDIOCPP_MODELS="a,b").
AUDIOCPP_MODEL_SET="${AUDIOCPP_MODEL_SET:-custom}"
AUDIOCPP_MODELS="${AUDIOCPP_MODELS:-chatterbox}"
# The patch step is a separate script; an override lets hermetic tests stub it
# out (the real one needs a full ggml source tree to patch).
PATCH_AUDIOCPP_GGML_SH="${PATCH_AUDIOCPP_GGML_SH:-$REPO_ROOT/scripts/patch_audio_cpp_ggml.sh}"

WITH_MODEL=0
for arg in "$@"; do
  case "$arg" in
    --with-model)
      WITH_MODEL=1
      ;;
    -h | --help)
      echo "Usage: $0 [--with-model]"
      echo
      echo "Builds the audio.cpp Vulkan CLI (audiocpp_cli) used by the Oracle's"
      echo "opt-in Vulkan backend, applying The Oracle's vendored ggml fixes."
      echo
      echo "Options:"
      echo "  --with-model   also fetch the Chatterbox model afterwards (one"
      echo "                 command builds the CLI and fetches the model)."
      echo
      echo "Environment:"
      echo "  SKIP_MODEL_DOWNLOAD=1   skip the model download even with --with-model"
      echo "  AUDIOCPP_MODEL_PACKAGE   model precision (default chatterbox_q8_0)"
      echo "  AUDIOCPP_MODELS_ROOT     where the model is installed"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $arg (supported: --with-model, --help)" >&2
      exit 1
      ;;
  esac
done

# SKIP_MODEL_DOWNLOAD=1 is the belt-and-suspenders override for scripts/CI
# that always pass --with-model but must not fetch in some environments.
if [[ "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  WITH_MODEL=0
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake is required (plus GCC 13+ and the Vulkan SDK)." >&2
  exit 1
fi

if [[ ! -d "$AUDIOCPP_DIR/.git" ]]; then
  echo "[1/4] Cloning pinned audio.cpp tree into $AUDIOCPP_DIR (ggml is vendored in-tree)"
  git clone "$AUDIOCPP_REPO" "$AUDIOCPP_DIR"
else
  echo "[1/4] audio.cpp already present at $AUDIOCPP_DIR"
fi

if [[ "${AUDIOCPP_SKIP_PIN_VERIFY:-0}" != "1" ]]; then
  if ! git -C "$AUDIOCPP_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "ERROR: audio.cpp is not a valid Git checkout; refusing an unpinned build." >&2
    exit 1
  fi
  actual_commit="$(git -C "$AUDIOCPP_DIR" rev-parse HEAD)"
  if [[ "$actual_commit" != "$AUDIOCPP_PINNED_COMMIT" ]]; then
    echo "[1/4] Fetching pinned audio.cpp commit $AUDIOCPP_PINNED_COMMIT"
    git -C "$AUDIOCPP_DIR" fetch --quiet origin "$AUDIOCPP_PINNED_COMMIT"
    git -C "$AUDIOCPP_DIR" checkout --quiet --detach "$AUDIOCPP_PINNED_COMMIT"
  fi
  actual_commit="$(git -C "$AUDIOCPP_DIR" rev-parse HEAD)"
  if [[ "$actual_commit" != "$AUDIOCPP_PINNED_COMMIT" ]]; then
    echo "ERROR: audio.cpp commit mismatch: expected $AUDIOCPP_PINNED_COMMIT, got $actual_commit" >&2
    exit 1
  fi
  echo "[1/4] Verified audio.cpp commit $actual_commit"
else
  echo "[1/4] WARNING: AUDIOCPP_SKIP_PIN_VERIFY=1; pin verification disabled for a test harness" >&2
fi

echo "[2/4] Applying The Oracle's vendored RDNA1 ggml fix (idempotent)"
"$PATCH_AUDIOCPP_GGML_SH"

echo "[3/4] Building audiocpp_cli with the Vulkan backend (ENGINE_ENABLE_VULKAN=ON, model-set=$AUDIOCPP_MODEL_SET)"
# audio.cpp's build helper resolves the CMake source tree from its cwd.
(
  cd "$AUDIOCPP_DIR"
  ./scripts/build_linux.sh \
    --backend vulkan \
    --model-set "$AUDIOCPP_MODEL_SET" \
    --models "$AUDIOCPP_MODELS" \
    --target audiocpp_cli
)

BINARY="$AUDIOCPP_DIR/build/linux-vulkan-release/bin/audiocpp_cli"

if [[ "$WITH_MODEL" -eq 1 ]]; then
  if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: expected binary not found at $BINARY -- the build did not" >&2
    echo "produce audiocpp_cli (or the build layout changed). Refusing to" >&2
    echo "download a model for a CLI that does not exist." >&2
    exit 1
  fi
  echo "[4/4] Built: $BINARY"
  echo "[+] Fetching the Chatterbox model for the Vulkan backend"
  AUDIOCPP_DIR="$AUDIOCPP_DIR" "$REPO_ROOT/scripts/download_audio_cpp_model.sh"
  cat <<EOF

Build and model complete. The Oracle applies these paths automatically for
--inference-backend vulkan renders and in the GUI -- no shell exports needed.
EOF
else
  echo "[4/4] Built: $BINARY"
  cat <<EOF

Next steps:
1. Fetch the Chatterbox model (or re-run this script with --with-model to
   build and fetch in one command):
     "$REPO_ROOT/scripts/download_audio_cpp_model.sh"
   (use --dry-run to preview, AUDIOCPP_MODEL_PACKAGE=chatterbox_f16 to pick
   a different precision, or --overwrite to re-download)
2. Render -- The Oracle locates the binary and model automatically:
     the-oracle render --input Input/cli_short.txt --outdir Output \\
       --speakerA-ref "Seashells/generic/english_reader_1.wav" \\
       --speakerB-ref "Seashells/generic/english_reader_2.wav" \\
       --inference-backend vulkan
EOF
fi

cat <<EOF

The vendored RDNA1 fix (ggml_vk_buffer_memset on the compute queue for RDNA1)
was applied to audio.cpp/external/ggml and is marked with ORACLE_VENDORED
comments; scripts/patch_audio_cpp_ggml.sh re-applies it idempotently and fails
loudly on ggml version bumps. If audio.cpp still reports VK_ERROR_DEVICE_LOST,
the Python backend surfaces it with a clear error and you can fall back to
--inference-backend pytorch.
EOF
