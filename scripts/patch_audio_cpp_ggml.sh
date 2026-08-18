#!/usr/bin/env bash
# Apply The Oracle's vendored patches to the cloned audio.cpp tree (ggml is
# vendored in-tree at external/ggml, so these are local, clearly-marked edits).
#
# Patch 1 - RDNA1 buffer memset (ggml): AMD RDNA1 (gfx1010/gfx1012) SDMA
#   transfer queues reject vkCmdFillBuffer (EINVAL), surfacing as
#   VK_ERROR_DEVICE_LOST during buffer init (ggml-org/whisper.cpp#3611).
#   ggml_vk_buffer_memset() runs its memset through the transfer queue's
#   command pool; this patch switches it to the compute queue on RDNA1.
#   vk_command_pool binds its queue, so ggml_vk_submit follows the same queue.
# Patch 2 - space-safe build (sentencepiece): audio.cpp fails to build from
#   checkout paths containing spaces because sentencepiece's
#   -fmacro-prefix-map=... flag is emitted unquoted and split on the space.
#   The value is quoted so builds work from any path.
#
# Both patches are idempotent (no-op when their ORACLE_VENDORED marker is
# present) and FAIL LOUDLY if the source changes shape on an audio.cpp/ggml
# update, so the fixes can never be silently lost on a version bump.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIOCPP_DIR="${AUDIOCPP_DIR:-$REPO_ROOT/audio.cpp}"
GGML_FILE="$AUDIOCPP_DIR/external/ggml/src/ggml-vulkan/ggml-vulkan.cpp"
SP_FILE="$AUDIOCPP_DIR/external/sentencepiece/CMakeLists.txt"
GGML_MARKER="===== BEGIN ORACLE VENDORED PATCH (RDNA1 buffer memset) ====="
SP_MARKER="===== BEGIN ORACLE VENDORED PATCH (space-safe macro-prefix-map) ====="
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$GGML_FILE" ]]; then
  echo "ERROR: ggml-vulkan.cpp not found at $GGML_FILE (run scripts/build_audio_cpp.sh first)." >&2
  exit 1
fi

apply_patch() {
  local name="$1" marker="$2" file="$3" original_text="$4" patched_text="$5"

  if grep -q "$marker" "$file"; then
    echo "[patch] $name: already applied to $file (no-op)."
    return 0
  fi

  if ! "$PYTHON_BIN" - "$file" "$original_text" "$patched_text" <<'PYEOF'
import sys

path, original, patched = sys.argv[1], sys.argv[2], sys.argv[3]
o_lines = original.split("\n")
p_lines = patched.split("\n")
with open(path, "rb") as handle:
    raw = handle.read()
# Split on "\n" only: each element keeps its trailing "\r" when the file uses
# CRLF. Only the matched anchor region is replaced, so every other byte (and
# line ending) in the file is preserved byte-for-byte -- no diff churn.
file_lines = raw.split(b"\n")
norm = [line.rstrip(b"\r") for line in file_lines]
target = [line.encode() for line in o_lines]
for i in range(len(norm) - len(target) + 1):
    if norm[i : i + len(target)] == target:
        anchor_crlf = file_lines[i].endswith(b"\r")
        new_lines = [line.encode() + (b"\r" if anchor_crlf else b"") for line in p_lines]
        result = file_lines[:i] + new_lines + file_lines[i + len(target) :]
        open(path, "wb").write(b"\n".join(result))
        sys.exit(0)
sys.exit(
    f"ANCHOR NOT FOUND in {path} -- source changed shape; a vendored fix "
    f"must be re-applied by hand."
)
PYEOF
  then
    echo "ERROR: $name patch application failed (see message above). The vendored fix is NOT in place." >&2
    exit 1
  fi

  if ! grep -q "$marker" "$file"; then
    echo "ERROR: post-apply verification failed for $name; marker missing. Do not proceed until fixed." >&2
    exit 1
  fi
  echo "[patch] Applied $name to $file"
}

# --- Patch 1: RDNA1 buffer memset in ggml-vulkan.cpp -------------------------
GGML_ORIGINAL="$(printf '%s\n' \
  '        memset((uint8_t*)dst->ptr + offset, c, size);' \
  '        return;' \
  '    }' \
  '' \
  '    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);' \
  '    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue.cmd_pool);')"
GGML_PATCHED="$(printf '%s\n' \
  '        memset((uint8_t*)dst->ptr + offset, c, size);' \
  '        return;' \
  '    }' \
  '' \
  '    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);' \
  "    // $GGML_MARKER" \
  '    // AMD RDNA1 (gfx1010/gfx1012) SDMA transfer queues reject vkCmdFillBuffer' \
  '    // (EINVAL), surfacing as VK_ERROR_DEVICE_LOST during buffer init' \
  '    // (ggml-org/whisper.cpp#3611). Run the memset on the compute queue for' \
  '    // RDNA1; vk_command_pool binds its queue so ggml_vk_submit follows it.' \
  "    // Vendored against ggml $(git -C "$AUDIOCPP_DIR/external/ggml" rev-parse --short HEAD 2>/dev/null || echo unknown)." \
  '    // Re-check this block after any audio.cpp/ggml update.' \
  '    // ===== END ORACLE VENDORED PATCH (RDNA1 buffer memset) =====' \
  '    vk_context subctx = ggml_vk_create_temporary_context(' \
  '        dst->device->architecture == vk_device_architecture::AMD_RDNA1' \
  '            ? dst->device->compute_queue.cmd_pool' \
  '            : dst->device->transfer_queue.cmd_pool);')"
apply_patch "RDNA1 buffer-memset fix" "$GGML_MARKER" "$GGML_FILE" "$GGML_ORIGINAL" "$GGML_PATCHED"

# --- Patch 2: space-safe macro-prefix-map in sentencepiece -------------------
SP_ORIGINAL="  string(APPEND CMAKE_CXX_FLAGS \" -fmacro-prefix-map=\${CMAKE_SOURCE_DIR}/=''\")"
# CMake comments use '#' (not '//'). The CMake line must contain literal
# backslash-quote escapes so the path stays inside the quotes:
#   string(APPEND CMAKE_CXX_FLAGS " -fmacro-prefix-map=\"...\").
SP_PATCHED="  # $SP_MARKER
  # audio.cpp fails to build from checkout paths containing spaces: the
  # unquoted -fmacro-prefix-map=... flag is split on the space. Quote the
  # value so builds work from any path.
  # ===== END ORACLE VENDORED PATCH (space-safe macro-prefix-map) =====
  string(APPEND CMAKE_CXX_FLAGS \" -fmacro-prefix-map=\\\"\${CMAKE_SOURCE_DIR}/=\\\"\")"
apply_patch "space-safe sentencepiece build fix" "$SP_MARKER" "$SP_FILE" "$SP_ORIGINAL" "$SP_PATCHED"

# Regenerate the reviewable patch artifacts so they always match what is applied.
mkdir -p "$REPO_ROOT/scripts/patches"
git -C "$AUDIOCPP_DIR/external/ggml" diff -- src/ggml-vulkan/ggml-vulkan.cpp > "$REPO_ROOT/scripts/patches/ggml_rdna1_buffer_memset.patch" || true
git -C "$AUDIOCPP_DIR" diff -- external/sentencepiece/CMakeLists.txt > "$REPO_ROOT/scripts/patches/audio_cpp_space_safe_build.patch" || true
echo "[patch] ggml commit: $(git -C "$AUDIOCPP_DIR/external/ggml" rev-parse HEAD 2>/dev/null || echo '?')"
echo "[patch] Patch artifacts: $REPO_ROOT/scripts/patches/"
