#!/usr/bin/env bash
set -Eeuo pipefail

# Chatterbox-only bootstrap for The Oracle TTS / DualVoice Studio style repos.
# Creates a dedicated Python 3.11 environment, installs GPU or CPU PyTorch,
# installs chatterbox-tts, warms the model cache, runs a smoke test, clones the
# upstream repo for reference/examples, and writes activation + report files for Codex.

REPO_ROOT="${REPO_ROOT:-$PWD}"
DEVICE_MODE="auto"
SKIP_WARMUP="0"
FORCE_RECREATE_ENV="0"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --repo-root PATH       Target repo root. Default: current working directory
  --device MODE          auto | cuda | cpu. Default: auto
  --skip-warmup          Install only; do not preload model or run smoke test generation
  --force-recreate-env   Delete and recreate the chatterbox env
  -h, --help             Show this help

Environment overrides:
  CHATTERBOX_DEVICE=auto|cuda|cpu
  CHATTERBOX_SKIP_WARMUP=0|1
  CHATTERBOX_FORCE_RECREATE_ENV=0|1
  CHATTERBOX_PYTORCH_VERSION=2.6.0
  CHATTERBOX_TORCHAUDIO_VERSION=2.6.0
  CHATTERBOX_TORCHVISION_VERSION=0.21.0
  CHATTERBOX_VERSION=0.1.6
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --device)
      DEVICE_MODE="$2"
      shift 2
      ;;
    --skip-warmup)
      SKIP_WARMUP="1"
      shift
      ;;
    --force-recreate-env)
      FORCE_RECREATE_ENV="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

DEVICE_MODE="${CHATTERBOX_DEVICE:-$DEVICE_MODE}"
SKIP_WARMUP="${CHATTERBOX_SKIP_WARMUP:-$SKIP_WARMUP}"
FORCE_RECREATE_ENV="${CHATTERBOX_FORCE_RECREATE_ENV:-$FORCE_RECREATE_ENV}"

TORCH_VERSION="${CHATTERBOX_PYTORCH_VERSION:-2.6.0}"
TORCHAUDIO_VERSION="${CHATTERBOX_TORCHAUDIO_VERSION:-2.6.0}"
TORCHVISION_VERSION="${CHATTERBOX_TORCHVISION_VERSION:-0.21.0}"
CHATTERBOX_VERSION="${CHATTERBOX_VERSION:-0.1.6}"

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
TOOLS_DIR="$REPO_ROOT/.tools"
ENGINE_SETUP_DIR="$REPO_ROOT/.engine-setup"
ENV_DIR="$REPO_ROOT/.envs/chatterbox-py311"
UPSTREAM_DIR="$REPO_ROOT/third_party/chatterbox-upstream"
MODEL_CACHE_DIR="$REPO_ROOT/.model_cache/chatterbox"
MAMBA_ROOT_PREFIX="$TOOLS_DIR/micromamba-root"
MICROMAMBA_BIN="$TOOLS_DIR/bin/micromamba"
ACTIVATE_SCRIPT="$ENGINE_SETUP_DIR/chatterbox_env.sh"
REPORT_JSON="$ENGINE_SETUP_DIR/chatterbox_install_report.json"
SMOKE_PY="$ENGINE_SETUP_DIR/chatterbox_smoke_test.py"
SMOKE_WAV="$ENGINE_SETUP_DIR/chatterbox_smoke.wav"

mkdir -p "$TOOLS_DIR/bin" "$ENGINE_SETUP_DIR" "$REPO_ROOT/.envs" "$REPO_ROOT/third_party" "$MODEL_CACHE_DIR"

log() {
  printf '\n==> %s\n' "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

need_one_of() {
  for candidate in "$@"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      return 0
    fi
  done
  echo "Missing required command. Need one of: $*" >&2
  exit 1
}

need_cmd bash
need_cmd tar
need_one_of curl wget

bootstrap_micromamba() {
  if [[ -x "$MICROMAMBA_BIN" ]]; then
    return 0
  fi

  log "Bootstrapping local micromamba"
  local arch
  case "$(uname -m)" in
    x86_64|amd64) arch="linux-64" ;;
    aarch64|arm64) arch="linux-aarch64" ;;
    *)
      echo "Unsupported architecture for micromamba bootstrap: $(uname -m)" >&2
      exit 1
      ;;
  esac

  local tmp_archive extract_dir
  tmp_archive="$(mktemp)"
  extract_dir="$(mktemp -d)"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "https://micro.mamba.pm/api/micromamba/${arch}/latest" -o "$tmp_archive"
  else
    wget -qO "$tmp_archive" "https://micro.mamba.pm/api/micromamba/${arch}/latest"
  fi

  tar -xjf "$tmp_archive" -C "$extract_dir"
  mv "$extract_dir/bin/micromamba" "$MICROMAMBA_BIN"
  chmod +x "$MICROMAMBA_BIN"
  rm -f "$tmp_archive"
  rm -rf "$extract_dir"
}

activate_env() {
  eval "$($MICROMAMBA_BIN shell hook -s bash -r "$MAMBA_ROOT_PREFIX")"
  micromamba activate "$ENV_DIR"
}

create_env() {
  bootstrap_micromamba
  if [[ "$FORCE_RECREATE_ENV" == "1" && -d "$ENV_DIR" ]]; then
    log "Removing existing chatterbox env"
    rm -rf "$ENV_DIR"
  fi
  if [[ ! -d "$ENV_DIR" ]]; then
    log "Creating Python 3.11 chatterbox env"
    "$MICROMAMBA_BIN" create -y -p "$ENV_DIR" python=3.11 pip
  fi
  activate_env
  python --version
  pip install -U pip "setuptools<81" wheel packaging
  pip install -U "numpy>=2.0"
}

detect_device() {
  if [[ "$DEVICE_MODE" == "cpu" || "$DEVICE_MODE" == "cuda" ]]; then
    printf '%s' "$DEVICE_MODE"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf 'cuda'
  else
    printf 'cpu'
  fi
}

install_pytorch() {
  local resolved_device="$1"
  log "Installing PyTorch for ${resolved_device}"
  if [[ "$resolved_device" == "cuda" ]]; then
    pip install \
      "torch==${TORCH_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      --index-url https://download.pytorch.org/whl/cu124
  else
    pip install \
      "torch==${TORCH_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      --index-url https://download.pytorch.org/whl/cpu
  fi
}

install_chatterbox() {
  log "Installing chatterbox-tts ${CHATTERBOX_VERSION}"
  pip install -U "chatterbox-tts==${CHATTERBOX_VERSION}" soundfile huggingface_hub
}

clone_upstream() {
  if command -v git >/dev/null 2>&1; then
    if [[ -d "$UPSTREAM_DIR/.git" ]]; then
      log "Refreshing upstream chatterbox reference repo"
      git -C "$UPSTREAM_DIR" fetch --depth 1 origin master || true
      git -C "$UPSTREAM_DIR" reset --hard origin/master || true
    elif [[ ! -e "$UPSTREAM_DIR" ]]; then
      log "Cloning upstream chatterbox repo for reference/examples"
      git clone --depth 1 https://github.com/resemble-ai/chatterbox.git "$UPSTREAM_DIR" || true
    fi
  fi
}

write_smoke_test() {
  cat > "$SMOKE_PY" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

out_path = Path(os.environ["CHATTERBOX_SMOKE_WAV"])
device = os.environ.get("CHATTERBOX_DEVICE_RESOLVED", "cpu")
text = os.environ.get(
    "CHATTERBOX_SMOKE_TEXT",
    "The Oracle is online. This is a Chatterbox smoke test.",
)

model = ChatterboxTTS.from_pretrained(device=device)
wav = model.generate(text)
ta.save(str(out_path), wav, model.sr)

payload = {
    "device": device,
    "sample_rate": int(model.sr),
    "output": str(out_path),
    "torch_cuda_available": bool(torch.cuda.is_available()),
    "torch_version": torch.__version__,
}
print(json.dumps(payload, indent=2))
PY
}

write_activate_script() {
  cat > "$ACTIVATE_SCRIPT" <<EOF2
#!/usr/bin/env bash
set -Eeuo pipefail
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX"
export HF_HOME="$MODEL_CACHE_DIR"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export TORCH_HOME="$MODEL_CACHE_DIR/torch"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT/src"
MICROMAMBA_BIN="$MICROMAMBA_BIN"
if [[ ! -x "\$MICROMAMBA_BIN" ]]; then
  echo "micromamba not found at \$MICROMAMBA_BIN" >&2
  exit 1
fi

eval "\$(\$MICROMAMBA_BIN shell hook -s bash -r \"$MAMBA_ROOT_PREFIX\")"
micromamba activate "$ENV_DIR"
EOF2
  chmod +x "$ACTIVATE_SCRIPT"
}

write_report() {
  local resolved_device="$1"
  CHATTERBOX_DEVICE_RESOLVED="$resolved_device" CHATTERBOX_SMOKE_WAV="$SMOKE_WAV" python - <<'PY' > "$REPORT_JSON"
from __future__ import annotations

import json
import os
import platform
import shutil
import sys

report = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "device_requested": os.environ.get("CHATTERBOX_DEVICE_RESOLVED"),
    "env_prefix": sys.prefix,
    "ffmpeg": shutil.which("ffmpeg") is not None,
    "smoke_wav": os.environ.get("CHATTERBOX_SMOKE_WAV"),
}

modules = {}
for mod in ["torch", "torchaudio", "soundfile", "chatterbox"]:
    try:
        module = __import__(mod)
        modules[mod] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        modules[mod] = f"missing: {exc}"
report["modules"] = modules

try:
    import torch
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device_count"] = int(torch.cuda.device_count())
except Exception as exc:
    report["cuda_available"] = False
    report["cuda_error"] = str(exc)

print(json.dumps(report, indent=2))
PY
}

run_warmup() {
  local resolved_device="$1"
  write_smoke_test
  export HF_HOME="$MODEL_CACHE_DIR"
  export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
  export TORCH_HOME="$MODEL_CACHE_DIR/torch"
  export XDG_CACHE_HOME="$REPO_ROOT/.cache"
  export TOKENIZERS_PARALLELISM=false
  export CHATTERBOX_DEVICE_RESOLVED="$resolved_device"
  export CHATTERBOX_SMOKE_WAV="$SMOKE_WAV"

  if [[ "$SKIP_WARMUP" == "1" ]]; then
    log "Skipping warmup/smoke test"
    return 0
  fi

  log "Running chatterbox smoke test and warming model cache"
  python "$SMOKE_PY"
}

main() {
  log "Repo root: $REPO_ROOT"
  create_env
  local resolved_device
  resolved_device="$(detect_device)"
  log "Resolved device: $resolved_device"
  install_pytorch "$resolved_device"
  install_chatterbox
  clone_upstream
  write_activate_script
  run_warmup "$resolved_device"
  write_report "$resolved_device"

  log "Finished"
  echo "Activation script: $ACTIVATE_SCRIPT"
  echo "Install report:    $REPORT_JSON"
  echo "Smoke test audio:  $SMOKE_WAV"
  echo
  echo "Next shell:"
  echo "  source '$ACTIVATE_SCRIPT'"
}

main "$@"
