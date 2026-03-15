#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
WRAPPER_PATH="$HOME/.local/bin/the-oracle"
LEGACY_WRAPPER_PATH="$HOME/.local/bin/dualvoice"
WRAPPER_BACKUP_SUFFIX="$(date +%Y%m%d_%H%M%S)"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

pass() {
  printf 'PASS: %s\n' "$*"
}

info() {
  printf '%s\n' "$*"
}

fail() {
  printf 'FAIL: %s\n' "$*" >&2
}

select_python() {
  local candidate
  for candidate in python3.12 python3.11 python3; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 11) <= sys.version_info[:3] < (3, 13) else 1)
PY
    then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

cleanup_legacy_wrapper() {
  if [[ ! -f "$LEGACY_WRAPPER_PATH" ]]; then
    return 0
  fi

  if grep -q "ORACLE_TTS_WRAPPER" "$LEGACY_WRAPPER_PATH"; then
    rm -f "$LEGACY_WRAPPER_PATH"
    pass "Removed legacy managed wrapper at $LEGACY_WRAPPER_PATH"
    return 0
  fi

  info "Leaving non-managed legacy wrapper in place at $LEGACY_WRAPPER_PATH"
}

cleanup_legacy_venv_entrypoint() {
  local legacy_entrypoint="$VENV_DIR/bin/dualvoice"
  if [[ -e "$legacy_entrypoint" ]]; then
    rm -f "$legacy_entrypoint"
    pass "Removed legacy venv entrypoint at $legacy_entrypoint"
  fi
}

install_oracle_wrapper() {
  mkdir -p "$(dirname "$WRAPPER_PATH")"
  if [[ -f "$WRAPPER_PATH" ]] && ! grep -q "ORACLE_TTS_WRAPPER" "$WRAPPER_PATH"; then
    mv "$WRAPPER_PATH" "${WRAPPER_PATH}.pre_oracle_tts.${WRAPPER_BACKUP_SUFFIX}"
    info "Backed up existing $WRAPPER_PATH to ${WRAPPER_PATH}.pre_oracle_tts.${WRAPPER_BACKUP_SUFFIX}"
  fi

  cat > "$WRAPPER_PATH" <<EOF
#!/usr/bin/env bash
# ORACLE_TTS_WRAPPER
set -Eeuo pipefail

REPO_ROOT="$REPO_ROOT"
VENV_ENTRYPOINT="\$REPO_ROOT/.venv/bin/the-oracle"

if [[ ! -x "\$VENV_ENTRYPOINT" ]]; then
  printf 'the-oracle is not installed in %s\nRun %s/bootstrap_oracle_tts.sh first.\n' "\$REPO_ROOT/.venv" "\$REPO_ROOT" >&2
  exit 1
fi

exec "\$VENV_ENTRYPOINT" "\$@"
EOF
  chmod +x "$WRAPPER_PATH"
}

install_python_dependencies() {
  "$VENV_DIR/bin/python" -m pip install \
    --index-url "$PYTORCH_INDEX_URL" \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    torchvision==0.21.0
  "$VENV_DIR/bin/python" -m pip install -e ".[ml]"
  "$VENV_DIR/bin/python" -m pip install \
    librosa==0.11.0 \
    s3tokenizer \
    diffusers==0.29.0 \
    resemble-perth==1.0.1 \
    conformer==0.3.2 \
    safetensors==0.5.3 \
    spacy-pkuseg \
    pykakasi==2.3.0 \
    pyloudnorm \
    omegaconf
  "$VENV_DIR/bin/python" -m pip install --no-deps chatterbox-tts==0.1.6
}

main() {
  local python_bin
  if ! python_bin="$(select_python)"; then
    fail "Need Python 3.11 or 3.12 with venv support."
    printf 'Next step: sudo apt install python3.12 python3.12-venv\n' >&2
    exit 1
  fi

  pass "Using $("$python_bin" --version 2>&1) from $(command -v "$python_bin")"

  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$python_bin" -m venv "$VENV_DIR"
    pass "Created project venv at $VENV_DIR"
  else
    pass "Reusing project venv at $VENV_DIR"
  fi

  "$VENV_DIR/bin/python" -m pip install --upgrade pip "setuptools<81" wheel
  install_python_dependencies
  cleanup_legacy_venv_entrypoint
  pass "Installed The Oracle, CPU PyTorch, and the Chatterbox runtime bundle into $VENV_DIR"

  cleanup_legacy_wrapper
  install_oracle_wrapper
  pass "Installed managed the-oracle wrapper at $WRAPPER_PATH"

  if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    info "Note: ~/.local/bin is not on this shell PATH. The doctor will print the exact export command if fresh-shell detection fails."
  fi

  info ""
  info "Verification report:"
  set +e
  "$REPO_ROOT/doctor_oracle_tts.sh"
  local doctor_status=$?
  set -e
  if (( doctor_status != 0 )); then
    fail "Bootstrap verification found one or more blocking issues."
    printf 'Next step: follow the FAIL lines above, then rerun ./bootstrap_oracle_tts.sh\n' >&2
    exit "$doctor_status"
  fi

  pass "Bootstrap complete. Launch The Oracle GUI with ./run_oracle_tts.sh"
}

main "$@"
