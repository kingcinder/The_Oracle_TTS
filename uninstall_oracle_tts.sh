#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
WRAPPER_PATH="$HOME/.local/bin/the-oracle"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$DESKTOP_DIR/the-oracle.desktop"

info() { printf '[INFO] %s\n' "$*"; }
pass() { printf 'PASS: %s\n' "$*"; }

remove_wrapper() {
  if [[ -f "$WRAPPER_PATH" ]] && grep -q "ORACLE_TTS_WRAPPER" "$WRAPPER_PATH"; then
    rm -f "$WRAPPER_PATH"
    pass "Removed managed wrapper $WRAPPER_PATH"
  fi
}

remove_desktop() {
  if [[ -f "$DESKTOP_FILE" ]] && grep -q "ORACLE_TTS_DESKTOP" "$DESKTOP_FILE"; then
    rm -f "$DESKTOP_FILE"
    pass "Removed desktop entry $DESKTOP_FILE"
  fi
}

remove_venv() {
  if [[ -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
    pass "Removed virtualenv $VENV_DIR"
  fi
}

main() {
  remove_wrapper
  remove_desktop
  remove_venv
  info "User projects, settings, and cached voices remain under your home directories."
}

main "$@"
