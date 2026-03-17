#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
WRAPPER_PATH="$HOME/.local/bin/the-oracle"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$DESKTOP_DIR/the-oracle.desktop"

info() { printf '[INFO] %s\n' "$*"; }
pass() { printf 'PASS: %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

ensure_bootstrap() {
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    pass "Reusing existing venv at $VENV_DIR"
    return
  fi
  info "Bootstrapping The Oracle in $REPO_ROOT"
  "$REPO_ROOT/bootstrap_oracle_tts.sh"
}

write_desktop_file() {
  mkdir -p "$DESKTOP_DIR"
  cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Name=The Oracle
Comment=Chatterbox-based two-speaker TTS
Exec=the-oracle gui
Terminal=false
Type=Application
Categories=AudioVideo;Utility;
StartupNotify=true
# ORACLE_TTS_DESKTOP
EOF
  pass "Installed desktop entry at $DESKTOP_FILE"
}

install_wrapper() {
  if [[ ! -x "$WRAPPER_PATH" ]]; then
    fail "Wrapper missing after bootstrap: $WRAPPER_PATH"
  fi
  pass "CLI/launcher wrapper available at $WRAPPER_PATH"
}

post_checks() {
  info "Running doctor to confirm install"
  if "$REPO_ROOT/doctor_oracle_tts.sh"; then
    pass "Doctor checks passed"
  else
    fail "Doctor reported issues; see output above"
  fi
}

main() {
  ensure_bootstrap
  install_wrapper
  write_desktop_file
  post_checks
  info ""
  pass "Install complete. Launch via desktop menu or run: the-oracle gui"
}

main "$@"
