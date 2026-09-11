#!/usr/bin/env bash
# _launcher.sh — backend bootstrap for the .desktop launchers (not user-facing).
#
# The user double-clicks "Install The Oracle TTS" / "Uninstall The Oracle TTS
# Engine" (.desktop files with real icons). This script:
#   1. registers the dedicated icons + app-grid launchers at user level,
#   2. re-runs itself with sudo for the privileged phase,
#   3. execs the real worker (install.sh / uninstall.sh).
set -euo pipefail

ACTION="${1:?usage: _launcher.sh <install|uninstall>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_SLUG="oracle-tts"
DEFAULT_PREFIX="/opt/oracle-tts"

register_user_assets() {
  local icon_dir="$HOME/.local/share/icons/hicolor/256x256/apps"
  local app_dir="$HOME/.local/share/applications"
  mkdir -p "$icon_dir" "$app_dir"

  # Dedicated iconography (idempotent).
  local png
  for png in "$HERE"/icons/*.png; do
    [ -e "$png" ] || continue
    cp -f "$png" "$icon_dir/"
  done

  # App-grid launchers (idempotent). The install flow later rewrites the
  # uninstall entry with absolute paths once $PREFIX exists.
  local dt
  for dt in "$HERE"/${APP_SLUG}-*.desktop; do
    [ -e "$dt" ] || continue
    cp -f "$dt" "$app_dir/"
    chmod +x "$app_dir/$(basename "$dt")"
    # Mark trusted so GNOME launches without the "untrusted" prompt.
    gio set "$app_dir/$(basename "$dt")" metadata::trusted true 2>/dev/null || true
  done

  gtk-update-icon-cache -f "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1 || true
}

write_uninstall_entry() {
  # Absolute-path app-grid entry, written after a successful install so the
  # uninstaller keeps working even if this repo checkout is gone.
  local user_home="$1" prefix="$2"
  local app_dir="$user_home/.local/share/applications"
  mkdir -p "$app_dir"
  cat > "$app_dir/${APP_SLUG}-uninstall.desktop" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Uninstall The Oracle TTS
Comment=Remove The Oracle TTS service, files and user
Icon=${APP_SLUG}-uninstall
Exec=${prefix}/uninstall.sh
Terminal=true
Categories=System;
Keywords=hype;coin;crypto;uninstall;remove;
StartupNotify=false
EOF
  chmod +x "$app_dir/${APP_SLUG}-uninstall.desktop"
}

case "$ACTION" in
  install)
    register_user_assets
    if [[ "$(id -u)" -ne 0 ]]; then
      echo "Requesting admin rights for the install…"
      exec sudo "$HERE/_launcher.sh" __install_root "$HOME" "$(whoami)"
    else
      exec "$HERE/_launcher.sh" __install_root "$HOME" "$(whoami)"
    fi
    ;;
  __install_root)
    # args: user_home user_name
    "$HERE/install.sh"
    rc=$?
    if [[ $rc -eq 0 ]]; then
      # PREFIX may be overridden in the environment; default matches install.sh.
      prefix="${PREFIX:-$DEFAULT_PREFIX}"
      # Ship the uninstaller with the install tree so the app-grid uninstall
      # entry keeps working even if this repo checkout is gone.
      cp -f "$HERE/uninstall.sh" "$prefix/uninstall.sh"
      chmod +x "$prefix/uninstall.sh"
      write_uninstall_entry "$2" "$prefix"
      echo "Registered 'Uninstall The Oracle TTS' in the app grid."
    else
      echo "Install failed (exit $rc)." >&2
    fi
    echo
    if [ -t 0 ]; then read -rp "Press Enter to close this window. " _ || true; fi
    exit $rc
    ;;
  uninstall)
    register_user_assets
    if [[ "$(id -u)" -ne 0 ]]; then
      echo "Requesting admin rights for the uninstall…"
      exec sudo "$HERE/_launcher.sh" __uninstall_root
    else
      exec "$HERE/_launcher.sh" __uninstall_root
    fi
    ;;
  __uninstall_root)
    "$HERE/uninstall.sh"
    rc=$?
    echo
    if [ -t 0 ]; then read -rp "Press Enter to close this window. " _ || true; fi
    exit $rc
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 1
    ;;
esac
