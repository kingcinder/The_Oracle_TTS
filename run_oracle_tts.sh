#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  printf 'The Oracle is not bootstrapped yet.\nRun %s/bootstrap_oracle_tts.sh first.\n' "$REPO_ROOT" >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  printf 'No GUI display detected. Set DISPLAY or WAYLAND_DISPLAY, then rerun %s/run_oracle_tts.sh.\n' "$REPO_ROOT" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
if [[ -z "${QT_QPA_PLATFORM:-}" && -n "${DISPLAY:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

if command -v the-oracle >/dev/null 2>&1; then
  exec the-oracle gui
fi

exec python -m the_oracle gui
