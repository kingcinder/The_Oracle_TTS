#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  printf 'Oracle TTS venv is missing at %s\nRun %s/bootstrap_oracle_tts.sh first.\n' "$VENV_DIR" "$REPO_ROOT" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
exec python "$REPO_ROOT/scripts/doctor.py" --repo-root "$REPO_ROOT" "$@"
