#!/usr/bin/env bash
# Wrapper: delegates to the real installer.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "../install_oracle_tts.sh" "$@"
