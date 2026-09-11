#!/usr/bin/env bash
# Wrapper: delegates to the real uninstaller.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "../uninstall_oracle_tts.sh" "$@"
