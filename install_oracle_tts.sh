#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

main() {
  local python_bin
  if ! python_bin="$(select_python)"; then
    printf 'FAIL: Need Python 3.11 or 3.12 with venv support.\n' >&2
    exit 1
  fi
  exec "$python_bin" "$REPO_ROOT/scripts/manage_install.py" install "$@"
}

main "$@"
