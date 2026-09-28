#!/usr/bin/env bash
# U4.2 crash catcher — recreated 2026-09-28 (previous instance lost to a
# machine restart on 2026-09-11; original spec: gdb batch launcher, native
# backtrace -> /tmp/gui_crash_backtrace.log, Python stacks ->
# /tmp/gui_crash_pyfaulthandler.log).
#
# Usage: bash /tmp/gui_crash_catcher.sh [python-script-or-args...]
#   no args: interactive GUI session under gdb (original mode)
#   args:    run the given script under gdb batch, e.g.
#            bash /tmp/gui_crash_catcher.sh scripts/u42_crash_repro.py 2
#
# Outputs:
#   /tmp/gui_crash_backtrace.log        gdb native backtrace on crash (empty
#                                       if no crash)
#   /tmp/gui_crash_pyfaulthandler.log   PYTHONFAULTHANDLER=1 native Python stacks
#   /tmp/gui_crash_gdb_full.log         full gdb session (for context)
set -u

REPO="/home/cody/Documents/the oracle tts"
PY="$REPO/.venv/bin/python"
LOG_BT="/tmp/gui_crash_backtrace.log"
LOG_PY="/tmp/gui_crash_pyfaulthandler.log"
LOG_FULL="/tmp/gui_crash_gdb_full.log"

: > "$LOG_BT"
: > "$LOG_PY"
: > "$LOG_FULL"

export PYTHONFAULTHANDLER=1
cd "$REPO"

GDB_CMDS=/tmp/gui_crash_gdb_cmds
if [ "$#" -eq 0 ]; then
  # Original mode: interactive GUI session under gdb.
  cat > "$GDB_CMDS" <<'EOF'
set pagination off
set confirm off
handle SIGSEGV stop print
run -m the_oracle.cli gui
bt full
python-thread-apply-all-bt
EOF
  echo "[catcher] interactive GUI under gdb; crash backtrace -> $LOG_BT"
  gdb -q -batch -x "$GDB_CMDS" -args "$PY" >"$LOG_FULL" 2>&1
  gdb -q -batch -ex "set pagination off" -ex "run -m the_oracle.cli gui" -ex "bt full" -args "$PY" >"$LOG_BT" 2>&1
else
  # Scripted mode: drive a repro pass under gdb batch.
  cat > "$GDB_CMDS" <<'EOF'
set pagination off
set confirm off
run
bt full
EOF
  echo "[catcher] driving: $*"
  gdb -q -batch -x "$GDB_CMDS" -args "$PY" "$@" >"$LOG_FULL" 2>&1
  cp "$LOG_FULL" "$LOG_BT"
fi

# Only keep the tail with the actual crash evidence for the backtrace log.
if grep -q "SIGSEGV\|Segmentation fault" "$LOG_FULL"; then
  echo "[catcher] CRASH CAPTURED — backtrace in $LOG_BT, python stacks in $LOG_PY"
  exit 42
else
  echo "[catcher] no native crash (session ended normally)"
  exit 0
fi
