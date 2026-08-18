#!/usr/bin/env bash
# Benchmark The Oracle's Vulkan backend batching (--inference-backend vulkan).
#
# The batched path synthesizes every cache-missing stem through audio.cpp
# --request-sequence processes: one model load + Vulkan shader compile per
# process of up to --max-batch requests (forwarded to the CLI as
# --audio-cpp-max-batch), instead of a fresh audiocpp_cli per utterance. This script times a real render on the Vulkan
# backend and reports, from logs/render_timings.json, the per-process overhead
# (model load + shader compile + process spawn) vs the per-utterance cost it
# replaces -- the amortization the batching exists to achieve.
#
# Usage:
#   scripts/benchmark_vulkan_batching.sh --input Input/cli_short.txt \
#     --speakerA-ref "Seashells/Cody's Seashell.wav" \
#     --speakerB-ref "Seashells/Cody's Seashell1.wav"
#
# Options:
#   --input FILE       dialogue to render (required in render mode)
#   --outdir DIR       parent for benchmark runs (default Output/benchmark;
#                      each run renders into a fresh subdirectory so the cache
#                      is cold and every stem is a cache miss)
#   --speakerA-ref / --speakerB-ref   reference wavs (required in render mode)
#   --max-batch N      batch cap (default 32; 1 = one process per utterance,
#                      the naive per-utterance baseline)
#   --sweep            run caps 1 2 4 8 16 32 (or --sweep-caps "1 4 16") and
#                      print a comparison table of the amortization curve
#   --report-only DIR  skip rendering; re-analyze an existing run's
#                      logs/render_timings.json (no binary/model required)
#   --device N / --threads N   forwarded as --audio-cpp-device/--audio-cpp-threads
#   --no-stems         forward --no-stems to the render (skip stem exports)
#   -h | --help
#
# Environment: ORACLE_AUDIOCPP_CLI (built audiocpp_cli; required to render),
# ORACLE_AUDIOCPP_MODEL (chatterbox GGUF path; required to render). The
# per-process batch cap is set via --max-batch and forwarded to the CLI as
# --audio-cpp-max-batch (the same knob the GUI and saved projects use).
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORACLE_CLI="${ORACLE_CLI:-the-oracle}"

INPUT=""
OUTDIR="$REPO_ROOT/Output/benchmark"
SPEAKER_A=""
SPEAKER_B=""
MAX_BATCH=32
SWEEP=0
SWEEP_CAPS="1 2 4 8 16 32"
REPORT_ONLY=""
DEVICE=""
THREADS=""
NO_STEMS=0

usage() {
  sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'
}

# Shared analysis logic, embedded as a python module so every heredoc below
# (report-only, sweep, single-run) stays in lockstep with one definition.
_ANALYZE_PY='
import json


def analyze(data, cap, human=True):
    timeline = data.get("timeline", {})
    summary = data.get("summary", {})
    processes = int(timeline.get("vulkan_batch_processes", 0) or 0)
    requests = int(timeline.get("vulkan_batch_requests", 0) or 0)
    dispatch_window = max(
        0.0,
        float(timeline.get("results_ready_seconds", 0.0) or 0.0)
        - float(timeline.get("dispatch_start_seconds", 0.0) or 0.0),
    )
    pure_synth = float(summary.get("total_synthesize_seconds", 0.0) or 0.0)
    per_entry_overhead = float(summary.get("total_overhead_seconds", 0.0) or 0.0)
    cache_hits = int(summary.get("cache_hits", 0) or 0)
    cache_misses = int(summary.get("cache_misses", 0) or 0)
    render_wall = max(
        0.0,
        float(timeline.get("flac_write_end_seconds", 0.0) or 0.0)
        - float(timeline.get("render_entry_seconds", 0.0) or 0.0),
    )
    # The per-process overhead is everything in the batch window that is not
    # per-request synthesis or per-entry I/O: model load + Vulkan shader
    # compile + process spawn, amortized across the processes actually used.
    per_process_overhead = 0.0
    if processes > 0:
        per_process_overhead = max(
            0.0, dispatch_window - pure_synth - per_entry_overhead
        ) / processes
    # What the render would have cost with one process per utterance (cap=1)
    # vs the batched cost actually paid.
    naive_total = pure_synth + per_entry_overhead + per_process_overhead * requests
    amortized_total = pure_synth + per_entry_overhead + per_process_overhead * processes
    saved = naive_total - amortized_total
    saved_pct = (saved / naive_total * 100.0) if naive_total > 0 else 0.0

    if not human:
        print(
            "{cap}\t{processes}\t{requests}\t{dispatch:.2f}\t{naive:.2f}\t"
            "{amortized:.2f}\t{saved_pct:.1f}".format(
                cap=cap,
                processes=processes,
                requests=requests,
                dispatch=dispatch_window,
                naive=naive_total,
                amortized=amortized_total,
                saved_pct=saved_pct,
            )
        )
        return

    title = f"== Vulkan batching report (max-batch={cap}) ==" if cap else "== Vulkan batching report =="
    print(title)
    print(f"  render wall:             {render_wall:8.2f} s")
    print(f"  batch dispatch window:   {dispatch_window:8.2f} s")
    print(f"  pure synthesis:          {pure_synth:8.2f} s  (per-request [TIMING] wall_ms)")
    print(f"  per-entry overhead:      {per_entry_overhead:8.2f} s  (stem load/write)")
    print(f"  audio.cpp processes:     {processes:8d}  (<= {requests} requests at cap {cap})")
    if processes == 0:
        print("  NOTE: no batch processes recorded (all stems cached? non-vulkan run?).")
        print("  Re-run with a fresh --outdir so every stem is a cache miss.")
        return
    print(f"  per-process overhead:    {per_process_overhead:8.2f} s  (model load + shader compile + spawn)")
    print(f"  naive per-utterance:     {naive_total:8.2f} s  (overhead x {requests} requests)")
    print(f"  batched (actual):        {amortized_total:8.2f} s  (overhead x {processes} processes)")
    print(f"  saved by batching:       {saved:8.2f} s  ({saved_pct:.1f}%)")
    print(f"  cache hits/misses:       {cache_hits}/{cache_misses}")
'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="${2:?--input requires a value}"
      shift 2
      ;;
    --outdir)
      OUTDIR="${2:?--outdir requires a value}"
      shift 2
      ;;
    --speakerA-ref)
      SPEAKER_A="${2:?--speakerA-ref requires a value}"
      shift 2
      ;;
    --speakerB-ref)
      SPEAKER_B="${2:?--speakerB-ref requires a value}"
      shift 2
      ;;
    --max-batch)
      MAX_BATCH="${2:?--max-batch requires a value}"
      shift 2
      ;;
    --sweep)
      SWEEP=1
      shift
      ;;
    --sweep-caps)
      SWEEP_CAPS="${2:?--sweep-caps requires a value}"
      shift 2
      ;;
    --report-only)
      REPORT_ONLY="${2:?--report-only requires a directory}"
      shift 2
      ;;
    --device)
      DEVICE="${2:?--device requires a value}"
      shift 2
      ;;
    --threads)
      THREADS="${2:?--threads requires a value}"
      shift 2
      ;;
    --no-stems)
      NO_STEMS=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1 (try --help)" >&2
      exit 1
      ;;
  esac
done

# The report-only analysis needs neither the render inputs nor the audio.cpp
# binary/model; it just re-parses an existing run's timings JSON.
if [[ -n "$REPORT_ONLY" ]]; then
  TIMINGS="$REPORT_ONLY/logs/render_timings.json"
  if [[ ! -f "$TIMINGS" ]]; then
    echo "ERROR: no render timings found at $TIMINGS" >&2
    echo "Run the benchmark against the Vulkan backend first." >&2
    exit 1
  fi
  python3 - "$TIMINGS" "" <<PY
$_ANALYZE_PY
import json, sys
path, cap = sys.argv[1], sys.argv[2]
data = json.load(open(path, encoding="utf-8"))
analyze(data, cap, human=True)
PY
  exit 0
fi

if [[ -z "$INPUT" || -z "$SPEAKER_A" || -z "$SPEAKER_B" ]]; then
  echo "ERROR: render mode requires --input, --speakerA-ref, and --speakerB-ref." >&2
  usage
  exit 1
fi

if ! command -v "$ORACLE_CLI" >/dev/null 2>&1; then
  echo "ERROR: $ORACLE_CLI is not on PATH (set ORACLE_CLI if it is elsewhere)." >&2
  exit 1
fi
if [[ -z "${ORACLE_AUDIOCPP_CLI:-}" || ! -x "$ORACLE_AUDIOCPP_CLI" ]]; then
  echo "ERROR: ORACLE_AUDIOCPP_CLI is unset or not executable." >&2
  echo "Build it first with scripts/build_audio_cpp.sh" >&2
  exit 1
fi
if [[ -z "${ORACLE_AUDIOCPP_MODEL:-}" || ! -f "$ORACLE_AUDIOCPP_MODEL" ]]; then
  echo "ERROR: ORACLE_AUDIOCPP_MODEL is unset or the file is missing." >&2
  echo "Fetch it first with scripts/download_audio_cpp_model.sh" >&2
  exit 1
fi

# Current wall clock in nanoseconds since epoch. GNU date supports %s%N;
# elsewhere fall back to whole-second precision scaled to the same unit.
now_ns() {
  local ns
  if ns="$(date +%s%N 2>/dev/null)"; then
    echo "$ns"
  else
    echo "$(date +%s)000000000"
  fi
}

# Run one render at a given cap in a fresh, cold-cache outdir, then analyze.
# Diagnostics go to stderr; stdout carries ONLY the timings path, because the
# caller captures this function's output into a variable. The cap is passed
# as --audio-cpp-max-batch (the CLI-threaded setting) rather than the raw
# env var, so the sweep exercises the same knob the GUI/manifest use.
run_render() {
  local cap="$1" run_dir="$2"
  mkdir -p "$run_dir"
  echo "[bench] rendering with --audio-cpp-max-batch=$cap into $run_dir" >&2
  local start_s end_s
  start_s="$(now_ns)"
  "$ORACLE_CLI" render \
    --input "$INPUT" \
    --outdir "$run_dir" \
    --speakerA-ref "$SPEAKER_A" \
    --speakerB-ref "$SPEAKER_B" \
    --inference-backend vulkan \
    --audio-cpp-max-batch "$cap" \
    ${DEVICE:+--audio-cpp-device "$DEVICE"} \
    ${THREADS:+--audio-cpp-threads "$THREADS"} \
    ${NO_STEMS:+--no-stems} >/dev/null
  end_s="$(now_ns)"
  awk -v s="$start_s" -v e="$end_s" 'BEGIN { printf "[bench] CLI wall time: %.2f s\n", (e - s) / 1e9 }' >&2
  local timings="$run_dir/logs/render_timings.json"
  if [[ ! -f "$timings" ]]; then
    echo "ERROR: render finished but no $timings was written." >&2
    exit 1
  fi
  echo "$timings"
}

if [[ "$SWEEP" -eq 1 ]]; then
  echo "[bench] sweep across max-batch caps: $SWEEP_CAPS"
  results=()
  dirs=()
  for cap in $SWEEP_CAPS; do
    run_dir="$OUTDIR/benchmark-cap-$cap-$(date +%Y%m%d-%H%M%S)"
    timings="$(run_render "$cap" "$run_dir")"
    dirs+=("$run_dir")
    row="$(python3 - "$timings" "$cap" <<PY
$_ANALYZE_PY
import json, sys
path, cap = sys.argv[1], sys.argv[2]
data = json.load(open(path, encoding="utf-8"))
analyze(data, cap, human=False)
PY
)"
    results+=("$row")
  done
  echo
  echo "[bench] amortization curve (fresh cold-cache render per cap):"
  printf "  %-6s %-10s %-9s %-12s %-12s %-12s %s\n" "cap" "processes" "requests" "dispatch(s)" "naive(s)" "batched(s)" "saved%"
  for row in "${results[@]}"; do
    IFS=$'\t' read -r cap processes requests dispatch naive batched saved_pct <<<"$row"
    printf "  %-6s %-10s %-9s %-12s %-12s %-12s %s\n" "$cap" "$processes" "$requests" "$dispatch" "$naive" "$batched" "$saved_pct"
  done
  echo
  echo "[bench] run directories (keep these for --report-only):"
  for d in "${dirs[@]}"; do
    echo "  $d"
  done
else
  run_dir="$OUTDIR/benchmark-cap-$MAX_BATCH-$(date +%Y%m%d-%H%M%S)"
  timings="$(run_render "$MAX_BATCH" "$run_dir")"
  echo
  python3 - "$timings" "$MAX_BATCH" <<PY
$_ANALYZE_PY
import json, sys
path, cap = sys.argv[1], sys.argv[2]
data = json.load(open(path, encoding="utf-8"))
analyze(data, cap, human=True)
PY
  echo
  echo "[bench] re-analyze later with: $0 --report-only $run_dir"
fi
