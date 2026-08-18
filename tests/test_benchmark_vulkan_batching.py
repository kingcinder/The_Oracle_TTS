"""Deterministic tests for scripts/benchmark_vulkan_batching.sh (offline).

The benchmark renders a real dialogue on the Vulkan backend, which needs a
built audiocpp_cli and the chatterbox GGUF -- far too heavy for tests. These
tests exercise the two offline halves instead:

* the argument parsing / preflight guards, which must fail fast with clear
  messages before any render is attempted, and
* the ``--report-only`` analysis path, which re-parses an existing run's
  logs/render_timings.json. A fixture with hand-checkable numbers verifies
  the per-process overhead math (model load + shader compile amortization)
  without any binary, model, or GPU.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "benchmark_vulkan_batching.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("python3") is None,
    reason="python3 is not installed (the report step requires it)",
)


def _run_script(
    *args: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=cwd or REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _env_without(*keys: str, **overrides: str) -> dict[str, str]:
    """os.environ minus the audio.cpp vars, plus overrides.

    The benchmark reads ORACLE_AUDIOCPP_CLI / ORACLE_AUDIOCPP_MODEL /
    ORACLE_AUDIOCPP_MAX_BATCH from the environment, and a dev machine may
    legitimately have them exported. Tests that probe the preflight guards
    must be hermetic, so those keys are always stripped.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("ORACLE_AUDIOCPP_CLI", "ORACLE_AUDIOCPP_MODEL", "ORACLE_AUDIOCPP_MAX_BATCH")
    }
    env.update(overrides)
    return env


FAKE_CLI = """#!/usr/bin/env bash
# Minimal stand-in for the-oracle render: parses --outdir and copies a fixture
# logs/render_timings.json into it, so the benchmark's render path runs end to
# end without a GPU, model, or binary. FAKE_TIMINGS_FIXTURE must point at the
# fixture JSON. All other args are recorded verbatim into
# $OUTDIR/logs/cli_args.txt (for asserting the benchmark forwards the right
# flags, e.g. --audio-cpp-max-batch) but otherwise ignored.
set -euo pipefail
OUTDIR=""
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--outdir" ]]; then
    OUTDIR="$arg"
  fi
  prev="$arg"
done
if [[ -z "$OUTDIR" ]]; then
  echo "fake cli: missing --outdir" >&2
  exit 1
fi
mkdir -p "$OUTDIR/logs"
printf '%s\n' "$@" > "$OUTDIR/logs/cli_args.txt"
cp "$FAKE_TIMINGS_FIXTURE" "$OUTDIR/logs/render_timings.json"
echo "fake cli: rendered into $OUTDIR"
"""


def _write_fake_cli(tmp_path: Path) -> Path:
    cli = tmp_path / "fake-the-oracle"
    cli.write_text(FAKE_CLI, encoding="utf-8")
    cli.chmod(0o755)
    return cli


def _render_env(tmp_path: Path, fixture_json: Path, fake_cli: Path) -> dict[str, str]:
    """A hermetic environment that passes the benchmark's preflight guards:
    a fake render CLI on PATH (via ORACLE_CLI), an executable audiocpp_cli
    stub, and an existing model file -- then runs the render against the
    fixture timings the fake CLI writes."""
    cli = tmp_path / "audiocpp_cli"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stub")
    return _env_without(
        ORACLE_CLI=str(fake_cli),
        ORACLE_AUDIOCPP_CLI=str(cli),
        ORACLE_AUDIOCPP_MODEL=str(model),
        FAKE_TIMINGS_FIXTURE=str(fixture_json),
    )


def _fixture_timings(outdir: Path) -> Path:
    """A realistic logs/render_timings.json with hand-checkable numbers.

    dispatch window 10.0 s (12.0 - 2.0), pure synthesis 5.0 s, per-entry
    overhead 1.0 s, 2 processes / 8 requests:

      per-process overhead = (10.0 - 5.0 - 1.0) / 2 = 2.0 s
      naive per-utterance  = 5.0 + 1.0 + 2.0 * 8   = 22.0 s
      batched (actual)     = 5.0 + 1.0 + 2.0 * 2   = 10.0 s
      saved                = 12.0 s (54.5%)
    """
    entries = [
        {
            "type": "utterance",
            "segment_number": i + 1,
            "index": i,
            "speaker": "A" if i % 2 == 0 else "B",
            "cache_hit": False,
            "chunk_hash": f"hash{i}",
            "duration_seconds": 2.0,
            "synthesize_seconds": round(5.0 / 8, 6),
            "load_audio_seconds": round(1.0 / 8, 6),
            "segment_total_seconds": round(6.0 / 8, 6),
            "inter_segment_overhead_seconds": round(1.0 / 8, 6),
        }
        for i in range(8)
    ]
    payload = {
        "count": 8,
        "entries": entries,
        "output": {"path": str(outdir / "out.flac")},
        "timeline": {
            "render_entry_seconds": 0.0,
            "flac_write_end_seconds": 15.0,
            "dispatch_start_seconds": 2.0,
            "results_ready_seconds": 12.0,
            "vulkan_batch_processes": 2.0,
            "vulkan_batch_requests": 8.0,
        },
        "summary": {
            "utterance_count": 8,
            "segment_count": 8,
            "join_count": 7,
            "cache_hits": 0,
            "cache_misses": 8,
            "total_synthesize_seconds": 5.0,
            "total_overhead_seconds": 1.0,
        },
    }
    logs = outdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "render_timings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return outdir


def test_help_lists_flags_and_exits_without_side_effects(tmp_path: Path) -> None:
    result = _run_script("--help", cwd=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    for flag in ("--report-only", "--sweep", "--max-batch", "--input", "--speakerA-ref"):
        assert flag in result.stdout
    # Help must not try to render or touch the audio.cpp binary.
    assert "--inference-backend vulkan" in result.stdout


def test_unknown_argument_fails(tmp_path: Path) -> None:
    result = _run_script("--bogus", cwd=tmp_path)

    assert result.returncode != 0
    assert "unknown argument" in result.stderr


def test_render_mode_requires_input_and_refs(tmp_path: Path) -> None:
    result = _run_script("--input", str(tmp_path / "x.txt"), cwd=tmp_path)

    assert result.returncode != 0
    assert "--speakerA-ref" in result.stderr


def test_render_mode_preflight_requires_audio_cpp_binary(tmp_path: Path) -> None:
    """Render mode must fail fast when ORACLE_AUDIOCPP_CLI is unset, before
    attempting any render. ORACLE_CLI points at a real executable so the
    preflight reaches the audio.cpp check."""
    result = _run_script(
        "--input",
        str(tmp_path / "in.txt"),
        "--outdir",
        str(tmp_path / "out"),
        "--speakerA-ref",
        "a.wav",
        "--speakerB-ref",
        "b.wav",
        env=_env_without(ORACLE_CLI="/bin/true"),
    )

    assert result.returncode != 0
    assert "ORACLE_AUDIOCPP_CLI" in result.stderr
    assert "build_audio_cpp.sh" in result.stderr


def test_render_mode_preflight_requires_model_file(tmp_path: Path) -> None:
    cli = tmp_path / "audiocpp_cli"
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    cli.chmod(0o755)
    result = _run_script(
        "--input",
        str(tmp_path / "in.txt"),
        "--outdir",
        str(tmp_path / "out"),
        "--speakerA-ref",
        "a.wav",
        "--speakerB-ref",
        "b.wav",
        env=_env_without(ORACLE_CLI="/bin/true", ORACLE_AUDIOCPP_CLI=str(cli)),
    )

    assert result.returncode != 0
    assert "ORACLE_AUDIOCPP_MODEL" in result.stderr
    assert "download_audio_cpp_model.sh" in result.stderr


def test_report_only_missing_timings_fails(tmp_path: Path) -> None:
    result = _run_script("--report-only", str(tmp_path), cwd=tmp_path)

    assert result.returncode != 0
    assert "no render timings found" in result.stderr


def test_report_only_analyzes_fixture_timings(tmp_path: Path) -> None:
    outdir = _fixture_timings(tmp_path)
    result = _run_script("--report-only", str(outdir), cwd=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "== Vulkan batching report ==" in out
    # Window math from the fixture timeline.
    assert "render wall" in out and "15.00" in out
    assert "dispatch window" in out and "10.00" in out
    assert "pure synthesis" in out and "5.00" in out
    assert "per-entry overhead" in out and "1.00" in out
    assert "2" in out and "8 requests" in out
    # The amortization numbers (see _fixture_timings docstring).
    assert "per-process overhead" in out and "2.00" in out
    assert "naive per-utterance" in out and "22.00" in out
    assert "batched (actual)" in out and "10.00" in out
    assert re.search(r"saved by batching:.*12\.00.*54\.5%", out)
    assert "cache hits/misses" in out and "0/8" in out
    # Processes/requests line: 2 processes serving 8 requests.
    assert re.search(r"audio\.cpp processes:.*\b2\b.*8 requests", out)


def test_render_mode_runs_end_to_end_with_fake_cli(tmp_path: Path) -> None:
    """The full render path (preflight -> run_render -> report) works when the
    CLI writes a timings JSON: run_render's stdout must carry ONLY the timings
    path (diagnostics go to stderr) or the python report could not open it.
    This is the regression test for exactly that capture bug."""
    fake_cli = _write_fake_cli(tmp_path)
    fixture_source = _fixture_timings(tmp_path / "fixture-src")
    fixture_json = fixture_source / "logs" / "render_timings.json"

    result = _run_script(
        "--input",
        str(tmp_path / "in.txt"),
        "--outdir",
        str(tmp_path / "bench"),
        "--speakerA-ref",
        "a.wav",
        "--speakerB-ref",
        "b.wav",
        "--max-batch",
        "8",
        env=_render_env(tmp_path, fixture_json, fake_cli),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "== Vulkan batching report (max-batch=8) ==" in out
    # The fake CLI wrote the fixture timings into the fresh run dir, and the
    # report analyzed them (fixture: naive 22.00 s, batched 10.00 s, 54.5%).
    assert re.search(r"naive per-utterance.*22\.00", out)
    assert re.search(r"batched \(actual\).*10\.00", out)
    assert re.search(r"saved by batching:.*54\.5%", out)
    # Wall-time diagnostics belong on stderr, not in the captured path.
    assert "CLI wall time" in result.stderr
    # The cap is forwarded as the CLI-threaded setting (--audio-cpp-max-batch),
    # not the raw ORACLE_AUDIOCPP_MAX_BATCH env var.
    args_files = list((tmp_path / "bench").rglob("logs/cli_args.txt"))
    assert args_files, "fake CLI never recorded its argv"
    args_text = args_files[0].read_text(encoding="utf-8")
    assert "--audio-cpp-max-batch" in args_text
    assert "8" in args_text.split()
    assert "--inference-backend" in args_text and "vulkan" in args_text


def test_sweep_mode_renders_each_cap_and_prints_table(tmp_path: Path) -> None:
    fake_cli = _write_fake_cli(tmp_path)
    fixture_source = _fixture_timings(tmp_path / "fixture-src")
    fixture_json = fixture_source / "logs" / "render_timings.json"

    result = _run_script(
        "--input",
        str(tmp_path / "in.txt"),
        "--outdir",
        str(tmp_path / "bench"),
        "--speakerA-ref",
        "a.wav",
        "--speakerB-ref",
        "b.wav",
        "--sweep",
        "--sweep-caps",
        "1 8",
        env=_render_env(tmp_path, fixture_json, fake_cli),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "amortization curve" in out
    # Table header plus one full row per cap. The fixture yields identical
    # numbers for both caps (2 processes / 8 requests / dispatch 10.00 /
    # naive 22.00 / batched 10.00 / 54.5%), so the row regex also locks in
    # the TSV field mapping the bash table parses: cap, processes, requests,
    # dispatch, naive, batched, saved_pct in that exact order.
    assert re.search(r"\b1\s+2\s+8\s+10\.00\s+22\.00\s+10\.00\s+54\.5", out)
    assert re.search(r"\b8\s+2\s+8\s+10\.00\s+22\.00\s+10\.00\s+54\.5", out)
    assert out.count("54.5") >= 2
    # Fresh run directory per cap, listed for --report-only reuse.
    assert "benchmark-cap-1-" in out
    assert "benchmark-cap-8-" in out
    assert "--report-only" in out
