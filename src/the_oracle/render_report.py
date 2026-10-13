"""Studio render report — a machine-readable conformance summary of a render.

This is the first **real** Studio-gated capability, added in 1.4.0 and
deliberately **additive**: nothing that shipped before is gated, so the
community edition loses no workflow (docs/LICENSING_DESIGN.md §4 — community is
everything that shipped, and the first gated feature is a new one). The owner
can move the gate to a different feature, or drop it, by changing the single
``policy.require(status, STUDIO_ENTITLEMENT)`` call site — see
docs/LICENSING_OPS.md.

The report is derived entirely from artifacts the render already produced: the
assembled FLAC and its ``render_plan.json``. It measures, it does not re-render,
so producing one is cheap and cannot alter the audio.

What it answers, for a customer doing paid work: is this deliverable technically
sound? Peak/RMS level and clipping, silence ratio, sample rate and channel
count, utterance and speaker counts from the plan, and whether the plan's
declared durations agree with the audio that was actually assembled.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

from the_oracle import __version__

#: The entitlement this capability requires. One name, one call site.
STUDIO_ENTITLEMENT = "studio_features"

#: Written next to the render by ``--out``-less runs when a path is derived.
REPORT_SUFFIX = ".report.json"

#: Below this, a sample counts as silence (``-80 dBFS``).
_SILENCE_AMPLITUDE = 1e-4

#: At or above this, a sample counts as full scale. FLAC is integer PCM, so a
#: full-scale sample read back as float is 1 - 2**-15 (16-bit) or 1 - 2**-23
#: (24-bit), never exactly 1.0; a naive ``>= 1.0`` test would therefore report
#: zero clipping on a render that is entirely clipped (measured: reading a
#: full-scale 16-bit FLAC back through soundfile yields 0.999969).
_CLIP_LEVEL = 0.9999

#: Reported floor for a completely silent (or empty) render.
_DBFS_FLOOR = -120.0


class ReportError(Exception):
    """The report cannot be produced (missing project, no render, unreadable)."""


def _dbfs(amplitude: float) -> float:
    if amplitude <= 0:
        return _DBFS_FLOOR
    return round(20.0 * math.log10(amplitude), 2)


def find_render(project_dir: Path) -> Path:
    """The single assembled FLAC in ``project_dir``, or ReportError.

    Deliberately strict: more than one FLAC means the caller pointed at a folder
    holding several renders (or a stray export), and guessing which one the
    report is about would make the report a lie.
    """
    if not project_dir.is_dir():
        raise ReportError(f"{project_dir} is not a directory.")
    flacs = sorted(path for path in project_dir.glob("*.flac") if path.is_file())
    if not flacs:
        raise ReportError(
            f"no rendered .flac in {project_dir} — point the report at the project "
            "directory a render wrote (it holds the FLAC and render_plan.json)."
        )
    if len(flacs) > 1:
        names = ", ".join(path.name for path in flacs[:5])
        raise ReportError(
            f"{project_dir} holds {len(flacs)} .flac files ({names}); the report "
            "covers exactly one render, so pass a project directory with one."
        )
    return flacs[0]


def _plan_summary(plan_path: Path) -> tuple[dict[str, object], list[str]]:
    """Utterance/speaker counts from ``render_plan.json``; never fatal.

    A missing or unreadable plan degrades the report (a warning) instead of
    failing it: the audio measurements are the part a deliverable check needs,
    and the plan is corroborating context.
    """
    warnings: list[str] = []
    if not plan_path.is_file():
        warnings.append(f"no render_plan.json beside the render ({plan_path.name} is absent)")
        return {"utterances": 0, "speakers": {}, "declared_duration_seconds": None}, warnings
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        warnings.append(f"render_plan.json could not be read ({error})")
        return {"utterances": 0, "speakers": {}, "declared_duration_seconds": None}, warnings

    utterances = plan.get("utterances")
    if not isinstance(utterances, list):
        warnings.append("render_plan.json has no utterance list")
        return {"utterances": 0, "speakers": {}, "declared_duration_seconds": None}, warnings

    speakers: dict[str, int] = {}
    statuses: dict[str, int] = {}
    declared = 0.0
    for entry in utterances:
        if not isinstance(entry, dict):
            continue
        speaker = str(entry.get("speaker") or "?")
        speakers[speaker] = speakers.get(speaker, 0) + 1
        status = str(entry.get("status") or "?")
        statuses[status] = statuses.get(status, 0) + 1
        try:
            declared += float(entry.get("duration_seconds") or 0.0)
        except (TypeError, ValueError):
            pass
    return (
        {
            "utterances": len(utterances),
            "speakers": dict(sorted(speakers.items())),
            "speaker_count": len(speakers),
            "status_counts": dict(sorted(statuses.items())),
            "declared_duration_seconds": round(declared, 3) if declared else None,
        },
        warnings,
    )


def build_report(project_dir: Path) -> dict[str, object]:
    """Measure ``project_dir``'s render and return the report document.

    Raises ReportError when there is nothing to measure; every other failure is
    carried in ``warnings`` so a partly-degraded report is still useful.
    """
    import soundfile  # local import: the report is a leaf, keep import cost off CLI start

    project_dir = Path(project_dir)
    flac = find_render(project_dir)
    try:
        info = soundfile.info(str(flac))
        data, rate = soundfile.read(str(flac), always_2d=True, dtype="float32")
    except (OSError, RuntimeError) as error:
        raise ReportError(f"could not read {flac.name}: {error}") from error

    warnings: list[str] = []
    frames = int(data.shape[0]) if data.size else 0
    peak = 0.0
    rms = 0.0
    clipped = 0
    silent = 0
    if frames:
        import numpy as np

        magnitude = np.abs(data)
        peak = float(magnitude.max())
        rms = float(np.sqrt(np.mean(np.square(data))))
        clipped = int(np.count_nonzero(magnitude >= _CLIP_LEVEL))
        silent = int(np.count_nonzero(magnitude.max(axis=1) < _SILENCE_AMPLITUDE))
        if clipped:
            warnings.append(
                f"{clipped} sample(s) at or above full scale — the render clips; "
                "lower the speaker level or the loudness setting and re-render."
            )

    plan_summary, plan_warnings = _plan_summary(project_dir / "render_plan.json")
    warnings.extend(plan_warnings)

    duration = round(frames / float(rate), 3) if rate else 0.0
    declared = plan_summary.get("declared_duration_seconds")
    if isinstance(declared, (int, float)) and declared and abs(float(declared) - duration) > 1.0:
        warnings.append(
            f"the plan declares {float(declared):.3f}s of speech but the audio is "
            f"{duration:.3f}s — they disagree by more than 1s; check the render completed."
        )

    return {
        "report_version": 1,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "oracle_version": __version__,
        "project_dir": str(project_dir),
        "output": {
            "file": flac.name,
            "format": info.format,
            "subtype": info.subtype,
            "sample_rate": int(rate),
            "channels": int(info.channels),
            "frames": frames,
            "duration_seconds": duration,
        },
        "loudness": {
            "peak_dbfs": _dbfs(peak),
            "rms_dbfs": _dbfs(rms),
            "clipped_samples": clipped,
            "silent_frame_ratio": round(silent / frames, 4) if frames else 0.0,
        },
        "plan": plan_summary,
        # Honest watermark line: a render is watermarked by Chatterbox's Perth
        # stack by design, and this project never removes it. The report states
        # which detector would confirm that, and whether it is importable here —
        # it does not claim to have run detection.
        "watermark": {
            "expected": True,
            "detector": "perth.perth_net",
            "detector_available": _perth_available(),
            "note": "renders carry Chatterbox/Perth watermarking by design; "
            "this report does not run detection.",
        },
        "warnings": warnings,
    }


def _perth_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("perth") is not None


def report_text(report: dict[str, object]) -> str:
    """A short human rendition of the same document."""
    output = report["output"]  # type: ignore[index]
    loudness = report["loudness"]  # type: ignore[index]
    plan = report["plan"]  # type: ignore[index]
    lines = [
        f"Render report — {output['file']}",  # type: ignore[index]
        f"  format          : {output['format']}/{output['subtype']}, "
        f"{output['sample_rate']} Hz, {output['channels']} ch, {output['frames']} frames",  # type: ignore[index]
        f"  duration        : {output['duration_seconds']} s",  # type: ignore[index]
        f"  level           : peak {loudness['peak_dbfs']} dBFS, rms {loudness['rms_dbfs']} dBFS",  # type: ignore[index]
        f"  clipped samples : {loudness['clipped_samples']}",  # type: ignore[index]
        f"  silence ratio   : {loudness['silent_frame_ratio']}",  # type: ignore[index]
        f"  utterances      : {plan['utterances']} across {plan.get('speaker_count', 0)} speaker(s)",  # type: ignore[index]
        f"  speakers        : {plan['speakers']}",  # type: ignore[index]
        f"  watermark       : expected={report['watermark']['expected']}, "  # type: ignore[index]
        f"detector_available={report['watermark']['detector_available']}",  # type: ignore[index]
    ]
    for warning in report.get("warnings") or []:  # type: ignore[union-attr]
        lines.append(f"  WARNING: {warning}")
    return "\n".join(lines)


def write_report(report: dict[str, object], path: Path) -> Path:
    """Write the report JSON atomically (temp + replace) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
