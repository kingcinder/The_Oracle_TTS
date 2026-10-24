"""User-initiated, redacted support bundle.

A support bundle is the one artifact an alpha user can hand to support without
also handing over their manuscript, audio, recordings, serial number, or home
directory. Everything is collected locally, redacted before it is written, and
placed in a zip the user chooses to send; The Oracle never uploads it.

Import discipline: stdlib plus the Oracle's own small modules. No Qt, no torch,
no network, and nothing runs at import time.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from the_oracle import __version__

BUNDLE_VERSION = 2
MAX_LOG_FILES = 3
MAX_LOG_BYTES = 256 * 1024
MAX_CRASH_RECORDS = 20
DOCTOR_TIMEOUT_SECONDS = 120

#: Key fragments whose value is never included, even redacted.
_SENSITIVE_KEY_FRAGMENTS = (
    "token",
    "license",
    "licensee",
    "machine_hash",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "credential",
    "private_key",
    "signing",
)

#: Key fragments whose string value is replaced because it names a file.
_PATH_KEY_FRAGMENTS = (
    "path",
    "file",
    "dir",
    "folder",
    "recent",
    "input",
    "output",
    "home",
)

_TOKEN_RE = re.compile(r"ORACLE1\.[A-Za-z0-9_\-=.]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_WINDOWS_HOME_RE = re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE)
_API_KEY_RES = (
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{10,}\b"),
)


@dataclass(frozen=True, slots=True)
class SupportBundleResult:
    path: Path
    manifest: dict[str, Any]
    included: tuple[str, ...]
    skipped: tuple[str, ...]


def _home_text() -> str:
    try:
        return str(Path.home())
    except (OSError, RuntimeError):  # pragma: no cover - pathological
        return ""


def redact_text(text: str) -> str:
    """Remove tokens, credentials, emails, and home-directory prefixes."""
    if not text:
        return ""
    value = str(text)
    home = _home_text()
    if home and home != "/" and home in value:
        value = value.replace(home, "~")
    value = _WINDOWS_HOME_RE.sub("%USERPROFILE%", value)
    value = _TOKEN_RE.sub("[REDACTED LICENSE TOKEN]", value)
    for pattern in _API_KEY_RES:
        value = pattern.sub("[REDACTED CREDENTIAL]", value)
    value = _EMAIL_RE.sub("[redacted email]", value)
    return value


def _redact_key_value(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS):
        return "[REDACTED]"
    if isinstance(value, str) and any(fragment in lowered for fragment in _PATH_KEY_FRAGMENTS):
        return "[redacted path]"
    return redact_json(value)


def redact_json(value: Any) -> Any:
    """Recursively redact a JSON-compatible value."""
    if isinstance(value, dict):
        return {str(key): _redact_key_value(str(key), item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_json(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _environment_payload(root: Path) -> dict[str, Any]:
    names = sorted(
        name
        for name in os.environ
        if name.startswith("ORACLE_")
        or name in {"HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES", "QT_QPA_PLATFORM", "PYTHONPATH"}
    )
    return {
        "app_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        # The interpreter's full path can carry a username (a home directory or
        # a mounted volume); support only needs to know which interpreter ran.
        "executable": Path(sys.executable).name or "<python>",
        "repository_root": "install",
        "environment_variables_present": names,
        "environment_values_included": False,
    }


def _license_payload(root: Path) -> dict[str, Any]:
    from the_oracle.licensing import current_license

    status = current_license(root)
    return {
        "ok": status.ok,
        "state": status.state,
        "key_id": status.key_id,
        "exp": status.exp,
        "machine_locked": status.machine_locked,
        "detail": redact_text(status.detail or ""),
        "token_included": False,
        "licensee_included": False,
    }


def _settings_payload(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = root / "user_settings" / "app_settings.json"
    if not path.is_file():
        return None, "user_settings/app_settings.json is absent"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, f"user_settings/app_settings.json could not be read ({error})"
    if not isinstance(data, dict):
        return None, "user_settings/app_settings.json is not a JSON object"
    return redact_json(data), None


def _log_entries(root: Path) -> tuple[dict[str, bytes], list[str]]:
    entries: dict[str, bytes] = {}
    skipped: list[str] = []
    directory = root / "logs"
    if not directory.is_dir():
        return entries, ["logs/ is absent"]
    candidates = sorted(
        (path for path in directory.iterdir() if path.is_file()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )[:MAX_LOG_FILES]
    for index, path in enumerate(candidates, start=1):
        try:
            raw = path.read_bytes()[:MAX_LOG_BYTES]
        except OSError as error:
            skipped.append(f"logs/{path.name} could not be read ({error})")
            continue
        text = raw.decode("utf-8", errors="replace")
        entries[f"logs/{index:02d}-{path.name}"] = redact_text(text).encode("utf-8")
    if not entries:
        skipped.append("no readable log files")
    return entries, skipped


def _crash_entries(root: Path, include: bool) -> tuple[dict[str, bytes], list[str]]:
    from the_oracle import crash

    if not include:
        return {}, ["crash reports excluded (pass --include-crash-reports to opt in)"]
    if not crash.read_consent(root):
        return {}, ["crash reports not included: local crash-reporting consent is off"]
    entries: dict[str, bytes] = {}
    records = crash.list_records(root)[:MAX_CRASH_RECORDS]
    for index, path in enumerate(records, start=1):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            entries[f"crash_reports/{index:02d}-{path.name}.error.txt"] = redact_text(
                f"could not read {path.name}: {error}"
            ).encode("utf-8")
            continue
        entries[f"crash_reports/{index:02d}-{path.name}"] = json.dumps(
            redact_json(data), indent=2, ensure_ascii=False
        ).encode("utf-8")
    native = crash.crash_dir(root) / "native-crash.txt"
    if native.is_file():
        try:
            entries["crash_reports/native-crash.txt"] = redact_text(
                native.read_bytes()[:MAX_LOG_BYTES].decode("utf-8", errors="replace")
            ).encode("utf-8")
        except OSError:
            pass
    if not entries:
        return {}, ["crash-reporting consent is on but no records exist"]
    return entries, []


def run_doctor_report(
    root: str | Path,
    *,
    timeout: int = DOCTOR_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run the install's doctor in its non-interactive CI mode.

    Returns ``(report, problem)``: exactly one of the two is set. The report
    is the doctor's parsed ``--json`` document; ``problem`` is a human-readable
    reason it could not be collected (missing script, failure exit, timeout).
    Read-only: the doctor never writes to the install.
    """
    root = Path(root)
    script = root / "scripts" / "doctor.py"
    if not script.is_file():
        return None, "scripts/doctor.py is not present in this install"
    try:
        completed = subprocess.run(
            [sys.executable, str(script), "--json", "--skip-model-init", "--ci"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"doctor could not run: {error}"
    if completed.returncode == 0:
        try:
            report = json.loads(completed.stdout)
        except ValueError:
            report = None
        if isinstance(report, dict):
            return report, None
    detail = (
        f"doctor exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout[-4000:]}\n"
        f"stderr:\n{completed.stderr[-4000:]}\n"
    )
    return None, detail


def _doctor_entry(
    root: Path,
    doctor_report: dict[str, Any] | None,
    run_doctor: bool,
) -> tuple[dict[str, bytes], list[str]]:
    if doctor_report is not None:
        return {"doctor.json": json.dumps(redact_json(doctor_report), indent=2).encode("utf-8")}, []
    if not run_doctor:
        return {}, ["doctor report excluded (--no-doctor)"]
    report, problem = run_doctor_report(root)
    if report is not None:
        return {"doctor.json": json.dumps(redact_json(report), indent=2).encode("utf-8")}, []
    return {"doctor_error.txt": redact_text(problem or "doctor failed").encode("utf-8")}, []


def _readme_text() -> str:
    return (
        "The Oracle support bundle\n"
        "========================\n\n"
        "This archive is created only when you ask for it. It is not uploaded\n"
        "anywhere. Review it before sharing it with support.\n\n"
        "Included: the app version and platform, whether ORACLE_* environment\n"
        "variables are present (not their values), the license state (never the\n"
        "serial number or licensee), redacted application settings, the newest redacted\n"
        "log files, and a doctor report when available. Crash reports are included\n"
        "only when you pass --include-crash-reports and local crash-reporting\n"
        "consent is on.\n\n"
        "Not included: dialogue files, manuscripts, rendered audio, voice\n"
        "recordings, reference clips, serial numbers, licensees, home-directory\n"
        "paths, or environment-variable values.\n\n"
        "Redaction removes ORACLE1 serial numbers, common API-key shapes, email\n"
        "addresses, and home-directory prefixes. It is best-effort: read the zip\n"
        "before sending it.\n"
    )


def _validated_extra_entries(extra_entries: Mapping[str, bytes] | None) -> dict[str, bytes]:
    """Extra files a caller attaches to the bundle (a bug report, for one).

    Names are archive paths: relative, forward-slash, no ``..`` segment, and
    never a name the bundle already owns (manifest.json, README.txt, and the
    collectors' outputs), so a caller cannot quietly rewrite the manifest of
    the archive it is describing. Bytes only — a str would be ambiguous about
    encoding at the archive boundary.
    """
    if not extra_entries:
        return {}
    reserved = {"manifest.json", "README.txt"}
    validated: dict[str, bytes] = {}
    for name, payload in extra_entries.items():
        if not isinstance(name, str) or not name:
            raise ValueError("support bundle attachment names must be non-empty strings")
        if not isinstance(payload, bytes):
            raise TypeError(f"support bundle attachment {name!r} must be bytes")
        parts = name.split("/")
        if "\\" in name or name.startswith("/") or ".." in parts or any(not part for part in parts):
            raise ValueError(f"support bundle attachment name is not a safe relative path: {name!r}")
        if name in reserved:
            raise ValueError(f"support bundle attachment name is reserved: {name!r}")
        validated[name] = payload
    return validated


def build_support_bundle(
    root: str | Path,
    out_path: str | Path | None = None,
    *,
    include_crash_reports: bool = False,
    doctor_report: dict[str, Any] | None = None,
    run_doctor: bool = True,
    extra_entries: Mapping[str, bytes] | None = None,
    now: datetime | None = None,
) -> SupportBundleResult:
    """Build one redacted support zip and return its manifest."""
    root = Path(root).resolve()
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if out_path is None:
        output = root / "Output" / f"oracle-support-{__version__}-{stamp}.zip"
    else:
        output = Path(out_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    included: list[str] = ["manifest.json", "README.txt", "environment.json", "license.json"]
    skipped: list[str] = []
    entries: dict[str, bytes] = {
        "README.txt": _readme_text().encode("utf-8"),
        "environment.json": json.dumps(_environment_payload(root), indent=2).encode("utf-8"),
        "license.json": json.dumps(_license_payload(root), indent=2).encode("utf-8"),
    }

    settings, settings_note = _settings_payload(root)
    if settings is not None:
        entries["settings.json"] = json.dumps(settings, indent=2, ensure_ascii=False).encode("utf-8")
        included.append("settings.json")
    if settings_note:
        skipped.append(settings_note)

    log_entries, log_notes = _log_entries(root)
    entries.update(log_entries)
    included.extend(sorted(log_entries))
    skipped.extend(log_notes)

    crash_entries, crash_notes = _crash_entries(root, include_crash_reports)
    entries.update(crash_entries)
    included.extend(sorted(crash_entries))
    skipped.extend(crash_notes)

    doctor_entries, doctor_notes = _doctor_entry(root, doctor_report, run_doctor)
    entries.update(doctor_entries)
    included.extend(sorted(doctor_entries))
    skipped.extend(doctor_notes)

    attachments = _validated_extra_entries(extra_entries)
    collisions = sorted(set(attachments) & set(entries))
    if collisions:
        raise ValueError(
            "support bundle attachment name collides with a collected entry: "
            + ", ".join(collisions)
        )
    entries.update(attachments)
    included.extend(sorted(attachments))

    manifest: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": moment.astimezone(timezone.utc).isoformat(),
        "app_version": __version__,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "included": sorted(included),
        "attachments": sorted(attachments),
        "skipped": sorted(skipped),
        "redaction": (
            "Serial numbers, API-key shapes, email addresses, and home-directory "
            "prefixes are removed. Environment-variable values and licensee data "
            "are never included."
        ),
    }
    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    entries["manifest.json"] = manifest_bytes

    handle, temporary = tempfile.mkstemp(prefix=".oracle-support-", suffix=".zip", dir=str(output.parent))
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ["manifest.json", "README.txt", *sorted(name for name in entries if name not in {"manifest.json", "README.txt"})]:
                archive.writestr(name, entries[name])
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    return SupportBundleResult(
        path=output,
        manifest=manifest,
        included=tuple(sorted(included)),
        skipped=tuple(sorted(skipped)),
    )
