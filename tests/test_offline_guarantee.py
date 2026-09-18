"""The offline guarantee, per model-loading entry point.

An offline install writes ``.oracle_offline`` in the repo root and seeds the
pinned models into the local Hugging Face cache. Every process that resolves a
model must then resolve it locally: no network attempt, and a fast, clear
failure when the cache cannot satisfy the request.

The mechanism is import-order sensitive — ``huggingface_hub`` reads
``HF_HUB_OFFLINE`` once, at import — so these tests cover each entry point the
way it actually starts: the console script (render and GUI), the doctor, the
real-engine smoke, and the installer's run action. Resolution itself is checked
against a cache in the installer's seeded layout with outbound connections
forbidden, so "resolved from the cache" cannot quietly mean "fetched".

No test downloads anything or touches the network.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script_module():
    return _load_script_module


@pytest.fixture()
def offline_module():
    from the_oracle import offline

    return offline


@pytest.fixture()
def offline_install(tmp_path, monkeypatch, offline_module):
    """An install root carrying the offline marker, with the package pointed at it.

    Restores ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` and the imported
    ``huggingface_hub`` constant afterwards, since the code under test writes all
    three by design.
    """
    root = tmp_path / "offline install"
    root.mkdir()
    (root / offline_module.OFFLINE_MARKER_FILENAME).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(offline_module, "repo_root", lambda: root)

    saved_env = {name: os.environ.get(name) for name in offline_module.OFFLINE_ENV}
    for name in offline_module.OFFLINE_ENV:
        monkeypatch.delenv(name, raising=False)
    # Force the "imported while online" state and let monkeypatch undo whatever
    # the entry point does to it. Read through sys.modules: the package-level
    # attribute is lazy and raises for parts not yet imported.
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", False)

    yield root

    for name, value in saved_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture()
def online_install(tmp_path, monkeypatch, offline_module):
    """The same install root without the marker (the control case)."""
    root = tmp_path / "online install"
    root.mkdir()
    monkeypatch.setattr(offline_module, "repo_root", lambda: root)
    for name in offline_module.OFFLINE_ENV:
        monkeypatch.delenv(name, raising=False)
    yield root


@pytest.fixture()
def no_network(monkeypatch):
    """Forbid outbound connections, so a network attempt fails loudly and fast."""
    attempts: list = []

    def blocked(self, address):  # noqa: ANN001
        attempts.append(address)
        raise AssertionError(f"outbound connection attempted to {address}")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    return attempts


@pytest.fixture()
def seeded_cache(tmp_path, monkeypatch):
    """A hub cache in the layout the installer seeds: snapshots/<sha> + refs/main."""
    from the_oracle.models.pins import (
        CHATTERBOX_REPO,
        GO_EMOTIONS_REPO,
        TURBO_REPO_ID,
        pin_for,
    )

    cache = tmp_path / "hub"
    for repo_id, filenames in (
        (CHATTERBOX_REPO, ("ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt")),
        (GO_EMOTIONS_REPO, ("config.json", "model.safetensors")),
        (TURBO_REPO_ID, ("t3_turbo.safetensors", "config.json")),
    ):
        sha = pin_for(repo_id)
        model_dir = cache / ("models--" + repo_id.replace("/", "--"))
        snapshot = model_dir / "snapshots" / sha
        snapshot.mkdir(parents=True)
        for name in filenames:
            (snapshot / name).write_bytes(b"fake")
        (model_dir / "refs").mkdir()
        (model_dir / "refs" / "main").write_text(sha, encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    return cache


# --- the shared owner ---------------------------------------------------------


def test_apply_is_scoped_to_the_marker(tmp_path, monkeypatch, offline_module) -> None:
    markerless = tmp_path / "no marker"
    markerless.mkdir()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # inherited from a launcher

    assert offline_module.apply_offline_environment(markerless) is False
    # An inherited value is left alone: child processes must not undo the launcher.
    assert os.environ["HF_HUB_OFFLINE"] == "1"

    with_marker = tmp_path / "with marker"
    with_marker.mkdir()
    (with_marker / offline_module.OFFLINE_MARKER_FILENAME).write_text("", encoding="utf-8")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    assert offline_module.apply_offline_environment(with_marker) is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert offline_module.apply_offline_environment(with_marker) is True  # idempotent


def test_apply_corrects_an_already_imported_huggingface_hub(offline_install, offline_module) -> None:
    """The library reads HF_HUB_OFFLINE at import, so a late env is not enough."""
    import huggingface_hub.constants as hf_constants

    sys.modules["huggingface_hub.constants"].HF_HUB_OFFLINE = False  # as if imported while online

    assert offline_module.apply_offline_environment() is True
    # Without this, an unseeded model takes 23s of retries instead of failing
    # locally, because the library captured the variable during its own import.
    assert hf_constants.HF_HUB_OFFLINE is True


# --- one test per model-loading entry point -----------------------------------


def test_cli_render_entry_point_enables_offline_mode(offline_install, monkeypatch, tmp_path) -> None:
    """`the-oracle render` must have offline mode on by the time it renders."""
    from the_oracle import cli

    seen: list = []

    class _RecordingPipeline:
        def __init__(self) -> None:
            self.output = tmp_path / "out.flac"

        def prepare_plan(self, input_path, output_dir, speaker_settings, settings):
            seen.append(os.environ.get("HF_HUB_OFFLINE"))

            class _Plan:
                output_dir = str(tmp_path)

            return _Plan()

        def render(self, plan, settings):
            seen.append(os.environ.get("HF_HUB_OFFLINE"))
            return self.output

    monkeypatch.setattr("the_oracle.cli.OraclePipeline", lambda: _RecordingPipeline())

    status = cli.main(
        [
            "render",
            "--input", "in.txt",
            "--outdir", str(tmp_path),
            "--speakerA-ref", "a.wav",
            "--speakerB-ref", "b.wav",
        ]
    )

    assert status == 0
    assert seen == ["1", "1"], f"pipeline saw {seen}"
    # The CLI imports the engine (and so huggingface_hub) before main() runs, so
    # the import-time constant has to be corrected too, not just the env.
    assert sys.modules["huggingface_hub.constants"].HF_HUB_OFFLINE is True


def test_cli_gui_entry_point_enables_offline_mode(offline_install, monkeypatch) -> None:
    """`the-oracle gui` (and the desktop entries that exec it) likewise."""
    from the_oracle import app_gui, cli

    seen: list = []
    monkeypatch.setattr(app_gui, "launch_gui", lambda: seen.append(os.environ.get("HF_HUB_OFFLINE")))

    assert cli.main(["gui"]) == 0
    assert seen == ["1"], f"launch_gui saw {seen}"


def test_doctor_entry_point_probes_inherit_offline_mode(offline_install, monkeypatch, tmp_path, script_module) -> None:
    """The doctor constructs the Chatterbox model, so its probes must be offline."""
    doctor = script_module("oracle_doctor_offline", "doctor.py")

    seen: list = []

    def fake_run(repo_root, **kwargs):
        seen.append(doctor._probe_environment(Path(repo_root), None).get("HF_HUB_OFFLINE"))
        return {"ok": True, "overall_ready": True}

    monkeypatch.setattr(doctor, "run", fake_run)

    assert doctor.main(["--repo-root", str(tmp_path), "--skip-model-init", "--ci", "--json"]) == 0
    assert seen == ["1"], f"probe environment carried {seen}"
    assert sys.modules["huggingface_hub.constants"].HF_HUB_OFFLINE is True


def test_real_engine_smoke_entry_point_enables_offline_mode(
    offline_install, monkeypatch, script_module
) -> None:
    """`scripts/real_engine_smoke.py` renders with the real model."""
    import huggingface_hub

    from the_oracle import real_engine_smoke

    seen: list = []

    def fake_smoke(*_args, **_kwargs):
        # This module imports the engine (and so huggingface_hub) at import time,
        # which is exactly why the constant has to be forced too.
        seen.append((os.environ.get("HF_HUB_OFFLINE"), huggingface_hub.constants.HF_HUB_OFFLINE))
        return _FakeSmokeResult()

    monkeypatch.setattr(real_engine_smoke, "run_real_engine_smoke", fake_smoke)

    assert real_engine_smoke.main(["--json"]) == 0
    assert seen == [("1", True)], f"smoke saw {seen}"


def test_manage_install_run_path_uses_the_shared_owner(offline_install, monkeypatch, script_module, offline_module) -> None:
    """The installer's run action resolves the marker through the one owner."""
    manage = script_module("oracle_manage_install_offline_guarantee", "manage_install.py")
    entrypoint = manage.venv_entrypoint_path(offline_install, "the-oracle")
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(manage, "REPO_ROOT", offline_install)
    monkeypatch.setenv("DISPLAY", ":0")

    captured: dict = {}

    def fake_run(args, **kwargs):
        captured["args"] = list(args)
        captured["env"] = dict(kwargs.get("env") or {})
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(manage.subprocess, "run", fake_run)

    assert manage.run_gui() == 0
    assert captured["env"].get("HF_HUB_OFFLINE") == "1"
    # One definition of the marker name for the launchers, the installer, and
    # the runtime check.
    assert manage.OFFLINE_MARKER_FILENAME == offline_module.OFFLINE_MARKER_FILENAME
    for name, value in offline_module.OFFLINE_ENV.items():
        assert captured["env"].get(name) == value


def test_entry_points_leave_offline_mode_off_without_the_marker(
    online_install, monkeypatch, tmp_path, script_module
) -> None:
    """The control: no marker, no forcing."""
    from the_oracle import cli, real_engine_smoke

    from the_oracle import app_gui

    launched: list = []
    monkeypatch.setattr(app_gui, "launch_gui", lambda: launched.append(True))
    monkeypatch.setattr(real_engine_smoke, "run_real_engine_smoke", lambda **_kwargs: _FakeSmokeResult())
    doctor = script_module("oracle_doctor_no_marker", "doctor.py")

    class _RecordingDoctor:
        def __init__(self) -> None:
            self.seen: list = []

        def run(self, repo_root, **_kwargs):
            self.seen.append(doctor._probe_environment(Path(repo_root), None).get("HF_HUB_OFFLINE"))
            return {"ok": True, "overall_ready": True}

    recorder = _RecordingDoctor()
    monkeypatch.setattr(doctor, "run", recorder.run)

    # Invoking each entry point far enough to run its startup path, without
    # letting it do model work.
    assert cli.main(["gui"]) == 0
    assert doctor.main(["--repo-root", str(tmp_path), "--skip-model-init", "--ci", "--json"]) == 0
    assert real_engine_smoke.main(["--json"]) == 0

    assert launched == [True]
    assert recorder.seen == [None]
    assert "HF_HUB_OFFLINE" not in os.environ


# --- resolution: from the seeded cache, or a fast clear failure ---------------


class _FakeSmokeResult:
    """Stands in for the smoke result the entry point prints when only --json is used."""

    runtime_seconds = 0.0
    model_variant = "standard"
    device = "cpu"
    output_path = ""
    report_path = ""

    def to_dict(self):
        return {"ok": True}


#: Resolves every model loader and reports where each landed. Run in a fresh
#: process because both libraries capture their environment at import time, so a
#: test process that already imported them cannot represent a real launch.
_SEEDED_CACHE_PROBE = r'''
import json, os, socket, sys

attempts = []

def guard(self, address):
    attempts.append(str(address))
    raise AssertionError(f"connection attempted: {address}")

socket.socket.connect = guard

from pathlib import Path
from the_oracle import offline

applied = offline.apply_offline_environment(sys.argv[1])   # before ANY model import

from the_oracle.models.pins import GO_EMOTIONS_REPO, pin_for
from the_oracle.tts_engines.chatterbox_engine import download_chatterbox_repo, download_turbo_checkpoint
from transformers.utils import cached_file

payload = {
    "applied": applied,
    "hf_offline": os.environ.get("HF_HUB_OFFLINE"),
    "chatterbox": str(download_chatterbox_repo()),
    "turbo": str(download_turbo_checkpoint()),
    "go_emotions": str(cached_file(GO_EMOTIONS_REPO, "config.json", revision=pin_for(GO_EMOTIONS_REPO))),
    "connections": attempts,
}
print("<<<PROBE>>>" + json.dumps(payload))
'''


def test_seeded_cache_resolves_every_loader_without_network(
    offline_install, seeded_cache, tmp_path
) -> None:
    """With the marker present, every loader resolves inside the seeded cache."""
    from the_oracle.models.pins import CHATTERBOX_REPO, GO_EMOTIONS_REPO, TURBO_REPO_ID

    probe = tmp_path / "seeded_probe.py"
    probe.write_text(_SEEDED_CACHE_PROBE, encoding="utf-8")
    env = {
        **os.environ,
        "HF_HUB_CACHE": str(seeded_cache),
        "HF_ENDPOINT": "http://127.0.0.1:9",  # closed port: any traffic fails
        # Import the code under test from this tree, not from whatever editable
        # install the interpreter happens to carry.
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)

    completed = subprocess.run(
        [sys.executable, str(probe), str(offline_install)],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(next(l for l in completed.stdout.splitlines() if "<<<PROBE>>>" in l).split("<<<PROBE>>>")[1])

    assert payload["applied"] is True
    assert payload["hf_offline"] == "1"
    cache = seeded_cache.resolve()
    for label in ("chatterbox", "turbo", "go_emotions"):
        resolved = Path(payload[label]).resolve()
        assert resolved.is_relative_to(cache), f"{label} resolved outside the seeded cache: {resolved}"
    assert payload["connections"] == [], f"loaders attempted connections: {payload['connections']}"
    # Every pinned repo the installer seeds is reachable offline (the cache also
    # gains huggingface_hub's own version marker).
    seeded = {p.name for p in cache.iterdir()}
    assert {
        "models--" + repo.replace("/", "--") for repo in (CHATTERBOX_REPO, GO_EMOTIONS_REPO, TURBO_REPO_ID)
    } <= seeded, seeded


def test_unseeded_model_fails_fast_instead_of_reaching_out(
    offline_install, no_network, offline_module
) -> None:
    """A model the cache cannot satisfy must fail locally, not retry the network.

    Measured on this machine for an unsatisfied resolution: offline mode 0.0s,
    network resolution 23.0s across five retries with an 8s backoff. The bound
    below is deliberately loose; the socket guard is the real assertion.
    """
    from huggingface_hub import hf_hub_download

    offline_module.apply_offline_environment()

    started = time.perf_counter()
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the type is asserted
        hf_hub_download(repo_id="oracle-tests/model-that-is-not-seeded", filename="weights.safetensors")
    elapsed = time.perf_counter() - started

    assert "NotFound" in type(excinfo.value).__name__, type(excinfo.value).__name__
    assert elapsed < 10.0, f"took {elapsed:.1f}s: that is the retry storm, not a local miss"
    assert not no_network, f"resolution attempted a connection: {no_network}"
