"""Regression tests for the review-fixes pass on installers, scripts, and model
downloading (scripts/, entry points, doctor/bootstrap/install tooling).

Covers:
  * scripts/download_models.py: download failures now exit non-zero and name
    the failed model; Hugging Face per-model revision pins exist and are
    forwarded as ``revision=`` to the download call.
  * scripts/manage_install.py: generated Unix wrappers shell-quote paths with
    shlex.quote instead of bare double quotes.
  * oracle (bash): the ``gui`` action is wired to the backend parser's real
    GUI action (``run``) instead of being passed through dead.
  * scripts/doctor.py: the deterministic smoke status no longer reports
    success unless the smoke output file exists and is non-empty; the
    real-engine readiness check (which only evaluates prerequisites and never
    runs the render) no longer presents a smoke output path as though it had
    produced one on a fresh machine; and the whole gate is read-only and
    idempotent — it used to generate the smoke's reference clips, which the
    voice-source count then picked up, so run 2 disagreed with run 1.
  * scripts/manage_install.py: install() skips the doctor inside bootstrap and
    runs it once, after the launchers are registered — each doctor run builds
    the Chatterbox model, so a fresh install used to pay for two identical
    model loads. That single run stays a full (non-CI) run, so a broken
    ``the-oracle`` entrypoint still fails the install.
  * oracle.ps1: static checks that Invoke-Expression is gone and extra
    arguments are forwarded (PowerShell cannot execute in this Linux
    runtime, so runtime behavior of the .ps1 still needs verification on
    Windows).

No test downloads models, touches the network, or spends anything.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def download_models(monkeypatch):
    """Load scripts/download_models.py with the heavy chatterbox engine stubbed.

    The real ``the_oracle.tts_engines.chatterbox_engine`` imports
    huggingface_hub/torch, which are unavailable (and unwanted) in this
    runtime; stubbing keeps these tests dependency-light.
    """
    tts_pkg = types.ModuleType("the_oracle.tts_engines")
    tts_pkg.__path__ = []  # type: ignore[attr-defined]
    engine_mod = types.ModuleType("the_oracle.tts_engines.chatterbox_engine")

    class FakeChatterboxEngine:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_model_ready(self):
            pass

    engine_mod.ChatterboxEngine = FakeChatterboxEngine
    monkeypatch.setitem(sys.modules, "the_oracle.tts_engines", tts_pkg)
    monkeypatch.setitem(sys.modules, "the_oracle.tts_engines.chatterbox_engine", engine_mod)
    return _load_script_module("oracle_download_models", "download_models.py")


@pytest.fixture()
def manage_install():
    return _load_script_module("oracle_manage_install_scripts", "manage_install.py")


@pytest.fixture()
def doctor_module():
    return _load_script_module("oracle_doctor_scripts", "doctor.py")


# --- download_models.py: failure propagation ---------------------------------


def test_download_models_hf_failure_returns_nonzero_and_names_model(
    download_models, tmp_path: Path, capsys
) -> None:
    def boom(repo_id, cache_dir, revision=None):
        raise RuntimeError("network is down")

    import unittest.mock as mock

    with mock.patch.object(download_models, "download_hf_model", side_effect=boom), mock.patch.object(
        download_models, "warm_chatterbox"
    ):
        status = download_models.main(
            ["--variant", "all", "--include-helper-models", "--cache-dir", str(tmp_path)]
        )
    assert status != 0
    err = capsys.readouterr().err
    assert "go_emotions" in err
    assert "punctuation" in err


def test_download_models_chatterbox_failure_returns_nonzero_and_names_variant(
    download_models, tmp_path: Path, capsys
) -> None:
    import unittest.mock as mock

    def boom(variant, device=None):
        raise RuntimeError("no torch here")

    with mock.patch.object(download_models, "download_hf_model"), mock.patch.object(
        download_models, "warm_chatterbox", side_effect=boom
    ):
        status = download_models.main(["--variant", "standard", "--cache-dir", str(tmp_path)])
    assert status != 0
    assert "chatterbox-standard" in capsys.readouterr().err


def test_download_models_success_returns_zero(download_models, tmp_path: Path) -> None:
    import unittest.mock as mock

    with mock.patch.object(download_models, "download_hf_model"), mock.patch.object(
        download_models, "warm_chatterbox"
    ):
        status = download_models.main(
            ["--variant", "all", "--include-helper-models", "--cache-dir", str(tmp_path)]
        )
    assert status == 0


# --- download_models.py: revision pins ----------------------------------------


def test_hf_model_revisions_cover_every_model(download_models) -> None:
    assert set(download_models.HF_MODEL_REVISIONS) == set(download_models.HF_MODELS)


def test_hf_model_revisions_are_none_or_real_shas(download_models) -> None:
    sha_re = re.compile(r"^[0-9a-f]{40}$")
    for name, revision in download_models.HF_MODEL_REVISIONS.items():
        assert revision is None or sha_re.match(revision), (
            f"{name}: revision pins must be None (unpinned) or a 40-char hex "
            f"commit SHA from huggingface.co/<repo>/commits, got {revision!r}"
        )


def test_download_hf_model_forwards_revision(download_models, tmp_path: Path, monkeypatch) -> None:
    calls: list[dict] = []

    fake_hub = types.ModuleType("huggingface_hub")

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    download_models.download_hf_model("org/some-model", tmp_path, revision="a" * 40)
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "org/some-model"
    assert calls[0]["revision"] == "a" * 40


def test_download_hf_model_default_revision_is_unpinned(download_models, tmp_path: Path, monkeypatch) -> None:
    calls: list[dict] = []
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda **kwargs: calls.append(kwargs) or str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    download_models.download_hf_model("org/some-model", tmp_path)
    assert calls[0]["revision"] is None


# --- manage_install.py: wrapper shell quoting ----------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="bash scripts require a POSIX shell")
def test_unix_wrapper_shell_quotes_nasty_paths(manage_install, monkeypatch, tmp_path: Path) -> None:
    nasty = tmp_path / 'dir with spaces and "quotes" and $dollar `backtick`'
    monkeypatch.setattr(manage_install, "REPO_ROOT", nasty)
    contents = manage_install.managed_wrapper_contents()
    # Paths must appear in shlex.quote (single-quote) form, never in bare
    # double quotes that the shell would re-expand.
    assert f"REPO_ROOT={shlex.quote(str(nasty))}" in contents
    assert f'REPO_ROOT="{nasty}"' not in contents
    assert f'VENV_ENTRYPOINT="{manage_install.venv_entrypoint_path(nasty, "the-oracle")}"' not in contents


@pytest.mark.skipif(sys.platform == "win32", reason="bash scripts require a POSIX shell")
def test_unix_wrapper_is_valid_bash(manage_install, monkeypatch, tmp_path: Path) -> None:
    nasty = tmp_path / 'weird "dir" $x'
    monkeypatch.setattr(manage_install, "REPO_ROOT", nasty)
    contents = manage_install.managed_wrapper_contents()
    probe = tmp_path / "wrapper.sh"
    probe.write_text(contents, encoding="utf-8")
    completed = subprocess.run(["bash", "-n", str(probe)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="bash scripts require a POSIX shell")
def test_unix_wrapper_still_execs_entrypoint(manage_install, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(manage_install, "REPO_ROOT", tmp_path / "plain")
    contents = manage_install.managed_wrapper_contents()
    assert 'exec "$VENV_ENTRYPOINT" "$@"' in contents
    assert "ORACLE_TTS_WRAPPER" in contents


# --- oracle (bash): gui action wiring ------------------------------------------


def _build_fake_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Copy the real bash `oracle` into a fake repo with a shim python.

    The shim exits 0 when probed with no args (the version check) and
    otherwise records its argv, so we can assert exactly what the wrapper
    would have exec'd without running any real installer code.
    """
    fakerepo = tmp_path / "fakerepo"
    (fakerepo / "scripts").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "oracle", fakerepo / "oracle")
    recorded = tmp_path / "recorded_args.txt"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = (
        "#!/usr/bin/env bash\n"
        # The version probe invokes the candidate as `candidate -` with the
        # check script on stdin: treat that as a successful probe.
        'if [ "$#" -eq 1 ] && [ "$1" = "-" ]; then exit 0; fi\n'
        'printf \'%s\\n\' "$@" >> "$ORACLE_TEST_RECORDED"\n'
    )
    for name in ("python3.12", "python3.11", "python3"):
        path = bindir / name
        path.write_text(shim, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return fakerepo, bindir, recorded


def _run_oracle(fakerepo: Path, bindir: Path, recorded: Path, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["ORACLE_TEST_RECORDED"] = str(recorded)
    return subprocess.run(
        ["bash", str(fakerepo / "oracle"), *args],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="bash scripts require a POSIX shell")
def test_bash_oracle_gui_is_wired_to_run(tmp_path: Path) -> None:
    fakerepo, bindir, recorded = _build_fake_repo(tmp_path)
    completed = _run_oracle(fakerepo, bindir, recorded, "gui", "--skip-model-init")
    assert completed.returncode == 0, completed.stderr
    argv = recorded.read_text(encoding="utf-8").splitlines()
    # gui must reach the backend parser as its real GUI action, with extras forwarded.
    assert argv == [f"{fakerepo}/scripts/manage_install.py", "run", "--skip-model-init"]


@pytest.mark.skipif(sys.platform == "win32", reason="bash scripts require a POSIX shell")
def test_bash_oracle_start_still_maps_to_run(tmp_path: Path) -> None:
    fakerepo, bindir, recorded = _build_fake_repo(tmp_path)
    completed = _run_oracle(fakerepo, bindir, recorded, "start")
    assert completed.returncode == 0, completed.stderr
    argv = recorded.read_text(encoding="utf-8").splitlines()
    assert argv == [f"{fakerepo}/scripts/manage_install.py", "run"]


@pytest.mark.skipif(sys.platform == "win32", reason="bash scripts require a POSIX shell")
def test_bash_oracle_rejects_unknown_action(tmp_path: Path) -> None:
    fakerepo, bindir, recorded = _build_fake_repo(tmp_path)
    completed = _run_oracle(fakerepo, bindir, recorded, "bogus")
    assert completed.returncode != 0
    assert not recorded.exists()


# --- doctor.py: deterministic smoke must verify output --------------------------


def _stub_smoke(monkeypatch, output_path: Path):
    # The doctor delegates the output-verdict policy to the smoke module, so
    # the fake module carries the real helper -- these tests then still
    # exercise the real policy (missing/empty fail, present passes).
    from the_oracle.smoke import smoke_output_problem

    fake_smoke = types.ModuleType("the_oracle.smoke")
    fake_smoke.run_deterministic_smoke_render = lambda output_root, source_format="txt": SimpleNamespace(
        output_path=output_path,
        project_dir=output_root / "project",
        cache_reused_on_second_pass=False,
    )
    fake_smoke.smoke_output_problem = smoke_output_problem
    monkeypatch.setitem(sys.modules, "the_oracle.smoke", fake_smoke)


def test_deterministic_smoke_fails_when_output_missing(
    doctor_module, tmp_path: Path, monkeypatch
) -> None:
    _stub_smoke(monkeypatch, tmp_path / "nope.flac")
    status = doctor_module._deterministic_smoke_status(tmp_path)
    assert status["ok"] is False
    assert "no output" in status["error"].lower()


def test_deterministic_smoke_fails_when_output_empty(
    doctor_module, tmp_path: Path, monkeypatch
) -> None:
    empty = tmp_path / "empty.flac"
    empty.write_bytes(b"")
    _stub_smoke(monkeypatch, empty)
    status = doctor_module._deterministic_smoke_status(tmp_path)
    assert status["ok"] is False
    assert "empty" in status["error"].lower()


def test_deterministic_smoke_ok_when_output_present(
    doctor_module, tmp_path: Path, monkeypatch
) -> None:
    real = tmp_path / "out.flac"
    real.write_bytes(b"RIFF....fake-audio")
    _stub_smoke(monkeypatch, real)
    status = doctor_module._deterministic_smoke_status(tmp_path)
    assert status["ok"] is True
    assert status["output_path"] == str(real)


# --- oracle.ps1: static regression checks ---------------------------------------
# PowerShell cannot execute in this Linux runtime; these assertions guard the
# source against reintroducing Invoke-Expression / dropped arguments. The
# runtime behavior still needs a check on Windows.


def test_oracle_ps1_has_no_invoke_expression() -> None:
    text = (REPO_ROOT / "oracle.ps1").read_text(encoding="utf-8")
    # No *command* usage of Invoke-Expression (mentions in comments are fine).
    assert not re.search(r"(?m)^\s*(\$null\s*=\s*)?Invoke-Expression\b", text)


def test_oracle_ps1_forwards_remaining_arguments() -> None:
    text = (REPO_ROOT / "oracle.ps1").read_text(encoding="utf-8")
    assert "ValueFromRemainingArguments" in text
    assert "RemainingArgs" in text
    # The final dispatch must use the call operator, not string evaluation.
    assert "& $pythonTokens[0]" in text


# --- doctor.py: real-engine readiness must not imply an artifact exists ----------
# This check evaluates prerequisites only; it never runs the render. Printing the
# smoke output path as though the check produced it tells a freshly installed
# user about a file that has never existed on their machine.


def _stub_real_engine(monkeypatch, output: Path, *, ready: bool = True):
    fake = types.ModuleType("the_oracle.real_engine_smoke")
    fake.ensure_real_engine_inputs = lambda output_root: {}
    fake.real_engine_smoke_prerequisites = lambda output_root: {
        "ready": ready,
        "expected_paths": {"output": str(output)},
    }
    monkeypatch.setitem(sys.modules, "the_oracle.real_engine_smoke", fake)


def test_real_engine_readiness_flags_missing_output(doctor_module, tmp_path: Path, monkeypatch) -> None:
    """A pristine tree has no build/real_engine_smoke/*.flac: readiness must be
    reported without claiming an output exists."""
    missing = tmp_path / "build" / "real_engine_smoke" / "real_engine_smoke.flac"
    _stub_real_engine(monkeypatch, missing)
    status = doctor_module._real_engine_smoke_status(tmp_path)
    assert status["ok"] is True  # prerequisites are met ...
    assert status["output_exists"] is False  # ... but nothing was produced


def test_real_engine_readiness_flags_present_output(doctor_module, tmp_path: Path, monkeypatch) -> None:
    present = tmp_path / "real_engine_smoke.flac"
    present.write_bytes(b"RIFF....fake-audio")
    _stub_real_engine(monkeypatch, present)
    status = doctor_module._real_engine_smoke_status(tmp_path)
    assert status["ok"] is True
    assert status["output_exists"] is True


def _report_with_real_engine(real_engine: dict) -> dict:
    return {
        "repo_root": "/repo",
        "platform": "linux",
        "ci_mode": True,
        "python": {"ok": True, "executable": "python3", "version": "3.12"},
        "ffmpeg": {"ok": True, "path": "ffmpeg"},
        "entrypoint": {
            "ok": True, "fresh_shell_path": "the-oracle", "path_entrypoint": "",
            "venv_entrypoint": "", "fresh_shell_error": "", "help_error": "",
        },
        "chatterbox_import": {"ok": True, "target": "x", "error": ""},
        "chatterbox_init": {"ok": True, "seconds": 1.0, "skipped": False, "error": ""},
        "perth": {"ok": True, "watermarker_symbol": "w", "error": ""},
        "turbo": {"ok": True, "checkpoint_dir": "", "error": ""},
        "voice_sources": {
            "ok": True,
            "default_voice_assessment": "ok",
            "seashell_clip_count": 2,
            "fallback_clip_count": 0,
            "better_local_assets_detail": "",
            "voice_mixing_detail": "",
        },
        "qt": {
            "ok": True, "plugin_path": "", "qt_platform": "offscreen", "error": "",
            "missing_libraries": [], "suggested_packages": [], "offscreen_error": "", "ldd_error": "",
        },
        "deterministic_smoke": {"ok": True, "output_path": "/o", "error": ""},
        "real_engine_smoke": real_engine,
        "vulkan_backend": {
            "ok": True, "binary_built": True, "model_override_set": True,
            "model_file_exists": True, "model_path": "/models/x.gguf",
            "vulkan_device": True, "device_name": "GPU A", "rdna1_device": False,
            "vendored_patch_applied": True, "device_index_env": "", "threads_env": "",
            "audio_cpp_devices": [], "caveat": "", "error": "",
        },
        "next_steps": [],
    }


def test_human_report_does_not_claim_absent_smoke_output(doctor_module, capsys) -> None:
    """The PASS line must scope itself to prerequisites when no smoke has run."""
    report = _report_with_real_engine({
        "ok": True, "ready": True, "output_exists": False, "error": "",
        "expected_paths": {"output": "/repo/build/real_engine_smoke/real_engine_smoke.flac"},
    })
    doctor_module._print_human_report(report)
    line = next(l for l in capsys.readouterr().out.splitlines() if "Real-engine smoke readiness" in l)
    assert line.startswith("PASS")
    assert "no smoke output yet" in line
    assert "present at" not in line


def test_human_report_cites_present_smoke_output(doctor_module, capsys) -> None:
    report = _report_with_real_engine({
        "ok": True, "ready": True, "output_exists": True, "error": "",
        "expected_paths": {"output": "/repo/build/real_engine_smoke/real_engine_smoke.flac"},
    })
    doctor_module._print_human_report(report)
    line = next(l for l in capsys.readouterr().out.splitlines() if "Real-engine smoke readiness" in l)
    assert "present at" in line
    assert "real_engine_smoke.flac" in line


# --- doctor.py: the launch gate must not change its own input -------------------
# The real-engine check used to generate the smoke's reference clips. Those land
# in build/real_engine_smoke/inputs, which the voice-source audit counts, so run 1
# reported fallback=0 and run 2 reported fallback=2 on an unchanged machine: the
# gate's verdict depended on how many times it had been run.


def _stub_prereqs_only(monkeypatch, output: Path, *, ready: bool = True) -> None:
    """Stub only the prerequisites probe, keeping the real
    ``ensure_real_engine_inputs`` so any write the check performs is caught."""
    real_mod = importlib.import_module("the_oracle.real_engine_smoke")
    monkeypatch.setattr(
        real_mod,
        "real_engine_smoke_prerequisites",
        lambda output_root: {"ready": ready, "expected_paths": {"output": str(output)}},
    )


def test_real_engine_readiness_does_not_write(doctor_module, tmp_path: Path, monkeypatch) -> None:
    """A check must not create state. Without this, running the gate fed the next
    run's voice-source count."""
    output_root = tmp_path / "build" / "real_engine_smoke"
    _stub_prereqs_only(monkeypatch, output_root / "real_engine_smoke.flac")
    status = doctor_module._real_engine_smoke_status(tmp_path)
    assert status["ok"] is True
    assert not output_root.exists(), f"check wrote {list(output_root.rglob('*')) if output_root.exists() else ''}"


def _stub_heavy_probes(doctor_module, monkeypatch) -> None:
    """Stub everything expensive so the two-run test exercises the
    voice-source <-> real-engine coupling without torch, Qt, or a real render."""
    monkeypatch.setattr(doctor_module, "_python_status", lambda: {"ok": True, "executable": "python3", "version": "3.12"})
    monkeypatch.setattr(doctor_module, "_ffmpeg_status", lambda: {"ok": True, "path": "ffmpeg"})
    monkeypatch.setattr(doctor_module, "_entrypoint_status", lambda repo: {
        "ok": True, "venv_entrypoint": "", "venv_entrypoint_exists": False,
        "path_entrypoint": "the-oracle", "managed_wrapper_path": "", "managed_wrapper_installed": True,
        "help_ok": True, "help_error": "", "fresh_shell_help_ok": True,
        "fresh_shell_path": "the-oracle", "fresh_shell_error": "", "path_has_local_bin": True,
    })
    monkeypatch.setattr(doctor_module, "_chatterbox_probe", lambda *a, **k: {
        "import_ok": True, "perth_ok": True, "watermarker_callable": True, "init_skipped": True,
    })
    monkeypatch.setattr(doctor_module, "_turbo_status", lambda *a, **k: {
        "ok": True, "cached": True, "checkpoint_dir": "", "sample_rate": 24000, "error": "",
    })
    monkeypatch.setattr(doctor_module, "_cuda_backend_status", lambda repo: {
        "ok": False, "runtime_available": False, "reason": "no CUDA", "devices": [],
    })
    monkeypatch.setattr(doctor_module, "_qt_status", lambda *a, **k: {
        "ok": True, "plugin_path": "", "qt_platform": "offscreen", "error": "",
        "missing_libraries": [], "suggested_packages": [], "offscreen_error": "", "ldd_error": "",
    })
    monkeypatch.setattr(doctor_module, "_deterministic_smoke_status", lambda repo: {
        "ok": True, "output_path": "/o", "error": "",
    })
    monkeypatch.setattr(doctor_module, "_vulkan_backend_status", lambda repo: {
        "ok": True, "binary_built": False, "model_override_set": False, "model_file_exists": False,
        "model_path": "", "vulkan_device": False, "device_name": "", "rdna1_device": False,
        "vendored_patch_applied": None, "device_index_env": "", "threads_env": "",
        "audio_cpp_devices": [], "caveat": "", "error": "",
    })


def test_doctor_output_is_identical_across_consecutive_runs(
    doctor_module, tmp_path: Path, monkeypatch, capsys
) -> None:
    """Two runs on an unchanged machine must print byte-identical reports."""
    repo = tmp_path / "repo"
    (repo / "Seashells" / "generic").mkdir(parents=True)
    (repo / "Seashells" / "generic" / "english_1.wav").write_bytes(b"RIFF")
    (repo / "Seashells" / "curated.wav").write_bytes(b"RIFF")
    _stub_heavy_probes(doctor_module, monkeypatch)

    rendered: list[str] = []
    for _ in range(2):
        report = doctor_module.run(repo, model_timeout=1.0, qt_timeout=1.0, skip_model_init=True, ci_mode=True)
        doctor_module._print_human_report(report)
        rendered.append(capsys.readouterr().out)

    differing = [a for a, b in zip(rendered[0].splitlines(), rendered[1].splitlines()) if a != b]
    assert rendered[0] == rendered[1], f"report changed between runs: {differing}"


# --- manage_install.py: a fresh install verifies once, and that run must count --
# install() called bootstrap() (which runs the doctor) and then ran the doctor
# again at the end. Each doctor run constructs the Chatterbox model, so a fresh
# install loaded it twice for the same answer.


def _stub_bootstrap_steps(manage_install, monkeypatch, venv_python: Path) -> None:
    """Stub every bootstrap step that would touch the system, so install()'s
    call graph is observable without creating a venv or installing anything."""
    monkeypatch.setattr(manage_install, "ensure_supported_python", lambda: None)
    monkeypatch.setattr(manage_install, "ensure_venv", lambda: venv_python)
    monkeypatch.setattr(manage_install, "install_dependencies", lambda *a, **k: None)
    monkeypatch.setattr(manage_install, "install_managed_wrapper", lambda: None)


def test_install_verifies_once_after_the_launcher(manage_install, monkeypatch, tmp_path: Path) -> None:
    """Exactly one doctor run, and it happens after the launchers are in place."""
    _stub_bootstrap_steps(manage_install, monkeypatch, tmp_path / "python")
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(manage_install, "install_desktop_launcher", lambda: events.append(("launcher", {})))
    monkeypatch.setattr(
        manage_install, "run_doctor", lambda *a, **k: (events.append(("doctor", k)), 0)[1]
    )

    assert manage_install.install() == 0

    names = [name for name, _ in events]
    assert names == ["launcher", "doctor"], f"install step order was {names}"
    # CI mode exempts the entrypoint from the required checks, so the single
    # verification must stay a full run.
    assert not events[1][1].get("ci_mode"), "verification must not run in CI mode"


def test_install_fails_when_the_entrypoint_is_broken(
    manage_install, doctor_module, monkeypatch, tmp_path: Path, capsys
) -> None:
    """Bootstrap's doctor is skipped, so install()'s single run is the only thing
    between a broken ``the-oracle`` entrypoint and a report of success."""
    repo = tmp_path / "repo"
    (repo / "Seashells" / "generic").mkdir(parents=True)
    (repo / "Seashells" / "generic" / "english_1.wav").write_bytes(b"RIFF")
    _stub_heavy_probes(doctor_module, monkeypatch)
    _stub_real_engine(monkeypatch, repo / "build" / "real_engine_smoke" / "real_engine_smoke.flac")
    monkeypatch.setattr(doctor_module, "_entrypoint_status", lambda _repo: {
        "ok": False,
        "venv_entrypoint": str(repo / ".venv" / "bin" / "the-oracle"),
        "venv_entrypoint_exists": False,
        "path_entrypoint": "",
        "managed_wrapper_path": "",
        "managed_wrapper_installed": False,
        "help_ok": False,
        "help_error": "the-oracle: command not found",
        "fresh_shell_help_ok": False,
        "fresh_shell_path": "",
        "fresh_shell_error": "the-oracle is not available in a fresh shell PATH",
        "path_has_local_bin": False,
    })

    # The doctor's own verdict for a broken entrypoint is "not ready".
    verdict = doctor_module.main(["--repo-root", str(repo), "--skip-model-init"])
    assert verdict == 1
    capsys.readouterr()

    _stub_bootstrap_steps(manage_install, monkeypatch, tmp_path / "python")
    monkeypatch.setattr(manage_install, "install_desktop_launcher", lambda: None)
    monkeypatch.setattr(manage_install, "run_doctor", lambda *a, **k: verdict)

    # Install must delegate verification to its own final run, so that run is
    # the one that reports the broken entrypoint.
    real_bootstrap = manage_install.bootstrap
    seen: dict = {}

    def _spy_bootstrap(*a, **k):
        seen.update(k)
        return real_bootstrap(*a, **k)

    monkeypatch.setattr(manage_install, "bootstrap", _spy_bootstrap)

    assert manage_install.install() == verdict
    assert seen.get("skip_doctor") is True, "install must skip the doctor inside bootstrap"
    assert "Install complete." not in capsys.readouterr().out
