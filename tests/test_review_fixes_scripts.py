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
    success unless the smoke output file exists and is non-empty.
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


def test_unix_wrapper_shell_quotes_nasty_paths(manage_install, monkeypatch, tmp_path: Path) -> None:
    nasty = tmp_path / 'dir with spaces and "quotes" and $dollar `backtick`'
    monkeypatch.setattr(manage_install, "REPO_ROOT", nasty)
    contents = manage_install.managed_wrapper_contents()
    # Paths must appear in shlex.quote (single-quote) form, never in bare
    # double quotes that the shell would re-expand.
    assert f"REPO_ROOT={shlex.quote(str(nasty))}" in contents
    assert f'REPO_ROOT="{nasty}"' not in contents
    assert f'VENV_ENTRYPOINT="{manage_install.venv_entrypoint_path(nasty, "the-oracle")}"' not in contents


def test_unix_wrapper_is_valid_bash(manage_install, monkeypatch, tmp_path: Path) -> None:
    nasty = tmp_path / 'weird "dir" $x'
    monkeypatch.setattr(manage_install, "REPO_ROOT", nasty)
    contents = manage_install.managed_wrapper_contents()
    probe = tmp_path / "wrapper.sh"
    probe.write_text(contents, encoding="utf-8")
    completed = subprocess.run(["bash", "-n", str(probe)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


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


def test_bash_oracle_gui_is_wired_to_run(tmp_path: Path) -> None:
    fakerepo, bindir, recorded = _build_fake_repo(tmp_path)
    completed = _run_oracle(fakerepo, bindir, recorded, "gui", "--skip-model-init")
    assert completed.returncode == 0, completed.stderr
    argv = recorded.read_text(encoding="utf-8").splitlines()
    # gui must reach the backend parser as its real GUI action, with extras forwarded.
    assert argv == [f"{fakerepo}/scripts/manage_install.py", "run", "--skip-model-init"]


def test_bash_oracle_start_still_maps_to_run(tmp_path: Path) -> None:
    fakerepo, bindir, recorded = _build_fake_repo(tmp_path)
    completed = _run_oracle(fakerepo, bindir, recorded, "start")
    assert completed.returncode == 0, completed.stderr
    argv = recorded.read_text(encoding="utf-8").splitlines()
    assert argv == [f"{fakerepo}/scripts/manage_install.py", "run"]


def test_bash_oracle_rejects_unknown_action(tmp_path: Path) -> None:
    fakerepo, bindir, recorded = _build_fake_repo(tmp_path)
    completed = _run_oracle(fakerepo, bindir, recorded, "bogus")
    assert completed.returncode != 0
    assert not recorded.exists()


# --- doctor.py: deterministic smoke must verify output --------------------------


def _stub_smoke(monkeypatch, output_path: Path):
    fake_smoke = types.ModuleType("the_oracle.smoke")
    fake_smoke.run_deterministic_smoke_render = lambda output_root, source_format="txt": SimpleNamespace(
        output_path=output_path,
        project_dir=output_root / "project",
        cache_reused_on_second_pass=False,
    )
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
