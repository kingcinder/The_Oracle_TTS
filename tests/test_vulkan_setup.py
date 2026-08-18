"""Hermetic tests for the automatic CPU→GPU (Vulkan) backend setup.

The orchestrator (the_oracle.vulkan_setup) completes the switch by itself:
it builds audiocpp_cli when missing, downloads the Chatterbox model when
missing, and sets ORACLE_AUDIOCPP_CLI/ORACLE_AUDIOCPP_MODEL for the session.
These tests exercise that with fake build/download scripts in a temp repo so
nothing touches the real audio.cpp checkout, the network, or the GPU.

Also covers the CLI wiring: ``handle_render`` with ``--inference-backend
vulkan`` auto-runs the setup (unless ``--no-audio-cpp-setup``), and the
``setup-vulkan`` subcommand performs an explicit one-shot setup.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from the_oracle import vulkan_setup


def _fake_repo(tmp_path: Path, binary: Path, model: Path) -> Path:
    """A hermetic repo root with fake setup scripts.

    ``build_audio_cpp.sh`` creates the ``audiocpp_cli`` file (unless
    FAKE_BUILD_FAIL=1 is set), and ``download_audio_cpp_model.sh`` creates the
    model file and prints the ORACLE_AUDIOCPP_MODEL export line (unless
    FAKE_DOWNLOAD_FAIL=1 is set). ``model_path`` is baked in so the parse step
    has something to resolve.
    """
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "build_audio_cpp.sh").write_text(
        "#!/usr/bin/env bash\n"
        'echo "fake build: compiling audiocpp_cli"\n'
        'if [[ "${FAKE_BUILD_FAIL:-0}" == "1" ]]; then\n'
        "  echo 'compiler exploded' >&2\n"
        "  exit 1\n"
        "fi\n"
        f'mkdir -p "$(dirname {binary})"\n'
        f'touch {binary}\n',
        encoding="utf-8",
    )
    (scripts / "download_audio_cpp_model.sh").write_text(
        "#!/usr/bin/env bash\n"
        'echo "fake download: fetching chatterbox q8_0"\n'
        'if [[ "${FAKE_DOWNLOAD_FAIL:-0}" == "1" ]]; then\n'
        "  echo 'network timeout' >&2\n"
        "  exit 1\n"
        "fi\n"
        f'mkdir -p "$(dirname {model})"\n'
        f'touch {model}\n'
        f'echo \'export ORACLE_AUDIOCPP_MODEL="{model}"\'\n',
        encoding="utf-8",
    )
    return root


def test_parse_model_export_extracts_path() -> None:
    sample = (
        "Chatterbox model installed: /tmp/x.gguf\n\n"
        '    export ORACLE_AUDIOCPP_MODEL="/tmp/x.gguf"\n'
    )
    assert vulkan_setup.parse_model_export(sample) == "/tmp/x.gguf"
    assert vulkan_setup.parse_model_export("no export line here") == ""


def test_vulkan_setup_needed_when_configured(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("x")
    model = tmp_path / "model.gguf"
    model.write_text("m")
    monkeypatch.setattr(vulkan_setup, "find_audiocpp_binary", lambda: binary)
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model))
    assert vulkan_setup.vulkan_setup_needed() == []


def test_vulkan_setup_needed_reports_missing_pieces(monkeypatch) -> None:
    monkeypatch.setattr(vulkan_setup, "find_audiocpp_binary", lambda: None)
    monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
    missing = vulkan_setup.vulkan_setup_needed()
    assert any("audiocpp_cli is not built" in item for item in missing)
    assert any("ORACLE_AUDIOCPP_MODEL" in item for item in missing)


def test_run_vulkan_setup_already_configured_no_scripts(
    monkeypatch, tmp_path: Path
) -> None:
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("x")
    model = tmp_path / "model.gguf"
    model.write_text("m")
    monkeypatch.setattr(vulkan_setup, "find_audiocpp_binary", lambda: binary)
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model))
    root = _fake_repo(tmp_path, binary, model)

    result = vulkan_setup.run_vulkan_setup(repo_root=root)

    assert result.ok
    assert any("already configured" in msg for msg in result.messages)


def _restore_env_after_setup(saved: dict[str, str | None]) -> None:
    """Restore env vars that run_vulkan_setup mutated directly.

    ``monkeypatch.delenv(name, raising=False)`` records nothing when the var
    is absent, so a test that lets the real ``run_vulkan_setup`` write
    ``os.environ`` would leak the session model/CLI paths into later tests
    (e.g. vulkan_backend's missing-model check). Capture the pre-test values
    and restore them explicitly instead.
    """
    import os

    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_run_vulkan_setup_builds_and_downloads(monkeypatch, tmp_path: Path) -> None:
    import os

    binary = tmp_path / "bin" / "audiocpp_cli"
    model = tmp_path / "models" / "chatterbox-q8_0.gguf"
    root = _fake_repo(tmp_path, binary, model)
    saved = {
        "ORACLE_AUDIOCPP_MODEL": os.environ.get("ORACLE_AUDIOCPP_MODEL"),
        "ORACLE_AUDIOCPP_CLI": os.environ.get("ORACLE_AUDIOCPP_CLI"),
    }
    monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
    monkeypatch.delenv("ORACLE_AUDIOCPP_CLI", raising=False)
    progress: list[str] = []
    # The probe sees the binary only after the fake build creates it.
    def probe() -> Path | None:
        return binary if binary.exists() else None

    monkeypatch.setattr(vulkan_setup, "find_audiocpp_binary", probe)

    try:
        result = vulkan_setup.run_vulkan_setup(progress=progress.append, repo_root=root)

        assert result.ok
        assert result.binary == str(binary)
        assert result.model == str(model)

        # run_vulkan_setup mutates os.environ directly; read the live value.
        assert os.environ.get("ORACLE_AUDIOCPP_CLI") == str(binary)
        assert os.environ.get("ORACLE_AUDIOCPP_MODEL") == str(model)
        assert any("Building audiocpp_cli" in line for line in progress)
        assert any("Model ready" in line for line in progress)
    finally:
        _restore_env_after_setup(saved)


def test_run_vulkan_setup_skips_download_when_model_present(
    monkeypatch, tmp_path: Path
) -> None:
    binary = tmp_path / "bin" / "audiocpp_cli"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("x")
    model = tmp_path / "models" / "existing.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("m")
    root = _fake_repo(tmp_path, binary, model)
    monkeypatch.setattr(vulkan_setup, "find_audiocpp_binary", lambda: binary)
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model))
    progress: list[str] = []

    result = vulkan_setup.run_vulkan_setup(progress=progress.append, repo_root=root)

    assert result.ok
    # No build (binary exists) and no download (model exists).
    assert not any("Building audiocpp_cli" in line for line in progress)
    assert not any("Downloading the Chatterbox model" in line for line in progress)


def test_run_vulkan_setup_build_failure(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "bin" / "audiocpp_cli"
    model = tmp_path / "models" / "chatterbox-q8_0.gguf"
    root = _fake_repo(tmp_path, binary, model)
    monkeypatch.setattr(vulkan_setup, "find_audiocpp_binary", lambda: None)
    monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
    monkeypatch.setenv("FAKE_BUILD_FAIL", "1")

    result = vulkan_setup.run_vulkan_setup(repo_root=root)

    assert not result.ok
    assert "build failed" in result.error
    assert "compiler exploded" in result.error


def test_run_vulkan_setup_download_failure(monkeypatch, tmp_path: Path) -> None:
    import os

    binary = tmp_path / "bin" / "audiocpp_cli"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("x")
    model = tmp_path / "models" / "chatterbox-q8_0.gguf"
    root = _fake_repo(tmp_path, binary, model)
    monkeypatch.setattr(vulkan_setup, "find_audiocpp_binary", lambda: binary)
    monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
    monkeypatch.setenv("FAKE_DOWNLOAD_FAIL", "1")
    # The build step runs before the download fails, so run_vulkan_setup sets
    # ORACLE_AUDIOCPP_CLI directly; restore it so later tests are unaffected.
    saved_cli = os.environ.get("ORACLE_AUDIOCPP_CLI")

    try:
        result = vulkan_setup.run_vulkan_setup(repo_root=root)

        assert not result.ok
        assert "download failed" in result.error
    finally:
        _restore_env_after_setup({"ORACLE_AUDIOCPP_CLI": saved_cli})


def test_run_vulkan_setup_respects_cancel_before_scripts(
    monkeypatch, tmp_path: Path
) -> None:
    binary = tmp_path / "bin" / "audiocpp_cli"
    model = tmp_path / "models" / "chatterbox-q8_0.gguf"
    root = _fake_repo(tmp_path, binary, model)
    monkeypatch.setattr(vulkan_setup, "find_audiocpp_binary", lambda: None)
    monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
    cancel = threading.Event()
    cancel.set()

    result = vulkan_setup.run_vulkan_setup(cancel=cancel, repo_root=root)

    assert not result.ok
    assert result.cancelled is True
    assert not binary.exists()  # the build script never ran


def test_handle_render_auto_runs_vulkan_setup(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from the_oracle.cli import build_parser, handle_render

    class _FakePipeline:
        def __init__(self) -> None:
            self.output = tmp_path / "out.flac"

        def prepare_plan(self, *args, **kwargs):
            class _Plan:
                output_dir = str(tmp_path)

            return _Plan()

        def render(self, plan, settings):
            return self.output

    monkeypatch.setattr("the_oracle.cli.OraclePipeline", lambda: _FakePipeline())
    setup_calls: list[str] = []

    def ok_result(progress=None, cancel=None, **kwargs):
        if progress:
            progress("stub ok")
        setup_calls.append("ran")
        return type("R", (), {"ok": True, "messages": ["stub setup done"], "error": ""})()

    monkeypatch.setattr("the_oracle.vulkan_setup.run_vulkan_setup", ok_result)

    args = build_parser().parse_args(
        [
            "render",
            "--input",
            "in.txt",
            "--outdir",
            str(tmp_path / "out"),
            "--speakerA-ref",
            "a.wav",
            "--speakerB-ref",
            "b.wav",
            "--inference-backend",
            "vulkan",
        ]
    )
    assert handle_render(args) == 0
    assert setup_calls == ["ran"], "auto-setup should run before a Vulkan render"
    assert "stub setup done" in capsys.readouterr().err


def test_handle_render_no_audio_cpp_setup_skips_auto_setup(
    monkeypatch, tmp_path: Path
) -> None:
    from the_oracle.cli import build_parser, handle_render

    class _FakePipeline:
        def __init__(self) -> None:
            self.output = tmp_path / "out.flac"

        def prepare_plan(self, *args, **kwargs):
            class _Plan:
                output_dir = str(tmp_path)

            return _Plan()

        def render(self, plan, settings):
            return self.output

    monkeypatch.setattr("the_oracle.cli.OraclePipeline", lambda: _FakePipeline())
    called: list[str] = []

    def fail_if_called(**kwargs):
        called.append("boom")
        raise AssertionError("auto-setup must not run with --no-audio-cpp-setup")

    monkeypatch.setattr("the_oracle.vulkan_setup.run_vulkan_setup", fail_if_called)

    args = build_parser().parse_args(
        [
            "render",
            "--input",
            "in.txt",
            "--outdir",
            str(tmp_path / "out"),
            "--speakerA-ref",
            "a.wav",
            "--speakerB-ref",
            "b.wav",
            "--inference-backend",
            "vulkan",
            "--no-audio-cpp-setup",
        ]
    )
    assert handle_render(args) == 0
    assert not called


def test_setup_vulkan_subcommand_runs_setup(monkeypatch, tmp_path: Path, capsys) -> None:
    """``the-oracle setup-vulkan`` performs the one-shot auto-setup and prints
    the export lines, exiting 0 on success."""
    from the_oracle.cli import build_parser, handle_setup_vulkan

    binary = tmp_path / "bin" / "audiocpp_cli"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("x")
    model = tmp_path / "models" / "chatterbox-q8_0.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("m")
    root = _fake_repo(tmp_path, binary, model)
    monkeypatch.setattr(vulkan_setup, "find_audiocpp_binary", lambda: binary)
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model))
    # Point the CLI at the hermetic repo so no real scripts run.
    monkeypatch.setattr(vulkan_setup, "_repo_root", lambda: root)

    parser = build_parser()
    args = parser.parse_args(["setup-vulkan"])
    assert args.command == "setup-vulkan"
    assert handle_setup_vulkan() == 0
    captured = capsys.readouterr()
    assert "already configured" in captured.err
    assert str(binary) in captured.out
    assert str(model) in captured.out
