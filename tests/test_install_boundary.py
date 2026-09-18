"""What an install executes and writes, pinned exactly.

Uses :mod:`tests.install_recorder`, which replaces ``subprocess.run`` with a
recorder and redirects ``HOME``/``XDG_*``/``HF_HUB_CACHE``/``APPDATA`` into
scratch, so a whole install runs here without pip-installing anything, without
touching the real ``~/.local``, and without network access.

Asserted end to end for both paths: the online install and the offline
``--offline-bundle`` install — their exact command lists, their exact file
sets, and the contents of what gets written. Also pinned: the fresh-repo
``python -m venv`` step, the hostile space-containing repo path this project
actually lives at, the Windows launcher branch, and the invariant that no
recorded command touches a path outside scratch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.install_recorder import install_boundary, load_manage_install

DESKTOP_ENTRY_LABEL = "home/.local/share/applications/the-oracle.desktop"
DESKTOP_SHORTCUT_LABEL = "home/Desktop/the-oracle.desktop"


@pytest.fixture()
def manage_install():
    return load_manage_install()


def _online_pip_commands(venv_python: Path) -> list[tuple[str, ...]]:
    """The five pip installs an online install runs, verbatim."""
    python = str(venv_python)
    return [
        (python, "-m", "pip", "install", "--upgrade", "pip", "setuptools<81", "wheel"),
        (
            python, "-m", "pip", "install",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "torch==2.6.0", "torchaudio==2.6.0", "torchvision==0.21.0",
        ),
        (python, "-m", "pip", "install", "-e", ".[ml]"),
        (
            python, "-m", "pip", "install",
            "librosa==0.11.0", "s3tokenizer", "diffusers==0.29.0", "resemble-perth==1.0.1",
            "conformer==0.3.2", "safetensors==0.5.3", "spacy-pkuseg", "pykakasi==2.3.0",
            "pyloudnorm", "omegaconf",
        ),
        (python, "-m", "pip", "install", "--no-deps", "chatterbox-tts==0.1.6"),
    ]


def _desktop_commands(boundary) -> list[tuple[str, ...]]:
    """The best-effort desktop integration calls, in order."""
    return [
        ("update-desktop-database", str(boundary.home / ".local" / "share" / "applications")),
        ("gio", "set", str(boundary.home / "Desktop" / "the-oracle.desktop"), "metadata::trusted", "true"),
    ]


def _doctor_command(boundary) -> tuple[str, ...]:
    """The single verification run, exactly as install invokes it."""
    return (
        str(boundary.venv_python),
        str(boundary.repo / "scripts" / "doctor.py"),
        "--repo-root",
        str(boundary.repo),
    )


# --- online install -----------------------------------------------------------


def test_online_install_runs_exactly_these_commands(manage_install, install_boundary) -> None:
    boundary = install_boundary(manage_install)

    assert boundary.install(pytorch_runtime="cpu") == 0

    expected = _online_pip_commands(boundary.venv_python) + _desktop_commands(boundary) + [_doctor_command(boundary)]
    assert [command.args for command in boundary.commands] == expected, "\n".join(boundary.command_lines())
    # pip and the verification run from the repo root (pip installs `-e .[ml]`
    # from there, and the doctor is given --repo-root explicitly). The desktop
    # tools take absolute arguments and inherit the caller's own directory.
    assert {command.cwd for command in boundary.pip_commands} == {str(boundary.repo)}
    assert boundary.commands[-1].cwd == str(boundary.repo)
    assert all(command.env.get("PYTHONNOUSERSITE") == "1" for command in boundary.pip_commands)


def test_online_install_verifies_once_after_registering_the_launchers(
    manage_install, install_boundary
) -> None:
    """One full (non-CI) doctor run, and it comes last."""
    boundary = install_boundary(manage_install)

    boundary.install(pytorch_runtime="cpu")

    assert boundary.doctor_commands == [boundary.commands[-1]]
    assert boundary.commands[-1].args == _doctor_command(boundary)
    # CI mode exempts the entrypoint from the doctor's required checks, so the
    # install's single verification must never use it.
    assert "--ci" not in boundary.commands[-1].args
    assert "--skip-model-init" not in boundary.commands[-1].args


def test_online_install_writes_exactly_these_files(manage_install, install_boundary) -> None:
    boundary = install_boundary(manage_install)

    boundary.install(pytorch_runtime="cpu")

    assert boundary.created_labels() == [
        "home/.local/bin/the-oracle",
        DESKTOP_ENTRY_LABEL,
        DESKTOP_SHORTCUT_LABEL,
    ]

    wrapper = boundary.created_file("home/.local/bin/the-oracle")
    assert wrapper.executable, "the managed launcher must be executable"
    assert wrapper.text == manage_install.managed_wrapper_contents()
    assert wrapper.text.startswith("#!/usr/bin/env bash\n")
    assert "exec \"$VENV_ENTRYPOINT\" \"$@\"\n" in wrapper.text
    # The space-containing repo path is quoted, so the generated script cannot
    # split it into two arguments.
    assert f"REPO_ROOT='{boundary.repo}'" in wrapper.text

    entry = boundary.created_file(DESKTOP_ENTRY_LABEL)
    assert not entry.executable, "the applications-menu entry is not a launcher file"
    assert f'Exec="{boundary.launcher}" gui' in entry.text
    assert "Trusted=true" not in entry.text

    shortcut = boundary.created_file(DESKTOP_SHORTCUT_LABEL)
    assert shortcut.executable, "GNOME needs the desktop shortcut executable"
    assert "Trusted=true" in shortcut.text


def test_online_install_touches_nothing_outside_scratch(manage_install, install_boundary) -> None:
    boundary = install_boundary(manage_install)

    boundary.install(pytorch_runtime="cpu")

    boundary.assert_touches_only_scratch()


# --- offline install ----------------------------------------------------------


def _offline_pip_commands(venv_python: Path, wheels_dir: Path) -> list[tuple[str, ...]]:
    """The same five installs, each pinned to the bundle's wheel directory."""
    wheel_args = ("--no-index", "--find-links", str(wheels_dir))
    python = str(venv_python)
    return [
        (python, "-m", "pip", "install", *wheel_args, "--upgrade", "pip", "setuptools<81", "wheel"),
        (
            python, "-m", "pip", "install", *wheel_args,
            "torch==2.6.0", "torchaudio==2.6.0", "torchvision==0.21.0",
        ),
        (python, "-m", "pip", "install", *wheel_args, "-e", ".[ml]"),
        (
            python, "-m", "pip", "install", *wheel_args,
            "librosa==0.11.0", "s3tokenizer", "diffusers==0.29.0", "resemble-perth==1.0.1",
            "conformer==0.3.2", "safetensors==0.5.3", "spacy-pkuseg", "pykakasi==2.3.0",
            "pyloudnorm", "omegaconf",
        ),
        (python, "-m", "pip", "install", *wheel_args, "--no-deps", "chatterbox-tts==0.1.6"),
    ]


def _expected_offline_labels(boundary) -> list[str]:
    """Every file an offline install creates, derived from the pinned models."""
    from the_oracle.models.pins import MODEL_PINS

    labels = {
        "home/.local/bin/the-oracle",
        DESKTOP_ENTRY_LABEL,
        DESKTOP_SHORTCUT_LABEL,
        "repo/.oracle_offline",
    }
    for repo_id, sha in MODEL_PINS.items():
        cache = f"home/.cache/huggingface/hub/models--{repo_id.replace('/', '--')}"
        labels |= {f"{cache}/snapshots/{sha}/config.json", f"{cache}/refs/main"}
    return sorted(labels)


def test_offline_install_commands_cannot_reach_the_network(manage_install, install_boundary, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    boundary = install_boundary(manage_install, bundle=bundle)
    wheels = bundle / "wheels" / "linux"

    assert boundary.install(pytorch_runtime="cpu", offline_bundle=bundle) == 0

    expected = (
        _offline_pip_commands(boundary.venv_python, wheels)
        + _desktop_commands(boundary)
        + [_doctor_command(boundary)]
    )
    assert [command.args for command in boundary.commands] == expected, "\n".join(boundary.command_lines())

    joined = " ".join(boundary.command_lines())
    assert "--no-index" in joined
    assert "--index-url" not in joined, "an offline install must not name a wheel index"
    assert "https://" not in joined, f"an offline install must not reach the network:\n{joined}"


def test_offline_install_seeds_the_pinned_models_and_writes_the_marker(
    manage_install, install_boundary, tmp_path: Path
) -> None:
    from the_oracle.models.pins import MODEL_PINS

    bundle = tmp_path / "bundle"
    boundary = install_boundary(manage_install, bundle=bundle)

    boundary.install(pytorch_runtime="cpu", offline_bundle=bundle)

    assert boundary.created_labels() == _expected_offline_labels(boundary)
    cache = boundary.home / ".cache" / "huggingface" / "hub"
    for repo_id, sha in MODEL_PINS.items():
        ref = cache / ("models--" + repo_id.replace("/", "--")) / "refs" / "main"
        assert ref.read_text(encoding="utf-8").strip() == sha, f"{repo_id}: refs/main must pin the revision"
    marker = boundary.created_file("repo/.oracle_offline").text
    assert json.loads(marker) == {"offline_bundle_manifest": '{"app": "the-oracle"}'}
    # The launcher is what makes the promise true at run time.
    assert "HF_HUB_OFFLINE=1" in boundary.created_file("home/.local/bin/the-oracle").text


# --- fresh repo, missing tools, Windows ---------------------------------------


def test_fresh_repo_creates_the_venv_before_installing_anything(manage_install, install_boundary) -> None:
    """With no .venv, `python -m venv <repo>/.venv` is the very first step."""
    boundary = install_boundary(manage_install, with_venv=False)

    assert boundary.install(pytorch_runtime="cpu") == 0

    assert boundary.commands[0].args[:3] == (str(manage_install.sys.executable), "-m", "venv")
    assert boundary.commands[0].args[3] == str(boundary.repo / ".venv")
    assert [command.args for command in boundary.pip_commands] == _online_pip_commands(
        boundary.venv_python
    )
    # The rest of the install ran against the venv the recorded command created.
    assert boundary.venv_python.is_file()
    assert boundary.doctor_commands == [boundary.commands[-1]]


def test_missing_desktop_tools_are_skipped_silently(manage_install, install_boundary) -> None:
    """No update-desktop-database / gio on the box: fewer commands, same files."""
    without = install_boundary(manage_install, tools=())
    with_tools = install_boundary(manage_install)

    without.install(pytorch_runtime="cpu")
    with_tools.install(pytorch_runtime="cpu")

    expected = _online_pip_commands(without.venv_python) + [_doctor_command(without)]
    assert [command.args for command in without.commands] == expected
    assert without.created_labels() == with_tools.created_labels()
    assert without.created_file(DESKTOP_SHORTCUT_LABEL).executable


def test_windows_install_registers_a_start_menu_launcher(manage_install, install_boundary) -> None:
    boundary = install_boundary(manage_install, platform="windows")

    assert boundary.install(pytorch_runtime="cpu") == 0

    expected = _online_pip_commands(boundary.venv_python) + [_doctor_command(boundary)]
    assert [command.args for command in boundary.commands] == expected, "\n".join(boundary.command_lines())
    assert boundary.created_labels() == [
        "home/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/The Oracle.cmd",
        "home/AppData/Roaming/Python/Scripts/the-oracle.cmd",
    ]

    wrapper = boundary.created_file("home/AppData/Roaming/Python/Scripts/the-oracle.cmd")
    assert "\r\n" in wrapper.text, "a .cmd launcher needs CRLF line endings"
    assert 'set "REPO_ROOT=' in wrapper.text
    assert "HF_HUB_OFFLINE=1" in wrapper.text

    start_menu = boundary.created_file(
        "home/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/The Oracle.cmd"
    )
    assert f'call "{boundary.launcher}" gui %*' in start_menu.text
