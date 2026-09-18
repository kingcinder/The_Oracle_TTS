"""Tests for the Vulkan CI orchestrator and the job that runs it.

Building audio.cpp and downloading the ggml model are far too slow (and need too
much network) to exercise here, so what is pinned is everything that decides
whether those steps are worth running: the preflight, the gate check that has to
agree with the smoke test's *own* guard, the strict-skip handoff to the suite, and
the CI job's declared contract.

The gate check is the important one. It means the job cannot report success while
the smoke would still skip, which is the failure this whole job exists to remove.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from the_oracle.tts_engines import vulkan_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture()
def smoke():
    spec = importlib.util.spec_from_file_location("oracle_vulkan_ci_smoke", SCRIPTS_DIR / "vulkan_ci_smoke.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- preflight ------------------------------------------------------------------


def test_preflight_is_quiet_when_every_tool_is_present(smoke) -> None:
    assert smoke.missing_tools(which=lambda _command: "/usr/bin/whatever") == []


def test_preflight_names_the_command_and_the_package_that_provides_it(smoke) -> None:
    missing = smoke.missing_tools(which=lambda command: None if command == "glslc" else "/usr/bin/ok")

    assert len(missing) == 1
    # Both halves matter: which command is missing, and what to install for it.
    assert missing[0].startswith("glslc is not on PATH")
    assert "`glslc`" in missing[0]


def test_every_required_tool_carries_an_install_hint(smoke) -> None:
    for command, package in smoke.REQUIRED_TOOLS:
        assert command.strip()
        assert package.strip()


def test_vulkan_tools_and_the_compiler_are_required(smoke) -> None:
    """The two the Vulkan smoke cannot run without, beyond git/cmake."""
    commands = {command for command, _ in smoke.REQUIRED_TOOLS}

    assert {"vulkaninfo", "cmake", "git"} <= commands
    assert any(command in {"g++", "clang++", "c++"} for command in commands)


# --- the gate check (must agree with the smoke test's own guard) ------------------


def _gates(monkeypatch, *, device=True, binary=Path("/tmp/audiocpp_cli"), model=Path("/tmp/model.gguf")):
    monkeypatch.setattr(vulkan_backend, "vulkan_device_available", lambda: device)
    monkeypatch.setattr(vulkan_backend, "find_audiocpp_binary", lambda: binary)
    monkeypatch.setattr(vulkan_backend, "find_audiocpp_model", lambda: model)


def test_gate_check_resolves_every_gate_the_smoke_tests(smoke, monkeypatch) -> None:
    _gates(monkeypatch)

    gates = smoke.check_gates()

    assert gates["vulkan_device_available"] is True
    assert gates["audiocpp_cli"] == Path("/tmp/audiocpp_cli")
    assert gates["chatterbox_model"] == Path("/tmp/model.gguf")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"device": False}, "mesa-vulkan-drivers"),
        ({"binary": None}, "audiocpp_cli"),
        ({"model": None}, "Chatterbox ggml model"),
    ],
)
def test_gate_check_refuses_to_claim_readiness(smoke, monkeypatch, kwargs, expected: str) -> None:
    _gates(monkeypatch, **kwargs)

    with pytest.raises(smoke.VulkanSmokeError) as excinfo:
        smoke.check_gates()

    message = str(excinfo.value)
    assert expected in message
    # The point of the check: it says the smoke would still skip.
    assert "would still skip" in message


def test_gate_check_reports_every_open_gate_at_once(smoke, monkeypatch) -> None:
    _gates(monkeypatch, device=False, binary=None, model=None)

    with pytest.raises(smoke.VulkanSmokeError) as excinfo:
        smoke.check_gates()

    message = str(excinfo.value)
    assert message.count("- ") == 3


# --- the strict-skip handoff -----------------------------------------------------


class _Recorder:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[dict] = []

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": list(argv), **kwargs})
        return type("Completed", (), {"returncode": self.returncode})()


def test_the_suite_is_run_with_skips_disallowed(smoke, monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(smoke.subprocess, "run", recorder)

    smoke.run_suite(strict=True, extra_args=("-q",))

    assert recorder.calls[0]["env"]["ORACLE_FAIL_ON_SKIP"] == "1"
    assert recorder.calls[0]["argv"] == [sys.executable, "-m", "pytest", "-q"]


def test_allow_skips_leaves_the_variable_unset(smoke, monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(smoke.subprocess, "run", recorder)
    monkeypatch.delenv("ORACLE_FAIL_ON_SKIP", raising=False)

    smoke.run_suite(strict=False, extra_args=("-q",))

    assert "ORACLE_FAIL_ON_SKIP" not in recorder.calls[0]["env"]


def test_a_failing_suite_is_reported_as_a_failure(smoke, monkeypatch) -> None:
    monkeypatch.setattr(smoke.subprocess, "run", _Recorder(returncode=1))

    with pytest.raises(smoke.VulkanSmokeError, match="exit 1"):
        smoke.run_suite(strict=True, extra_args=("-q",))


# --- main() ---------------------------------------------------------------------


def _stub_pipeline(smoke, monkeypatch, *, gates=None, suite_error: Exception | None = None):
    """Replace the slow steps with recorders, keeping main()'s decisions real."""
    steps: list[list[str]] = []
    suite: list[bool] = []

    def fake_run(argv, **kwargs):
        steps.append(list(argv))
        return type("Completed", (), {"returncode": 0})()

    def fake_gates():
        if isinstance(gates, Exception):
            raise gates
        return gates if gates is not None else {
            "vulkan_device_available": True,
            "audiocpp_cli": Path("/tmp/audiocpp_cli"),
            "chatterbox_model": Path("/tmp/model.gguf"),
        }

    def fake_suite(*, strict, extra_args):
        suite.append(strict)
        if suite_error is not None:
            raise suite_error

    monkeypatch.setattr(smoke, "_run", fake_run)
    monkeypatch.setattr(smoke, "check_gates", fake_gates)
    monkeypatch.setattr(smoke, "run_suite", fake_suite)
    monkeypatch.setattr(smoke, "missing_tools", lambda: [])
    return steps, suite


def test_main_builds_then_fetches_then_runs_strictly(smoke, monkeypatch) -> None:
    steps, suite = _stub_pipeline(smoke, monkeypatch)

    assert smoke.main([]) == 0
    assert [Path(step[0]).name for step in steps] == ["build_audio_cpp.sh", "download_audio_cpp_model.sh"]
    assert suite == [True]


def test_main_can_reuse_an_existing_build_and_model(smoke, monkeypatch) -> None:
    steps, suite = _stub_pipeline(smoke, monkeypatch)

    assert smoke.main(["--skip-build", "--skip-model"]) == 0
    assert steps == []
    assert suite == [True]


def test_main_never_runs_the_suite_when_a_gate_is_still_open(smoke, monkeypatch, capsys) -> None:
    """The whole point: a run that would skip must fail before it can pass."""
    failure = smoke.VulkanSmokeError("the smoke would still skip: no Vulkan device")
    steps, suite = _stub_pipeline(smoke, monkeypatch, gates=failure)

    assert smoke.main(["--skip-build", "--skip-model"]) == 1
    assert suite == []
    assert "FAIL" in capsys.readouterr().err


def test_main_fails_before_building_when_a_tool_is_missing(smoke, monkeypatch, capsys) -> None:
    monkeypatch.setattr(smoke, "missing_tools", lambda: ["cmake is not on PATH (install `cmake`)"])
    monkeypatch.setattr(smoke, "check_gates", lambda: pytest.fail("gates must not be checked yet"))

    assert smoke.main([]) == 2
    assert "cmake" in capsys.readouterr().err


def test_main_reports_a_failed_suite_step(smoke, monkeypatch, capsys) -> None:
    _stub_pipeline(smoke, monkeypatch, suite_error=smoke.VulkanSmokeError("the suite failed (exit 1)"))

    assert smoke.main(["--skip-build", "--skip-model"]) == 1
    assert "the suite failed" in capsys.readouterr().err


def test_main_can_run_the_suite_without_demanding_no_skips(smoke, monkeypatch) -> None:
    _, suite = _stub_pipeline(smoke, monkeypatch)

    assert smoke.main(["--skip-build", "--skip-model", "--allow-skips"]) == 0
    assert suite == [False]


def test_main_passes_pytest_arguments_through(smoke, monkeypatch) -> None:
    _stub_pipeline(smoke, monkeypatch)
    captured: list[tuple] = []
    monkeypatch.setattr(smoke, "run_suite", lambda *, strict, extra_args: captured.append(extra_args))

    smoke.main(["--skip-build", "--skip-model", "--", "-q", "tests/test_vulkan_backend.py"])

    assert captured == [("-q", "tests/test_vulkan_backend.py")]


# --- the CI job's declared contract ---------------------------------------------


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _vulkan_job() -> dict:
    return _workflow()["jobs"]["vulkan-smoke"]


def test_the_job_is_gated_off_pull_requests() -> None:
    job = _vulkan_job()

    assert job["runs-on"] == "ubuntu-latest"
    assert job["if"] == "github.event_name != 'pull_request'"
    # A heavy build on a shared runner needs a bound; the matrix gate stays fast.
    assert job["timeout-minutes"] >= 30


def _apt_packages(script: str) -> list[str]:
    """The packages an `apt-get install` invocation actually asks for.

    Comments are stripped and continuations joined first, because a substring
    search over the raw step text is satisfied by the step's own comment -- which
    is how a workflow docstring can cover for a missing package.
    """
    tokens: list[str] = []
    for line in script.splitlines():
        tokens.extend(line.split("#")[0].replace("\\", " ").split())
    if "install" not in tokens:
        return []
    return [token for token in tokens[tokens.index("install") + 1 :] if not token.startswith("-")]


def test_the_job_installs_every_package_the_build_and_the_probe_need() -> None:
    job = _vulkan_job()
    steps = job["steps"]
    install = next(step for step in steps if step.get("name") == "Install Vulkan And Build Dependencies")

    packages = _apt_packages(install["run"])
    # glslc is required by audio.cpp's ggml-vulkan CMakeLists; without the Mesa
    # driver a GPU-less runner has no ICD at all and the smoke would skip.
    for package in ("cmake", "g++", "git", "glslc", "libvulkan-dev", "mesa-vulkan-drivers", "ninja-build", "vulkan-tools"):
        assert package in packages, package


def test_the_job_verifies_the_device_before_it_builds_anything() -> None:
    steps = _vulkan_job()["steps"]
    names = [step.get("name") for step in steps]

    probe = next(step for step in steps if step.get("name") == "Verify A Vulkan Device Is Visible")
    assert "vulkaninfo --summary" in probe["run"]
    assert names.index("Verify A Vulkan Device Is Visible") < names.index(
        "Build audio.cpp, Fetch The Model, Run The Suite Without Skips"
    )


def test_the_job_runs_the_orchestrator_from_the_venv() -> None:
    steps = _vulkan_job()["steps"]

    run_step = next(step for step in steps if step.get("name") == "Build audio.cpp, Fetch The Model, Run The Suite Without Skips")
    assert run_step["run"].strip() == "./.venv/bin/python scripts/vulkan_ci_smoke.py"
    # The suite needs the venv, so bootstrap has to come first.
    names = [step.get("name") for step in steps]
    assert names.index("Bootstrap Runtime And Test Dependencies") < names.index(
        "Build audio.cpp, Fetch The Model, Run The Suite Without Skips"
    )


def test_the_cached_model_lives_outside_the_clone_path() -> None:
    """A cache restored inside audio.cpp/ makes git clone refuse the directory."""
    job = _vulkan_job()

    models_root = job["env"]["AUDIOCPP_MODELS_ROOT"]
    assert "audio.cpp" not in models_root
    cache = next(step for step in job["steps"] if step.get("uses") == "actions/cache@v4")
    assert cache["with"]["path"] == models_root
    assert "chatterbox" in cache["with"]["key"]


def test_the_workflow_has_triggers_that_keep_the_job_running() -> None:
    workflow = _workflow()
    # PyYAML reads the bare `on:` key as the boolean True.
    triggers = workflow.get("on", workflow.get(True))

    assert "workflow_dispatch" in triggers
    assert "schedule" in triggers
    assert triggers["schedule"]
