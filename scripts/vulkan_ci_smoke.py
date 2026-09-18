#!/usr/bin/env python3
"""Provision the Vulkan backend in CI and run the smoke that needs it.

The Vulkan smoke (``tests/test_vulkan_backend.py::test_vulkan_backend_smoke_requires_device``)
has skipped in CI since it was written. It needs three things the runner did not
have — ``vulkaninfo`` reporting a device, a built ``audiocpp_cli`` and the
Chatterbox ggml model — and a skip is indistinguishable from a pass in a green
run, so the suite stayed green while covering nothing.

This prepares all three and then refuses to let anything skip: it runs the suite
with ``ORACLE_FAIL_ON_SKIP=1`` (see ``tests/skip_audit.py``), so a skip fails the
job instead of hiding in it.

Before running the suite it asserts the smoke's *own* three gates resolve — the
same ``vulkan_device_available`` / ``find_audiocpp_binary`` / ``find_audiocpp_model``
the test's guard calls. That is what makes "the smoke no longer skips" a checked
claim rather than an expectation: if any gate is still open, this fails with the
reason and the suite is never run.

The build itself is not reimplemented here: ``scripts/build_audio_cpp.sh`` and
``scripts/download_audio_cpp_model.sh`` own it, exactly as the README documents, so
the job exercises the instructions a user follows.

Run by ``.github/workflows/ci.yml`` (the ``vulkan-smoke`` job)::

    .venv/bin/python scripts/vulkan_ci_smoke.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: Commands this job needs before it can do anything, with the Debian/Ubuntu
#: package that provides each (GitHub's ubuntu runner). The hints are in the
#: failure text because a missing tool is the most common way this job breaks and
#: the runner's message for it is otherwise a bare "command not found".
REQUIRED_TOOLS: tuple[tuple[str, str], ...] = (
    ("git", "git"),
    ("cmake", "cmake"),
    # audio.cpp's ggml-vulkan CMakeLists does
    # `find_package(Vulkan COMPONENTS glslc REQUIRED)`, so this is a build
    # requirement, not a nicety.
    ("glslc", "glslc"),
    ("vulkaninfo", "vulkan-tools"),
    ("g++", "g++ (or set CXX to another C++17 compiler)"),
)


class VulkanSmokeError(RuntimeError):
    """A step of the job did not produce what the next step requires."""


def missing_tools(which: Callable[[str], str | None] = shutil.which) -> list[str]:
    """One actionable line per missing command, empty when everything is present."""
    missing: list[str] = []
    for command, package in REQUIRED_TOOLS:
        if which(command) is None:
            missing.append(f"{command} is not on PATH (install the `{package}` package)")
    return missing


def _run(argv: Sequence[str], *, env: Mapping[str, str] | None = None, description: str) -> None:
    """Run a step, echoing its output, and fail with a clear message if it fails."""
    print(f"\n=== {description} ===", flush=True)
    completed = subprocess.run(
        list(argv),
        cwd=str(REPO_ROOT),
        env=dict(os.environ if env is None else env),
        check=False,
    )
    if completed.returncode != 0:
        raise VulkanSmokeError(f"{description} exited {completed.returncode}")


def check_gates() -> dict[str, object]:
    """The smoke's own three gates, resolved in this process.

    Imported from the application rather than re-derived, so this cannot drift
    from what the test checks. Raises with the failing gate named.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from the_oracle.tts_engines.vulkan_backend import (
        find_audiocpp_binary,
        find_audiocpp_model,
        vulkan_device_available,
    )

    gates: dict[str, object] = {
        "vulkan_device_available": vulkan_device_available(),
        "audiocpp_cli": find_audiocpp_binary(),
        "chatterbox_model": find_audiocpp_model(),
    }
    problems: list[str] = []
    if not gates["vulkan_device_available"]:
        problems.append(
            "no Vulkan device: `vulkaninfo --summary` failed. On a GPU-less runner the "
            "software ICD from `mesa-vulkan-drivers` (+ `vulkan-tools`) is what makes "
            "this pass; check `vulkaninfo --summary` by hand."
        )
    if gates["audiocpp_cli"] is None:
        problems.append("audiocpp_cli not found: scripts/build_audio_cpp.sh did not produce it")
    if gates["chatterbox_model"] is None:
        problems.append(
            "the Chatterbox ggml model not found: scripts/download_audio_cpp_model.sh did not"
            " install it (expected under audio.cpp/models/Chatterbox-GGUF)"
        )
    if problems:
        raise VulkanSmokeError(
            "the Vulkan smoke would still skip, so this job would prove nothing:\n  - "
            + "\n  - ".join(problems)
        )
    return gates


def run_suite(*, strict: bool, extra_args: Sequence[str]) -> None:
    """Run the suite, demanding that no test skip when ``strict``."""
    env = dict(os.environ)
    if strict:
        # tests/skip_audit.py turns any skip into a failed run.
        env["ORACLE_FAIL_ON_SKIP"] = "1"
    argv = [sys.executable, "-m", "pytest", *extra_args]
    print(f"\n=== {' '.join(argv)}  (ORACLE_FAIL_ON_SKIP={env.get('ORACLE_FAIL_ON_SKIP', 'unset')}) ===", flush=True)
    completed = subprocess.run(argv, cwd=str(REPO_ROOT), env=env, check=False)
    if completed.returncode != 0:
        raise VulkanSmokeError(f"the suite failed (exit {completed.returncode}) with skips disallowed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-build", action="store_true", help="Reuse an audiocpp_cli that is already built.")
    parser.add_argument("--skip-model", action="store_true", help="Reuse an already-downloaded model.")
    parser.add_argument("--allow-skips", action="store_true", help="Run the suite without ORACLE_FAIL_ON_SKIP.")
    # Everything after the known flags goes to pytest, so `-- -q tests/...` works
    # with pytest's own option syntax instead of argparse eating the leading dash.
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Arguments passed to pytest (after --).")
    args = parser.parse_args(argv)
    pytest_args = [arg for arg in args.pytest_args if arg != "--"] or ["-q"]

    print(f"Repo root: {REPO_ROOT}")
    absent = missing_tools()
    if absent:
        print("\nFAIL: this job cannot build audio.cpp yet:", file=sys.stderr)
        for line in absent:
            print(f"  - {line}", file=sys.stderr)
        return 2

    try:
        if not args.skip_build:
            _run([str(SCRIPTS_DIR / "build_audio_cpp.sh")], description="build audio.cpp (Vulkan backend)")
        if not args.skip_model:
            _run(
                [str(SCRIPTS_DIR / "download_audio_cpp_model.sh")],
                description="fetch the Chatterbox ggml model",
            )

        gates = check_gates()
        print("\nThe smoke's gates, resolved:")
        for name, value in gates.items():
            print(f"  {name}: {value}")

        run_suite(strict=not args.allow_skips, extra_args=tuple(pytest_args))
    except VulkanSmokeError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1

    print("\nPASS: the Vulkan smoke's gates are all satisfied and the suite ran without skipping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
