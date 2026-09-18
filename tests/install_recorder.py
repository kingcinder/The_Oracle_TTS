"""Record exactly what an install would execute and write, with no side effects.

The launch-readiness passes validated ``bootstrap``/``install`` by intercepting at
the process boundary: ``subprocess.run`` replaced by a recorder, and the user's
``HOME``/XDG/HF roots redirected into scratch space. That harness lived in an
uncommitted scratch script, so the exact commands and files an install produces
were never pinned by the suite. This module is that harness, committed.

What a test gets from :class:`InstallBoundary`:

* every command, as its exact argv list and in order (``commands``);
* every file the run created, relative to a scratch root, with its mode and
  contents (``created_files``);
* a deterministic environment: ``HOME``/``XDG_*``/``HF_HUB_CACHE``/``APPDATA``
  point into scratch, ``Path.home()`` follows, the platform is forced (so the
  recorded boundary is byte-identical on Linux and Windows CI), and
  ``shutil.which`` is pinned (so a missing ``gio`` on the runner cannot change
  the recorded command list).

Nothing here installs a package, touches the network, or writes outside the
scratch roots: intercepted commands return a fake ``CompletedProcess``, and a
recorded ``python -m venv`` is simulated by writing a placeholder interpreter
file so the rest of an install still proceeds.
"""

from __future__ import annotations

import importlib.util
import os
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: Best-effort desktop-integration helpers the installer looks up with
#: ``shutil.which``. Pinned so the recorded command list does not depend on what
#: the runner happens to have installed.
DESKTOP_TOOLS = ("update-desktop-database", "gio")


def load_manage_install(name: str = "oracle_manage_install_boundary"):
    """Load ``scripts/manage_install.py`` without executing its ``__main__`` path."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / "manage_install.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Command:
    """One intercepted subprocess command."""

    args: tuple[str, ...]
    cwd: str
    env: Mapping[str, str]

    @property
    def line(self) -> str:
        """The command as one readable string (for assertion messages)."""
        return " ".join(f"'{arg}'" if " " in arg else arg for arg in self.args)

    def __str__(self) -> str:  # convenience when a test prints the boundary
        return self.line


@dataclass(frozen=True)
class WrittenFile:
    """One file the run created under a scratch root, with its mode and text."""

    label: str  # "<home|repo>/<relative path>"
    mode: int
    text: str

    @property
    def executable(self) -> bool:
        return bool(self.mode & 0o111)


class InstallBoundary:
    """Drive an install with the process boundary and the user's roots captured."""

    def __init__(
        self,
        module,
        monkeypatch,
        *,
        home: Path,
        repo_root: Path | None = None,
        with_venv: bool = True,
        tools: Sequence[str] = DESKTOP_TOOLS,
        platform: str = "linux",
        bundle: Path | None = None,
    ) -> None:
        self.module = module
        self.monkeypatch = monkeypatch
        self.home = Path(home)
        # A space in the repo path is the hostile case this project actually
        # lives at, so it is the default here rather than a special case.
        self.repo = Path(repo_root) if repo_root is not None else self.home.parent / "repo with space"
        self.bundle = Path(bundle) if bundle is not None else None
        self.calls: list[Command] = []
        self._platform = platform
        self._tools = tuple(tools)

        self.home.mkdir(parents=True, exist_ok=True)
        self.repo.mkdir(parents=True, exist_ok=True)
        self._configure()
        if self.bundle is not None:
            self._write_bundle(self.bundle)
        if with_venv:
            self.venv_python.parent.mkdir(parents=True, exist_ok=True)
            self.venv_python.write_text("# placeholder interpreter\n", encoding="utf-8")
        monkeypatch.setattr(module.subprocess, "run", self._record)
        self._before = self._snapshot()

    def _configure(self) -> None:
        """Apply this boundary's wiring to the module and the environment."""
        self._force_platform(self._platform)
        self._redirect_roots()
        self._pin_tools()
        self.monkeypatch.setattr(self.module, "REPO_ROOT", self.repo)

    # -- environment ---------------------------------------------------------

    @property
    def venv_python(self) -> Path:
        return self.module.venv_python_path(self.repo)

    @property
    def launcher(self) -> Path:
        return self.module.managed_launcher_path()

    def _force_platform(self, platform: str) -> None:
        """Force Linux or Windows behaviour in both namespaces.

        ``manage_install`` imported ``is_windows``/``is_linux`` by name while
        ``platform_support``'s own path helpers call their module-local copies,
        so patching only one of them yields an inconsistent boundary (a Windows
        wrapper body around a POSIX venv path).
        """
        from the_oracle import platform_support

        windows = platform == "windows"
        for target in (platform_support, self.module):
            self.monkeypatch.setattr(target, "is_windows", lambda: windows)
            self.monkeypatch.setattr(target, "is_linux", lambda: not windows)

    def _redirect_roots(self) -> None:
        home = self.home
        env = {
            "HOME": str(home),
            "USERPROFILE": str(home),  # Path.home() on Windows
            "APPDATA": str(home / "AppData" / "Roaming"),
            "XDG_BIN_HOME": str(home / ".local" / "bin"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "HF_HUB_CACHE": str(home / ".cache" / "huggingface" / "hub"),
        }
        for key, value in env.items():
            self.monkeypatch.setenv(key, value)
        self.monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    def _pin_tools(self) -> None:
        found = {name: f"/usr/bin/{name}" for name in self._tools}
        self.monkeypatch.setattr(self.module.shutil, "which", lambda name: found.get(name))

    def _write_bundle(self, bundle: Path) -> None:
        """A minimal *valid* offline bundle: manifest, this platform's wheel dir,
        and every pinned model in hub-cache layout. A partial bundle is rejected
        before anything is seeded, so a partial fixture would only ever record an
        aborted install."""
        from the_oracle.models.pins import MODEL_PINS

        platform_dir = "windows" if self.module.is_windows() else "linux"
        (bundle / "wheels" / platform_dir).mkdir(parents=True, exist_ok=True)
        (bundle / "manifest.json").write_text('{"app": "the-oracle"}', encoding="utf-8")
        for repo_id, sha in MODEL_PINS.items():
            snapshot = (
                bundle / "hf_cache" / ("models--" + repo_id.replace("/", "--")) / "snapshots" / sha
            )
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")

    # -- interception --------------------------------------------------------

    def _record(self, args, **kwargs):
        assert not isinstance(args, (str, bytes)), "installer commands must be argv lists"
        assert not kwargs.get("shell"), "installer commands must not use a shell"
        argv = tuple(str(arg) for arg in args)
        self.calls.append(
            Command(args=argv, cwd=str(kwargs.get("cwd") or ""), env=dict(kwargs.get("env") or {}))
        )
        if len(argv) > 2 and argv[1:3] == ("-m", "venv"):
            # `python -m venv <dir>` really creates the interpreter; the rest of
            # an install (ensure_venv, run_doctor) checks for it, so materialize
            # the placeholder rather than pretending and then failing.
            self.venv_python.parent.mkdir(parents=True, exist_ok=True)
            self.venv_python.write_text("# placeholder interpreter\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    # -- running -------------------------------------------------------------

    def _attach(self) -> None:
        """Re-wire this boundary before a run and record into it.

        Several boundaries can exist in one test (an online install and an
        offline one, say), and every boundary patches the same module-level
        names — the environment roots, ``REPO_ROOT``, ``shutil.which`` and
        ``subprocess.run``. Re-applying them per run means each boundary records
        its own run into its own roots, instead of the most recently constructed
        boundary capturing every call.
        """
        self._configure()
        self.monkeypatch.setattr(self.module.subprocess, "run", self._record)

    def install(self, **kwargs: Any) -> int:
        self._attach()
        return self.module.install(**kwargs)

    def bootstrap(self, **kwargs: Any) -> int:
        self._attach()
        return self.module.bootstrap(**kwargs)

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run anything else against this boundary (e.g. ``update``/``run_gui``)."""
        self._attach()
        return fn(*args, **kwargs)

    # -- recorded evidence ---------------------------------------------------

    @property
    def commands(self) -> tuple[Command, ...]:
        return tuple(self.calls)

    def command_lines(self) -> list[str]:
        return [command.line for command in self.calls]

    @property
    def pip_commands(self) -> list[Command]:
        return [c for c in self.calls if "-m" in c.args and "pip" in c.args]

    @property
    def doctor_commands(self) -> list[Command]:
        return [c for c in self.calls if any("doctor.py" in arg for arg in c.args)]

    @property
    def created_files(self) -> tuple[WrittenFile, ...]:
        after = self._snapshot()
        return tuple(after[label] for label in sorted(set(after) - set(self._before)))

    def created_labels(self) -> list[str]:
        return [entry.label for entry in self.created_files]

    def created_file(self, label: str) -> WrittenFile:
        for entry in self.created_files:
            if entry.label == label:
                return entry
        raise AssertionError(f"{label} was not created; created: {self.created_labels()}")

    def _snapshot(self) -> dict[str, WrittenFile]:
        found: dict[str, WrittenFile] = {}
        for root_name, root in (("home", self.home), ("repo", self.repo)):
            for path in sorted(root.rglob("*")):
                if path.is_dir() or (path.is_symlink() and not path.exists()):
                    continue
                found[f"{root_name}/{path.relative_to(root)}"] = WrittenFile(
                    label=f"{root_name}/{path.relative_to(root)}",
                    mode=path.stat().st_mode & 0o777,
                    text=_read_text(path),
                )
        return found

    def assert_touches_only_scratch(self) -> None:
        """Every absolute path in every command must live in a scratch root."""
        roots = [self.home, self.repo] + ([self.bundle] if self.bundle is not None else [])
        allowed = tuple(str(root) for root in roots)
        offenders = [
            arg
            for command in self.calls
            for arg in command.args
            if _looks_absolute(arg) and not arg.startswith(allowed)
        ]
        assert not offenders, f"commands reference paths outside scratch: {offenders}"


def _read_text(path: Path) -> str:
    # Read bytes, not text: Path.read_text translates newlines, which would hide
    # whether a generated .cmd launcher really carries the CRLF endings Windows
    # needs (the whole point of recording file contents).
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return f"<binary {path.stat().st_size}B>"


def _looks_absolute(value: str) -> bool:
    return value.startswith("/") or (len(value) > 2 and value[1:3] in (":\\", ":/"))


@pytest.fixture()
def install_boundary(tmp_path: Path, monkeypatch):
    """Factory fixture: ``install_boundary(module, ...)`` returns a wired boundary.

    Each call gets its own scratch ``home`` (and repo root) under ``tmp_path`` so
    a test can record more than one install — e.g. online and offline — without
    the two runs shadowing each other's files.
    """
    made: list[InstallBoundary] = []

    def _make(module, **kwargs: Any) -> InstallBoundary:
        home = kwargs.pop("home", None) or tmp_path / f"home{len(made)}"
        # Each boundary gets its own repo root as well as its own HOME: a shared
        # one would let one run's files (an offline marker, say) show up in the
        # other's created-file diff.
        repo_root = kwargs.pop("repo_root", None) or tmp_path / f"repo{len(made)} with space"
        boundary = InstallBoundary(
            module, monkeypatch, home=Path(home), repo_root=Path(repo_root), **kwargs
        )
        made.append(boundary)
        return boundary

    return _make
