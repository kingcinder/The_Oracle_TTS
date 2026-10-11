"""The shared certifier preamble: one owner, tested where it lives.

``scripts/certifier_isolation.py`` holds the throwaway config root, the two
environment redirects and the reap that used to be copy-pasted into
``certify_gui.py`` and ``certify_theme_switch.py``. The certifier tests still
prove the END of that contract (a real sweep leaves the repo and the real config
untouched, and leaves no root in /tmp); these tests prove the helper's own
halves, so the claims the certifier tests delegate — both variables redirected,
the reap armed on every exit path, a survivor reported rather than swallowed —
have somewhere to fail.

The helper writes ``os.environ`` directly (it has to: the whole point is that
the redirect is in place before anything reads it), so every test here restores
the environment itself and points ``TMPDIR`` at pytest's own scratch directory.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
HELPER_PATH = SCRIPTS_DIR / "certifier_isolation.py"

CERTIFIERS = (SCRIPTS_DIR / "certify_gui.py", SCRIPTS_DIR / "certify_theme_switch.py")


@pytest.fixture(scope="module")
def isolation():
    """The helper, imported by path — the idiom this repo's tests use."""
    spec = importlib.util.spec_from_file_location("oracle_certifier_isolation", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config_env(monkeypatch, tmp_path: Path) -> Path:
    """A scratch TMPDIR, and the config environment put back afterwards.

    Roots land in ``tmp_path`` so pytest removes them, and the two variables the
    helper overwrites are restored from copies taken before the test ran.
    """
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    saved = {name: os.environ.get(name) for name in ("XDG_CONFIG_HOME", "APPDATA")}
    yield scratch
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_the_helper_redirects_both_config_variables(isolation, config_env) -> None:
    """Both variables, because the real reader picks one per platform.

    ``platform_support.user_config_root`` reads ``XDG_CONFIG_HOME`` on POSIX and
    ``%APPDATA%`` on Windows, so redirecting one alone leaves the other
    platform's real config root readable and writable. Asserted through that
    reader as well as against the names, which is the claim that matters.
    """
    from the_oracle.platform_support import user_config_root

    # Spelled out here rather than read off THROWAWAY_ENV_VARS: a loop over that
    # tuple would keep passing after a variable was dropped from it, which is
    # exactly the regression this test exists to catch (found by mutation —
    # removing APPDATA from the tuple left all eight tests green).
    redirects = ("XDG_CONFIG_HOME", "APPDATA")
    assert set(isolation.THROWAWAY_ENV_VARS) == set(redirects), (
        "the helper no longer redirects both config variables"
    )

    root = isolation.isolated_config_root("probe_cfg_")
    try:
        assert isinstance(root, str), "callers hand this to functions that take a path"
        assert Path(root).is_dir()
        for name in redirects:
            assert os.environ.get(name) == root, f"{name} was not redirected to the throwaway"
        assert user_config_root() == Path(root), (
            "the real reader did not resolve inside the throwaway — the redirect "
            "is cosmetic, not hermetic"
        )
    finally:
        isolation.reap_isolated_root(root)


def test_the_helper_is_package_free(isolation) -> None:
    """It has to be importable BEFORE the config root exists.

    That is the ordering rule the certifiers' pins depend on: if importing this
    module pulled in ``the_oracle``, the redirect would necessarily happen after
    the package was imported. Proven in a fresh interpreter, because this test
    process has ``the_oracle`` loaded for other reasons — and the module's own
    prose is allowed to name the package, since the rule it documents is about
    that package; what must stay absent is an import of it.
    """
    code = (
        f"import sys; sys.path.insert(0, {str(SCRIPTS_DIR)!r});"
        " import certifier_isolation;"
        " print('the_oracle' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", (
        "importing the isolation helper pulled in the_oracle, so the config "
        "redirect cannot precede the package import"
    )
    source = HELPER_PATH.read_text(encoding="utf-8")
    assert not re.search(r"^(?:from|import) the_oracle", source, re.MULTILINE), (
        "the helper imports the package it must be importable without"
    )


def test_reap_removes_the_root_and_an_absent_root_is_success(isolation, config_env) -> None:
    """Idempotent on purpose: teardown reaps, then the atexit hook reaps again."""
    root = isolation.isolated_config_root("probe_reap_", attempts=1, pause=0.0)
    assert Path(root).is_dir()
    marker = Path(root) / "the_oracle" / "app_settings.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")

    assert isolation.reap_isolated_root(root) is True
    assert not Path(root).exists()
    # The second call is the one interpreter exit makes.
    assert isolation.reap_isolated_root(root) is True
    assert config_env.is_dir()


def test_a_root_that_survives_is_reported_not_swallowed(isolation, config_env, capsys) -> None:
    """Litter is a defect: the warning names the root, and the verdict is False.

    Made undeletable the way a real one can be — the directory itself refuses
    writes, so ``rmtree`` cannot unlink what is inside it. Whatever the platform
    does with that, the contract is the same: the helper must not report success
    for a root that is still there.
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can unlink inside a read-only directory; the survivor path is unreachable")

    root = isolation.isolated_config_root("probe_survives_", attempts=2, pause=0.0)
    (Path(root) / "inner").mkdir()
    Path(root).chmod(0o500)
    try:
        assert isolation.reap_isolated_root(root, attempts=2, pause=0.0) is False
        assert Path(root).exists()
        assert f"warning: {root} survived 2 reap attempts" in capsys.readouterr().err
    finally:
        Path(root).chmod(0o700)


def test_the_root_is_reaped_at_interpreter_exit(isolation, config_env) -> None:
    """The exits that never reach a teardown: argparse's SystemExit, a usage error.

    Measured when the certifiers' litter was chased down: those were exactly the
    exits that left a root behind, one per run. This is the atexit half, run for
    real — a child that creates a root and dies of ``SystemExit(2)`` must leave
    nothing.
    """
    code = (
        f"import sys; sys.path.insert(0, {str(SCRIPTS_DIR)!r});"
        " from certifier_isolation import isolated_config_root;"
        " print(isolated_config_root('probe_exit_'));"
        " raise SystemExit(2)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "TMPDIR": str(config_env)},
    )
    assert proc.returncode == 2, proc.stderr
    root = proc.stdout.strip()
    assert root, f"the child did not report its root\n{proc.stderr}"
    assert not Path(root).exists(), "the root outlived the interpreter that made it"
    assert sorted(path.name for path in config_env.iterdir()) == [], (
        "the child's scratch directory is not empty after its exit"
    )


def _preamble_shapes(source: str) -> list[str]:
    """The syntax of the preamble that was removed, found in CODE rather than prose.

    A textual scan would flag the scripts' own docstrings, which describe the
    mkdtemp and the redirect they no longer implement — prose explaining a rule
    is not a second copy of it. So this walks the parse tree.
    """
    shapes: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in {"tempfile.mkdtemp", "shutil.rmtree", "atexit.register"}:
                shapes.append(f"{name}(...)")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and _dotted(target.value) == "os.environ"
                ):
                    shapes.append(f"os.environ[{_literal_key(target)}] = ...")
    return shapes


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _literal_key(node: ast.Subscript) -> str:
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return "..."


@pytest.mark.parametrize("script", CERTIFIERS, ids=lambda path: path.name)
def test_both_certifiers_use_the_helper_instead_of_a_copy(script: Path) -> None:
    """The dedupe itself, pinned: a second copy of the policy is a second owner.

    Looks for the SHAPE of the copy that was removed rather than for a helper
    call alone, because a new sweep pasting the preamble back in is the
    regression this refactor exists to prevent.
    """
    source = script.read_text(encoding="utf-8")
    duplicated = _preamble_shapes(source)
    assert not duplicated, (
        f"{script.name} carries its own copy of the isolation preamble ({sorted(set(duplicated))}); "
        "it belongs in scripts/certifier_isolation.py, where it is tested"
    )
    assert "from certifier_isolation import isolated_config_root, reap_isolated_root" in source
    assert "_CONFIG_ROOT = isolated_config_root(" in source
    assert "reap_isolated_root(_CONFIG_ROOT)" in source, (
        f"{script.name} created a throwaway root but never reaps it at its own teardown"
    )


def test_each_certifier_keeps_its_own_root_prefix() -> None:
    """A leaked root has to name the sweep that leaked it."""
    prefixes = {
        "certify_gui.py": "oracle_gui_cert_cfg_",
        "certify_theme_switch.py": "oracle_theme_switch_cfg_",
    }
    for name, prefix in prefixes.items():
        source = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        assert f'isolated_config_root("{prefix}")' in source, (
            f"{name} no longer uses its own prefix {prefix!r} — a root left in /tmp "
            "would not say which sweep left it"
        )
