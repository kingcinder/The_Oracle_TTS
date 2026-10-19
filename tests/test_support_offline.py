"""The support layer is local-only by construction.

Two pins, mirroring the crash unit's offline contract:

1. **Source:** nothing in the support package (or ``support_bundle``) imports
   a network client, and nothing calls a browser or URL opener. Links are
   strings; opening one is the GUI's injected callback.
2. **Runtime:** building a report and a package succeeds with outbound
   connections blocked, in a fresh interpreter that never imports torch, Qt,
   or huggingface_hub just by importing the support layer.
"""

from __future__ import annotations

import ast
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from the_oracle import support

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "the_oracle"
SUPPORT_FILES = sorted((SRC / "support").glob("*.py")) + [SRC / "support_bundle.py"]

FORBIDDEN_IMPORTS = {
    "socket",
    "http",
    "http.client",
    "urllib.request",
    "urllib.error",
    "requests",
    "urllib3",
    "webbrowser",
    "aiohttp",
    "ftplib",
    "smtplib",
    "imaplib",
}


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_support_sources_import_no_network_client() -> None:
    offenders: list[str] = []
    for path in SUPPORT_FILES:
        for name in _imported_names(path):
            root = name.split(".")[0]
            if name in FORBIDDEN_IMPORTS or root in {"socket", "requests", "urllib3", "aiohttp"}:
                if name != "urllib.parse":
                    offenders.append(f"{path.name}: {name}")
    assert offenders == [], f"support layer imports a network client: {offenders}"


def test_support_sources_never_open_a_browser_or_url() -> None:
    offenders: list[str] = []
    for path in SUPPORT_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"urlopen", "openUrl", "open_new"}:
                offenders.append(f"{path.name}:{node.lineno} {node.attr}")
    assert offenders == [], f"support layer opens a URL itself: {offenders}"


def test_report_and_package_work_with_sockets_blocked(tmp_path: Path, monkeypatch) -> None:
    def blocked(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("the support layer attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    root = tmp_path / "install"
    root.mkdir()
    report = support.new_report(
        str(root),
        title="Render stops at the last line",
        summary="Nothing was uploaded.",
        run_doctor=False,
    )
    support.save_report(root, report)
    result = support.build_report_package(root, report, run_doctor=False)
    assert result.path.is_file()
    assert result.issue_url.startswith("https://github.com/")
    assert result.mailto_url.startswith("mailto:")


def test_importing_the_support_layer_stays_light() -> None:
    code = (
        "import sys\n"
        "import the_oracle.support\n"
        "heavy = [name for name in ('torch', 'PySide6', 'huggingface_hub', 'transformers') "
        "if name in sys.modules]\n"
        "print('|'.join(heavy))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", f"support import loaded heavy modules: {completed.stdout}"


@pytest.mark.parametrize("path", SUPPORT_FILES, ids=lambda path: path.name)
def test_support_module_has_no_import_time_side_effects(path: Path) -> None:
    """Module bodies only define names: no calls at import time beyond the
    standard dunder assignments Python itself adds."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert calls == [], f"{path.name} executes a call at import time"
