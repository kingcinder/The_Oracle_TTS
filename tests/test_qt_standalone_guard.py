"""Standalone-run guard for Qt widget tests.

The suite's ``qt_app`` fixtures are per-file, so a test that constructs Qt
widgets without requesting one only survives because an *earlier* file in the
same session created the QApplication — file order becomes a silent
dependency, and standalone the process dies on Qt's abort (SIGABRT) with no
pytest report at all. Proven real: ``test_help_stage_is_first...`` aborted
rc=134 standalone and passed in grouped runs.

These tests pin the conftest guard that converts that class into a loud,
immediate pytest failure instead.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import _called_function_names, widget_builds_in_source

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --- marker scanning (pure) ---------------------------------------------------


def test_direct_widget_construction_is_detected() -> None:
    src = "def test_x():\n    w = QWidget()\n    w.deleteLater()\n"
    assert widget_builds_in_source(src) == "QWidget("


def test_dotted_construction_and_wizard_builders_are_detected() -> None:
    assert widget_builds_in_source("x = QtWidgets.QPushButton()\n") == "QPushButton("
    assert widget_builds_in_source("w = InferenceSetupWizard(parent)\n") == "Wizard("
    assert widget_builds_in_source("win, p = _build_window(mp, tmp)\n") == "_build_window("
    assert widget_builds_in_source("w = MainWindow()\n") == "MainWindow("


def test_widget_names_in_strings_and_comments_do_not_trigger() -> None:
    # Source-scanning tests legitimately quote widget constructors in string
    # literals and comments; those are not constructions.
    src = 'def test_x():\n    text = "build a QWidget( here"\n    assert "QMenu(" in text\n'
    assert widget_builds_in_source(src) is None
    assert widget_builds_in_source("def test_x():\n    # QWidget(\n    pass\n") is None


def test_non_widget_qobjects_and_plain_code_do_not_trigger() -> None:
    # QObject-ish types never abort without an app; only QWidgets do.
    assert widget_builds_in_source("b = QButtonGroup()\n") is None
    assert widget_builds_in_source("i = QTableWidgetItem()\n") is None
    assert widget_builds_in_source("app = QApplication([])\n") is None
    assert widget_builds_in_source("def test_x():\n    assert 1 + 1 == 2\n") is None


def test_called_function_names_extracts_plain_and_attribute_calls() -> None:
    src = (
        "def f(x):\n"
        "    a = helper()\n"
        "    b = mod.other(1)\n"
        "    c = len(x)\n"
        "    d = obj.method()\n"
    )
    names = _called_function_names(ast.parse(src))
    assert {"helper", "other", "len", "method"} <= names


# --- end-to-end: the guard converts abort into a loud failure -----------------


def _run_probe(probe_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", f"tests/{probe_name}", "-q", "-p", "no:cacheprovider"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_widget_build_without_app_fixture_fails_loudly_not_aborts(tmp_path) -> None:
    """The exact regression class: standalone it must be a pytest failure
    with an actionable message, never Qt's silent SIGABRT."""
    probe = _REPO_ROOT / "tests" / "_qt_guard_probe_bad.py"
    try:
        probe.write_text(
            "from PySide6.QtWidgets import QWidget\n"
            "\n"
            "\n"
            "def test_bare_widget_probe():\n"
            "    w = QWidget()\n"
            "    w.deleteLater()\n",
            encoding="utf-8",
        )
        result = _run_probe(probe.name)
    finally:
        probe.unlink(missing_ok=True)
    assert result.returncode != 0, "widget-without-app probe must fail"
    assert "requests no" in result.stdout, result.stdout[-2000:]
    assert "qt_app" in result.stdout, result.stdout[-2000:]
    assert "dumped core" not in result.stdout, "guard must prevent the abort"


def test_widget_build_behind_unmarked_helper_fails_loudly() -> None:
    """The extension under test: a helper the test CALLS (not a named
    builder) constructs the widget — the guard must follow the call and name
    the helper in its message."""
    probe = _REPO_ROOT / "tests" / "_qt_guard_probe_helper.py"
    try:
        probe.write_text(
            "from PySide6.QtWidgets import QWidget\n"
            "\n"
            "\n"
            "def _make_widget():\n"
            "    return QWidget()\n"
            "\n"
            "\n"
            "def test_via_helper_probe():\n"
            "    w = _make_widget()\n"
            "    w.deleteLater()\n",
            encoding="utf-8",
        )
        result = _run_probe(probe.name)
    finally:
        probe.unlink(missing_ok=True)
    assert result.returncode != 0, "helper-hidden widget build must fail"
    assert "_make_widget" in result.stdout, result.stdout[-2000:]
    assert "requests no" in result.stdout, result.stdout[-2000:]
    assert "dumped core" not in result.stdout, "guard must prevent the abort"


def test_widget_build_behind_fixture_parameter_is_caught() -> None:
    """Fixture functions are helpers too: a test whose fixture builds a
    widget, without requesting qt_app, must also fail loudly."""
    probe = _REPO_ROOT / "tests" / "_qt_guard_probe_fixture.py"
    try:
        probe.write_text(
            "import pytest\n"
            "from PySide6.QtWidgets import QWidget\n"
            "\n"
            "\n"
            "@pytest.fixture()\n"
            "def built_widget():\n"
            "    w = QWidget()\n"
            "    yield w\n"
            "    w.deleteLater()\n"
            "\n"
            "\n"
            "def test_fixture_built_probe(built_widget):\n"
            "    assert built_widget is not None\n",
            encoding="utf-8",
        )
        result = _run_probe(probe.name)
    finally:
        probe.unlink(missing_ok=True)
    assert result.returncode != 0, "fixture-built widget must fail"
    assert "built_widget" in result.stdout, result.stdout[-2000:]
    assert "dumped core" not in result.stdout, "guard must prevent the abort"


def test_widget_build_with_app_fixture_passes_standalone() -> None:
    """Control: the same construction with the fixture is green."""
    probe = _REPO_ROOT / "tests" / "_qt_guard_probe_good.py"
    try:
        probe.write_text(
            "import pytest\n"
            "from PySide6.QtWidgets import QApplication, QWidget\n"
            "\n"
            "\n"
            "@pytest.fixture(scope=\"module\")\n"
            "def qt_app():\n"
            "    app = QApplication.instance() or QApplication([])\n"
            "    yield app\n"
            "\n"
            "\n"
            "def test_widget_with_app_probe(qt_app):\n"
            "    w = QWidget()\n"
            "    w.deleteLater()\n",
            encoding="utf-8",
        )
        result = _run_probe(probe.name)
    finally:
        probe.unlink(missing_ok=True)
    assert result.returncode == 0, result.stdout[-2000:]
