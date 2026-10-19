"""The GUI Report a Problem dialog: wiring, not pixel dressing.

The dialog is Qt-presentation only; all rules live in ``the_oracle.support``.
These tests drive the real widgets (fields, checkboxes, buttons) on an
offscreen QApplication and pin the three outcomes the Help menu can produce:
a saved draft, a packaged zip, and a copied issue link. Validation failures
must leave nothing on disk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QWidget,
)

from the_oracle import crash, gui_crash, support


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def parent(qt_app):
    widget = QWidget()
    yield widget
    widget.deleteLater()


class _MessageBox:
    instances: list["_MessageBox"] = []

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        self.parent = parent
        self.title = ""
        self.text = ""
        _MessageBox.instances.append(self)

    def setWindowTitle(self, title: str) -> None:  # noqa: N802 - Qt name
        self.title = title

    def setText(self, text: str) -> None:  # noqa: N802 - Qt name
        self.text = text

    def exec(self) -> int:
        return 0


class _AutoDialog(QDialog):
    """Fills the real fields, then clicks one real button."""

    action = "save_button"
    fields: dict[str, str] = {}
    choices: dict[str, str] = {}

    def exec(self) -> int:
        for name, value in self.fields.items():
            widget = self.findChild(QLineEdit, name) or self.findChild(QPlainTextEdit, name)
            assert widget is not None, f"missing field {name}"
            if isinstance(widget, QPlainTextEdit):
                widget.setPlainText(value)
            else:
                widget.setText(value)
        for name, value in self.choices.items():
            combo = self.findChild(QComboBox, name)
            assert combo is not None, f"missing picker {name}"
            combo.setCurrentText(value)
        button = self.findChild(QPushButton, self.action)
        assert button is not None, f"missing button {self.action}"
        button.click()
        return 1


def _dialog(action: str, fields: dict[str, str] | None = None, choices: dict[str, str] | None = None):
    return type(
        "AutoDialog",
        (_AutoDialog,),
        {
            "action": action,
            "fields": fields if fields is not None else {},
            "choices": choices or {},
        },
    )


def _base_fields() -> dict[str, str]:
    return {
        "title": "Preview closes the window",
        "summary": "Clicking Preview exits the app.",
        "steps": "1. Open a project\n2. Click Preview",
        "expected": "A preview render",
        "actual": "The app closes",
        "contact": "writer@example.com",
    }


@pytest.fixture(autouse=True)
def _clear_messages():
    _MessageBox.instances.clear()


def test_steps_text_is_split_and_numbering_stripped() -> None:
    assert gui_crash._steps_from_text("1. Open\n\n 2) Click\n- Wait") == ["Open", "Click", "Wait"]


def test_save_writes_a_local_draft(parent, tmp_path: Path) -> None:
    dialog_cls = _dialog("save_button", _base_fields(), {"area": "gui", "severity": "normal"})
    outcome = gui_crash.run_problem_report_dialog(
        parent,
        tmp_path,
        dialog_cls=dialog_cls,
        message_box_cls=_MessageBox,
        run_doctor=False,
    )
    assert outcome == "saved"
    reports = support.list_reports(tmp_path)
    assert len(reports) == 1
    report = reports[0]
    assert report.title == "Preview closes the window"
    assert report.steps == ["Open a project", "Click Preview"]
    assert report.area == "gui"
    assert report.severity == "normal"
    assert report.contact == "writer@example.com"
    assert report.diagnostics
    assert "Draft Saved" in _MessageBox.instances[-1].title
    assert report.report_id in _MessageBox.instances[-1].text


def test_package_writes_a_zip_under_output(parent, tmp_path: Path) -> None:
    dialog_cls = _dialog("package_button", _base_fields())
    opened: list[str] = []
    outcome = gui_crash.run_problem_report_dialog(
        parent,
        tmp_path,
        dialog_cls=dialog_cls,
        message_box_cls=_MessageBox,
        open_path_fn=opened.append,
        run_doctor=False,
    )
    assert outcome == "packaged"
    report = support.list_reports(tmp_path)[0]
    package = tmp_path / "Output" / f"oracle-report-{report.report_id}.zip"
    assert package.is_file()
    assert opened == [str(package.parent)]
    assert "Support Package Created" in _MessageBox.instances[-1].title


def test_issue_copies_a_prefilled_link(parent, tmp_path: Path) -> None:
    copied: list[str] = []
    dialog_cls = _dialog("issue_button", _base_fields())
    outcome = gui_crash.run_problem_report_dialog(
        parent,
        tmp_path,
        dialog_cls=dialog_cls,
        message_box_cls=_MessageBox,
        copy_text_fn=copied.append,
        run_doctor=False,
    )
    assert outcome == "saved"
    assert copied and copied[0].startswith("https://github.com/")
    report = support.list_reports(tmp_path)[0]
    assert copied[0] == support.issue_url(report)
    assert "Copied" in _MessageBox.instances[-1].title


def test_email_opens_the_prefilled_mailto(parent, tmp_path: Path) -> None:
    opened: list[str] = []
    dialog_cls = _dialog("email_button", _base_fields())
    outcome = gui_crash.run_problem_report_dialog(
        parent,
        tmp_path,
        dialog_cls=dialog_cls,
        message_box_cls=_MessageBox,
        open_url_fn=opened.append,
        run_doctor=False,
    )
    assert outcome == "saved"
    assert opened and opened[0].startswith("mailto:")


def test_short_title_is_refused_and_writes_nothing(parent, tmp_path: Path) -> None:
    fields = _base_fields()
    fields["title"] = "bad"
    dialog_cls = _dialog("save_button", fields)
    outcome = gui_crash.run_problem_report_dialog(
        parent,
        tmp_path,
        dialog_cls=dialog_cls,
        message_box_cls=_MessageBox,
        run_doctor=False,
    )
    assert outcome == "closed"
    assert support.list_reports(tmp_path) == []
    assert "Not Saved" in _MessageBox.instances[-1].title
    assert "at least" in _MessageBox.instances[-1].text


def test_crash_checkbox_references_the_newest_record_only_with_consent(parent, tmp_path: Path) -> None:
    crash.write_consent(tmp_path, True)
    crash.write_record(
        tmp_path, crash.build_record(exception_type="RuntimeError", exception_message="boom")
    )
    dialog_cls = _dialog("save_button", _base_fields())
    outcome = gui_crash.run_problem_report_dialog(
        parent,
        tmp_path,
        dialog_cls=dialog_cls,
        message_box_cls=_MessageBox,
        run_doctor=False,
    )
    assert outcome == "saved"
    report = support.list_reports(tmp_path)[0]
    assert report.crash_record == crash.list_records(tmp_path)[0].name


def test_diagnostics_checkbox_keeps_the_snapshot_out(parent, tmp_path: Path) -> None:
    class _NoDiagnosticsDialog(_AutoDialog):
        action = "save_button"
        fields = _base_fields()

        def exec(self) -> int:  # noqa: D102 - fill, uncheck, save
            for name, value in self.fields.items():
                widget = self.findChild(QLineEdit, name) or self.findChild(QPlainTextEdit, name)
                if isinstance(widget, QPlainTextEdit):
                    widget.setPlainText(value)
                elif widget is not None:
                    widget.setText(value)
            from PySide6.QtWidgets import QCheckBox

            self.findChild(QCheckBox, "include_diagnostics").setChecked(False)
            self.findChild(QPushButton, "save_button").click()
            return 1

    outcome = gui_crash.run_problem_report_dialog(
        parent,
        tmp_path,
        dialog_cls=_NoDiagnosticsDialog,
        message_box_cls=_MessageBox,
        run_doctor=False,
    )
    assert outcome == "saved"
    assert support.list_reports(tmp_path)[0].diagnostics == {}
