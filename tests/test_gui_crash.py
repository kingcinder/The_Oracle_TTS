"""gui_crash: the consent and post-crash review dialogs (U3, CRASH §11 steps 3–4).

The dialogs are Qt-presentation only; every decision (consent writes, purge,
faulthandler arming) goes through the crash package's real API with a sandbox
root, so the tests pin behavior, not pixel dressing. The injection contract
is exercised with fake dialog/message-box classes — the same
``dialog_cls=`` seam MainWindow uses with its own QDialog.

Click-through fakes: the functions under test build REAL QPushButtons and
wire them; the fake dialog's exec() replays a programmed list of button
texts by clicking the matching real child button — so the wiring itself
(lambda -> _set_choice -> done -> verdict -> state change) is what's pinned,
not a stubbed shortcut.

**Mutation contract:** a consent dialog that reports "enable" without writing
consent must fail test_first_run_consent_enable_writes_true; a review dialog
whose delete button does not purge must fail test_review_delete_purges.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtCore import QMessageLogContext, QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from the_oracle import gui_crash
from the_oracle.crash import bundle, consent
from the_oracle.crash import handlers as crash_handlers


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def parent(qt_app):
    widget = QWidget()
    yield widget
    widget.deleteLater()


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


class _FakeDialog(QWidget):
    """Qt-shaped fake whose exec() clicks programmed real child buttons."""

    DialogCode = type("DialogCode", (), {"Accepted": 1, "Rejected": 0})

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.programmed: list[str] = []  # button texts to click at exec()
        self.titles: list[str] = []
        self._done_code: int | None = None

    def setWindowTitle(self, title: str) -> None:
        self.titles.append(title)

    def setModal(self, modal: bool) -> None:
        pass

    def resize(self, w: int, h: int) -> None:
        pass

    def done(self, code: int) -> None:
        self._done_code = code

    def accept(self) -> None:
        self.done(1)

    def reject(self) -> None:
        self.done(0)

    def exec(self) -> int:  # noqa: A003 - Qt API
        for text in self.programmed:
            wanted = text.replace("&", "")
            clicked = False
            for button in self.findChildren(QPushButton):
                if button.text().replace("&", "") == wanted:
                    button.click()
                    clicked = True
                    break
            assert clicked, f"programmed button not found: {text!r}"
        return self._done_code if self._done_code is not None else 0


class _FakeMessageBox(_FakeDialog):
    def setIcon(self, *_args, **_kwargs) -> None:
        pass

    def setText(self, text: str) -> None:
        self.titles.append(text)


def _plant_record(root: Path, exception_type: str = "SegmentationFault") -> Path:
    return bundle.write_record(
        root,
        {
            "record_version": 1,
            "exception": {"type": exception_type, "message": "boom"},
            "traceback": [],
            "log_tail": [],
            "edition": "community",
        },
    )


# --- consent_state ------------------------------------------------------------


def test_consent_state_never_asked_is_false_false(root: Path) -> None:
    assert gui_crash.consent_state(root) == (False, False)


def test_consent_state_after_decline_is_false_true(root: Path) -> None:
    consent.write_consent(root, False)
    assert gui_crash.consent_state(root) == (False, True)


def test_consent_state_after_optin_is_true_true(root: Path) -> None:
    consent.write_consent(root, True)
    assert gui_crash.consent_state(root) == (True, True)


# --- first-run consent dialog ---------------------------------------------------


def test_first_run_consent_enable_writes_true(root: Path, parent) -> None:
    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Enable Local Reports"]
        return dialog

    assert gui_crash.run_first_run_consent(parent, root, dialog_cls=factory) is True
    assert consent.read_consent(root) is True


def test_first_run_consent_not_now_records_decline(root: Path, parent) -> None:
    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Not Now"]
        return dialog

    assert gui_crash.run_first_run_consent(parent, root, dialog_cls=factory) is False
    # Fail-closed either way, but the decision IS recorded: never asked again.
    assert consent.read_consent(root) is False
    assert gui_crash.consent_state(root) == (False, True)


def test_first_run_consent_privacy_leaves_the_decision_open(root: Path, parent) -> None:
    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Open Privacy Policy"]
        return dialog

    assert gui_crash.run_first_run_consent(parent, root, dialog_cls=factory) is False
    assert gui_crash.consent_state(root) == (False, False)  # still never asked


# --- next-session review --------------------------------------------------------


def test_review_without_records_never_opens_a_dialog(root: Path, parent) -> None:
    def factory(parent_widget):
        raise AssertionError("no dialog may open when there is nothing to review")

    assert (
        gui_crash.run_next_session_review(parent, root, dialog_cls=factory, message_box_cls=_FakeMessageBox)
        == "dismissed"
    )


def test_review_delete_purges(root: Path, parent) -> None:
    _plant_record(root)

    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Delete All Reports"]
        return dialog

    verdict = gui_crash.run_next_session_review(
        parent, root, dialog_cls=factory, message_box_cls=_FakeMessageBox
    )
    assert verdict == "deleted"
    assert bundle.list_records(root) == []


def test_review_open_report_hands_the_file_to_the_viewer(root: Path, parent) -> None:
    record_path = _plant_record(root)
    opened: list[str] = []

    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Open Report"]
        return dialog

    verdict = gui_crash.run_next_session_review(
        parent, root, dialog_cls=factory, message_box_cls=_FakeMessageBox, open_path_fn=opened.append
    )
    assert verdict == "reviewed"
    assert opened == [str(record_path)]


def test_review_close_keeps_the_reports(root: Path, parent) -> None:
    _plant_record(root)

    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Close"]
        return dialog

    assert (
        gui_crash.run_next_session_review(parent, root, dialog_cls=factory, message_box_cls=_FakeMessageBox)
        == "dismissed"
    )
    assert bundle.list_records(root), "close must not purge"


def test_review_summary_names_the_exception(root: Path, parent) -> None:
    _plant_record(root, "SegmentationFault")
    labels: list[str] = []

    real_label = gui_crash.QLabel

    class _RecordingLabel(real_label):
        def __init__(self, text: str = "", *a, **k) -> None:
            super().__init__(text, *a, **k)
            labels.append(text)

    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Close"]
        return dialog

    import the_oracle.gui_crash as gc

    gc.QLabel = _RecordingLabel
    try:
        gui_crash.run_next_session_review(parent, root, dialog_cls=factory, message_box_cls=_FakeMessageBox)
    finally:
        gc.QLabel = real_label
    assert any("SegmentationFault" in text for text in labels), labels


# --- the startup branch ----------------------------------------------------------


def test_startup_flow_shows_consent_only_when_never_asked(root: Path, parent) -> None:
    shown: list[str] = []

    def consent_factory(parent_widget):
        shown.append("consent")
        dialog = _FakeDialog()
        dialog.programmed = ["Not Now"]
        return dialog

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(gui_crash, "run_next_session_review", lambda *a, **k: shown.append("review"))
        assert (
            gui_crash.maybe_run_startup_flow(parent, consent_root=root, dialog_cls=consent_factory, message_box_cls=_FakeMessageBox)
            == "declined"
        )
        assert shown == ["consent"]

        # A declined install is never asked again.
        shown.clear()
        assert (
            gui_crash.maybe_run_startup_flow(parent, consent_root=root, dialog_cls=consent_factory, message_box_cls=_FakeMessageBox)
            is None
        )
        assert shown == []
    finally:
        monkey.undo()


def test_startup_flow_prefers_the_review_when_reports_exist(root: Path, parent) -> None:
    _plant_record(root)
    consent.write_consent(root, True)  # a consented install that crashed

    def review_factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Close"]
        return dialog

    result = gui_crash.maybe_run_startup_flow(
        parent, consent_root=root, dialog_cls=review_factory, message_box_cls=_FakeMessageBox
    )
    assert result == "dismissed"
    assert bundle.list_records(root)


def test_startup_flow_enable_arms_faulthandler(root: Path, parent) -> None:
    def consent_factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Enable Local Reports"]
        return dialog

    result = gui_crash.maybe_run_startup_flow(
        parent, consent_root=root, dialog_cls=consent_factory, message_box_cls=_FakeMessageBox
    )
    assert result == "enabled"
    assert consent.read_consent(root) is True
    assert crash_handlers._STATE.get("native_dump_path") is not None
    crash_handlers.disable_faulthandler_catch()


def test_plain_relaunch_with_consent_on_arms_faulthandler(root: Path, parent) -> None:
    """The already-asked-and-consented relaunch (no reports, no dialogs) must
    still come up armed — the 08:06:51 GUI segfault left native-crash.txt at
    0 bytes because this branch returned None without arming (STATE.md
    Noticed, 2026-09-28). M: dropping the arm at the top of
    maybe_run_startup_flow fails exactly this test."""
    consent.write_consent(root, True)  # consented AND asked; no records on disk

    def unexpected_dialog(parent_widget):  # a plain relaunch shows nothing
        raise AssertionError("no dialog should be shown on a plain relaunch")

    result = gui_crash.maybe_run_startup_flow(
        parent, consent_root=root, dialog_cls=unexpected_dialog, message_box_cls=_FakeMessageBox
    )
    assert result is None
    assert crash_handlers._STATE.get("native_dump_handle") is not None
    crash_handlers.disable_faulthandler_catch()


def test_launch_gui_arms_before_mainwindow_is_built() -> None:
    """Source pin (same convention as tests/test_render_subprocess_arming.py):
    launch_gui is a public entry reachable without cli.main, so it must arm
    the net itself — before MainWindow() is constructed, not after (the
    window-build phase is native-crash territory; see the 08:06:51 segfault)."""
    from the_oracle import app_gui

    source = Path(app_gui.__file__).read_text(encoding="utf-8")
    assert "crash_handlers.arm_native_capture()" in source
    assert source.index("crash_handlers.arm_native_capture()") < source.index(
        "window = MainWindow()"
    )


# --- the Qt message handler --------------------------------------------------------


def test_qt_message_handler_records_crash_class_only(root: Path) -> None:
    consent.write_consent(root, True)
    crash_handlers.install(root)
    gui_crash.install_qt_message_handler()
    try:
        # Retrieve the just-installed handler: installing None returns the
        # previous one (PySide6 contract).
        oracle_handler = qInstallMessageHandler(None)
        assert callable(oracle_handler)
        context = QMessageLogContext()

        oracle_handler(QtMsgType.QtFatalMsg, context, "native fatal from Qt")
        assert not bundle.list_records(root) or json.loads(
            bundle.list_records(root)[0].read_text(encoding="utf-8")
        )["exception"]["type"].startswith("Qt")
        oracle_handler(QtMsgType.QtWarningMsg, context, "a warning must not become a crash record")
        records = bundle.list_records(root)
        assert len(records) == 1, "only the fatal message is crash-class"
        payload = json.loads(records[0].read_text(encoding="utf-8"))
        assert payload["exception"]["type"].startswith("Qt")
        assert "native fatal from Qt" in payload["exception"]["message"]
    finally:
        crash_handlers.disable_faulthandler_catch()


def test_qt_message_handler_is_installable_repeatedly() -> None:
    # qInstallMessageHandler replaces; a second install must not raise.
    gui_crash.install_qt_message_handler()
    gui_crash.install_qt_message_handler()
    qInstallMessageHandler(None)
