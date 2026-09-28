"""gui_license: activation and About dialogs (U3, LICENSING §8 step 6).

Click-through fakes (see test_gui_crash.py's docstring): the dialogs build
REAL QPushButtons; the fake exec() clicks programmed texts, so the wiring —
token read, verify, save-gate, typed-state surfacing — is what's pinned.
The license root is sandboxed like the CLI flow tests (offline.repo_root
monkeypatched where the production handler resolves it; the dialogs take it
as a plain argument).

**Mutation contract:** an activation that stores a token WITHOUT verifying
must fail test_activation_bad_token_writes_nothing (the save gate is the
privacy-critical ordering).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from the_oracle import gui_license
from the_oracle.licensing import store, tokens

SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")


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
    DialogCode = type("DialogCode", (), {"Accepted": 1, "Rejected": 0})

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.programmed: list[str] = []
        self.text_to_type: str | None = None
        self.titles: list[str] = []
        self._done_code: int | None = None

    def setWindowTitle(self, title: str) -> None:
        self.titles.append(title)

    def setModal(self, modal: bool) -> None:
        pass

    def resize(self, w: int, h: int) -> None:
        pass

    def setPlaceholderText(self, text: str) -> None:
        pass

    def done(self, code: int) -> None:
        self._done_code = code

    def accept(self) -> None:
        self.done(1)

    def reject(self) -> None:
        self.done(0)

    def exec(self) -> int:  # noqa: A003
        # The real QPlainTextEdit's textChanged signal enables the Activate
        # button once text exists; the fake token view cannot fire it, so
        # simulate the typed state here (the load-bearing guards — empty-text
        # refusal and the save gate — live in _try_activate, not the button).
        for button in self.findChildren(QPushButton):
            button.setEnabled(True)
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


class _FakeTokenView(QWidget):
    """Stands in for the token QPlainTextEdit: records program, replays text."""

    def __init__(self) -> None:
        super().__init__()
        self._text = ""
        self.toPlainText = lambda: self._text
        self.textChanged = type("S", (), {"connect": staticmethod(lambda slot: None)})()

    def set_text(self, text: str) -> None:
        self._text = text


class _FakeMessageBox(_FakeDialog):
    def setText(self, text: str) -> None:
        self.titles.append(text)


def _good_token(exp: int | None = None) -> str:
    payload = {
        "v": 1,
        "key_id": "k1",
        "lic_id": "55555555-5555-4555-5555-555555555555",
        "edition": "studio",
        "licensee": "Cody",
        "machine_hash": "",
        "iat": 1700000000,
        "exp": exp,
    }
    return tokens.mint(payload, SEED)


def _patch_token_view(monkeypatch: pytest.MonkeyPatch, token: str) -> None:
    """Route the dialog's token view to a fake whose toPlainText yields the
    token the Activate click should try."""

    real_view = gui_license.QPlainTextEdit

    class _View(QWidget):
        def __init__(self, *a, **k) -> None:
            super().__init__()
            self._text = token

        def toPlainText(self) -> str:  # noqa: N802 - Qt API
            return self._text

        def setPlaceholderText(self, text: str) -> None:  # noqa: N802
            pass

        def setReadOnly(self, value: bool) -> None:  # noqa: N802
            pass

        def setPlainText(self, text: str) -> None:  # noqa: N802
            self._text = text

    # textChanged.connect must exist; give it a no-op signal shim.
    class _Signal:
        def connect(self, slot) -> None:
            pass

    _View.textChanged = _Signal()
    monkeypatch.setattr(gui_license, "QPlainTextEdit", _View)


def test_activation_good_token_stores_and_reports_true(root: Path, parent, monkeypatch) -> None:
    _patch_token_view(monkeypatch, _good_token())

    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Activate"]
        return dialog

    assert gui_license.run_activation_dialog(parent, root, dialog_cls=factory, message_box_cls=_FakeMessageBox) is True
    result = store.load_token(root)
    assert result.state == "loaded"
    # StoredToken carries the raw token; verify its payload says studio.
    from the_oracle.licensing import tokens as _tokens

    verdict = _tokens.verify_token(result.stored.token)
    assert verdict.ok and verdict.edition == "studio"


def test_activation_bad_token_writes_nothing_and_names_the_state(root: Path, parent, monkeypatch) -> None:
    _patch_token_view(monkeypatch, "ORACLE1.not-a-real-token")

    shown: list[tuple[str, str]] = []

    def message_factory(parent_widget):
        box = _FakeMessageBox()
        return box

    real_show = gui_license._show

    def recording_show(message_box_cls, parent_widget, title, text):
        shown.append((title, text))
        return None

    monkeypatch.setattr(gui_license, "_show", recording_show)

    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Activate"]
        return dialog

    assert gui_license.run_activation_dialog(parent, root, dialog_cls=factory, message_box_cls=message_factory) is False
    assert store.load_token(root).state == "no_token", "a bad token must never be stored"
    assert shown, "the typed failure state must be surfaced"
    assert shown[0][0].startswith("Token Not Valid")


def test_activation_empty_token_refuses_to_activate(root: Path, parent, monkeypatch) -> None:
    _patch_token_view(monkeypatch, "   ")
    activated: list[bool] = []

    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Activate"]
        return dialog

    assert gui_license.run_activation_dialog(parent, root, dialog_cls=factory, message_box_cls=_FakeMessageBox) is False
    assert store.load_token(root).state == "no_token"


def test_activation_cancel_writes_nothing(root: Path, parent, monkeypatch) -> None:
    _patch_token_view(monkeypatch, _good_token())

    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["Cancel"]
        return dialog

    assert gui_license.run_activation_dialog(parent, root, dialog_cls=factory, message_box_cls=_FakeMessageBox) is False
    assert store.load_token(root).state == "no_token"


def test_about_panel_names_edition_and_privacy(root: Path, parent, monkeypatch) -> None:
    labels: list[str] = []
    real_label = gui_license.QLabel

    class _RecordingLabel(real_label):
        def __init__(self, text: str = "", *a, **k) -> None:
            super().__init__(text, *a, **k)
            labels.append(text)

    monkeypatch.setattr(gui_license, "QLabel", _RecordingLabel)

    def factory(parent_widget):
        dialog = _FakeDialog()
        dialog.programmed = ["OK"]
        return dialog

    # Unlicensed install: community with everything included.
    gui_license.run_about_panel(parent, dialog_cls=factory)
    assert any("community" in text for text in labels), labels
    assert any("PRIVACY.md" in text for text in labels), labels

    # Activated install: names the edition and the licensee.
    labels.clear()
    store.save_token(root, _good_token(), verify_before_save=tokens.verify_token)
    from the_oracle.licensing import current_license as _real_current

    def fake_current_license():
        from the_oracle.licensing.tokens import LicenseStatus

        return LicenseStatus(
            ok=True, state="valid", edition="studio", licensee="Cody",
            lic_id="x", key_id="k1", exp=None, machine_locked=False, detail="",
        )

    monkeypatch.setattr(gui_license, "current_license", fake_current_license)
    gui_license.run_about_panel(parent, dialog_cls=factory)
    assert any("studio" in text and "Cody" in text for text in labels), labels
