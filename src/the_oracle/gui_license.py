"""License activation and About dialogs (GUI presentation layer).

The GUI half of the licensing unit's step 6 (LICENSING_DESIGN §8): paste-a-
token activation with the typed verification states surfaced readably, and
an About panel that shows which edition this install runs.

Injection discipline (the ``gui_ingest`` pattern): the caller passes
``dialog_cls``/``message_box_cls``; everything else is plain arguments. No
network exists on this path by construction — activation is a paste-and-
verify against embedded keys, offline by construction (LICENSING §6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from the_oracle.licensing import current_license
from the_oracle.licensing import store as license_store
from the_oracle.licensing import tokens as license_tokens

RootPath = Union[str, Path]


def run_activation_dialog(parent: QWidget, root: RootPath, *, dialog_cls: type, message_box_cls: type) -> bool:
    """Paste-a-token activation. Returns True when the store now holds a
    valid license.

    A bad token never writes anything (the store's verify-before-save gate)
    and the typed state is shown verbatim — the same words the CLI and the
    doctor use, so support can ask for exactly this line.
    """
    dialog = dialog_cls(parent)
    dialog.setWindowTitle("Activate The Oracle")
    dialog.setModal(True)
    dialog.resize(560, 320)
    layout = QVBoxLayout(dialog)

    summary = QLabel(
        "Paste the license token you received after purchase.\n\n"
        "Activation is offline: the token is verified against keys embedded\n"
        "in this install — no server is contacted, and nothing is sent\n"
        "anywhere."
    )
    summary.setWordWrap(True)
    layout.addWidget(summary)

    token_view = QPlainTextEdit()
    token_view.setPlaceholderText("ORACLE1.…")
    layout.addWidget(token_view, 1)

    buttons = QDialogButtonBox()
    activate_button = buttons.addButton("Activate", QDialogButtonBox.AcceptRole)
    cancel_button = buttons.addButton(QDialogButtonBox.Cancel)
    if activate_button is not None:
        activate_button.setToolTip("Verify the token and store it locally (offline; nothing is sent).")
    if cancel_button is not None:
        cancel_button.setToolTip("Leave the install as it is — the community edition keeps everything it ships with.")
    layout.addWidget(buttons)

    outcome = {"activated": False}
    token_view.textChanged.connect(
        lambda: activate_button.setEnabled(bool(token_view.toPlainText().strip())) if activate_button is not None else None
    )
    if activate_button is not None:
        activate_button.setEnabled(False)

    def _try_activate() -> None:
        token = token_view.toPlainText().strip()
        if not token:
            return
        verdict = license_tokens.verify_token(token)
        if verdict.ok:
            saved = license_store.save_token(root, token, verify_before_save=license_tokens.verify_token)
            if saved.state == "saved":
                outcome["activated"] = True
                # Done-code 1 (= QDialog.DialogCode.Accepted) semantically:
                # written as the literal because dialog_cls is an injected
                # factory, not necessarily a QDialog subclass with the
                # DialogCode attribute (the injection contract).
                dialog.done(1)
                return
            _show(message_box_cls, dialog, "Could Not Save License", saved.detail or f"unexpected store state: {saved.state}")
            return
        _show(message_box_cls, dialog, f"Token Not Valid ({verdict.state})", verdict.detail)

    activate_button.clicked.connect(_try_activate)
    cancel_button.clicked.connect(dialog.reject)
    dialog.exec()
    return outcome["activated"]


def run_eula_acceptance(parent: QWidget, root: RootPath, *, dialog_cls: type) -> bool:
    """First-run EULA click-through. Returns True only on an explicit Accept.

    Declining (or closing the dialog) records nothing; the caller exits the
    app. Acceptance is recorded against the digest of the exact text shown.
    """
    from the_oracle import eula_acceptance

    current = eula_acceptance.eula_text_and_digest(root)
    if current is None:
        return False
    text, _digest = current

    dialog = dialog_cls(parent)
    dialog.setWindowTitle("The Oracle — License Agreement")
    dialog.setModal(True)
    dialog.resize(640, 560)
    layout = QVBoxLayout(dialog)

    body = QPlainTextEdit()
    body.setReadOnly(True)
    body.setPlainText(text)
    layout.addWidget(body, 1)

    buttons = QDialogButtonBox()
    accept_button = buttons.addButton("I Accept", QDialogButtonBox.AcceptRole)
    decline_button = buttons.addButton("Decline", QDialogButtonBox.RejectRole)
    layout.addWidget(buttons)

    accept_button.clicked.connect(dialog.accept)
    decline_button.clicked.connect(dialog.reject)
    if dialog.exec() != 1:
        return False
    eula_acceptance.record_acceptance(root)
    return True


def _show(message_box_cls, parent, title: str, text: str) -> None:  # noqa: ANN001
    box = message_box_cls(parent)
    box.setWindowTitle(title)
    box.setText(text)
    box.exec()


def run_about_panel(parent: QWidget, *, dialog_cls: type) -> None:
    """The About panel: what this install runs, license first.

    Edition/licensee/key_id/expiry come from the same typed status the CLI
    and the doctor read — one source of truth, no GUI-specific parsing.
    """
    status = current_license()
    lines = ["The Oracle — all-in-one TTS rendering."]
    if status.ok and status.state == "valid":
        licensee_line = "Licensed"
        if status.licensee:
            licensee_line += f" to {status.licensee}"
        lines.append(licensee_line)
        if status.exp:
            lines.append(f"License expires: {status.exp}")
        lines.append(f"License key: {status.key_id}")
    else:
        lines.append("Not activated — enter a product key to unlock the program.")
        if status.state not in ("no_token", ""):
            lines.append(f"(Stored license state: {status.state})")
    lines.append("")
    lines.append("Privacy: no telemetry, no phone-home; crash reports stay on")
    lines.append("your disk and are shared only by you. See PRIVACY.md.")

    dialog = dialog_cls(parent)
    dialog.setWindowTitle("About The Oracle")
    dialog.setModal(True)
    layout = QVBoxLayout(dialog)
    label = QLabel("\n".join(lines))
    label.setWordWrap(True)
    layout.addWidget(label)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok)
    layout.addWidget(buttons)
    buttons.accepted.connect(dialog.accept)
    dialog.exec()
