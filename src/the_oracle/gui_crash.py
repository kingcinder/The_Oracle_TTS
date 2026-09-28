"""Crash-consent and post-crash review dialogs (GUI presentation layer).

The GUI half of the crash/privacy unit (CRASH_TELEMETRY_DESIGN §11 steps 3–4):
the first-run consent dialog, the post-crash next-session review, and the Qt
message handler the U1.2 core deliberately left unwired (app_gui was the
concurrent engine thread's in-flight surface; this module is the safe home).

Injection discipline (the ``gui_ingest`` pattern): the caller passes the Qt
classes this module instantiates — ``dialog_cls`` (QDialog subclass factory),
``message_box_cls`` — so MainWindow resolves them from its own namespace and
tests can substitute fakes without patching this module. Everything else the
dialogs need arrives as plain arguments: no network, no settings I/O, no
import-time Qt construction.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from the_oracle.crash import bundle, consent


def consent_state(root: str | Path) -> tuple[bool, bool]:
    """(consented, asked) — the two bits the startup flow needs.

    ``asked`` is False only when no consent decision has ever been recorded
    (missing file). A file present with any value counts as asked: the
    fail-closed reader treats malformed as opted-out, and re-asking a user
    who already declined would be consent fatigue.
    """
    path = consent.consent_path(root)
    return consent.read_consent(root), path.exists()


def run_first_run_consent(
    parent: QWidget,
    root: str | Path,
    *,
    dialog_cls: type,
) -> bool:
    """The first-run consent dialog. Returns the user's verdict.

    Three outcomes map to the design contract: enable (write consent True),
    open privacy policy (no state change; the caller opens the document),
    not now (writes consent False — 'no' recorded, never asked again this
    install). Enabling later stays available from the Help menu.
    """
    dialog = dialog_cls(parent)
    dialog.setWindowTitle("Local Crash Reports")
    dialog.setModal(True)
    layout = QVBoxLayout(dialog)

    summary = QLabel(
        "Help improve The Oracle by keeping a record of crashes on this\n"
        "machine?\n\n"
        "Reports stay in this install's crash_reports/ folder — sanitized,\n"
        "capped at 20 small files, and never uploaded. There is no upload\n"
        "code: sharing one is always your deliberate act. Your manuscripts\n"
        "and audio are never included."
    )
    summary.setWordWrap(True)
    layout.addWidget(summary)

    buttons = QHBoxLayout()
    enable_button = QPushButton("Enable Local Reports")
    enable_button.setDefault(True)
    privacy_button = QPushButton("Open Privacy Policy")
    not_now_button = QPushButton("Not Now")
    buttons.addWidget(enable_button)
    buttons.addWidget(privacy_button)
    buttons.addWidget(not_now_button)
    layout.addLayout(buttons)

    choice = {"value": "not_now"}
    enable_button.clicked.connect(lambda: _set_choice(dialog, choice, "enable"))
    privacy_button.clicked.connect(lambda: _set_choice(dialog, choice, "privacy"))
    not_now_button.clicked.connect(lambda: _set_choice(dialog, choice, "not_now"))

    dialog.exec()
    if choice["value"] == "enable":
        consent.write_consent(root, True)
        return True
    if choice["value"] == "not_now":
        consent.write_consent(root, False)
    return False


def _set_choice(dialog, choice: dict, value: str) -> None:  # noqa: ANN001
    choice["value"] = value
    dialog.done(_DIALOG_CODES.get(value, 0))


_DIALOG_CODES = {"enable": 1, "privacy": 2, "not_now": 0}


def run_next_session_review(
    parent: QWidget,
    root: str | Path,
    *,
    dialog_cls: type,
    message_box_cls: type,
    open_path_fn=None,  # noqa: ANN001 - injected: opens a file with the OS viewer
) -> str:
    """The post-crash next-session review. Returns ``"reviewed"``,
    ``"deleted"``, or ``"dismissed"``.

    D8: the notice happens at next session, never a modal at crash time.
    Share = open the sanitized JSON with the OS viewer, then the user copies
    or saves it themselves — there is no upload code anywhere in this flow.
    """
    records = bundle.list_records(root)
    if not records:
        return "dismissed"

    dialog = dialog_cls(parent)
    dialog.setWindowTitle("Crash Report From Last Session")
    dialog.setModal(True)
    dialog.resize(720, 520)
    layout = QVBoxLayout(dialog)

    newest = records[0]
    exception = ""
    detail = ""
    try:
        payload = json.loads(newest.read_text(encoding="utf-8"))
        exception = str(payload.get("exception", {}).get("type", ""))
        detail = str(payload.get("exception", {}).get("message", ""))[:200]
    except (OSError, ValueError):
        detail = "(report unreadable — delete it below)"
    count = len(records)

    summary = QLabel(
        f"The last session captured {count} crash report(s). The most\n"
        f"recent one recorded: {exception or 'an unknown error'}"
        + (f" — {detail}" if detail else "")
        + "\n\n"
        "The report is sanitized (no manuscript or audio content) and has\n"
        "never left this machine. You can review exactly what is in it\n"
        "before deciding to share it with anyone."
    )
    summary.setWordWrap(True)
    layout.addWidget(summary)

    preview = QPlainTextEdit()
    preview.setReadOnly(True)
    try:
        preview.setPlainText(newest.read_text(encoding="utf-8", errors="replace")[:8000])
    except OSError:
        preview.setPlainText("(unreadable)")
    layout.addWidget(preview, 1)

    buttons = QDialogButtonBox()
    review_button = buttons.addButton("Open Report", QDialogButtonBox.ActionRole)
    delete_button = buttons.addButton("Delete All Reports", QDialogButtonBox.DestructiveRole)
    close_button = buttons.addButton(QDialogButtonBox.Close)
    for button, tip in (
        (review_button, "Open the sanitized report in your system viewer — then copy or save it yourself if you choose to share it."),
        (delete_button, "Delete every stored crash report now."),
        (close_button, "Keep the reports; maybe review later."),
    ):
        if button is not None:
            button.setToolTip(tip)
    layout.addWidget(buttons)

    verdict = {"value": "dismissed"}
    if review_button is not None:
        review_button.clicked.connect(
            lambda: _open_report(open_path_fn, newest, verdict)
        )
    if delete_button is not None:
        delete_button.clicked.connect(
            lambda: _delete_all(bundle, root, verdict, dialog, message_box_cls)
        )
    if close_button is not None:
        close_button.clicked.connect(lambda: _set_choice(dialog, verdict, "dismissed"))
    dialog.exec()
    return verdict["value"]


def _open_report(open_path_fn, newest: Path, verdict: dict) -> None:  # noqa: ANN001
    verdict["value"] = "reviewed"
    if open_path_fn is not None:
        try:
            open_path_fn(str(newest))
        except Exception:  # noqa: BLE001 - a failed viewer open must not loop the dialog
            pass


def _delete_all(bundle_module, root, verdict, dialog, message_box_cls) -> None:  # noqa: ANN001
    removed = bundle_module.clear_records(root)
    verdict["value"] = "deleted"
    if message_box_cls is not None:
        box = message_box_cls(dialog)
        box.setWindowTitle("Reports Deleted")
        box.setText(f"Deleted {removed} stored report(s).")
        box.exec()


def maybe_run_startup_flow(
    parent: QWidget,
    *,
    consent_root: str | Path,
    dialog_cls: type,
    message_box_cls: type,
    open_path_fn=None,  # noqa: ANN001
) -> str | None:
    """Startup branch (D8): what, if anything, to show this session.

    Crash reports present → the next-session review (D8). No reports and no
    recorded decision → the first-run consent dialog. Otherwise None: the
    install has already decided and there is nothing to show. Every dialog
    is modal-on-parent but reached through a queued single-shot, so startup
    itself never blocks.
    """
    root = consent_root
    if bundle.list_records(root):
        return run_next_session_review(
            parent, root, dialog_cls=dialog_cls, message_box_cls=message_box_cls, open_path_fn=open_path_fn
        )
    consented, asked = consent_state(root)
    if asked:
        return None
    if run_first_run_consent(parent, root, dialog_cls=dialog_cls):
        from the_oracle.crash import handlers

        handlers.enable_faulthandler_catch(root)
        return "enabled"
    return "declined"


def install_qt_message_handler() -> None:
    """Route Qt's own warning/error channel into the crash pipeline.

    The U1.2 core deliberately left this unwired (app_gui was in flight);
    this module is the safe home. QtFatal/QtCritical are treated as crash-
    class signals: they go through the record path (consent-checked, never
    raising) with the thread name, so a Qt-internal failure leaves the same
    evidence as a Python exception. QtWarning/QtInfo stay out of the crash
    store — warnings are not crashes.

    Idempotent by design: qInstallMessageHandler replaces, and reinstalling
    with the same context-handler idempotence the rest of install() has.
    """
    from PySide6.QtCore import qInstallMessageHandler, QtMsgType

    from the_oracle.crash import handlers

    def _on_qt_message(msg_type, context, message):  # noqa: ANN001
        crash_class = (QtMsgType.QtFatalMsg, QtMsgType.QtCriticalMsg)
        if msg_type not in crash_class:
            return
        handlers._capture(
            exception_type=f"Qt{msg_type.name}",
            exception_message=str(message)[:2000],
            traceback_frames=None,
            thread_name="qt-main",
        )

    qInstallMessageHandler(_on_qt_message)
