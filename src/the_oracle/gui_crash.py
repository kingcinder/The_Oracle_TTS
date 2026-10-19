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
    QApplication,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
    # Arm on EVERY launch when consent is already on, not only on consent
    # transitions (STATE.md Noticed, 2026-09-28: the 08:06:51 GUI segfault
    # ran unarmed because a plain relaunch reached the ``asked`` branch
    # below without ever arming). Idempotent and fail-closed: consent-off
    # installs get no capture machinery, a consent transition arm later in
    # this function re-hits the armed early-return.
    from the_oracle.crash import handlers

    handlers.arm_native_capture(root)
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


def _steps_from_text(text: str) -> list[str]:
    """One reproduction step per non-empty line, numbering stripped so a user
    can paste an existing list."""
    steps: list[str] = []
    for line in str(text or "").splitlines():
        value = line.strip().lstrip("0123456789.)-• ").strip()
        if value:
            steps.append(value)
    return steps


def _inform(message_box_cls, parent, title: str, text: str) -> None:  # noqa: ANN001
    if message_box_cls is None:
        return
    box = message_box_cls(parent)
    box.setWindowTitle(title)
    box.setText(text)
    box.exec()


def _report_fields(
    root: str | Path,
    *,
    title: str,
    summary: str,
    steps_text: str,
    expected: str,
    actual: str,
    area: str,
    severity: str,
    contact: str,
    crash_record: str | None,
    include_diagnostics: bool,
    run_doctor: bool,
):  # noqa: ANN001, ANN202 - returns a support.BugReport
    """The one mapping from dialog text to the report model. Kept pure so the
    dialog's wiring can be tested without a second copy of these rules."""
    from the_oracle.support import new_report

    return new_report(
        str(root),
        title=title,
        summary=summary,
        steps=_steps_from_text(steps_text),
        expected=expected,
        actual=actual,
        area=area,
        severity=severity,
        contact=contact,
        crash_record=crash_record,
        include_diagnostics=include_diagnostics,
        run_doctor=run_doctor,
    )


def run_problem_report_dialog(
    parent: QWidget,
    root: str | Path,
    *,
    dialog_cls: type,
    message_box_cls: type,
    open_path_fn=None,  # noqa: ANN001 - injected: opens a folder/file with the OS
    open_url_fn=None,  # noqa: ANN001 - injected: opens a browser/mail client
    copy_text_fn=None,  # noqa: ANN001 - injected: clipboard writer
    run_doctor: bool = False,
) -> str:
    """Compose, preview, save, and package a bug report. Returns ``"saved"``,
    ``"packaged"``, or ``"closed"``.

    Nothing is sent from here. "Copy Issue Link" and "Email Support" hand a
    prefilled link to the injected opener, which is the user clicking a
    button; there is no transport code in the support package. The dialog is
    Qt-presentation only — every rule (validation, redaction, consent gating,
    storage) lives in :mod:`the_oracle.support` and the crash unit.
    """
    from the_oracle import crash as crash_module
    from the_oracle.support import (
        AREAS,
        SEVERITIES,
        build_report_package,
        issue_url,
        mailto_url,
        report_markdown,
        save_report,
    )

    records = crash_module.list_records(root)
    newest = records[0].name if records else None
    consented = crash_module.read_consent(root)

    dialog = dialog_cls(parent)
    dialog.setWindowTitle("Report a Problem")
    dialog.setModal(True)
    dialog.resize(820, 760)
    layout = QVBoxLayout(dialog)

    intro = QLabel(
        "Describe what happened. The report is saved on this machine and is\n"
        "only shared when you choose to share it. Manuscripts, audio, voice\n"
        "recordings, tokens, and home-directory paths are never included."
    )
    intro.setWordWrap(True)
    layout.addWidget(intro)

    form = QFormLayout()
    title_edit = QLineEdit()
    title_edit.setObjectName("title")
    title_edit.setPlaceholderText("Render stops at the last line")
    area_combo = QComboBox()
    area_combo.setObjectName("area")
    area_combo.addItems(list(AREAS))
    severity_combo = QComboBox()
    severity_combo.setObjectName("severity")
    severity_combo.addItems(list(SEVERITIES))
    severity_combo.setCurrentText("normal")
    summary_edit = QPlainTextEdit()
    summary_edit.setObjectName("summary")
    summary_edit.setFixedHeight(68)
    steps_edit = QPlainTextEdit()
    steps_edit.setObjectName("steps")
    steps_edit.setFixedHeight(88)
    steps_edit.setPlaceholderText("One step per line, in the order they happen")
    expected_edit = QLineEdit()
    expected_edit.setObjectName("expected")
    actual_edit = QLineEdit()
    actual_edit.setObjectName("actual")
    contact_edit = QLineEdit()
    contact_edit.setObjectName("contact")
    contact_edit.setPlaceholderText("Optional — how support should reply")
    for label, widget in (
        ("Title", title_edit),
        ("Area", area_combo),
        ("Severity", severity_combo),
        ("Summary", summary_edit),
        ("Steps to reproduce", steps_edit),
        ("Expected behaviour", expected_edit),
        ("Actual behaviour", actual_edit),
        ("Contact", contact_edit),
    ):
        form.addRow(label, widget)
    layout.addLayout(form)

    crash_check = QCheckBox(
        f"Reference the newest crash report ({newest})"
        if newest
        else "No local crash report to reference"
    )
    crash_check.setObjectName("attach_crash")
    crash_check.setEnabled(bool(newest))
    crash_check.setChecked(bool(newest) and consented)
    crash_check.setToolTip(
        "Adds the crash record's name to the report. The record itself is only "
        "packaged when local crash-reporting consent is on."
    )
    diagnostics_check = QCheckBox("Add the redacted diagnostics section to the report")
    diagnostics_check.setObjectName("include_diagnostics")
    diagnostics_check.setChecked(True)
    diagnostics_check.setToolTip(
        "Version, platform, CPU/memory/disk, engine availability, settings "
        "summary, and the doctor verdict. No paths, usernames, tokens, or "
        "file contents."
    )
    layout.addWidget(crash_check)
    layout.addWidget(diagnostics_check)

    layout.addWidget(QLabel("Preview (what you would be sharing):"))
    preview = QPlainTextEdit()
    preview.setObjectName("preview")
    preview.setReadOnly(True)
    preview.setFixedHeight(200)
    layout.addWidget(preview)

    buttons = QHBoxLayout()
    preview_button = QPushButton("Preview")
    preview_button.setObjectName("preview_button")
    save_button = QPushButton("Save Draft")
    save_button.setObjectName("save_button")
    save_button.setDefault(True)
    package_button = QPushButton("Save && Package…")
    package_button.setObjectName("package_button")
    issue_button = QPushButton("Copy Issue Link")
    issue_button.setObjectName("issue_button")
    email_button = QPushButton("Email Support")
    email_button.setObjectName("email_button")
    close_button = QPushButton("Close")
    close_button.setObjectName("close_button")
    for button in (
        preview_button,
        save_button,
        package_button,
        issue_button,
        email_button,
        close_button,
    ):
        buttons.addWidget(button)
    layout.addLayout(buttons)

    state: dict[str, object] = {"report": None, "outcome": "closed"}

    def build(*, save: bool):  # noqa: ANN202 - BugReport | None
        try:
            report = _report_fields(
                root,
                title=title_edit.text(),
                summary=summary_edit.toPlainText(),
                steps_text=steps_edit.toPlainText(),
                expected=expected_edit.text(),
                actual=actual_edit.text(),
                area=area_combo.currentText(),
                severity=severity_combo.currentText(),
                contact=contact_edit.text(),
                crash_record=newest if crash_check.isChecked() else None,
                include_diagnostics=diagnostics_check.isChecked(),
                run_doctor=run_doctor,
            )
        except ValueError as error:
            _inform(message_box_cls, dialog, "Report Not Saved", str(error))
            return None
        state["report"] = report
        preview.setPlainText(report_markdown(report)[:12000])
        if save:
            save_report(root, report)
        return report

    def on_preview() -> None:
        build(save=False)

    def on_save() -> None:
        report = build(save=True)
        if report is None:
            return
        state["outcome"] = "saved"
        _inform(
            message_box_cls,
            dialog,
            "Draft Saved",
            f"Saved report {report.report_id} on this machine.\n\n"
            "Nothing was sent. Use Copy Issue Link or Save && Package… when "
            "you are ready to share it.",
        )
        dialog.done(0)

    def on_package() -> None:
        report = build(save=True)
        if report is None:
            return
        try:
            result = build_report_package(
                root,
                report,
                include_crash_reports=crash_check.isChecked(),
                refresh_diagnostics=False,
                run_doctor=run_doctor,
            )
        except (OSError, ValueError) as error:
            _inform(message_box_cls, dialog, "Package Failed", str(error))
            return
        state["outcome"] = "packaged"
        crash_note = (
            "Crash records were included (local consent is on)."
            if result.crash_reports_included
            else "No crash records were included."
        )
        _inform(
            message_box_cls,
            dialog,
            "Support Package Created",
            f"Saved to:\n{result.path}\n\n{crash_note}\n\n"
            "Nothing was uploaded. Review the archive, then attach it to the "
            "issue or email you send.",
        )
        if open_path_fn is not None:
            try:
                open_path_fn(str(result.path.parent))
            except Exception:  # noqa: BLE001 - a failed viewer open must not lose the package
                pass
        dialog.done(0)

    def on_issue() -> None:
        report = build(save=True)
        if report is None:
            return
        state["outcome"] = "saved"
        url = issue_url(report)
        copied = False
        try:
            if copy_text_fn is not None:
                copy_text_fn(url)
            else:
                QApplication.clipboard().setText(url)
            copied = True
        except Exception:  # noqa: BLE001 - no clipboard is a fallback, not a failure
            copied = False
        _inform(
            message_box_cls,
            dialog,
            "Issue Link Copied" if copied else "Issue Link",
            "The prefilled issue link is on your clipboard. Open it, review it, "
            "and attach the support package."
            if copied
            else url,
        )
        dialog.done(0)

    def on_email() -> None:
        report = build(save=True)
        if report is None:
            return
        state["outcome"] = "saved"
        url = mailto_url(report)
        if open_url_fn is not None:
            try:
                open_url_fn(url)
            except Exception:  # noqa: BLE001 - a missing mail client is not a crash
                _inform(message_box_cls, dialog, "Support Email", url)
        else:
            _inform(message_box_cls, dialog, "Support Email", url)
        dialog.done(0)

    preview_button.clicked.connect(on_preview)
    save_button.clicked.connect(on_save)
    package_button.clicked.connect(on_package)
    issue_button.clicked.connect(on_issue)
    email_button.clicked.connect(on_email)
    close_button.clicked.connect(lambda: dialog.done(0))

    dialog.exec()
    return str(state["outcome"])


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
