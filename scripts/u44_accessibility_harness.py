"""U4.4 accessibility harness: the two new dialogs, offscreen and scripted.

Runs the REAL dialog code - gui_crash.run_first_run_consent /
run_next_session_review and gui_license.run_activation_dialog /
run_about_panel - under QT_QPA_PLATFORM=offscreen through their own
injection seam (dialog_cls / message_box_cls), the same click-through-fake
pattern tests/test_gui_crash.py uses: the injected dialog's exec() dumps
the accessibility-relevant tree (classes/roles, accessible names, text,
placeholders, tooltips, focus policies, default button), walks the real
Tab-focus chain, performs scripted actions (TYPE:<text> into the first
editable text widget, or click a button by its exact text), and returns
without ever entering a blocking event loop.

Every write (consent files, seeded crash records, the license store) goes
to a throwaway sandbox root; the real user config is redirected away via
XDG_CONFIG_HOME. Read-only audit tool: prints one JSON document, edits no
code.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("XDG_CONFIG_HOME", tempfile.mkdtemp(prefix="u44-config-"))
if sys.platform == "win32":
    os.environ.setdefault("APPDATA", os.environ["XDG_CONFIG_HOME"])

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialog,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QWidget,
)

from the_oracle import gui_crash, gui_license  # noqa: E402
from the_oracle.crash import bundle, consent  # noqa: E402

AUDITS: list[dict] = []


def _describe(widget: QWidget) -> dict:
    text = widget.text() if hasattr(widget, "text") else ""
    info: dict = {
        "class": widget.metaObject().className(),
        "object_name": widget.objectName() or None,
        "accessible_name": widget.accessibleName() or None,
        "text": text or None,
        "has_mnemonic": "&" in text,
        "tab_focus": bool(widget.focusPolicy() & 1),
        "enabled": widget.isEnabled(),
    }
    if hasattr(widget, "placeholderText"):
        info["placeholder"] = widget.placeholderText() or None
    if hasattr(widget, "toolTip"):
        info["tooltip"] = widget.toolTip() or None
    if isinstance(widget, QPushButton):
        info["is_default"] = widget.isDefault()
    return info


class _AuditDialog(QDialog):
    """exec(): audit the built dialog, run scripted actions, return."""

    program: tuple[str, ...] = ()

    def exec(self) -> int:
        buttons = self.findChildren(QPushButton)
        edits = [
            w
            for w in self.findChildren(QPlainTextEdit) + self.findChildren(QLineEdit)
            if not (hasattr(w, "isReadOnly") and w.isReadOnly())
        ]
        AUDITS.append(
            {
                "title": self.windowTitle(),
                "modal": self.isModal(),
                "default_button": next((b.text() for b in buttons if b.isDefault()), None),
                "buttons": [_describe(b) for b in buttons],
                "text_edits": [_describe(e) for e in edits],
                "focus_chain": self._focus_chain(),
                "buttons_without_tooltip": [b.text() for b in buttons if not b.toolTip()],
                "buttons_without_mnemonic": [b.text() for b in buttons if "&" not in b.text()],
                "edits_without_accessible_name": [
                    e.placeholderText() or "?" for e in edits if not e.accessibleName()
                ],
            }
        )
        for action in self.program:
            if action.startswith("TYPE:"):
                if edits:
                    edits[0].setPlainText(action[5:])
            else:
                for button in buttons:
                    if button.text() == action and button.isEnabled():
                        button.click()
                        break
        return 0

    def _focus_chain(self) -> list[str]:
        chain: list[str] = []
        first = self.focusWidget()
        if first is None:
            first = next((w for w in self.findChildren(QWidget) if w.focusPolicy() & 1), None)
        current = first
        for _ in range(24):
            if current is None:
                break
            text = current.text() if hasattr(current, "text") else ""
            chain.append(f"{current.metaObject().className()}:{text or current.objectName() or '?'}")
            if not self.focusNextChild():
                break
            nxt = self.focusWidget()
            if nxt is current or nxt is first:
                break
            current = nxt
        return chain


class _AuditMessageBox(QMessageBox):
    def exec(self) -> int:
        AUDITS.append({"message_box": {"title": self.windowTitle(), "text": self.text()}})
        return 0


def scripted(program: tuple[str, ...]) -> type:
    return type("ScriptedDialog", (_AuditDialog,), {"program": tuple(program)})


def main() -> int:
    # Launch-path arming (the entry-point audit, 2026-10-08): this harness
    # runs the REAL gui_crash/gui_license dialogs (Qt-native widget code)
    # without ever passing through cli.main's install+arm. Idempotent;
    # fails closed without consent.
    from the_oracle.crash import handlers as crash_handlers

    crash_handlers.arm_native_capture()
    (QApplication.instance() or QApplication([]))
    parent = QWidget()
    sandbox = Path(tempfile.mkdtemp(prefix="u44-sandbox-"))
    scenarios: dict = {}

    def run_consent(label: str, action: str) -> None:
        consent.consent_path(sandbox).unlink(missing_ok=True)
        AUDITS.clear()
        verdict = gui_crash.run_first_run_consent(parent, sandbox, dialog_cls=scripted((action,)))
        path = consent.consent_path(sandbox)
        scenarios[label] = {
            "verdict": verdict,
            "consent_file_exists": path.exists(),
            "consent_value": consent.read_consent(sandbox),
            "audit": [dict(a) for a in AUDITS],
        }

    for label, action in (
        ("consent_enable", "Enable Local Reports"),
        ("consent_not_now", "Not Now"),
        ("consent_privacy", "Open Privacy Policy"),
    ):
        run_consent(label, action)

    def seed_record() -> None:
        bundle.write_record(sandbox, {"exception": {"type": "RuntimeError", "message": "synthetic audit record"}})

    seed_record()
    AUDITS.clear()
    verdict = gui_crash.run_next_session_review(
        parent, sandbox, dialog_cls=scripted(("Delete All Reports",)), message_box_cls=_AuditMessageBox, open_path_fn=None
    )
    scenarios["review_delete_all"] = {
        "verdict": verdict,
        "records_after": len(bundle.list_records(sandbox)),
        "audit": [dict(a) for a in AUDITS],
    }

    seed_record()
    opened: list[str] = []
    AUDITS.clear()
    verdict = gui_crash.run_next_session_review(
        parent, sandbox, dialog_cls=scripted(("Open Report",)), message_box_cls=_AuditMessageBox, open_path_fn=opened.append
    )
    scenarios["review_open_report"] = {
        "verdict": verdict,
        "opened": list(opened),
        "audit": [dict(a) for a in AUDITS],
    }

    AUDITS.clear()
    activated = gui_license.run_activation_dialog(
        parent, sandbox, dialog_cls=scripted(()), message_box_cls=_AuditMessageBox
    )
    scenarios["activation_empty"] = {"activated": activated, "audit": [dict(a) for a in AUDITS]}

    AUDITS.clear()
    activated = gui_license.run_activation_dialog(
        parent,
        sandbox,
        dialog_cls=scripted(("TYPE:ORACLE1.not-a-real-token", "Activate")),
        message_box_cls=_AuditMessageBox,
    )
    scenarios["activation_invalid_token"] = {"activated": activated, "audit": [dict(a) for a in AUDITS]}

    AUDITS.clear()
    gui_license.run_about_panel(parent, dialog_cls=scripted(()))
    scenarios["about_panel"] = {"outcome": None, "audit": [dict(a) for a in AUDITS]}

    print(json.dumps({"sandbox_root": str(sandbox), "scenarios": scenarios}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
