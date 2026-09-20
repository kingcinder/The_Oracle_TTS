"""Ingestion-transformer popups and preview dialogs (GUI presentation layer).

Everything the user sees when the transformer inspects an input file lives
here: the warning popup, the side-by-side fix preview, and the batch folder
preview. MainWindow drives the flow around them (analysis, file fixes,
settings persistence); this module builds and runs the dialogs themselves.

The Qt classes the test suite patches (``QDialog``, ``QMessageBox``) are
injected by the caller rather than imported, so MainWindow resolves them from
its own module namespace at call time and this module stays patch-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QColor, QTextCharFormat
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from the_oracle.ingest_transformer import (
    labeled_fixed_diff,
    rule_color,
    rule_label,
    side_by_side_diff_rows,
)


class FixPreviewResult(NamedTuple):
    """Outcome of the fix preview dialog, for the caller to act on."""

    accepted: bool
    remember: bool
    geometry: str | None


def color_rule_labels_in_view(view, rules: list[str]) -> None:
    """Color every ``[rule label]`` occurrence in *view* by its rule.

    Each fix rule gets its own stable hue (see ``rule_color``), so a
    mixed batch reads as distinct colors rather than uniform text.
    Uses extra selections appended after any existing ones (row
    backgrounds), so both layers render.
    """
    selections = list(view.extraSelections())
    document = view.document()
    for rule in dict.fromkeys(rules):  # unique, order-preserving
        needle = f" [{rule_label(rule)}]"
        fmt = QTextCharFormat()
        red, green, blue = rule_color(rule)
        fmt.setForeground(QColor(red, green, blue))
        found = document.find(needle, 0)
        while not found.isNull():
            selection = QTextEdit.ExtraSelection()
            selection.cursor = found
            selection.format = fmt
            selections.append(selection)
            found = document.find(needle, found.selectionEnd())
    view.setExtraSelections(selections)


def format_warning_choice(
    parent: QWidget,
    *,
    fixable_issues: list,
    warning_issues: list,
    hint_lines: list[str],
    message_box_cls: type,
) -> str:
    """Show the input-formatting warning popup and return the user's choice.

    ``"fix"`` when the user asked for the automatic correction, ``"cancel"``
    when they chose to stop, ``"continue"`` to proceed without fixing
    (clean-of-fixables file, Ignore, or any non-fix button).
    """
    lines: list[str] = []
    if fixable_issues:
        lines.append(
            f"Found {len(fixable_issues)} formatting problem(s) that can be corrected automatically:"
        )
        for issue in fixable_issues[:5]:
            where = f"line {issue.line_number}" if issue.line_number else "file"
            lines.append(f"  \u2022 {where}: {issue.fix_description}")
        if len(fixable_issues) > 5:
            lines.append(f"  \u2026 and {len(fixable_issues) - 5} more.")
    if warning_issues:
        lines.append(
            f"Found {len(warning_issues)} formatting warning(s) that cannot be corrected automatically:"
        )
        for issue in warning_issues[:5]:
            where = f"line {issue.line_number}" if issue.line_number else "file"
            lines.append(f"  \u2022 {where}: {issue.description}")
        if len(warning_issues) > 5:
            lines.append(f"  \u2026 and {len(warning_issues) - 5} more.")
    # Cast suggestions are meaningful even pre-fix: they come from the
    # transformed text, so a subtitle conversion's cast is included.
    lines.extend(hint_lines)

    box = message_box_cls(parent)
    box.setIcon(message_box_cls.Icon.Warning)
    box.setWindowTitle("Input Formatting")
    box.setText(
        "The input file's formatting may prevent correct speaker attribution."
    )
    box.setInformativeText("\n".join(lines))
    if fixable_issues:
        fix_button = box.addButton("Fix File Automatically", message_box_cls.ButtonRole.AcceptRole)
        box.addButton(message_box_cls.StandardButton.Cancel)
        box.addButton(message_box_cls.StandardButton.Ignore)
        box.setDefaultButton(fix_button)
    else:
        box.addButton(message_box_cls.StandardButton.Ok)
        box.setDefaultButton(message_box_cls.StandardButton.Ok)
    box.exec()
    clicked = box.clickedButton()
    if not fixable_issues:
        return "continue"
    if clicked is fix_button:
        return "fix"
    # standardButton() maps the clicked widget back to its role; a bare
    # `is` comparison against the enum can never be true for a button.
    if box.standardButton(clicked) == message_box_cls.StandardButton.Cancel:
        return "cancel"
    return "continue"  # Ignore (or any non-fix button) proceeds without a fix


def run_fix_preview(
    parent: QWidget,
    original_text: str,
    fixed_text: str,
    fix_count: int,
    line_fixes: list | None = None,
    input_file: str | None = None,
    *,
    dialog_cls: type,
    saved_geometry: str | None = None,
) -> FixPreviewResult:
    """Show the exact rewrite for review side by side; the user's verdict.

    Two aligned panes: the original script on the left, the corrected
    script on the right. Rewritten rows are paired line-for-line and the
    right-hand cell is labeled with the fix rule that produced it (e.g.
    ``[dash/pipe separator]``); rows scroll in sync so long scripts stay
    reviewable. When *input_file* is given, the dialog also offers the
    remember checkbox; the caller persists the returned geometry blob and
    acts on *remember*.
    """
    dialog = dialog_cls(parent)
    dialog.setWindowTitle("Preview Fixed Text")
    dialog.setModal(True)
    dialog.resize(900, 560)
    # Restore the size/position remembered from the last session, so the
    # dialog reopens the way the user left it (multi-monitor safe: the
    # blob encodes position relative to the OS's virtual desktop).
    if isinstance(saved_geometry, str) and saved_geometry:
        dialog.restoreGeometry(QByteArray.fromBase64(saved_geometry.encode("ascii")))
    layout = QVBoxLayout(dialog)

    summary = QLabel(
        f"The correction rewrites {fix_count} line(s). Original on the "
        "left, corrected on the right \u2014 review the changes before "
        "accepting; the original is backed up either way."
    )
    summary.setWordWrap(True)
    layout.addWidget(summary)

    panes = QHBoxLayout()
    left_column = QVBoxLayout()
    left_header = QLabel("Original")
    left_view = QPlainTextEdit()
    left_view.setReadOnly(True)
    left_view.setFont(parent.font())
    left_column.addWidget(left_header)
    left_column.addWidget(left_view, 1)
    panes.addLayout(left_column, 1)

    right_column = QVBoxLayout()
    right_header = QLabel("Corrected (with fix rule)")
    right_view = QPlainTextEdit()
    right_view.setReadOnly(True)
    right_view.setFont(parent.font())
    right_column.addWidget(right_header)
    right_column.addWidget(right_view, 1)
    panes.addLayout(right_column, 1)
    layout.addLayout(panes, 1)

    rows = side_by_side_diff_rows(original_text, fixed_text, line_fixes or [])
    left_lines: list[str] = []
    right_lines: list[str] = []
    for row in rows:
        if row.kind == "same":
            left_lines.append(row.left or "")
            right_lines.append(row.right or "")
        elif row.kind == "changed":
            left_lines.append(f"- {row.left}")
            label = f" [{rule_label(row.rule)}]" if row.rule else ""
            right_lines.append(f"+ {row.right}{label}")
        elif row.kind == "removed":
            left_lines.append(f"- {row.left}")
            right_lines.append("")
        else:  # added
            label = f" [{rule_label(row.rule)}]" if row.rule else ""
            left_lines.append("")
            right_lines.append(f"+ {row.right}{label}")
    left_view.setPlainText("\n".join(left_lines))
    right_view.setPlainText("\n".join(right_lines))

    # Color-code the changed rows: soft red on the original pane, soft
    # green on the corrected pane, so rewrites scan at a glance. Extra
    # selections give full-width per-line backgrounds without rich text.
    # Softened so the text stays readable in both light and dark themes.
    removed_format = QTextCharFormat()
    removed_format.setBackground(QColor(255, 106, 106, 70))
    added_format = QTextCharFormat()
    added_format.setBackground(QColor(108, 220, 108, 70))
    left_selections: list[QTextEdit.ExtraSelection] = []
    right_selections: list[QTextEdit.ExtraSelection] = []
    cursor = left_view.textCursor()
    cursor.movePosition(cursor.MoveOperation.Start)
    for line_index, row in enumerate(rows):
        block = left_view.document().findBlockByLineNumber(line_index)
        if not block.isValid():
            break
        cursor.setPosition(block.position())
        cursor.movePosition(cursor.MoveOperation.EndOfBlock, cursor.MoveMode.KeepAnchor)
        if row.kind == "changed" or row.kind == "removed":
            selection = QTextEdit.ExtraSelection()
            selection.cursor = cursor
            selection.format = removed_format
            left_selections.append(selection)
    cursor = right_view.textCursor()
    cursor.movePosition(cursor.MoveOperation.Start)
    for line_index, row in enumerate(rows):
        block = right_view.document().findBlockByLineNumber(line_index)
        if not block.isValid():
            break
        cursor.setPosition(block.position())
        cursor.movePosition(cursor.MoveOperation.EndOfBlock, cursor.MoveMode.KeepAnchor)
        if row.kind == "changed" or row.kind == "added":
            selection = QTextEdit.ExtraSelection()
            selection.cursor = cursor
            selection.format = added_format
            right_selections.append(selection)
    left_view.setExtraSelections(left_selections)
    right_view.setExtraSelections(right_selections)

    # Tint each [rule label] by its rule's color so a mixed set of
    # fixes reads as distinct hues, not uniform text. Must run after
    # the row backgrounds above (setExtraSelections replaces the list;
    # the helper appends to whatever exists).
    present_rules = [row.rule for row in rows if row.rule]
    color_rule_labels_in_view(right_view, present_rules)

    # Synchronized scrolling: either pane's scroll drives the other.
    left_bar = left_view.verticalScrollBar()
    right_bar = right_view.verticalScrollBar()
    left_bar.valueChanged.connect(right_bar.setValue)
    right_bar.valueChanged.connect(left_bar.setValue)

    remember = None
    if input_file:
        remember = QCheckBox("Remember this choice and fix this file automatically in the future")
        remember.setToolTip(
            "Pre-approve fixes for this exact file: future Analyze/Render runs "
            "correct it silently (a backup is still kept) without this popup. "
            "Clear via Settings > Forget remembered auto-fix approvals."
        )
        layout.addWidget(remember)

    buttons = QHBoxLayout()
    buttons.addStretch(1)
    accept = QPushButton("Accept Fix")
    accept.setDefault(True)
    cancel = QPushButton("Cancel")
    buttons.addWidget(accept)
    buttons.addWidget(cancel)
    layout.addLayout(buttons)
    accept.clicked.connect(dialog.accept)
    cancel.clicked.connect(dialog.reject)
    accepted = dialog.exec() == dialog_cls.DialogCode.Accepted
    # Hand the dialog's size and position back for the caller to persist
    # for the next session (alongside the workspace in app settings).
    geometry = dialog.saveGeometry()
    geometry_blob = bytes(geometry.toBase64()).decode("ascii") if not geometry.isNull() else None
    remember_checked = bool(remember is not None and remember.isChecked())
    return FixPreviewResult(accepted=accepted, remember=remember_checked, geometry=geometry_blob)


def run_batch_fix_preview(
    parent: QWidget,
    folder: str,
    fixes: list,
    warnings_only: list,
    *,
    dialog_cls: type,
) -> list:
    """Folder tree + per-file diff preview of the whole batch.

    A tree on the left lists every fixable file under the chosen folder
    (recursed, shown as paths relative to the folder) with its fix
    count and an include-checkbox (all ticked by default); selecting
    one shows its rule-labeled diff on the right. A rule-filter row of
    checkboxes unticks whole fix rules (e.g. keep only timestamp
    fixes). Unfixable warnings are listed beneath the tree. Returns
    the list of fixes to apply on Accept — recomputed against the rule
    filter and limited to the ticked files (empty when cancelled,
    nothing ticked, or every fix filtered out).
    """
    from the_oracle.ingest_transformer import preview_folder_fixes

    dialog = dialog_cls(parent)
    dialog.setWindowTitle("Preview Batch Fix")
    dialog.setModal(True)
    dialog.resize(860, 560)
    layout = QVBoxLayout(dialog)

    total = sum(fix.fix_count for fix in fixes)
    summary = QLabel(
        f"{len(fixes)} file(s) under the folder can be corrected "
        f"({total} fix(es) total). Tick the files to include; untick "
        "any file to leave it untouched. Selecting a file shows its "
        "changes; every included original is backed up."
    )
    summary.setWordWrap(True)
    layout.addWidget(summary)

    body = QHBoxLayout()

    # Rule filter: unticking a fix rule leaves those lines untouched
    # everywhere in the batch (e.g. accept only timestamp fixes). The
    # Apply is recomputed against the filter, so a file whose fixes are
    # all filtered out is skipped entirely.
    present_rules: list[str] = []
    for fix in fixes:
        for line_fix in fix.line_fixes:
            if line_fix.rule not in present_rules:
                present_rules.append(line_fix.rule)
    filter_row = QHBoxLayout()
    filter_label = QLabel("Fix rules to apply:")
    filter_row.addWidget(filter_label)
    rule_checkboxes: dict[str, QCheckBox] = {}
    for rule in present_rules:
        box = QCheckBox(rule_label(rule))
        box.setChecked(True)
        occurrences = sum(1 for fix in fixes for line_fix in fix.line_fixes if line_fix.rule == rule)
        box.setToolTip(
            f"{occurrences} fix(es) of this rule in the batch. Untick to "
            "leave every line this rule would rewrite unchanged."
        )
        filter_row.addWidget(box)
        rule_checkboxes[rule] = box

    def _excluded_rules() -> set[str]:
        return {rule for rule, box in rule_checkboxes.items() if not box.isChecked()}

    if rule_checkboxes:
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

    tree_column = QVBoxLayout()
    tree_header = QLabel("Files to fix (folder tree)")
    tree = QTreeWidget()
    tree.setHeaderLabels(["File", "Fixes"])
    tree.setColumnWidth(0, 260)
    root_item = QTreeWidgetItem(tree, [Path(folder).name or folder, ""])
    fixes_by_relative: dict[str, object] = {}
    for fix in fixes:
        relative = str(fix.path.relative_to(Path(folder)))
        fixes_by_relative[relative] = fix
        item = QTreeWidgetItem(root_item, [relative, str(fix.fix_count)])
        # Every file is included by default; a checkbox excludes it.
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked)
    root_item.setExpanded(True)
    tree_column.addWidget(tree_header)
    tree_column.addWidget(tree, 1)
    body.addLayout(tree_column, 2)

    def _checked_fixes() -> list:
        """The fixes whose tree items are ticked, in scan order."""
        return [
            fixes_by_relative[root_item.child(i).text(0)]
            for i in range(root_item.childCount())
            if root_item.child(i).checkState(0) == Qt.Checked
        ]

    def _update_counts() -> None:
        excluded = _excluded_rules()
        checked = [fix for fix in _checked_fixes() if any(lf.rule not in excluded for lf in fix.line_fixes)]
        count = len(checked)
        fix_total = sum(
            sum(1 for lf in fix.line_fixes if lf.rule not in excluded)
            for fix in checked
        )
        accept.setText(
            f"Apply Fixes to {count} of {len(fixes)} File(s) ({fix_total} fix(es))"
        )
        accept.setEnabled(bool(checked))

    tree.itemChanged.connect(lambda _item, _column: _update_counts())
    for box in rule_checkboxes.values():
        box.toggled.connect(lambda _checked: _update_counts())

    diff_column = QVBoxLayout()
    diff_header = QLabel("Diff (selected file)")
    diff_view = QPlainTextEdit()
    diff_view.setReadOnly(True)
    diff_view.setFont(parent.font())
    diff_column.addWidget(diff_header)
    diff_column.addWidget(diff_view, 1)
    body.addLayout(diff_column, 3)
    layout.addLayout(body, 1)

    if warnings_only:
        warning_lines = [
            f"! {Path(a.path).relative_to(Path(folder))}, line {issue.line_number}: "
            f"{issue.description} (cannot be fixed automatically)"
            for a in warnings_only
            for issue in a.warning_issues[:2]
        ]
        if warning_lines:
            warnings_label = QLabel("\n".join(warning_lines[:6]))
            warnings_label.setWordWrap(True)
            layout.addWidget(warnings_label)

    first = fixes[0] if fixes else None

    def _show_diff() -> None:
        selected = tree.selectedItems()
        if not selected or selected[0] is root_item:
            if first is not None:
                relative = str(first.path.relative_to(Path(folder)))
            else:
                return
        else:
            relative = selected[0].text(0)
        fix = fixes_by_relative.get(relative)
        if fix is None:
            return
        diff_view.setPlainText(
            labeled_fixed_diff(fix.original_text, fix.fixed_text, fix.line_fixes)
        )
        color_rule_labels_in_view(
            diff_view, [line_fix.rule for line_fix in fix.line_fixes]
        )

    tree.itemSelectionChanged.connect(_show_diff)
    # Preselect the first fixable file so the diff pane is never empty.
    if fixes:
        first_relative = str(fixes[0].path.relative_to(Path(folder)))
        for index in range(root_item.childCount()):
            child = root_item.child(index)
            if child.text(0) == first_relative:
                tree.setCurrentItem(child)
                break
        _show_diff()

    buttons = QHBoxLayout()
    buttons.addStretch(1)
    accept = QPushButton(f"Apply Fixes to {len(fixes)} File(s)")
    accept.setDefault(True)
    cancel = QPushButton("Cancel")
    buttons.addWidget(accept)
    buttons.addWidget(cancel)
    layout.addLayout(buttons)
    accept.clicked.connect(dialog.accept)
    cancel.clicked.connect(dialog.reject)
    # After the button exists, so the label/enablement can be computed.
    _update_counts()
    accepted = dialog.exec() == dialog_cls.DialogCode.Accepted
    if not accepted:
        return []
    checked_relatives = {
        root_item.child(i).text(0)
        for i in range(root_item.childCount())
        if root_item.child(i).checkState(0) == Qt.Checked
    }
    excluded = _excluded_rules()
    if not checked_relatives:
        return []
    # Recompute the batch against the rule filter: the precomputed
    # rewrites include every rule, so applying them as-is would ignore
    # the filter. A file whose fixes are all excluded drops out here.
    try:
        refixes, _warnings = preview_folder_fixes(folder, exclude_rules=excluded or None)
    except ValueError:
        return []  # every fix was filtered out
    return [
        fix for fix in refixes
        if str(fix.path.relative_to(Path(folder))) in checked_relatives
    ]
