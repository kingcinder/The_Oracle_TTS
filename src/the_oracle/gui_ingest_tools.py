"""Ingest-tools handler flows (extraction slice 10, 2026-10-08).

The orchestration bodies behind MainWindow's input-formatting actions live
here: the per-file transformer check, the batch folder fix, and the
restore-most-recent-backup settings action. MainWindow keeps the methods
(thin wrappers with the exact signatures the tests drive) and stays the
owner of app-settings state, the input/output fields, and the status panel;
this module owns the flows around them.

The patch seam: the test suite rebinds ``app_gui.QMessageBox`` and
``app_gui.QDialog`` wholesale (module-attribute patches a moved body
resolving the bare names would silently miss -- the Vulkan-slice
``find_audiocpp_binary`` harm), so every Qt class the flows touch crosses
this boundary as an explicit parameter resolved from app_gui's globals at
call time (the VulkanPreflightThread ``preflight_report=`` precedent).
This module imports no Qt classes at all, which the patch-couple net
enforces mechanically.
"""

from __future__ import annotations

from pathlib import Path

from the_oracle.gui_settings import drop_next_format_backup, next_format_backup, save_app_settings


def run_ingest_transformer_check(window, *, message_box_cls: type) -> bool:
    """Check the input file's formatting with the ingestion transformer.

    When fixable or unfixable format issues are found, show a warning
    popup explaining them; the popup offers a one-click in-place fix
    (with a timestamped backup) when the issues are fixable.

    Returns True when analysis should proceed (no issues, user fixed the
    file, or user chose to continue anyway); False when the user wants to
    stop and fix the file themselves.
    """
    from the_oracle.gui_ingest import format_warning_choice
    from the_oracle.ingest_transformer import analyze_input_file, fix_input_file, preview_fixed_text

    input_file = window.input_path.text().strip()
    if not input_file or not Path(input_file).is_file():
        return True
    try:
        analysis = analyze_input_file(input_file)
    except (OSError, ValueError, UnicodeDecodeError):
        # An unreadable file will fail on its own in prepare_plan with a
        # clear error; the transformer must never block analysis.
        return True
    if not analysis.has_issues:
        return True
    if window._input_file_is_trusted(input_file):
        # The user pre-approved fixes for this file: apply them silently
        # (backup kept) and continue, no popup.
        try:
            _text, fix_count, backup_path = fix_input_file(input_file)
        except (OSError, ValueError):
            return True  # fall back to the normal flow if the fix fails
        window._remember_format_backup(input_file, backup_path)
        window.error_panel.append(
            f"Input formatting: auto-corrected {fix_count} problem(s) in "
            f"{Path(input_file).name} (trusted file; backup kept)."
        )
        return True

    choice = format_warning_choice(
        window,
        fixable_issues=analysis.fixable_issues,
        warning_issues=analysis.warning_issues,
        hint_lines=window._speaker_ref_hint_lines(input_file),
        message_box_cls=message_box_cls,
    )
    if choice == "cancel":
        return False
    if choice != "fix":
        return True  # clean of fixables, Ignore, or any non-fix button
    try:
        original_text, fixed_text, fix_count, _issues, line_fixes = preview_fixed_text(input_file)
    except (OSError, ValueError) as exc:
        message_box_cls.critical(window, "Fix Failed", f"The file could not be corrected:\n{exc}")
        return False
    if not window._show_fix_preview_dialog(original_text, fixed_text, fix_count, line_fixes, input_file=input_file):
        window.error_panel.append("Fix cancelled: the input file was left unchanged.")
        return False
    try:
        _text, fix_count, backup_path = fix_input_file(input_file)
    except (OSError, ValueError) as exc:
        message_box_cls.critical(window, "Fix Failed", f"The file could not be corrected:\n{exc}")
        return False
    window._remember_format_backup(input_file, backup_path)
    message = f"Corrected {fix_count} formatting problem(s) in {Path(input_file).name}."
    if backup_path:
        message += f"\n\nA backup of the original was saved to:\n{backup_path}"
    hint_lines = window._speaker_ref_hint_lines(input_file)
    if hint_lines:
        message += "\n\n" + "\n".join(hint_lines)
    message_box_cls.information(window, "File Corrected", message)
    window.error_panel.append(
        f"Input formatting: corrected {fix_count} problem(s) in "
        f"{Path(input_file).name}"
        + (f" (backup: {backup_path})" if backup_path else "")
        + " \u2014 re-analyzing the corrected file now."
    )
    for hint in window._speaker_ref_hint_lines(input_file):
        window.error_panel.append(f"  {hint}")
    return True


def batch_fix_input_folder(window, *, message_box_cls: type, file_dialog_cls: type) -> None:
    """Scan a folder for misformatted input scripts and fix them in bulk.

    One folder picker, one combined preview of every file's exact
    rewrite, then a single Apply that writes every accepted file (each
    original backed up) and reports per-file results in the status panel.
    """
    from the_oracle.ingest_transformer import analyze_folder, apply_folder_fixes, preview_folder_fixes

    folder = file_dialog_cls.getExistingDirectory(window, "Choose Folder of Input Scripts", "")
    if not folder:
        return
    try:
        analyses = analyze_folder(folder)
    except (OSError, ValueError, UnicodeDecodeError):
        message_box_cls.critical(window, "Scan Failed", f"The folder could not be read:\n{folder}")
        return
    if not analyses:
        message_box_cls.information(
            window,
            "Input Formatting",
            f"No formatting problems found in:\n{folder}\n\nEvery file is already in a format the engine attributes correctly.",
        )
        return

    warnings_only = [a for a in analyses if not a.fixable_issues]
    try:
        fixes, _warnings = preview_folder_fixes(folder)
    except ValueError:
        # Nothing fixable: only warnings. Report them, offer no fix.
        lines = [f"Found {len(warnings_only)} file(s) with unfixable formatting warnings:"]
        for analysis in warnings_only[:10]:
            for issue in analysis.warning_issues[:3]:
                lines.append(f"  \u2022 {Path(analysis.path).name}, line {issue.line_number}: {issue.description}")
        message_box_cls.warning(window, "Input Formatting", "\n".join(lines))
        return

    accepted = window._show_batch_fix_preview_dialog(folder, fixes, warnings_only)
    if not accepted:
        window.error_panel.append("Batch fix cancelled: no files were changed.")
        return
    try:
        written = apply_folder_fixes(accepted)
    except OSError as exc:
        message_box_cls.critical(window, "Batch Fix Failed", f"The files could not be corrected:\n{exc}")
        return
    total = sum(count for _path, count, _backup in written)
    window.error_panel.append(
        f"Batch fix: corrected {total} formatting problem(s) across "
        f"{len(written)} file(s) in {folder}."
    )
    for path, count, backup_path in written:
        window.error_panel.append(
            f"  \u2022 {Path(path).name}: {count} fix(es)"
            + (f" (backup: {backup_path})" if backup_path else "")
        )
    message_box_cls.information(
        window,
        "Batch Fix Complete",
        f"Corrected {total} formatting problem(s) across {len(written)} file(s).\n\n"
        "Backups of every original were saved next to the files.",
    )


def restore_most_recent_format_backup(window, *, message_box_cls: type) -> None:
    """Settings action: undo the most recent input-file fix from its backup."""
    record = next_format_backup(window._app_settings)
    if record is None:
        message_box_cls.information(
            window,
            "Restore Backup",
            "No input-file backups are recorded yet. Backups are noted "
            "whenever a formatting fix corrects a file in place.",
        )
        return
    fixed_file = Path(record["file"])
    backup_file = Path(record["backup"])
    if not backup_file.is_file():
        message_box_cls.critical(
            window,
            "Restore Backup",
            f"The backup file no longer exists:\n{backup_file}",
        )
        drop_next_format_backup(window._app_settings)
        if window._app_settings_ready:
            try:
                save_app_settings(window._app_settings)
            except OSError:
                pass
        return
    stamp = record.get("stamp") or "an earlier time"
    answer = message_box_cls.question(
        window,
        "Restore Backup",
        f"Restore {fixed_file.name} from its most recent backup?\n\n"
        f"Backup: {backup_file}\nTaken: {stamp}\n\n"
        "The corrected version will be overwritten by the original.",
        message_box_cls.StandardButton.Yes | message_box_cls.StandardButton.No,
        message_box_cls.StandardButton.No,
    )
    if answer != message_box_cls.StandardButton.Yes:
        return
    try:
        text = backup_file.read_text(encoding="utf-8")
        fixed_file.write_text(text, encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        message_box_cls.critical(
            window,
            "Restore Failed",
            f"The backup could not be restored:\n{exc}",
        )
        return
    drop_next_format_backup(window._app_settings)
    if window._app_settings_ready:
        try:
            save_app_settings(window._app_settings)
        except OSError:
            pass
    window.error_panel.append(
        f"Restored {fixed_file.name} from its backup (fix of {stamp} undone)."
    )
    # Re-point the input field when the restored file is the remembered
    # input (or nothing is loaded), so the next Analyze/render uses it.
    current = window.input_path.text().strip()
    if not current or Path(current).resolve() == fixed_file:
        window.input_path.setText(str(fixed_file))
    message_box_cls.information(
        window,
        "Backup Restored",
        f"{fixed_file.name} was restored from its backup.\n\n"
        "The corrected version was overwritten by the original.",
    )
