"""The main window's menu bar: File, Settings, Theme, Help, and the
menu-bar Recording Studio action.

Extraction slice 12 (2026-10-10). ``MainWindow._build_menu`` moved here as
:func:`build_menu`, the campaign's biggest remaining method (187 lines, 24
public action attributes, the ``inference_wizard_menu`` / ``templates_menu``
menus, the ``_theme_actions`` dict, and the three ``textEdited``
workspace-persistence connections).

Two seams, both deliberate:

- **Widgets, actions and callbacks ride ``main_window``** (the
  gui_ingest_tools direction rule): nothing in this module imports app_gui, so
  the window is the only route to its own state and the delegate is the only
  call site.
- **``THEMES`` is injected** (``themes`` is a required keyword-only parameter)
  rather than imported here. ``app_gui`` stays the single resolution point for
  a mapping its own ``_apply_theme_selection`` also reads, which forecloses a
  latent app_gui-level ``THEMES`` split: a module-global binding here would
  turn every ``app_gui.THEMES`` rebind into a silent no-op for the merged body
  (the harm the patch-couple net exists to catch, and the seam this slice is
  mutation-proven on). The mapping's values are ``gui_themes.ThemeTokens``, and
  only their ``name``/``description``/``key`` attributes are read here.

``QAction`` is imported by this module; no test patches ``app_gui.QAction``,
and the patch-couple census does not list it.
"""


from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from PySide6.QtGui import QAction

if TYPE_CHECKING:
    from the_oracle.gui_themes import ThemeTokens


def build_menu(main_window, *, themes: Mapping[str, ThemeTokens]) -> None:
    """Build the window's whole menu bar, resolved against ``main_window``.

    Every action attribute the rest of the window (and the suite) reaches —
    ``new_action`` ... ``about_action``, ``_theme_actions``,
    ``inference_wizard_menu``, ``templates_menu`` — is set on ``main_window``,
    exactly as the method set it on ``self``.

    ``themes`` is injected by the caller (see the module docstring): the theme
    menu iterates it and connects each action to
    ``main_window._apply_theme_selection``.
    """
    file_menu = main_window.menuBar().addMenu("File")
    new_action = QAction("New Project", main_window)
    new_action.triggered.connect(main_window.new_project)
    open_action = QAction("Open Project", main_window)
    open_action.triggered.connect(main_window.open_project)
    save_action = QAction("Save Project", main_window)
    save_action.triggered.connect(main_window.save_project)
    save_as_action = QAction("Save Project As", main_window)
    save_as_action.triggered.connect(main_window.save_project_as)
    for action in (new_action, open_action, save_action, save_as_action):
        file_menu.addAction(action)
    file_menu.addSeparator()
    save_profile_action = QAction("Save Profile…", main_window)
    save_profile_action.setToolTip(
        "Save every current GUI option and slider position (shared render "
        "settings and all speaker voice settings) to a reusable profile "
        "file. Same as Settings → Save Settings..."
    )
    save_profile_action.triggered.connect(main_window.save_settings_profile)
    file_menu.addAction(save_profile_action)
    load_profile_action = QAction("Load Profile…", main_window)
    load_profile_action.setToolTip(
        "Load a saved profile file, restoring all options and slider "
        "positions. Same as Settings → Load Settings..."
    )
    load_profile_action.triggered.connect(main_window.load_settings_profile)
    file_menu.addAction(load_profile_action)
    file_menu.addSeparator()
    batch_fix_action = QAction("Batch Fix Input Folder...", main_window)
    batch_fix_action.setToolTip(
        "Scan a folder of input scripts for formatting problems the engine "
        "would misread, show one combined preview of every proposed "
        "correction, and fix all accepted files in place (each original "
        "backed up)."
    )
    batch_fix_action.triggered.connect(main_window._batch_fix_input_folder)
    file_menu.addAction(batch_fix_action)

    settings_menu = main_window.menuBar().addMenu("Settings")
    reset_defaults_action = QAction("Reset to Defaults", main_window)
    reset_defaults_action.triggered.connect(main_window.reset_settings_to_defaults)
    save_settings_action = QAction("Save Settings...", main_window)
    save_settings_action.triggered.connect(main_window.save_settings_profile)
    load_settings_action = QAction("Load Settings...", main_window)
    load_settings_action.triggered.connect(main_window.load_settings_profile)
    save_template_action = QAction("Save Current as Template...", main_window)
    save_template_action.triggered.connect(main_window.save_template_profile)
    settings_menu.addAction(reset_defaults_action)
    settings_menu.addSeparator()
    settings_menu.addAction(save_settings_action)
    settings_menu.addAction(load_settings_action)
    settings_menu.addSeparator()
    settings_menu.addAction(save_template_action)
    settings_menu.addSeparator()
    main_window.inference_wizard_menu = settings_menu.addMenu("Replay Setup & Tutorial")
    main_window.replay_full_wizard_action = QAction("Replay Entire Setup & Tutorial", main_window)
    main_window.replay_full_wizard_action.triggered.connect(lambda: main_window._start_inference_wizard("full", force=True))
    main_window.replay_discovery_wizard_action = QAction("Replay Hardware Discovery Only", main_window)
    main_window.replay_discovery_wizard_action.triggered.connect(lambda: main_window._start_inference_wizard("discovery", force=True))
    main_window.replay_main_wizard_action = QAction("Replay Main GUI Tour Only", main_window)
    main_window.replay_main_wizard_action.triggered.connect(lambda: main_window._start_inference_wizard("main", force=True))
    for wizard_action in (main_window.replay_full_wizard_action, main_window.replay_discovery_wizard_action, main_window.replay_main_wizard_action):
        main_window.inference_wizard_menu.addAction(wizard_action)
    main_window.recording_wizard_action = QAction("Replay Recording Studio Setup & Guide", main_window)
    main_window.recording_wizard_action.triggered.connect(lambda: main_window._start_recording_wizard(force=True))
    settings_menu.addAction(main_window.recording_wizard_action)
    main_window.output_filename_warning_action = QAction("Warn before generic output filenames", main_window)
    main_window.output_filename_warning_action.setCheckable(True)
    main_window.output_filename_warning_action.setChecked(main_window.output_filename_warning_enabled)
    main_window.output_filename_warning_action.setToolTip(
        "Show a caution before a render uses the automatically derived input-based "
        "filename. Re-enable this setting here after disabling it in the warning."
    )
    main_window.output_filename_warning_action.toggled.connect(main_window._on_output_filename_warning_toggled)
    settings_menu.addAction(main_window.output_filename_warning_action)

    main_window.templates_menu = settings_menu.addMenu("Load Template")
    main_window.templates_menu.aboutToShow.connect(main_window._rebuild_templates_menu)
    main_window.confirmation_action = QAction("Re-enable delete confirmations", main_window)
    main_window.confirmation_action.triggered.connect(main_window._enable_delete_confirmation)
    clear_trust_action = QAction("Forget remembered auto-fix approvals", main_window)
    clear_trust_action.setToolTip(
        "Files approved via 'Remember this choice' in the fix preview will "
        "prompt for review again before any automatic correction."
    )
    clear_trust_action.triggered.connect(main_window._clear_trusted_input_files)
    restore_backup_action = QAction("Restore most recent input-file backup", main_window)
    restore_backup_action.setToolTip(
        "Undo the most recent input-formatting fix by restoring the file from "
        "the timestamped backup taken before the correction. Regret a fix? "
        "This puts the original back."
    )
    restore_backup_action.triggered.connect(main_window._restore_most_recent_format_backup)
    settings_menu.addSeparator()
    settings_menu.addAction(main_window.confirmation_action)
    settings_menu.addAction(clear_trust_action)
    settings_menu.addAction(restore_backup_action)
    settings_menu.addSeparator()
    main_window.remember_backend_action = QAction("Remember GPU/CPU choice", main_window)
    main_window.remember_backend_action.setCheckable(True)
    main_window.remember_backend_action.setChecked(bool(main_window._app_settings.get("remember_backend", True)))
    main_window.remember_backend_action.setToolTip(
        "Remember which inference backend was chosen (PyTorch CPU or Vulkan "
        "GPU) plus the audiocpp_cli and Chatterbox model paths, and restore "
        "them automatically at next launch so the GPU needs no setup again. "
        "Uncheck to always start on PyTorch (CPU)."
    )
    # toggled (not triggered): PySide6's triggered binding here takes no
    # checked argument, so a real menu click would not deliver the bool to
    # the handler. toggled(bool) always does, and nothing programmatically
    # toggles the action after the setChecked above (which precedes the
    # connect), so no spurious early emission is possible.
    main_window.remember_backend_action.toggled.connect(main_window._on_remember_backend_toggled)
    settings_menu.addAction(main_window.remember_backend_action)
    main_window.download_vulkan_model_action = QAction("Download Vulkan Model...", main_window)
    main_window.download_vulkan_model_action.setToolTip(
        "Run scripts/download_audio_cpp_model.sh in the background and report "
        "the resulting ORACLE_AUDIOCPP_MODEL path to use for the Vulkan backend."
    )
    main_window.download_vulkan_model_action.triggered.connect(main_window.download_vulkan_model)
    settings_menu.addAction(main_window.download_vulkan_model_action)

    # Top-of-window Recording Studio: a plain menu-bar action (not a menu)
    # that opens the voice-recording window in its own dialog.
    main_window.recording_studio_action = QAction("Recording Studio…", main_window)
    main_window.recording_studio_action.setToolTip(
        "Open the Custom Voice Recording Studio: record a new reference "
        "voice (a 'Seashell') from a microphone with a teleprompter, then "
        "pick it for any speaker."
    )
    main_window.recording_studio_action.triggered.connect(main_window.open_recording_studio)
    main_window.menuBar().addAction(main_window.recording_studio_action)

    # Help menu: privacy/crash reports, license activation, About (U3).
    help_menu = main_window.menuBar().addMenu("Help")
    main_window.privacy_action = QAction("Privacy && Crash Reports…", main_window)
    main_window.privacy_action.setToolTip(
        "Review crash reports from last session, change crash-report consent, "
        "and open the privacy policy. Reports never leave this machine unless "
        "you share one yourself."
    )
    main_window.privacy_action.triggered.connect(main_window._open_crash_privacy_dialog)
    help_menu.addAction(main_window.privacy_action)
    main_window.activate_license_action = QAction("Activate License…", main_window)
    main_window.activate_license_action.setToolTip(
        "Paste your license token (offline activation — no server is contacted)."
    )
    main_window.activate_license_action.triggered.connect(main_window._open_license_activation)
    help_menu.addAction(main_window.activate_license_action)
    main_window.about_action = QAction("About The Oracle…", main_window)
    main_window.about_action.setToolTip("Which edition this install runs, and the privacy posture in one glance.")
    main_window.about_action.triggered.connect(main_window._open_about_panel)
    help_menu.addAction(main_window.about_action)

    # Keep references for Ctrl+hover help registration (see
    # _register_ctrl_help_descriptions).
    main_window.new_action = new_action
    main_window.open_action = open_action
    main_window.save_action = save_action
    main_window.save_as_action = save_as_action
    main_window.reset_defaults_action = reset_defaults_action
    main_window.save_settings_action = save_settings_action
    main_window.load_settings_action = load_settings_action
    main_window.save_template_action = save_template_action
    main_window.save_profile_action = save_profile_action
    main_window.load_profile_action = load_profile_action

    # Theme menu: six built-in looks; the choice is persisted immediately.
    theme_menu = main_window.menuBar().addMenu("Theme")
    main_window._theme_actions: dict[str, QAction] = {}
    for key in themes:
        tokens = themes[key]
        action = QAction(tokens.name, main_window)
        action.setCheckable(True)
        action.setToolTip(f"{tokens.description} Contrast-checked for legibility.")
        action.triggered.connect(lambda _checked=False, current=key: main_window._apply_theme_selection(current))
        theme_menu.addAction(action)
        main_window._theme_actions[key] = action

    # Workspace persistence wiring for the path fields (input file joins
    # backend choice, theme, splitters, section states, and the configured
    # default folders in the app settings file; textEdited deliberately
    # excludes programmatic restore/default changes during startup) lives in
    # app_gui._build_paths_content, with the fields it wires: a
    # restored-collapsed Script & Cast section must not be built just because
    # the menu was constructed (lazy collapsed sections, 2026-10-10).
