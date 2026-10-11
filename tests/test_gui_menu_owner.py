"""Ownership tests for the menu-bar extraction (gui_menu.py).

Extraction slice 12 (2026-10-10) moved ``MainWindow._build_menu`` — the
campaign's biggest remaining method — into ``the_oracle.gui_menu`` as the
module-level :func:`build_menu`, with two seams both deliberate and both
mutation-proven on a real window during the slice: widgets, actions and
callbacks ride ``main_window`` (the gui_ingest_tools direction rule), and
``THEMES`` is injected as a required keyword-only parameter so app_gui stays
the single resolution point for a mapping its own ``_apply_theme_selection``
also reads.

The THEMES seam's behavioral half is owned by ``tests/test_gui_menu_themes_seam.py``
(rebound-app_gui.THEMES leg + baseline leg, both building the real window);
the patch-couple couple net in ``tests/test_app_gui_patch_surface.py`` scans
gui_menu's bare-name loads against the suite's own census. This file pins the
owner-side shape those files do not:

1. ONE OWNER — ``build_menu`` is defined in gui_menu and is its ONLY public
   function; app_gui holds NO module-level binding for it (the slice-10
   function-local delegate convention), so an app_gui-level patch is a quiet
   no-op and gui_menu is the single patch point, exactly what
   ``moved_owners.the_oracle.gui_menu`` records.
2. THE DELEGATE PASSES THE BARE THEMES NAME — app_gui's ``_build_menu`` body
   resolves ``THEMES`` from app_gui's own globals at call time
   (``build_menu(self, themes=THEMES)``). A captured or owner-resolved
   mapping would go deaf to app_gui-level rebinds (the injected-seam
   precedent of the Vulkan preflight report).
3. THE SEAM IS A SIGNATURE, NOT A CONVENTION — ``themes`` is a required
   keyword-only parameter of ``build_menu``. A defaulted or positional-only
   mapping could be swapped by a caller without anyone noticing at the seam.
4. DIRECTION — gui_menu imports app_gui in NO spelling (the direction rule;
   the ``main_window`` parameter is the only route to window state).
5. HARM CLASS — the moved body resolves NO app_gui-patched name as a bare
   global. Its bare-name loads are exactly the names it imports itself
   (``QAction``; the census deliberately does not list it) plus its own
   parameters — anything else colliding with an app_gui census name is an
   offender to inject across the call boundary.
6. THE MANIFEST RECORD — ``moved_owners.the_oracle.gui_menu`` lists exactly
   ``build_menu``, and gui_menu defines exactly that one function as its
   public surface: a helper defined here without being registered would make
   the ownership record silently incomplete.
7. BEHAVIOR — the moved body is self-sufficient on a stub window driven with
   an INJECTED mapping: the menu bar carries File/Settings/Theme/Help plus
   the menu-bar Recording Studio action, and the Theme menu iterates the
   injected mapping exactly (order included). This proves the owner works
   independent of app_gui and that ``themes`` is the body's only theme
   source — the seam tested by construction rather than only by rebind.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QMainWindow

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "scripts" / "patch_surface_manifest.json"

_OWNER = "the_oracle.gui_menu"
_OWNER_NAME = "build_menu"

#: Every attribute the moved body must SET on ``main_window``: the 24 public
#: action attributes, the two menus the rest of the window reads
#: (``inference_wizard_menu``, ``templates_menu``), and ``_theme_actions``.
#: A body that stopped setting any of these would strand a read elsewhere in
#: app_gui or the suite with an AttributeError at first use, so the set is
#: pinned exactly — new attributes assert here before anywhere else relies.
_MOVED_BODY_ASSIGNMENTS = (
    "inference_wizard_menu",
    "replay_full_wizard_action",
    "replay_discovery_wizard_action",
    "replay_main_wizard_action",
    "recording_wizard_action",
    "output_filename_warning_action",
    "templates_menu",
    "confirmation_action",
    "remember_backend_action",
    "download_vulkan_model_action",
    "recording_studio_action",
    "privacy_action",
    "activate_license_action",
    "about_action",
    "new_action",
    "open_action",
    "save_action",
    "save_as_action",
    "reset_defaults_action",
    "save_settings_action",
    "load_settings_action",
    "save_template_action",
    "save_profile_action",
    "load_profile_action",
    "_theme_actions",
)

#: App_gui census names (measured 2026-10-10 with the same AST census the
#: patch-couple net derives): a moved body resolving any of these as a bare
#: global would be invisible to the suite's app_gui-level rebinds — the
#: harm class. QAction is deliberately absent: gui_menu imports it itself
#: and nothing in the suite patches app_gui.QAction.
#: Vacuity floor: if the census shrinks below this, the scan may be going
#: blind — grow the list from the current suite instead of trusting a
#: smaller scan.
_APP_GUI_PATCHED_POLICY_NAMES = frozenset(
    {
        "OraclePipeline",
        "ChatterboxEngine",
        "RenderWorker",
        "PreviewWorker",
        "VulkanDeviceProbeThread",
        "VulkanPreflightThread",
        "VulkanSetupThread",
        "ModelDownloadThread",
        "find_audiocpp_binary",
        "_vulkan_preflight_report",
        "QMediaPlayer",
        "QAudioOutput",
        "QDialog",
        "QMessageBox",
        "QFileDialog",
        "ensure_repo_default_paths",
        "load_recent_reference_paths",
        "blend_voice_choices",
        "default_voice_choices",
        "cuda_devices",
        "cuda_reason",
        "launch_gui",
        "THEMES",
    }
)
_MIN_CENSUS_NAMES = 20

pytestmark = pytest.mark.usefixtures("qt_app")


def _import(name: str):
    return importlib.import_module(name)


def _gui_menu_tree():
    owner = _import(_OWNER)
    source = Path(owner.__file__).read_text(encoding="utf-8")
    return owner, ast.parse(source)


# --- contract 1: one owner, no module-level re-export ------------------------


def test_owner_defines_build_menu_as_its_only_public_function() -> None:
    owner, tree = _gui_menu_tree()
    public = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    assert public == {_OWNER_NAME}, (
        f"gui_menu's public function surface must be exactly {{'{_OWNER_NAME}'}}; "
        f"got {sorted(public)} — a helper added here must be registered in the "
        "manifest or stay private"
    )
    assert callable(getattr(owner, _OWNER_NAME))


def test_app_gui_holds_no_module_level_binding_for_the_owner() -> None:
    """The owner is reachable ONLY through the function-local import inside
    ``_build_menu``: app_gui's module namespace carries no ``build_menu``
    binding of any form (assignment, import, import-from). A module-level
    re-export would invite app_gui-level patches that are silent no-ops for
    the moved body — the exact shape the moved-owners record forbids."""
    app_gui = _import("the_oracle.app_gui")
    assert not hasattr(app_gui, _OWNER_NAME), (
        "app_gui must NOT re-export build_menu: MainWindow reaches the menu "
        "bar through the _build_menu delegate's function-local import, so a "
        "module-level binding would only invite app_gui-level patches that "
        "never intercept the moved body"
    )
    tree = ast.parse(Path(app_gui.__file__).read_text(encoding="utf-8"))
    # No import/import-from/assignment of the name at module top level in ANY
    # spelling (the_oracle.gui_menu, .gui_menu, alias included).
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname == _OWNER_NAME or alias.name.split(".")[-1] == _OWNER_NAME:
                    offenders.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname == _OWNER_NAME or alias.name == _OWNER_NAME:
                    offenders.append(f"line {node.lineno}: from {node.module} import {alias.name}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == _OWNER_NAME:
                    offenders.append(f"line {node.lineno}: assignment to {_OWNER_NAME}")
    assert offenders == [], (
        f"app_gui binds {_OWNER_NAME} at module level: {offenders} — the "
        "moved-owner rule pins the function-local import"
    )


# --- contract 2: the delegate passes the bare THEMES name --------------------


def test_delegate_resolves_themes_bare_from_app_gui_globals() -> None:
    """app_gui's ``_build_menu`` body must contain the function-local owner
    import and hand over ``themes=THEMES``, the bare app_gui-global name. A
    body that resolved the mapping another way (qualified lookup, captured
    object, owner-imported THEMES) would go deaf to app_gui-level rebinds —
    the seam the VULKAN preflight seam pins as ``find_audiocpp_binary``."""
    app_gui = _import("the_oracle.app_gui")
    source = Path(app_gui.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_window = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    defs = {node.name: node for node in main_window.body if isinstance(node, ast.FunctionDef)}
    assert "_build_menu" in defs, "app_gui must keep the _build_menu delegate"
    body_source = ast.get_source_segment(source, defs["_build_menu"]) or ""
    assert f"from {_OWNER} import {_OWNER_NAME}" in body_source, (
        "the delegate must import the owner function function-locally"
    )
    assert "build_menu(self, themes=THEMES)" in body_source, (
        "the delegate must pass the BARE THEMES name — a captured or "
        "qualified lookup would go deaf to app_gui-level rebinds"
    )


# --- contract 3: the seam is a required keyword-only parameter ---------------


def test_themes_is_a_required_keyword_only_parameter() -> None:
    owner, tree = _gui_menu_tree()
    signature = inspect.signature(owner.build_menu)
    parameter = signature.parameters.get("themes")
    assert parameter is not None, "build_menu must carry the themes seam parameter"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
        "themes must be KEYWORD-ONLY: a positional mapping could be swapped "
        "at a call site without the seam anyone adjudicated"
    )
    assert parameter.default is inspect.Parameter.empty, (
        "themes must be REQUIRED: a defaulted mapping would let a caller "
        "build a menu silently from a hard-coded look list"
    )
    # The injected spelling is also the read site: the signature-only probe
    # would pass over a body still reading its own module global.
    func_def = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == _OWNER_NAME
    )
    body_source = ast.get_source_segment(
        Path(owner.__file__).read_text(encoding="utf-8"), func_def
    ) or ""
    assert "for key in themes:" in body_source, (
        "build_menu's body must iterate the INJECTED mapping, not a module global"
    )


# --- contract 4: direction — nothing imports app_gui -------------------------


def test_gui_menu_imports_nothing_from_app_gui() -> None:
    from tests.test_gui_import_direction import _module_root, _scan_imports

    _forbids_app_gui = lambda module: _module_root(module) == "app_gui"
    owner, _tree = _gui_menu_tree()
    source = Path(owner.__file__).read_text(encoding="utf-8")
    scan = _scan_imports(source, forbidden=_forbids_app_gui)
    assert not scan.violations and not scan.unresolved, (
        f"gui_menu imports app_gui: violations={scan.violations} "
        f"unresolved={scan.unresolved} — the main window's collaborators "
        "cross the seam as the main_window parameter, never as imports"
    )


# --- contract 5: the harm class — no census name resolved bare ---------------


def test_moved_body_resolves_no_app_gui_patched_name_as_a_bare_global() -> None:
    assert len(_APP_GUI_PATCHED_POLICY_NAMES) >= _MIN_CENSUS_NAMES, (
        "the scan's census copy collapsed — fix the names list, not this pin"
    )
    app_gui = _import("the_oracle.app_gui")
    del app_gui  # the census is derived from the SUITE, not from app_gui's own source
    literal_targets: set[str] = set()
    for test_file in (REPO_ROOT / "tests").glob("test_*.py"):
        try:
            tree = ast.parse(test_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setattr":
                args = node.args
                if len(args) >= 2 and isinstance(args[1], ast.Constant) and isinstance(args[1].value, str):
                    target = args[0]
                    spelled = None
                    if isinstance(target, ast.Name) and target.id in {"app_gui", "the_oracle"}:
                        spelled = "app_gui"
                    elif isinstance(target, ast.Attribute):
                        spelled = getattr(target.value, "id", None)
                    elif isinstance(target, ast.Constant) and isinstance(target.value, str) and "app_gui" in target.value:
                        spelled = "app_gui"
                    if spelled == "app_gui":
                        literal_targets.add(args[1].value)
    assert literal_targets & _APP_GUI_PATCHED_POLICY_NAMES, (
        "none of the census names are still rebound at app_gui level — "
        "re-measure the census instead of scanning against a stale list"
    )
    _owner, owner_tree = _gui_menu_tree()
    bare = {
        node.id
        for node in ast.walk(owner_tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    offenders = sorted(bare & _APP_GUI_PATCHED_POLICY_NAMES)
    assert offenders == [], (
        f"gui_menu resolves {offenders} as bare globals — an app_gui-level "
        "rebind of them would be silently ignored by the moved body. Inject "
        "the name across the call boundary (the vulkan preflight pattern) or "
        "adjudicate the split in the patch-couple record"
    )


# --- test-side discipline: the suite must patch the owner, never the no-op ---


def test_suite_never_patches_app_gui_build_menu() -> None:
    """The recording/cast owner nets' test-side discipline, for this cluster.

    ``app_gui.build_menu`` cannot exist (the delegate imports function-
    locally), so an app_gui-level monkeypatch of it would be a silent no-op:
    a suite test doing that looks green while patching a name the moved body
    never reads. Any menu patch belongs on ``the_oracle.gui_menu`` — the
    moved-owner record's designed spelling, and the only place the patch
    intercepts.
    """
    offenders: list[str] = []
    own_file = Path(__file__).name
    for test_file in (REPO_ROOT / "tests").glob("test_*.py"):
        if test_file.name == own_file:
            # This net's source legitimately names the forbidden spelling
            # (that is what an instruction is); the scan takes the SUITE's
            # word for it elsewhere.
            continue
        text = test_file.read_text(encoding="utf-8")
        if '"app_gui.build_menu"' in text or "the_oracle.app_gui.build_menu" in text:
            offenders.append(test_file.name + " (literal path patch)")
        if 'setattr(app_gui, "build_menu"' in text:
            offenders.append(test_file.name + " (setattr(app_gui, ...))")
    assert offenders == [], (
        f"tests patch app_gui.build_menu: {offenders} — the owner function is "
        "imported function-locally inside _build_menu, so an app_gui-level "
        "patch is a silent no-op. Patch the_oracle.gui_menu.build_menu instead"
    )


def test_the_scan_classifies_a_qualified_lookup_as_non_bare() -> None:
    """Vacuity: a qualified `app_gui.THEMES` load must NOT satisfy the bare
    predicate, or the pin above would pass green over the exact spelling it
    exists to refuse."""
    probe = ast.parse("def f():\n    return app_gui.THEMES\n")
    bare = {
        node.id
        for node in ast.walk(probe)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert "THEMES" not in bare


# --- contract 6: the manifest record ----------------------------------------


def test_manifest_records_build_menu_and_nothing_else() -> None:
    moved = json.loads(MANIFEST.read_text(encoding="utf-8"))["moved_owners"]
    assert moved.get(_OWNER, {}).get("names") == [_OWNER_NAME], (
        f"moved_owners.{_OWNER} must list exactly [['{_OWNER_NAME}']]; the "
        "record is what makes app_gui-level patches of it an offender to "
        "repoint"
    )
    # And the recorded name is current: the owner really defines it (the
    # patch-surface net's drift check reads the same record the other way).
    owner, _tree = _gui_menu_tree()
    assert hasattr(owner, _OWNER_NAME)


def test_every_attribute_the_body_sets_is_pinned() -> None:
    """The OLD method is gone, so the moved body IS the assignment surface:
    the pinned set is asserted against gui_menu's own body (an attribute the
    body stops setting must disappear from the pin deliberately, not silently).

    The other half of the delegation surface is the OTHER side of the seam —
    nothing in app_gui references these attributes at MODULE level (they ride
    ``self``/``main_window``), so an app_gui-level reference check would be
    vacuous here; the consumer reads live in MainWindow methods and the suite
    at large, which other nets and the suite itself cover.
    """
    _owner, owner_tree = _gui_menu_tree()
    func_def = next(
        node for node in owner_tree.body if isinstance(node, ast.FunctionDef) and node.name == _OWNER_NAME
    )
    assigned = {
        (node.targets[0].attr if isinstance(node, ast.Assign) else node.target.attr)
        for node in ast.walk(func_def)
        if (isinstance(node, ast.Assign) or isinstance(node, ast.AnnAssign))
        and isinstance(node.targets[0] if isinstance(node, ast.Assign) else node.target, ast.Attribute)
        and isinstance(node.targets[0].value if isinstance(node, ast.Assign) else node.target.value, ast.Name)
        and (node.targets[0].value.id if isinstance(node, ast.Assign) else node.target.value.id) == "main_window"
    }  # type: ignore[union-attr]
    assert assigned == set(_MOVED_BODY_ASSIGNMENTS), (
        "the moved body's main_window.* assignments drifted from the pinned "
        f"record: missing={sorted(set(_MOVED_BODY_ASSIGNMENTS) - assigned)} "
        f"extra={sorted(assigned - set(_MOVED_BODY_ASSIGNMENTS))} — update the "
        "pin first so the new surface is a deliberate act"
    )


# --- contract 7: behavior on a stub window with an INJECTED mapping ----------


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class _StubWindow(QMainWindow):
    """The pieces ``build_menu`` drives, real where Qt needs real.

    QMainWindow (QAction needs a QObject parent), recorder attributes for
    every method the moved body connects to — the probe proves the OWNER is
    self-sufficient: it needs app_gui for nothing but the mapping it is
    HANDED, so a hard-coded look list or an imported callback would fail here
    by name.
    """

    def __init__(self) -> None:
        super().__init__()
        self._app_settings = {"remember_backend": False}
        self.output_filename_warning_enabled = True
        callbacks = (
            "new_project",
            "open_project",
            "save_project",
            "save_project_as",
            "save_settings_profile",
            "load_settings_profile",
            "_batch_fix_input_folder",
            "reset_settings_to_defaults",
            "save_template_profile",
            "_start_inference_wizard",
            "_start_recording_wizard",
            "_on_output_filename_warning_toggled",
            "_rebuild_templates_menu",
            "_enable_delete_confirmation",
            "_clear_trusted_input_files",
            "_restore_most_recent_format_backup",
            "_on_remember_backend_toggled",
            "download_vulkan_model",
            "open_recording_studio",
            "_open_crash_privacy_dialog",
            "_open_license_activation",
            "_open_about_panel",
            "_apply_theme_selection",
        )
        for name in callbacks:
            setattr(self, name, lambda *args, _name=name: None)

    def menuBar(self):  # noqa: D102 - QMainWindow's own method is real here
        return super().menuBar()


class _Tokens:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.key = name


def _stub_mapping() -> dict:
    """A deliberately NON-production look list: if the body built this menu
    from anything but the parameter, the probe below fails by name."""
    return {
        "stub_probe_dark": _Tokens("Stub Probe Dark", "injected mapping for the owner probe"),
        "stub_probe_light": _Tokens("Stub Probe Light", "injected mapping for the owner probe"),
    }


def test_build_menu_is_self_sufficient_on_the_injected_mapping() -> None:
    from PySide6.QtWidgets import QMenu

    from the_oracle.gui_menu import build_menu

    window = _StubWindow()
    build_menu(window, themes=_stub_mapping())
    bar = window.menuBar()
    titles = [action.text() for action in bar.actions()]
    assert "File" in titles and "Settings" in titles and "Theme" in titles and "Help" in titles, (
        f"the moved body did not build the four menus on a stub window: {titles}"
    )
    assert "Recording Studio…" in titles, (
        f"the menu-bar Recording Studio action is missing: {titles}"
    )
    theme_menu = next(menu for menu in bar.findChildren(QMenu) if menu.title() == "Theme")
    labels = [action.text() for action in theme_menu.actions()]
    assert labels == ["Stub Probe Dark", "Stub Probe Light"], (
        f"the Theme menu must iterate the INJECTED mapping exactly, order "
        f"included; got {labels} — a hard-coded look list would fail here"
    )
    assert set(window._theme_actions) == set(_stub_mapping())


def test_build_menu_assignments_land_on_the_window() -> None:
    from the_oracle.gui_menu import build_menu

    window = _StubWindow()
    build_menu(window, themes=_stub_mapping())
    missing = [name for name in _MOVED_BODY_ASSIGNMENTS if not hasattr(window, name)]
    assert missing == [], (
        f"build_menu did not set {missing} on the window — a public action "
        "attribute stranded app_gui or the suite's reads with an AttributeError"
    )
