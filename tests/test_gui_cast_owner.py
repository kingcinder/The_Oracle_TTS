"""Ownership tests for the cast-management extraction (gui_cast.py).

The ninth MainWindow extraction slice (2026-10-05) moved the cast cluster —
``SpeakerGroup`` (one speaker's full voice panel), ``CastManagementDialog``
(the modal whole-cast editor), their shared helpers
``_speaker_settings_from_group`` / ``_apply_speaker_settings_to_group`` and
the dialog's row widget ``_CastRow`` — out of ``app_gui`` with bodies
verbatim, after the 23 pre-flight gates ran green. app_gui re-imports the
four cluster names so its construction sites resolve to identical objects.

The contract this file pins:

1. IDENTITY — the four re-exported names are the gui_cast objects themselves
   (identity is load-bearing: MainWindow constructs/reads them from bare
   names in app_gui's globals, so app_gui-level patches intercept only while
   the names are the same objects — the ``RenderWorker`` precedent). The
   split readership is the patch-surface net's PARTIAL_OWNED record: an
   app_gui-level patch of these names covers only app_gui's sites, because
   CastManagementDialog reads them from gui_cast's globals too.
2. DIRECTION — gui_cast imports app_gui in NO spelling. The main window's
   collaborators cross the seam as values (the ``main_window`` parameter),
   never as imports.
3. THE _CASTROW MOVED-WORKER RULE — ``_CastRow`` is dialog-only: the dialog
   builds its rows from gui_cast's globals, so an app_gui-level patch of it
   would be a silent no-op. It is therefore NOT re-exported (app_gui has no
   ``_CastRow`` name at all), it is the sole entry in
   ``moved_owners.the_oracle.gui_cast``, MainWindow never references it in
   any form, and any suite-side patch of it must target gui_cast — the
   designed spelling — never app_gui.

The exact-equality of gui_cast's sanctioned bare names (QDialog, QMessageBox,
blend_voice_choices, default_voice_choices, load_recent_reference_paths)
belongs to the patch-couple net in test_app_gui_patch_surface.py; this file
pins the cluster's own owner-side shape.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "scripts" / "patch_surface_manifest.json"

#: The cluster names app_gui re-imports (identity contract 1).
_REEXPORTED_CLUSTER = (
    "CastManagementDialog",
    "SpeakerGroup",
    "_speaker_settings_from_group",
    "_apply_speaker_settings_to_group",
)

#: The dialog-only row widget (contract 3): defined in gui_cast, never
#: re-exported, never referenced by MainWindow.
_MOVED_WORKER = "_CastRow"

#: App_gui-resident names the suite patches at app_gui module level and that
#: a moved cast body would have to receive as a VALUE, never resolve as a
#: bare global (such a read resolves from gui_cast's own namespace and is
#: invisible to app_gui-level monkeypatching — the harm class). Names
#: sanctioned for gui_cast by the couple net (QDialog, QMessageBox,
#: blend_voice_choices, default_voice_choices, load_recent_reference_paths)
#: are deliberately absent: they are gui_cast-owned reads by design.
_APP_GUI_PATCHED_POLICY_NAMES = frozenset(
    {
        "OraclePipeline",
        "ChatterboxEngine",
        "RenderWorker",
        "PreviewWorker",
        "RecordingStudioDialog",
        "RecordStudioWorker",
        "VulkanPreflightThread",
        "ModelDownloadThread",
        "find_audiocpp_binary",
        "find_audiocpp_model",
        "_vulkan_preflight_report",
        "_vulkan_prerequisite_missing",
        "QMediaPlayer",
        "QAudioOutput",
    }
)


def _import(name: str):
    return importlib.import_module(name)


def _gui_cast_tree():
    owner = _import("the_oracle.gui_cast")
    source = Path(owner.__file__).read_text(encoding="utf-8")
    return owner, ast.parse(source)


def test_cluster_lives_in_gui_cast() -> None:
    """The cluster defines itself in gui_cast — a stub or re-export cannot
    pass, because the pins read the module's own AST and class bases."""
    owner, tree = _gui_cast_tree()
    defined_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    defined_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("SpeakerGroup", "CastManagementDialog", _MOVED_WORKER):
        assert name in defined_classes, f"gui_cast must define class {name}"
    for name in ("_speaker_settings_from_group", "_apply_speaker_settings_to_group"):
        assert name in defined_functions, f"gui_cast must define {name}"
    # Behavior probe: the real bases travelled with the bodies.
    from PySide6.QtWidgets import QDialog  # noqa: PLC0415 - probe of the real class

    assert issubclass(owner.SpeakerGroup, QDialog) is False
    assert issubclass(owner.CastManagementDialog, QDialog), (
        "CastManagementDialog must still be the modal QDialog it was in app_gui"
    )


def test_app_gui_reimports_the_cluster_as_identical_objects() -> None:
    """Identity is load-bearing: MainWindow constructs/reads the cluster from
    bare names in app_gui's globals, so app_gui-level class patches intercept
    only while app_gui.X IS gui_cast.X — a wrapper or subclass would silently
    defeat them while the suite stayed green."""
    owner = _import("the_oracle.gui_cast")
    app_gui = _import("the_oracle.app_gui")
    for name in _REEXPORTED_CLUSTER:
        assert getattr(app_gui, name) is getattr(owner, name), (
            f"app_gui.{name} must be the gui_cast object itself"
        )


def test_main_window_constructs_and_reads_the_cluster_from_bare_names() -> None:
    """Every app_gui construction of the cluster classes must use the bare
    app_gui-global name; the two settings helpers must be read bare too. A
    qualified ``gui_cast.X`` lookup would go deaf to app_gui-level patches."""
    app_gui = _import("the_oracle.app_gui")
    tree = ast.parse(Path(app_gui.__file__).read_text(encoding="utf-8"))
    constructions = {
        name: [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        ]
        for name in ("SpeakerGroup", "CastManagementDialog")
    }
    for name, sites in constructions.items():
        assert sites, f"no {name} construction found in app_gui — the pin went blind"
        for node in sites:
            assert isinstance(node.func, ast.Name), (
                f"line {node.lineno}: {name} must be constructed from the bare "
                "app_gui-global name"
            )
    bare_helper_reads = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in {"_speaker_settings_from_group", "_apply_speaker_settings_to_group"}
    }
    assert bare_helper_reads == {"_speaker_settings_from_group", "_apply_speaker_settings_to_group"}, (
        "the settings helpers must still be read from app_gui's globals — "
        f"missing: {bare_helper_reads}"
    )


def test_gui_cast_imports_nothing_from_app_gui() -> None:
    """No import cycle and no seam bypass: gui_cast must never reach back
    into app_gui — in ANY spelling. The main window crosses the seam as a
    value (the ``main_window`` parameter); someone "fixing" a missing name
    with ``from the_oracle.app_gui import X`` would plant a cycle under the
    next slice's feet."""
    from tests.test_gui_import_direction import _module_root, _scan_imports

    _forbids_app_gui = lambda module: _module_root(module) == "app_gui"  # noqa: E731
    owner = _import("the_oracle.gui_cast")
    source = Path(owner.__file__).read_text(encoding="utf-8")
    scan = _scan_imports(source, forbidden=_forbids_app_gui)
    assert scan.violations == () and scan.unresolved == (), (
        "gui_cast must not import app_gui in any form:\n  "
        + "\n  ".join(scan.violations + scan.unresolved)
    )
    # Vacuity: each forbidden spelling must still trip this exact gate.
    for form in (
        "import the_oracle.app_gui",
        "import app_gui",
        "from the_oracle.app_gui import CastManagementDialog",
        "from the_oracle import app_gui",
        "from . import app_gui",
        "from .app_gui import SpeakerGroup",
        "importlib.import_module('the_oracle.app_gui')",
    ):
        assert _scan_imports(form, forbidden=_forbids_app_gui).violations, form
    # ...and the scan must see gui_cast's real imports (which are legal):
    assert {
        "the_oracle.gui_sections",
        "the_oracle.gui_settings",
        "the_oracle.gui_utils",
        "the_oracle.gui_widgets",
        "the_oracle.models.project",
        "the_oracle.models.settings",
        "the_oracle.voice_catalog",
    } <= set(scan.modules), scan.modules


def test_moved_worker_is_dialog_only_and_never_reexported() -> None:
    """The _CastRow moved-worker rule, ownership half: the row widget defines
    in gui_cast and app_gui has NO ``_CastRow`` name at all — a re-export
    would invite app_gui-level patches that the dialog (which resolves the
    name from gui_cast's globals) would silently never see."""
    owner, tree = _gui_cast_tree()
    defined = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert _MOVED_WORKER in defined, (
        f"gui_cast must define {_MOVED_WORKER} — the pin went blind"
    )
    app_gui = _import("the_oracle.app_gui")
    assert not hasattr(app_gui, _MOVED_WORKER), (
        "app_gui must NOT re-export _CastRow: the dialog builds rows from "
        "gui_cast's globals, so an app_gui-level patch would be a silent no-op"
    )
    # And the manifest records exactly that disposition:
    moved = json.loads(MANIFEST.read_text(encoding="utf-8"))["moved_owners"][
        "the_oracle.gui_cast"
    ]["names"]
    assert moved == [_MOVED_WORKER], (
        "moved_owners.the_oracle.gui_cast must list exactly the dialog-only "
        f"row widget; the split-readership names stay in PARTIAL_OWNED, got {moved}"
    )


def test_dialog_builds_rows_from_gui_cast_globals() -> None:
    """The _CastRow moved-worker rule, construction half: every row
    construction inside CastManagementDialog uses the bare gui_cast-global
    name. A qualified lookup (or an injected re-route through app_gui) would
    move the patch surface without anyone deciding to."""
    owner, tree = _gui_cast_tree()
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    dialog = classes["CastManagementDialog"]
    constructions = [
        node
        for node in ast.walk(dialog)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _MOVED_WORKER
    ]
    assert constructions, (
        f"no {_MOVED_WORKER} construction found in CastManagementDialog — "
        "the moved-worker pin went blind"
    )
    # Vacuity: a qualified spelling must NOT satisfy the bare-name predicate —
    # otherwise this pin passes green over the exact drift it exists to refuse.
    probe = ast.parse(
        "class CastManagementDialog:\n"
        "    def _row(self):\n"
        f"        return gui_cast.{_MOVED_WORKER}('A', None)\n"
    ).body[0]
    probe_bare = [
        node
        for node in ast.walk(probe)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _MOVED_WORKER
    ]
    assert probe_bare == [], "the vacuity probe must classify a qualified lookup as non-bare"


def test_main_window_never_references_the_moved_worker() -> None:
    """MainWindow must not read, construct, or attribute-lookup ``_CastRow``
    in any form: the row widget is the dialog's private implementation. A
    window-side reference would mean rows moved back across the seam without
    the manifest's moved_owners record being true anymore."""
    app_gui = _import("the_oracle.app_gui")
    tree = ast.parse(Path(app_gui.__file__).read_text(encoding="utf-8"))
    offenders = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if (isinstance(node, ast.Name) and node.id == _MOVED_WORKER)
        or (isinstance(node, ast.Attribute) and node.attr == _MOVED_WORKER)
    ]
    assert offenders == [], (
        f"app_gui references {_MOVED_WORKER}: {offenders} — it is dialog-only"
    )
    # Vacuity: the same predicate flags an attribute reference (the form a
    # "harmless" gui_cast._CastRow re-use would take).
    probe = ast.parse(f"x = gui_cast.{_MOVED_WORKER}\n")
    assert any(
        isinstance(node, ast.Attribute) and node.attr == _MOVED_WORKER
        for node in ast.walk(probe)
    )


def test_suite_patches_the_moved_worker_only_on_the_owner() -> None:
    """Any suite-side patch of ``_CastRow`` must target gui_cast — the
    designed spelling (the dialog reads it there). An app_gui-level patch
    looks green while faking a module the dialog never reads.

    Vacuity: the classifier is proven on synthetic patch sites so a silent
    widening of the predicate cannot hide a real offender later."""
    offenders: list[str] = []
    found_owner_side = 0
    for test_file in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "setattr" or len(node.args) < 2:
                continue
            target, name_arg = node.args[0], node.args[1]
            if not (isinstance(name_arg, ast.Constant) and name_arg.value == _MOVED_WORKER):
                continue
            if isinstance(target, ast.Name) and target.id == "gui_cast":
                found_owner_side += 1
                continue
            if isinstance(target, ast.Name) and target.id == "app_gui":
                offenders.append(f"{test_file.name}: line {node.lineno} patches app_gui.{_MOVED_WORKER}")
    assert offenders == [], (
        "the moved worker must be patched on the_oracle.gui_cast, never "
        "app_gui:\n  " + "\n  ".join(offenders)
    )
    # Classifier vacuity, both directions:
    owner_side = ast.parse("monkeypatch.setattr(gui_cast, '_CastRow', fake)\n")
    app_gui_side = ast.parse("monkeypatch.setattr(app_gui, '_CastRow', fake)\n")

    def classify(tree_):
        for node in ast.walk(tree_):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == _MOVED_WORKER
            ):
                return node.args[0].id
        return None

    assert classify(owner_side) == "gui_cast"
    assert classify(app_gui_side) == "app_gui"


def test_cast_bodies_never_resolve_app_gui_patched_policy_names() -> None:
    """The harm-class pin: EVERY top-level body in gui_cast — the moved
    cluster and anything added beside it later — must not read an
    app_gui-resident, suite-patched policy name as a bare global; such a
    read resolves from gui_cast's own namespace and is invisible to
    app_gui-level monkeypatching. Collaborators cross as values (the
    ``main_window`` parameter, the ``on_save_blend`` callback)."""
    owner, tree = _gui_cast_tree()
    scopes = [
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    defined = {node.name for node in scopes}
    for name in (*_REEXPORTED_CLUSTER, _MOVED_WORKER):
        assert name in defined, f"{name} must be findable in gui_cast — the scan went blind"
    offenders: list[str] = []
    for scope in scopes:
        for node in ast.walk(scope):
            if isinstance(node, ast.Name) and node.id in _APP_GUI_PATCHED_POLICY_NAMES:
                offenders.append(f"{scope.name}: {node.id} (line {node.lineno})")
    assert offenders == [], (
        "cast bodies resolve app_gui-patched names from their own globals; "
        "inject the collaborator as a value or patch the owner:\n  "
        + "\n  ".join(offenders)
    )
    # Vacuity: a blind walk passes green over anything — the scan must flag a
    # synthetic offender.
    probe = ast.parse(
        "class P:\n    def run(self):\n        return OraclePipeline()\n"
    ).body[0]
    flagged = [
        node.id
        for node in ast.walk(probe)
        if isinstance(node, ast.Name) and node.id in _APP_GUI_PATCHED_POLICY_NAMES
    ]
    assert flagged == ["OraclePipeline"]
