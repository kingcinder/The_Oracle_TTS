"""Ownership tests for the cast-management extraction (gui_cast.py).

The ninth MainWindow extraction slice (2026-10-05) moved the cast cluster —
``SpeakerGroup`` (one speaker's full voice panel), ``CastManagementDialog``
(the modal whole-cast editor), their shared helpers
``_speaker_settings_from_group`` / ``_apply_speaker_settings_to_group`` and
the dialog's row widget ``_CastRow`` — out of ``app_gui`` with bodies
verbatim, after the 23 pre-flight gates ran green. app_gui re-imports the
split-readership names so its construction sites resolve to identical
objects.

The thirteenth slice (the cast-bar cluster, 2026-10-10) then moved the ten
``MainWindow`` cast-bar methods' bodies here — ``cast_keys``,
``speaker_names``, ``_all_speaker_groups``, ``apply_cast``,
``_update_speaker_group_title``, ``_refresh_cast_bar``,
``open_cast_manager``, ``_sync_extra_speaker_groups``, ``_speaker_settings``,
``_speaker_config_from_payload`` — as module-level functions taking the
window as ``main_window``.

The contract this file pins:

1. IDENTITY — the re-exported names are the gui_cast objects themselves
   (identity is load-bearing: MainWindow constructs/reads them from bare
   names in app_gui's globals, so app_gui-level patches intercept only while
   the names are the same objects — the ``RenderWorker`` precedent). The
   split readership is the patch-surface net's PARTIAL_OWNED record: an
   app_gui-level patch of these names covers only app_gui's sites, because
   the dialog reads them from gui_cast's globals too.
2. DIRECTION — gui_cast imports app_gui in NO spelling. The main window's
   collaborators cross the seam as values (the ``main_window`` parameter),
   never as imports.
3. THE MOVED-OWNER RULE — ``_CastRow`` and, as of the cast-bar slice,
   ``CastManagementDialog`` are defined in gui_cast with NO app_gui name at
   all: the moved ``open_cast_manager`` builds the dialog, and the dialog
   builds its rows, from gui_cast's globals, so an app_gui-level patch of
   either would be a silent no-op. Both are listed in
   ``moved_owners.the_oracle.gui_cast``, MainWindow never references either
   in any form, and any suite-side patch must target gui_cast — the designed
   spelling — never app_gui.
4. THE TEN DELEGATES + THE FOUR INJECTIONS — each cast-bar method stays in
   app_gui as a thin delegate whose body imports its owner function
   FUNCTION-LOCALLY (the settled entry-point delegate convention), so
   app_gui holds no module-level binding and gui_cast is the single patch
   point. Four names a staying app_gui site still reads (``SpeakerGroup``,
   ``_speaker_settings_from_group``, ``_apply_speaker_settings_to_group``,
   ``normalize_cast_keys``) cross the boundary as INJECTIONS resolved from
   app_gui's globals at call time; the moved bodies must never read them as
   their own globals, or an app_gui-level rebind would silently miss the
   moved half of the flow (the harm class these probes drive).

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
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "scripts" / "patch_surface_manifest.json"

#: The cluster names app_gui re-imports (identity contract 1).
_REEXPORTED_CLUSTER = (
    "SpeakerGroup",
    "_speaker_settings_from_group",
    "_apply_speaker_settings_to_group",
)

#: The dialog-only row widget (contract 3): defined in gui_cast, never
#: re-exported, never referenced by MainWindow.
_MOVED_WORKER = "_CastRow"

#: The modal cast editor: moved-owned as of the cast-bar cluster slice
#: (2026-10-10) — its only app_gui runtime reader was ``open_cast_manager``,
#: whose body moved, so app_gui holds no binding for it at all.
_MOVED_DIALOG = "CastManagementDialog"

#: The ten MainWindow cast-bar methods (contract 4): bodies moved to gui_cast
#: as module-level functions; their app_gui selves are thin delegates.
_CLUSTER_METHODS = (
    "cast_keys",
    "speaker_names",
    "_all_speaker_groups",
    "apply_cast",
    "_update_speaker_group_title",
    "_refresh_cast_bar",
    "open_cast_manager",
    "_sync_extra_speaker_groups",
    "_speaker_settings",
    "_speaker_config_from_payload",
)

#: Owner-function keyword -> the app_gui-global name the delegate injects
#: (contract 4). A moved body reading one of these as a module global would
#: turn every app_gui-level rebind into a silent no-op for the moved path.
_DELEGATE_INJECTIONS = {
    "_sync_extra_speaker_groups": {"speaker_group_cls": "SpeakerGroup"},
    "_speaker_settings": {
        "settings_from_group_fn": "_speaker_settings_from_group",
    },
    "apply_cast": {
        "apply_settings_fn": "_apply_speaker_settings_to_group",
        "normalize_keys_fn": "normalize_cast_keys",
    },
}

#: Every injected app_gui-global name, as one set.
_INJECTED_NAMES = frozenset(
    name for mapping in _DELEGATE_INJECTIONS.values() for name in mapping.values()
)

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
    for name in (
        "_speaker_settings_from_group",
        "_apply_speaker_settings_to_group",
        *_CLUSTER_METHODS,
    ):
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


def test_main_window_constructs_speaker_groups_from_bare_names_only() -> None:
    """Every app_gui construction of the re-exported cluster class must use
    the bare app_gui-global name; the two settings helpers must be read bare
    too. A qualified ``gui_cast.X`` lookup would go deaf to app_gui-level
    patches. ``CastManagementDialog`` is deliberately absent: the cast-bar
    slice moved its last app_gui constructor into gui_cast."""
    app_gui = _import("the_oracle.app_gui")
    tree = ast.parse(Path(app_gui.__file__).read_text(encoding="utf-8"))
    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SpeakerGroup"
    ]
    assert sites, "no SpeakerGroup construction found in app_gui — the pin went blind"
    for node in sites:
        assert isinstance(node.func, ast.Name), (
            f"line {node.lineno}: SpeakerGroup must be constructed from the "
            "bare app_gui-global name"
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
    # And the manifest records exactly that disposition: the row widget and
    # (since the cast-bar slice) the modal dialog.
    moved = json.loads(MANIFEST.read_text(encoding="utf-8"))["moved_owners"][
        "the_oracle.gui_cast"
    ]["names"]
    assert moved == [_MOVED_WORKER, _MOVED_DIALOG], (
        "moved_owners.the_oracle.gui_cast must list exactly the two names no "
        "app_gui site reads at runtime; the split-readership names stay in "
        f"PARTIAL_OWNED, got {moved}"
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


def test_the_moved_dialog_has_no_app_gui_name_and_no_window_reference() -> None:
    """The CastManagementDialog moved-owner rule (cast-bar slice, 2026-10-10):
    gui_cast defines it, app_gui has NO name for it at all, and MainWindow
    never constructs or mentions it in any form — the moved
    ``open_cast_manager`` builds it from gui_cast's globals, so an
    app_gui-level patch would be a silent no-op."""
    owner, tree = _gui_cast_tree()
    defined = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert _MOVED_DIALOG in defined, (
        f"gui_cast must define {_MOVED_DIALOG} — the pin went blind"
    )
    app_gui = _import("the_oracle.app_gui")
    assert not hasattr(app_gui, _MOVED_DIALOG), (
        f"app_gui must NOT re-export {_MOVED_DIALOG}: the moved "
        "open_cast_manager constructs it from gui_cast's globals, so an "
        "app_gui-level patch would be a silent no-op"
    )
    app_tree = ast.parse(Path(app_gui.__file__).read_text(encoding="utf-8"))
    # Runtime reads only: MainWindow.__init__ still carries the
    # ``set[CastManagementDialog]`` *annotation* for the dialog tracker (no
    # runtime effect, so no app_gui-level rebind can be missed there — the
    # reason the pre-flight counted this name as movable).
    assert _MOVED_DIALOG not in _runtime_reads(app_tree), (
        f"app_gui reads {_MOVED_DIALOG} in a runtime position — it is "
        "moved-owned; only the __init__ annotation may name it"
    )


def test_every_cast_bar_method_is_a_function_local_import_delegate() -> None:
    """Contract 4, shape half: each of the ten cast-bar methods survives in
    app_gui as a thin delegate whose body imports its owner function
    FUNCTION-LOCALLY (the settled entry-point delegate convention). app_gui
    therefore holds no module-level binding for any of the ten — an
    app_gui-level patch is a silent no-op and gui_cast is the single patch
    point, which is what makes the manifest's listing literal."""
    app_gui = _import("the_oracle.app_gui")
    tree = ast.parse(Path(app_gui.__file__).read_text(encoding="utf-8"))
    window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    methods = {m.name: m for m in window.body if isinstance(m, ast.FunctionDef)}
    module_bindings = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for name in _CLUSTER_METHODS:
        assert name in methods, f"MainWindow lost the {name} delegate"
        local_imports = [
            f"{node.module}.{alias.name}"
            for node in ast.walk(methods[name])
            if isinstance(node, ast.ImportFrom) and node.module == "the_oracle.gui_cast"
            for alias in node.names
        ]
        assert f"the_oracle.gui_cast.{name}" in local_imports, (
            f"{name} must import its owner function FUNCTION-LOCALLY "
            f"(found {local_imports})"
        )
        assert name not in module_bindings, (
            f"app_gui holds a module-level binding for {name} — an "
            "app_gui-level patch would then look live while gui_cast stays the "
            "real patch point"
        )


def test_the_four_injection_names_cross_the_boundary_and_never_sit_in_gui_cast() -> None:
    """Contract 4, seam half (AST): every injected name is declared as a
    keyword parameter of its owner function, passed by keyword from the
    delegate body out of app_gui's globals, and — the harm this closes —
    never read as a module global inside the moved bodies."""
    owner, tree = _gui_cast_tree()
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in _CLUSTER_METHODS:
        assert name in funcs, f"gui_cast lost the {name} owner function"

    app_gui = _import("the_oracle.app_gui")
    app_source = Path(app_gui.__file__).read_text(encoding="utf-8")
    app_tree = ast.parse(app_source)
    app_window = next(
        node
        for node in app_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    app_methods = {m.name: m for m in app_window.body if isinstance(m, ast.FunctionDef)}

    for method_name, mapping in _DELEGATE_INJECTIONS.items():
        owner_fn = funcs[method_name]
        keyword_params = {arg.arg for arg in owner_fn.args.kwonlyargs}
        missing = set(mapping) - keyword_params
        assert not missing, (
            f"gui_cast.{method_name} must declare {sorted(missing)} as "
            "keyword-only parameters"
        )
        delegate_source = ast.get_source_segment(app_source, app_methods[method_name])
        for keyword, global_name in mapping.items():
            assert f"{keyword}={global_name}" in delegate_source, (
                f"the {method_name} delegate must pass {keyword}={global_name} "
                "from app_gui's globals"
            )

    for name in _CLUSTER_METHODS:
        loads = _runtime_reads(funcs[name])
        offenders = sorted(loads & _INJECTED_NAMES)
        assert offenders == [], (
            f"gui_cast.{name} reads {offenders} as module globals — inject the "
            "name at the call boundary instead, or an app_gui-level rebind is "
            "a silent no-op for this flow"
        )


def _runtime_reads(node: ast.AST) -> set[str]:
    """Names a node loads in runtime position (annotations are not reads)."""
    from tests.test_app_gui_patch_surface import _RuntimeReads

    visitor = _RuntimeReads()
    visitor.visit(node)
    return visitor.loads


class _StubWindow:
    """The minimum window surface the moved cast-bar bodies touch.

    The moved functions take the window as a plain parameter, so the seam
    probes below can drive them without constructing a real MainWindow (and
    without Qt): what they verify is exactly the boundary an app_gui-level
    rebind has to cross, independent of the widget tree behind it.
    """

    def __init__(self) -> None:
        self._cast = ["A", "B"]
        self._speaker_names = {"A": "Ada", "B": ""}
        self.speaker_a = _StubGroup()
        self.speaker_b = _StubGroup()
        self.extra_speaker_groups: dict[str, object] = {}
        self.extra_speaker_layout = _StubLayout()
        self.extra_voices_section = _StubLayout()
        self.monologue_check = _StubCheck()
        self.paths = SimpleNamespace(voice_dir=Path("/tmp"))
        self.variant_combo = _StubRead("Narrator")
        self.crossfade_spin = _StubRead(120)
        self._cast_dialogs: set[object] = set()
        self.refreshed = 0
        self.titles: dict[str, str] = {}

    def _all_speaker_groups(self) -> dict[str, object]:
        from the_oracle.gui_cast import _all_speaker_groups

        return _all_speaker_groups(self)

    def _update_speaker_group_title(self, key: str, group) -> None:
        self.titles[key] = f"Speaker {key}"

    def _sync_extra_speaker_groups(self, speakers) -> None:
        # Mirror the real window: route through the app_gui delegate so the
        # seam probe exercises the injection exactly as production does.
        app_gui = _import("the_oracle.app_gui")
        app_gui.MainWindow._sync_extra_speaker_groups(self, speakers)

    def _refresh_cast_bar(self) -> None:
        self.refreshed += 1

    def _refresh_reference_pickers(self) -> None:
        self.refreshed += 1

    def _refresh_language_options(self) -> None:
        self.refreshed += 1

    def _save_blend_as_voice(self, group) -> None:
        pass

    def _section_content_built(self, key: str) -> bool:
        return True


class _StubGroup:
    def is_content_built(self) -> bool:
        return True

    def setTitle(self, text: str) -> None:  # noqa: N802 - Qt casing
        self.title = text

    def setVisible(self, visible: bool) -> None:  # noqa: N802 - Qt casing
        self.visible = visible

    def deleteLater(self) -> None:  # noqa: N802 - Qt casing
        pass


class _StubLayout:
    def addWidget(self, widget) -> None:  # noqa: N802 - Qt casing
        pass

    def removeWidget(self, widget) -> None:  # noqa: N802 - Qt casing
        pass

    def hide(self) -> None:
        pass


class _StubCheck:
    def setChecked(self, value: bool) -> None:  # noqa: N802 - Qt casing
        self.checked = value


class _StubRead:
    def __init__(self, value) -> None:
        self._value = value

    def currentText(self) -> str:  # noqa: N802 - Qt casing
        return self._value

    def value(self) -> int:
        return self._value


def test_the_moved_bodies_use_the_collaborators_they_are_handed() -> None:
    """Contract 4, behavior half (owner side): each moved body must USE the
    collaborator it was handed, never a module global of its own — inject
    recording fakes and watch them run.

    The app_gui-level rebind that has to reach them through the delegate is
    proven by the slice's EPHEMERAL seam probe instead of a durable test:
    the patch-couple census is derived from this suite, so a lasting
    app_gui-level rebind of these names would be the patch-surface net's
    business (and that net refuses app_gui-level patches of the three
    PARTIAL_OWNED ones outright — the split is the design, not the test's
    target). What this test can pin without perturbing the census is that the
    injection, once made, is what runs."""
    owner = _import("the_oracle.gui_cast")
    window = _StubWindow()

    built: list[str] = []

    class FakeGroup:
        def __init__(self, key, voice_dir, on_save_blend=None, on_process_voice=None) -> None:
            built.append(key)

        def setTitle(self, text: str) -> None:  # noqa: N802 - Qt casing
            self.title = text

        def deleteLater(self) -> None:  # noqa: N802 - Qt casing
            pass

    owner._sync_extra_speaker_groups(window, ["C"], speaker_group_cls=FakeGroup)
    assert built == ["C"], (
        "the moved _sync_extra_speaker_groups ignored the class it was handed "
        f"(constructed {built})"
    )

    seen: list[tuple] = []

    def fake_settings(group, variant, crossfade_ms):
        seen.append((group, variant, crossfade_ms))
        return f"settings:{variant}:{crossfade_ms}"

    settings = owner._speaker_settings(window, settings_from_group_fn=fake_settings)
    assert [entry[1:] for entry in seen] == [("Narrator", 120), ("Narrator", 120)], (
        f"the moved _speaker_settings ignored the helper it was handed: {seen}"
    )
    assert set(settings) == {"A", "B"}

    applied: list[tuple] = []
    normalized: list[list[str]] = []

    def fake_apply(group, panel_settings) -> None:
        applied.append((group, panel_settings))

    def fake_normalize(keys):
        normalized.append(list(keys))
        return list(keys)

    owner.apply_cast(
        window,
        ["A", "B"],
        {"A": "Ada"},
        {"A": "panel"},
        apply_settings_fn=fake_apply,
        normalize_keys_fn=fake_normalize,
    )
    assert normalized == [["A", "B"]], (
        f"the moved apply_cast ignored normalize_keys_fn: {normalized}"
    )
    assert [entry[1] for entry in applied] == ["panel"], (
        f"the moved apply_cast ignored apply_settings_fn: {applied}"
    )
    assert window.titles.get("A") == "Speaker A"


def test_the_moved_cast_dialog_construction_resolves_gui_cast_globals(monkeypatch) -> None:
    """Contract 3, behavior half: patching ``gui_cast.CastManagementDialog``
    — the designed spelling (patching it at app_gui is impossible: there is
    no name) — is what the moved ``open_cast_manager`` constructs, driven
    through the app_gui delegate so the whole path is exercised."""
    owner = _import("the_oracle.gui_cast")
    app_gui = _import("the_oracle.app_gui")
    built: list[object] = []

    class FakeDialog:
        def __init__(self, main_window) -> None:
            built.append(main_window)
            self.finished = SimpleNamespace(connect=lambda _fn: None)

        def raise_(self) -> None:
            pass

        def activateWindow(self) -> None:  # noqa: N802 - Qt casing
            pass

        def exec(self) -> None:
            pass

    assert not hasattr(app_gui, _MOVED_DIALOG)
    monkeypatch.setattr(owner, _MOVED_DIALOG, FakeDialog)
    window = _StubWindow()
    app_gui.MainWindow.open_cast_manager(window)
    assert built == [window], (
        "the moved open_cast_manager must build its dialog from gui_cast's "
        f"globals (built {built})"
    )


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
