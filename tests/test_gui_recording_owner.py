"""Ownership tests for the Recording Studio extraction (gui_recording.py).

The eighth MainWindow extraction slice (2026-09-28) moved the record-a-
Seashell cluster — ``RecordStudioWorker`` (the sounddevice capture thread)
and ``RecordingStudioDialog`` (teleprompter, mic/rate pickers, level meter,
audition player) — out of ``app_gui`` with bodies verbatim, after both
pre-flight nets (the patch-surface net and the payload-policy net) ran green.

The contract this file pins:

1. ONE OWNER — the dialog defines itself in ``gui_recording`` and
   ``app_gui`` re-imports the identical object (identity is load-bearing:
   MainWindow constructs the dialog from a bare name in app_gui's globals,
   so app_gui-level class patches intercept only while the names are the
   same objects — the ``RenderWorker`` precedent from slice 4).
2. MOVED-WORKER RULE — ``RecordStudioWorker`` is dialog-only (the
   ``_CastRow`` precedent): the manifest lists it, the dialog builds it
   from gui_recording's globals, app_gui does NOT re-export it (the
   original slice's dead re-export was removed 2026-10-08 — it only
   invited app_gui-level patches that would be silent no-ops), MainWindow
   never references it, and any suite patch targets the owner.
3. DIRECTION — gui_recording imports app_gui in NO spelling. The studio's
   MainWindow collaborators cross the seam as values (constructor arguments,
   the ``on_assign`` callback), never as imports.
4. HARM CLASS — a moved body must not read an app_gui-patched name as a bare
   global. The two media classes are the sanctioned exception BY DESIGN: the
   dialog resolves ``QMediaPlayer``/``QAudioOutput`` from gui_recording's own
   globals, and the dialog tests patch them THERE (the window-fixture tests
   keep patching app_gui for the window's own preview player).
5. TEST-SIDE DISCIPLINE — app_gui-level media patches in the studio's test
   file may exist only inside the MainWindow fixture; any dialog test that
   quietly repatches app_gui for the dialog's player would look green while
   actually patching a module the moved dialog never reads.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "scripts" / "patch_surface_manifest.json"

#: The dialog-only worker (the _CastRow moved-worker precedent): defined in
#: gui_recording, constructed only by the dialog's body from its own globals,
#: never re-exported, never referenced by MainWindow.
_MOVED_WORKER = "RecordStudioWorker"

#: App_gui-level names the dialog tests (or window fixtures) patch, from the
#: vantage of a body that might wrongly resolve them as bare globals. The
#: media pair is sanctioned: it is read from gui_recording's own globals by
#: design and patched there (contract 4/5 above).
_APP_GUI_PATCHED_POLICY_NAMES = frozenset(
    {
        "QMediaPlayer",
        "QAudioOutput",
        "OraclePipeline",
        "ChatterboxEngine",
        "find_audiocpp_binary",
        "VulkanPreflightThread",
        "ModelDownloadThread",
    }
)

#: The media classes the moved dialog resolves from its own module globals —
#: the bare-name reads the harm-class scan must tolerate.
_SANCTIONED_BARE_NAMES = frozenset({"QMediaPlayer", "QAudioOutput"})


def _import(name: str):
    return importlib.import_module(name)


def _gui_recording_tree():
    owner = _import("the_oracle.gui_recording")
    source = Path(owner.__file__).read_text(encoding="utf-8")
    return owner, ast.parse(source)


def test_patch_free_dialog_lives_in_gui_recording() -> None:
    owner = _import("the_oracle.gui_recording")
    for name in ("RecordStudioWorker", "RecordingStudioDialog"):
        assert hasattr(owner, name), name
    # Behavior probe so the pin cannot pass over a stub module.
    assert callable(owner.RecordingStudioDialog.apply_setup_preferences)
    assert callable(owner.RecordingStudioDialog.set_assign_speakers)


def test_app_gui_delegates_the_recording_cluster_to_the_owner() -> None:
    """Identity is load-bearing for the dialog: MainWindow constructs it from
    a bare name in app_gui's globals, so app_gui-level class patches intercept
    only while app_gui.RecordingStudioDialog IS gui_recording's object — a
    wrapper or subclass would silently defeat those patches while the suite
    stayed green."""
    owner = _import("the_oracle.gui_recording")
    app_gui = _import("the_oracle.app_gui")
    assert app_gui.RecordingStudioDialog is owner.RecordingStudioDialog


def test_moved_worker_is_dialog_only_and_never_reexported() -> None:
    """The RecordStudioWorker moved-worker rule (the _CastRow precedent):
    the worker defines in gui_recording and app_gui has NO such name at all.
    The original slice re-exported it, but MainWindow never constructs it —
    the dialog's body does, resolving the name from gui_recording's globals —
    so a re-export only invites app_gui-level patches that would be silent
    no-ops. The dead import was removed 2026-10-08 and this pin refuses its
    return in any form."""
    owner, tree = _gui_recording_tree()
    defined = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert _MOVED_WORKER in defined, (
        f"gui_recording must define {_MOVED_WORKER} — the pin went blind"
    )
    app_gui = _import("the_oracle.app_gui")
    assert not hasattr(app_gui, _MOVED_WORKER), (
        f"app_gui must NOT re-export {_MOVED_WORKER}: the dialog builds its "
        "worker from gui_recording's globals, so an app_gui-level patch "
        "would be a silent no-op"
    )
    # And the manifest records exactly that disposition:
    moved = json.loads(MANIFEST.read_text(encoding="utf-8"))["moved_owners"][
        "the_oracle.gui_recording"
    ]["names"]
    assert moved == [_MOVED_WORKER], (
        "moved_owners.the_oracle.gui_recording must list exactly the "
        f"dialog-only worker; got {moved}"
    )


def test_dialog_builds_its_worker_from_gui_recording_globals() -> None:
    """The moved-worker rule, construction half: every worker construction
    inside RecordingStudioDialog uses the bare gui_recording-global name.
    A qualified lookup (or an injected re-route through app_gui) would move
    the patch surface without anyone deciding to."""
    owner, tree = _gui_recording_tree()
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    dialog = classes["RecordingStudioDialog"]
    constructions = [
        node
        for node in ast.walk(dialog)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _MOVED_WORKER
    ]
    assert constructions, (
        f"no {_MOVED_WORKER} construction found in RecordingStudioDialog — "
        "the moved-worker pin went blind"
    )
    # Vacuity: a qualified spelling must NOT satisfy the bare-name predicate —
    # otherwise this pin passes green over the exact drift it exists to refuse.
    probe = ast.parse(
        "class RecordingStudioDialog:\n"
        "    def _start(self):\n"
        f"        self._worker = gui_recording.{_MOVED_WORKER}('A', None)\n"
    ).body[0]
    probe_bare = [
        node
        for node in ast.walk(probe)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _MOVED_WORKER
    ]
    assert probe_bare == [], "the vacuity probe must classify a qualified lookup as non-bare"


def test_main_window_never_references_the_moved_worker() -> None:
    """MainWindow must not read, construct, or attribute-lookup the worker
    in any form: it is the dialog's private implementation. A window-side
    reference would mean the worker moved back across the seam without the
    manifest's moved_owners record being true anymore."""
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
    # "harmless" gui_recording.RecordStudioWorker re-use would take).
    probe = ast.parse(f"x = gui_recording.{_MOVED_WORKER}\n")
    assert any(
        isinstance(node, ast.Attribute) and node.attr == _MOVED_WORKER
        for node in ast.walk(probe)
    )


def test_suite_patches_the_moved_worker_only_on_the_owner() -> None:
    """Any suite-side patch of the worker must target gui_recording — the
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
            if isinstance(target, ast.Name) and target.id == "gui_recording":
                found_owner_side += 1
                continue
            if isinstance(target, ast.Name) and target.id == "app_gui":
                offenders.append(f"{test_file.name}: line {node.lineno} patches app_gui.{_MOVED_WORKER}")
    assert offenders == [], (
        "the moved worker must be patched on the_oracle.gui_recording, never "
        "app_gui:\n  " + "\n  ".join(offenders)
    )
    # Classifier vacuity, both directions:
    owner_side = ast.parse(f"monkeypatch.setattr(gui_recording, '{_MOVED_WORKER}', fake)\n")
    app_gui_side = ast.parse(f"monkeypatch.setattr(app_gui, '{_MOVED_WORKER}', fake)\n")

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

    assert classify(owner_side) == "gui_recording"
    assert classify(app_gui_side) == "app_gui"


def test_recording_dialog_constructs_only_from_app_gui_globals() -> None:
    """MainWindow constructs RecordingStudioDialog from a bare name.

    The construction resolving from app_gui's globals is what keeps the
    app_gui-level class patches live; if it ever became a qualified lookup
    (``gui_recording.RecordingStudioDialog``), those patches would go silent
    while the suite stayed green.
    """
    app_gui = _import("the_oracle.app_gui")
    tree = ast.parse(Path(app_gui.__file__).read_text(encoding="utf-8"))
    constructions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RecordingStudioDialog"
    ]
    assert constructions, (
        "no RecordingStudioDialog construction found in app_gui — the "
        "construction pin went blind"
    )
    for node in constructions:
        assert isinstance(node.func, ast.Name), (
            f"line {node.lineno}: RecordingStudioDialog must be constructed "
            "from the bare app_gui-global name"
        )


def test_gui_recording_imports_nothing_from_app_gui() -> None:
    """No import cycle and no seam bypass: gui_recording must never reach
    back into app_gui — in ANY spelling.

    The studio's MainWindow collaborators (the assign callback, the settings
    payload) cross the seam as values; someone "fixing" a missing name with
    ``from the_oracle.app_gui import X`` would plant a cycle under the next
    slice's feet.
    """
    from tests.test_gui_import_direction import _module_root, _scan_imports

    _forbids_app_gui = lambda module: _module_root(module) == "app_gui"
    owner = _import("the_oracle.gui_recording")
    source = Path(owner.__file__).read_text(encoding="utf-8")
    scan = _scan_imports(source, forbidden=_forbids_app_gui)
    assert scan.violations == () and scan.unresolved == (), (
        "gui_recording must not import app_gui in any form:\n  "
        + "\n  ".join(scan.violations + scan.unresolved)
    )
    # Vacuity: each forbidden spelling must still trip this exact gate.
    for form in (
        "import the_oracle.app_gui",
        "import app_gui",
        "from the_oracle.app_gui import RecordingStudioDialog",
        "from the_oracle import app_gui",
        "from . import app_gui",
        "from .app_gui import RecordStudioWorker",
        "importlib.import_module('the_oracle.app_gui')",
    ):
        assert _scan_imports(form, forbidden=_forbids_app_gui).violations, form
    # ...and the scan must see gui_recording's real imports (which are legal):
    assert {
        "the_oracle.gui_utils",
        "the_oracle.audio.recorder",
    } <= set(scan.modules), scan.modules


def test_recording_bodies_never_resolve_app_gui_patched_policy_names() -> None:
    """The harm-class pin: the moved bodies must not read an app_gui-patched
    name as a bare global — except the sanctioned media pair.

    A bare global resolves from gui_recording's own namespace, so an
    app_gui-level patch never reaches it. The media classes are the designed
    exception: they belong to gui_recording's namespace and the dialog tests
    patch them there (see the test-side pin below).
    """
    owner = _import("the_oracle.gui_recording")
    source = Path(owner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    offenders: list[str] = []
    # EVERY top-level scope in gui_recording — the moved cluster and
    # anything added beside it later (the gui_cast harm-class widening:
    # a new helper beside the cluster gets the same rule, not a free pass).
    scopes = [
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    defined = {node.name for node in scopes}
    for name in (_MOVED_WORKER, "RecordingStudioDialog"):
        assert name in defined, (
            f"{name} must be findable in gui_recording — the scan went blind"
        )
    for scope in scopes:
        for node in ast.walk(scope):
            if (
                isinstance(node, ast.Name)
                and node.id in _APP_GUI_PATCHED_POLICY_NAMES
                and node.id not in _SANCTIONED_BARE_NAMES
            ):
                offenders.append(f"{scope.name}: {node.id} (line {node.lineno})")
    assert offenders == [], (
        "recording bodies resolve app_gui-patched names from their own "
        "globals; inject the collaborator as a value or patch the owner:\n  "
        + "\n  ".join(offenders)
    )
    # Vacuity: the scan must flag a synthetic offender — a blind walk passes
    # green over anything.
    probe = ast.parse(
        "class P:\n    def run(self):\n        return OraclePipeline()\n"
    ).body[0]
    flagged = [
        node.id
        for node in ast.walk(probe)
        if isinstance(node, ast.Name)
        and node.id in _APP_GUI_PATCHED_POLICY_NAMES
        and node.id not in _SANCTIONED_BARE_NAMES
    ]
    assert flagged == ["OraclePipeline"]


def test_app_gui_level_media_patches_live_only_in_the_window_fixture() -> None:
    """The studio test file's app_gui-level media patches must stay inside
    the MainWindow fixture.

    The window's own preview player still resolves the media classes from
    app_gui globals, so the fixture's patches are legitimate. A dialog test
    that patches app_gui instead of gui_recording looks green while silently
    faking a module the moved dialog never reads — the exact drift this pin
    refuses.
    """
    source = (REPO_ROOT / "tests" / "test_recording_studio.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    fixture: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_build_main_window":
            fixture = node
            break
    assert fixture is not None, (
        "the MainWindow fixture (_build_main_window) vanished from "
        "test_recording_studio.py — the pin went blind"
    )
    fixture_lines = range(
        fixture.lineno,
        getattr(fixture, "end_lineno", fixture.lineno + 1) + 1,
    )
    offenders: list[str] = []
    app_gui_media_patches = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "setattr" or not node.args:
            continue
        target = node.args[0]
        name_arg = node.args[1] if len(node.args) >= 2 else None
        if not (
            isinstance(target, ast.Name)
            and target.id in {"app_gui", "gui_recording"}
            and isinstance(name_arg, ast.Constant)
            and name_arg.value in {"QMediaPlayer", "QAudioOutput"}
        ):
            continue
        if target.id == "gui_recording":
            continue  # the designed dialog-test spelling
        app_gui_media_patches += 1
        if node.lineno not in fixture_lines:
            offenders.append(f"line {node.lineno}: app_gui-level media patch outside the window fixture")
    assert offenders == [], (
        "dialog-level media fakes must patch the_oracle.gui_recording, not "
        "app_gui:\n  " + "\n  ".join(offenders)
    )
    assert app_gui_media_patches >= 1, (
        "no app_gui-level media patch found — the window fixture's fakes "
        "went missing and the pin went blind"
    )
