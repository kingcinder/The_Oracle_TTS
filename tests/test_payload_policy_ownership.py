"""Ownership tests for the WidgetSnapshot / GUI-settings payload policy.

The settings-payload slice (2026-09-20) established the policy: the window's
single ``_widget_snapshot`` method is the only place payload-relevant widgets
are read; ``gui_settings`` owns payload building/normalization; MainWindow's
payload methods are thin delegates. This test pins that shape so later
extractions cannot silently re-scatter widget reads or hand-roll the schema.

The rule data (payload widgets, read accessors, sanctioned readers, policy
reference set, policy owners, schema keys and exempt modules) used to be
hardcoded in this file. It now lives in the ``payload_policy`` section of
``scripts/patch_surface_manifest.json`` — the same validated record the
patch-surface net reads — and is loaded through
``tests.test_app_gui_patch_surface.load_payload_policy``, which fails loudly
on a malformed section rather than letting the net shrink to match nothing.
One record per rule set; none of it spelled out here.

Four pins, all AST-based (no import of app_gui at collection cost beyond
the policy owner):
0. The record is live: every payload widget the manifest lists is assigned
   somewhere in ``MainWindow.__init__``'s construction tree (an over-typed
   manifest would blind the reader net), and the field floors hold (no
   vacuous record).
1. ``MainWindow`` defines exactly ONE ``_widget_snapshot`` (the audit found a
   shadowed duplicate with an unreachable tail — last-def-wins kept the code
   working while the dead copy invited drift).
2. ``_widget_snapshot`` is the only MainWindow method calling the widget
   READ accessors the payload schema is built from (text/currentText/value/
   isChecked/currentData on the payload widgets) outside the explicitly
   sanctioned set (render-settings builder, apply-payload, project load,
   outdir echo). ``_render_settings`` is sanctioned: it is the render-path
   builder, deliberately separate from the persisted payload.
3. The payload policy functions are referenced only from ``gui_settings``
   (definition) and ``app_gui`` (delegates). No other module builds or
   normalizes the schema directly.
4. No dict literal outside ``gui_settings`` carries a majority of the payload
   schema keys — a hand-rolled schema is the exact drift the single-owner
   policy exists to prevent.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from tests.test_app_gui_patch_surface import load_payload_policy

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_GUI = REPO_ROOT / "src" / "the_oracle" / "app_gui.py"

#: The record itself — one validated read at collection, shared by both nets.
_PAYLOAD_POLICY: dict[str, Any] = load_payload_policy()

#: Widget attributes whose READ accessors feed the payload schema.
PAYLOAD_WIDGETS: frozenset[str] = _PAYLOAD_POLICY["payload_widgets"]

#: Read accessors that constitute "reading a widget" for policy purposes.
READ_ACCESSORS: frozenset[str] = _PAYLOAD_POLICY["read_accessors"]

#: Methods allowed to read payload widgets. Widget-construction and
#: apply/restore methods write or wire; they never make payload decisions.
#: The record annotates each entry's role; the roles are documentation for
#: the record's readers, not enforcement data (the net needs the names only).
SANCTIONED_READERS: frozenset[str] = _PAYLOAD_POLICY["sanctioned_readers"]

#: Where the payload policy functions may be referenced.
POLICY: frozenset[str] = _PAYLOAD_POLICY["policy_references"]

#: Policy owner modules, as ``module -> role`` ("definition" / "delegates").
POLICY_OWNERS: dict[str, str] = _PAYLOAD_POLICY["policy_owners"]

#: Schema keys whose hand-rolled copies outside gui_settings the net forbids.
SCHEMA_KEYS: frozenset[str] = _PAYLOAD_POLICY["schema_keys"]

#: Modules exempt from the hand-rolled-schema scan.
SCHEMA_EXEMPT_MODULES: frozenset[str] = _PAYLOAD_POLICY["schema_exempt_modules"]

#: Vacuity floors: smallest self-defending record sizes. A legit edit can
#: shrink the sets a little; only blindness shrinks them past these.
_MIN_WIDGETS = 9  # one per payload-relevant control in the window
_MIN_SANCTIONED = 25  # the single reader + construction/apply/refresh cluster


def _mainwindow_methods(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow":
            return {
                item.name: item
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            }
    raise AssertionError("MainWindow class not found in app_gui.py")


def _annotate_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]


def test_payload_policy_record_is_live_in_the_window() -> None:
    """Vacuity guards for the manifest-fed rule data.

    A record that loads but matches nothing is the quiet failure this file
    exists to prevent, so the loaded sets must clear explicit floors, and
    every payload widget the record names must actually be assigned in
    MainWindow (an over-typed widget name would blind the reader net to a
    real offender). A genuine removal edits the record consciously.
    """
    assert len(PAYLOAD_WIDGETS) >= _MIN_WIDGETS, (
        f"the record lists only {len(PAYLOAD_WIDGETS)} payload widgets; "
        f"the window has at least {_MIN_WIDGETS}. The record went blind — "
        "fix the record, not this assertion."
    )
    assert len(SANCTIONED_READERS) >= _MIN_SANCTIONED, (
        f"the record sanctions only {len(SANCTIONED_READERS)} readers; the "
        f"known floor is {_MIN_SANCTIONED}. The record went blind — fix the "
        "record, not this assertion."
    )
    assert "the_oracle.gui_settings" in POLICY_OWNERS, (
        "the record lost gui_settings as the payload policy owner"
    )
    assert len(SCHEMA_KEYS) >= 17, (
        f"the record carries only {len(SCHEMA_KEYS)} schema keys; the known "
        "floor is 17. The record went blind — fix the record, not this "
        "assertion."
    )

    source = APP_GUI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    _annotate_parents(tree)
    assigned: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        in_ctor = False
        parent = node._parent  # type: ignore[attr-defined]
        while parent is not None:
            if (
                isinstance(parent, ast.FunctionDef)
                and parent.name in {"__init__", "_build_ui", "_build_project_settings"}
            ):
                in_ctor = True
                break
            parent = getattr(parent, "_parent", None)
            if parent is None:
                break
        if not in_ctor:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assigned.add(target.attr)
    missing = sorted(PAYLOAD_WIDGETS - assigned)
    assert not missing, (
        f"the record's payload widgets not assigned in MainWindow's window-assembly "
        f"methods (__init__/_build_ui/_build_project_settings): {missing} — "
        "the record and the window have drifted; correct the record deliberately "
        "if a control was genuinely removed."
    )


def test_main_window_defines_exactly_one_widget_snapshot() -> None:
    """One snapshot method: the single place payload widgets are read.

    The audit found a shadowed duplicate (last-def-wins kept behavior, but
    the dead copy had an unreachable tail and invited drift). Pin the count.
    """
    tree = ast.parse(APP_GUI.read_text(encoding="utf-8"))
    methods = _mainwindow_methods(tree)
    snapshot_defs = [
        item
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "_widget_snapshot"
    ]
    assert len(snapshot_defs) == 1, (
        f"MainWindow defines _widget_snapshot {len(snapshot_defs)} times; "
        "a shadowed duplicate re-scatters widget reads and defeats the "
        "single-read-site policy. Keep exactly one."
    )
    assert "_widget_snapshot" in methods


def test_widget_snapshot_is_the_only_payload_widget_reader() -> None:
    """Outside the sanctioned set, no MainWindow method reads payload widgets.

    The sanctioned set is deliberately explicit: runtime render/preview paths
    (which build RenderSettings, not the persisted payload), restore paths
    (which read to write), and widget construction/refresh helpers. Anything
    NEW that reads these widgets must be added to the record consciously —
    that's the review gate this test provides.
    """
    source = APP_GUI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    _annotate_parents(tree)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "MainWindow"):
            continue
        for method in node.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if method.name in SANCTIONED_READERS:
                continue
            for call in ast.walk(method):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                if not (isinstance(func, ast.Attribute) and func.attr in READ_ACCESSORS):
                    continue
                receiver = func.value
                if (
                    isinstance(receiver, ast.Attribute)
                    and isinstance(receiver.value, ast.Name)
                    and receiver.value.id == "self"
                    and receiver.attr in PAYLOAD_WIDGETS
                ):
                    offenders.append(f"{method.name} (line {call.lineno}: {receiver.attr}.{func.attr}())")
    assert offenders == [], (
        "payload-widget reads outside the sanctioned single-reader policy:\n  "
        + "\n  ".join(offenders)
        + "\nWidget reads belong in _widget_snapshot (persisted payload) or "
        "_render_settings (runtime). If a new method legitimately reads a "
        "payload widget, add it to the record's sanctioned_readers with a reason."
    )


def test_payload_policy_referenced_only_from_owners() -> None:
    """No module outside the record's policy owners touches the payload
    policy functions — schema building has one address."""
    offenders: list[str] = []
    src_root = REPO_ROOT / "src"
    for path in sorted(src_root.rglob("*.py")):
        module_key = path.relative_to(src_root).with_suffix("").as_posix().replace("/", ".")
        if module_key in POLICY_OWNERS:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name) and node.id in POLICY:
                name = node.id
            elif isinstance(node, ast.Attribute) and node.attr in POLICY:
                name = node.attr
            if name:
                offenders.append(f"{rel}:{node.lineno} {name}")
    assert offenders == [], (
        "payload policy referenced outside its owners:\n  " + "\n  ".join(offenders)
    )


def test_no_hand_rolled_payload_schema_literals() -> None:
    """A dict literal carrying a majority of the payload-schema keys outside
    the exempt modules is a hand-rolled schema — the exact drift the
    single-owner policy exists to prevent (it diverges silently when the
    owner evolves).

    Threshold: more than half of the schema keys in one literal. Ordinary
    dicts sharing a key or two with the schema (kwargs passthroughs, CLI
    flags) are untouched; whole-schema copies are flagged. The record names
    the exempt modules: the window constructs the schema's inputs, it does
    not define the schema.
    """
    majority = len(SCHEMA_KEYS) // 2 + 1
    offenders: list[str] = []
    src_root = REPO_ROOT / "src"
    for path in sorted(src_root.rglob("*.py")):
        module_key = path.relative_to(src_root).with_suffix("").as_posix().replace("/", ".")
        if module_key in SCHEMA_EXEMPT_MODULES:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            overlap = keys & SCHEMA_KEYS
            if len(overlap) >= majority:
                offenders.append(
                    f"{rel}:{node.lineno} carries {len(overlap)} payload-schema keys "
                    f"{sorted(overlap)[:8]}... — build the payload via gui_settings, "
                    "not by copying the schema."
                )
    assert offenders == [], (
        "hand-rolled payload-schema literals found:\n  " + "\n  ".join(offenders)
    )
