"""Ownership tests for the WidgetSnapshot / GUI-settings payload policy.

The settings-payload slice (2026-09-20) established the policy: the window's
single ``_widget_snapshot`` method is the only place payload-relevant widgets
are read; ``gui_settings`` owns payload building/normalization; MainWindow's
payload methods are thin delegates. This test pins that shape so later
extractions cannot silently re-scatter widget reads or hand-roll the schema.

Three pins, all AST-based (no import of app_gui at collection cost beyond
the policy owner):
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
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_GUI = REPO_ROOT / "src" / "the_oracle" / "app_gui.py"

#: Widget attributes whose READ accessors feed the payload schema.
PAYLOAD_WIDGETS: frozenset[str] = frozenset(
    {
        "variant_combo",
        "correction_mode_combo",
        "loudness_combo",
        "crossfade_spin",
        "inference_backend_combo",
        "outdir_path",
        "output_name",
        "export_srt_check",
        "monologue_check",
    }
)

#: Read accessors that constitute "reading a widget" for policy purposes.
READ_ACCESSORS: frozenset[str] = frozenset(
    {"text", "currentText", "value", "isChecked", "currentData"}
)

#: Methods allowed to read payload widgets. Widget-construction and
#: apply/restore methods write or wire; they never make payload decisions.
SANCTIONED_READERS: frozenset[str] = frozenset(
    {
        "_widget_snapshot",  # THE reader — the single place by design
        "_render_settings",  # render-path builder (runtime, not persisted)
        "_apply_gui_settings_payload",  # restore path (reads to decide writes)
        "_load_project_into_ui",  # manifest restore (same shape as apply)
        "_handle_outdir_changed",  # echoes output_name into the field
        "prepare_project",  # runtime render entry (uses _render_settings)
        "render_project",  # runtime: resolve_output_filename before render
        "preview_utterance",  # runtime: backend + variant for this preview
        "_sync_plan_from_table",  # runtime plan refresh (metadata echo)
        "new_project",  # preserves the typed output name across clears
        "_speaker_settings",  # variant/crossfade feed SpeakerGroup decode
        "_build_project_settings",  # widget construction
        "_build_ui",  # widget construction
        "_build_menu",  # menu wiring
        "_register_ctrl_help_descriptions",  # help text binds widget states
        "_set_correction_mode",  # combo writer helper
        "_apply_inference_wizard_selection",  # combo writer helper
        "_apply_inference_wizard_preferences",  # field writer helper
        "_apply_remembered_backend",  # combo restore writer
        "_pick_outdir",  # dialog write-back
        "_persist_remembered_settings",  # remembers backend choice
        "_refresh_vulkan_preflight_button",  # button state from backend
        "_handle_vulkan_model_downloaded",  # post-download re-enable
        "_refresh_audio_cpp_knob_options",  # knob enable/disable
        "_refresh_cast_bar",  # monologue checkbox state for the cast bar
        "_refresh_inference_backend_options",  # variant-dependent options
        "_refresh_language_options",  # variant-dependent languages
        "apply_cast",  # monologue default from cast size
    }
)

#: Where the payload policy functions may be referenced.
POLICY: frozenset[str] = frozenset(
    {
        "current_gui_settings_payload",
        "default_gui_settings_payload",
        "cast_request_from_payload",
        "speaker_config_from_payload",
        "WidgetSnapshot",
        "PayloadDefaults",
    }
)
POLICY_OWNERS: dict[str, str] = {
    "the_oracle.gui_settings": "definition",
    "the_oracle.app_gui": "delegates",
}


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
    NEW that reads these widgets must be added here consciously — that's the
    review gate this test provides.
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
        "payload widget, add it to SANCTIONED_READERS with a reason."
    )


def test_payload_policy_referenced_only_from_owners() -> None:
    """No module outside gui_settings (owner) and app_gui (delegates) touches
    the payload policy functions — schema building has one address."""
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
    gui_settings is a hand-rolled schema — the exact drift the single-owner
    policy exists to prevent (it diverges silently when the owner evolves).

    Threshold: more than half of the schema keys in one literal. Ordinary
    dicts sharing a key or two with the schema (kwargs passthroughs, CLI
    flags) are untouched; whole-schema copies are flagged. app_gui is
    exempt: the window constructs the schema's inputs, it does not define
    the schema.
    """
    schema_keys = {
        "model_variant", "correction_mode", "loudness_preset", "crossfade_ms",
        "inference_backend", "device_mode", "cuda_device", "output_dir",
        "output_filename", "export_srt", "monologue", "delete_confirm_enabled",
        "output_filename_warning", "audio_cpp_device", "audio_cpp_threads",
        "audio_cpp_timeout", "audio_cpp_max_batch",
    }
    majority = len(schema_keys) // 2 + 1
    exempt_modules = {"the_oracle.gui_settings", "the_oracle.app_gui"}
    offenders: list[str] = []
    src_root = REPO_ROOT / "src"
    for path in sorted(src_root.rglob("*.py")):
        module_key = path.relative_to(src_root).with_suffix("").as_posix().replace("/", ".")
        if module_key in exempt_modules:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            overlap = keys & schema_keys
            if len(overlap) >= majority:
                offenders.append(
                    f"{rel}:{node.lineno} carries {len(overlap)} payload-schema keys "
                    f"{sorted(overlap)[:8]}... — build the payload via gui_settings, "
                    "not by copying the schema."
                )
    assert offenders == [], (
        "hand-rolled payload-schema literals found:\n  " + "\n  ".join(offenders)
    )
