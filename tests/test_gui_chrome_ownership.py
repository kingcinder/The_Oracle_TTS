"""Ownership tests for the gui_chrome slice — the Live sidebar chrome.

``LivePanel`` and the Live column's section chrome moved from ``app_gui`` to
``the_oracle.gui_chrome`` (2026-09-21 extraction slice). The safety nets each
slice relies on (``test_app_gui_patch_surface``, ``test_payload_policy_ownership``)
ran green as this slice's pre-flight gates; these tests pin the slice itself.

Four contracts keep the move honest:

1. ONE OWNER — ``gui_chrome`` defines the Live chrome and no other module
   defines a competing ``LivePanel``.
2. RE-EXPORT — ``app_gui.LivePanel`` IS ``gui_chrome.LivePanel``, so
   MainWindow's construction, the two progress handlers that drive it, and the
   existing ``from the_oracle.app_gui import LivePanel`` test import keep
   resolving exactly as before.
3. ASSEMBLY — MainWindow's Live column comes from ``gui_chrome``'s
   ``build_live_section`` (the collapsible/resizable section chrome), while the
   splitter and ``_register_section`` persistence wiring stay in ``app_gui``.
   That is the split-readership the patch-surface net registers as
   ``PARTIAL_OWNED``.
4. MIRROR — the sidebar is the *persistent* half of the progress mirror (the
   render's ``progress_dialog`` AND the preview's ``preview_dialog``), so
   wherever MainWindow dismisses one it must reset the sidebar with it; a
   dismissal site that forgets ``set_idle()`` leaves the finished or failed
   operation's last frame on screen indefinitely. Dismissals are watched in
   every spelling — dropping the reference (``self.X = None``) or ending the
   dialog (``close()`` and friends) — so a teardown that skips the
   None-assign shape (a ``closeEvent`` path, say) cannot slide past the rule;
   unclassifiable spellings fail the scan loudly.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
APP_GUI = SRC / "the_oracle" / "app_gui.py"
GUI_CHROME = SRC / "the_oracle" / "gui_chrome.py"

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def _calls(node: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _callee_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_gui_chrome_is_the_single_owner_of_the_live_panel() -> None:
    """gui_chrome defines the Live chrome; no other module competes."""
    tree = ast.parse(GUI_CHROME.read_text(encoding="utf-8"))
    live_panel = _class(tree, "LivePanel")
    assert live_panel is not None, "gui_chrome must define LivePanel"

    builders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_live_section"
    ]
    assert len(builders) == 1, "gui_chrome must define exactly one build_live_section"

    # No second definition anywhere in the package (a leftover copy in app_gui
    # would mean two live definitions, only one of which the window uses).
    others = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if path != GUI_CHROME
        and _class(ast.parse(path.read_text(encoding="utf-8")), "LivePanel") is not None
    ]
    assert others == [], f"LivePanel defined outside gui_chrome: {others}"


def test_app_gui_no_longer_defines_the_live_panel() -> None:
    """The moved class is gone from app_gui — it must be re-exported, not copied."""
    tree = ast.parse(APP_GUI.read_text(encoding="utf-8"))
    assert _class(tree, "LivePanel") is None, (
        "app_gui still defines LivePanel; the slice must leave exactly one "
        "definition (in gui_chrome) and re-export it."
    )


def test_app_gui_reexports_the_gui_chrome_names() -> None:
    """The re-export contract the patch surface and tests depend on.

    ``tests/test_app_gui_progress.py`` imports ``LivePanel`` from app_gui, and
    MainWindow constructs it from app_gui globals — both require the identical
    object, not a lookalike.
    """
    import the_oracle.app_gui as app_gui
    import the_oracle.gui_chrome as gui_chrome

    assert app_gui.LivePanel is gui_chrome.LivePanel
    assert app_gui.build_live_section is gui_chrome.build_live_section


def test_main_window_builds_the_live_column_from_gui_chrome() -> None:
    """Assembly: the Live section comes from gui_chrome; the splitter wiring
    and section registration stay in app_gui (where the layout state lives)."""
    tree = ast.parse(APP_GUI.read_text(encoding="utf-8"))
    main_window = _class(tree, "MainWindow")
    assert main_window is not None
    build_ui = _method(main_window, "_build_ui")
    assert build_ui is not None

    builder_calls = [c for c in _calls(build_ui) if _callee_name(c) == "build_live_section"]
    assert len(builder_calls) == 1, "MainWindow._build_ui must call build_live_section once"
    # ...and hand it the panel it owns.
    passed = builder_calls[0].args
    assert len(passed) == 1
    assert isinstance(passed[0], ast.Attribute) and passed[0].attr == "live_panel", (
        "build_live_section must receive MainWindow's self.live_panel"
    )

    # The Live section is no longer constructed in app_gui from app_gui's
    # QHSectionGroup — that is exactly the PARTIAL_OWNED split.
    live_sections_here = [
        c
        for c in _calls(build_ui)
        if _callee_name(c) == "QHSectionGroup"
        and c.args
        and isinstance(c.args[0], ast.Constant)
        and c.args[0].value == "Live"
    ]
    assert live_sections_here == [], (
        "app_gui still builds the Live QHSectionGroup; it belongs to "
        "gui_chrome.build_live_section now."
    )

    # Wiring app_gui still owns: the section is registered for layout
    # persistence and added to the main splitter.
    registered = [
        c
        for c in _calls(build_ui)
        if _callee_name(c) == "addWidget"
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "addWidget"
        and isinstance(c.func.value, ast.Attribute)
        and c.func.value.attr == "_main_splitter"
    ]
    assert len(registered) == 2, "the main splitter must still receive both arms"
    register_calls = [
        c
        for c in _calls(build_ui)
        if _callee_name(c) == "_register_section"
        and c.args
        and isinstance(c.args[0], ast.Constant)
        and c.args[0].value == "live"
    ]
    assert len(register_calls) == 1, "the 'live' section must stay registered for persistence"


#: The progress dialogs MainWindow dismisses at the end of an operation. The
#: sidebar mirrors both (renders drive ``progress_dialog``, previews drive
#: ``preview_dialog``); a dismissal of either must reset the mirror.
_PROGRESS_DIALOG_ATTRS = frozenset({"progress_dialog", "preview_dialog"})

#: Ways a dialog can die that mean "the operation is over": dropping the
#: reference or ending the dialog through one of its own methods. A dismissal
#: that skips the None-assign shape (a ``closeEvent`` teardown, say) must not
#: slide past the mirror rule.
_DISMISSAL_METHODS = frozenset({"close", "done", "accept", "reject", "deleteLater"})

#: Interactions that are *not* dismissals: dialog liveness only.
_BENIGN_METHODS = frozenset({"show", "update_progress"})


class _DialogScan(NamedTuple):
    """The classified result of one scan pass over a window class."""

    dismissers: list[str]  # methods containing any dismissal shape
    offenders: list[str]  # dismissers lacking a same-method sidebar reset
    dismissed_dialogs: set[str]  # dialog attr names seen being dismissed
    dismissal_shapes: set[str]  # e.g. {"call:close", "assign:None"}
    unclassified: list[str]  # interactions the scan refuses to guess about


def _scan_dialog_interactions(window_cls: ast.ClassDef) -> _DialogScan:
    """Classify every interaction with the two mirrored progress dialogs.

    Dismissal shapes covered: dropping the reference (``self.X = None``)
    and ending the dialog via one of its methods (``close``/``done``/
    ``accept``/``reject``/``deleteLater``) — either way the sidebar must be
    reset in the same method. ``__init__``'s ``= None`` is initialization,
    not dismissal; ``show``/``update_progress`` and construction assignments
    are benign liveness. Anything else lands in ``unclassified`` and fails
    the scan loudly: a new spelling must be classified deliberately (add it
    to ``_DISMISSAL_METHODS`` or ``_BENIGN_METHODS``), never absorbed by
    guesswork — a dismissal hiding in an unclassified spelling is exactly
    the bug this scan exists to catch.
    """
    dismissers: list[str] = []
    offenders: list[str] = []
    dismissed_dialogs: set[str] = set()
    dismissal_shapes: set[str] = set()
    unclassified: list[str] = []

    for method in window_cls.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        dismissed = False
        idles_sidebar = False
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "set_idle"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "live_panel"
            ):
                idles_sidebar = True

            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if not (
                        isinstance(target, ast.Attribute)
                        and target.attr in _PROGRESS_DIALOG_ATTRS
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        continue
                    value = node.value
                    if value is None:
                        continue  # bare annotation declares, never dismisses
                    if isinstance(value, ast.Constant) and value.value is None:
                        if method.name == "__init__":
                            continue  # initialization, not dismissal
                        dismissed = True
                        dismissed_dialogs.add(target.attr)
                        dismissal_shapes.add("assign:None")
                    elif isinstance(value, ast.Call):
                        pass  # (re)construction — benign liveness
                    else:
                        unclassified.append(
                            f"{method.name} (line {node.lineno}): self.{target.attr} = "
                            f"<{type(value).__name__}> — dismissal or not? Classify "
                            "it deliberately, do not let the scan guess."
                        )

            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr in _PROGRESS_DIALOG_ATTRS
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self"
            ):
                callee = node.func.attr
                if callee in _DISMISSAL_METHODS:
                    dismissed = True
                    dismissed_dialogs.add(node.func.value.attr)
                    dismissal_shapes.add(f"call:{callee}")
                elif callee not in _BENIGN_METHODS:
                    unclassified.append(
                        f"{method.name} (line {node.lineno}): "
                        f"self.{node.func.value.attr}.{callee}() — is this a "
                        "dismissal? Add it to _DISMISSAL_METHODS or "
                        "_BENIGN_METHODS deliberately."
                    )
        if dismissed:
            dismissers.append(method.name)
            if not idles_sidebar:
                offenders.append(f"{method.name} (line {method.lineno})")

    return _DialogScan(
        dismissers, offenders, dismissed_dialogs, dismissal_shapes, unclassified
    )


def test_dismissing_the_progress_dialog_also_idles_the_sidebar() -> None:
    """Every progress-dialog dismissal resets the persistent sidebar with it.

    The sidebar mirrors the dialog but is the half that stays on screen, so a
    dismissal site that forgets ``live_panel.set_idle()`` freezes the finished
    or failed operation's last frame there forever — the bug fixed in
    ``_finish_render``/``_fail_render``, and found again in the preview paths
    (``_finish_preview``/``_fail_preview``, the queued-after-setup path, and
    the worker-start failure teardown) once previews were seen mirroring into
    the sidebar too. Pinning the shape keeps the next dismissal site (or the
    next mirror) from re-introducing it.
    """
    tree = ast.parse(APP_GUI.read_text(encoding="utf-8"))
    main_window = _class(tree, "MainWindow")
    assert main_window is not None

    scan = _scan_dialog_interactions(main_window)

    assert scan.dismissers, (
        "no MainWindow method dismisses a progress dialog; the scan went "
        "blind — fix the scanner, not this assertion."
    )
    assert scan.dismissed_dialogs == _PROGRESS_DIALOG_ATTRS, (
        f"the scan saw dismissals of {sorted(scan.dismissed_dialogs)} only; both "
        f"dialog families {sorted(_PROGRESS_DIALOG_ATTRS)} must stay visible. "
        "If one side genuinely stops being dismissed, update this guard "
        "deliberately — do not let the scan narrow silently."
    )
    assert {"call:close", "assign:None"} <= scan.dismissal_shapes, (
        f"the scan saw dismissal shapes {sorted(scan.dismissal_shapes)} only; "
        "both the close-call and the reference-drop shapes must stay visible "
        "(a close-only dismissal is exactly what a closeEvent teardown would "
        "introduce). If a shape genuinely disappears, update this guard "
        "deliberately — do not let the scan narrow silently."
    )
    assert scan.unclassified == [], (
        "these dialog interactions use a spelling the scan has not "
        "classified; decide whether each is a dismissal (add to "
        "_DISMISSAL_METHODS) or benign liveness (add to _BENIGN_METHODS) — "
        "a dismissal hiding in an unclassified spelling is the bug this "
        "scan exists to catch:\n  " + "\n  ".join(scan.unclassified)
    )
    assert scan.offenders == [], (
        "these MainWindow methods dismiss a progress dialog but leave the "
        "persistent Live sidebar showing stale progress:\n  "
        + "\n  ".join(scan.offenders)
        + "\nAdd live_panel.set_idle() alongside the dismissal."
    )


def test_the_dismissal_scanner_covers_every_shape_form_by_form() -> None:
    """Form-by-form proofs the classifier sees the shapes it claims to.

    The live source only exercises close()+clear together, so a close-only
    dismissal (the shape a ``closeEvent`` teardown would introduce) or a
    result-based dismissal could regress the mirror rule with every scan pass
    green. These synthetic forms run in every CI pass: every dismissal shape
    without a reset must be flagged as that shape (not merely flagged), every
    shape with a reset must pass *having been seen as a dismissal*, benign
    interactions must not be flagged, and unknown spellings must fail loudly
    rather than pass unseen.
    """

    def scan(method_body: str) -> _DialogScan:
        indented = "\n".join("        " + line for line in method_body.splitlines())
        tree = ast.parse(f"class _W:\n    def m(self) -> None:\n{indented}\n")
        window_cls = _class(tree, "_W")
        assert window_cls is not None
        return _scan_dialog_interactions(window_cls)

    dismissal_forms = (
        "self.progress_dialog.close()",
        "self.preview_dialog.close()",
        "self.progress_dialog.done(0)",
        "self.preview_dialog.accept()",
        "self.progress_dialog.reject()",
        "self.preview_dialog.deleteLater()",
        "self.progress_dialog = None",
        "self.preview_dialog = None",
    )
    for form in dismissal_forms:
        got = scan(form)
        assert got.dismissers == ["m"], f"{form!r} was not seen as a dismissal"
        assert got.offenders and got.offenders[0].startswith("m ("), (
            f"{form!r} without a same-method reset must be flagged"
        )
        assert got.unclassified == [], f"{form!r} misread: {got.unclassified}"

        got = scan(form + "\nself.live_panel.set_idle()")
        assert got.dismissers == ["m"], f"{form!r} was not seen as a dismissal"
        assert got.offenders == [], f"{form!r} with a same-method reset must pass"
        assert got.unclassified == [], f"{form!r} misread: {got.unclassified}"

    # __init__'s = None is initialization, not dismissal.
    tree = ast.parse(
        "class _W:\n"
        "    def __init__(self) -> None:\n"
        "        self.progress_dialog = None\n"
        "        self.preview_dialog: object | None = None\n"
    )
    init_scan = _scan_dialog_interactions(_class(tree, "_W"))  # type: ignore[arg-type]
    assert init_scan.dismissers == [] and init_scan.offenders == []
    assert init_scan.unclassified == []

    for form in (
        "self.progress_dialog.show()",
        "self.preview_dialog.update_progress(progress)",
        "self.preview_dialog = RenderProgressDialog(self)",
    ):
        got = scan(form)
        assert got.dismissers == [], f"{form!r} is benign liveness, not a dismissal"
        assert got.unclassified == [], f"{form!r} misread: {got.unclassified}"

    for form in (
        "self.progress_dialog.hide()",
        "self.preview_dialog = other",
    ):
        got = scan(form)
        assert got.unclassified, f"{form!r} must fail loudly as unclassified"
        assert got.dismissers == [], f"{form!r} must not be absorbed as a dismissal"


@pytest.mark.slow
def test_live_panel_retry_tally_state_machine_is_pinned() -> None:
    """The tally is a three-state contract on the persistent mirror.

    The retry count may only change at the two deliberate transitions:
    a note-bearing progress event increments it (and set_idle never touches
    it — a self-heal is part of the session's record), and reset_retry_tally
    starts the next session at zero. The unit pins in test_app_gui_progress
    exercise a LivePanel instance; this pins the CLASS: the set_idle body
    must not assign _retry_count, the increment must live in
    update_from_progress, and reset must be its own method — so a refactor
    that "tidies" the persistence away fails the net, named, before the
    behavior silently changes.
    """
    tree = ast.parse(GUI_CHROME.read_text(encoding="utf-8"))
    panel = _class(tree, "LivePanel")
    assert panel is not None, "gui_chrome must define LivePanel"

    methods = {
        node.name: node
        for node in panel.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("update_from_progress", "set_idle", "reset_retry_tally"):
        assert name in methods, f"LivePanel.{name} must exist — the tally net went blind"

    def _count_assignments(method: ast.FunctionDef) -> int:
        return sum(
            1
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
                and t.attr == "_retry_count"
                for t in node.targets
            )
        )

    # set_idle preserves; reset clears. Exactly one assignment each — an
    # extra assignment in either body would be a rule change to review.
    assert _count_assignments(methods["set_idle"]) == 0, (
        "set_idle must not touch the retry tally — the persistent mirror's "
        "point is that a self-heal survives it"
    )
    assert _count_assignments(methods["reset_retry_tally"]) == 1
    # The increment lives in update_from_progress (+= is an AugAssign, so it
    # does not count as a plain assignment there).
    assert _count_assignments(methods["update_from_progress"]) == 0
    increments = [
        node
        for node in ast.walk(methods["update_from_progress"])
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Attribute)
        and isinstance(node.target.value, ast.Name)
        and node.target.value.id == "self"
        and node.target.attr == "_retry_count"
    ]
    assert len(increments) == 1, (
        "update_from_progress must contain exactly one tally increment"
    )


@pytest.mark.slow
def test_build_live_section_wraps_the_panel_in_section_chrome() -> None:
    """The chrome itself: a collapsible, resizable 'Live' section holding the
    panel, with the Live column's narrow floor."""
    from PySide6.QtWidgets import QApplication

    from the_oracle.gui_chrome import LivePanel, build_live_section
    from the_oracle.gui_sections import QHSectionGroup

    QApplication.instance() or QApplication([])

    panel = LivePanel()
    section = build_live_section(panel)

    assert isinstance(section, QHSectionGroup)
    assert section.title() == "Live"
    assert section.has_size_slider(), "the section must be resizable"
    assert not section.is_collapsed()
    assert section.minimumWidth() == 220
    assert panel.parent() is section, "the panel must be reparented into the section"

    wider = build_live_section(LivePanel(), minimum_width=300)
    assert wider.minimumWidth() == 300


@pytest.mark.slow
def test_live_column_updates_through_the_wrapped_panel() -> None:
    """Driving the panel through its section chrome behaves identically: the
    progress mirror still updates and still resets."""
    from PySide6.QtWidgets import QApplication

    from the_oracle.gui_chrome import LivePanel, build_live_section
    from the_oracle.pipeline import RenderProgress

    QApplication.instance() or QApplication([])

    panel = LivePanel()
    # The section owns the panel (addWidget reparents), so the reference must
    # outlive the test body — dropping it would let Qt delete the whole
    # subtree, panel and progress bar included. MainWindow keeps the same
    # ownership via its splitter.
    section = build_live_section(panel)
    assert section is not None

    panel.update_from_progress(
        RenderProgress(
            stage="Synthesizing",
            detail="utterance 3/10",
            current_step=3,
            total_steps=10,
            current_segment=3,
            total_segments=10,
            elapsed_seconds=12.5,
            eta_seconds=8.0,
            fraction=0.3,
            backend="vulkan",
            device_label="AMD RX 5700 XT",
            synth_seconds_total=9.2,
            synth_seconds_latest=1.5,
        )
    )
    assert panel.progress_bar.value() == 30
    assert "Vulkan" in panel.backend_label.text()
    assert "AMD RX 5700 XT" in panel.backend_label.text()
    assert "Synthesizing" in panel.stage_label.text()
    assert "3/10" in panel.segment_label.text()

    panel.set_idle()
    assert panel.progress_bar.value() == 0
    assert panel.backend_label.text() == "Backend: idle"
    assert panel.stage_label.text() == ""
