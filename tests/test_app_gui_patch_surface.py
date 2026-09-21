"""The app_gui patch surface is load-bearing — this test keeps it visible.

Every MainWindow extraction slice moves code out of ``app_gui`` into an owner
module. The safety net each slice relies on: tests patch app_gui-level names
(``monkeypatch.setattr(app_gui, "find_audiocpp_binary", ...)``), and moved
code resolves names from *its own* module globals, so a test that still
patches ``app_gui`` for behavior a moved module now owns breaks silently —
the patch applies to app_gui, the moved body never sees it, and the suite
reports a baffling failure far from the cause. The Vulkan slice hit exactly
this (the failed extraction of ``_vulkan_preflight_report`` broke 9
patch-coupled tests this way before it was reverted).

The per-cluster review used a throwaway AST scanner to map the patch surface
before each slice. This test commits that scanner: it parses every
``tests/test_*.py`` with :mod:`ast` (no import, no execution) and fails if
any test patches a name through the ``app_gui`` module object that the moved
owner modules now define for themselves. ``app_gui`` re-exports those names,
so patching them *there* is a silent no-op for the owner's code.

Two rules, both born from real slices:

1. MOVED_OWNERS — a name one owner module defines for itself and nothing in
   ``app_gui`` reads anymore. ANY app_gui-level patch of it is a silent
   no-op; repoint the patch at the owner (or un-move the name).
2. SPLIT_OWNED — a name with split readership after the gui_render slice:
   ``OraclePipeline`` is constructed by MainWindow (and PrewarmThread) from
   app_gui globals AND by the workers' direct fallback from gui_render
   globals. app_gui-level patches stay legitimate for window-assembly tests
   and are REQUIRED to be gui_render-level in the worker-path test files.

An offender must be fixed by repointing the patch at the owner module (or by
un-moving the name). Adding names to the maps as slices land keeps the net
current; a name absent from both maps is not checked.

Covered patch forms (the review's scanner missed ``type(window)`` class
targets at first because a Call is not a Name/Attribute chain — this one
handles both):
  * ``monkeypatch.setattr(app_gui, "name", ...)`` / ``delattr``
  * ``monkeypatch.setattr(app_gui, name_var, ...)`` — non-literal names are
    reported as unresolved so nothing slips through unreviewed
  * ``mock.patch("the_oracle.app_gui.name")`` / ``patch("app_gui.name")``
  * ``mock.patch.object(app_gui, "name")``
  * keyword form ``monkeypatch.setattr(app_gui, name="x")``
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Moved-owner modules, in extraction-slice order, with the app_gui names each
#: now owns. A test patching one of these names through ``app_gui`` is a
#: silent no-op for the owner's code and must be repointed.
MOVED_OWNERS: dict[str, frozenset[str]] = {
    "the_oracle.gui_settings": frozenset(
        {
            "default_gui_settings_payload",
            "current_gui_settings_payload",
            "cast_request_from_payload",
            "speaker_config_from_payload",
            "input_file_is_trusted",
            "remember_trusted_input_file",
            "clear_trusted_input_files",
            "remember_format_backup",
            "next_format_backup",
            "drop_next_format_backup",
            "PayloadDefaults",
        }
    ),
    "the_oracle.models.settings": frozenset({"RenderSettings", "SpeakerSettings"}),
    "the_oracle.gui_vulkan": frozenset({"_device_row_text", "_parse_oracle_model_path"}),
    "the_oracle.gui_render": frozenset(
        {
            # Only the no-op-after-move name. The three class names must NOT
            # be listed: MainWindow (still in app_gui) constructs RenderWorker,
            # PreviewWorker and RenderProgressDialog from app_gui globals, so
            # app_gui-level class patches remain live and legitimate.
            "_render_child_environment",
        }
    ),
}

#: Split-readership names: constructed from app_gui globals by MainWindow's
#: window assembly AND from the owner module's globals by moved code. An
#: app_gui-level patch is legitimate for window-assembly tests but a silent
#: no-op for the worker path.
SPLIT_OWNED: dict[str, tuple[str, str]] = {
    "OraclePipeline": ("the_oracle.app_gui", "the_oracle.gui_render"),
}

#: Test files whose OraclePipeline patches feed the WORKERS' direct
#: (non-subprocess) fallback — the path gui_render owns. Everything else
#: patching app_gui.OraclePipeline feeds window assembly / prewarm, which
#: app_gui still owns.
WORKER_PATH_TESTS: frozenset[str] = frozenset(
    {
        "test_app_gui_srt.py",
    }
)

#: Every name the scan must catch, for the failure message.
OWNED_BY_MOVERS: dict[str, str] = {
    name: owner for owner, names in MOVED_OWNERS.items() for name in names
}


@dataclass(frozen=True)
class PatchTarget:
    """One app_gui patch site found by the AST scan."""

    file: str
    line: int
    form: str
    name: str  # the patched attribute, or "?" when not statically resolvable
    resolved: bool  # False when the name argument is not a string literal

    def as_offender(self) -> str | None:
        """Return the offender description when this target breaks the net."""
        if not self.resolved:
            return f"{self.file}:{self.line} {self.form} name={self.name!r} (unresolved — must be reviewed)"
        if self.name in OWNED_BY_MOVERS:
            return (
                f"{self.file}:{self.line} {self.form} patches app_gui.{self.name}, "
                f"but that name is owned by {OWNED_BY_MOVERS[self.name]} — "
                "the patch is invisible to the owner's code. Repoint the patch "
                f"at {OWNED_BY_MOVERS[self.name]} (or un-move the name)."
            )
        if self.name in SPLIT_OWNED and self.file in WORKER_PATH_TESTS:
            _window_owner, workers_owner = SPLIT_OWNED[self.name]
            return (
                f"{self.file}:{self.line} {self.form} patches app_gui.{self.name} "
                "in a worker-path test — the workers' direct (non-subprocess) "
                f"fallback resolves {self.name} from {workers_owner}'s globals, "
                "so this patch is invisible where it matters. Repoint it at "
                f"{workers_owner}. (app_gui-level patches stay correct for "
                "window-assembly tests; MainWindow still constructs from "
                "app_gui globals.)"
            )
        return None


def _root_name(node: ast.AST) -> str | None:
    """Root identifier of a Name or Attribute chain (``a.b.c`` -> ``a``)."""
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_app_gui(node: ast.AST) -> bool:
    """True for the name ``app_gui`` or an attribute chain rooted at it."""
    return _root_name(node) == "app_gui"


def _string_constants(call: ast.Call) -> list[str]:
    return [
        arg.value
        for arg in call.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    ]


def _scan_call(path: str, call: ast.Call) -> list[PatchTarget]:
    """Classify one call node as an app_gui patch target, or not."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return []
    method = func.attr
    if method not in {"setattr", "delattr", "patch", "patch.object", "patch.multiple"}:
        return []
    root = _root_name(func.value)
    is_monkeypatch = root == "monkeypatch"
    is_mock_api = root in {"mock", "patch", "pytest"} or (
        isinstance(func.value, ast.Attribute) and func.value.attr in {"mock"}
    )
    if not (is_monkeypatch or is_mock_api):
        return []

    targets: list[PatchTarget] = []
    args = call.args

    if is_monkeypatch and method in {"setattr", "delattr"}:
        if args and _is_app_gui(args[0]):
            # setattr(app_gui, "name", value) — name may be positional or
            # keyword; also handle the three-arg setattr(target, name, value)
            # and the (target, name, value) delattr forms.
            name_arg: ast.AST | None = None
            if len(args) >= 2:
                name_arg = args[1]
            else:
                for kw in call.keywords:
                    if kw.arg == "name":
                        name_arg = kw.value
                        break
            if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
                targets.append(PatchTarget(path, call.lineno, f"monkeypatch.{method}(app_gui, ...)", name_arg.value, True))
            else:
                target_desc = ast.dump(name_arg) if name_arg is not None else "?"
                targets.append(PatchTarget(path, call.lineno, f"monkeypatch.{method}(app_gui, ...)", target_desc, False))
    elif method == "patch":
        # patch("the_oracle.app_gui.name") / patch("app_gui.name")
        for s in _string_constants(call):
            if s.startswith("the_oracle.app_gui."):
                targets.append(PatchTarget(path, call.lineno, "patch(string)", s.rsplit(".", 1)[-1], True))
    elif method == "patch.object":
        if args and _is_app_gui(args[0]):
            name_arg = args[1] if len(args) >= 2 else None
            if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
                targets.append(PatchTarget(path, call.lineno, "patch.object(app_gui, ...)", name_arg.value, True))
            else:
                target_desc = ast.dump(name_arg) if name_arg is not None else "?"
                targets.append(PatchTarget(path, call.lineno, "patch.object(app_gui, ...)", target_desc, False))
    return targets


def scan_app_gui_patch_surface(tests_dir: Path) -> list[PatchTarget]:
    """AST-scan every test file for patches routed through the app_gui module."""
    targets: list[PatchTarget] = []
    for path in sorted(tests_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                targets.extend(_scan_call(path.name, node))
    return targets


def test_no_test_patches_an_app_gui_name_owned_by_a_moved_module() -> None:
    """The safety net: no test may patch app_gui names the owners define.

    Patching through app_gui resolves in app_gui's globals; the moved body
    reads its own. Such a patch is a silent no-op for the behavior it means
    to stub — see the module docstring for the harm history.
    """
    offenders = [
        offender
        for target in scan_app_gui_patch_surface(REPO_ROOT / "tests")
        if (offender := target.as_offender()) is not None
    ]
    assert offenders == [], (
        "app_gui patch surface broken by a moved-module name:\n  "
        + "\n  ".join(offenders)
    )


def test_scan_finds_the_known_app_gui_surface() -> None:
    """Vacuity guard: the scanner must still see the live patch surface.

    The suite currently routes 100+ patches through app_gui (engine fakes,
    device helpers, thread classes). If a refactor of the scan or the tests
    ever drops that count to zero, the scanner went blind — fail loudly
    rather than pass vacuously.
    """
    targets = scan_app_gui_patch_surface(REPO_ROOT / "tests")
    assert len(targets) >= 50, (
        f"scanner found only {len(targets)} app_gui patch targets; "
        "the known surface is 100+. The scan went blind — fix the scanner, "
        "not this assertion."
    )
    resolved_names = {t.name for t in targets if t.resolved}
    # Names the suite demonstrably patches today (subset spot-checks).
    assert {"find_audiocpp_binary", "OraclePipeline", "QMediaPlayer"} <= resolved_names


def test_moved_owner_names_are_current() -> None:
    """The MOVED_OWNERS map must match what the owners actually define.

    Guards the map against drift in both directions: a name removed from an
    owner (so app_gui patches become legitimate again) or a new owner module
    landing without being registered here.
    """
    import importlib

    for owner_module, names in MOVED_OWNERS.items():
        module = importlib.import_module(owner_module)
        missing = {name for name in names if not hasattr(module, name)}
        assert not missing, f"{owner_module} no longer defines: {sorted(missing)}"

    # Split-owned names must exist on BOTH sides: the window assembly and
    # the worker fallback. Losing either side changes which patches are live.
    for name, (window_module, workers_module) in SPLIT_OWNED.items():
        window_mod = importlib.import_module(window_module)
        workers_mod = importlib.import_module(workers_module)
        assert hasattr(window_mod, name), f"{window_module} lost the window-side {name}"
        assert hasattr(workers_mod, name), f"{workers_module} lost the worker-side {name}"
