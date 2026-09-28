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

Three rules, all born from real slices:

1. MOVED_OWNERS — a name one owner module defines for itself and nothing in
   ``app_gui`` reads anymore. ANY app_gui-level patch of it is a silent
   no-op; repoint the patch at the owner (or un-move the name).
2. SPLIT_OWNED — a name with split readership after the gui_render slice:
   ``OraclePipeline`` is constructed by MainWindow (and PrewarmThread) from
   app_gui globals AND by the workers' direct fallback from gui_render
   globals. app_gui-level patches stay legitimate for window-assembly tests
   and are REQUIRED to be gui_render-level in the worker-path test files.
3. PARTIAL_OWNED — a name with construction/read sites in two modules after
   the gui_chrome slice: an app_gui-level patch is not a no-op but is
   PARTIAL, covering app_gui's sites and silently missing the owner's.

An offender must be fixed by repointing the patch at the owner module (or by
un-moving the name). Adding names to the maps as slices land keeps the net
current; a name absent from both maps is not checked.

MOVED_OWNERS is not spelled out in this file: the ownership record lives in
``scripts/patch_surface_manifest.json``, read here (and validated — a manifest
that no longer parses fails loudly rather than silently matching nothing) and
readable by future extraction-slice tooling. SPLIT_OWNED, PARTIAL_OWNED and
WORKER_PATH_TESTS stay here, because they encode where patches are *allowed*
(the review policy for split-readership names), not who owns what.

The same record also carries the payload-policy net's rule data (the
``payload_policy`` section, read by ``tests/test_payload_policy_ownership.py``
through :func:`load_payload_policy` with the same validate-loudly rule) — one
validated record per rule set, none of it hardcoded in a test file.

Covered patch forms (the review's scanner missed ``type(window)`` class
targets at first because a Call is not a Name/Attribute chain — this one
handles both):
  * ``monkeypatch.setattr(app_gui, "name", ...)`` / ``delattr``
  * ``monkeypatch.setattr(app_gui, name_var, ...)`` — non-literal names are
    reported as unresolved so nothing slips through unreviewed
  * ``mock.patch("the_oracle.app_gui.name")`` / ``patch("app_gui.name")``
  * ``monkeypatch.setattr("the_oracle.app_gui.name", ...)`` — the string
    form resolves ``the_oracle.app_gui`` by import; identical hazard, and
    invisible to any scan that only handles module-object first arguments
  * ``mock.patch.object(app_gui, "name")``
  * ``mock.patch.multiple("the_oracle.app_gui", name=value, ...)`` — each
    keyword is one patch site (the string form again); the module-object
    form would need the target as a *value*, which AST cannot bind, so
    only the string form is scanned
  * keyword form ``monkeypatch.setattr(app_gui, name="x")``
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The ownership record itself lives in ``scripts/`` rather than here, so the
#: same file can be read by extraction-slice tooling that has no interest in
#: this test. This module is its enforcer, not its only reader.
MANIFEST_PATH = REPO_ROOT / "scripts" / "patch_surface_manifest.json"


def load_moved_owners(manifest_path: Path = MANIFEST_PATH) -> dict[str, frozenset[str]]:
    """Read ``moved_owners`` from the manifest, validating its shape.

    Moved-owner modules, in extraction-slice order, with the app_gui names each
    now owns. A test patching one of these names through ``app_gui`` is a
    silent no-op for the owner's code and must be repointed.

    Each entry is ``{"names": [...], "note": "..."}``; the note is rationale
    for humans and tooling, so it is optional. Anything else fails loudly here:
    a typo that silently yielded zero names would leave the whole net passing
    vacuously, which is the one failure mode this file exists to prevent.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("moved_owners")
    if not isinstance(entries, dict) or not entries:
        raise AssertionError(f"{manifest_path.name}: 'moved_owners' must be a non-empty object")
    moved: dict[str, frozenset[str]] = {}
    for owner_module, entry in entries.items():
        if not isinstance(entry, dict):
            raise AssertionError(
                f"{manifest_path.name}: {owner_module!r} must map to an object with 'names'"
            )
        names = entry.get("names")
        if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
            raise AssertionError(
                f"{manifest_path.name}: {owner_module!r} needs a non-empty 'names' list of strings"
            )
        moved[owner_module] = frozenset(names)
    return moved


MOVED_OWNERS: dict[str, frozenset[str]] = load_moved_owners()

#: Top-level manifest sections. Every section of the record must be routed
#: through a validated loader; a section landing in the JSON without one is a
#: bug to fix, not a free pass to ship unvalidated policy.
_MANIFEST_SECTIONS = frozenset({"moved_owners", "payload_policy"})


def load_payload_policy(manifest_path: Path = MANIFEST_PATH) -> dict[str, object]:
    """Read the ``payload_policy`` section from the manifest, validating it.

    The payload-policy net's rule data used to be hardcoded in
    ``tests/test_payload_policy_ownership.py``; it lives in the record now so
    both safety nets read the same validated manifest. Every field must be
    present with the right shape — a typo that quietly yielded an empty set
    would leave the net that reads it passing vacuously, the exact failure
    mode this record exists to prevent. Anything malformed fails loudly here.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"{manifest_path.name}: top level must be an object")
    unknown = set(data) - _MANIFEST_SECTIONS - {"description"}
    if unknown:
        raise AssertionError(
            f"{manifest_path.name}: unrecognized top-level section(s) {sorted(unknown)} — "
            "every section of this record must be routed through a validated loader; "
            "add one rather than letting unvalidated policy ship."
        )
    policy = data.get("payload_policy")
    if not isinstance(policy, dict):
        raise AssertionError(f"{manifest_path.name}: 'payload_policy' must be an object")

    def string_list(field: str, *, allow_empty: bool = False) -> frozenset[str]:
        values = policy.get(field)
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise AssertionError(
                f"{manifest_path.name}: payload_policy.{field} must be a list of strings"
            )
        if not allow_empty and not values:
            raise AssertionError(
                f"{manifest_path.name}: payload_policy.{field} must not be empty — "
                "an empty record makes the net that reads it pass vacuously"
            )
        return frozenset(values)

    owners = policy.get("policy_owners")

    def owners_dict() -> dict[str, str]:
        if (
            not isinstance(owners, dict)
            or not owners
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in owners.items())
        ):
            raise AssertionError(
                f"{manifest_path.name}: payload_policy.policy_owners must be a "
                "non-empty object of string -> string"
            )
        return dict(owners)

    # Field checks run in this order, so a malformed test fixture reports the
    # first broken field, not whichever one a refactor happened to hoist.
    return {
        "payload_widgets": string_list("payload_widgets"),
        "read_accessors": string_list("read_accessors"),
        "sanctioned_readers": string_list("sanctioned_readers"),
        "policy_references": string_list("policy_references"),
        "policy_owners": owners_dict(),
        "schema_keys": string_list("schema_keys"),
        "schema_exempt_modules": string_list("schema_exempt_modules"),
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

#: PARTIAL-readership names: construction/read sites live in BOTH modules, so
#: an app_gui-level patch is neither live-everywhere nor a no-op — it covers
#: app_gui's sites and silently misses the owner module's. (The gui_chrome
#: slice: ``build_live_section`` builds the Live column's ``QHSectionGroup``
#: from gui_chrome's globals, while the shared-settings and status sections
#: are still built from app_gui's.) Unlike SPLIT_OWNED this is not scoped to
#: worker-path files: every app_gui-level patch of the name is partial.
PARTIAL_OWNED: dict[str, tuple[str, str]] = {
    "QHSectionGroup": ("the_oracle.app_gui", "the_oracle.gui_chrome"),
}

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
        if self.name in PARTIAL_OWNED:
            _window_owner, owner_module = PARTIAL_OWNED[self.name]
            return (
                f"{self.file}:{self.line} {self.form} patches app_gui.{self.name} — "
                f"PARTIAL coverage: {self.name} has construction/read sites in "
                f"both app_gui's globals and {owner_module}'s, so this patch "
                f"covers only app_gui's sites and silently misses "
                f"{owner_module}'s. Patch the site you mean, or patch both "
                "modules."
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


def _patch_string_target(value: str) -> tuple[str, str] | None:
    """Split a dotted patch string into (module, name) if it targets app_gui.

    Both the fully-qualified ``the_oracle.app_gui.X`` and the abbreviated
    ``app_gui.X`` (mock's string form also resolves the last two components
    against the calling namespace) route through app_gui's globals — the
    same silent-no-op hazard as ``setattr(app_gui, "X", ...)``. Everything
    else returns None.
    """
    if value.startswith("the_oracle.app_gui."):
        module, _, name = value.rpartition(".")
        return module, name
    if value.startswith("app_gui."):
        module, _, name = value.rpartition(".")
        return module, name
    return None


def _scan_call(path: str, call: ast.Call) -> list[PatchTarget]:
    """Classify one call node as an app_gui patch target, or not."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return []
    root = _root_name(func.value)
    method = func.attr
    # ast exposes only the LAST attribute component (mock.patch.object ->
    # "object"), so the mock API's two-component methods must be rebuilt —
    # otherwise the patch.object/patch.multiple handling below is dead code
    # and those forms bypass the net entirely.
    if method in {"object", "multiple"} and (
        root in {"patch", "mock"}
        or (isinstance(func.value, ast.Attribute) and func.value.attr == "patch")
    ):
        method = f"patch.{method}"
    if method not in {"setattr", "delattr", "patch", "patch.object", "patch.multiple"}:
        return []
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
        elif args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
            # setattr("the_oracle.app_gui.name", value): the string form
            # imports the module and sets the attribute on it — identical
            # hazard to the module-object form, and invisible to any scan
            # that only inspects first-argument expressions.
            target = _patch_string_target(args[0].value)
            if target is not None:
                module, name = target
                targets.append(PatchTarget(path, call.lineno, f"monkeypatch.{method}({module!r}, ...)", name, True))
    elif method == "patch":
        # patch("the_oracle.app_gui.name") / patch("app_gui.name") — mock's
        # string form imports the module, so both spellings hit app_gui's
        # globals (the abbreviated form resolves the last two components
        # against the calling module's namespace).
        for s in _string_constants(call):
            target = _patch_string_target(s)
            if target is not None:
                targets.append(PatchTarget(path, call.lineno, "patch(string)", target[1], True))
    elif method == "patch.object":
        if args and _is_app_gui(args[0]):
            name_arg = args[1] if len(args) >= 2 else None
            if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
                targets.append(PatchTarget(path, call.lineno, "patch.object(app_gui, ...)", name_arg.value, True))
            else:
                target_desc = ast.dump(name_arg) if name_arg is not None else "?"
                targets.append(PatchTarget(path, call.lineno, "patch.object(app_gui, ...)", target_desc, False))
    elif method == "patch.multiple":
        # patch.multiple("the_oracle.app_gui", name=value, ...): the string is
        # the BARE module (names come from the keywords), so the module.name
        # splitter does not apply — only the exact module spellings route
        # through app_gui's globals. (The module-object form would need the
        # target bound to a value AST cannot statically know.)
        if any(s in {"the_oracle.app_gui", "app_gui"} for s in _string_constants(call)):
            for kw in call.keywords:
                if kw.arg is not None:
                    targets.append(PatchTarget(path, call.lineno, "patch.multiple(app_gui)", kw.arg, True))
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
    # The string form is part of the live surface: this very module patches
    # 'the_oracle.app_gui.QFileDialog.getOpenFileName' by string.
    assert any(t.form.startswith("monkeypatch.setattr('") or t.form.startswith('monkeypatch.setattr("') for t in targets), (
        "no string-form setattr target was found; the string-form scan went blind"
    )


def _scanner(tmp_path: Path, source: str) -> list[PatchTarget]:
    """Run the scanner over a single synthetic test file."""
    source = dedent(source)
    test_file = tmp_path / "test_synthetic_patch_forms.py"
    test_file.write_text(source, encoding="utf-8")
    tree = ast.parse(source)
    targets: list[PatchTarget] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            targets.extend(_scan_call(test_file.name, node))
    return targets


def test_string_form_setattr_is_caught(tmp_path: Path) -> None:
    """The bypass this extension closes: monkeypatch's string form imports
    'the_oracle.app_gui' and patches it — invisible to any scan that only
    inspects module-object first arguments."""
    targets = _scanner(
        tmp_path,
        """
        import monkeypatch_target as app_gui

        def test_uses_string_form(monkeypatch):
            monkeypatch.setattr("the_oracle.app_gui.OraclePipeline", object)
            monkeypatch.setattr("app_gui.find_audiocpp_binary", lambda: None)
            monkeypatch.delattr("the_oracle.app_gui.QMediaPlayer")
        """,
    )
    names = {t.name for t in targets}
    assert names == {"OraclePipeline", "find_audiocpp_binary", "QMediaPlayer"}
    assert all(t.resolved for t in targets)


def test_string_form_owned_name_is_an_offender(tmp_path: Path) -> None:
    """The caught string-form patch feeds the SAME offender logic: an owned
    name patched by string is flagged exactly like the module-object form."""
    targets = _scanner(
        tmp_path,
        """
        def test_offender(monkeypatch):
            monkeypatch.setattr("the_oracle.app_gui.default_gui_settings_payload", lambda: {})
        """,
    )
    offenders = [t.as_offender() for t in targets]
    assert len(offenders) == 1 and offenders[0] is not None
    assert "default_gui_settings_payload" in offenders[0]
    assert "gui_settings" in offenders[0]


def test_bare_app_gui_prefix_is_caught(tmp_path: Path) -> None:
    """mock's abbreviated string form ('app_gui.X') also resolves through
    app_gui — the old scanner comment claimed it, the code never did it."""
    targets = _scanner(
        tmp_path,
        """
        def test_abbreviated(monkeypatch):
            mock.patch("app_gui.QMediaPlayer")
        """,
    )
    assert [t.name for t in targets] == ["QMediaPlayer"]


def test_patch_multiple_keywords_are_individual_targets(tmp_path: Path) -> None:
    """patch.multiple("the_oracle.app_gui", a=1, b=2) is two patch sites; an
    owned name hidden in a keyword must not escape the net."""
    targets = _scanner(
        tmp_path,
        """
        def test_multiple(monkeypatch):
            mock.patch.multiple("the_oracle.app_gui", QMediaPlayer=object, OraclePipeline=object)
        """,
    )
    names = {t.name for t in targets}
    assert names == {"QMediaPlayer", "OraclePipeline"}
    assert all("patch.multiple" in t.form for t in targets)


def test_unrelated_strings_are_ignored(tmp_path: Path) -> None:
    """Strings that merely contain 'app_gui' without patching through it —
    other modules, unrelated imports — must not produce targets."""
    targets = _scanner(
        tmp_path,
        """
        def test_unrelated(monkeypatch):
            monkeypatch.setattr("the_oracle.gui_render.RenderWorker", object)
            monkeypatch.setattr("the_oracle.pipeline.OraclePipeline", object)
            mock.patch("the_oracle.app_gui_render.Thing")
        """,
    )
    assert targets == []


def test_partial_owned_name_patch_is_flagged(tmp_path: Path) -> None:
    """An app_gui-level patch of a name built in two modules is PARTIAL, not a
    no-op: it stubs app_gui's construction sites and silently misses the
    owner module's (the gui_chrome Live column). The net must say so rather
    than let the patch look complete."""
    targets = _scanner(
        tmp_path,
        """
        def test_partial(monkeypatch):
            monkeypatch.setattr(app_gui, "QHSectionGroup", FakeSection)
        """,
    )
    offenders = [t.as_offender() for t in targets]
    assert len(offenders) == 1, offenders
    assert offenders[0] is not None
    assert "PARTIAL" in offenders[0]
    assert "gui_chrome" in offenders[0]


def test_string_form_scan_matches_the_live_suite(tmp_path: Path) -> None:
    """End-to-end: the real suite contains one string-form setattr today
    (QFileDialog.getOpenFileName); the scanner must see it unflagged."""
    targets = scan_app_gui_patch_surface(REPO_ROOT / "tests")
    string_form = [t for t in targets if "setattr(" in t.form and ("'" in t.form or '"' in t.form)]
    assert any(t.name == "getOpenFileName" for t in string_form)


def test_a_malformed_manifest_fails_loudly(tmp_path: Path) -> None:
    """A manifest that no longer parses must break the net, not shrink it.

    The dangerous shape is the quiet one: an empty or restructured section
    that loads fine and simply stops matching, leaving every offender
    untested while the suite stays green. Both loaders — ``moved_owners``
    and ``payload_policy`` — are proven here.
    """
    bad = tmp_path / "patch_surface_manifest.json"

    bad.write_text('{"moved_owners": {}}', encoding="utf-8")
    with pytest.raises(AssertionError, match="non-empty object"):
        load_moved_owners(bad)

    bad.write_text('{"moved_owners": {"the_oracle.gui_render": []}}', encoding="utf-8")
    with pytest.raises(AssertionError, match="must map to an object"):
        load_moved_owners(bad)

    bad.write_text('{"moved_owners": {"the_oracle.gui_render": {"names": []}}}', encoding="utf-8")
    with pytest.raises(AssertionError, match="non-empty 'names' list"):
        load_moved_owners(bad)

    bad.write_text('{"moved_owners": {"the_oracle.gui_render": {"names": "x"}}}', encoding="utf-8")
    with pytest.raises(AssertionError, match="non-empty 'names' list"):
        load_moved_owners(bad)

    bad.write_text('{"owners": {}}', encoding="utf-8")
    with pytest.raises(AssertionError, match="'moved_owners' must be a non-empty object"):
        load_moved_owners(bad)

    # The payload-policy section gets the same loud-failure rule. Each case
    # starts from a complete valid section and breaks exactly one field, so
    # the reported error is the one the case means to prove.
    def policy_json(**overrides: object) -> str:
        policy: dict[str, object] = {
            "payload_widgets": ["variant_combo"],
            "read_accessors": ["text"],
            "sanctioned_readers": ["_widget_snapshot"],
            "policy_references": ["WidgetSnapshot"],
            "policy_owners": {"the_oracle.gui_settings": "definition"},
            "schema_keys": ["model_variant"],
            "schema_exempt_modules": ["the_oracle.gui_settings"],
        }
        policy.update(overrides)
        return json.dumps(
            {
                "moved_owners": {"the_oracle.gui_render": {"names": ["x"]}},
                "payload_policy": policy,
            }
        )

    bad.write_text('{"moved_owners": {"the_oracle.gui_render": {"names": ["x"]}}}', encoding="utf-8")
    with pytest.raises(AssertionError, match="'payload_policy' must be an object"):
        load_payload_policy(bad)

    bad.write_text(policy_json(payload_widgets=[]), encoding="utf-8")
    with pytest.raises(AssertionError, match=r"payload_policy\.payload_widgets must not be empty"):
        load_payload_policy(bad)

    bad.write_text(policy_json(schema_keys="x"), encoding="utf-8")
    with pytest.raises(AssertionError, match=r"payload_policy\.schema_keys must be a list of strings"):
        load_payload_policy(bad)

    bad.write_text(policy_json(policy_owners={}), encoding="utf-8")
    with pytest.raises(AssertionError, match="policy_owners must be a non-empty object"):
        load_payload_policy(bad)

    bad.write_text(policy_json()[:-1] + ', "surprise_section": {}}', encoding="utf-8")
    with pytest.raises(AssertionError, match="unrecognized top-level section"):
        load_payload_policy(bad)


def test_moved_owner_names_are_current() -> None:
    """The manifest's moved-owner entries must match what the owners define.

    Guards the manifest against drift in both directions: a name removed from
    an owner (so app_gui patches become legitimate again) or a new owner module
    landing without being registered there.
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

    # Partial-owned names must also exist on both sides — a split that loses a
    # side stops being partial (and the map entry would then over-flag).
    for name, (window_module, owner_module) in PARTIAL_OWNED.items():
        window_mod = importlib.import_module(window_module)
        owner_mod = importlib.import_module(owner_module)
        assert hasattr(window_mod, name), f"{window_module} lost its {name} site"
        assert hasattr(owner_mod, name), f"{owner_module} lost the {name} it owns"
