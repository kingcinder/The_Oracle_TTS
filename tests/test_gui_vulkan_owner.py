"""Ownership tests for the Vulkan/audio.cpp GUI extraction (gui_vulkan.py).

The per-cluster damage assessment said the Vulkan cluster's wholesale patch
surface must be preserved. The first extraction attempt moved the two policy
functions and failed exactly the tests that patch
``app_gui.find_audiocpp_binary`` / ``app_gui._vulkan_preflight_report`` by
name — a function body resolving those names from its own module globals is
invisible to app_gui-level monkeypatching.

Two slices landed under that constraint:

1. (2026-09-20) only the patch-free helpers moved — ``_device_row_text``,
   ``_parse_oracle_model_path`` — and the patch-coupled policies
   (``_vulkan_prerequisite_missing``, ``_vulkan_preflight_report``) STAY on
   app_gui, with their source-level patch behavior intact.
2. (2026-09-27) the four background threads — ``VulkanDeviceProbeThread``,
   ``VulkanPreflightThread``, ``ModelDownloadThread``, ``VulkanSetupThread``
   — moved with their bodies verbatim, because those bodies resolve only
   move-safe names (module-attribute patches of ``subprocess``/``os`` and
   class-target patches of ``AudioCppVulkanEngine`` intercept identically
   from any importer). The one patch-coupled dependency, the preflight
   report builder, crosses the seam by CONSTRUCTOR INJECTION: MainWindow
   passes the bare ``_vulkan_preflight_report`` name, which resolves from
   app_gui's globals at call time, so app_gui-level patches reach it exactly
   as they reached the old in-module call.

The contracts pinned here: one owner per class (gui_vulkan defines, app_gui
re-imports the identical objects so wholesale construction patches stay
live); the thread bodies never resolve an app_gui-patched policy name as a
bare global (the harm class itself); the preflight seam is injected as a
bare name from app_gui's namespace; and gui_vulkan imports app_gui in NO
spelling — the seam must never be "fixed" with a back-import.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import(name: str):
    return importlib.import_module(name)


def test_patch_free_helpers_live_in_gui_vulkan() -> None:
    owner = _import("the_oracle.gui_vulkan")
    for name in ("_device_row_text", "_parse_oracle_model_path"):
        assert callable(getattr(owner, name)), name


def test_app_gui_delegates_device_row_text_to_the_owner() -> None:
    owner = _import("the_oracle.gui_vulkan")
    app_gui = _import("the_oracle.app_gui")
    assert app_gui._device_row_text is owner._device_row_text
    assert app_gui._parse_oracle_model_path is owner._parse_oracle_model_path
    assert owner._device_row_text(2, "AMD Radeon") == "Device 2: AMD Radeon"


def test_patch_coupled_policies_stay_on_app_gui() -> None:
    """The two policy functions' bodies resolve find_audiocpp_binary from
    app_gui's own globals — the property the patch surface depends on."""
    app_gui = _import("the_oracle.app_gui")
    assert callable(app_gui._vulkan_prerequisite_missing)
    assert callable(app_gui._vulkan_preflight_report)
    app_gui_source = Path(app_gui.__file__).read_text(encoding="utf-8")
    assert "def _vulkan_prerequisite_missing" in app_gui_source
    assert "def _vulkan_preflight_report" in app_gui_source


def test_prerequisite_missing_sees_app_gui_level_binary_patch(monkeypatch) -> None:
    """The load-bearing patch property, exercised directly: patching
    app_gui.find_audiocpp_binary must change what the policy function sees."""
    app_gui = _import("the_oracle.app_gui")
    from the_oracle.tts_engines import vulkan_backend

    monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: None)
    monkeypatch.delenv("ORACLE_AUDIOCPP_MODEL", raising=False)
    # find_audiocpp_model() is patched too so the check isolates the binary probe.
    monkeypatch.setattr(vulkan_backend, "find_audiocpp_model", lambda: None)
    missing = app_gui._vulkan_prerequisite_missing()
    assert missing and "audiocpp_cli is not built" in missing[0]


def test_gui_vulkan_has_no_wholesale_patch_targets() -> None:
    """Nothing in gui_vulkan is monkeypatched by name anywhere in the suite."""
    owner = _import("the_oracle.gui_vulkan")
    owner_names = {name for name in vars(owner) if not name.startswith("__")}
    tests_dir = REPO_ROOT / "tests"
    offenders: list[str] = []
    for test_file in tests_dir.glob("test_*.py"):
        text = test_file.read_text(encoding="utf-8")
        for name in owner_names:
            if f'"gui_vulkan", "{name}"' in text or f"'gui_vulkan', '{name}'" in text:
                offenders.append(f"{test_file.name}:{name}")
    assert offenders == []


def test_gui_vulkan_imports_nothing_from_app_gui() -> None:
    """No import cycle and no seam bypass: gui_vulkan must never reach back
    into app_gui — in ANY spelling.

    The bare-name check this replaces (added 2026-09-20) only caught
    ``import app_gui`` / ``from app_gui import x``; it would pass green over
    ``from the_oracle import app_gui``, ``from . import app_gui`` and friends.
    The harm this guards is someone "fixing" the injected preflight seam with
    ``from the_oracle.app_gui import _vulkan_preflight_report`` — a cycle the
    direction guards exist to refuse.
    """
    from tests.test_gui_import_direction import _module_root, _scan_imports

    _forbids_app_gui = lambda module: _module_root(module) == "app_gui"
    owner = _import("the_oracle.gui_vulkan")
    source = Path(owner.__file__).read_text(encoding="utf-8")
    scan = _scan_imports(source, forbidden=_forbids_app_gui)
    assert scan.violations == () and scan.unresolved == (), (
        "gui_vulkan must not import app_gui in any form:\n  "
        + "\n  ".join(scan.violations + scan.unresolved)
    )
    # Vacuity: each forbidden spelling must still trip this exact gate.
    for form in (
        "import the_oracle.app_gui",
        "import app_gui",
        "from the_oracle.app_gui import _vulkan_preflight_report",
        "from the_oracle import app_gui",
        "from . import app_gui",
        "from .app_gui import _vulkan_preflight_report",
        "importlib.import_module('the_oracle.app_gui')",
    ):
        assert _scan_imports(form, forbidden=_forbids_app_gui).violations, form
    # ...and the scan must see gui_vulkan's real imports (which are legal):
    assert {
        "the_oracle.gui_utils",
        "the_oracle.tts_engines.vulkan_backend",
        "the_oracle.vulkan_setup",
    } <= set(scan.modules), scan.modules


def test_vulkan_thread_cluster_has_one_owner() -> None:
    """The four background threads define themselves in gui_vulkan and
    app_gui holds the identical objects.

    Identity is load-bearing: MainWindow constructs the threads from bare
    names in app_gui's namespace, so the wholesale class patches
    (``monkeypatch.setattr(app_gui, "VulkanPreflightThread", fake)``)
    intercept construction only while ``app_gui.X is gui_vulkan.X``.
    """
    owner = _import("the_oracle.gui_vulkan")
    app_gui = _import("the_oracle.app_gui")
    owner_tree = ast.parse(Path(owner.__file__).read_text(encoding="utf-8"))
    defined = {
        node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)
    }
    for name in (
        "VulkanDeviceProbeThread",
        "VulkanPreflightThread",
        "ModelDownloadThread",
        "VulkanSetupThread",
    ):
        assert name in defined, f"gui_vulkan must define {name}"
        assert getattr(app_gui, name) is getattr(owner, name), (
            f"app_gui.{name} must be the gui_vulkan class itself — a wrapper "
            "or subclass would silently defeat the wholesale construction patches"
        )


#: Names whose app_gui-global identity the patch surface depends on. A body
#: resolving any of these from its own module globals is invisible to
#: app_gui-level monkeypatching — the harm class the 2026-09-20 extraction
#: attempt hit (9 failed tests, journaled).
_APP_GUI_PATCHED_POLICY_NAMES = frozenset(
    {
        "_vulkan_preflight_report",
        "_vulkan_prerequisite_missing",
        "find_audiocpp_binary",
        "find_audiocpp_model",
    }
)


def _bare_policy_refs(cls: ast.ClassDef) -> list[str]:
    """The app_gui-patched policy names a class body reads as bare globals."""
    return sorted(
        {
            node.id
            for node in ast.walk(cls)
            if isinstance(node, ast.Name) and node.id in _APP_GUI_PATCHED_POLICY_NAMES
        }
    )


def test_thread_bodies_never_resolve_app_gui_patched_policy_names() -> None:
    """The harm-class pin: the moved thread bodies must not read an
    app_gui-patched policy name as a bare global.

    Such a reference resolves from gui_vulkan's globals — the patch applies
    to app_gui, the moved body never sees it. ``VulkanPreflightThread`` gets
    its builder by constructor injection instead (the seam pin below); the
    other three have no patch-coupled dependency at all.
    """
    owner = _import("the_oracle.gui_vulkan")
    source = Path(owner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    threads = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    offenders: list[str] = []
    for name in (
        "VulkanDeviceProbeThread",
        "VulkanPreflightThread",
        "ModelDownloadThread",
        "VulkanSetupThread",
    ):
        assert name in threads, f"{name} must be findable in gui_vulkan — the scan went blind"
        offenders += [f"{name}: {ref}" for ref in _bare_policy_refs(threads[name])]
    assert offenders == [], (
        "thread bodies resolve app_gui-patched names from their own globals; "
        "inject the collaborator instead (see VulkanPreflightThread):\n  "
        + "\n  ".join(offenders)
    )
    # Vacuity: the scan must flag a synthetic offender — a blind walk passes
    # green over anything.
    probe = ast.parse(
        "class P:\n    def run(self):\n        return _vulkan_preflight_report(0)\n"
    ).body[0]
    assert _bare_policy_refs(probe) == ["_vulkan_preflight_report"]


def test_preflight_builder_is_injected_from_app_gui_namespace() -> None:
    """MainWindow must construct VulkanPreflightThread with
    ``preflight_report=_vulkan_preflight_report`` as a bare name.

    The bare name resolves from app_gui's globals at call time, so
    ``monkeypatch.setattr(app_gui, "_vulkan_preflight_report", stub)``
    reaches the seam; a captured reference, literal, or attribute lookup
    would pin the unpatched builder and re-open the harm class.
    """
    app_gui = _import("the_oracle.app_gui")
    tree = ast.parse(Path(app_gui.__file__).read_text(encoding="utf-8"))
    injections: list[ast.AST | None] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "VulkanPreflightThread":
                keywords = {kw.arg: kw.value for kw in node.keywords}
                injections.append(keywords.get("preflight_report"))
    assert injections, (
        "no VulkanPreflightThread construction found in app_gui — "
        "the seam pin went blind"
    )
    for value in injections:
        assert isinstance(value, ast.Name) and value.id == "_vulkan_preflight_report", (
            "every construction must inject the bare _vulkan_preflight_report "
            f"name, got: {ast.dump(value) if value is not None else None}"
        )
