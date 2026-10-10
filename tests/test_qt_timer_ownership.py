"""Deferred-callback ownership pin for the GUI source.

``QTimer.singleShot(msec, bound_method)`` holds the receiver through
PySide6's undocumented overload semantics — measured on this build, weakly:
destroy the receiver first and the call is silently skipped. The explicit
context form ``QTimer.singleShot(msec, context, callable)`` (or a child
``QTimer`` of the owner, which every startup/launch callback already uses)
carries a documented guarantee instead: the timer lives and dies with its
owner, so a pending call can neither outlive the window nor depend on a
binding implementation detail.

The 2026-10-10 dialog audit found the last two context-less deferrals
(``RecordingStudioDialog._on_playback_status`` and
``MainWindow._on_preview_playback_status`` — both QtMultimedia EndOfMedia
stop-deferrals) and converted them; this test keeps the pattern from
coming back.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "the_oracle"


def _contextless_single_shots() -> list[str]:
    """Every ``QTimer.singleShot`` call in src that passes no receiver."""
    offenders: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "singleShot"
                and isinstance(func.value, ast.Name)
                and func.value.id == "QTimer"
            ):
                continue
            if len(node.args) < 3:
                rel = path.relative_to(_SRC_ROOT.parent.parent)
                offenders.append(f"{rel}:{node.lineno}: needs an explicit context receiver")
    return offenders


def test_every_singleshot_passes_an_explicit_context() -> None:
    offenders = _contextless_single_shots()
    assert not offenders, (
        "context-less QTimer.singleShot calls found — pass the owner as the "
        "context argument (QTimer.singleShot(ms, self, callable)) or use a "
        "child QTimer:\n" + "\n".join(offenders)
    )


def test_the_scanner_sees_a_synthetic_offender() -> None:
    """Self-check of the AST rule: a 2-arg QTimer.singleShot is caught, a
    3-arg context form is not (guards against the scan silently matching
    nothing)."""
    import tempfile

    src = (
        "from PySide6.QtCore import QTimer\n"
        "def f(self):\n"
        "    QTimer.singleShot(0, self._stop)\n"
        "    QTimer.singleShot(0, self, self._stop)\n"
        "    QTimer.singleShot(0, self, lambda: None)\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "the_oracle"
        root.mkdir()
        (root / "probe_mod.py").write_text(src, encoding="utf-8")
        global _SRC_ROOT
        real = _SRC_ROOT
        _SRC_ROOT = Path(tmp)
        try:
            offenders = _contextless_single_shots()
        finally:
            _SRC_ROOT = real
    assert len(offenders) == 1, offenders
    assert "probe_mod.py:3" in offenders[0]
