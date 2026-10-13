#!/usr/bin/env python3
"""Run the GUI certifier with CtrlHoverHelp.eventFilter replaced by a no-op.

This diagnostic separates failure in Python event-filter dispatch from failure
in the filter's event handling. It is not a product entry point.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from the_oracle.gui_tooltips import CtrlHoverHelp  # noqa: E402


def _ignore_events(self, _watched, _event) -> bool:
    return False


CtrlHoverHelp.eventFilter = _ignore_events
sys.argv[0] = str(REPO / "scripts" / "certify_gui.py")
runpy.run_path(sys.argv[0], run_name="__main__")
