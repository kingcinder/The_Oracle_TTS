"""Release scripts must not depend on the original developer checkout."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_debug_tools_derive_their_checkout_path() -> None:
    shell_source = (REPO_ROOT / "scripts/u42_crash_catcher.sh").read_text(encoding="utf-8")
    python_source = (REPO_ROOT / "scripts/u42_crash_repro.py").read_text(encoding="utf-8")
    assert "/home/" not in shell_source
    assert "/home/" not in python_source
    assert 'dirname "${BASH_SOURCE[0]}"' in shell_source
    assert "Path(__file__).resolve().parents[1]" in python_source


def test_sanitizer_tests_use_a_synthetic_home() -> None:
    source = (REPO_ROOT / "tests/test_crash_sanitize.py").read_text(encoding="utf-8")
    assert 'HOME = "/home/cody"' not in source
    assert 'HOME = "/home/test-user"' in source


def test_shell_scripts_fail_closed_on_checkout_change_and_quote_launcher_glob() -> None:
    catcher = (REPO_ROOT / "scripts/u42_crash_catcher.sh").read_text(encoding="utf-8")
    launcher = (REPO_ROOT / "packaging/_launcher.sh").read_text(encoding="utf-8")
    assert 'cd "$REPO" || exit 1' in catcher
    assert 'for dt in "$HERE/${APP_SLUG}-"*.desktop; do' in launcher
