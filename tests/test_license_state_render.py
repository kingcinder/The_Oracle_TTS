"""Pin the product gate at the real pipeline render entry point."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import cli, pipeline, smoke  # noqa: E402
from the_oracle.licensing import policy, store, tokens  # noqa: E402
from tests.licensing_fixtures import TEST_SEED, payload  # noqa: E402

pytestmark = pytest.mark.usefixtures("test_license_registry")


def _isolate(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("the_oracle.offline.repo_root", lambda: root)
    return root


def _unlicensed(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """No token at all: the fail-closed default."""
    _isolate(monkeypatch, root)


def _studio(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    root = _isolate(monkeypatch, root)
    token = tokens.mint(payload(), TEST_SEED)
    result = store.save_token(root, token, verify_before_save=lambda t: tokens.verify_token(t))
    assert result.state == "saved", result


def _expired_studio(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """A token that was valid at activation and has since passed its ``exp``.

    `save_token` refuses an already-expired token (by design), so the store is
    written directly — the shape a real aged token has on disk.
    """
    import json

    root = _isolate(monkeypatch, root)
    token = tokens.mint(payload(exp=int(time.time()) - 10), TEST_SEED)
    store.store_path(root).write_text(json.dumps({"token": token}), encoding="utf-8")


def _render(output_root: Path) -> smoke.SmokeRenderResult:
    return smoke.run_deterministic_smoke_render(output_root, source_format="txt")


@pytest.mark.parametrize(
    ("setup_state", "expected_state"),
    [(_unlicensed, "no_token"), (_expired_studio, "expired")],
    ids=["no-token", "expired-key"],
)
def test_a_render_refuses_unlicensed_states(
    setup_state,
    expected_state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Absent and expired serial numbers cannot reach synthesis."""
    setup_state(monkeypatch, tmp_path / "install")

    # Assert the state really is the state (vacuity guard): the licence verdict
    # is read through the same call the feature gates use.
    from the_oracle.licensing import current_license

    state = current_license().state
    assert state == expected_state, f"expected {expected_state}, got {state}"

    with pytest.raises(policy.LicenseRequired, match="serial number"):
        _render(tmp_path / "render")
    capsys.readouterr()


def test_a_valid_serial_number_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid serial number enables the normal render path."""
    _studio(monkeypatch, tmp_path / "install")
    from the_oracle.licensing import current_license

    assert current_license().state == "valid"

    result = _render(tmp_path / "render")
    capsys.readouterr()
    assert smoke.smoke_output_problem(result) is None


def test_the_render_path_contains_the_serial_gate() -> None:
    """Structurally: CLI dispatch and pipeline retain the core gate."""
    cli_source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "def handle_render(" in cli_source
    handler = cli_source.split("def handle_render(", 1)[1].split("\ndef ", 1)[0]
    assert "require(" not in handler, "CLI dispatch owns the command gate"

    pipeline_source = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert "from the_oracle.licensing import current_license, require" in pipeline_source
    assert pipeline_source.count("require(current_license()") >= 3
