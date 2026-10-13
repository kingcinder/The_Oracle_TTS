"""Rendering works in every license state — the Phase 1 promise, by execution.

`docs/LICENSING_DESIGN.md` §4 promises that an expired or absent licence
degrades to community, that renders in flight finish, and that **nothing is ever
locked away**. Until now that promise was pinned only at the policy layer
(entitlement sets and `require`), which is a statement about a table, not about a
render. This file drives a real end-to-end render — the deterministic smoke
render, which exercises the same prepare/synthesize/assemble/export path the
product uses without loading the model — under each of the three licence states
and asserts the audio is produced and the result is clean.

**Mutation contract:** adding a `policy.require(...)` call to the render path
must fail ``test_the_render_path_contains_no_entitlement_gate``; making an
expired licence block rendering must fail
``test_a_render_succeeds_under_every_license_state``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from the_oracle import cli, pipeline, smoke  # noqa: E402
from the_oracle.licensing import store, tokens  # noqa: E402
from the_oracle.licensing.policy import edition_entitlements  # noqa: E402
from tests.licensing_fixtures import TEST_SEED, payload  # noqa: E402

pytestmark = pytest.mark.usefixtures("test_license_registry")


def _isolate(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("the_oracle.offline.repo_root", lambda: root)
    return root


def _community(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """No token at all: the fail-closed default (community)."""
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
    ("setup_state", "expected_edition"),
    [(_community, "community"), (_expired_studio, "community"), (_studio, "studio")],
    ids=["community-no-token", "expired-studio", "valid-studio"],
)
def test_a_render_succeeds_under_every_license_state(
    setup_state,
    expected_edition: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each state renders; none of them can take a render away."""
    setup_state(monkeypatch, tmp_path / "install")

    # Assert the state really is the state (vacuity guard): the licence verdict
    # is read through the same call the feature gates use.
    from the_oracle.licensing import current_license
    from the_oracle.licensing.policy import current_edition

    edition = current_edition(current_license())
    assert edition == expected_edition, f"expected {expected_edition}, got {edition}"

    result = _render(tmp_path / "render")
    capsys.readouterr()  # the smoke prints progress; keep the test output clean

    assert result.output_path.exists(), result.to_dict()
    assert result.output_path.suffix == ".flac"
    assert result.stem_count > 0
    assert smoke.smoke_output_problem(result) is None


def test_a_valid_studio_license_also_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The licensed state is not a special path: it renders identically."""
    _studio(monkeypatch, tmp_path / "install")
    from the_oracle.licensing import current_license
    from the_oracle.licensing.policy import current_edition

    assert current_edition(current_license()) == "studio"

    result = _render(tmp_path / "render")
    capsys.readouterr()
    assert smoke.smoke_output_problem(result) is None


def test_the_render_path_contains_no_entitlement_gate() -> None:
    """Structurally: nothing on the render path consults an entitlement.

    The policy-layer pin alone would not catch a gate added to `handle_render`
    or the pipeline, which is exactly how "nothing is ever locked away" would
    quietly stop being true.
    """
    cli_source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "def handle_render(" in cli_source
    handler = cli_source.split("def handle_render(", 1)[1].split("\ndef ", 1)[0]
    assert "require(" not in handler, "handle_render must not gate on an entitlement"
    assert "current_license(" not in handler, "handle_render must not read the licence at all"

    pipeline_source = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert "from the_oracle.licensing" not in pipeline_source, (
        "the pipeline must not import licensing at all"
    )
    assert "policy.require" not in pipeline_source
    assert "current_license" not in pipeline_source

    # And the entitlement table agrees: render is community's, unconditionally.
    assert "render" in edition_entitlements("community")
