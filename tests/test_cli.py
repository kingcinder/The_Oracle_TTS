import pytest

from the_oracle import __version__
from the_oracle.cli import build_parser


def test_cli_version_flag_reports_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--version"])

    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_description_mentions_conversation_and_monologue() -> None:
    parser = build_parser()

    assert "conversation and monologue" in parser.description
