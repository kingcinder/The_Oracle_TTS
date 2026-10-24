from __future__ import annotations

import json
from pathlib import Path

import pytest

from the_oracle import cli
from the_oracle.licensing import policy, store, tokens
from the_oracle.licensing.tokens import LicenseStatus
from tests.licensing_fixtures import TEST_KEY_ID, TEST_SEED

pytestmark = pytest.mark.usefixtures("test_license_registry")


def _unlicensed() -> LicenseStatus:
    return LicenseStatus(ok=False, state="no_token", detail="No serial number is activated.")


def _token() -> str:
    return tokens.mint(
        {
            "v": 2,
            "key_id": TEST_KEY_ID,
            "lic_id": "77777777-7777-4777-7777-777777777777",
            "licensee": "Cody",
            "machine_hash": "",
            "iat": 1700000000,
            "exp": None,
        },
        TEST_SEED,
    )


def test_unlicensed_status_rejects_every_functional_feature() -> None:
    with pytest.raises(policy.LicenseRequired, match="serial number"):
        policy.require(_unlicensed(), "render")


@pytest.mark.parametrize("status", [
    tokens.verify_token("not-a-serial-number"),
    tokens.verify_token(
        tokens.mint(
            {
                "v": 2,
                "key_id": "unknown",
                "lic_id": "x",
                "licensee": "",
                "machine_hash": "",
                "iat": 1,
                "exp": None,
            },
            TEST_SEED,
        )
    ),
    tokens.verify_token(_token()[:-1] + ("A" if _token()[-1] != "A" else "B")),
    tokens.verify_token(_token(), machine_hash="b" * 64),
    tokens.verify_token(
        tokens.mint(
            {
                "v": 2,
                "key_id": TEST_KEY_ID,
                "lic_id": "x",
                "licensee": "",
                "machine_hash": "",
                "iat": 1,
                "exp": 1,
            },
            TEST_SEED,
        ),
        now=2,
    ),
])
def test_real_invalid_serial_states_refuse(status: LicenseStatus) -> None:
    assert status.ok is False
    with pytest.raises(policy.LicenseRequired, match="serial number"):
        policy.require(status, "render")


def test_license_status_reports_licensed_boolean_without_edition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from the_oracle import offline

    monkeypatch.setattr(offline, "repo_root", lambda: tmp_path)
    assert cli.handle_license_status() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["licensed"] is False
    assert "edition" not in payload
    store.save_token(tmp_path, _token(), verify_before_save=tokens.verify_token)
    assert cli.handle_license_status() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["licensed"] is True
    assert "edition" not in payload


def test_cli_functional_command_requires_serial_number(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("the_oracle.licensing.current_license", _unlicensed)
    assert cli.main(["voices"]) == 1
    assert "serial number" in capsys.readouterr().err


def test_pipeline_render_entry_requires_serial_number(monkeypatch: pytest.MonkeyPatch) -> None:
    from the_oracle.pipeline import OraclePipeline

    monkeypatch.setattr("the_oracle.licensing.current_license", _unlicensed)
    pipeline = OraclePipeline.__new__(OraclePipeline)
    with pytest.raises(policy.LicenseRequired, match="serial number"):
        pipeline.render(None, None)  # type: ignore[arg-type]
    with pytest.raises(policy.LicenseRequired, match="serial number"):
        pipeline.prepare_plan(None, None, None, None)  # type: ignore[arg-type]
    with pytest.raises(policy.LicenseRequired, match="serial number"):
        pipeline.render_preview(None, None, "standard")  # type: ignore[arg-type]


def test_gui_functional_entry_points_require_serial_number() -> None:
    from the_oracle.app_gui import MainWindow

    class FakeWindow:
        def _require_serial(self, feature: str) -> bool:
            self.feature = feature
            return False

    for method, expected in (
        (MainWindow.new_project, "start a new project"),
        (MainWindow._batch_fix_input_folder, "fix input files"),
        (MainWindow._run_ingest_transformer_check, "check input formatting"),
        (MainWindow.prepare_project, "analyze a project"),
        (MainWindow.open_recording_studio, "record a voice"),
        (MainWindow.download_vulkan_model, "download a model"),
        (MainWindow._save_blend_as_voice, "save a blended voice"),
        (MainWindow._process_selected_voice, "process a voice recording"),
        (MainWindow.preview_utterance, "preview audio"),
    ):
        fake = FakeWindow()
        args = (None, False) if method is MainWindow._process_selected_voice else ((None,) if method in {MainWindow._save_blend_as_voice, MainWindow.preview_utterance} else ())
        result = method(fake, *args)  # type: ignore[arg-type]
        assert result is None or result is False
        assert fake.feature == expected


def test_ingest_entry_points_require_serial_number(monkeypatch: pytest.MonkeyPatch) -> None:
    from the_oracle import gui_ingest_tools

    monkeypatch.setattr("the_oracle.licensing.current_license", _unlicensed)
    with pytest.raises(policy.LicenseRequired, match="serial number"):
        gui_ingest_tools.run_ingest_transformer_check(object(), message_box_cls=object)
    with pytest.raises(policy.LicenseRequired, match="serial number"):
        gui_ingest_tools.batch_fix_input_folder(object(), message_box_cls=object, file_dialog_cls=object)


def test_recording_capture_requires_serial_number(monkeypatch: pytest.MonkeyPatch) -> None:
    from the_oracle import gui_recording

    class Status:
        def __init__(self) -> None:
            self.text = ""

        def setText(self, value: str) -> None:
            self.text = value

    class MessageBox:
        @staticmethod
        def information(_parent, _title: str, _message: str) -> None:
            return None

    monkeypatch.setattr("the_oracle.licensing.current_license", _unlicensed)
    monkeypatch.setattr(gui_recording, "QMessageBox", MessageBox)
    fake = gui_recording.RecordingStudioDialog.__new__(gui_recording.RecordingStudioDialog)
    fake.status = Status()
    fake._start_recording()
    assert "serial number" in fake.status.text


def test_every_cli_subcommand_is_explicitly_classified() -> None:
    choices = cli.build_parser()._subparsers._group_actions[0].choices
    classified = (cli._UNLICENSED_ALLOWED_COMMANDS | cli._SERIAL_REQUIRED_COMMANDS) & set(choices)
    assert set(choices) == classified
    assert {"doctor", "install", "uninstall"} <= cli._UNLICENSED_ALLOWED_COMMANDS


def test_every_gui_action_is_explicitly_classified() -> None:
    from the_oracle.app_gui import SERIAL_REQUIRED_GUI_ACTIONS, UNLICENSED_GUI_ACTIONS

    assert UNLICENSED_GUI_ACTIONS.isdisjoint(SERIAL_REQUIRED_GUI_ACTIONS)
    assert UNLICENSED_GUI_ACTIONS | SERIAL_REQUIRED_GUI_ACTIONS == {
        "activate_license_action",
        "about_action",
        "report_problem_action",
        "privacy_action",
        "new_action",
        "open_action",
        "save_action",
        "save_as_action",
        "save_profile_action",
        "load_profile_action",
        "batch_fix_action",
        "reset_defaults_action",
        "save_settings_action",
        "load_settings_action",
        "save_template_action",
        "replay_full_wizard_action",
        "replay_discovery_wizard_action",
        "replay_main_wizard_action",
        "recording_wizard_action",
        "output_filename_warning_action",
        "confirmation_action",
        "clear_trust_action",
        "restore_backup_action",
        "remember_backend_action",
        "download_vulkan_model_action",
        "recording_studio_action",
        "_theme_actions",
        "templates_menu",
    }


def test_unlicensed_gui_leaves_only_help_and_activation_actions_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from the_oracle.app_gui import MainWindow

    class Action:
        def __init__(self) -> None:
            self.enabled = True

        def isEnabled(self) -> bool:
            return self.enabled

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

    class Central:
        def __init__(self) -> None:
            self.enabled = True

        def findChildren(self, _kind):
            return []

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

    class MenuBar:
        def __init__(self, actions):
            self.actions = actions

        def findChildren(self, _kind):
            return self.actions

    class Panel:
        def append(self, _message: str) -> None:
            return None

    actions = [Action() for _ in range(5)]
    fake = MainWindow.__new__(MainWindow)
    fake.centralWidget = lambda: Central()
    fake.menuBar = lambda: MenuBar(actions)
    fake.error_panel = Panel()
    fake.activate_license_action, fake.about_action, fake.report_problem_action, fake.privacy_action = actions[:4]
    fake._serial_gate_action_states = {}
    monkeypatch.setattr("the_oracle.licensing.current_license", _unlicensed)
    MainWindow._apply_serial_gate(fake)  # type: ignore[arg-type]
    assert all(action.enabled for action in actions[:4])
    assert actions[4].enabled is False
    assert cli._UNLICENSED_ALLOWED_COMMANDS.isdisjoint(cli._SERIAL_REQUIRED_COMMANDS)
