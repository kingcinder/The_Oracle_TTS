"""GUI tests for the input-formatting preview dialog and popup flow."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from the_oracle.ingest_transformer import preview_fixed_text

pytestmark = pytest.mark.slow


def _fake_pipeline_class():
    from tests.test_app_gui_profiles import _FakePipeline

    return _FakePipeline


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class _FakeAudioOutput:
    def __init__(self, *_args, **_kwargs) -> None:
        pass


class _FakeMediaPlayer:
    def __init__(self, *_args, **_kwargs) -> None:
        from tests.test_app_gui_profiles import _FakeSignal

        self.audio_output = None
        self.source = None
        self.played = 0
        self.stopped = 0
        self.mediaStatusChanged = _FakeSignal()

    def setAudioOutput(self, output) -> None:
        self.audio_output = output

    def setSource(self, source) -> None:
        self.source = source

    def play(self) -> None:
        self.played += 1

    def stop(self) -> None:
        self.stopped += 1


def _build_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from tests.test_app_gui_profiles import (
        _FakeChatterboxEngine,
        _FakeModelDownloadThread,
        _FakePipeline,
        _FakeSignal,
        _FakeVulkanPreflightThread,
        _FakeVulkanProbeThread,
        _FakeVulkanSetupThread,
    )
    import the_oracle.app_gui as app_gui
    from the_oracle.app_paths import ensure_repo_default_paths

    paths = ensure_repo_default_paths(tmp_path / "repo")
    monkeypatch.setattr(app_gui, "QAudioOutput", _FakeAudioOutput)
    monkeypatch.setattr(app_gui, "QMediaPlayer", _FakeMediaPlayer)
    monkeypatch.setattr(app_gui, "OraclePipeline", _FakePipeline)
    monkeypatch.setattr(app_gui, "ChatterboxEngine", _FakeChatterboxEngine)
    monkeypatch.setattr(app_gui, "VulkanDeviceProbeThread", _FakeVulkanProbeThread)
    monkeypatch.setattr(app_gui, "VulkanPreflightThread", _FakeVulkanPreflightThread)
    monkeypatch.setattr(app_gui, "VulkanSetupThread", _FakeVulkanSetupThread)
    monkeypatch.setattr(app_gui, "ModelDownloadThread", _FakeModelDownloadThread)
    monkeypatch.setattr(app_gui, "ensure_repo_default_paths", lambda _repo_root: paths)
    monkeypatch.setattr(app_gui, "default_voice_choices", lambda _repo_root, **_kwargs: [])
    monkeypatch.setattr(app_gui, "load_recent_reference_paths", lambda limit=10: [])
    model_file = tmp_path / "chatterbox-model"
    model_file.write_text("model", encoding="utf-8")
    monkeypatch.setattr(app_gui, "find_audiocpp_binary", lambda: tmp_path / "audiocpp_cli")
    monkeypatch.setenv("ORACLE_AUDIOCPP_MODEL", str(model_file))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    window = app_gui.MainWindow()
    return window, paths


def test_preview_dialog_shows_side_by_side_and_accepts(qt_app, monkeypatch, tmp_path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    original = "A - Hello there.\nB - Hi back.\n"
    fixed = "A: Hello there.\nB: Hi back.\n"

    # Invoke the preview builder directly and simulate acceptance by calling
    # the dialog's accept() from inside a patched exec().
    import the_oracle.app_gui as app_gui

    class _PreviewDialog(app_gui.QDialog):
        accepted = False

        def exec(self):  # noqa: D102 - test double
            _PreviewDialog.accepted = True
            from PySide6.QtWidgets import QPlainTextEdit

            views = self.findChildren(QPlainTextEdit)
            assert len(views) == 2, "side-by-side preview needs two panes"
            left, right = views[0].toPlainText(), views[1].toPlainText()
            # Left pane: the original; right pane: corrections with their
            # fix-rule label as a suffix.
            assert "A - Hello there." in left
            assert "B - Hi back." in left
            assert "+ A: Hello there. [dash/pipe separator]" in right
            assert "+ B: Hi back. [dash/pipe separator]" in right
            # Pane headers identify the layout.
            from PySide6.QtWidgets import QLabel

            labels = [w.text() for w in self.findChildren(QLabel)]
            assert "Original" in labels
            assert any("Corrected" in label for label in labels)
            return app_gui.QDialog.DialogCode.Accepted

    monkeypatch.setattr(app_gui, "QDialog", _PreviewDialog)
    from the_oracle.ingest_transformer import transform_text_detailed

    _fixed, line_fixes = transform_text_detailed(original)
    result = window._show_fix_preview_dialog(original, fixed, 2, line_fixes=line_fixes)
    assert result is True
    assert _PreviewDialog.accepted


def test_preview_dialog_synchronizes_pane_scrolling(qt_app, monkeypatch, tmp_path) -> None:
    """Scrolling one pane drives the other (long-script reviewability)."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    original = "A - Hello there.\nB - Hi back.\n"
    fixed = "A: Hello there.\nB: Hi back.\n"
    import the_oracle.app_gui as app_gui

    captured: dict = {}

    class _PreviewDialog(app_gui.QDialog):
        def exec(self):  # noqa: D102 - test double: capture scroll coupling
            from PySide6.QtWidgets import QPlainTextEdit

            views = self.findChildren(QPlainTextEdit)
            left_bar = views[0].verticalScrollBar()
            right_bar = views[1].verticalScrollBar()
            # Drive the left scrollbar; the right one must follow.
            left_bar.setValue(int(left_bar.maximum() / 2))
            captured["followed"] = right_bar.value() == left_bar.value()
            # And the reverse direction.
            right_bar.setValue(0)
            captured["followed_back"] = left_bar.value() == 0
            return app_gui.QDialog.DialogCode.Accepted

    monkeypatch.setattr(app_gui, "QDialog", _PreviewDialog)
    window._show_fix_preview_dialog(original, fixed, 2)
    assert captured["followed"] and captured["followed_back"]


def test_preview_dialog_reject_leaves_file(qt_app, monkeypatch, tmp_path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui

    class _PreviewDialog(app_gui.QDialog):
        def exec(self):  # noqa: D102 - test double
            return app_gui.QDialog.DialogCode.Rejected

    monkeypatch.setattr(app_gui, "QDialog", _PreviewDialog)
    result = window._show_fix_preview_dialog("A - hi.\n", "A: hi.\n", 1)
    assert result is False


def test_transformer_check_shows_popup_and_previews_before_fix(
    qt_app, monkeypatch, tmp_path
) -> None:
    """The full popup flow: warning popup -> preview dialog -> file fixed."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui
    from tests.test_app_gui_profiles import _FakeSignal  # noqa: F401

    target = tmp_path / "input" / "messy.txt"
    target.parent.mkdir(exist_ok=True)
    target.write_text("A - Hello there.\nB - Hi back.\n", encoding="utf-8")
    window.input_path.setText(str(target))

    # The warning popup: user clicks the "Fix File Automatically" button.
    popup_buttons: list[object] = []

    class _Popup(app_gui.QMessageBox):
        fix_button = None

        # Static popups the check flow raises after the fix must not block
        # offscreen: stub them like exec above.
        information = staticmethod(lambda *a, **k: 0)
        critical = staticmethod(lambda *a, **k: 0)

        def exec(self):  # noqa: D102 - test double
            assert self.icon() == app_gui.QMessageBox.Icon.Warning
            assert "formatting" in self.text().lower()
            _Popup.fix_button = self.buttons()[0]
            self.clickedButton = lambda: _Popup.fix_button  # type: ignore[method-assign]
            return 0

    monkeypatch.setattr(app_gui, "QMessageBox", _Popup)

    # The preview dialog: user accepts.
    class _PreviewDialog(app_gui.QDialog):
        def exec(self):  # noqa: D102 - test double
            from PySide6.QtWidgets import QPlainTextEdit

            views = self.findChildren(QPlainTextEdit)
            assert views, "preview dialog must appear before the fix is applied"
            return app_gui.QDialog.DialogCode.Accepted

    real_qdialog = app_gui.QDialog
    monkeypatch.setattr(
        app_gui, "QDialog", type("QDialog", (real_qdialog,), {"exec": _PreviewDialog.exec})
    )

    proceed = window._run_ingest_transformer_check()
    assert proceed is True
    assert target.read_text(encoding="utf-8") == "A: Hello there.\nB: Hi back.\n"
    # A timestamped backup of the original exists next to the file.
    backups = list(target.parent.glob("messy.txt.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "A - Hello there.\nB - Hi back.\n"


def test_transformer_check_cancel_in_preview_keeps_file(qt_app, monkeypatch, tmp_path) -> None:
    """Cancelling inside the preview dialog leaves the file untouched."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui

    target = tmp_path / "input" / "messy2.txt"
    target.parent.mkdir(exist_ok=True)
    target.write_text("A - Hello there.\n", encoding="utf-8")
    window.input_path.setText(str(target))

    class _Popup(app_gui.QMessageBox):
        information = staticmethod(lambda *a, **k: 0)
        critical = staticmethod(lambda *a, **k: 0)

        def exec(self):  # noqa: D102 - test double
            _Popup.fix_button = self.buttons()[0]
            self.clickedButton = lambda: _Popup.fix_button  # type: ignore[method-assign]
            return 0

    monkeypatch.setattr(app_gui, "QMessageBox", _Popup)

    real_qdialog = app_gui.QDialog

    class _PreviewDialog(real_qdialog):
        def exec(self):  # noqa: D102 - test double
            return real_qdialog.DialogCode.Rejected

    monkeypatch.setattr(app_gui, "QDialog", _PreviewDialog)

    proceed = window._run_ingest_transformer_check()
    assert proceed is False
    # File untouched, no backup written.
    assert target.read_text(encoding="utf-8") == "A - Hello there.\n"
    assert not list(target.parent.glob("messy2.txt.bak-*"))


def test_clean_file_skips_popup_entirely(qt_app, monkeypatch, tmp_path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui

    target = tmp_path / "input" / "clean.txt"
    target.parent.mkdir(exist_ok=True)
    target.write_text("A: Hello there.\nB: Hi back.\n", encoding="utf-8")
    window.input_path.setText(str(target))

    def _fail_exec(self):  # pragma: no cover - must never run
        raise AssertionError("no popup should appear for a clean file")

    monkeypatch.setattr(app_gui.QMessageBox, "exec", _fail_exec)
    assert window._run_ingest_transformer_check() is True
    assert target.read_text(encoding="utf-8") == "A: Hello there.\nB: Hi back.\n"


def test_analyze_reruns_automatically_after_fix(qt_app, monkeypatch, tmp_path) -> None:
    """One Analyze click on a messy file must produce a plan built from the
    CORRECTED content: fix -> Analyze re-runs itself with no second click."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui
    from the_oracle.models.project import RenderPlan, Utterance, VoiceProfile, VoiceSettings

    target = tmp_path / "input" / "messy3.txt"
    target.parent.mkdir(exist_ok=True)
    target.write_text("A - Hello there.\nB - Hi back.\n", encoding="utf-8")
    window.input_path.setText(str(target))

    seen_at_plan_time: dict[str, str] = {}

    class _PlanPipeline(_fake_pipeline_class()):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

        def prepare_plan(self, input_path, output_dir, speaker_settings, settings):
            seen_at_plan_time["content"] = Path(input_path).read_text(encoding="utf-8")
            profile = VoiceProfile(name="A", speaker="A", reference_audio=[], engine_params=VoiceSettings())
            plan = RenderPlan(
                title="t",
                source_path=input_path,
                output_dir=str(output_dir),
                engine="chatterbox",
                correction_mode="moderate",
                metadata={"model_variant": "standard"},
                voice_profiles={"A": profile, "B": profile},
            )
            plan.utterances = [
                Utterance(index=0, original_text="Hello there.", repaired_text="Hello there.", speaker="A", emotion="neutral"),
                Utterance(index=1, original_text="Hi back.", repaired_text="Hi back.", speaker="B", emotion="neutral"),
            ]
            return plan

    monkeypatch.setattr(app_gui, "OraclePipeline", _PlanPipeline)

    class _Popup(app_gui.QMessageBox):
        information = staticmethod(lambda *a, **k: None)
        critical = staticmethod(lambda *a, **k: None)

        def exec(self):  # noqa: D102 - test double: user clicks the fix button
            _Popup.fix_button = self.buttons()[0]
            self.clickedButton = lambda: _Popup.fix_button  # type: ignore[method-assign]
            return 0

    real_qdialog = app_gui.QDialog

    class _PreviewDialog(real_qdialog):
        def exec(self):  # noqa: D102 - test double: user accepts the preview
            return real_qdialog.DialogCode.Accepted

    monkeypatch.setattr(app_gui, "QMessageBox", _Popup)
    monkeypatch.setattr(app_gui, "QDialog", _PreviewDialog)

    window.prepare_project()  # a single Analyze click

    # The plan was built from the corrected file, not the original.
    assert seen_at_plan_time["content"] == "A: Hello there.\nB: Hi back.\n"
    assert window.plan is not None
    assert [u.speaker for u in window.plan.utterances] == ["A", "B"]
    # The file and its backup are on disk exactly as the flow promises.
    assert target.read_text(encoding="utf-8") == "A: Hello there.\nB: Hi back.\n"
    backups = list(target.parent.glob("messy3.txt.bak-*"))
    assert len(backups) == 1


def test_status_panel_reports_the_auto_fix(qt_app, monkeypatch, tmp_path) -> None:
    """The error/status panel records the correction and the automatic
    re-analysis, so the one-click flow is visible instead of silent."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui

    target = tmp_path / "input" / "messy4.txt"
    target.parent.mkdir(exist_ok=True)
    target.write_text("A - Hello there.\n", encoding="utf-8")
    window.input_path.setText(str(target))

    class _Popup(app_gui.QMessageBox):
        information = staticmethod(lambda *a, **k: None)
        critical = staticmethod(lambda *a, **k: None)

        def exec(self):  # noqa: D102 - test double: user clicks the fix button
            _Popup.fix_button = self.buttons()[0]
            self.clickedButton = lambda: _Popup.fix_button  # type: ignore[method-assign]
            return 0

    real_qdialog = app_gui.QDialog

    class _PreviewDialog(real_qdialog):
        def exec(self):  # noqa: D102 - test double: user accepts the preview
            return real_qdialog.DialogCode.Accepted

    monkeypatch.setattr(app_gui, "QMessageBox", _Popup)
    monkeypatch.setattr(app_gui, "QDialog", _PreviewDialog)

    assert window._run_ingest_transformer_check() is True
    panel_text = window.error_panel.toPlainText()
    assert "corrected 1 problem(s) in messy4.txt" in panel_text
    assert "re-analyzing the corrected file now" in panel_text
    assert "backup:" in panel_text


# ---------------------------------------------------------------------------
# Batch folder fix flow
# ---------------------------------------------------------------------------


def test_batch_fix_scans_previews_and_applies(qt_app, monkeypatch, tmp_path) -> None:
    """Folder picker -> combined preview -> apply writes every file + backups,
    and the status panel reports per-file results."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui
    from the_oracle.ingest_transformer import FolderFix

    folder = tmp_path / "inputs"
    folder.mkdir()
    (folder / "messy.txt").write_text("A - Hello there.\n", encoding="utf-8")
    (folder / "brackets.md").write_text("[A]: Greetings.\n", encoding="utf-8")
    (folder / "clean.txt").write_text("A: All good here.\n", encoding="utf-8")
    sub = folder / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("B - Deep.\n", encoding="utf-8")

    monkeypatch.setattr(
        app_gui.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder))
    )

    real_qdialog = app_gui.QDialog

    class _BatchDialog(real_qdialog):
        def exec(self):  # noqa: D102 - test double: user accepts the batch preview
            from PySide6.QtWidgets import QPlainTextEdit, QTreeWidget

            trees = self.findChildren(QTreeWidget)
            assert trees, "batch preview has no folder tree"
            tree = trees[0]
            labels = [tree.topLevelItem(0).child(i).text(0) for i in range(tree.topLevelItem(0).childCount())]
            # Tree lists every fixable file by path relative to the folder,
            # including the subfolder file (recursive scan).
            assert sorted(labels) == ["brackets.md", "messy.txt", "sub/nested.txt"]
            views = self.findChildren(QPlainTextEdit)
            assert views, "batch preview has no diff view"
            diff = views[0].toPlainText()
            # The preselected first file's rule-labeled diff is shown.
            assert "+ [bracketed label] A: Greetings." in diff
            return real_qdialog.DialogCode.Accepted

    information_calls: list[str] = []
    monkeypatch.setattr(app_gui, "QDialog", _BatchDialog)
    monkeypatch.setattr(
        app_gui.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: information_calls.append(str(k)) or information_calls.append(str(a))),
    )

    window._batch_fix_input_folder()

    # Files were rewritten, originals backed up.
    assert (folder / "messy.txt").read_text(encoding="utf-8") == "A: Hello there.\n"
    assert (folder / "brackets.md").read_text(encoding="utf-8") == "A: Greetings.\n"
    assert (folder / "clean.txt").read_text(encoding="utf-8") == "A: All good here.\n"
    assert (sub / "nested.txt").read_text(encoding="utf-8") == "B: Deep.\n"
    backups = list(folder.rglob("*.bak-*"))
    assert len(backups) == 3
    # Status panel reports the batch and each file (with subfolder paths).
    panel = window.error_panel.toPlainText()
    assert "Batch fix: corrected 3 formatting problem(s) across 3 file(s)" in panel
    assert "messy.txt: 1 fix(es)" in panel
    assert "brackets.md: 1 fix(es)" in panel
    assert "nested.txt: 1 fix(es)" in panel


def test_batch_fix_cancel_changes_nothing(qt_app, monkeypatch, tmp_path) -> None:
    """Rejecting the combined preview leaves every file untouched."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui

    folder = tmp_path / "inputs"
    folder.mkdir()
    (folder / "messy.txt").write_text("A - Hello there.\n", encoding="utf-8")

    monkeypatch.setattr(
        app_gui.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder))
    )

    real_qdialog = app_gui.QDialog

    class _BatchDialog(real_qdialog):
        def exec(self):  # noqa: D102 - test double: user cancels
            return real_qdialog.DialogCode.Rejected

    monkeypatch.setattr(app_gui, "QDialog", _BatchDialog)

    window._batch_fix_input_folder()

    assert (folder / "messy.txt").read_text(encoding="utf-8") == "A - Hello there.\n"
    assert not list(folder.glob("*.bak-*"))
    assert "Batch fix cancelled: no files were changed." in window.error_panel.toPlainText()


def test_batch_fix_clean_folder_shows_no_preview(qt_app, monkeypatch, tmp_path) -> None:
    """A folder with nothing to fix gets an informational popup, no dialog."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui

    folder = tmp_path / "inputs"
    folder.mkdir()
    (folder / "clean.txt").write_text("A: All good here.\n", encoding="utf-8")

    monkeypatch.setattr(
        app_gui.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder))
    )

    messages: list[str] = []

    class _Popup(app_gui.QMessageBox):
        @staticmethod
        def information(*args, **kwargs) -> None:  # noqa: D102 - test double
            messages.append(str(args[2]) if len(args) >= 3 else str(kwargs))

        @staticmethod
        def critical(*args, **kwargs) -> None:  # noqa: D102 - test double
            messages.append("CRITICAL: " + str(args))

    monkeypatch.setattr(app_gui, "QMessageBox", _Popup)

    window._batch_fix_input_folder()

    assert any("No formatting problems found" in m for m in messages), messages
    # No fix dialogs were opened and nothing was written.
    assert (folder / "clean.txt").read_text(encoding="utf-8") == "A: All good here.\n"


# ---------------------------------------------------------------------------
# "Remember my choice" / trusted-file auto-accept
# ---------------------------------------------------------------------------


def test_trusted_file_is_autocorrected_without_popup(qt_app, monkeypatch, tmp_path) -> None:
    """A remembered file is fixed silently: no popup, no preview dialog."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui

    target = tmp_path / "input" / "trusted.txt"
    target.parent.mkdir(exist_ok=True)
    target.write_text("A - Hello there.\n", encoding="utf-8")
    window.input_path.setText(str(target))
    window._remember_trusted_input_file(str(target))

    opened: list[str] = []

    class _NoPopup(app_gui.QMessageBox):
        def exec(self):  # noqa: D102 - test double: must never open
            opened.append("popup")
            return 0

    class _NoDialog(app_gui.QDialog):
        def exec(self):  # noqa: D102 - test double: must never open
            opened.append("dialog")
            return 0

    monkeypatch.setattr(app_gui, "QMessageBox", _NoPopup)
    monkeypatch.setattr(app_gui, "QDialog", _NoDialog)

    assert window._run_ingest_transformer_check() is True
    assert opened == [], "trusted file must not open popup or preview"
    assert target.read_text(encoding="utf-8") == "A: Hello there.\n"
    assert "auto-corrected 1 problem(s)" in window.error_panel.toPlainText()
    backups = list(target.parent.glob("*.bak-*"))
    assert len(backups) == 1


def test_remember_checkbox_persists_trust_on_accept(qt_app, monkeypatch, tmp_path) -> None:
    """Ticking the checkbox and accepting adds the file to the trusted list."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui
    from PySide6.QtWidgets import QCheckBox

    target = tmp_path / "messy.txt"
    target.write_text("A - Hello there.\n", encoding="utf-8")
    original, fixed, fix_count, _issues, line_fixes = preview_fixed_text(target)

    class _PreviewDialog(app_gui.QDialog):
        def exec(self):  # noqa: D102 - user accepts with the checkbox ticked
            boxes = self.findChildren(QCheckBox)
            assert len(boxes) == 1, "preview dialog must offer the remember checkbox"
            assert "Remember this choice" in boxes[0].text()
            boxes[0].setChecked(True)
            return app_gui.QDialog.DialogCode.Accepted

    monkeypatch.setattr(app_gui, "QDialog", _PreviewDialog)
    assert window._show_fix_preview_dialog(original, fixed, fix_count, line_fixes, input_file=str(target)) is True
    assert window._input_file_is_trusted(str(target))


def test_unchecked_remember_box_does_not_trust(qt_app, monkeypatch, tmp_path) -> None:
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui
    from PySide6.QtWidgets import QCheckBox

    target = tmp_path / "messy.txt"
    target.write_text("A - Hello there.\n", encoding="utf-8")
    original, fixed, fix_count, _issues, line_fixes = preview_fixed_text(target)

    class _PreviewDialog(app_gui.QDialog):
        def exec(self):  # noqa: D102 - user accepts WITHOUT the checkbox
            boxes = self.findChildren(QCheckBox)
            assert boxes and not boxes[0].isChecked()
            return app_gui.QDialog.DialogCode.Accepted

    monkeypatch.setattr(app_gui, "QDialog", _PreviewDialog)
    assert window._show_fix_preview_dialog(original, fixed, fix_count, line_fixes, input_file=str(target)) is True
    assert not window._input_file_is_trusted(str(target))


def test_untrusted_file_still_prompts(qt_app, monkeypatch, tmp_path) -> None:
    """Trust is per-file: other files keep the full popup + preview flow."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui

    trusted_file = tmp_path / "trusted.txt"
    trusted_file.write_text("A - Hello there.\n", encoding="utf-8")
    window._remember_trusted_input_file(str(trusted_file))

    other = tmp_path / "other.txt"
    other.write_text("B - Hi back.\n", encoding="utf-8")
    window.input_path.setText(str(other))

    popup_opened: list[bool] = []

    class _Popup(app_gui.QMessageBox):
        information = staticmethod(lambda *a, **k: None)
        critical = staticmethod(lambda *a, **k: None)

        def exec(self):  # noqa: D102 - user clicks the fix button (Cancel role)
            popup_opened.append(True)
            cancel_button = next(
                button
                for button in self.buttons()
                if button.text().lower().startswith("cancel")
            )
            self.clickedButton = lambda: cancel_button  # type: ignore[method-assign]
            return 0

    monkeypatch.setattr(app_gui, "QMessageBox", _Popup)

    assert window._run_ingest_transformer_check() is False
    assert popup_opened == [True]
    # The untrusted file was not touched.
    assert other.read_text(encoding="utf-8") == "B - Hi back.\n"


def test_trust_persists_across_windows(qt_app, monkeypatch, tmp_path) -> None:
    """Remembered trust round-trips through the app-settings file."""
    window, paths = _build_window(monkeypatch, tmp_path)
    window._app_settings_ready = True  # persistence is enabled after GUI shown
    target = tmp_path / "longterm.txt"
    target.write_text("A - Hello there.\n", encoding="utf-8")
    window._remember_trusted_input_file(str(target))

    # A fresh window loads persisted settings from disk.
    window2, _paths2 = _build_window(monkeypatch, tmp_path)
    assert window2._input_file_is_trusted(str(target))
    # ...and the file itself holds the resolved path.
    from the_oracle.gui_settings import load_app_settings

    reloaded = load_app_settings()
    assert str(target.resolve()) in reloaded.get("trusted_format_files", [])


def test_batch_tree_selection_switches_diff(qt_app, monkeypatch, tmp_path) -> None:
    """Selecting another file in the tree shows that file's labeled diff."""
    window, _paths = _build_window(monkeypatch, tmp_path)
    import the_oracle.app_gui as app_gui

    folder = tmp_path / "inputs"
    folder.mkdir()
    (folder / "dash.txt").write_text("A - Hello.\n", encoding="utf-8")
    (folder / "brackets.md").write_text("[A]: Greetings.\n", encoding="utf-8")

    monkeypatch.setattr(
        app_gui.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder))
    )

    real_qdialog = app_gui.QDialog
    seen: dict = {}

    class _BatchDialog(real_qdialog):
        def exec(self):  # noqa: D102 - select each tree entry, capture diffs
            from PySide6.QtWidgets import QPlainTextEdit, QTreeWidget

            tree = self.findChildren(QTreeWidget)[0]
            view = self.findChildren(QPlainTextEdit)[0]
            root = tree.topLevelItem(0)
            for i in range(root.childCount()):
                child = root.child(i)
                tree.setCurrentItem(child)
                seen[child.text(0)] = view.toPlainText()
            return real_qdialog.DialogCode.Accepted

    monkeypatch.setattr(app_gui, "QDialog", _BatchDialog)
    assert window._show_batch_fix_preview_dialog(str(folder), *_fixes_and_warnings(folder)) is True
    assert "dash/pipe separator" in seen["dash.txt"]
    assert "bracketed label" in seen["brackets.md"]


def _fixes_and_warnings(folder):
    from the_oracle.ingest_transformer import preview_folder_fixes

    return preview_folder_fixes(folder)
