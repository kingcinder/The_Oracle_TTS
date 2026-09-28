"""Recording Studio extraction: the record-a-Seashell dialog and its worker.

The eighth MainWindow extraction slice (2026-09-28) moved this cluster out of
``app_gui`` verbatim: ``RecordStudioWorker`` (the sounddevice capture thread)
and ``RecordingStudioDialog`` (the teleprompter/mic-picker/level-meter window
that records mono WAV "Seashell" reference voices into the voice folder).

Ownership and patch-surface contract (enforced by
``tests/test_gui_recording_owner.py``):

- ``RecordStudioWorker`` is a MOVED owner (listed in
  ``scripts/patch_surface_manifest.json``): MainWindow constructs it only via
  the dialog, so an app_gui-level patch of the name is a silent no-op and the
  patch-surface net repoints it here.
- ``RecordingStudioDialog`` is deliberately NOT listed, same as the
  ``RenderWorker`` precedent: MainWindow constructs it from its own globals,
  so app_gui-level class patches remain live and legitimate.
- The dialog's audition player resolves ``QMediaPlayer``/``QAudioOutput`` as
  bare names from THIS module's globals; tests faking those classes patch
  ``the_oracle.gui_recording``, not ``app_gui``.
- The recording back-end seam is ``the_oracle.audio.recorder``'s module
  attributes (``_sd``, ``list_input_devices``, ``samplerates_for_device``,
  ``next_seashell_name``, ``save_recording_wav``) — the seam tests already
  patch, preserved identically by importing the module object, not its names.
- Shared helpers (``recording_target_path``, ``sanitize_recording_filename``)
  come from ``gui_utils`` — the same below-the-gui-layer home the ingest
  slice established; the auto-incrementing name helper stays on the recorder
  seam (``recorder.next_seashell_name``).

``app_gui`` re-imports both classes so its names keep resolving to identical
objects. gui_recording imports app_gui in no spelling — the dependency
direction stays one-way (``app_gui -> gui_* -> pipeline/models``).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QTimer, QThread, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from the_oracle.audio import recorder
from the_oracle.gui_utils import recording_target_path, sanitize_recording_filename

class RecordStudioWorker(QThread):
    """Records microphone input until stopped, emitting live level updates.

    Runs recorder.record_until_stop (sounddevice) on a worker thread so the
    Recording Studio UI stays responsive while the take is in progress.
    """

    level = Signal(float)
    captured = Signal(object)  # numpy float32 mono array
    failed = Signal(str)

    def __init__(self, device_index: int, samplerate: int, channels: int = 1) -> None:
        super().__init__()
        self._device_index = int(device_index)
        self._samplerate = int(samplerate)
        self._channels = max(1, int(channels))
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            audio = recorder.record_until_stop(
                self._device_index,
                self._samplerate,
                channels=self._channels,
                stop_event=self._stop_event,
                on_level=lambda level: self.level.emit(level),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.captured.emit(audio)


class RecordingStudioDialog(QDialog):
    """Separate window for recording new reference "Seashell" voices.

    Reads a script from a teleprompter area (fed by the repo's Input/ folder),
    picks the microphone/sample-rate, and records mono WAV into the voice
    folder under an auto-incrementing ``Seashell_No_<n>.wav`` name that never
    overwrites an existing file. The parent window refreshes its voice pickers
    on this dialog's ``finished`` signal - success or not.
    """

    def __init__(
        self,
        repo_root: str | Path,
        voice_dir: str | Path,
        input_dir: str | Path,
        parent: QWidget | None = None,
        on_assign: Callable[[str, Path], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Recording Studio - record a new Seashell voice")
        self.resize(760, 640)
        self.repo_root = Path(repo_root)
        self.voice_dir = Path(voice_dir)
        self.input_dir = Path(input_dir)
        self.on_assign = on_assign
        self._worker: RecordStudioWorker | None = None
        self._stop_requested = False
        self._last_saved_path: Path | None = None
        self._name_edited = False
        self._player: QMediaPlayer | None = None

        self.script_combo = QComboBox()
        self.script_combo.currentIndexChanged.connect(self._load_script)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(8, 48)
        self.font_spin.setValue(16)
        self.font_spin.valueChanged.connect(self._update_prompt_font)
        self.prompt_area = QPlainTextEdit()
        self.prompt_area.setPlaceholderText(
            "Pick a script from the teleprompter file list (the repo's Input/ "
            "folder) or paste lines here to read while recording."
        )
        self._update_prompt_font()

        # --- capture row ---
        self.mic_combo = QComboBox()
        self.mic_combo.currentIndexChanged.connect(self._on_mic_selected)
        self.rate_combo = QComboBox()
        self.rate_combo.setEnabled(False)
        self.outdir_combo = QComboBox()
        self.outdir_combo.setEditable(True)
        self.outdir_combo.addItem(str(self.voice_dir), str(self.voice_dir))
        self.outdir_combo.currentTextChanged.connect(self._on_output_folder_changed)
        self.outdir_browse = QPushButton("Browse...")
        self.outdir_browse.clicked.connect(self._browse_outdir)
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(lambda: setattr(self, "_name_edited", True))
        self.generic_name_warning_enabled = True
        self.remember_input_default = True
        self.remember_output_default = True
        self.name_edit.textChanged.connect(self._refresh_target)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setFormat("input level: %p%")
        self.elapsed_label = QLabel("0:00")
        self.record_button = QPushButton("Record")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self._toggle_record)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        # Take actions: audition the saved take and point a speaker at it.
        self.audition_check = QCheckBox("Auto-play new take")
        self.audition_check.setChecked(True)
        self.audition_check.setToolTip(
            "Play each finished take back immediately so you can judge the voice "
            "before recording the next one."
        )
        self.play_button = QPushButton("Listen again")
        self.play_button.setEnabled(False)
        self.play_button.setToolTip("Play the last saved take again.")
        self.play_button.clicked.connect(self._play_take)
        self.assign_speaker_combo = QComboBox()
        self.assign_speaker_combo.setToolTip(
            "Which speaker's voice to point at the Seashell you just recorded "
            "(the voice pickers refresh automatically)."
        )
        self.assign_speaker_button = QPushButton("Use for Speaker")
        self.assign_speaker_button.setEnabled(False)
        self.assign_speaker_button.setToolTip(
            "Point the selected speaker's voice at the Seashell you just recorded."
        )
        self.assign_speaker_button.clicked.connect(self._assign_to_selected_speaker)

        self._elapsed_seconds = 0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        self._build_layout()
        self._populate_scripts()
        self._refresh_devices()
        self._refresh_target()

    def _on_output_folder_changed(self, folder: str) -> None:
        """Keep the suggested name aligned with a newly chosen folder."""
        if not self._name_edited:
            self.name_edit.blockSignals(True)
            self.name_edit.setText(f"{recorder.next_seashell_name(folder or self.voice_dir)}.wav")
            self.name_edit.blockSignals(False)
        self._refresh_target()

    def apply_setup_preferences(self, payload: dict) -> None:
        """Apply Recording Studio wizard settings to this live dialog."""
        mic = payload.get("microphone_index")
        if mic is not None:
            index = self.mic_combo.findData(mic)
            if index >= 0:
                self.mic_combo.setCurrentIndex(index)
        rate = payload.get("samplerate")
        if rate is not None:
            index = self.rate_combo.findData(rate)
            if index >= 0:
                self.rate_combo.setCurrentIndex(index)
        script = str(payload.get("input_file") or "")
        if script:
            index = self.script_combo.findData(script)
            if index >= 0:
                self.script_combo.setCurrentIndex(index)
        folder = str(payload.get("output_dir") or "").strip()
        if folder:
            self.outdir_combo.setCurrentText(folder)
        name = str(payload.get("output_filename") or "").strip()
        if name:
            self.name_edit.setText(name)
            self._name_edited = not name.lower().startswith("seashell_no_")
        self.generic_name_warning_enabled = bool(payload.get("generic_name_warning", True))
        self.remember_input_default = bool(payload.get("remember_input_default", True))
        self.remember_output_default = bool(payload.get("remember_output_default", True))
        self._refresh_target()

    # -- layout -------------------------------------------------------------

    def _build_layout(self) -> None:
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Teleprompter script:"))
        top_row.addWidget(self.script_combo, 1)
        top_row.addWidget(QLabel("Font:"))
        top_row.addWidget(self.font_spin)

        prompt_box = QGroupBox("Reading prompt")
        prompt_layout = QVBoxLayout(prompt_box)
        prompt_layout.addWidget(self.prompt_area)

        capture_box = QGroupBox("Capture")
        capture_layout = QVBoxLayout(capture_box)
        form = QFormLayout()
        mic_row = QHBoxLayout()
        mic_row.addWidget(self.mic_combo, 1)
        form.addRow("Microphone", mic_row)
        form.addRow("Sample rate", self.rate_combo)
        outdir_row = QHBoxLayout()
        outdir_row.addWidget(self.outdir_combo, 1)
        outdir_row.addWidget(self.outdir_browse)
        form.addRow("Save to", outdir_row)
        form.addRow("File name", self.name_edit)

        control_row = QHBoxLayout()
        control_row.addWidget(self.record_button)
        control_row.addWidget(self.elapsed_label)
        control_row.addWidget(self.level_bar, 1)
        capture_layout.addLayout(form)
        capture_layout.addLayout(control_row)

        take_row = QHBoxLayout()
        take_row.addWidget(self.audition_check)
        take_row.addWidget(self.play_button)
        take_row.addWidget(self.assign_speaker_combo)
        take_row.addWidget(self.assign_speaker_button)
        take_row.addStretch(1)
        capture_layout.addLayout(take_row)
        capture_layout.addWidget(self.status)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addWidget(prompt_box, 1)
        layout.addWidget(capture_box, 0)

    def _populate_scripts(self) -> None:
        self.script_combo.blockSignals(True)
        self.script_combo.clear()
        self.script_combo.addItem("<no script - type your own lines>", "")
        if self.input_dir.is_dir():
            for path in sorted(self.input_dir.glob("*")):
                if path.suffix.lower() in (".txt", ".md") and path.is_file():
                    self.script_combo.addItem(path.name, str(path))
        # The bundled alphabet/phonetic coverage script is the safest first
        # take: it exercises a broad range of sounds while remaining easy to
        # follow from the teleprompter. A remembered script is applied later
        # by MainWindow, so this only governs a genuinely fresh studio.
        default_script = self.input_dir / "What is, reality.txt"
        default_index = self.script_combo.findData(str(default_script))
        if default_index >= 0:
            self.script_combo.setCurrentIndex(default_index)
        self.script_combo.blockSignals(False)
        if default_index >= 0:
            self._load_script(default_index)

    def _load_script(self, _index: int) -> None:
        path = self.script_combo.currentData()
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            self.status.setText(f"Could not read script: {exc}")
            return
        self.prompt_area.setPlainText(text)
        self.status.setText(f"Loaded {Path(path).name} into the prompt.")

    def _update_prompt_font(self) -> None:
        font = self.prompt_area.font()
        font.setPointSize(self.font_spin.value())
        self.prompt_area.setFont(font)

    # -- capture state ------------------------------------------------------

    def _refresh_devices(self) -> None:
        devices = recorder.list_input_devices()
        self.mic_combo.blockSignals(True)
        self.mic_combo.clear()
        self._devices = devices
        if devices:
            for device in devices:
                self.mic_combo.addItem(device.name, device.index)
            self.mic_combo.setCurrentIndex(0)
            self.mic_combo.blockSignals(False)
            self._on_mic_selected(0)
            self.record_button.setEnabled(True)
        else:
            self.mic_combo.addItem(
                "(no microphone found)" if recorder.have_capture_backend()
                else "(recording backend not installed)",
                None,
            )
            self.mic_combo.blockSignals(False)
            self.rate_combo.setEnabled(False)
            self.record_button.setEnabled(False)

    def _on_mic_selected(self, _index: int) -> None:
        index = self.mic_combo.currentData()
        self.rate_combo.clear()
        self.rate_combo.setEnabled(False)
        if index is None:
            return
        try:
            rates = recorder.samplerates_for_device(int(index))
        except recorder.RecorderUnavailableError:
            self.rate_combo.addItem("(backend unavailable)", None)
            return
        for rate in rates:
            self.rate_combo.addItem(f"{rate} Hz", rate)
        self.rate_combo.setEnabled(True)

    def _browse_outdir(self) -> None:
        start = self.voice_dir if self.voice_dir.is_dir() else self.repo_root
        chosen = QFileDialog.getExistingDirectory(self, "Save recordings to", str(start))
        if chosen:
            self.outdir_combo.setCurrentText(chosen)

    def _refresh_target(self) -> None:
        folder = self.outdir_combo.currentText().strip() or str(self.voice_dir)
        if not self.name_edit.text().strip():
            # Empty name: fall back to the next free auto-name.
            self._name_edited = False
            self.name_edit.blockSignals(True)
            try:
                self.name_edit.setText(f"{recorder.next_seashell_name(folder)}.wav")
            finally:
                self.name_edit.blockSignals(False)
        # Sanitize the typed name so path separators / '..' can never escape
        # the chosen output folder (the name is confined to its final path
        # segment and forced to a .wav extension).
        safe = sanitize_recording_filename(self.name_edit.text())
        if safe != self.name_edit.text():
            self.name_edit.blockSignals(True)
            try:
                self.name_edit.setText(safe)
            finally:
                self.name_edit.blockSignals(False)
        self._target_path = Path(folder) / safe
        self._warn_if_overwrite()

    def _warn_if_overwrite(self) -> None:
        target = getattr(self, "_target_path", None)
        if target is not None and target.exists():
            self.status.setText(
                f"{target.name} already exists - recording would overwrite it. "
                "Consider renaming (or the next Seashell_No_x will be used)."
            )

    def _default_filename(self) -> str:
        return f"{recorder.next_seashell_name(self._out_folder())}.wav"

    def _out_folder(self) -> Path:
        folder = self.outdir_combo.currentText().strip()
        return Path(folder) if folder else self.voice_dir

    def _toggle_record(self) -> None:
        # isRunning() guards the brief window where the terminal captured/
        # failed slot has run but the finished cleanup has not yet detached
        # the worker — a finished thread must never be "stopped" again.
        if self._worker is None or not self._worker.isRunning():
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        if self.mic_combo.currentData() is None:
            self.status.setText("Pick a microphone first.")
            return
        if self.rate_combo.currentData() is None:
            self.status.setText("Pick a supported sample rate first.")
            return
        # A generated boilerplate filename gets one actionable caution. It is
        # intentionally shown at record time, when the user can still close it
        # and replace the name; the checkbox preference is persisted by MainWindow.
        if self.generic_name_warning_enabled and not self._name_edited and self.name_edit.text().strip().lower().startswith("seashell_no_"):
            warning = QMessageBox(self)
            warning.setIcon(QMessageBox.Icon.Warning)
            warning.setWindowTitle("Generic Seashell filename")
            warning.setText("This recording will use a generic default filename.")
            warning.setInformativeText("Close this warning and choose a descriptive filename now if you want this take to be easy to identify later.")
            disable = QCheckBox("Click here to disable this warning; re-enable it in the Settings menu")
            warning.setCheckBox(disable)
            warning.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            warning.setDefaultButton(QMessageBox.StandardButton.Cancel)
            result = warning.exec()
            if disable.isChecked():
                self.generic_name_warning_enabled = False
                self.status.setText("Generic filename warning disabled; change it in Settings if needed.")
            if result == QMessageBox.StandardButton.Cancel:
                return
        # Never overwrite: if the typed name collides and the user declines the
        # overwrite prompt, fall back to the next free Seashell_No_x name.
        # The name is sanitized (no separators / '..') so the target can never
        # escape the chosen output folder.
        target = recording_target_path(self._out_folder(), self.name_edit.text().strip())
        if target.exists():
            answer = QMessageBox.question(
                self,
                "File exists",
                f"{target.name} already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                fresh = self._default_filename()
                self.name_edit.setText(fresh)
                self._name_edited = True
                self._refresh_target()
        self._target_path = recording_target_path(self._out_folder(), self.name_edit.text().strip())
        self._stop_requested = False
        self._elapsed_seconds = 0
        self._elapsed_label_text()
        self.level_bar.setValue(0)
        self.record_button.setText("Stop")
        self.record_button.setEnabled(True)
        self.mic_combo.setEnabled(False)
        self.rate_combo.setEnabled(False)
        self._elapsed_timer.start()
        self._stop_playback()
        self._set_take_actions_enabled(False)
        self.status.setText("Recording... click Stop when done.")
        self._worker = RecordStudioWorker(
            device_index=int(self.mic_combo.currentData()),
            samplerate=int(self.rate_combo.currentData()),
            channels=1,
        )
        self._worker.level.connect(self._on_level)
        self._worker.captured.connect(self._on_captured)
        self._worker.failed.connect(self._on_recording_failed)
        # Delete the thread object only after run() has fully returned. The
        # captured/failed slots fire from run()'s final lines while the thread
        # is still alive — deleteLater there races the thread's own exit and
        # is the classic "QThread destroyed while running" use-after-free.
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _stop_recording(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
        self._stop_requested = True
        self.record_button.setEnabled(False)
        self.record_button.setText("Finishing...")

    def _on_level(self, level: float) -> None:
        # RMS in [-0..~0.5 for speech]; scale so typical speech sits mid-meter.
        self.level_bar.setValue(int(min(100.0, level * 500.0)))

    def _on_captured(self, audio) -> None:
        self._elapsed_timer.stop()
        self.record_button.setText("Record")
        self.record_button.setEnabled(True)
        self.mic_combo.setEnabled(True)
        self.rate_combo.setEnabled(True)
        # The worker detaches itself via finished -> _on_worker_finished;
        # never deleteLater from here (thread may still be exiting).
        if audio is None or len(audio) == 0:
            self.status.setText("No audio was captured - check the microphone and try again.")
            return
        try:
            saved = recorder.save_recording_wav(self._target_path, audio, int(self.rate_combo.currentData()))
        except Exception as exc:
            self.status.setText(f"Could not save the recording: {exc}")
            return
        self._last_saved_path = saved
        seconds = len(audio) / float(self.rate_combo.currentData())
        self.status.setText(f"Saved {saved.name} ({seconds:.1f}s) in {saved.parent}.")
        # Next take auto-names one higher so nothing is ever overwritten.
        self._name_edited = False
        self.name_edit.setText(self._default_filename())
        self._refresh_target()
        # Offer audition + quick-assign now that a take exists on disk.
        self._refresh_take_actions()
        if self.audition_check.isChecked():
            self._play_take()

    def _on_recording_failed(self, message: str) -> None:
        self._elapsed_timer.stop()
        self.record_button.setText("Record")
        self.record_button.setEnabled(True)
        self.mic_combo.setEnabled(True)
        self.rate_combo.setEnabled(True)
        # Worker detaches itself via finished (see _on_worker_finished).
        self.status.setText(f"Recording failed: {message}")
        if self._last_saved_path is not None:
            self._refresh_take_actions()

    def _on_worker_finished(self) -> None:
        # Run on the GUI thread only after run() has returned, so detaching
        # and deleting the QThread object here is safe.
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

    def _tick_elapsed(self) -> None:
        self._elapsed_seconds += 1
        self._elapsed_label_text()

    def _elapsed_label_text(self) -> None:
        minutes, seconds = divmod(self._elapsed_seconds, 60)
        self.elapsed_label.setText(f"{minutes}:{seconds:02d}")

    # -- audition + assign --------------------------------------------------

    def set_assign_speakers(self, items: list[tuple[str, str]]) -> None:
        """Populate the 'Use for Speaker' picker from the current cast.

        ``items`` is ``[(key, name), ...]``; the combo shows "A (Alice)" style
        labels and carries the speaker key as item data.
        """
        self.assign_speaker_combo.blockSignals(True)
        try:
            self.assign_speaker_combo.clear()
            for key, name in items:
                label = f"{key} ({name})" if name else key
                self.assign_speaker_combo.addItem(label, key)
        finally:
            self.assign_speaker_combo.blockSignals(False)

    def _assign_to_selected_speaker(self) -> None:
        key = self.assign_speaker_combo.currentData()
        if key:
            self._use_for_speaker(str(key))

    def _set_take_actions_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled and self._last_saved_path is not None)
        self.assign_speaker_combo.setEnabled(enabled and self._last_saved_path is not None)
        self.assign_speaker_button.setEnabled(enabled and self._last_saved_path is not None)

    def _refresh_take_actions(self) -> None:
        self._set_take_actions_enabled(True)

    def _play_take(self) -> None:
        if self._last_saved_path is None or not self._last_saved_path.exists():
            return
        self._stop_playback()
        # One persistent player per dialog, created lazily on first audition.
        # Reusing a single QMediaPlayer (as MainWindow does for previews) is
        # safe; creating and deleteLater'ing a fresh player per take races the
        # FFmpeg backend's internal threads (use-after-free).
        if self._player is None:
            audio_output = QAudioOutput(self)
            player = QMediaPlayer(self)
            player.setAudioOutput(audio_output)
            player.mediaStatusChanged.connect(self._on_playback_status)
            self._player = player
            self._player_output = audio_output
        self._player.setSource(QUrl.fromLocalFile(str(self._last_saved_path)))
        self._player.play()
        self.status.setText(f"Auditioning {self._last_saved_path.name}...")

    def _on_playback_status(self, status) -> None:  # type: ignore[no-untyped-def]
        # NEVER call stop()/deleteLater() on the player from inside its own
        # mediaStatusChanged emission — that is a QtMultimedia use-after-free
        # (backend threads still mid-emission). Defer any stop out of the
        # handler with a zero-timer so the backend finishes delivering first.
        end_state = getattr(status, "EndOfMedia", None)
        if end_state is not None and status == end_state:
            QTimer.singleShot(0, self._stop_playback)

    def _stop_playback(self) -> None:
        player = self._player
        if player is not None:
            try:
                player.stop()
            except Exception:
                pass

    def _use_for_speaker(self, speaker: str) -> None:
        if self._last_saved_path is None:
            return
        if self.on_assign is not None:
            try:
                self.on_assign(speaker, self._last_saved_path)
            except Exception as exc:
                self.status.setText(f"Could not assign to Speaker {speaker}: {exc}")
                return
        self.status.setText(f"Assigned {self._last_saved_path.name} as the voice for Speaker {speaker}.")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt casing)
        # A take in progress is stopped and its thread joined before the window
        # closes; the take itself is discarded, but the parent still refreshes
        # pickers via the finished signal (which always fires on close).
        # Destroying a live QThread is a hard Qt abort, so wait (bounded) and
        # refuse to close if the thread cannot stop.
        worker = self._worker
        if worker is not None:
            self._stop_recording()
            try:
                joined = bool(worker.wait(2000))
            except Exception:
                joined = True  # test double without a real thread
            if not joined:
                event.ignore()
                self.status.setText("Still finishing the take - click Stop first.")
                return
            worker.deleteLater()
            self._worker = None
        self._stop_playback()
        super().closeEvent(event)


