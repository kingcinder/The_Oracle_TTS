"""The cast-management cluster (2026-10-05, slice 9).

:class:`SpeakerGroup` — one speaker's full voice panel (reference picker,
hybrid second voice, dominance weight, combine mode, and the perceptual
sliders) — and :class:`CastManagementDialog`, the modal window that edits the
whole cast as a snapshot handed back on close, plus their shared helpers
(``_speaker_settings_from_group``, ``_apply_speaker_settings_to_group``,
``_CastRow``). Extracted from ``app_gui`` in the extraction campaign; the main
window re-imports the names so its construction sites and existing imports
resolve to identical objects.

Dependency direction: this module never imports ``app_gui`` — the main
window's collaborators cross the seam as values (the ``main_window``
parameter), per the gui_recording precedent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from the_oracle.gui_sections import QHSectionGroup
from the_oracle.gui_settings import load_recent_reference_paths
from the_oracle.gui_utils import MAX_CAST_SPEAKERS, CastModel, next_speaker_key
from the_oracle.gui_widgets import PerceptualSlider
from the_oracle.models.project import VoiceSettings
from the_oracle.models.settings import SpeakerSettings
from the_oracle.voice_catalog import (
    VoiceChoice,
    blend_voice_choices,
    default_voice_choices,
)


class SpeakerGroup(QHSectionGroup):
    def __init__(
        self,
        speaker: str,
        custom_reference_dir: Path,
        on_save_blend: Callable[["SpeakerGroup"], None] | None = None,
    ) -> None:
        super().__init__(f"Speaker {speaker}")
        self.custom_reference_dir = custom_reference_dir
        self.on_save_blend = on_save_blend
        self.reference_path = QLineEdit()
        self.reference_picker = QComboBox()
        # activated alone (not currentIndexChanged): a user pick fires BOTH
        # signals, and connecting both made _pick_audio() run twice — two
        # stacked modal file dialogs. activated also covers the
        # already-current custom option that currentIndexChanged would miss.
        self.reference_picker.activated.connect(self._handle_reference_selection)
        self._available_reference_paths: set[str] = set()

        # Voice blending: optionally derive this speaker's conditioning
        # reference from TWO clips (base + blend target), with a preference
        # weight choosing which voice's qualities dominate and a mode
        # choosing how they are combined (see audio/blend.py).
        self.blend_picker = QComboBox()
        self.blend_picker.setToolTip(
            "Combine this speaker's voice with a second voice: the render "
            "conditions Chatterbox on a deterministic blend of the two clips. "
            "None keeps the single-voice behavior."
        )
        self.blend_weight_spin = PerceptualSlider(
            minimum=0,
            maximum=100,
            value=50,
            suffix="%",
            int_mode=True,
            caption=(
                "0% = the blend target's voice dominates; 100% = this speaker's "
                "base voice dominates."
            ),
        )
        self.blend_weight_spin.setToolTip(
            "The preference weight for the two-voice hybrid (0-100%). The "
            "conditioning clip is derived from your base voice and the second "
            "voice: 100% keeps the base voice's qualities alone, 0% hands the "
            "voice entirely to the second clip, 50% meets in the middle. The "
            "derived clip is deterministic and content-hashed, so the same "
            "blend always renders the same voice on CPU and Vulkan."
        )
        self.blend_mode_combo = QComboBox()
        self.blend_mode_combo.addItem("Mix (voices blended together)", "mix")
        self.blend_mode_combo.addItem("Alternate (voices take turns)", "alternate")
        self.blend_mode_combo.addItem("Layer (base + texture under)", "layer")
        self.blend_mode_combo.setToolTip(
            "How the two clips are physically combined into the conditioning "
            "reference: Mix weight-averages the waveforms by the dominance "
            "weight, Alternate concatenates segments of each so the voices "
            "take turns, and Layer keeps the base at full level with the "
            "second voice mixed underneath."
        )
        self.save_blend_button = QPushButton("Save Blend As...")
        self.save_blend_button.setToolTip(
            "Save the current blend (base voice + blend target + presence + "
            "mode) as a named voice that appears under 'Saved Blends' in the "
            "voice picker, just like the bundled generic voices."
        )
        self.save_blend_button.clicked.connect(self._save_blend_clicked)

        self.language_combo = QComboBox()
        # Perceptual voice-modulation sliders: the scale and caption say what
        # the change SOUNDS like. Values still map onto the exact engine ranges
        # the spin boxes used, so profiles/projects/manifests are unchanged.
        # Perceptual voice-modulation sliders: the scale and caption say what
        # the change SOUNDS like; the tooltip states the exact engine knob,
        # domain, and mechanics. Values still map onto the exact engine ranges
        # the spin boxes used, so profiles/projects/manifests are unchanged.
        self.cfg_weight = PerceptualSlider(
            minimum=0.0, maximum=1.5, value=0.5,
            caption=(
                "How strictly this voice stays glued to its reference. High = "
                "steadier and closer to the original; low = freer, drifts more."
            ),
        )
        self.cfg_weight.setToolTip(
            "Engine: cfg_weight (0.0-1.5; forwarded to audio.cpp as "
            "--guidance-scale). Each spoken token is sampled twice - once "
            "conditioned on your reference audio, once unconditioned - and "
            "this weight sets how strongly the conditioned guess wins "
            "(classifier-free guidance). High values keep the delivery glued "
            "to the reference clip's cadence; low values let the model wander "
            "from it. Applied identically on CPU and Vulkan."
        )
        self.exaggeration = PerceptualSlider(
            minimum=0.0, maximum=1.5, value=0.5,
            caption=(
                "How hard emphasis lands. High = bigger swings in pitch and "
                "stress; low = flat, matter-of-fact delivery."
            ),
        )
        self.exaggeration.setToolTip(
            "Engine: exaggeration (0.0-1.5), Chatterbox's own expression knob. "
            "Scales how much pitch and stress swing around the neutral read: "
            "0.0 is flat and matter-of-fact, 0.5 is the model default, 1.5 is "
            "theatrical. Shapes emphasis WITHIN each line; the silence AFTER "
            "each line belongs to the Timing slider below."
        )
        self.temperature = PerceptualSlider(
            minimum=0.1, maximum=1.5, value=0.8,
            caption=(
                "How surprising the delivery can be. High = livelier, more "
                "varied takes; low = steady, predictable, closer to the "
                "reference."
            ),
        )
        self.temperature.setToolTip(
            "Engine: temperature (0.1-1.5). Controls randomness when the model "
            "picks each sound: low = the most predictable, steady "
            "pronunciation every time; high = livelier, more varied takes that "
            "can occasionally stumble. Held at one value per speaker for the "
            "whole render so the voice character does not drift between lines."
        )
        self.emotion_intensity = PerceptualSlider(
            minimum=0.0, maximum=2.0, value=1.0,
            caption=(
                "How strongly detected emotions color the performance "
                "(emphasis and pacing only - the voice's character stays "
                "locked)."
            ),
        )
        self.emotion_intensity.setToolTip(
            "Blend weight (0.0-2.0) for the emotion auto-detected on each line. "
            "At 1.0 a detected emotion fully applies its preset (that preset's "
            "emphasis and pause values replace the sliders above for that "
            "line); 0.5 halfway blends your slider value with the preset; 0.0 "
            "ignores detection; above 1.0 over-drives past the preset. Only "
            "emphasis and pacing move - the voice's timbre stays locked so one "
            "speaker never changes character mid-render."
        )
        self.naturalness = PerceptualSlider(
            minimum=0.0, maximum=1.0, value=0.0,
            caption=(
                "Loosens sampling for a more natural, less mechanical voice; "
                "held steady for the whole render so the character stays "
                "consistent."
            ),
        )
        self.naturalness.setToolTip(
            "Engine heuristic (0.0-1.0), applied once per speaker and constant "
            "for the whole render. Moves four sampling knobs together: per 0.1 "
            "of Human Drift, cfg_weight drops 0.12, temperature rises 0.18, "
            "repetition_penalty falls 0.25, min_p rises 0.03, and breaths "
            "lengthen by 12% - the combination reads as a relaxed human take "
            "instead of a mechanical one. 0.0 leaves every other slider "
            "exactly as set."
        )
        self.pause_spin = PerceptualSlider(
            minimum=0, maximum=2000, value=180, suffix=" ms", int_mode=True,
            caption=(
                "Silence after this speaker's turns, scaled by how each line "
                "ends - longer after !? and ellipses, shorter when a line "
                "trails on."
            ),
        )
        self.pause_spin.setToolTip(
            "Silence (0-2000 ms) inserted after this speaker's turn ends. The "
            "value is scaled by how the line ends: x1.0 after a period, x1.3 "
            "after !, x1.25 after ?, x1.6 after an ellipsis, and x0.7 when a "
            "line trails off on a comma or no punctuation. Between chunks of "
            "one long line the gap is ~35% of this value (never below 40 ms) "
            "instead of the full pause."
        )

        form = QFormLayout(self)
        self.form = form  # kept so Ctrl+hover help can register row labels

        def _section_divider(text: str) -> QLabel:
            label = QLabel(text)
            label.setStyleSheet("font-weight: 700; letter-spacing: 1px; padding-top: 8px; color: palette(mid);")
            return label

        form.addRow("Voice Reference", self.reference_picker)
        form.addRow("Language", self.language_combo)
        form.addRow(_section_divider("VOICE CHARACTER"))
        form.addRow("Identity Lock", self.cfg_weight)
        form.addRow("Emphasis Punch", self.exaggeration)
        form.addRow("Delivery Variety", self.temperature)
        form.addRow("Emotion Depth", self.emotion_intensity)
        form.addRow("Human Drift", self.naturalness)
        form.addRow(_section_divider("TIMING"))
        form.addRow("Breath After This Speaker", self.pause_spin)
        form.addRow(_section_divider("HYBRIDIZE VOICES"))
        blend_row = QHBoxLayout()
        blend_row.addWidget(self.blend_picker, 1)
        blend_row.addWidget(self.save_blend_button, 0)
        form.addRow("Hybrid Second Voice", blend_row)
        form.addRow("Voice Dominance", self.blend_weight_spin)
        form.addRow("Hybridize Mode", self.blend_mode_combo)

    def _pick_audio(self) -> None:
        current_reference = Path(self.reference_path.text()).expanduser()
        start_dir = current_reference.parent if current_reference.exists() else self.custom_reference_dir
        path, _ = QFileDialog.getOpenFileName(self, "Choose Reference Audio", str(start_dir), "Audio Files (*.wav *.flac *.mp3)")
        if path:
            self.reference_path.setText(path)

    def set_language_options(self, languages: dict[str, str], enabled: bool) -> None:
        selected = self.language_combo.currentData() or "en"
        self.language_combo.clear()
        for code, name in languages.items():
            self.language_combo.addItem(f"{code} - {name}", code)
        index = self.language_combo.findData(selected if enabled else "en")
        if index < 0:
            index = self.language_combo.findData("en")
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        self.language_combo.setEnabled(enabled)

    def set_reference_choices(
        self,
        defaults: list[VoiceChoice],
        recents: list[str],
        selected_path: str = "",
        blends: list[VoiceChoice] | None = None,
    ) -> None:
        current_path = selected_path or self.reference_path.text()
        self.reference_picker.blockSignals(True)
        self.reference_picker.clear()
        self._available_reference_paths = set()
        if defaults:
            header_index = self.reference_picker.count()
            self.reference_picker.addItem("Default Voices")
            header_item = self.reference_picker.model().item(header_index)
            if header_item is not None:
                header_item.setEnabled(False)
            for voice in defaults:
                self.reference_picker.addItem(f"  {voice.label}", voice.path)
                self._available_reference_paths.add(voice.path)
        if blends:
            header_index = self.reference_picker.count()
            self.reference_picker.addItem("Saved Blends")
            header_item = self.reference_picker.model().item(header_index)
            if header_item is not None:
                header_item.setEnabled(False)
            for voice in blends:
                self.reference_picker.addItem(f"  {voice.label}", voice.path)
                self._available_reference_paths.add(voice.path)
        if recents:
            header_index = self.reference_picker.count()
            self.reference_picker.addItem("Recent Custom Clips")
            header_item = self.reference_picker.model().item(header_index)
            if header_item is not None:
                header_item.setEnabled(False)
            for path in recents[:10]:
                resolved = str(Path(path).expanduser())
                self.reference_picker.addItem(f"  {Path(resolved).name}", resolved)
                self._available_reference_paths.add(resolved)
        self.reference_picker.addItem("Custom Voice Reference Audio...", "__custom__")
        target_index = self.reference_picker.findData(current_path)
        if target_index < 0:
            target_index = self.reference_picker.findData("__custom__")
        self.reference_picker.setCurrentIndex(target_index)
        self.reference_picker.blockSignals(False)
        if current_path in self._available_reference_paths:
            self.reference_path.setText(current_path)

    def _handle_reference_selection(self, _index=None) -> None:
        data = self.reference_picker.currentData()
        if data == "__custom__":
            self._pick_audio()
            return
        if isinstance(data, str) and data:
            self.reference_path.setText(data)

    def set_blend_choices(
        self,
        defaults: list[VoiceChoice],
        recents: list[str],
        selected_path: str = "",
        blends: list[VoiceChoice] | None = None,
    ) -> None:
        """Populate the 'Blend With Voice' picker (None + the same voice lists)."""
        current = selected_path or self.blend_target_path()
        self.blend_picker.blockSignals(True)
        self.blend_picker.clear()
        self.blend_picker.addItem("None (single voice)", "__none__")
        if defaults:
            header_index = self.blend_picker.count()
            self.blend_picker.addItem("Default Voices")
            header_item = self.blend_picker.model().item(header_index)
            if header_item is not None:
                header_item.setEnabled(False)
            for voice in defaults:
                self.blend_picker.addItem(f"  {voice.label}", voice.path)
        if blends:
            header_index = self.blend_picker.count()
            self.blend_picker.addItem("Saved Blends")
            header_item = self.blend_picker.model().item(header_index)
            if header_item is not None:
                header_item.setEnabled(False)
            for voice in blends:
                self.blend_picker.addItem(f"  {voice.label}", voice.path)
        if recents:
            header_index = self.blend_picker.count()
            self.blend_picker.addItem("Recent Custom Clips")
            header_item = self.blend_picker.model().item(header_index)
            if header_item is not None:
                header_item.setEnabled(False)
            for path in recents[:10]:
                resolved = str(Path(path).expanduser())
                self.blend_picker.addItem(f"  {Path(resolved).name}", resolved)
        index = self.blend_picker.findData(current)
        self.blend_picker.setCurrentIndex(index if index >= 0 else 0)
        self.blend_picker.blockSignals(False)

    def select_blend_path(self, path: str) -> None:
        """Select a specific blend target (restoring a saved project)."""
        if not path:
            self.blend_picker.setCurrentIndex(0)
            return
        index = self.blend_picker.findData(str(Path(path).expanduser()))
        if index < 0:
            # The saved clip is not in the standard lists; keep the selection
            # honest by staying on None rather than inventing an entry.
            self.blend_picker.setCurrentIndex(0)
            return
        self.blend_picker.setCurrentIndex(index)

    def blend_target_path(self) -> str:
        """The selected second clip, or "" when blending is off."""
        data = self.blend_picker.currentData()
        if isinstance(data, str) and data and data != "__none__":
            return data
        return ""

    def _save_blend_clicked(self) -> None:
        if self.on_save_blend is not None:
            self.on_save_blend(self)


def _speaker_settings_from_group(
    group: SpeakerGroup,
    variant: str,
    crossfade_ms: int,
) -> SpeakerSettings:
    """Read one speaker panel's widgets into engine :class:`SpeakerSettings`.

    Module-level so both the main window and the cast-management dialog (which
    owns its own panels) build settings identically.
    """
    blend_target = group.blend_target_path()
    return SpeakerSettings(
        reference_path=group.reference_path.text(),
        voice_settings=VoiceSettings(
            variant=variant,
            language=group.language_combo.currentData() or "en",
            cfg_weight=group.cfg_weight.value(),
            exaggeration=group.exaggeration.value(),
            temperature=group.temperature.value(),
            emotion_intensity=group.emotion_intensity.value(),
            naturalness=group.naturalness.value(),
            pause_ms=group.pause_spin.value(),
            crossfade_ms=crossfade_ms,
        ),
        blend_references=[group.reference_path.text(), blend_target] if blend_target else [],
        blend_weight=group.blend_weight_spin.value() / 100.0,
        blend_mode=group.blend_mode_combo.currentData() or "mix",
    )


def _apply_speaker_settings_to_group(group: SpeakerGroup, settings: SpeakerSettings) -> None:
    """Write engine :class:`SpeakerSettings` back into one speaker panel's widgets.

    Covers the hybrid (blend) controls too, so a saved blend target, dominance
    weight, and combine mode survive a settings round-trip instead of being
    silently reset.
    """
    voice = VoiceSettings.from_mapping(settings.voice_settings)
    group.reference_path.setText(settings.reference_path)
    language_index = group.language_combo.findData(voice.language)
    if language_index >= 0:
        group.language_combo.setCurrentIndex(language_index)
    group.cfg_weight.setValue(voice.cfg_weight)
    group.exaggeration.setValue(voice.exaggeration)
    group.temperature.setValue(voice.temperature)
    group.emotion_intensity.setValue(voice.emotion_intensity)
    group.naturalness.setValue(voice.naturalness)
    group.pause_spin.setValue(voice.pause_ms)
    group.blend_weight_spin.setValue(int(round(settings.blend_weight * 100)))
    mode_index = group.blend_mode_combo.findData(settings.blend_mode)
    if mode_index >= 0:
        group.blend_mode_combo.setCurrentIndex(mode_index)
    group.select_blend_path(settings.blend_references[1] if len(settings.blend_references) >= 2 else "")


class _CastRow:
    """One editable speaker row inside the cast-management dialog."""

    def __init__(self, key: str, group: SpeakerGroup) -> None:
        self.key = key
        self.group = group
        self.container = QWidget()
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        self.key_label = QLabel(f"Speaker {key}")
        self.key_label.setStyleSheet("font-weight: 700;")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Character name (optional)")
        self.name_edit.setToolTip(
            "Optional display name for this speaker, shown in the main window's cast summary."
        )
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip(
            "Remove this speaker from the cast. Removing every speaker but one "
            "returns the main window to monologue mode."
        )
        header.addWidget(self.key_label)
        header.addWidget(self.name_edit, 1)
        header.addWidget(self.remove_button)
        layout.addLayout(header)
        layout.addWidget(group)

    def set_name(self, name: str) -> None:
        self.name_edit.blockSignals(True)
        try:
            self.name_edit.setText(name)
        finally:
            self.name_edit.blockSignals(False)
        self._retitle(name)

    def _retitle(self, name: str) -> None:
        name = (name or "").strip()
        self.group.setTitle(f"Speaker {self.key} — {name}" if name else f"Speaker {self.key}")


class CastManagementDialog(QDialog):
    """Modal window for managing the speaker cast.

    The main window keeps a single narrator-oriented layout; when the user
    wants more than one speaker, this dialog opens and every cast member gets
    a full :class:`SpeakerGroup` panel (voice reference picker, hybrid second
    voice, dominance weight, combine mode, and all voice sliders) plus an
    optional character name. Speakers can be added freely up to the engine's
    voice capacity (:data:`MAX_CAST_SPEAKERS`); the narrator (A) can never be
    removed — deleting every other speaker is the graceful path back to
    monologue.

    Changes apply when the window closes (Done or the window's close button);
    the main window then rebuilds its panels from the dialog's cast.
    """

    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__(main_window)
        self._main = main_window
        self.setWindowTitle("Manage cast")
        self.setModal(True)
        self.resize(760, 680)

        # Snapshot the cast; the dialog edits the snapshot and hands it back
        # on close, so the main window's live panels are never half-edited.
        self._model = CastModel.from_parts(
            main_window.cast_keys(),
            main_window.speaker_names(),
            {key: self._settings_to_dict(settings) for key, settings in main_window.speaker_settings().items()},
        )
        self._variant = main_window.variant_combo.currentText()

        layout = QVBoxLayout(self)
        info = QLabel(
            "Each speaker gets a full voice panel: reference voice, hybrid second voice, "
            "and voice sliders. Changes apply when this window closes. "
            f"The engine voices up to {MAX_CAST_SPEAKERS} speakers distinctly."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.addStretch(1)
        scroll.setWidget(self._rows_host)
        layout.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        self.add_button = QPushButton("+ Add speaker")
        self.add_button.setToolTip("Add another speaker to the cast.")
        self.add_button.clicked.connect(self._add_speaker)
        self.done_button = QPushButton("Done")
        self.done_button.setDefault(True)
        self.done_button.clicked.connect(self._apply_and_close)
        buttons.addWidget(self.add_button)
        buttons.addStretch(1)
        buttons.addWidget(self.done_button)
        layout.addLayout(buttons)

        self._rows: dict[str, _CastRow] = {}
        for key in self._model.keys():
            self._add_row(key)
        self._refresh_pickers()
        self._refresh_add_button()

    # -- row management ---------------------------------------------------

    def _add_row(self, key: str) -> None:
        member = next((m for m in self._model.members if m.key == key), None)
        group = SpeakerGroup(key, self._main.paths.voice_dir, on_save_blend=self._on_save_blend)
        if member is not None:
            _apply_speaker_settings_to_group(group, self._dict_to_settings(member.settings))
        row = _CastRow(key, group)
        if member is not None:
            row.set_name(member.name)
        # The narrator anchors the cast (and monologue mode); it can never be
        # removed, so its Remove button is disabled with an explanation.
        if key == "A":
            row.remove_button.setEnabled(False)
            row.remove_button.setToolTip("The narrator (Speaker A) can't be removed.")
        else:
            row.remove_button.clicked.connect(lambda _checked=False, k=key: self._remove_speaker(k))
        row.name_edit.textChanged.connect(lambda _text, k=key: self._rename_speaker(k))
        self._rows[key] = row
        # Insert above the bottom stretch item.
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row.container)

    def _add_speaker(self) -> None:
        member = self._model.add()
        if member is None:
            QMessageBox.information(
                self,
                "Cast Full",
                f"The engine voices up to {MAX_CAST_SPEAKERS} speakers distinctly, "
                "so the cast can't grow further.",
            )
            return
        member.settings = self._settings_to_dict(
            SpeakerSettings(voice_settings=VoiceSettings(variant=self._variant))
        )
        self._add_row(member.key)
        self._refresh_pickers()
        self._refresh_add_button()

    def _remove_speaker(self, key: str) -> None:
        if not self._model.remove(key):
            return
        row = self._rows.pop(key, None)
        if row is not None:
            self._rows_layout.removeWidget(row.container)
            row.container.deleteLater()
        self._refresh_add_button()

    def _rename_speaker(self, key: str) -> None:
        row = self._rows.get(key)
        if row is None:
            return
        name = row.name_edit.text().strip()
        self._model.rename(key, name)
        row._retitle(name)

    def _refresh_add_button(self) -> None:
        full = next_speaker_key(self._model.keys()) is None
        self.add_button.setEnabled(not full)
        if full:
            self.add_button.setToolTip(
                f"The engine voices up to {MAX_CAST_SPEAKERS} speakers distinctly."
            )

    # -- blends / pickers ---------------------------------------------------

    def _on_save_blend(self, group: SpeakerGroup) -> None:
        self._main._save_blend_as_voice(group)
        self._refresh_pickers()

    def _refresh_pickers(self) -> None:
        defaults = default_voice_choices(self._main.repo_root, limit=40)
        blends = blend_voice_choices(self._main.paths.profile_dir)
        recents = [path for path in load_recent_reference_paths() if Path(path).exists()]
        for row in self._rows.values():
            group = row.group
            group.set_reference_choices(defaults, recents, group.reference_path.text(), blends=blends)
            group.set_blend_choices(defaults, recents, group.blend_target_path(), blends=blends)

    # -- apply / close ------------------------------------------------------

    @staticmethod
    def _settings_to_dict(settings: SpeakerSettings) -> dict:
        voice = settings.voice_settings
        voice_dict = voice.to_dict() if hasattr(voice, "to_dict") else dict(voice)
        return {
            "reference_path": settings.reference_path,
            "voice_settings": voice_dict,
            "emotion_reference_paths": dict(settings.emotion_reference_paths),
            "blend_references": list(settings.blend_references),
            "blend_weight": settings.blend_weight,
            "blend_mode": settings.blend_mode,
        }

    @staticmethod
    def _dict_to_settings(data: dict) -> SpeakerSettings:
        data = data if isinstance(data, dict) else {}
        return SpeakerSettings(
            reference_path=str(data.get("reference_path", "")),
            voice_settings=dict(data.get("voice_settings", {})),
            emotion_reference_paths=dict(data.get("emotion_reference_paths", {})),
            blend_references=list(data.get("blend_references", []) or []),
            blend_weight=float(data.get("blend_weight", 0.5)),
            blend_mode=str(data.get("blend_mode", "mix")),
        )

    def _collect(self) -> tuple[list[str], dict[str, str], dict[str, SpeakerSettings]]:
        """Read every row's widgets back into (keys, names, settings)."""
        keys: list[str] = []
        names: dict[str, str] = {}
        settings: dict[str, SpeakerSettings] = {}
        variant = self._main.variant_combo.currentText()
        crossfade_ms = self._main.crossfade_spin.value()
        for key in self._model.keys():
            row = self._rows.get(key)
            if row is None:
                continue
            keys.append(key)
            names[key] = row.name_edit.text().strip()
            settings[key] = _speaker_settings_from_group(row.group, variant, crossfade_ms)
        return keys, names, settings

    def _apply_and_close(self) -> None:
        keys, names, settings = self._collect()
        self._main.apply_cast(keys, names, settings)
        self.accept()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt casing)
        # The window close button applies pending edits, just like Done:
        # closing with one speaker left gracefully returns the main window to
        # monologue mode. No worker threads live here, so close is immediate.
        try:
            keys, names, settings = self._collect()
            self._main.apply_cast(keys, names, settings)
        finally:
            super().closeEvent(event)
