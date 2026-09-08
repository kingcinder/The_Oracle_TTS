# Voice Craft + Recording Studio — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax. This run is executed
> inline with **no commits** — the user reviews the working tree before any commit/push.

**Goal:** Fix punctuation-blind pacing and per-line timbre drift in TTS output, replace
the engine-named voice knobs with perceptually named sliders, and add a Custom Voice
Recording Studio window with teleprompter, mic/rate/output pickers, and picker refresh
on close.

**Architecture:**
- A: a pure pause-profile module (`utils/pacing.py`) consumed by `pipeline.py`, plus a
  render-loop change so only an utterance's final chunk carries the turn pause.
  Emotion blends only performance keys (`exaggeration`, `pause_ms`), locking
  temperature/CFG per speaker.
- B: a `PerceptualSlider` composite widget (`gui_widgets.py`) replacing the five
  engine-named spin boxes (plus pause/blend) in `SpeakerGroup` with visceral naming.
- C: a pure capture module (`audio/recorder.py`) behind a `sounddevice`-guarded import,
  and a `RecordingStudioDialog` opened from a top-level menu-bar action; the pickers
  refresh on the dialog's `finished` signal unconditionally.

**Tech stack:** Python 3.11/3.12, PySide6 6.8.2.1, NumPy, soundfile; `sounddevice`
added for capture.

## Global Constraints
- Attribute/field names persist: `cfg_weight`, `exaggeration`, `temperature`,
  `emotion_intensity`, `naturalness`, `pause_spin`, `blend_weight_spin` — serialization
  keys, profiles, manifests, tests stay valid.
- Period-terminated utterances must keep the current default pause (factor 1.0 → 180 ms)
  so existing plan tests remain green.
- No commits, no push. All work stays uncommitted on
  `feature/voice-craft-and-recording-studio` for review.
- GUI additions degrade gracefully when `sounddevice` is missing or no mic exists.
- Reference ingestion remains mono-normalized to 24 kHz; new recordings save mono WAV.

---

### Task 1: `utils/pacing.py` — punctuation pause profile + chunk seam pause

**Files:**
- Create: `src/the_oracle/utils/pacing.py`
- Test: `tests/test_pacing.py`

- [ ] Write tests:

```python
from the_oracle.utils.pacing import chunk_seam_pause_ms, pause_for_utterance

def test_period_keeps_base_pause():
    assert pause_for_utterance("This is a calm line.", 180) == 180

def test_question_and_exclamation_lengthen():
    assert pause_for_utterance("Wait, really?", 180) == int(180 * 1.25)
    assert pause_for_utterance("Stop!", 180) == int(180 * 1.3)

def test_ellipsis_breathes_longest():
    assert pause_for_utterance("And then…", 180) == int(180 * 1.6)

def test_clause_or_no_terminal_shortens():
    assert pause_for_utterance("He paused,", 180) == int(180 * 0.7)
    assert pause_for_utterance("and kept going", 180) == int(180 * 0.7)

def test_quotes_after_terminal_still_scaled():
    assert pause_for_utterance("No!\"", 180) == int(180 * 1.3)

def test_clamped_to_domain():
    assert 0 <= pause_for_utterance("…", 2000) <= 2000

def test_chunk_seam_is_small_breath():
    assert chunk_seam_pause_ms(180) == 63
    assert chunk_seam_pause_ms(0) == 40
```

- [ ] Run to verify failure: `pytest tests/test_pacing.py -q` (import error).
- [ ] Implement module (terminal-char scan from the end, quote/closing-bracket
  tolerance, factor table, clamp; seam = `max(40, round(0.35 * turn))`).
- [ ] Run to verify pass.
- [ ] No commit (per plan constraints).

### Task 2: pipeline — apply pause profile + timbre-locked emotion

**Files:**
- Modify: `src/the_oracle/pipeline.py` (`_apply_emotion_and_naturalness`,
  `prepare_plan` utterance loop)
- Test: `tests/test_emotion_controls.py`, `tests/test_pacing.py`

- [ ] In `_apply_emotion_and_naturalness`, blend emotion controls only for the keys
  `("exaggeration", "pause_ms")`; keep the naturalness block untouched; update the
  docstring.
- [ ] In `prepare_plan`, between `_apply_emotion_and_naturalness` and
  `apply_directives`, apply `pause_for_utterance(cleaned_text, merged.pause_ms)` to
  `merged.pause_ms` (author directives still override afterwards).
- [ ] Extend `tests/test_pacing.py` with a prepare_plan-style assertion is NOT required
  (slow suite) — assert at the pure function level; run `pytest tests/test_emotion_controls.py
  -q -m ""`? No: keep slow file untouched; verify behavior via:
  `pytest tests/test_directives.py tests/test_wpm_controls.py -q` after Task 3.
- [ ] Verify: `pytest tests/test_pacing.py -q` still green.

### Task 3: pipeline render — one turn pause per utterance (final chunk only)

**Files:**
- Modify: `src/the_oracle/pipeline.py` (task build loop + results loop segment creation)
- Test: `tests/test_render_worker_integration.py`, `tests/test_audio_assemble.py`

- [ ] After `raw_tasks` is built, compute `last_task_per_source: dict[int, int]` =
  `{task.source_index: task.utterance_index}` iterating in order (final occurrence wins).
- [ ] In the results loop where `AudioSegment` is appended, set
  `pause_after_ms = utterance.pause_after_ms` when
  `result.utterance_index == last_task_per_source.get(utterance.index)` else
  `chunk_seam_pause_ms(utterance.pause_after_ms)`.
- [ ] Run: `pytest tests/test_audio_assemble.py tests/test_render_worker_integration.py -q`
  (existing expectations must hold — non-chunked single stems keep the full pause).

### Task 4: `gui_widgets.py` — PerceptualSlider

**Files:**
- Create: `src/the_oracle/gui_widgets.py`
- Test: `tests/test_sliders.py`

- [ ] Write tests: float mapping round-trip (`value() == setValue`), int-mode, range
  change re-maps readout, caption string rendered, tooltip forwards to the slider.
- [ ] Implement `PerceptualSlider(QWidget)`: `QSlider(Horizontal)` ticks 0–1000 +
  readout `QLabel` + caption `QLabel`; `value()/setValue()/setRange()`; `int_mode`
  rounds; `setToolTip` sets slider + widget tooltip.
- [ ] `pytest tests/test_sliders.py -q` green.

### Task 5: SpeakerGroup conversion + visceral copy

**Files:**
- Modify: `src/the_oracle/app_gui.py` (SpeakerGroup constructor + row labels +
  help registration block)
- Test: `tests/test_app_gui_profiles.py`, `tests/test_gui_tooltips.py`

- [ ] Swap the five `_double_box` knobs + pause + blend-weight widgets to
  `PerceptualSlider` with new labels/captions (Voice Lock, Emotional Punch, Delivery
  Variety, Emotion Strength, Human Drift, Breath After This Speaker, Which Voice Wins).
- [ ] Rename `.form.addRow` strings; register the new captions/tooltips through the
  existing `_register_ctrl_help_descriptions` speaker-group block and
  `register_form_labels` lists (attribute references unchanged).
- [ ] Run `pytest tests/test_app_gui_profiles.py tests/test_gui_tooltips.py -q` green.

### Task 6: voice catalog — show curated Seashells recordings

**Files:**
- Modify: `src/the_oracle/voice_catalog.py`, `src/the_oracle/app_gui.py`
  (`_refresh_reference_pickers`, `SpeakerGroup.set_reference_choices`)

- [ ] `set_reference_choices` stops slicing `defaults[:10]` (show all passed-in);
  `_refresh_reference_pickers` calls `default_voice_choices(self.repo_root, limit=30)`.
- [ ] `pytest tests/test_voice_catalog.py tests/test_blend_voices.py -q` green.

### Task 7: `audio/recorder.py` — pure capture module

**Files:**
- Create: `src/the_oracle/audio/recorder.py`
- Test: `tests/test_recorder.py`

- [ ] Write tests with an injected fake `sounddevice` object (module-level
  `_sounddevice` seam): device list shape/default, samplerate probe honors the fake's
  `check_input_settings` successes, auto-name `Seashell_No_<next>.wav` increments across
  existing files, `save_wav` writes a file soundfile can read.
- [ ] Implement: `have_capture_backend`, `InputDevice` dataclass,
  `list_input_devices`, `samplerates_for_device`, `record_blocking` (sd.rec-style via
  InputStream collection is kept in the GUI thread worker; pure module exposes
  `capture_stream` returning a generator? — simpler: GUI worker calls
  `record_until_stop(device, samplerate, channels, stop_event, on_level)`), and
  `next_seashell_name(voice_dir)`.
- [ ] `pytest tests/test_recorder.py -q` green.

### Task 8: RecordingStudioDialog + top-level action

**Files:**
- Modify: `src/the_oracle/app_gui.py` (menu build, new dialog class, close wiring)
- Test: `tests/test_recording_studio.py`

- [ ] Dialog builds from `repo_root`; teleprompter combo lists `Input/*.txt|*.md`;
  mic/rate/outdir/filename rows; Record/Stop with worker thread (record + level
  callbacks); saving to `Seashells/Seashell_No_<n>.wav` (overwrite prompt guarded).
- [ ] `MainWindow`: `self.recording_studio_action = QAction("Recording Studio…")` added
  to the menu bar; handler opens the dialog; `dialog.finished.connect(
  self._refresh_reference_pickers)`; register ctrl-help for the action.
- [ ] Tests: offscreen dialog opens without a backend (record disabled), teleprompter
  defaults from `Input/`, closing fires refresh even with a failed take; with a fake
  recorder + fake save, a successful take also refreshes.
- [ ] `pytest tests/test_recording_studio.py -q` green.

### Task 8b: audition + quick-assign of a fresh take

**Files:**
- Modify: `src/the_oracle/app_gui.py` (`RecordingStudioDialog`, `MainWindow`)
- Test: `tests/test_recording_studio.py`

- [ ] Dialog gains an "Auto-play new take" checkbox (default on), a **Listen
      again** button, and **Use for Speaker A/B** buttons (enabled only after a
      successful save).
- [ ] `_play_take` auditions via the existing QtMultimedia
      `QMediaPlayer`/`QAudioOutput` pair; playback stops on close/new take.
- [ ] `_use_for_speaker` fires the `on_assign(speaker, path)` callback;
      `MainWindow` passes `_assign_recording_to_speaker`, which sets that
      speaker group's reference path and refreshes the pickers.
- [ ] Tests: auto-audition fires on save (fake media player), disabled when the
      checkbox is off, assign buttons invoke the callback with the saved path,
      and the MainWindow wiring points the group reference at the new file.
- [ ] `pytest tests/test_recording_studio.py -q` green.

### Task 9: dependency + install + full suite

**Files:**
- Modify: `pyproject.toml` (add `sounddevice>=0.4.6,<0.6` to dependencies)

- [ ] Install into dev env: `.venv/bin/python -m pip install "sounddevice>=0.4.6,<0.6"`
  (failure tolerated — guards keep the app functional; report if offline).
- [ ] Full suite: `.venv/bin/python -m pytest -q`. Fix regressions. Re-run until green.
- [ ] Report status + leave working tree uncommitted for review.
