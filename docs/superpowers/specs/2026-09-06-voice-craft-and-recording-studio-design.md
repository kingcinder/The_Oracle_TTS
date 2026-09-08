# Voice Craft + Recording Studio — Design

Date: 2026-09-06 · Branch: `feature/voice-craft-and-recording-studio` (uncommitted until review)

This spec covers three related improvements to The Oracle's voice pipeline and GUI,
decided after three brainstorming passes (architecture review, requirements/approach
design, and product lens):

1. **Audio quality:** fix flat "punctuation-blind" pacing and inconsistent emotional
   inflection in rendered speech.
2. **Perceptual controls:** turn the speaker voice-modulation controls into sliders
   whose names and captions say what the change sounds like, not what the engine knob
   is called.
3. **Custom Voice Recording Studio:** a top-level window for recording new reference
   voices from a microphone, with a teleprompter, device/rate/output pickers, and
   automatic picker refresh on close.

Everything is on top of the merged `main` at `1ebe925` (fast-forwarded serpent-circle
renovation). No commits are made until the user reviews.

---

## 1. Audio quality — punctuation-aware pacing + timbre-locked emotion

### Problem evidence
- `prepare_plan` gives every utterance the same flat `pause_after_ms` (default 180 ms)
  no matter the terminal punctuation. A one-word "Yes." breathes identically to a long
  run-on clause.
- A chunked utterance (every stem in `pipeline.render`) carries the **same full
  turn-pause after every chunk**, so a long sentence split at 250 chars gets stacked
  pauses at seams that contain no punctuation — the "weird pacing".
- `_apply_emotion_and_naturalness` moves `temperature` and `cfg_weight` **per line**
  (via the emotion label map). Temperature/CFG are timbre/identity knobs; per-line
  jitter makes one speaker's character drift sentence to sentence — the "inconsistent
  inflection".

### Changes
1. New pure helper module `src/the_oracle/utils/pacing.py`:
   - `pause_for_utterance(text, base_pause_ms) -> int` — scales the base pause by the
     utterance's terminal punctuation: `.` → 1.0×, `!` → 1.3×, `?` → 1.25×, trailing
     `…/...` → 1.6×, weak/no terminal punctuation (comma, clause) → 0.7×. Clamped to
     the slider's 0–2000 ms domain. Deterministic and unit-testable.
   - `chunk_seam_pause_ms(turn_pause_ms) -> int` — a short "breath" (≈35%, min 40 ms)
     placed between chunks **of the same utterance**, replacing the stacked full pauses.
2. `pipeline.prepare_plan`: apply `pause_for_utterance` to the utterance text before
   author directives are applied (an explicit `[pause=...]`/rate directive still wins).
3. `pipeline.render`: when building `AudioSegment`s, the full turn pause is attached
   only to each utterance's **final** chunk; intermediate chunk seams get the small
   breath pause. Implemented by precomputing, from `raw_tasks`, the last task index per
   source utterance and comparing against `result.utterance_index`.
4. Timbre lock in `_apply_emotion_and_naturalness`: per-line emotion blends only the
   **performance** keys (`exaggeration`, `pause_ms`). `temperature`, `cfg_weight`,
   `repetition_penalty`, `min_p` never move per line from emotion; they are per-speaker
   constants (the speaker-level Naturalness heuristic may still move them — it is
   constant for the whole render by design). Emotion mapping values remain in
   `emotion/goemotions.py` (used for the performance subset).

### Non-goals / deferred
- No new emotion model, no whisper/emphasis acoustic modeling, no per-word prosody.
- Chatterbox's own internal prosody at punctuation remains canonical (documented,
   identical on both backends).

---

## 2. Perceptual voice-modulation sliders

### Problem
SpeakerGroup exposes `QDoubleSpinBox`es labelled with engine names (CFG Weight,
Exaggeration, Temperature, Emotion Intensity, Naturalness) plus a ms pause spin and a
% blend spin. A user cannot tell what a number will do to the voice.

### Changes
1. New reusable widget `PerceptualSlider` (in `src/the_oracle/gui_widgets.py`):
   - A horizontal `QSlider` over an internal 0–1000 tick domain, mapped linearly onto a
     domain range; a live value readout; and an always-visible **caption** describing
     the audible effect.
   - Preserves the API the rest of the code and tests rely on: `value()`, `setValue(v)`,
     `setRange(lo, hi)`, and tooltips. Float domains for CFG/Exaggeration/Temperature/
     Emotion/Naturalness; integer domains for pause (ms) and blend presence (%).
   - The five perceptual speaker knobs are shown on a normalized 0–100 "how much do you
     hear it" feel scale; `pause` keeps true ms and `blend` keeps true %.
2. `SpeakerGroup` (`app_gui.py`) replaces the affected spin boxes with `PerceptualSlider`s
   and renames the form rows + captions + tooltips to visceral language, e.g.:
   - CFG Weight → **Voice Lock** — "How strictly this voice stays on its reference.
     High = steadier and closer to the original; low = looser, drifts more."
   - Exaggeration → **Emotional Punch** — "How hard emotional emphasis lands.
     High = bigger swings in pitch and stress; low = flat, matter-of-fact."
   - Temperature → **Delivery Variety** — "How surprising the delivery is. High =
     more varied takes; low = steadier, more predictable."
   - Emotion Intensity → **Emotion Strength** — "How strongly detected emotions color
     the performance (emphasis and pacing — the voice's character is unchanged)."
   - Naturalness → **Human Drift** — "Loosens the sampling for a more natural, less
     mechanical voice (still constant for the whole render)."
   - Pause After Speaker Turn → **Breath After This Speaker** (ms; punctuation-aware).
   - Base Voice Presence → **Which Voice Wins** (%).
3. Wire the renamed rows through Ctrl+hover help (`_register_ctrl_help_descriptions`,
   `gui_tooltips.register_form_labels`) with the new copy.
4. Persistence/round-trip untouched: attribute names (`cfg_weight`, `pause_spin`, …) and
   serialized field names are unchanged, so profiles, templates, projects, and existing
   tests that drive `.value()/.setValue()` keep working.

---

## 3. Custom Voice Recording Studio

### User story
A top-of-window **Recording Studio…** button opens a separate window. It lets the user
read a script from a teleprompter area while recording a microphone into a new
reference "Seashell" in `Seashells/`, saved as WAV with an auto-incrementing name that
never overwrites an existing recording. Whenever the window closes — success or not —
the Speaker A/B (and extra cast) voice pickers refresh so new recordings are selectable
immediately.

### Layout (new `RecordingStudioDialog` in `app_gui.py`, driven by pure logic in
`src/the_oracle/audio/recorder.py`)

Audition uses the GUI's existing QtMultimedia `QMediaPlayer`/`QAudioOutput` pair
(the same playback path used for review-table previews), guarded so no player is
constructed unless a take exists. Assignment goes through an `on_assign` callback
(`MainWindow._assign_recording_to_speaker`), which keeps the dialog decoupled
from `MainWindow` internals for testability.
- **Top-left toolbar:** teleprompter file dropdown, populated from the repo's `Input/`
  folder (`*.txt`, `*.md`) — the folder's samples appear by default. Selecting one loads
  it into the prompt area.
- **Prompt/reading area (largest region):** `QPlainTextEdit`; user-editable; font can be
  bumped with a size spin.
- **Capture row:**
  - Microphone dropdown (installed inputs, default first). Populated from `sounddevice`.
  - Sample-rate dropdown — enabled/visible only after a microphone is selected; lists
    rates the mic actually supports (probe via `sd.check_input_settings`), default rate
    first.
  - Output folder dropdown (defaults to `Seashells/`), plus a filename field defaulting
    to `Seashell_No_<next>.wav` computed from existing files; choosing an existing file
    prompts to overwrite (default: no, auto-bump instead).
  - Record/Stop button, elapsed-time label, and a live input level meter.
- **Status panel** at the bottom (last message / last saved file).
- **Take actions** (after a successful save): an "Auto-play new take" checkbox
  (default on) that auditions each take through the media player the moment it
  is saved, a **Listen again** button to re-play the last take, and **Use for
  Speaker A** / **Use for Speaker B** buttons that point that speaker's Custom
  Voice Reference Audio at the new Seashell and refresh the main window's
  voice pickers (the dialog stays open for further takes).

### Capture semantics
- Mono WAV (PCM_16, chosen sample rate). Reference ingestion normalizes mono→24 kHz and
  trims silence anyway; saving at the native rate keeps the take lossless.
- Recording runs in a worker `QThread` using a `sounddevice.InputStream` callback that
  accumulates float32 samples and emits level updates; Stop concatenates and hands back
  the array. Absence of `sounddevice` or of any microphone disables recording with a
  clear inline message (the window still opens, teleprompter still works).

### Refresh contract (explicit requirement)
`MainWindow._refresh_reference_pickers()` is connected to the dialog's `finished` signal,
so Speaker A/B + extra cast pickers and their blend pickers repopulate **whether or not**
a recording succeeded. `_save_blend_as_voice`-style status text is appended on success.

### Catalog fix (required for refresh to be visible)
`default_voice_choices()` caps at 10 and the generic bundled library already fills that,
so a brand-new `Seashells/Seashell_No_1.wav` would never appear. The GUI now requests a
larger limit (e.g. 30) and `SpeakerGroup.set_reference_choices` stops slicing defaults to
10, so curated `Seashells/` recordings show up under Default Voices after their generic
cousins.

### Non-goals / deferred
- No trimming/cropping editor, no multi-track punch-in, no pitch shifting.
- Microphone recording needs `sounddevice`; it is added to `pyproject.toml`
  (`sounddevice>=0.4.6`), installed into the dev `.venv`, and imported lazily so the app
  degrades gracefully if the wheel is absent.

---

## Decomposition & verification
Three sub-projects, each independently testable:
- **A (audio):** `utils/pacing.py` unit tests + pipeline test additions; existing suite
  must stay green (period-terminated utterances keep 180 ms, so current expectations
  hold).
- **B (sliders):** `gui_widgets.py` unit tests; GUI profile/tooltip tests stay green.
- **C (studio):** `recorder.py` pure-function tests with an injected fake `sounddevice`;
  dialog smoke test offscreen verifying picker refresh on close with a recorded and with
  a failed take.

Final gate: full `pytest` suite passes; `git status` reviewed by user before any commit.
