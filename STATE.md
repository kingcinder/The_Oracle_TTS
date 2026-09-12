# The Oracle — State (completeness manifest)

**Current release: V1.11 (CUDA inference + complete guided onboarding persistence)**

This file is the repo's self-designated completeness record. It is the
authoritative context for the Omega meta-skill loop (searched, never
rewritten by the loop).

## Done

- **Ingestion transformer (2026-09-11, this campaign)**: new
  `src/the_oracle/ingest_transformer.py` analyzes an input script before use
  and automatically corrects non-canonical speaker-turn formats. Detected and
  fixed in place (canonical `Label: dialogue` output the existing ingester
  attributes correctly): dash/pipe separators (`A - text`, `Alice | text`),
  bracketed labels (`[A]: text`, `[Speaker A] text`), bulleted/quote-marked
  turns (`- A: text`, `> A: text`), period separators (`A. text`), orphan
  label lines (dialogue on the next line), chat-export timestamp prefixes,
  and non-UTF-8 encodings (UTF-16/CP1252 transcoded to UTF-8). Prose is
  protected: `Note:`/`See:` labels and parenthetical em-dash sentences are
  never rewritten (spaced-dash-inside-text guard + document-level label
  evidence). The GUI's Analyze path (`prepare_project`) now runs the
  transformer check first: a warning popup explains every issue and offers a
  one-click in-place fix with a timestamped backup; unfixable issues are
  explained with a continue/cancel choice. Accepting the fix opens a
  **Preview Fixed Text** dialog first: a unified diff (original vs fixed,
  `-`/`+` marked lines) via `preview_fixed_text()`, and nothing is written
  until the user accepts; cancelling in the preview leaves the file
  untouched. Accepting the fix re-runs Analyze automatically: the same
  click's `prepare_plan` reads the corrected file from disk and populates
  the review table in one step (no second click needed), and the status
  panel records the correction (count, file, backup path) so the automatic
  re-analysis is visible rather than silent. Transforms are idempotent and verified end-to-end against the
  real `TextIngestor` attribution. The CLI render path runs the same check:
  issues are always reported on stderr, and `--fix-input` corrects the
  fixable ones in place (timestamped backup) before the pipeline loads;
  without the flag the file is never modified and the output includes a
  re-run hint. 27 module tests (`tests/test_ingest_transformer.py`), 5
  GUI-flow tests (`tests/test_ingest_transformer_gui.py`), and 6 CLI tests
  (`tests/test_cli.py`); full suite 727 passing.

- **Batch folder fix (2026-09-11, this campaign)**: File → Batch Fix Input
  Folder... scans every `.txt`/`.md` file in a chosen folder (non-recursive;
  `*.bak-*` fix backups and non-text extensions skipped) via
  `analyze_folder()`, computes all rewrites up front via
  `preview_folder_fixes()` (nothing written), and shows **one combined
  Preview Batch Fix dialog**: each file's unified diff grouped under a
  `=== filename ===` header, unfixable warnings listed at the bottom.
  Accepting applies every fix in one pass (`apply_folder_fixes()`), each
  original backed up, and the status panel reports the total plus per-file
  fix counts and backup paths; cancelling leaves the whole folder
  untouched. A folder with no problems gets an informational popup and no
  preview; a warnings-only folder gets an explanatory popup with no fix
  offered. The batch is idempotent (re-running finds nothing) and verified
  end-to-end through the real `TextIngestor`. 8 module tests
  (`tests/test_ingest_transformer_batch.py`) + 3 GUI tests; full suite 740
  passing.

- **check-input CLI subcommand (2026-09-11, this campaign)**:
  `the-oracle check-input FILE [--fix]` lints a dialogue file's formatting
  without loading the render pipeline (no LanguageTool download, no model).
  Exit codes: 0 = no issues (or all fixed with `--fix`), 1 = issues found
  (fixable ones listed with their exact rewrites, warnings marked `!`),
  2 = file missing/unreadable. Report-only by default; `--fix` corrects
  fixable issues in place (timestamped backup), then re-analyzes so the
  exit code reflects what remains. Warnings are never rewritten, and the
  re-run hint is suppressed when nothing is fixable. 6 tests in
  `tests/test_cli.py`; full suite 746 passing.

- **--check-input-json (2026-09-11, this campaign)**: `the-oracle render
  --check-input-json` emits the input-formatting report as exactly one JSON
  document on stdout instead of human-readable stderr text — `file`,
  `fixable_count`, `warning_count`, and an `issues` list (`line`,
  `snippet`, `fixable`, `description`); with `--fix-input` the same
  document gains `fixed_count` and `backup` (pre-fix issue list plus what
  was applied, in one object so consumers can always parse stdout as a
  single document). Missing/unreadable files report an `error` key rather
  than emitting nothing. 6 tests in `tests/test_cli.py`; full suite 752
  passing. The document builder is shared with `check-input --json`
  (below) so the two schemas can never drift.

- **check-input --json (2026-09-11, this campaign)**:
  `the-oracle check-input FILE [--fix] [--json]` emits the same
  machine-readable report the render path produces — one JSON document on
  stdout, nothing else printed in json mode so CI can parse stdout
  cleanly. Exit codes are unchanged and the document always agrees with
  them: 0 = clean (or fixed with `--fix`, where the document describes the
  file's **post-fix** state and gains `fixed_count`/`backup`), 1 = issues
  found, 2 = missing/unreadable (or fix failure) with an `error` key.
  `speaker_refs`/`rejected_labels` are always present (stable schema).
  5 tests in `tests/test_cli.py`, including a schema-parity pin against
  `_input_json_document`; full suite 804 passing.

- **--speaker-ref suggestions (2026-09-11, this campaign)**: when the
  transformer reports a file, the report now suggests the exact voice flags
  for the script's cast. `suggest_speaker_refs(text)` reads the cast from
  the **post-fix** text (so a speaker only visible after a transform is
  included) and computes the mapping via the pipeline's own
  `DualSpeakerAttributor._map_labels_to_voices` — first-appearance order
  onto keys A..X, so a suggestion can never drift from what a render does.
  A/B use `--speakerA-ref`/`--speakerB-ref`; keys C+ are marked `new` (the
  flags a user must ADD). `rejected_labels(text)` lists labels that fail
  speaker validation entirely ("Chapter:", "Note:") — no reference audio
  can attribute those; the user must edit the label. Surfaced in:
  `check-input` (stderr hints, `use`/`add` markers), the render path's
  human report (stderr), and `--check-input-json` (stable `speaker_refs` +
  `rejected_labels` keys, always present). 6 tests in `tests/test_cli.py`;
  full suite 758 passing.

- **SRT auto-detection (2026-09-11, this campaign)**: the CLI render path
  converts `.srt` subtitle inputs into dialogue scripts before analysis
  (`src/the_oracle/srt_ingest.py` + `cli._maybe_convert_srt`). Detection is
  strict: only files with the `.srt` extension whose content parses as
  complete SubRip cues convert; a `.srt` file without valid cues (and any
  other file) passes through untouched, and missing/unreadable files are
  reported by the ordinary error paths. Conversion strips timestamps, cue
  indexes, HTML markup (`<i>`, `<font>`), and ASS overrides (`{\\an8}`);
  a `Name:` prefix on a cue line names the speaker (validated by the
  ingester's own `canonical_speaker_label`, so `Note:`-style prose prefixes
  stay spoken); cues without names continue the previous speaker, falling
  back to `Narrator`; dashed cue lines split into separate turns with
  alternation to the other voice of the pair (and never re-merge within a
  cue); consecutive same-speaker cues merge into one turn. The converted
  script is written as a sibling `<name>.srt.txt` (subsequent runs reuse it;
  the subtitle file itself is never modified) and replaces the input for
  the transformer check and pipeline; the cast round-trips through the real
  `TextIngestor`. 12 module tests (`tests/test_srt_ingest.py`) + 3 CLI-path
  tests; full suite 773 passing.

- **--fix-input-interactive (2026-09-11, this campaign)**: the render path's
  terminal equivalent of the GUI popup + preview. Shows the issue summary
  and the rule-labeled diff of the exact rewrite on stderr, then prompts
  `Apply these fixes before rendering? [y/N]` before writing (backup kept).
  Any non-yes answer — or EOF/ctrl-D — leaves the file untouched and aborts
  the render with exit code 2 *before OraclePipeline() loads*. In
  non-interactive runs (piped stdin, CI) the flag degrades to a report-only
  check with a stderr notice, so automation never blocks on a prompt.
  Prompt function is injectable for tests. 7 tests in `tests/test_cli.py`;
  full suite 785 passing.

- **Rule-labeled preview diffs (2026-09-11, this campaign)**:
  `transform_text_detailed(text)` returns per-fix provenance (`LineFix`:
  output line number, rule name, original text) alongside the fixed text;
  `transform_text` is a thin wrapper preserving the old API. Rules: dash,
  period, bracket, timestamp, orphan, bullet, encoding (`RULE_LABELS` maps
  them to display names). `labeled_fixed_diff(original, fixed, line_fixes)`
  builds the unified diff with each rewritten `+` line prefixed by its rule
  label — e.g. `+ [dash/pipe separator] A: Hello there.` — attributing by
  *position* (hunk headers are walked, not text-matched), so identical
  source lines fixed by different rules label correctly. The GUI's
  single-file preview uses this via `preview_fixed_text` (now 5-tuple with
  `line_fixes`) and `FolderFix.line_fixes`. 5 module tests + updated
  batch-GUI assertions.

- **Recursive batch fix with folder tree (2026-09-11, this campaign)**:
  `analyze_folder` now recurses into subfolders (`rglob`), skipping hidden
  files/directories, `_BATCH_SKIP_DIRS` (`__pycache__`, `node_modules`,
  `venv`, `site-packages`), non-text extensions, and `*.bak-*` backups;
  results sort by path relative to the scanned folder so the tree reads
  naturally. The **Preview Batch Fix** dialog is now a two-pane layout: a
  folder tree on the left (root = folder name, children = fixable files by
  relative path with fix counts) and a rule-labeled diff pane on the right
  showing the selected file (first file preselected); unfixable warnings
  listed beneath. Apply/Cancel semantics unchanged (all-or-nothing, backups
  kept, per-file status-panel lines with relative paths). 4 module tests +
  2 GUI tests; full suite 799 passing.

- **Per-file batch checkboxes (2026-09-11, this campaign)**: every file
  row in the Preview Batch Fix tree gained an include-checkbox (all
  ticked by default), so a batch can be applied selectively instead of
  all-or-nothing. The apply button live-updates to "Apply Fixes to N of
  M File(s) (K fix(es))" as ticks change and disables when nothing is
  ticked; Apply returns only the ticked fixes and the caller applies
  exactly those (unticked files untouched, no backups created for them).
  The dialog now returns the included-fix list (was a bool). 1 new GUI
  test (exclusion end-to-end, default-all-ticked, live label/enable);
  full suite 805 passing.

- **WebVTT (.vtt) support (2026-09-11, this campaign)**: the subtitle
  converter now accepts WebVTT alongside SubRip — same module
  (`srt_ingest.py`), same convert-not-overwrite contract. Parser additions:
  `WEBVTT` header, `MM:SS.mmm` clocks without hours, cue settings after
  the end time, cue identifiers, and `NOTE`/`STYLE`/`REGION` metadata
  blocks (skipped). **WebVTT voice spans** (`<v Winston>`) are lifted to
  `Winston:` prefixes before generic markup stripping — the demo caught
  them being deleted as tags, silencing the speaker. Detection is by
  content everywhere: `analyze_input_file` flags a valid `.vtt` as one
  fixable issue, `fix_input_file` writes the sibling `<name>.vtt.txt`
  script (subtitle never modified), the CLI pre-flight accepts `.vtt`
  (converting or reusing the script), and the batch scan includes `.vtt`
  files. The conversion moment now also **suggests the cast's voice
  flags**: the CLI pre-flight prints `_print_speaker_ref_hints` for the
  converted script (both convert and reuse paths), the render path's
  human report includes the hints (clean files too — the cast is the
  useful content there), and the GUI fix flow adds them to the post-fix
  "File Corrected" popup and the status panel, computed from the
  post-transform text so a subtitle conversion's cast is suggested even
  before the user accepts the fix. Full suite 820 passing. Applying a batch now honors the contract for **both** formats:
  a latent pre-existing bug had `apply_folder_fixes` writing the
  converted script *into* the subtitle file; it now redirects to the
  sibling script and reports that path. 6 converter tests + 3
  transformer/batch tests + 3 CLI tests; full suite 817 passing.

- **SRT-aware transformer (2026-09-11, this campaign)**: the transformer's
  own APIs now convert subtitles, complementing the CLI's `_maybe_convert_srt`
  pre-flight. `analyze_input_file` flags a structurally-valid SubRip file
  (detection is by content, not extension) as one fixable issue — "ingested
  directly, every cue would be read as narration" — and offers the
  conversion; `transform_text_detailed` handles SRT as a single whole-file
  `srt` rule (rule label "SRT subtitle"); `fix_input_file` is the one
  non-in-place case: it writes the converted script to a sibling
  `<name>.srt.txt` (subtitle file backed up, never modified) and returns
  the *written path* as its new first element (was the fixed text; all
  callers updated — the CLI render path re-points `--input` at the
  converted script after `--fix-input`/`--fix-input-interactive`, and the
  GUI flow renders the script too); `analyze_folder` includes `.srt` in
  batch scans. The GUI popup/preview/trusted-file flow, `check-input
  --fix`, and the JSON report all handle SRT inputs through the same APIs.
  5 module tests; full suite 796 passing.

- **Remember-my-choice / trusted files (2026-09-11, this campaign)**: the
  single-file fix preview dialog offers a checkbox, "Remember this choice
  and fix this file automatically in the future". Ticking it + accepting
  adds the file's resolved path to `trusted_format_files` in the app
  settings (`gui_settings.py` schema pass-through with string hygiene); a
  later Analyze/Render on the same file auto-corrects it silently (backup
  still kept, status-panel line says "trusted file"), no popup or preview.
  Trust is per-file: other files keep the full popup + preview flow.
  Settings > **Forget remembered auto-fix approvals** clears the list.
  SECURITY-RELEVANT FIX found by the new tests: the popup's Cancel branch
  compared `clickedButton() is StandardButton.Cancel`, which can never be
  true (widget vs. enum) — clicking Cancel silently proceeded to Analyze.
  Now uses `box.standardButton(clicked)`. 5 GUI tests; full suite 791
  passing.

- **Side-by-side preview dialog (2026-09-11, this campaign)**: the
  single-file **Preview Fixed Text** dialog is now a two-pane layout —
  original on the left (with `-` markers on rewritten rows), corrected text
  on the right (with `+` markers and the rule label as a suffix, e.g.
  `+ A: Hello there. [dash/pipe separator]`), headers `Original` /
  `Corrected (with fix rule)`, and synchronized scrolling (either pane's
  scrollbar drives the other). Alignment comes from
  `side_by_side_diff_rows(original, fixed, line_fixes)` in the transformer:
  a `SequenceMatcher` opcode walk producing `DiffRow(kind=same/changed/
  removed/added, rule)` rows — rewrites pair line-for-line as `changed`,
  exact regardless of script length. The batch dialog keeps the unified
  labeled diff. 2 GUI tests (pane contents + behavioral scroll coupling);
  full suite 786 passing.

- **V1.01 GUI release** (2026-09-08, this campaign):
  - Pain-point `~` markers (`word~word`) in input scripts are replaced by a
    normal spoken word boundary at the single synthesis chokepoint
    (`Utterance.text_for_tts`) in every correction mode — including Verbatim —
    so the annotated junctions (e.g. `syncronized~lockstep` in
    `Input/What is, reality.txt`) no longer cause engine hang-ups/
    mispronunciations, while unattached `~` (e.g. `~42`) is still read
    verbatim and the review table keeps the annotation.
  - Theme system (`src/the_oracle/gui_themes.py`): six stylized themes
    (Studio Light, Night Deck, Sephiroth, Pulp Science Fiction, Tape Deck,
    Ocean Depths), each a full design-token set (surfaces, typography,
    geometry) machine-contrast-checked at import time (WCAG AA body,
    3.0:1 large/on-accent). Default stays Studio Light. Theme menu in the
    menu bar; the choice persists.
  - Voice-modifier sliders renamed to match their mechanics (Identity Lock,
    Emphasis Punch, Delivery Variety, Emotion Depth, Human Drift, Breath
    After This Speaker) and given exact-mechanics tooltips: engine knob,
    domain, formula-level behavior (CFG dual-guess sampling, emotion
    preset blend weights, the per-0.1 Human Drift deltas, punctuation
    pause multipliers). Hybridize Voices is its own labeled section inside
    each speaker panel (Hybrid Second Voice / Voice Dominance / Hybridize
    Mode).
  - Every GUI section (Shared Render Settings, Speaker A/B, Status/Errors,
    Live, extra cast voices) is a collapsible section with a per-section
    Section Size slider redistributing its splitter share; three nested
    splitters (main/sections/lower) are resizable by handle or slider.
  - Workspace persistence in `app_settings.json`: theme, last input file
    (defaults to `Input/What is, reality.txt` on fresh installs when no
    remembered file exists), configurable default Input/Output folders,
    generic output filename warning preference, ALL options + slider positions
    (shared and per-speaker), splitter sizes, section shares, collapses, and
    window size.
  - File → Save Profile… / Load Profile… (same payload as Settings →
    Save/Load Settings).
  - One-stop manager wrappers `./oracle` (Linux/macOS) and `oracle.ps1`
    (Windows) with Install / Start / Update / Uninstall actions
    delegating to `scripts/manage_install.py` (new `update` action keeps
    user data); release version now 1.1.1.
  - First-run inference onboarding (`src/the_oracle/inference_wizard.py`):
    hardware-aware CUDA/CPU/Vulkan discovery, unsuitable-GPU explanations,
    a dependency-ordered Continue-driven GUI tutorial, live control
    highlighting with beside-the-tooltip placement, and Settings replay modes
    for the full tour, hardware discovery only, and main GUI tour only. The
    chosen inference path is applied to the real picker and tutorial completion
    or dismissal is persisted.
  - First-open Recording Studio onboarding (`src/the_oracle/recording_wizard.py`):
    microphone and supported sample-rate selection, shared Input/ teleprompter
    script selection (bundled What is, reality.txt first), configurable
    Seashells/ destination, generic-name caution with disable checkbox,
    persisted last-used recording choices, and a staged
    dependency-ordered guide with detailed mic placement, room, plosive,
    breath, enunciation, emotional delivery, audition, and speaker-assignment
    instructions. Settings can replay the guide at any time.
  - Certification: `scripts/certify_gui_themes.py` builds the real window
    under all six themes offscreen and asserts no truncation (geometry
    sweep with a populated table), WCAG legibility, working section
    sliders, collapse behavior, and a full persistence round-trip; full
    suite at 499 passing tests, deterministic smoke render green.

- Chatterbox-only render pipeline: `standard`, `multilingual`, and `turbo`
  variants on PyTorch, with CPU/system-DRAM as the guaranteed fallback, an
  opt-in CUDA device mode for suitable NVIDIA GPUs, and an opt-in Vulkan
  backend via audio.cpp for AMD RDNA1-class GPUs (vendored RDNA1 device-lost
  fix). CUDA remains the existing PyTorch/Chatterbox path rather than a
  second synthesis engine: hardware discovery reports card name, VRAM, driver
  visibility, runtime availability, and rejects cards below the 4 GiB
  suitability floor with an actionable reason.
- Cross-platform bootstrap/install/doctor/run/uninstall for Linux and
  Windows; managed launcher + desktop integration.
- Desktop GUI: review table, per-row preview, repair, live progress panel,
  profiles/templates, saved project manifests, Ctrl+hover help.
- CLI render flow with saved project manifests and deterministic smoke
  render path for repo-local verification.
- Batched Vulkan rendering (`--request-sequence`, bounded 32-request
  groups, live per-request progress) with truthful timing logs.
- Voice catalog: bundled generic voices (`Seashells/generic/` — 5
  English + 4 Chinese references, Apache-2.0 / CC BY 4.0 attributed,
  English-first in the picker), curated `Seashells/` defaults, recent
  custom clips.
- Voice blending: deterministic derived reference from two clips with a
  preference weight and mix/alternate/layer modes; persisted in projects
  and profiles; honored by both backends.
- Saved blend voices: **Save Blend As...** creates a named voice that
  appears under "Saved Blends" in the picker (catalog
  `Profiles/blend_voices.json`, derived clips `Profiles/.blends/`).
- Correction Mode **Verbatim (no changes)**: true passthrough of the
  source text (no spelling/grammar/punctuation edits).
- Fidelity fixes (commit `d77c278`): assembly no longer drops chunks of
  chunked utterances (parallel loader keyed by position, regression-tested);
  monologue renders no longer require/construct an unused speaker profile.
- Test suite: 499 passing tests after CUDA, workspace-persistence, and guided-onboarding coverage.
- Test suite: 454 passing tests on the `feature/voice-craft-and-

  recording-studio` tree (416 on `main`) including the new assembly/
  blend/monologue/pacing/parity/recorder coverage.
- Voice-craft + Recording Studio (branch
  `feature/voice-craft-and-recording-studio`, **uncommitted pending review**):
  punctuation-aware pacing (turn pause scaled by terminal punctuation;
  chunk seams breathe ~35%, the full turn pause applies only after an
  utterance's final chunk), timbre-locked emotion (per-line emotion moves
  emphasis/pause only; temperature/CFG stay per-speaker so the voice
  character is consistent), perceptual voice sliders (Voice Lock, Emotional
  Punch, Delivery Variety, Emotion Strength, Human Drift, Breath After This
  Speaker, Which Voice Wins), and the Custom Voice Recording Studio window
  (teleprompter fed from `Input/`, mic + supported-sample-rate pickers,
  auto-incrementing `Seashell_No_x.wav` saving into `Seashells/`, live
  level meter, auto-audition of each take, Listen again, Use for Speaker
  A/B quick-assign that refreshes the voice pickers on close).
- **CPU (PyTorch) ⇄ Vulkan parity** for the voice-craft decisions:
  `tests/test_backend_parity.py` plans a mixed-punctuation dialogue with
  `inference_backend=pytorch` vs `vulkan` and asserts per-utterance pauses
  and engine settings are identical on both, with timbre-lock holding (one
  temperature per speaker); the Vulkan/engine-path suites pass
  (84 tests in the batching/backend/synthesis/render-worker files). Pacing
  and emotion live in shared plan/assembly code, never in the engine call,
  so both inference backends behave identically by construction.

## Next

- Re-render `Input/What is, reality.txt` on the Vulkan backend with generic
  voices as live verification of the assembly fix (hardware-dependent).

## Noticed, not yet actioned

- **Enum-vs-widget identity audit (2026-09-11, prompted by the Cancel-button
  bug)**: the full GUI was swept for the `clickedButton() is
  StandardButton.X` bug family — no further instances. Verified correct:
  the two `result == StandardButton.Cancel` checks after `exec()` (exec
  returns the enum, so that comparison is valid), `QMessageBox.question`
  result checks, `_confirm_delete`'s `result == QMessageBox.Ok`, the
  `clicked is fix_button` widget-vs-widget comparison (valid — both are
  the same QPushButton instance), `checkState(0) == Qt.Checked`, the
  `EndOfMedia` `getattr(status, "EndOfMedia")` pattern in both playback
  handlers, and the wizards' signal-based button wiring (no identity
  comparisons at all). One fragile pattern exists only in tests
  (`buttons()[0]` positional indexing in four transformer GUI tests —
  brittle if Qt reorders buttons, but test-only).

- **`tests/test_grammar_language_tool.py` leaks temp dirs (found
  2026-09-11 during the orphan sweep)**: each test creates its cache via
  `tempfile.mkdtemp(prefix="oracle_lt_cache_")` and never removes it —
  ~140 `oracle_lt_cache_*` directories accumulated in `/tmp` across
  sessions. Pre-existing test-code hygiene issue, not introduced by the
  transformer campaign; fix is a `tempfile.TemporaryDirectory` context or
  `shutil.rmtree` teardown (out of scope for this session's requests).

- Serpent-circle inventory scan updated (2026-09-08, in
  `~/.agents/skills/serpent-circle/`): the bloat scan and language
  histogram now honor `.gitignore` — untracked+gitignored residue
  (bytecode, `.venv/`, vendored clones) is treated as repo-declared
  retention, while tracked junk and non-ignored strays stay flagged. This
  lets the omega loop terminate **CONVERGED** instead of NO-PROGRESS on a
  fresh campaign; the change lives in the skill (outside this repo), so
  it is documented here rather than committed here.

## Deferred (intentional)

- **GUI native crash — root cause found & fixed (2026-09-08)**: kernel-log
  `python: segfault at 0 ip 0000000000000000` (execution through a
  null/corrupted function pointer → use-after-free signature). Two lifetime
  bugs in the Recording Studio (added by the voice-craft campaign) matched
  the signature exactly and were fixed:
  1. `_stop_playback` called `stop()` + `deleteLater()` on the audition
     `QMediaPlayer` from inside its own `mediaStatusChanged` handler —
     tearing the QtMultimedia FFmpeg backend down mid-emission is the
     canonical player use-after-free. Fixed: one persistent player per
     dialog (lazy, created on first audition, reused like MainWindow's
     preview player); EndOfMedia now defers the stop via a zero-timer and
     never deletes the player.
  2. The `RecordStudioWorker` QThread was `deleteLater`'d from the
     `captured`/`failed` slots (fired from `run()`'s final lines while the
     thread still exits) and `closeEvent` never joined it — destroying a
     live QThread is a hard Qt abort. Fixed: teardown moved to the
     `finished` handler; dialog close and MainWindow close now do a bounded
     `wait()` and refuse to close if the thread cannot stop.
  3. Same deferred-stop discipline applied to MainWindow's preview player:
     EndOfMedia defers the stop via a zero-timer (never stop/delete from
     inside `mediaStatusChanged`), one persistent player is reused across
     previews, and window close stops the player before teardown.
  4. A full-file audit of every QObject teardown site found one more
     in-handler hazard: `PrewarmThread` emitted `ready`/`failed` from the
     last lines of `run()` and `_handle_prewarm_ready`/`_handle_prewarm_failed`
     called `thread.deleteLater()` with no `finished` connection — the same
     QThread-destroyed-while-running race. Fixed to match every other worker
     (teardown only from `finished` → `_cleanup_prewarm_thread`).
  Regression tests: `tests/test_recording_studio.py` (EndOfMedia handler
  safety, single-player reuse, finished-based worker teardown, close waits
  for worker, MainWindow preview-player deferral and close-stop, prewarm
  teardown via finished for both ready and failed paths). The pre-voice-craft Render-click crash report remains
  separate and still blocked-on-repro; repro launcher: `bash
  /tmp/gui_crash_catcher.sh` (recreated 2026-09-11 after a machine restart
  wiped /tmp). Findings: `.serpent-circle/04-debug/root-causes.md`.
  **Update 2026-09-11: the crash is CONFIRMED LIVE** — journalctl shows four
  more null-pointer segfaults (Sep 09 x3, Sep 10 x1), one in libQt6Widgets,
  and the Sep 10 one lands ~5 s after a logged `render_click`
  (`gui_action_timing.json` entry 182). The Sep 10 session's re-render
  succeeded after relaunch, so the trigger is state-dependent
  (render-after-render / preview-playback-active are candidate conditions;
  see root-causes.md). Static re-review of the current render flow found
  all worker teardowns correct; a real repro under the catcher is still the
  designated next step.
- **audio.cpp punctuation normalization is NOT patched**: its replacement
  table (`:`→`,`, `;`→`, `, dashes, quotes) exactly mirrors the installed
  Chatterbox Python reference `punc_norm`, so diverging would reduce
  model fidelity, not improve it. "Inflection at punctuation" is canonical
  model behavior, identical on both backends.
- `turbo` variant on Vulkan (PyTorch-only by design, rejected clearly).
- AMD ROCm remains intentionally out of scope for this release: AMD systems
  use CPU or the existing Vulkan/audio.cpp path. CUDA support is now present
  for NVIDIA through the PyTorch path, including GUI/CLI pickers, persisted
  device selection, manifest round-tripping, installer runtime selection
  (`auto`, `cpu`, `cuda`), and doctor diagnostics. A live CUDA render still
  requires compatible NVIDIA hardware, driver, CUDA-enabled PyTorch wheels,
  model availability, and cannot be certified on this CPU-only CI host.
- `Seashells/` runtime caches and `Output/` renders stay gitignored;
  `.venv/` bytecode is documented retention (see `.omega/CHANGELIST.md`).
- Recording Studio extras (deferred): a waveform/trim editor for cropping a
  take before saving, punch-in/overdub re-recording from a chosen prompt
  line while keeping earlier audio, and per-Seashell metadata sidecars
  (recorded-in sample rate, speaker tag) surfaced in the voice picker.