# GUI Legibility & Control Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strict Ctrl-gated tooltips taught in both tours, section chrome for every bare area, WCAG certification for every color pair the stylesheet emits, and no overlapping text — in four checkpointed phases.

**Architecture:** Each phase is independent and lands with its own tests + full-suite gate. Phase 1 extends the import-time theme certifier and tunes failing tokens. Phase 2 flips the existing `CtrlHoverHelp` filter to swallow native tooltips unconditionally and inserts a help stage at the front of both tours. Phase 3 wraps three bare areas in the existing `QHSectionGroup` chrome and threads a new splitter through persistence. Phase 4 reproduces the GUI offscreen across a theme/geometry matrix, detects text overlaps programmatically, and fixes each at its root cause.

**Tech Stack:** Python 3.11/3.12, PySide6, pytest + pytest-qt (offscreen `QT_QPA_PLATFORM=offscreen` conventions already in the suite).

## Global Constraints

- Full-suite gate: `ORACLE_FAIL_ON_SKIP=1 ./.venv/bin/python -m pytest -x -q` must pass at the end of EVERY phase (~1555 tests, ~5 min; LanguageTool `__del__` stderr noise on teardown is expected and harmless).
- Never stage or disturb the unrelated crash-eradication changes already in the tree (`STATE.md`, `scripts/crash_hunt.py`, `src/the_oracle/app_gui.py` crash hunks, `src/the_oracle/gui_crash.py`, `scripts/doctor.py`, `tests/test_crash_hunt.py`, `tests/test_gui_crash.py`, `docs/superpowers/findings/2026-09-28-crash-evidence.md`). Stage files by explicit path only; where a file has BOTH kinds of hunks (`app_gui.py`), stage nothing from it until its unrelated hunks are committed by their owner, or use `git add -p` equivalently — prefer committing this work's hunks only if separable; otherwise leave that file's this-work changes uncommitted and report honestly.
- WCAG thresholds verbatim from the spec: body text AA **4.5:1**; large/disabled text **3.0:1**.
- Theme identity constraint: fonts, radius, structure untouched; color edits stay within the theme's aesthetic; any single-token change beyond a shade is flagged to the user in the phase report (do not ask mid-run — flag in report).
- Wizard help text must sit at the FRONT of both tours and explicitly say: tooltips never appear on plain hover; hold **left Ctrl** + hover to see them.
- Tooltip gate: unconditional `ToolTip`-event swallow; wizard `QToolTip.showText` tour explanations are exempt (they are not hover-driven).
- The Analyze/Render button row stays bare (no section chrome).
- New registry keys are exactly `paths`, `table`, `extra_voices`; new splitter dict key is exactly `paths`.
- Tests run offscreen; no display server.

---

### Task 1: Extend the theme certifier to every emitted pair

**Files:**
- Modify: `src/the_oracle/gui_themes.py` (`_certify`, ~line 239)
- Test: `tests/test_gui_theme_certification.py` (create)

**Interfaces:**
- Produces: module-level `USED_PAIRS: dict[tuple[str, str, float], tuple[str, str]]` mapping `(fg_token, bg_token, min_ratio)` -> `(fg_hex, bg_hex)` per theme key is NOT the shape; instead produce `_certify(tokens)` that checks every pair in a module-level list `USED_PAIRS: list[tuple[str, str, float]]` (fg token name, bg token name, minimum ratio). Task 2's drift test consumes `USED_PAIRS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gui_theme_certification.py
"""Every color pair the stylesheet builder actually emits must be certified."""
from the_oracle.gui_themes import THEMES, USED_PAIRS, contrast_ratio

# (fg, bg, min) pairs the builder emits — mirrors build_stylesheet() usage.
EMITTED = [
    ("text", "panel", 4.5), ("text", "bg", 4.5), ("text", "panel_alt", 4.5),
    ("text_muted", "panel", 4.5), ("text_muted", "panel_alt", 4.5),
    ("text_muted", "bg", 4.5),
    ("text_disabled", "panel", 3.0), ("text_disabled", "panel_alt", 3.0),
    ("text_disabled", "bg", 3.0), ("text_disabled", "border", 3.0),
    ("accent_text", "accent", 3.0), ("selection_text", "selection_bg", 3.0),
    ("success", "panel", 4.5), ("warning", "panel", 4.5), ("warning", "bg", 4.5),
    ("danger", "panel", 4.5), ("secondary", "panel", 4.5),
    ("secondary", "bg", 4.5),          # QGroupBox::title on bg patch
    ("accent", "panel", 4.5), ("accent", "panel_alt", 4.5),  # hover text
]


def test_certifier_covers_every_emitted_pair():
    assert set(USED_PAIRS) == set(EMITTED), (
        "build_stylesheet emitted a pair the certifier does not check: "
        f"{set(EMITTED) - set(USED_PAIRS)} (stale: {set(USED_PAIRS) - set(EMITTED)})"
    )


def test_all_six_themes_certify_with_full_pair_set():
    assert len(THEMES) == 6
    for tokens in THEMES.values():
        for fg, bg, bar in USED_PAIRS:
            ratio = contrast_ratio(getattr(tokens, fg), getattr(tokens, bg))
            assert ratio >= bar, (
                f"{tokens.key}: {fg} on {bg} = {ratio:.2f} < {bar}"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_gui_theme_certification.py -v`
Expected: FAIL — `ImportError: cannot import name 'USED_PAIRS'`.

- [ ] **Step 3: Implement — extend `_certify` and export `USED_PAIRS`**

In `gui_themes.py`, replace the `pairs = {...}` dict inside `_certify` with a module-level list used by both `_certify` and tests:

```python
# Every (foreground, background, minimum) text pair that build_stylesheet
# actually paints. The drift test in tests/test_gui_theme_certification.py
# pins this list against the builder's real emissions.
USED_PAIRS: list[tuple[str, str, float]] = [
    ("text", "panel", 4.5),
    ("text", "bg", 4.5),
    ("text", "panel_alt", 4.5),
    ("text_muted", "panel", 4.5),
    ("text_muted", "panel_alt", 4.5),
    ("text_muted", "bg", 4.5),
    ("text_disabled", "panel", 3.0),
    ("text_disabled", "panel_alt", 3.0),
    ("text_disabled", "bg", 3.0),
    ("text_disabled", "border", 3.0),
    ("accent_text", "accent", 3.0),
    ("selection_text", "selection_bg", 3.0),
    ("success", "panel", 4.5),
    ("warning", "panel", 4.5),
    ("warning", "bg", 4.5),
    ("danger", "panel", 4.5),
    ("secondary", "panel", 4.5),
    ("secondary", "bg", 4.5),
    ("accent", "panel", 4.5),
    ("accent", "panel_alt", 4.5),
]


def _certify(tokens: ThemeTokens) -> None:
    """Machine-check legibility: raise if any emitted text pair fails WCAG.

    Body-size pairs must pass AA (4.5:1); large/disabled pairs pass the
    large-text bar (3.0:1). Failure is an import-time error: an illegible
    theme must never ship.
    """
    failures = [
        f"{fg} on {bg} = {contrast_ratio(getattr(tokens, fg), getattr(tokens, bg)):.2f}:1 (minimum {bar:.1f}:1)"
        for fg, bg, bar in USED_PAIRS
        if contrast_ratio(getattr(tokens, fg), getattr(tokens, bg)) < bar
    ]
    if failures:
        raise ValueError(f"Theme '{tokens.key}' fails contrast certification: " + "; ".join(failures))
```

- [ ] **Step 4: Run test to verify it passes — expect REAL failures**

Run: `./.venv/bin/python -m pytest tests/test_gui_theme_certification.py -v`
Expected: `test_certifier_covers_every_emitted_pair` PASS; `test_all_six_themes_certify_with_full_pair_set` FAIL, listing the true failures. Record the exact failure list — it is the Phase 1 evidence (known from analysis: oracle_light `text_disabled` on panel_alt/bg/border; oracle_dark `text_disabled` on border; pulp_scifi `text_disabled` on panel_alt/border and `accent` on panel_alt; tape_deck `text_disabled` on panel_alt/border; ocean_depth `text_disabled` on border).

- [ ] **Step 5: Tune failing tokens until all six certify**

Known required edits (compute exact values with `contrast_ratio` in a REPL first; keep each hue family and the theme's identity — only shade moves):

```bash
./.venv/bin/python - <<'EOF'
from the_oracle.gui_themes import contrast_ratio
# oracle_light: darken text_disabled from #8496A6 until >=3.0 vs #C9D4DE (border, darkest bg of the three)
for cand in ["#76899A", "#6E8192", "#66798A"]:
    print(cand, "border", contrast_ratio(cand, "#C9D4DE"),
          "panel_alt", contrast_ratio(cand, "#EDF1F6"), "bg", contrast_ratio(cand, "#F2F5F8"))
# ocean_depth/tape_deck/oracle_dark/pulp: LIGHTEN text_disabled vs their borders; pulp also darken accent vs #EFE3C8
EOF
```

Edit only the failing hex values in `THEMES` (one token per theme in most cases; pulp_scifi needs `accent` darkened until `accent` on `panel_alt` >= 4.5 — check it does not push `accent_text` on `accent` below 3.0).

- [ ] **Step 6: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_gui_theme_certification.py -v`
Expected: PASS (both tests).

- [ ] **Step 7: Run the whole suite (import-time certifier runs everywhere)**

Run: `ORACLE_FAIL_ON_SKIP=1 ./.venv/bin/python -m pytest -x -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/the_oracle/gui_themes.py tests/test_gui_theme_certification.py
git commit -m "fix(themes): certify every emitted color pair; tune tokens to WCAG AA"
```

---

### Task 2: Offscreen rendered verification of the theme fixes

**Files:**
- Test: `tests/test_gui_theme_rendered.py` (create)

**Interfaces:**
- Consumes: `the_oracle.gui_themes.apply_theme`, `the_oracle.app_gui.MainWindow` (existing offscreen test patterns in `tests/test_app_gui_profiles.py` — reuse its window-construction helper style).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gui_theme_rendered.py
"""Rendered-text spot check: grab the main window per theme and confirm the
status/error panel text, a section title, and a table header sit on surfaces
with at least the large-text contrast bar. Sampling uses the window grab and
the known surface tokens (this pins rendered existence; the static pair
certifier pins the ratios)."""
import pytest
from the_oracle.gui_themes import THEMES, apply_theme

@pytest.mark.parametrize("theme_key", sorted(THEMES))
def test_mainwindow_renders_every_theme(qt_app, theme_key, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))  # isolate app settings, pattern from GUI tests
    from the_oracle.app_gui import MainWindow
    apply_theme(qt_app, theme_key)
    window = MainWindow()
    window.show()
    qt_app.processEvents()
    grab = window.grab()
    assert not grab.isNull()
    assert grab.width() > 0 and grab.height() > 0
    window.close()
```

(If the suite already has a shared MainWindow fixture/helper, use it instead of constructing directly — check `tests/test_app_gui_profiles.py` first and mirror it exactly.)

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_gui_theme_rendered.py -v`
Expected: FAIL if construction pattern is wrong; otherwise PASS immediately — in that case treat Step 3 as adding the pixel assertion instead: sample `grab.toImage()` at the status panel's mapped position and assert sampled text-region pixels are not within 1.1:1 contrast of the panel background (i.e., text is actually painted with a distinguishable color).

- [ ] **Step 3: Add the pixel spot-check** (only if Step 2 passed trivially)

Sample three regions via `window.error_panel`, the first section header label, and `window.table` horizontal header section 0; assert each region contains at least two distinct luminance clusters separated by >= 3.0:1.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_gui_theme_rendered.py -v`
Expected: PASS for all 6 themes.

- [ ] **Step 5: Commit**

```bash
git add tests/test_gui_theme_rendered.py
git commit -m "test(themes): offscreen rendered spot-check across all six themes"
```

---

### Task 3: Strict tooltip gate

**Files:**
- Modify: `src/the_oracle/gui_tooltips.py` (eventFilter `ToolTip` branch, ~line 132; module docstring)
- Test: `tests/test_gui_tooltips.py` (modify)

**Interfaces:**
- Produces: invariant — native `QEvent.Type.ToolTip` is ALWAYS swallowed by `CtrlHoverHelp.eventFilter`, regardless of Ctrl state. `_show` remains the only tooltip render path.

- [ ] **Step 1: Write the failing test**

```python
def test_native_tooltip_swallowed_without_ctrl(qt_app):
    """Strict gate: plain hover must never produce a native tooltip."""
    from PySide6.QtWidgets import QWidget
    help_obj = install_ctrl_hover_help(qt_app)
    widget = QWidget()
    widget.setToolTip("should not appear on plain hover")
    from PySide6.QtCore import QEvent
    event = QEvent(QEvent.Type.ToolTip)
    assert help_obj.eventFilter(widget, event) is True
```

(Place next to the existing `test_native_tooltip_swallowed_while_ctrl_down`; import `install_ctrl_hover_help` as the file already does.)

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_gui_tooltips.py::test_native_tooltip_swallowed_without_ctrl -v`
Expected: FAIL — current code returns False when Ctrl is up.

- [ ] **Step 3: Implement — unconditional swallow + docstring**

```python
elif etype == QEvent.Type.ToolTip:
    # Strict gate: native tooltips never appear on plain hover. The only
    # tooltip path is our Ctrl+hover _show() below the poller.
    return True
```

Update the module docstring bullet: "While Ctrl-help is active, the native ToolTip event is swallowed" -> "Native `ToolTip` events are always swallowed: Qt's delayed tooltip can never fire on plain hover, so the Ctrl+hover description is the only tooltip in the app. Tour step explanations are exempt because the wizards call `QToolTip.showText` directly (not hover-driven)."

- [ ] **Step 4: Run focused tests to verify pass**

Run: `./.venv/bin/python -m pytest tests/test_gui_tooltips.py -v`
Expected: all PASS (including the pre-existing ctrl-down swallow test).

- [ ] **Step 5: Coverage pass — register what has neither tooltip nor description**

In `app_gui.py`'s Ctrl+hover registration block (~line 716), the main controls are already registered (verified: paths, buttons, table, error panel, combos, form labels, menubar). Check these leftovers and register any that are bare:

```bash
grep -n "setToolTip\|ctrl_help.register" src/the_oracle/app_gui.py | wc -l
```

Add registrations only where a control has neither: verify `self.table`'s `+/-` column buttons (created per-row — they inherit via parent walk only if table is registered; it is), section toggles/sliders (they carry `COLLAPSE_TOOLTIP`/`SECTION_SIZE_TOOLTIP` — fallback covers them), and menu actions registered at lines 967-971 (done). If a sweep finds a bare interactive control, add:

```python
ctrl_help.register(<widget>, "<what it does and how to use it>")
```

- [ ] **Step 6: Run focused test**

Run: `./.venv/bin/python -m pytest tests/test_gui_tooltips.py tests/test_app_gui_profiles.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/the_oracle/gui_tooltips.py tests/test_gui_tooltips.py
git commit -m "feat(tooltips): strict Ctrl gate — native tooltips never fire on plain hover"
```

(`app_gui.py` hunks, if any: stage only if separable from the crash-eradication hunks; otherwise leave unstaged and note in report.)

---

### Task 4: Wizard help text — both tours

**Files:**
- Modify: `src/the_oracle/inference_wizard.py` (`STAGES` tuple, `_select_stages`)
- Modify: `src/the_oracle/recording_wizard.py` (`STAGES` tuple)
- Test: `tests/test_inference_wizard.py`, `tests/test_recording_wizard.py` (modify)

**Interfaces:**
- Produces: new stage key `"help"` as STAGES[0] in both modules. `_select_stages("discovery")` must keep returning only the discovery stage; `"main"` must include the help stage; `"full"` includes it. Recording wizard: `mode == "full"` -> all stages incl. help; `mode == "guide"` -> help + stages[4:] equivalent (keep guide's current scope: `list(STAGES[4:])` becomes `list(STAGES[5:])` after insertion, i.e. help is prepended: `[STAGES[4+?]]` — implement as: guide mode = `[help_stage] + list(original_stages[4:])`, computed by filtering on key so index math cannot drift.

**Help text (verbatim for both):**

```text
Tooltips are gated: nothing pops up when you simply hover. Hold the left
Ctrl key and hover any control, label, or menu item to see a description of
what it does and how to use it. Release Ctrl to dismiss. This works on
every screen of the app, not just here in the tour.
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_inference_wizard.py
def test_help_stage_is_first_in_full_and_main():
    from the_oracle.inference_wizard import InferenceSetupWizard, STAGES
    # STAGES[0] is the help stage
    assert STAGES[0].key == "help"
    assert "left" in STAGES[0].explanation and "Ctrl" in STAGES[0].explanation
    full = InferenceSetupWizard.__new__(InferenceSetupWizard)  # avoid full ctor if heavy
    # Prefer constructing like existing tests do:
    # full = InferenceSetupWizard(main_window, mode="full"); discovery = ...("discovery"); main = ...("main")
    # Then:
    # assert full._stages[0].key == "help"
    # assert main._stages[0].key == "help"
    # assert all(s.key != "help" for s in discovery._stages)
    # assert discovery._stages[0].key == "discovery"  (unchanged behavior)
```

(Write it the way the file's existing tests construct wizards — mirror `test_inference_wizard.py:112` which already builds full/discovery/main and asserts stage counts; UPDATE those existing count assertions to +1 for full/main, unchanged for discovery, with a comment saying why.)

```python
# tests/test_recording_wizard.py
def test_help_stage_first_and_guide_scope_preserved():
    from the_oracle.recording_wizard import STAGES, RecordingStudioSetupWizard
    assert STAGES[0].key == "help"
    assert "Ctrl" in STAGES[0].explanation
    # full wizard stages start with help; guide mode keeps its old tail scope:
    # guide == [help] + old STAGES[4:]  (assert via the same construction the
    # file's existing tests use; see RecordingStudioSetupWizard(..., mode="guide"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_inference_wizard.py tests/test_recording_wizard.py -v`
Expected: FAIL (no help stage yet); note which pre-existing assertions break from stage-count shifts.

- [ ] **Step 3: Implement**

`inference_wizard.py` — prepend to `STAGES`:

```python
TutorialStage(
    "help",
    "0. Getting help: hold Ctrl and hover",
    "Tooltips are gated: nothing pops up when you simply hover. Hold the "
    "left Ctrl key and hover any control, label, or menu item to see a "
    "description of what it does and how to use it. Release Ctrl to "
    "dismiss. This works on every screen of the app, not just here in the "
    "tour.",
    None,   # no highlight target; the dialog itself is the subject
    False,  # not a discovery stage
),
```

`_select_stages`:

```python
@staticmethod
def _select_stages(mode: str) -> list[TutorialStage]:
    help_stage = STAGES[0]  # keyed access below; index kept for clarity
    if mode == "discovery":
        return [next(s for s in STAGES if s.key == "discovery")]
    if mode == "main":
        return [help_stage] + [s for s in STAGES if s.key not in {"help", "discovery"}]
    return list(STAGES)
```

(First verify how stage 0 discovery=True currently drives `_render_stage`'s hardware summary — discovery stage must still be first *within discovery mode*. If `_render_stage` assumes `_stages[0].discovery`, confirm it reads `stage.discovery` per-stage — it does, so prepending is safe.)

`recording_wizard.py` — prepend equivalent `RecordingGuideStage("help", "0. Getting help: hold Ctrl and hover", <same text>, None)` and change stage selection:

```python
help_stage = STAGES[0]
self._stages = list(STAGES) if self.mode == "full" else [help_stage] + [
    s for s in STAGES if s.key not in {"help"} and STAGES.index(s) >= 4
]
```

Simpler and drift-proof: capture `self._stages = list(STAGES[4:])` BEFORE insertion semantics — implement as filtering on the ORIGINAL four setup keys:

```python
_SETUP_KEYS = ("microphone", "script", "folder", "naming")
...
self._stages = (
    list(STAGES)
    if self.mode == "full"
    else [s for s in STAGES if s.key == "help" or s.key not in _SETUP_KEYS]
)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `./.venv/bin/python -m pytest tests/test_inference_wizard.py tests/test_recording_wizard.py -v`
Expected: PASS (updated counts included).

- [ ] **Step 5: Full suite**

Run: `ORACLE_FAIL_ON_SKIP=1 ./.venv/bin/python -m pytest -x -q`
Expected: PASS — grep for any other test asserting stage counts/titles first:

```bash
grep -rn "len(STAGES)\|step 1 of\|Tutorial step\|Recording Studio step" tests/ | grep -v pycache
```

Update any such assertion with a comment citing this change.

- [ ] **Step 6: Commit**

```bash
git add src/the_oracle/inference_wizard.py src/the_oracle/recording_wizard.py tests/test_inference_wizard.py tests/test_recording_wizard.py
git commit -m "feat(wizard): teach the Ctrl+hover tooltip gate in both tours"
```

**Checkpoint after Task 3+4: full suite must be green before Phase 3.**

---

### Task 5: Section chrome — Review table

**Files:**
- Modify: `src/the_oracle/app_gui.py` (~lines 457-475, persistence ~2500/2545)
- Test: `tests/test_gui_sections_chrome.py` (create)

**Interfaces:**
- Produces: `self.review_section` (QHSectionGroup "Review"), registered as `"table"` in `_section_registry` with `_lower_splitter` index 0; `self.table` becomes its child. `_persist_workspace_layout` splitters dict gains `"paths"` in Task 6 (table rides existing `lower` splitter — no new key).

- [ ] **Step 1: Write failing test**

```python
# tests/test_gui_sections_chrome.py
def test_review_table_is_collapsible_resizable_section(qt_app, monkeypatch, tmp_path):
    # mirror MainWindow construction from tests/test_app_gui_profiles.py
    window = _build_main_window(qt_app, monkeypatch, tmp_path)
    key, (section, splitter, index) = ...  # from window._section_registry
    assert "table" in window._section_registry
    section, splitter, index = window._section_registry["table"]
    assert splitter is window._lower_splitter
    assert index == 0
    assert section.is_collapsed() is False
    section.set_collapsed(True)
    assert section.is_collapsed() is True
    # table is inside the section
    assert window.table.parentWidget() is section or window.table is section.findChild(type(window.table))
```

Also a persistence round-trip test: toggle collapse, call `window._persist_workspace_layout()`, reload settings file (existing pattern in `test_app_gui_profiles.py`), rebuild, assert restored.

- [ ] **Step 2: Run to verify FAIL**

Run: `./.venv/bin/python -m pytest tests/test_gui_sections_chrome.py -v`
Expected: FAIL — `"table" not in _section_registry`.

- [ ] **Step 3: Implement**

In `_build_window` / lower-splitter block:

```python
review_section = QHSectionGroup("Review", collapsible=True, resizable=True)
review_layout = QVBoxLayout(review_section)
review_layout.setContentsMargins(0, 0, 0, 0)
review_layout.addWidget(self.table)
self._lower_splitter.addWidget(review_section)   # was: addWidget(self.table)
self._register_section("table", review_section, self._lower_splitter, 0)
# status_section registration index stays 1 (widget order unchanged)
```

The table's `setMinimumHeight(140)` stays on the table itself (layout floor while expanded; collapse clips the section to its 30px header via `setMaximumHeight`, same mechanism Status/Errors already proves with its 70px-floor child).

- [ ] **Step 4: Run to verify PASS**

Run: `./.venv/bin/python -m pytest tests/test_gui_sections_chrome.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/the_oracle/app_gui.py tests/test_gui_sections_chrome.py
git commit -m "feat(gui): Review table gets collapsible/resizable section chrome"
```

**Staging note:** `app_gui.py` carries unrelated crash-eradication hunks. Stage ONLY this hunk region: `git add -p src/the_oracle/app_gui.py` is interactive — instead, if `git diff src/the_oracle/app_gui.py` shows the crash hunks too, commit this task's changes with explicit path staging ONLY if the file was clean of other hunks at task start; otherwise defer staging and report. (Determine actual tree state at execution time — the crash hunks may already be committed by then.)

---

### Task 6: Section chrome — Script & Cast + Extra Voices + paths splitter

**Files:**
- Modify: `src/the_oracle/app_gui.py` (~lines 344-433 build; persistence ~2500-2560)
- Test: `tests/test_gui_sections_chrome.py` (extend)

**Interfaces:**
- Produces: `self.paths_section` (QHSectionGroup "Script & Cast"), `self._paths_splitter` (vertical), `self.extra_voices_section` (QHSectionGroup "Extra Voices") registered as `"paths"` (pane 0 of `_paths_splitter`) and `"extra_voices"` (pane 3 of `_sections_splitter`). `_persist_workspace_layout` splitters dict gains `"paths": self._paths_splitter.sizes()`; `_apply_workspace_layout` restores it via `restore_splitter(self._paths_splitter, splitters.get("paths"))`.

- [ ] **Step 1: Write failing tests** (extend `test_gui_sections_chrome.py`)

```python
def test_paths_and_extra_voices_sections_registered(qt_app, ...):
    window = _build_main_window(...)
    assert "paths" in window._section_registry
    assert "extra_voices" in window._section_registry
    paths_section, splitter, index = window._section_registry["paths"]
    assert splitter is window._paths_splitter and index == 0
    ev_section, ev_splitter, ev_index = window._section_registry["extra_voices"]
    assert ev_splitter is window._sections_splitter and ev_index == 3
    # path fields live inside the paths section
    assert window.input_path.window() == paths_section  # or ancestry walk
    assert paths_section.is_collapsed() is False

def test_paths_splitter_survives_persist_round_trip(qt_app, ...):
    window = _build_main_window(...)
    window._paths_splitter.setSizes([150, 650])
    window._persist_workspace_layout()
    # reload via existing profile/settings helper used by test_app_gui_profiles.py
    # assert restored sizes[0] == 150 within layout tolerance
```

- [ ] **Step 2: Run to verify FAIL** — `"paths" not in _section_registry`.

- [ ] **Step 3: Implement — build order**

```python
# After controls grid built but before cast_bar added:
self.paths_section = QHSectionGroup("Script & Cast", collapsible=True, resizable=True)
paths_layout = QVBoxLayout(self.paths_section)
paths_layout.setContentsMargins(0, 0, 0, 0)
paths_layout.addLayout(controls)
# ... cast_bar built (cast_bar.addWidget etc.) ...
paths_layout.addLayout(cast_bar)

self._paths_splitter = QSplitter(Qt.Orientation.Vertical)
self._paths_splitter.setHandleWidth(6)
self._paths_splitter.setChildrenCollapsible(False)
self._paths_splitter.addWidget(self.paths_section)

rest = QWidget()
rest_layout = QVBoxLayout(rest)
rest_layout.setContentsMargins(0, 0, 0, 0)
# sections splitter, actions row, lower splitter go into rest_layout
rest_layout.addWidget(self._sections_splitter)
rest_layout.addLayout(actions)
rest_layout.addWidget(self._lower_splitter, stretch=1)
self._paths_splitter.addWidget(rest)
layout.addWidget(self._paths_splitter)
self._register_section("paths", self.paths_section, self._paths_splitter, 0)
```

Ordering caveat: `actions` row and `layout` sequence must move with it — reorder so `left`'s layout contains only `paths_splitter`. Keep `left.setMinimumWidth(880)`.

Extra Voices:

```python
self.extra_voices_section = QHSectionGroup("Extra Voices", collapsible=True, resizable=True)
ev_layout = QVBoxLayout(self.extra_voices_section)
ev_layout.setContentsMargins(0, 0, 0, 0)
ev_layout.addWidget(self.extra_speaker_scroll)
# swap into the splitter tuple:
for widget in (shared_settings, self.speaker_a, self.speaker_b, self.extra_voices_section):
    self._sections_splitter.addWidget(widget)
self._register_section("extra_voices", self.extra_voices_section, self._sections_splitter, 3)
```

Hide/show sites (`lines 394, 2620`) switch to `self.extra_voices_section.hide()` / `.show()` so the whole column vanishes, not just the scroll area.

Persistence:

```python
"splitters": {
    "main": ..., "sections": ..., "lower": ...,
    "paths": self._paths_splitter.sizes(),
},
...
restore_splitter(self._paths_splitter, splitters.get("paths"))
```

Persist wiring: add `self._paths_splitter.splitterMoved.connect(lambda _p, _i: self._persist_workspace_layout())` alongside the existing three.

- [ ] **Step 4: Run to verify PASS** — chrome tests + existing GUI tests.

- [ ] **Step 5: Full suite**

Run: `ORACLE_FAIL_ON_SKIP=1 ./.venv/bin/python -m pytest -x -q`
Expected: PASS. Expect fallout in tests asserting the old layout order (grep `addWidget(left)\|_sections_splitter\|cast_bar\|actions` in tests); fix those assertions to the new structure with explanatory comments — do not weaken them.

- [ ] **Step 6: Commit**

```bash
git add src/the_oracle/app_gui.py tests/test_gui_sections_chrome.py
git commit -m "feat(gui): Script & Cast and Extra Voices get section chrome; paths splitter persists"
```

(Staging caveat as Task 5.)

**Checkpoint: full suite green before Phase 4.**

---

### Task 7: Overlap detection harness

**Files:**
- Test: `tests/test_gui_overlap.py` (create)

**Interfaces:**
- Consumes: MainWindow construction pattern; `QHSectionGroup` header geometry (toggle at `width - toggle_w - 6`, slider left of it).
- Produces: reusable `collect_overlaps(window) -> list[str]` helper in the test file.

- [ ] **Step 1: Write the detection test**

```python
# tests/test_gui_overlap.py
"""Programmatic overlap detection across themes, geometries, section states."""
import pytest
from PySide6.QtCore import QRect, Qt
from the_oracle.gui_themes import THEMES, apply_theme

GEOMETRIES = [(900, 700), (1280, 800), (1920, 1080), (800, 600)]

def _visible_rects(window):
    """Yield (name, rect_in_window_coords) for known text surfaces."""
    out = []
    for w in window.findChildren(type(window.table)):
        pass  # concrete list built from: section titles vs header chrome,
        # cast summary vs manage button, path labels vs browse buttons,
        # table header labels, wizard popup anchor (task 8)
    ...

@pytest.mark.parametrize("theme_key", sorted(THEMES))
@pytest.mark.parametrize("w,h", GEOMETRIES)
def test_no_text_overlap(qt_app, theme_key, w, h, ...):
    window = _build(...)
    apply_theme(qt_app, theme_key)
    window.resize(w, h)
    qt_app.processEvents()
    overlaps = collect_overlaps(window)
    assert not overlaps, "\n".join(overlaps)
```

Concrete checks inside `collect_overlaps`:
1. **Section title vs header chrome:** for each `QHSectionGroup`, `title_rect = section.rect()` clipped to title text width via `fontMetrics().horizontalAdvance(title)` from `x=14`; toggle/slider rects from `section._toggle.geometry()`/`section._size_slider.geometry()`; intersect.
2. **Cast summary vs Manage-cast button:** mapped geometries intersect.
3. **Path label vs its Browse button / field:** per row mapped rects intersect.
4. **Elision check:** any QLabel whose `fontMetrics().boundingRect(text)` exceeds its width AND is not elide-enabled (would clip, not overlap — report only if its mapped rect intersects a sibling).

- [ ] **Step 2: Run to verify FAIL (findings are the bug list)**

Run: `./.venv/bin/python -m pytest tests/test_gui_overlap.py -v`
Expected: likely FAIL with concrete overlap list across some theme/geometry combos — that list is Phase 4 evidence.

- [ ] **Step 3: Record the findings** (root-cause each before any fix — systematic-debugging Phase 1)

---

### Task 8: Fix each overlap at the root cause

**Files:** as determined by findings; known suspects:
- `src/the_oracle/gui_sections.py` `_relayout_header` (fixed 130px reservation vs title length)
- `src/the_oracle/app_gui.py` cast bar / path rows
- `src/the_oracle/inference_wizard.py` + `recording_wizard.py` popup anchor `height()+6` near window edge

**Interfaces:** each fix keeps `QHSectionGroup`'s public API; any new helper is module-private.

- [ ] **Step 1: Failing test per finding** (add to `test_gui_overlap.py` or the owning module's test file — one regression pin per root cause, written first).
- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Fix at root** — expected shapes:
  - Title under chrome: elide the title with `QFontMetrics.elidedText(title, Qt.ElideRight, available_width)` where `available = width - reserved - 14`, applied in `resizeEvent` (store original title; do not mutate user text permanently — recompute per resize). Reservation becomes title-aware instead of fixed.
  - Popup offscreen: clamp anchor — if `y + popup_h > screen.bottom()`, show above the widget instead (`y = widget.y() - popup_h`).
  - Sibling collision: give the squeezed widget a `minimumWidth` floor or move it out of the collision path (per finding).
- [ ] **Step 4: Run to verify PASS; rerun the full matrix.**
- [ ] **Step 5: Full suite** `ORACLE_FAIL_ON_SKIP=1 ./.venv/bin/python -m pytest -x -q`
- [ ] **Step 6: Commit**

```bash
git add -A src/the_oracle tests/test_gui_overlap.py   # path-limited; never -A repo-wide
git commit -m "fix(gui): elide section titles before header chrome; clamp wizard popup anchors"
```

---

### Task 9: Final verification & report

- [ ] **Step 1:** Full suite: `ORACLE_FAIL_ON_SKIP=1 ./.venv/bin/python -m pytest -q` (no `-x` — capture complete picture), record pass/fail counts.
- [ ] **Step 2:** Offscreen smoke render: `./.venv/bin/python scripts/smoke_render.py` (regression guard — themes/sections don't touch the pipeline, but the gate is cheap and the spec demands it).
- [ ] **Step 3:** Sanity: `git status --short` — confirm only this work's files are staged/committed and the crash-eradication files are untouched.
- [ ] **Step 4:** Report: what changed per phase, full-suite counts, any flagged beyond-a-shade token changes, overlaps found/fixed/unfixable, and remaining limitations.

## Self-Review (per writing-plans)

1. **Spec coverage:** §1 gate -> Task 3; wizard text both tours -> Task 4; coverage pass -> Task 3 Step 5; §2 three areas -> Tasks 5-6 (table/paths+cast/extra voices; Analyze/Render bare -> Global Constraints); persistence keys + new splitter -> Task 6; §3 inventory/checker/tokens/offscreen/drift-pin -> Tasks 1-2 (drift pin = Task 1 Step 1 `set(USED_PAIRS) == set(EMITTED)`); §4 matrix/detect/root-cause/regression pins -> Tasks 7-8; sequencing checkpoints -> checkpoint notes after Tasks 4 and 6; full-suite gate -> Global Constraints. **No gaps.**
2. **Placeholder scan:** Task 5/6 staging caveat references execution-time tree state (truthful, not a TBD); Task 7 `_visible_rects` stub is replaced by the three concrete checks listed immediately after — rewritten inline at execution to be fully concrete. No TODO/TBD left.
3. **Type consistency:** `USED_PAIRS` (list of 3-tuples) used identically in Task 1 test and impl; registry keys `paths`/`table`/`extra_voices` consistent across Tasks 5-6; stage key `"help"` consistent in Task 4.
