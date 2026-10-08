# GUI Legibility & Control Pass — Design

**Date:** 2026-10-08
**Status:** approved by user (design sections 1–4 signed off individually via
interactive review; pending written-spec review gate)
**Scope:** desktop GUI (`src/the_oracle/app_gui.py` and its gui_* modules)

## Problem statement

Four coupled issues reported against the current GUI:

1. **Sections are not uniformly collapsible/resizable.** Five areas already
   use `QHSectionGroup` chrome (Shared Render Settings, Speaker A, Speaker B,
   Status/Errors, Live); the path rows, cast bar, review table, and
   extra-speakers column do not.
2. **Tooltips pop up on plain hover.** Desired behavior: tooltips are gated —
   they appear **only** while the left Ctrl key is held during hover, and this
   behavior is explicitly taught in the tutorial wizard(s).
3. **Theme text contrast.** Text is hard to read on some of the six themes,
   even though the theme system machine-checks a core pair set for WCAG AA at
   import — meaning real rendered text is escaping the checked set.
4. **Overlapping text.** Unspecified "weird overlapping text issues" — to be
   found systematically (user has no specific sightings to contribute).

## Approved decisions (from interactive review)

| Question | Decision |
|---|---|
| Tooltip gate strictness | **Gate everything**: no tooltip on plain hover anywhere (widgets, menus, sliders, headers). Wizard step explanations are exempt (tour-driven, not hover-driven). |
| Section scope | **Wrap remaining areas**: path rows + cast bar, review table, extra-speakers column get `QHSectionGroup` chrome. The one-line Analyze/Render button row stays bare (collapsing would hide primary actions). |
| Contrast method | **Audit all 6 themes myself**: static pair audit of the stylesheet builder + extended import-time checker + offscreen verification. |
| Overlap bugs | **I hunt them all**: reproduce offscreen across themes/sizes, root-cause, fix with regression tests. |
| Wizard target | **Both tours**: Inference Setup & Tour and Recording Studio setup guide. |
| Sequencing | **One design, phased build**: contrast first (style changes ripple), then gating, then chrome, then overlap; checkpoint after each phase. |

## Approach selection (alternatives considered)

**Tooltip gating** — chosen: extend the existing `CtrlHoverHelp` event filter
to swallow `ToolTip` events unconditionally. Rejected: (B) stripping every
`setToolTip` call (massive diff, loses registry fallback text), (C) per-widget
filters (brittle with dynamic widgets).

**Contrast audit** — chosen: enumerate every (fg, bg) pair the stylesheet
builder actually emits, machine-check all of them at import, fix failures by
token tuning, then verify offscreen. Rejected: (B) screenshot pixel-sampling
as primary method (noisy, finds symptoms not tokens — kept only as the
verification step), (C) raising thresholds on the existing 10-pair checker
(failing text isn't in the checked set).

**Section chrome** — no alternatives pursued: reuse `QHSectionGroup` (collapse
toggle + Section Size slider + persistence already exist). No new container
class.

---

## Section 1: Tooltip strict gate + wizard explanation

### Gate mechanics

`CtrlHoverHelp.eventFilter` (`gui_tooltips.py`) currently swallows Qt's
`ToolTip` event only while Ctrl is down. Change: swallow `ToolTip` events
**unconditionally** at the QApplication level. Qt's delayed native tooltip can
then never appear on plain hover — widgets, menu items, sliders, table
headers, combo boxes. The only remaining tooltip render path is our
`QToolTip.showText`, driven by the existing Ctrl+hover poller (90 ms poll,
widget-under-cursor resolution, registry-first then `widget.toolTip()`
fallback).

### Discoverability coverage pass

`_description_for` already falls back to `widget.toolTip()`, so all existing
`setToolTip(...)` calls (~68 sites across 10 files) automatically become
Ctrl+hover content. A coverage pass adds registrations for interactive
controls that currently have neither a curated description nor a tooltip:
Analyze/Render buttons, path fields (Input/Output/Filename), the review
table, and the section chrome controls (the latter two may rely on their
existing tooltips where present).

### Exemptions

- Both wizards' step explanations call `QToolTip.showText` directly and are
  driven by the tour's Continue flow, not hover — they render regardless of
  Ctrl.
- Modal explanation popups (cast dialog onboarding text, etc.) are unaffected.

### Wizard text

- **Inference Setup & Tour**: a new dedicated step at the **front** of the
  dependency-ordered tour: tooltips are gated; nothing pops up on plain
  hover; hold **left Ctrl** and hover any control, label, or menu item to see
  what it does.
- **Recording Studio guide**: the same explanation at the front of its staged
  text.

### Tests

- `tests/test_gui_tooltips.py`: (1) `ToolTip` event with no Ctrl held is
  swallowed (filter returns True; `QToolTip.showText` not called);
  (2) Ctrl held → description shows; (3) existing registry/fallback behavior
  unchanged.
- `tests/test_inference_wizard.py` / `tests/test_recording_wizard.py`: pin
  the explanation text and its position at the front of the tour.

---

## Section 2: Section chrome for the bare areas

Three new chrome-wrapped areas, all `QHSectionGroup` instances:

1. **"Script & Cast"** — the Input/Output/Output-Filename grid plus the cast
   bar (both currently loose rows at the top of the left column). A section
   slider only works inside a `QSplitter`, so the left column's top becomes a
   new vertical `_paths_splitter`: pane 0 = the new section, pane 1 = a
   container holding the existing settings-row splitter, actions row, and
   lower splitter. The slider redistributes the paths strip's height against
   everything below — mirroring how Status/Errors shares `_lower_splitter`
   with the table. Collapse clips to the header, same as every section.
2. **"Review"** — the existing `QTableWidget` wrapped as pane 0 of
   `_lower_splitter` (Status/Errors stays pane 1). Collapse clips to header
   (mechanism already proven: Status's 70px-floor child collapses fine). The
   table's 140px minimum height is preserved while expanded.
3. **"Extra Voices"** — `extra_speaker_scroll` wrapped before insertion into
   `_sections_splitter` (pane 3). Hidden when the cast is ≤ 2 speakers; the
   slider's visible-pane-only redistribution logic already handles hidden
   panes.

### Persistence

- New registry keys: `paths`, `table`, `extra_voices` — they flow through the
  existing `_persist_workspace_layout` / `_apply_workspace_layout`.
- The new `_paths_splitter` **must** be added to the splitter list those
  methods snapshot (design-level requirement — it's how splitter shares
  survive restart).
- Older settings files lack the new keys; the apply path uses `data.get(...)`
  so missing keys are tolerated (existing behavior).

### Explicitly out of scope

The Analyze/Render button row stays bare.

### Tests

- Each new area is a `QHSectionGroup` and is registered in
  `_section_registry`.
- Persistence round-trip covers the new keys and the new splitter (save →
  fresh settings load → restored).
- Collapsing the table clips it to the header.
- Hidden Extra Voices pane does not break slider redistribution.

---

## Section 3: Theme contrast audit + fixes

### Root-cause frame

`certify_theme` currently validates ~10 token pairs
(`text`/`text_muted`/`text_disabled`/`accent_text`/`selection_text`/
`success`/`warning`/`danger` against `panel`/`bg`/`accent`/`selection_bg`).
Unreadable text comes from pairs **outside** that set. The stylesheet builder
(`render_theme`, ~300 lines) also emits colors for: section titles, table
headers and zebra cells (on `panel_alt`), placeholder text, `QToolTip`
fg/bg, button text on `secondary`/`danger`/`warning` fills, disabled button
text, menu bar and menu hover items, combo popup lists, slider/splitter
handles, link-style accents, and status-panel text on `panel_alt`.

### Method

1. **Inventory pass** — enumerate every color reference `render_theme`
   actually emits, grouped as (foreground token, background/surface token,
   role) triples. Deliverable: table of all used pairs vs. the checked set.
2. **Extend the checker** — `certify_theme` checks *all* used pairs at the
   existing thresholds (WCAG AA 4.5:1 text; 3.0:1 for large/disabled).
   Failing pairs **raise at import**, preserving today's guarantee but for
   every pair.
3. **Fix failing themes by token tuning** — adjust offending tokens in
   `THEMES` until all six themes certify. Constraint: theme identity
   survives — fonts, radius, structure untouched; color edits stay within the
   theme's aesthetic, and any change bigger than a shade is flagged to the
   user before it lands.
4. **Offscreen verification** — render `MainWindow` per theme offscreen
   (existing offscreen GUI-test infrastructure), grab the window, spot-check
   the worst surfaces (tooltips, tables, titles) by sampling
   text-vs-background pixels — this covers widget-specific QSS overrides and
   palette fallbacks the static checker can't see.
5. **Regression pin** — a test asserting `len(checked_pairs) ==
   len(used_pairs)`, so future builder code introducing a new unchecked color
   pair fails the suite until the checker knows it.

---

## Section 4: Overlapping-text hunt + fixes

Method follows systematic-debugging phases 1–4 (root cause before fixes).

1. **Reproduce** — offscreen harness rendering `MainWindow` across the
   matrix: 6 themes × window geometries (minimum, default, maximized) ×
   section states (all expanded, all collapsed, mixed) × populated review
   table (existing geometry-sweep test already populates one).
2. **Detect** — programmatic overlap checks:
   - `QFontMetrics` elision vs. painted rect for key labels (section titles
     vs. the header's fixed right-side reservation — long titles can run
     under the toggle/slider chrome);
   - child-widget rect intersections on known text surfaces (path-row
     labels/buttons, cast summary vs. Manage-cast button, table header
     labels vs. resize handles, wizard highlight popup vs. its host);
   - grab-pixel inspection of suspects for confirmation.
3. **Root-cause** — expected families: fixed-width reservations in
   `gui_sections._relayout_header` ignoring title length; elided labels
   colliding at narrow pane widths; wizard `QToolTip.showText` popup anchored
   at `height()+6` near the window edge (renders offscreen/clipped);
   per-theme `font_display` metrics shifting widths.
4. **Fix at the cause** — e.g. elide titles with `...` before the reserved
   zone; title-aware reservations; constrained popup anchors. Failing-test
   first; each fix pinned by a regression test in the relevant test file.
5. **Re-run the full matrix** at the end: suite + all 6 themes × geometry
   sweep clean.

### Fix policy / honest limits

If a finding is a layout impossibility rather than a bug (two texts must
touch at minimum window size), the fix is to raise a floor or elide — never
to hide the finding. Anything not fully fixed is reported explicitly.

---

## Sequencing & checkpoints

| Phase | Content | Checkpoint (must pass before next phase) |
|---|---|---|
| 1 | Contrast audit + fixes (Section 3) | all 6 themes certify + offscreen spot-check |
| 2 | Tooltip strict gate + wizard text (Section 1) | focused tooltip/wizard tests + full suite |
| 3 | Section chrome (Section 2) | GUI tests + persistence round-trip + full suite |
| 4 | Overlap hunt + fixes (Section 4) | matrix sweep + full suite |

- Full suite = `ORACLE_FAIL_ON_SKIP=1` pytest run (~1555 tests at time of
  writing, ~5 min; LanguageTool `__del__` stderr noise on teardown is
  expected and harmless).
- **No commits until the user approves this spec and the implementation
  plan.** Working tree currently carries unrelated crash-eradication changes
  (`STATE.md`, `scripts/crash_hunt.py`, `src/the_oracle/app_gui.py`,
  `src/the_oracle/gui_crash.py`, `scripts/doctor.py`, two crash tests, one
  findings doc) — these must not be disturbed or staged with this work.

## Testing strategy (cross-cutting)

- TDD via `test-driven-development` skill: failing test before each fix.
- All new GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen` convention
  already used by the suite).
- Theme changes verified by both the import-time checker (static) and the
  offscreen grab spot-check (rendered).
- Existing tests that pin current tooltip/section/theme behavior are
  updated only where the approved behavior change requires it, with the
  change explained in the test's comment.
