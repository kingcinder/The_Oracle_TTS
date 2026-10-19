# Removing the Strong-Copyleft Dependencies — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `mutagen`, `language-tool-python`, and `pykakasi` from the 1.4.1 runtime so the commercial gate has no strong-copyleft blockers, without breaking community rendering.

**Architecture:** Replace the only real `mutagen` use (FLAC Vorbis comments) with a small stdlib writer. Remove the optional LanguageTool client and keep the existing local grammar fallback behind the same `GrammarCorrector` API. Remove `pykakasi` from the installer and from Chatterbox's patched metadata; Chatterbox already tolerates its absence and skips Japanese token conversion.

**Tech Stack:** Python 3.11/3.12 stdlib, `soundfile`, setuptools, pytest, existing release tooling.

## Global Constraints

- No new runtime dependency.
- Keep `GrammarCorrector(use_language_tool=...)` call-compatible so existing callers and patch points do not break.
- Keep FLAC metadata behavior: title, artist, comment, date, software, engine, and the existing render-plan keys are still written.
- The community edition must still render a deterministic FLAC end to end.
- Version this as **1.4.1**; do not rewrite the pushed 1.4.0 checksum record.

---

### Task 1: Native FLAC Vorbis comments

**Files:**
- Create: `src/the_oracle/audio/flac_tags.py`
- Modify: `src/the_oracle/audio/export_flac.py`
- Modify: `src/the_oracle/gui_render.py`
- Test: `tests/test_export_flac.py`, `tests/test_review_fixes_audio.py`, `tests/test_gui_render_import_safety.py`

**Interfaces:**
- Produces: `write_vorbis_comments(path: Path, metadata: dict[str, str]) -> None`
- Produces: `read_vorbis_comments(path: Path) -> dict[str, list[str]]`
- `write_flac` keeps its signature and best-effort tagging contract.

- [ ] Write a failing round-trip test for `TITLE`/`ARTIST` and for a second write preserving an existing comment.
- [ ] Implement the FLAC metadata-block parser/writer: validate `fLaC`, preserve non-comment blocks, replace or insert block type 4, set the last-block flag, write atomically.
- [ ] Switch `export_flac.write_flac` from `_tag_with_mutagen` to `_tag_flac`; update the warning test.
- [ ] Remove the mutagen imports/comment from `gui_render.py` and the mutagen entries from the import-safety tests.
- [ ] Run: `pytest tests/test_export_flac.py tests/test_review_fixes_audio.py tests/test_gui_render_import_safety.py tests/test_worker_import_sweep.py -q`

### Task 2: Remove the optional LanguageTool client

**Files:**
- Modify: `src/the_oracle/text_repair/grammar.py`
- Modify: `src/the_oracle/cli.py`, `src/the_oracle/smoke.py`, `src/the_oracle/pipeline.py`, `src/the_oracle/app_gui.py` (stale comments)
- Modify: `scripts/manage_install.py`
- Delete: `tests/test_grammar_language_tool.py`, `tests/test_grammar_self_join.py`
- Test: `tests/test_review_fixes_text.py`, `tests/test_vulkan_render.py`, `tests/test_worker_import_sweep.py`

**Interfaces:**
- `GrammarCorrector(use_language_tool: bool = True)` remains.
- `GrammarCorrector._try_load_language_tool()` remains and returns `None`, so existing patch points keep working.
- `GrammarCorrector.correct(text, aggressive=False, *, preserve_newlines=False)` keeps its behavior on the local path.

- [ ] Reduce `grammar.py` to the local `COMMON_FIXES`/punctuation/ellipsis path; remove all `language_tool_python` imports, downloads, threads, and timeout state.
- [ ] Keep the `_tool` branch for injected test doubles; default it to `None`.
- [ ] Delete the two tests that exist only for the removed wrapper.
- [ ] Remove the `LTP_JAR_DIR_PATH` setup from the installer and stale comments from callers.
- [ ] Run: `pytest tests/test_review_fixes_text.py tests/test_vulkan_render.py tests/test_worker_import_sweep.py -q`

### Task 3: Remove pykakasi from the runtime

**Files:**
- Modify: `scripts/manage_install.py` (remove the pinned package)
- Modify: `scripts/patch_chatterbox_metadata.py` (drop `Requires-Dist: pykakasi`)
- Test: `tests/test_install_boundary.py`, `tests/test_chatterbox_metadata_patch.py`, `tests/test_offline_bundle.py`

- [ ] Remove `pykakasi==2.3.0` from `CHATTERBOX_ENGINE_PACKAGES`.
- [ ] Strip every `Requires-Dist: pykakasi...` line when patching Chatterbox metadata; keep the patch idempotent.
- [ ] Update the online/offline install command expectations.
- [ ] Document that Japanese text conversion is skipped; English and non-Japanese multilingual rendering are unchanged.
- [ ] Run: `pytest tests/test_install_boundary.py tests/test_chatterbox_metadata_patch.py tests/test_offline_bundle.py -q`

### Task 4: Packaging, policy, version, and gate

**Files:**
- Modify: `pyproject.toml` (drop `mutagen`, drop `language-tool-python`)
- Modify: `scripts/license_review.json` (empty review table)
- Modify: `src/the_oracle/third_party.py` (drop LanguageTool as an invoked external component)
- Modify: `src/the_oracle/__init__.py`, `CHANGELOG.md`, `README.md`, `docs/SOFT_LAUNCH.md`
- Regenerate: `THIRD_PARTY_NOTICES.md`
- Test: `tests/test_third_party_notices.py`, `tests/test_soft_launch_gate.py`, `tests/test_release.py`, `tests/test_release_hygiene.py`, `tests/test_doctor_dependency_pins.py`

- [ ] Uninstall the three packages from the project venv and re-run the Chatterbox metadata patch.
- [ ] Bump to 1.4.1, stamp the changelog, and sync the README banner.
- [ ] Regenerate notices; confirm none of the three packages appears and `notice_problems()` is empty.
- [ ] Update the runbook: the only remaining gate blocker is the EULA; note the removed features and the LGPL warning.
- [ ] Run: `python scripts/soft_launch.py --check`; expect one blocker (`eula`) and the weak-copyleft warning.

### Task 5: End-to-end proof and release record

- [ ] Run the deterministic smoke: `python scripts/smoke_render.py --output-root build/smoke_render_1_4_1`; inspect the FLAC with `ffprobe` and `soundfile`.
- [ ] Run the full suite.
- [ ] Build 1.4.1 artifacts and the tracked checksum with `scripts/release.py`.
- [ ] Commit and push the branch.
