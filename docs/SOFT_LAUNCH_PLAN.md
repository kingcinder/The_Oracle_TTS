# The Oracle 1.4.0 Commercial Alpha Soft-Launch Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 1.4.0 tree shippable as a narrow, paid, commercial alpha: a production license key is registered, voice-asset provenance is truthful, third-party notices and a dependency-license gate exist, early users can produce a redacted support bundle, and a single go/no-go command says exactly what still blocks distribution.

**Architecture:** Keep the existing small-module, stdlib-first conventions. Add one pure-Python `third_party` inventory module (no network, no heavy imports), one stdlib-only `support_bundle` module, and one release-tooling gate script that composes the existing `release.py` checks with new legal, asset, and license checks. The gate is advisory for internal builds and authoritative for commercial distribution: it exits nonzero until every owner action is recorded.

**Tech Stack:** Python 3.11/3.12 stdlib, `importlib.metadata`, `packaging` (already a runtime dependency), pytest, Markdown, Git.

## Global Constraints

- Supported Python: `3.11` and `3.12`; supported platforms: Linux (supported) and Windows (experimental for this alpha).
- Do not add a third-party runtime dependency for any new feature.
- Do not perform network I/O from the new code paths.
- The production signing seed must never appear in the repo, logs, tests, or chat. Only the public verify key is committed.
- `Seashells/generic/` remains third-party licensed material with its existing attribution. The eight root-level recordings are Owner-owned, per the owner's 2026-10-19 decision.
- The EULA remains a draft for counsel review; the commercial gate must refuse distribution until the governing-law section is completed and approved.
- Version metadata has one source: `src/the_oracle/__init__.py`; banners and CHANGELOG are synchronized by `scripts/release.py`.
- No existing workflow is gated. Community remains everything that shipped before 1.4.0.

---

### Task 1: Provision the production verify key

**Files:**
- Modify: `src/the_oracle/licensing/keys.py`
- Create (outside repo, never committed): `~/.config/the-oracle/license-signing-seed`
- Modify: `docs/LICENSING_OPS.md`
- Test: `tests/test_licensing_registry_hygiene.py`, `tests/test_release.py`

**Interfaces:**
- Produces: `KEYS: tuple[VerifyKey, ...]` containing exactly one `VerifyKey(key_id="k2", public_key=bytes)`.
- Consumes: `the_oracle._ed25519.public_key(seed: bytes) -> bytes`.

- [ ] **Step 1: Generate the seed without printing it**

Run a Python one-off that imports `_ed25519`, generates 32 random bytes, writes the hex seed to `~/.config/the-oracle/license-signing-seed` with mode `0600`, and prints only the public verify key and the seed file path. Never print the seed.

- [ ] **Step 2: Register only the public key**

Replace the empty `KEYS` tuple with:

```python
KEYS: tuple[VerifyKey, ...] = (
    VerifyKey(key_id="k2", public_key=bytes.fromhex("<verify_key_hex>")),
)
```

- [ ] **Step 3: Verify the registry is non-empty and not a published vector**

Run: `.venv/bin/pytest tests/test_licensing_registry_hygiene.py tests/test_release.py -q`
Expected: PASS.

- [ ] **Step 4: Document custody**

Update `docs/LICENSING_OPS.md` §1 to say `k2` is registered, the seed is the only secret, and the seed file created during provisioning must be moved into the owner's secret manager.

---

### Task 2: Make voice-asset provenance truthful

**Files:**
- Modify: `Seashells/ASSETS.md`
- Modify: `tests/test_release_hygiene.py`
- Modify: `EULA-DRAFT.md`
- Test: `tests/test_release_hygiene.py`

**Interfaces:**
- Produces: `ASSETS.md` rows for every tracked `Seashells/**/*.wav`; root-level rows say `Owner-owned` and contain no `undocumented`/`unknown`/`TBD`/`TODO`.
- Consumes: `git ls-files Seashells`.

- [ ] **Step 1: Update the eight root-level rows**

For each root-level file, set:

```markdown
| `Seashells/<file>.wav` | Owner recording; supplied by the copyright holder | Owner-owned; all rights reserved | Original recording by the Owner. No third-party rights. Confirmed 2026-10-19. |
```

- [ ] **Step 2: Add the EULA cross-reference**

In `EULA-DRAFT.md` §3, add one sentence: `Seashells/ASSETS.md` is the authoritative inventory of recordings shipped with the Software.

- [ ] **Step 3: Harden the hygiene tests**

Replace the generic-only inventory test with a test over every tracked `Seashells/**/*.wav`, and add:

```python
def test_voice_inventory_has_no_undocumented_entries() -> None:
    inventory = (REPO_ROOT / "Seashells" / "ASSETS.md").read_text(encoding="utf-8").lower()
    for forbidden in ("undocumented", "unknown", "tbd", "todo"):
        assert forbidden not in inventory
    assert "owner-owned" in inventory
```

Add a reverse test that every path named in the inventory is a tracked file.

- [ ] **Step 4: Run the hygiene suite**

Run: `.venv/bin/pytest tests/test_release_hygiene.py -q`
Expected: PASS.

---

### Task 3: Generate the third-party inventory and notices

**Files:**
- Create: `src/the_oracle/third_party.py`
- Create: `scripts/generate_third_party_notices.py`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `THIRD_PARTY_LICENSES/LGPL-3.0.txt`
- Create: `THIRD_PARTY_LICENSES/Apache-2.0.txt`
- Create: `THIRD_PARTY_LICENSES/MIT.txt`
- Create: `THIRD_PARTY_LICENSES/BSD-3-Clause.txt`
- Create: `THIRD_PARTY_LICENSES/GPL-2.0.txt`
- Create: `THIRD_PARTY_LICENSES/GPL-3.0.txt`
- Create: `THIRD_PARTY_LICENSES/MPL-2.0.txt`
- Create: `MANIFEST.in`
- Test: `tests/test_third_party_notices.py`

**Interfaces:**
- Produces: `Component(name: str, version: str, license: str, kind: str, source: str, notice: str = "")`
- Produces: `runtime_components(pyproject: Path, distributions=None) -> list[Component]`
- Produces: `bundled_components(root: Path) -> list[Component]`
- Produces: `render_notices(components: list[Component], version: str) -> str`
- Produces: `notice_problems(root: Path) -> list[str]`
- Produces: `classify_license(text: str) -> str` returning `permissive`, `weak-copyleft`, `strong-copyleft`, `proprietary`, or `unknown`.

**Bundled component table (exact facts):**
- `PySide6` / `shiboken6` — LGPL-3.0 (commercial Qt license alternative), source `https://code.qt.io/cgit/pyside/pyside-setup.git/`.
- `chatterbox-tts` and `ResembleAI/chatterbox` — MIT, © 2025 Resemble AI.
- `ResembleAI/chatterbox-turbo` — MIT.
- `resemble-perth` — MIT, © 2025 Resemble AI.
- `SamLowe/roberta-base-go_emotions` — MIT, © 2023 SamLowe.
- `oliverguhr/fullstop-punctuation-multilang-large` — MIT, © 2021 Oliver Guhr.
- `audio.cpp` — Apache-2.0, © 2026 ShugoAI LLC.
- `ggml` — MIT, © 2023-2026 The ggml authors.
- `ffmpeg` — external host dependency, not bundled; LGPL/GPL depends on the build.
- `Seashells/generic/` — Apache-2.0 (audio.cpp demo voices) and CC BY 4.0 (LibriSpeech), attribution in `Seashells/generic/ATTRIBUTION.md`.
- Root-level `Seashells/*.wav` — Owner-owned.

- [ ] **Step 1: Write the failing tests**

Cover:
- every `[project] dependencies` name appears in the rendered notices;
- every `MODEL_PINS` repo appears;
- every tracked `Seashells/generic/*.wav` appears;
- no `DRAFT`, `TODO`, `TBD`, or `undocumented` text appears;
- `notice_problems()` is empty for the committed file;
- a fixture with an unknown license returns `unknown` and a non-empty problem list;
- `render_notices` is deterministic for the same input.

- [ ] **Step 2: Implement `third_party.py`**

Use `importlib.metadata` and `packaging.requirements.Requirement`/`Marker`. Resolve the runtime closure from `pyproject.toml` extras `ml` and `chatterbox`, plus the separately installed `chatterbox-tts`. Include the project itself. Sort by canonical name. Do not import torch or any ML module.

- [ ] **Step 3: Implement the generator**

`python scripts/generate_third_party_notices.py` writes `THIRD_PARTY_NOTICES.md` with:
- a header naming version and the command that regenerates it;
- a summary of obligations (LGPL relinking/source offer, GPL review blockers, MIT/BSD/Apache notice requirements);
- bundled assets/models/tools sections;
- a Python dependency table (`name`, `version`, `license`, `source`);
- a note that each installed Python distribution carries its own license files in its `.dist-info` directory.

`--check` re-renders and compares to the committed file; `--json` prints the component list.

- [ ] **Step 4: Add license texts and packaging**

Copy the standard license texts from `/usr/share/common-licenses/` into `THIRD_PARTY_LICENSES/` (commit them; do not fetch at runtime). Add `MANIFEST.in` lines so the sdist includes `THIRD_PARTY_NOTICES.md`, `THIRD_PARTY_LICENSES/*`, `EULA-DRAFT.md`, `PRIVACY.md`, `docs/SOFT_LAUNCH.md`, and `Seashells/ASSETS.md`. Add `license-files` to the setuptools config so the wheel carries `LICENSE` and `THIRD_PARTY_NOTICES.md`.

- [ ] **Step 5: Generate and verify**

Run: `.venv/bin/python scripts/generate_third_party_notices.py`
Run: `.venv/bin/pytest tests/test_third_party_notices.py -q`
Expected: PASS, and the generated file contains no draft/TODO language.

---

### Task 4: Add the dependency-license policy and soft-launch gate

**Files:**
- Create: `scripts/license_review.json`
- Create: `scripts/soft_launch.py`
- Test: `tests/test_soft_launch_gate.py`

**Interfaces:**
- Produces: `Finding(key: str, ok: bool, severity: str, detail: str, owner_action: str = "")`
- Produces: `audit(root: Path, *, today: date | None = None) -> list[Finding]`
- Produces: `report_text(findings: list[Finding]) -> str`
- Consumes: `release.check_report`, `release.licensing_key_problems`, `third_party.notice_problems`, `third_party.classify_license`.

**Gate checks:**
1. `version` — `release.check_report` has no problems.
2. `licensing_key` — `release.licensing_key_problems` is empty.
3. `eula` — `EULA-DRAFT.md` exists, is not marked draft, has no `[insert` placeholder, and names a governing law.
4. `third_party_notices` — committed notices exist and `notice_problems` is empty.
5. `license_policy` — no `strong-copyleft` or `unknown` component is unreviewed; `weak-copyleft` is a warning with obligations.
6. `voice_assets` — every `Seashells/**/*.wav` has a non-undocumented inventory row.
7. `checksums` — `release_checksums/checksums-<version>.sha256` exists.
8. `artifacts` — the version's sdist and wheel exist in `release_artifacts/`.
9. `windows_scope` — `docs/SOFT_LAUNCH.md` names Windows as experimental.
10. `support_bundle` — `the_oracle.support_bundle` imports.

**License policy file:**

```json
{
  "policy": "Strong-copyleft or unknown licenses block commercial distribution until the owner records a decision.",
  "allow": [],
  "review": {
    "mutagen": {"license": "GPL-2.0-or-later", "reason": "FLAC tagging; replace or obtain a commercial license."},
    "language-tool-python": {"license": "GPL-3.0", "reason": "Optional grammar wrapper; replace with a permissive client or drop."},
    "pykakasi": {"license": "GPL-3.0-or-later", "reason": "Transitive from chatterbox-tts; remove Japanese support or obtain terms."}
  }
}
```

- [ ] **Step 1: Write the failing tests**

Cover: a draft EULA is a blocker; a final EULA passes; a strong-copyleft package in `review` is a blocker with the recorded reason; an unknown license is a blocker; a clean fixture has no blockers; `--json` is valid JSON; the report names the owner action for every blocker.

- [ ] **Step 2: Implement the gate**

Default to human-readable output; `--json` emits a list of finding objects. Exit 0 only when no finding has `severity == "blocker"`. Include `--allow-draft-legal` only for internal diagnostics, clearly labelled; the commercial gate does not use it.

- [ ] **Step 3: Run the gate**

Run: `.venv/bin/python scripts/soft_launch.py --check`
Expected at this stage: nonzero, with blockers for the draft EULA and the three strong-copyleft packages. Every blocker names its owner action.

---

### Task 5: Implement the redacted support bundle

**Files:**
- Create: `src/the_oracle/support_bundle.py`
- Modify: `src/the_oracle/cli.py`
- Test: `tests/test_support_bundle.py`

**Interfaces:**
- Produces: `SupportBundleResult(path: Path, manifest: dict[str, object], included: tuple[str, ...], skipped: tuple[str, ...])`
- Produces: `build_support_bundle(root, out_path=None, *, include_crash_reports=False, doctor_report=None, now=None) -> SupportBundleResult`
- Produces: CLI `the-oracle support-bundle [--out PATH] [--include-crash-reports] [--no-doctor] [--json]`.

**Bundle contents:**
- `manifest.json` — bundle version, UTC timestamp, app version, platform, Python, included/skipped, redaction statement.
- `environment.json` — OS, Python, CPU count, presence (not values) of `ORACLE_*`/`HF_HUB_OFFLINE`/CUDA variables.
- `license.json` — `ok`, `state`, `edition`, `key_id`, `exp`, `machine_locked`; never the token or licensee.
- `settings.json` — redacted settings; no paths, tokens, or license values.
- `logs/` — newest 3 logs, capped at 256 KiB each, sanitized.
- `crash_reports/` — only when consent is on AND `--include-crash-reports`; otherwise listed as skipped.
- `doctor.json` — the supplied report, or the output of `scripts/doctor.py --json --skip-model-init --ci` when available; a doctor failure is recorded, not fatal.
- `README.txt` — what is included, what is not, and how to share it.

**Redaction rules:**
- Reuse `the_oracle.crash.sanitize.sanitize_text`.
- Redact `ORACLE1.` tokens, home-directory prefixes, email addresses in licensee fields, and common API-key shapes.
- Recursively redact keys named `token`, `license`, `licensee`, `machine_hash`, `secret`, `password`, `api_key`, `path`, `file`, `dir`, `recent`, `input`, `output`.
- Never include input text, audio, reference clips, or recordings.

- [ ] **Step 1: Write the failing tests**

Cover: bundle is a valid zip; manifest has required keys; token/home path/email are absent from every entry; crash reports are excluded when consent is off; included when consent is on and the flag is passed; `--no-doctor` skips doctor; a failing doctor records an error but still produces a bundle; default output lands under `Output/`; the CLI prints the path.

- [ ] **Step 2: Implement the module and CLI**

Use `zipfile`, `json`, `os.replace`, and `pathlib`. No network. No heavy imports. Keep the module importable without Qt or ML dependencies.

- [ ] **Step 3: Run the tests**

Run: `.venv/bin/pytest tests/test_support_bundle.py -q`
Expected: PASS.

---

### Task 6: Write the soft-launch runbook and customer path

**Files:**
- Create: `docs/SOFT_LAUNCH.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/LICENSING_OPS.md`
- Modify: `docs/TROUBLESHOOTING.md`

**Content:**
- Scope and status: 1.4.0 paid alpha; Linux supported; Windows experimental.
- Install: Python 3.11/3.12, ffmpeg, Git; `./oracle install`, `./oracle start`.
- Purchase: email `codysa90@gmail.com`; the owner mints a Studio token with `scripts/license_sign.py mint --key-id k2`; delivery is the token string only.
- Activation: `the-oracle activate <token>` or the in-app dialog; offline; `license-status`; community remains fully usable.
- Support: `the-oracle support-bundle`; what it includes/excludes; never sends anything automatically.
- Updates: manual, via `git pull && ./oracle update`; CHANGELOG is the source of truth.
- Known limitations: Windows experimental; first model download; GPL dependency review in progress.
- Owner go-live checklist: complete the EULA governing-law section and counsel review; record a decision for each strong-copyleft dependency; regenerate notices; run the gate; build artifacts; tag; publish.
- The gate command and the meaning of each blocker.

- [ ] **Step 1: Write the runbook and README section**

Add a top-level "Alpha Soft Launch" section to the README linking to the runbook, with the activation and support commands.

- [ ] **Step 2: Add 1.4.0 changelog entries**

Under `## [Unreleased]`, add entries for the production key, voice provenance, third-party notices, license gate, support bundle, and soft-launch runbook. Keep the existing 1.4.0 entries.

- [ ] **Step 3: Check documentation links**

Run: `grep -n "SOFT_LAUNCH\|support-bundle\|third-party" README.md docs/*.md CHANGELOG.md`
Expected: every new doc is linked from at least one existing entry point.

---

### Task 7: Cut 1.4.0 and verify the whole tree

**Files:**
- Modify: `src/the_oracle/__init__.py`
- Modify: `CHANGELOG.md` (via `--sync-changelog`)
- Modify: `README.md` (via `--sync-banners`)
- Create: `release_checksums/checksums-1.4.0.sha256` (via release build)

- [ ] **Step 1: Bump the version**

Set `__version__ = "1.4.0"`.

- [ ] **Step 2: Sync the changelog and banners**

Run: `.venv/bin/python scripts/release.py --sync-changelog`
Run: `.venv/bin/python scripts/release.py --sync-banners`
Run: `.venv/bin/python scripts/release.py --check`
Expected: exit 0.

- [ ] **Step 3: Run the targeted suites**

Run: `.venv/bin/pytest tests/test_release_hygiene.py tests/test_third_party_notices.py tests/test_soft_launch_gate.py tests/test_support_bundle.py tests/test_release.py tests/test_licensing_registry_hygiene.py -q`
Expected: PASS.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS. If the theme certifier reports a parked crash-report directory, confirm it is a leftover from an interrupted run, remove only that parked directory, and re-run; do not weaken the test.

- [ ] **Step 5: Build the release**

Commit the tree first so `release.py` sees a clean worktree, then run: `.venv/bin/python scripts/release.py`
Expected: full suite passes; `release_artifacts/the_oracle-1.4.0-*.{tar.gz,whl}` and `release_checksums/checksums-1.4.0.sha256` exist. Amend the commit to include the checksum (the commit is not pushed yet) or add a separate release-record commit.

- [ ] **Step 6: Run the commercial gate**

Run: `.venv/bin/python scripts/soft_launch.py --check`
Expected: nonzero, with only owner-action blockers remaining (draft EULA and the recorded strong-copyleft review items). This is the honest launch status; do not report the alpha as distribution-approved while it is red.

- [ ] **Step 7: Commit and push**

Commit with the repository's conventional style and push the branch to `origin/release/1.4.0` (the user explicitly requested the push). Report the commit hash, the gate status, and every blocker that remains.

---

## Self-Review

- **Spec coverage:** production key (Task 1), voice provenance (Task 2), third-party notices (Task 3), license policy/gate (Task 4), supportability (Task 5), customer/runbook docs (Task 6), release cut and verification (Task 7).
- **Placeholder scan:** no TBD/TODO implementation steps; the only intentional placeholders are the EULA governing-law field and the owner license decisions, both surfaced as blockers by design.
- **Type consistency:** `Component`, `Finding`, `SupportBundleResult`, and the `audit`/`build_support_bundle` signatures are named consistently across tasks.
- **Explicit deferrals:** replacing `mutagen`/`language-tool-python`/`pykakasi` is not attempted in this plan; the gate records them as owner/legal decisions. Windows crash certification remains open and is documented as experimental.
