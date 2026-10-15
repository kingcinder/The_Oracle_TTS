# Third-Party Notices (DRAFT)

**Draft inventory for owner and lawyer review. Do not publish as final until generated from the release environment.**

The release must include a machine-generated dependency inventory and the license text required by each component. The following items are identified from the project manifest and release notes and require verification against the exact lockfile/build:

- PySide6 — LGPL obligations and relinking/replacement instructions.
- LanguageTool / `language-tool-python` — verify LGPL terms for the shipped distribution.
- `audio.cpp` and `ggml` — verify the exact commit and license files included in the binary/source distribution.
- Chatterbox TTS and each Hugging Face model listed in `src/the_oracle/models/pins.py` — inspect each model card and record license, attribution, and commercial-use terms.
- ffmpeg, if bundled — record the exact build flavor and corresponding license obligations.
- All other installed packages — generate with `pip-licenses` from the release lock environment and attach exact license texts.

## Open questions

1. Which exact dependency lock and artifact set is the legal review approving?
2. Are any model or voice terms incompatible with paid proprietary distribution?
3. What relinking/replacement mechanism will be documented for LGPL components?
4. Is ffmpeg bundled, downloaded, or required from the host?
