# Third-Party Notices

This file is generated from the release tree by
`python scripts/generate_third_party_notices.py`. Do not edit it by hand;
regenerate it and review the diff. The inventory is also read by the
commercial launch gate (`python scripts/soft_launch.py --check`), so a
dependency that is missing here is a release blocker, not a documentation
omission.

Release version: **1.4.0**

## Distribution obligations at a glance

- **MIT / BSD / Apache-2.0 components** require their copyright and
  permission notices to accompany copies. Their license texts are in
  `THIRD_PARTY_LICENSES/` and in each Python distribution's `.dist-info`
  directory.
- **PySide6 / Qt (LGPL-3.0)** requires a prominent notice, a copy of the
  LGPL text, access to the corresponding source of the LGPL-covered
  libraries, and a way for users to replace/relink those libraries. A
  commercial Qt license is the alternative.
- **Strong-copyleft components (GPL/AGPL)** are not safe for a proprietary
  combined distribution without a recorded legal decision. The commercial
  gate reports them as blockers until `scripts/license_review.json` records
  the decision.
- **External tools** (ffmpeg, LanguageTool, audio.cpp/ggml) are not part of
  the Python wheel. The host supplies ffmpeg; the optional Vulkan backend
  builds audio.cpp/ggml locally; LanguageTool is downloaded separately.

## Bundled assets, models, and external components

| Component | Version / pin | License | Source |
|---|---|---|---|
| `PySide6 / shiboken6 / Qt` | 6.8.2.1 | LGPL-3.0 (or commercial Qt license) | [source](https://code.qt.io/cgit/pyside/pyside-setup.git/) |
| `audio.cpp` | built at install time from a pinned clone | Apache-2.0 | [source](https://github.com/0xShug0/audio.cpp) |
| `ffmpeg` | host-provided | LGPL/GPL depending on the build | [source](https://ffmpeg.org/) |
| `ggml` | vendored in the audio.cpp checkout | MIT | [source](https://github.com/ggml-org/ggml) |
| `LanguageTool` | downloaded at first use | LGPL-2.1 | [source](https://languagetool.org/) |
| `oliverguhr/fullstop-punctuation-multilang-large` | 345e80adc07e761d3a35feafd20f2f44a151f453 | MIT | [source](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) |
| `ResembleAI/chatterbox` | 5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18 | MIT | [source](https://huggingface.co/ResembleAI/chatterbox) |
| `ResembleAI/chatterbox-turbo` | 749d1c1a46eb10492095d68fbcf55691ccf137cd | MIT | [source](https://huggingface.co/ResembleAI/chatterbox-turbo) |
| `SamLowe/roberta-base-go_emotions` | d75048347613a25d77de8cf6412eaae9fa7b26be | MIT | [source](https://huggingface.co/SamLowe/roberta-base-go_emotions) |
| `chinese_female_1.wav` | — | Apache-2.0 | Seashells/generic/ATTRIBUTION.md |
| `chinese_male_1.wav` | — | Apache-2.0 | Seashells/generic/ATTRIBUTION.md |
| `chinese_male_2.wav` | — | Apache-2.0 | Seashells/generic/ATTRIBUTION.md |
| `chinese_speaker_1.wav` | — | Apache-2.0 | Seashells/generic/ATTRIBUTION.md |
| `Cody's Seashell.wav` | — | Owner-owned; all rights reserved | Seashells/ASSETS.md |
| `Cody's Seashell1.wav` | — | Owner-owned; all rights reserved | Seashells/ASSETS.md |
| `english_male_1.wav` | — | Apache-2.0 | Seashells/generic/ATTRIBUTION.md |
| `english_reader_1.wav` | — | CC BY 4.0 | Seashells/generic/ATTRIBUTION.md |
| `english_reader_2.wav` | — | CC BY 4.0 | Seashells/generic/ATTRIBUTION.md |
| `english_reader_3.wav` | — | CC BY 4.0 | Seashells/generic/ATTRIBUTION.md |
| `english_reader_4.wav` | — | CC BY 4.0 | Seashells/generic/ATTRIBUTION.md |
| `Fransisco_MONSTER.wav` | — | Owner-owned; all rights reserved | Seashells/ASSETS.md |
| `seashell fransisco.wav` | — | Owner-owned; all rights reserved | Seashells/ASSETS.md |
| `seashell fransisco_TTS_clean_v5_aggressive_targeted_cleanup.wav` | — | Owner-owned; all rights reserved | Seashells/ASSETS.md |
| `Seashell_ fransisco.wav` | — | Owner-owned; all rights reserved | Seashells/ASSETS.md |
| `Seashell_No_1.wav` | — | Owner-owned; all rights reserved | Seashells/ASSETS.md |
| `Seashell_No_2.wav` | — | Owner-owned; all rights reserved | Seashells/ASSETS.md |

## Python dependencies

Resolved from `pyproject.toml` and the installed package metadata. The
closure covers the runtime and the `ml`/`chatterbox` extras; development
tools are not part of the shipped environment.

| Component | Version | License | Source |
|---|---|---|---|
| `annotated-doc` | 0.0.5 | MIT | [source](https://github.com/fastapi/annotated-doc) |
| `annotated-types` | 0.8.0 | MIT | [source](https://github.com/annotated-types/annotated-types) |
| `antlr4-python3-runtime` | 4.9.3 | BSD-3-Clause | [source](http://www.antlr.org) |
| `anyio` | 4.15.1 | MIT | [source](https://anyio.readthedocs.io/en/latest/) |
| `audioread` | 3.1.0 | MIT | [source](https://github.com/beetbox/audioread/issues) |
| `brotli` | 1.2.0 | MIT | [source](https://github.com/google/brotli) |
| `catalogue` | 2.0.10 | MIT | [source](https://github.com/explosion/catalogue) |
| `certifi` | 2026.7.22 | MPL-2.0 | [source](https://github.com/certifi/python-certifi) |
| `cffi` | 2.1.1 | MIT-0 | [source](https://cffi.readthedocs.io/) |
| `cfgv` | 3.5.0 | MIT | [source](https://github.com/asottile/cfgv) |
| `charset-normalizer` | 3.4.9 | MIT | [source](https://github.com/jawah/charset_normalizer/blob/master/CHANGELOG.md) |
| `chatterbox-tts` | 0.1.7+oracle1 | MIT | [source](https://github.com/resemble-ai/chatterbox) |
| `click` | 8.5.0 | BSD-3-Clause | [source](https://click.palletsprojects.com/page/changes/) |
| `conformer` | 0.3.2 | MIT | [source](https://github.com/lucidrains/conformer) |
| `decorator` | 5.3.1 | BSD-3-Clause | — |
| `deepmultilingualpunctuation` | 1.0.1 | MIT | [source](https://github.com/oliverguhr/deepmultilingualpunctuation) |
| `Deprecated` | 1.3.1 | MIT | [source](https://github.com/laurent-laporte-pro/deprecated) |
| `diffusers` | 0.38.0 | Apache-2.0 | [source](https://github.com/huggingface/diffusers) |
| `distlib` | 0.4.3 | PSF-2.0 | [source](https://github.com/pypa/distlib) |
| `editdistpy` | 0.4.0 | MIT | [source](https://github.com/mammothb/editdistpy/blob/master/CHANGELOG.md) |
| `einops` | 0.8.2 | MIT | [source](https://github.com/arogozhnikov/einops) |
| `fastapi` | 0.141.1 | MIT | [source](https://github.com/fastapi/fastapi) |
| `filelock` | 3.32.2 | MIT | [source](https://py-filelock.readthedocs.io) |
| `fsspec` | 2026.7.0 | BSD-3-Clause | [source](https://filesystem-spec.readthedocs.io/en/latest/changelog.html) |
| `gradio` | 6.16.0 | Apache-2.0 | [source](https://github.com/gradio-app/gradio) |
| `gradio_client` | 2.5.0 | Apache-2.0 | [source](https://github.com/gradio-app/gradio) |
| `groovy` | 0.1.2 | MIT | [source](https://github.com/gradio-app/groovy) |
| `h11` | 0.16.0 | MIT | [source](https://github.com/python-hyper/h11) |
| `hf-gradio` | 0.4.1 | MIT | [source](https://github.com/gradio-app/hf-gradio#) |
| `hf-xet` | 1.6.0 | Apache-2.0 | [source](https://huggingface.co/docs/hub/xet/index) |
| `httpcore` | 1.0.9 | BSD-3-Clause | [source](https://www.encode.io/httpcore) |
| `httpx` | 0.28.1 | BSD-3-Clause | [source](https://github.com/encode/httpx/blob/master/CHANGELOG.md) |
| `huggingface_hub` | 1.33.0 | Apache-2.0 | [source](https://github.com/huggingface/huggingface_hub) |
| `identify` | 2.6.19 | MIT | [source](https://github.com/pre-commit/identify) |
| `idna` | 3.18 | BSD-3-Clause | [source](https://github.com/kjd/idna/blob/master/HISTORY.md) |
| `importlib_metadata` | 9.0.0 | Apache-2.0 | [source](https://github.com/python/importlib_metadata) |
| `jaconv` | 0.5.0 | MIT | [source](https://github.com/ikegami-yukino/jaconv) |
| `Jinja2` | 3.1.6 | BSD-3-Clause | [source](https://jinja.palletsprojects.com/changes/) |
| `joblib` | 1.5.3 | BSD-3-Clause | [source](https://joblib.readthedocs.io) |
| `language_tool_python` | 2.9.4 | GPL-3.0 | [source](https://github.com/jxmorris12/language_tool_python.git) |
| `lazy-loader` | 0.5 | BSD-3-Clause | [source](https://scientific-python.org/specs/spec-0001/) |
| `librosa` | 0.11.0 | ISC | [source](https://librosa.org) |
| `llvmlite` | 0.48.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | [source](http://llvmlite.readthedocs.io) |
| `markdown-it-py` | 3.0.0 | MIT | [source](https://markdown-it-py.readthedocs.io) |
| `MarkupSafe` | 3.0.3 | BSD-3-Clause | [source](https://palletsprojects.com/donate) |
| `mdurl` | 0.1.2 | MIT | [source](https://github.com/executablebooks/mdurl) |
| `ml_dtypes` | 0.5.4 | Apache-2.0 | [source](https://github.com/jax-ml/ml_dtypes) |
| `mpmath` | 1.3.0 | BSD-3-Clause | [source](http://mpmath.org/) |
| `msgpack` | 1.2.1 | Apache-2.0 | [source](https://msgpack.org/) |
| `mutagen` | 1.47.0 | GPL-2.0-or-later | [source](https://github.com/quodlibet/mutagen) |
| `networkx` | 3.6.1 | BSD-3-Clause | [source](https://networkx.org/) |
| `nodeenv` | 1.10.0 | BSD-3-Clause | [source](https://github.com/ekalinin/nodeenv) |
| `numba` | 0.66.0 | BSD-3-Clause | [source](https://numba.pydata.org) |
| `numpy` | 1.26.4 | BSD-3-Clause | [source](https://numpy.org) |
| `omegaconf` | 2.3.1 | BSD-3-Clause | [source](https://github.com/omry/omegaconf) |
| `onnx` | 1.22.0 | Apache-2.0 | [source](https://onnx.ai/) |
| `orjson` | 3.12.0 | MPL-2.0 AND (Apache-2.0 OR MIT) | [source](https://github.com/ijl/orjson/blob/master/CHANGELOG.md) |
| `packaging` | 26.3 | Apache-2.0 OR BSD-2-Clause | [source](https://packaging.pypa.io/) |
| `pandas` | 3.0.6 | BSD-3-Clause | [source](https://pandas.pydata.org) |
| `pillow` | 12.3.0 | MIT-CMU | [source](https://pillow.readthedocs.io/en/stable/releasenotes/index.html) |
| `platformdirs` | 4.11.1 | MIT | [source](https://platformdirs.readthedocs.io/en/latest/changelog.html) |
| `pooch` | 1.9.0 | BSD-3-Clause | [source](https://www.fatiando.org/pooch) |
| `pre_commit` | 4.6.1 | MIT | [source](https://github.com/pre-commit/pre-commit) |
| `protobuf` | 7.35.1 | BSD-3-Clause | [source](https://developers.google.com/protocol-buffers/) |
| `psutil` | 7.2.2 | BSD-3-Clause | [source](https://github.com/giampaolo/psutil) |
| `pycparser` | 3.0 | BSD-3-Clause | [source](https://github.com/eliben/pycparser) |
| `pydantic` | 2.13.5 | MIT | [source](https://github.com/pydantic/pydantic) |
| `pydantic_core` | 2.46.5 | MIT | [source](https://github.com/pydantic/pydantic) |
| `pydub` | 0.25.1 | MIT | [source](http://pydub.com) |
| `Pygments` | 2.21.0 | BSD-2-Clause | [source](https://pygments.org) |
| `pykakasi` | 2.3.0 | GPL-3.0-or-later | [source](https://codeberg.org/miurahr/pykakasi) |
| `pyloudnorm` | 0.2.0 | MIT | [source](https://github.com/csteinmetz1/pyloudnorm) |
| `PySide6` | 6.8.2.1 | LGPL-3.0 | [source](https://pyside.org) |
| `PySide6_Addons` | 6.8.2.1 | LGPL-3.0 | [source](https://pyside.org) |
| `PySide6_Essentials` | 6.8.2.1 | LGPL-3.0 | [source](https://pyside.org) |
| `python-dateutil` | 2.9.0.post0 | BSD-3-Clause | [source](https://github.com/dateutil/dateutil) |
| `python-discovery` | 1.6.3 | MIT | [source](https://github.com/tox-dev/python-discovery/releases) |
| `python-multipart` | 0.0.32 | Apache-2.0 | [source](https://github.com/Kludex/python-multipart) |
| `pytz` | 2026.3.post1 | MIT | [source](http://pythonhosted.org/pytz) |
| `PyYAML` | 6.0.3 | MIT | [source](https://pyyaml.org/) |
| `regex` | 2026.7.19 | Apache-2.0 AND CNRI-Python | [source](https://github.com/mrabarnett/mrab-regex) |
| `requests` | 2.34.2 | Apache-2.0 | [source](https://requests.readthedocs.io) |
| `resemble-perth` | 1.1.0 | MIT | [source](https://github.com/resemble-ai/perth) |
| `rich` | 15.0.0 | MIT | [source](https://rich.readthedocs.io/en/latest/) |
| `s3tokenizer` | 0.3.0 | Apache-2.0 | [source](https://github.com/xingchensong/S3Tokenizer) |
| `safehttpx` | 0.1.7 | MIT | [source](https://github.com/gradio-app/safehttpx) |
| `safetensors` | 0.8.0 | Apache-2.0 | [source](https://github.com/huggingface/safetensors) |
| `scikit-learn` | 1.6.1 | BSD-3-Clause | [source](https://scikit-learn.org) |
| `scipy` | 1.15.2 | BSD-3-Clause | [source](https://scipy.org/) |
| `semantic-version` | 2.10.0 | BSD-3-Clause | [source](https://github.com/rbarrois/python-semanticversion) |
| `setuptools` | 83.0.0 | MIT | [source](https://github.com/pypa/setuptools) |
| `shellingham` | 1.5.4 | ISC | [source](https://github.com/sarugaku/shellingham) |
| `shiboken6` | 6.8.2.1 | LGPL-3.0 | [source](https://pyside.org) |
| `six` | 1.17.0 | MIT | [source](https://github.com/benjaminp/six) |
| `sounddevice` | 0.5.6 | MIT | [source](https://python-sounddevice.readthedocs.io/) |
| `soundfile` | 0.13.1 | BSD-3-Clause | [source](https://github.com/bastibe/python-soundfile) |
| `soxr` | 1.1.0 | LGPL-2.1-or-later | [source](https://github.com/dofuuz/python-soxr) |
| `spacy_pkuseg` | 1.0.1 | MIT | [source](https://github.com/explosion/spacy-pkuseg) |
| `srsly` | 2.5.3 | MIT | [source](https://github.com/explosion/srsly) |
| `starlette` | 1.3.1 | BSD-3-Clause | [source](https://github.com/Kludex/starlette) |
| `sympy` | 1.13.1 | BSD-3-Clause | [source](https://sympy.org) |
| `symspellpy` | 6.9.0 | MIT | [source](https://github.com/mammothb/symspellpy) |
| `threadpoolctl` | 3.6.0 | BSD-3-Clause | [source](https://github.com/joblib/threadpoolctl) |
| `tokenizers` | 0.23.3 | Apache-2.0 | [source](https://github.com/huggingface/tokenizers) |
| `toml` | 0.10.2 | MIT | [source](https://github.com/uiri/toml) |
| `tomlkit` | 0.13.3 | MIT | [source](https://github.com/sdispater/tomlkit) |
| `torch` | 2.6.0+cpu | BSD-3-Clause | [source](https://pytorch.org/) |
| `torchaudio` | 2.6.0+cpu | BSD-3-Clause | [source](https://github.com/pytorch/audio) |
| `torchvision` | 0.21.0+cpu | BSD-3-Clause | [source](https://github.com/pytorch/vision) |
| `tqdm` | 4.70.0 | MPL-2.0 | [source](https://tqdm.github.io) |
| `transformers` | 5.20.0 | Apache-2.0 | [source](https://github.com/huggingface/transformers) |
| `typer` | 0.27.2 | MIT | [source](https://github.com/fastapi/typer) |
| `typing_extensions` | 4.16.0 | PSF-2.0 | [source](https://github.com/python/typing_extensions/issues) |
| `typing-inspection` | 0.4.4 | MIT | [source](https://github.com/pydantic/typing-inspection) |
| `urllib3` | 2.8.0 | MIT | [source](https://github.com/urllib3/urllib3/blob/main/CHANGES.rst) |
| `uvicorn` | 0.53.0 | BSD-3-Clause | [source](https://uvicorn.dev/release-notes) |
| `virtualenv` | 21.7.13 | MIT | [source](https://virtualenv.pypa.io) |
| `wrapt` | 2.3.0 | BSD-2-Clause | [source](https://github.com/GrahamDumpleton/wrapt) |
| `zipp` | 4.1.0 | MIT | [source](https://github.com/jaraco/zipp) |

## Required notices

The following notices must accompany the distribution. The standard
license texts are in `THIRD_PARTY_LICENSES/`.

- **PySide6 / shiboken6 / Qt** — LGPL-3.0. A proprietary application may use the LGPL path while keeping its own source proprietary, provided the LGPL-covered libraries remain replaceable/relinkable, their complete corresponding source is made available (or a written offer is given), and the LGPL notice and text accompany the distribution. A Qt commercial license is the alternative.
- **audio.cpp** — Copyright 2026 ShugoAI LLC. Apache-2.0. Modified by The Oracle's vendored patches; see scripts/patches/.
- **ffmpeg** — Not bundled. The host supplies ffmpeg; its own build determines the exact license.
- **ggml** — Copyright (c) 2023-2026 The ggml authors. MIT license.
- **LanguageTool** — The LanguageTool server is downloaded separately, not bundled in the wheel.
- **oliverguhr/fullstop-punctuation-multilang-large** — Copyright (c) 2021 Oliver Guhr. MIT license.
- **ResembleAI/chatterbox** — Copyright (c) 2025 Resemble AI. MIT license; commercial use permitted.
- **ResembleAI/chatterbox-turbo** — MIT license; commercial use permitted.
- **SamLowe/roberta-base-go_emotions** — Copyright (c) 2023 SamLowe. MIT license.
- **chinese_female_1.wav** — audio.cpp demo voice, Copyright 2026 ShugoAI LLC
- **chinese_male_1.wav** — audio.cpp demo voice, Copyright 2026 ShugoAI LLC
- **chinese_male_2.wav** — audio.cpp demo voice, Copyright 2026 ShugoAI LLC
- **chinese_speaker_1.wav** — audio.cpp demo voice, Copyright 2026 ShugoAI LLC
- **Cody's Seashell.wav** — Original recording by the Owner; no third-party rights.
- **Cody's Seashell1.wav** — Original recording by the Owner; no third-party rights.
- **english_male_1.wav** — audio.cpp demo voice, Copyright 2026 ShugoAI LLC
- **english_reader_1.wav** — LibriSpeech clip; attribution in Seashells/generic/ATTRIBUTION.md
- **english_reader_2.wav** — LibriSpeech clip; attribution in Seashells/generic/ATTRIBUTION.md
- **english_reader_3.wav** — LibriSpeech clip; attribution in Seashells/generic/ATTRIBUTION.md
- **english_reader_4.wav** — LibriSpeech clip; attribution in Seashells/generic/ATTRIBUTION.md
- **Fransisco_MONSTER.wav** — Original recording by the Owner; no third-party rights.
- **seashell fransisco.wav** — Original recording by the Owner; no third-party rights.
- **seashell fransisco_TTS_clean_v5_aggressive_targeted_cleanup.wav** — Original recording by the Owner; no third-party rights.
- **Seashell_ fransisco.wav** — Original recording by the Owner; no third-party rights.
- **Seashell_No_1.wav** — Original recording by the Owner; no third-party rights.
- **Seashell_No_2.wav** — Original recording by the Owner; no third-party rights.












## Standard license texts

- `Apache-2.0` — `THIRD_PARTY_LICENSES/Apache-2.0.txt`
- `BSD-3-Clause` — `THIRD_PARTY_LICENSES/BSD-3-Clause.txt`
- `GPL-2.0` — `THIRD_PARTY_LICENSES/GPL-2.0.txt`
- `GPL-3.0` — `THIRD_PARTY_LICENSES/GPL-3.0.txt`
- `LGPL-3.0` — `THIRD_PARTY_LICENSES/LGPL-3.0.txt`
- `MIT` — `THIRD_PARTY_LICENSES/MIT.txt`
- `MPL-2.0` — `THIRD_PARTY_LICENSES/MPL-2.0.txt`

## Full dependency license texts

Every installed Python distribution carries its own license files in its
`.dist-info` directory. The offline bundle ships the wheels, so those
texts travel with the install. The standard texts above cover the
components whose wheels do not carry a text (notably PySide6/Qt).
