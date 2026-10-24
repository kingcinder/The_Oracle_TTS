# Third-Party Notices

This file is generated from the release tree by
`python scripts/generate_third_party_notices.py`. Do not edit it by hand;
regenerate it and review the diff. The inventory is also read by the
commercial launch gate (`python scripts/soft_launch.py --check`), so a
dependency that is missing here is a release blocker, not a documentation
omission.

Release version: **1.4.2**

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
- **External tools** (ffmpeg, audio.cpp/ggml) are not part of the Python
  wheel. The host supplies ffmpeg; the optional Vulkan backend builds
  audio.cpp/ggml locally.

## Bundled assets, models, and external components

| Component | Version / pin | License | Source |
|---|---|---|---|
| `PySide6 / shiboken6 / Qt` | 6.8.2.1 | LGPL-3.0 (or commercial Qt license) | [source](https://code.qt.io/cgit/pyside/pyside-setup.git/) |
| `audio.cpp` | built at install time from a pinned clone | Apache-2.0 | [source](https://github.com/0xShug0/audio.cpp) |
| `ffmpeg` | host-provided | LGPL/GPL depending on the build | [source](https://ffmpeg.org/) |
| `ggml` | vendored in the audio.cpp checkout | MIT | [source](https://github.com/ggml-org/ggml) |
| `oliverguhr/fullstop-punctuation-multilang-large` | pinned (models/pins.py) | MIT | [source](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) |
| `ResembleAI/chatterbox` | pinned (models/pins.py) | MIT | [source](https://huggingface.co/ResembleAI/chatterbox) |
| `ResembleAI/chatterbox-turbo` | pinned (models/pins.py) | MIT | [source](https://huggingface.co/ResembleAI/chatterbox-turbo) |
| `SamLowe/roberta-base-go_emotions` | pinned (models/pins.py) | MIT | [source](https://huggingface.co/SamLowe/roberta-base-go_emotions) |
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

Model revisions are pinned in `src/the_oracle/models/pins.py`; this
table names each model and its license without duplicating the
revision hashes, which are external identifiers rather than citations
in this repository's history.

## Python dependencies

Resolved from `pyproject.toml` and the installed package metadata. The
closure covers the runtime and the `ml`/`chatterbox` extras; development
tools are not part of the shipped environment.

| Component | Version | License | Source |
|---|---|---|---|
| `annotated-doc` | 0.0.5 | MIT | [source](https://github.com/fastapi/annotated-doc) |
| `anyio` | 4.15.1 | MIT | [source](https://anyio.readthedocs.io/en/latest/) |
| `audioread` | 3.1.0 | MIT | [source](https://github.com/beetbox/audioread/issues) |
| `catalogue` | 2.0.10 | MIT | [source](https://github.com/explosion/catalogue) |
| `certifi` | 2026.10.23 | MPL-2.0 | [source](https://github.com/certifi/python-certifi) |
| `cffi` | 2.1.1 | MIT-0 | [source](https://cffi.readthedocs.io/) |
| `cfgv` | 3.5.0 | MIT | [source](https://github.com/asottile/cfgv) |
| `charset-normalizer` | 3.5.2 | MIT | [source](https://github.com/jawah/charset_normalizer/blob/master/CHANGELOG.md) |
| `chatterbox-tts` | 0.1.7+oracle1 | MIT | [source](https://github.com/resemble-ai/chatterbox) |
| `click` | 8.5.0 | BSD-3-Clause | [source](https://click.palletsprojects.com/page/changes/) |
| `cloudpickle` | 3.1.2 | BSD-3-Clause | [source](https://github.com/cloudpipe/cloudpickle) |
| `conformer` | 0.3.2 | MIT | [source](https://github.com/lucidrains/conformer) |
| `decorator` | 5.3.1 | BSD-3-Clause | — |
| `deepmultilingualpunctuation` | 1.0.1 | MIT | [source](https://github.com/oliverguhr/deepmultilingualpunctuation) |
| `diffusers` | 0.38.0 | Apache-2.0 | [source](https://github.com/huggingface/diffusers) |
| `distlib` | 0.4.4 | PSF-2.0 | [source](https://github.com/pypa/distlib) |
| `editdistpy` | 0.4.0 | MIT | [source](https://github.com/mammothb/editdistpy/blob/master/CHANGELOG.md) |
| `einops` | 0.8.2 | MIT | [source](https://github.com/arogozhnikov/einops) |
| `filelock` | 3.32.3 | MIT | [source](https://py-filelock.readthedocs.io) |
| `fsspec` | 2026.7.0 | BSD-3-Clause | [source](https://filesystem-spec.readthedocs.io/en/latest/changelog.html) |
| `h11` | 0.16.0 | MIT | [source](https://github.com/python-hyper/h11) |
| `hf-xet` | 1.7.0 | Apache-2.0 | [source](https://huggingface.co/docs/hub/xet/index) |
| `httpcore` | 1.0.9 | BSD-3-Clause | [source](https://www.encode.io/httpcore) |
| `httpx` | 0.28.1 | BSD-3-Clause | [source](https://github.com/encode/httpx/blob/master/CHANGELOG.md) |
| `huggingface_hub` | 1.33.0 | Apache-2.0 | [source](https://github.com/huggingface/huggingface_hub) |
| `identify` | 2.6.20 | MIT | [source](https://github.com/pre-commit/identify) |
| `idna` | 3.20 | BSD-3-Clause | [source](https://github.com/kjd/idna/blob/master/HISTORY.md) |
| `importlib_metadata` | 10.0.0 | Apache-2.0 | [source](https://github.com/python/importlib_metadata) |
| `Jinja2` | 3.1.6 | BSD-3-Clause | [source](https://jinja.palletsprojects.com/changes/) |
| `joblib` | 1.6.0 | BSD-3-Clause | [source](https://joblib.readthedocs.io) |
| `lazy-loader` | 0.6 | BSD-3-Clause | [source](https://scientific-python.org/specs/spec-0001/) |
| `librosa` | 0.11.0 | ISC | [source](https://librosa.org) |
| `llvmlite` | 0.50.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | [source](http://llvmlite.readthedocs.io) |
| `markdown-it-py` | 3.0.0 | MIT | [source](https://markdown-it-py.readthedocs.io) |
| `MarkupSafe` | 3.0.3 | BSD-3-Clause | [source](https://palletsprojects.com/donate) |
| `mdurl` | 0.1.2 | MIT | [source](https://github.com/executablebooks/mdurl) |
| `ml_dtypes` | 0.5.4 | Apache-2.0 | [source](https://github.com/jax-ml/ml_dtypes) |
| `mpmath` | 1.3.0 | BSD-3-Clause | [source](http://mpmath.org/) |
| `msgpack` | 1.2.3 | Apache-2.0 | [source](https://msgpack.org/) |
| `networkx` | 3.6.1 | BSD-3-Clause | [source](https://networkx.org/) |
| `nodeenv` | 1.11.0 | BSD-3-Clause | [source](https://github.com/ekalinin/nodeenv) |
| `numba` | 0.68.0 | BSD-3-Clause | [source](https://numba.pydata.org) |
| `numpy` | 1.26.4 | BSD-3-Clause | [source](https://numpy.org) |
| `omegaconf` | 2.4.0 | BSD-3-Clause | [source](https://github.com/hydra-ecosystem/omegaconf) |
| `onnx` | 1.23.2 | Apache-2.0 | [source](https://onnx.ai/) |
| `packaging` | 26.3 | Apache-2.0 OR BSD-2-Clause | [source](https://packaging.pypa.io/) |
| `pillow` | 13.0.0 | MIT-CMU | [source](https://pillow.readthedocs.io/en/stable/releasenotes/index.html) |
| `platformdirs` | 4.13.5 | MIT | [source](https://platformdirs.readthedocs.io/en/latest/changelog.html) |
| `pooch` | 1.9.0 | BSD-3-Clause | [source](https://www.fatiando.org/pooch) |
| `pre_commit` | 4.7.0 | MIT | [source](https://github.com/pre-commit/pre-commit) |
| `protobuf` | 7.36.2 | BSD-3-Clause | [source](https://developers.google.com/protocol-buffers/) |
| `pycparser` | 3.12 | BSD-3-Clause | [source](https://github.com/eliben/pycparser) |
| `pydub` | 0.25.1 | MIT | [source](http://pydub.com) |
| `Pygments` | 2.21.0 | BSD-2-Clause | [source](https://pygments.org) |
| `pyloudnorm` | 0.2.0 | MIT | [source](https://github.com/csteinmetz1/pyloudnorm) |
| `PySide6` | 6.8.2.1 | LGPL-3.0 | [source](https://pyside.org) |
| `PySide6_Addons` | 6.8.2.1 | LGPL-3.0 | [source](https://pyside.org) |
| `PySide6_Essentials` | 6.8.2.1 | LGPL-3.0 | [source](https://pyside.org) |
| `python-discovery` | 1.6.4 | MIT | [source](https://github.com/tox-dev/python-discovery/releases) |
| `PyYAML` | 6.0.3 | MIT | [source](https://pyyaml.org/) |
| `regex` | 2026.9.29 | Apache-2.0 AND CNRI-Python | [source](https://github.com/mrabarnett/mrab-regex) |
| `requests` | 2.34.2 | Apache-2.0 | [source](https://requests.readthedocs.io) |
| `resemble-perth` | 1.1.0 | MIT | [source](https://github.com/resemble-ai/perth) |
| `rich` | 15.0.0 | MIT | [source](https://rich.readthedocs.io/en/latest/) |
| `s3tokenizer` | 0.3.0 | Apache-2.0 | [source](https://github.com/xingchensong/S3Tokenizer) |
| `safetensors` | 0.8.0 | Apache-2.0 | [source](https://github.com/huggingface/safetensors) |
| `scikit-learn` | 1.6.1 | BSD-3-Clause | [source](https://scikit-learn.org) |
| `scipy` | 1.15.2 | BSD-3-Clause | [source](https://scipy.org/) |
| `setuptools` | 84.2.1 | MIT | [source](https://github.com/pypa/setuptools) |
| `shellingham` | 1.5.4 | ISC | [source](https://github.com/sarugaku/shellingham) |
| `shiboken6` | 6.8.2.1 | LGPL-3.0 | [source](https://pyside.org) |
| `sounddevice` | 0.5.6 | MIT | [source](https://python-sounddevice.readthedocs.io/) |
| `soundfile` | 0.13.1 | BSD-3-Clause | [source](https://github.com/bastibe/python-soundfile) |
| `soxr` | 1.1.0 | LGPL-2.1-or-later | [source](https://github.com/dofuuz/python-soxr) |
| `spacy_pkuseg` | 1.0.1 | MIT | [source](https://github.com/explosion/spacy-pkuseg) |
| `srsly` | 2.5.4 | MIT | [source](https://github.com/explosion/srsly) |
| `sympy` | 1.13.1 | BSD-3-Clause | [source](https://sympy.org) |
| `symspellpy` | 6.9.0 | MIT | [source](https://github.com/mammothb/symspellpy) |
| `threadpoolctl` | 3.7.0 | BSD-3-Clause | [source](https://github.com/joblib/threadpoolctl) |
| `tokenizers` | 0.23.3 | Apache-2.0 | [source](https://github.com/huggingface/tokenizers) |
| `torch` | 2.6.0+cpu | BSD-3-Clause | [source](https://pytorch.org/) |
| `torchaudio` | 2.6.0+cpu | BSD-3-Clause | [source](https://github.com/pytorch/audio) |
| `torchvision` | 0.21.0+cpu | BSD-3-Clause | [source](https://github.com/pytorch/vision) |
| `tqdm` | 4.70.1 | MPL-2.0 | [source](https://tqdm.github.io) |
| `transformers` | 5.20.0 | Apache-2.0 | [source](https://github.com/huggingface/transformers) |
| `typer` | 0.27.3 | MIT | [source](https://github.com/fastapi/typer) |
| `typing_extensions` | 4.16.0 | PSF-2.0 | [source](https://github.com/python/typing_extensions/issues) |
| `urllib3` | 2.8.0 | MIT | [source](https://github.com/urllib3/urllib3/blob/main/CHANGES.rst) |
| `virtualenv` | 21.14.13 | MIT | [source](https://thanks.dev/u/gh/gaborbernat) |
| `zipp` | 4.1.2 | MIT | [source](https://github.com/jaraco/zipp) |

## Required notices

The following notices must accompany the distribution. The standard
license texts are in `THIRD_PARTY_LICENSES/`.

- **PySide6 / shiboken6 / Qt** — LGPL-3.0. A proprietary application may use the LGPL path while keeping its own source proprietary, provided the LGPL-covered libraries remain replaceable/relinkable, their complete corresponding source is made available (or a written offer is given), and the LGPL notice and text accompany the distribution. A Qt commercial license is the alternative.
- **audio.cpp** — Copyright 2026 ShugoAI LLC. Apache-2.0. Modified by The Oracle's vendored patches; see scripts/patches/.
- **ffmpeg** — Not bundled. The host supplies ffmpeg; its own build determines the exact license.
- **ggml** — Copyright (c) 2023-2026 The ggml authors. MIT license.
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
