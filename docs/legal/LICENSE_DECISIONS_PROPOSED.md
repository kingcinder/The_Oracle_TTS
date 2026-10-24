# Proposed License Decisions — 1.5.0 RC

Status: proposal for owner and qualified counsel. This file does not approve
any distribution decision and must not be copied into
`scripts/license_review.json` without that approval.

## Installed component evidence

The versions and metadata below were read from the managed CPU environment at
`/media/cody/Data/oracle-clean-install-YvsR8r/tree/.venv` on 2026-10-23.

| Component | Installed version | Metadata license | Proposed handling | Open verification |
|---|---:|---|---|---|
| `torch` | 2.6.0+cpu | BSD-3-Clause | Separate notice and allow only after counsel confirms the PyTorch bundle terms | Read the exact wheel license files for the selected CPU/CUDA wheels |
| `torchaudio` | 2.6.0+cpu | missing | Separate notice; conditional approval only | Confirm upstream audio license and wheel notices |
| `torchvision` | 0.21.0+cpu | BSD | Separate notice; conditional approval only | Confirm exact upstream license text |
| `chatterbox-tts` | 0.1.7+oracle1 | MIT | Proposed allow, with upstream MIT notice and local metadata patch disclosure | Counsel confirms the patched distribution and model terms |
| `huggingface-hub` | 1.33.0 | Apache-2.0 | Proposed allow with Apache notice | Confirm installed license file is shipped |
| `PySide6` / Qt | 6.8.2.1 | LGPL | Proposed allow only with replaceable libraries, source offer, and relink instructions | Frozen-package replacement test and counsel review |
| `soundfile` | 0.13.1 | BSD 3-Clause | Proposed allow with notice | Confirm bundled libsndfile strategy |
| `symspellpy` | 6.9.0 | MIT | Proposed allow with notice | Confirm upstream text is shipped |
| `deepmultilingualpunctuation` | 1.0.1 | UNKNOWN | Hold until upstream license is confirmed; replace or separate if unresolved | Read repository/model license and record evidence |
| `soxr` | 1.1.0 | missing | Hold pending exact license text and replacement obligations | Confirm wheel/source license |
| `transformers` | 5.20.0 | Apache-2.0 | Proposed allow with Apache notice | Confirm exact version's notice set |
| `cloudpickle` | 3.1.2 | BSD-3-Clause | Proposed allow with notice | Included in regenerated notices |
| `certifi` | 2026.10.23 | MPL-2.0 | Proposed allow with MPL notice | Confirm notice/source obligations |
| `tqdm` | 4.70.1 | MPL-2.0 AND MIT | Proposed allow with both notices | Confirm exact wheel license files |

## Model-card evidence

The pinned model-card `README.md` files were fetched at the exact revisions in
`src/the_oracle/models/pins.py` on 2026-10-23. Each reported `license: mit`:

- `ResembleAI/chatterbox` — `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`.
- `ResembleAI/chatterbox-turbo` — `749d1c1a46eb10492095d68fbcf55691ccf137cd`.
- `oliverguhr/fullstop-punctuation-multilang-large` — `345e80adc07e761d3a35feafd20f2f44a151f453`.
- `SamLowe/roberta-base-go_emotions` — `d75048347613a25d77de8cf6412eaae9fa7b26be`.

The card text reviewed did not state a non-commercial restriction. That is an
engineering observation, not a legal conclusion; counsel should confirm that
the cards, model files, and any training-data terms permit the intended paid
distribution.

## LGPL replacement proposal

The frozen installer must keep PySide6/Qt libraries replaceable, publish a
corresponding-source offer or source package, and document relinking. The
current source/venv path can be replaced by changing the environment's
PySide6 libraries, but a frozen artifact replacement test has not yet been
performed. Until that test passes, frozen distribution is not commercially
approved.

## Owner/counsel action

Copy only approved decisions into `scripts/license_review.json`, then run
`scripts/soft_launch.py --check --json`. No decision is recorded in that file
by this proposal.
