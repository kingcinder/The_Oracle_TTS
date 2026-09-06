# Generic Voices — Attribution

The clips in this folder are **bundled generic voice references** for The
Oracle's "Default Voices" picker. They are **not** user recordings; they are
third-party reference clips redistributed with permission under their
respective licenses. The Oracle renders every voice through Chatterbox voice
cloning, so these clips condition the model the same way any user-recorded
reference does.

Language note: a reference clip conditions the model on the *speaker's
timbre*, and the cloned voice then speaks the rendered text. English clips
yield English narration without accent influence; clips recorded in other
languages yield that speaker's voice with a corresponding accent — pick
accordingly. English-conditioned clips are listed first in the voice picker
and are the CLI's default fallback.

## audio.cpp demo voices (Apache License 2.0)

Copyright 2026 ShugoAI LLC — source:
[audio.cpp](https://github.com/0xShug0/audio.cpp) (`webui/voice/`), license
https://www.apache.org/licenses/LICENSE-2.0. The demo clips' own prompt
texts (see `audio.cpp/webui/voice/prompt_text`) confirm the languages below.

| This file | Source file (audio.cpp `webui/voice/`) | Speaker |
|---|---|---|
| `english_male_1.wav` | `demo_01_man.wav` | male, English ("Cemo") |
| `chinese_male_1.wav` | `demo_3_man.wav` | male, Chinese |
| `chinese_male_2.wav` | `zh-Bowen_man.wav` | male, Chinese |
| `chinese_female_1.wav` | `demo_02_woman.wav` | female, Chinese |
| `chinese_speaker_1.wav` | `ref-w1.wav` | Chinese (gender unlabeled in source) |

## LibriSpeech ASR corpus (CC BY 4.0)

| This file | LibriSpeech clip | Provenance |
|---|---|---|
| `english_reader_1.wav` | `test-clean/6930/75918/6930-75918-0001` | speaker 6930, 14.2s |
| `english_reader_2.wav` | `test-clean/1995/1836/1995-1836-0004` | speaker 1995, 33.9s |
| `english_reader_3.wav` | `test-clean/2094/142345/2094-142345-0008` | speaker 2094, 31.6s |
| `english_reader_4.wav` | `test-clean/4970/29093/4970-29093-0006` | speaker 4970, 29.6s |

The clips are single-speaker, clean-condition audiobook narration (the
LibriSpeech corpus does not publish reader sex per speaker, so these are
labeled neutrally as "English Reader"). LibriSpeech is licensed under
**CC BY 4.0** (attribution required): *LibriSpeech: an ASR corpus based on
public domain audio books*, V. Panayotov, G. Chen, D. Povey, and S.
Khudanpur, ICASSP 2015. Distributed via OpenSLR
(https://www.openslr.org/12) / Hugging Face
(https://huggingface.co/datasets/openslr/librispeech_asr), original audio
from LibriVox (public domain audiobooks).