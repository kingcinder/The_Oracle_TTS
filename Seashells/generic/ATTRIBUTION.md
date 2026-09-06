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

| This file | Source file | Provenance |
|---|---|---|
| `english_reader_1.wav` | `librispeech_test_clean_6930-75918-0001.wav` | LibriSpeech test-clean, speaker 6930, 14.2s |

LibriSpeech is licensed under **CC BY 4.0** (attribution required):
*LibriSpeech: an ASR corpus based on public domain audio books*, V.
Panayotov, G. Chen, D. Povey, and S. Khudanpur, ICASSP 2015. Distributed via
OpenSLR (https://www.openslr.org/12) / Hugging Face
(https://huggingface.co/datasets/openslr/librispeech_asr), original audio
from LibriVox (public domain audiobooks).