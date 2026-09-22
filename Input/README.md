# Input Directory

Place dialogue source files here for rendering.

## Sample Files

- `Read Aloud transcript.txt` - Example dialogue
- `What is, reality.txt` - Example dialogue
- `stream_of_consciousness_dialogue_with_typos.txt` - Intentionally messy
  sample for testing the text-repair/correction modes
- `fransisco help.txt` - Short dramatic monologue sample
- `cli_short.txt` - Short CLI test input
- `local_audio_test.txt` - Local-audio render test input
- `test.txt` - Basic test input
- `READ_THIS_TO_RECORD_SEASHELLS.txt` - the teleprompter script the Recording
  Studio reads aloud when recording a new reference voice; it lives here so
  the GUI's script picker finds it, and is not a dialogue sample.

## Runtime Subdirectories

`test/` is created on demand by test renders to hold their cache and runtime
artifacts. It, and the other runtime outputs under this directory (`*.pkl`,
`*.json`, `*.diff`, `*.log`, `*.flac`, `*.wav`), are ignored by Git.
