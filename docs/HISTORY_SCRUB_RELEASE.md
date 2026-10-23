# History scrub release procedure

This repository currently contains removed personal recordings in historical
commits. Do not run these commands against a shared remote without the owner's
explicit approval.

> **Conflict with the current tree.** `Seashells/ASSETS.md` and `EULA-DRAFT.md`
> (section 3) record the root-level `Seashells/*.wav` clips and the `fransisco`
> clips as Owner-owned recordings that ship with the software. The `--path`
> list below names those same clips as "removed personal recordings". The
> `filter-repo` command here removes them from every commit, including the
> working tree the release ships from. Resolve which is true before running it.

## Verify a scrub in a disposable mirror

```sh
git clone --no-local . /tmp/oracle-history-scrub
cd /tmp/oracle-history-scrub
git filter-repo --force \
  --path "Seashells/Cody's Seashell.wav" \
  --path "Seashells/Cody's Seashell1.wav" \
  --path 'Seashells/Seashell_ fransisco.wav' \
  --path 'Seashells/Seashell_No_1.wav' \
  --path 'Seashells/Seashell_No_2.wav' \
  --path 'Seashells/seashell fransisco.wav' \
  --path 'Seashells/seashell fransisco_TTS_clean_v5_aggressive_targeted_cleanup.wav' \
  --path 'Input/fransisco help.txt' \
  --invert-paths
git log --all --name-only --format='' | grep -E "Cody's Seashell|fransisco|Seashell_No_[12]|Input/fransisco help" && exit 1 || true
```

The verified local mirror had 464 commits and zero matches after filtering.

## Publish choice

After review, either replace the remote history with the scrubbed refs using a
coordinated force push, or create a new orphan repository from the scrubbed tree.
The current remote is intentionally unchanged until that choice is approved.
