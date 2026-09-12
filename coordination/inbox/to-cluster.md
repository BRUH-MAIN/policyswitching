# Inbox: laptop -> cluster

What to implement/change next, decided from eval results. Insert new entries
directly below "## Open" (top of that section, most recent first) with a
dated header. Cluster session: when you pick one up, note it in
`coordination/log/<date>-cluster.md` rather than deleting it here, then move
it under "## Done" (with a one-line pointer to what changed) once implemented
and merged -- don't delete an entry you're not sure was seen.

<!-- Example entry:

## 2026-09-11 -- bump Gaps specialist's flat/rough blend if attempt 3 plateaus again
If job 11914 (20% stepping_stones / 40% flat / 40% random_rough) plateaus like
attempts 1-2 did, try 10% stepping_stones / 45% flat / 45% random_rough next --
see findings.md "Terrain specialists" table for the plateau signature to check for
(reward flat at -6 to -8, episodes ending in ~10-14 steps).

-->

## Open

## 2026-09-12 (3) -- bug #12 fix candidate CONFIRMED, ready to apply

Ran the test the previous entry proposed: same PAS oracle checkpoint, `--terrain flat
--difficulty 0.5`, generator overridden in a throwaway script (not `env_cfgs.py`) to
keep `num_rows` at its configured value (10) and set only `difficulty_range=(0.5, 0.5)`
plus `max_init_terrain_level=None`. Result: **0% fall, 100% full-length episodes** --
fully reproduces the difficulty-unpinned result (was 98.9% fall with `num_rows=1`).
Confirms `num_rows=1` is the cause, not the difficulty arithmetic, exactly as diagnosed
below. Please apply the proposed one-liner in `apply_eval_conditions`'s pinned branch:

```python
gen = replace(gen, difficulty_range=(difficulty, difficulty))
cfg.scene.terrain.max_init_terrain_level = None
```

in place of the current `gen = replace(gen, num_rows=1, difficulty_range=(difficulty,
difficulty))`. Once that lands, job 11919 (and any `--difficulties` run) should be safe
again. Full detail in `findings.md` bug #12 (now includes this confirmation). Moving on
to a difficulty-uniform PAS oracle-vs-estimator comparison next (unaffected by this bug),
per orchestrator-session's steer.

## 2026-09-12 (2) -- bug #12 diagnosis: `num_rows=1` is unnecessary, and is the variable that breaks the pinned path

`env_cfgs.py` is yours, so this is a diagnosis plus a proposed one-line fix rather
than a patch. Context: bug #12 in findings.md -- `apply_eval_conditions`'s
difficulty-pinning path scores PAS oracle at 98.9% fall on `flat` and 99.3% on
`rough` while `stairs` scores 0.8%, backwards from what capability predicts. The
laptop isolated it to the pinning path by re-running the same checkpoint and
terrain with no `--difficulty`: 0% fall, 100% full-length episodes.

**The pin does not need `num_rows=1`.** In mjlab's `terrain_generator.py`,
`_generate_curriculum_terrains` sets each patch's difficulty as:

```python
lower, upper = self.cfg.difficulty_range
difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
difficulty = lower + (upper - lower) * difficulty
```

When `lower == upper == d`, the second line evaluates to exactly `d` for **every**
row, independent of `num_rows` and independent of the random draw. So
`difficulty_range=(d, d)` on its own already pins every patch at exactly the
requested difficulty. The `num_rows=1` in

```python
gen = replace(gen, num_rows=1, difficulty_range=(difficulty, difficulty))
```

contributes nothing to the pinning, and is exactly the variable the laptop's
isolation implicates. It also collapses the terrain's x-extent to a single patch
and forces every env onto row 0 (`_compute_env_origins_curriculum` then clamps
`max_init_terrain_level=5` to `min(5, num_rows-1) = 0`, so the spread over rows
that the unpinned path gets is gone too -- note the `difficulty is None` branch
sets `max_init_terrain_level = None` explicitly while the pinned branch never
touches it).

**Proposed fix**, to be confirmed by the test below rather than applied blind:

```python
gen = replace(gen, difficulty_range=(difficulty, difficulty))
cfg.scene.terrain.max_init_terrain_level = None  # all rows now identical difficulty
```

i.e. drop `num_rows=1` from the pinned branch and let envs spread across rows that
are all generated at difficulty `d`. This keeps the pin exact, restores the terrain
extent, and makes the pinned and unpinned paths differ only in difficulty -- which
is what the eval condition was supposed to mean.

**Caveat, please don't skip it.** The laptop measured mean episode length 30.2
steps on the failing `flat` cells -- under a second. That is too fast to be
"walked off the end of a one-patch-deep terrain", and points at something wrong at
spawn/reset on a 1-row grid rather than at the difficulty arithmetic. So treat the
above as "`num_rows=1` is the culprit, and here is a fix candidate that removes it",
not as a confirmed causal chain. If the test still fails, the next suspect is env
origin z / initial base height on a 1-row grid, not the difficulty computation.

The laptop is running the check now, without editing `env_cfgs.py`: same checkpoint,
same terrain, generator overridden in a throwaway script to keep the configured
`num_rows` and set only `difficulty_range=(0.5, 0.5)` plus
`max_init_terrain_level=None`. If `flat`'s fall rate collapses from 98.9% toward
~0%, that confirms it. Result will follow here and in findings.md #12.

**Until this is settled, job 11919 as designed produces suspect numbers in every
difficulty-pinned cell** -- all of `--difficulties 0.25/0.5/0.75`, not just `flat`.
`stairs` looking healthy in the same run is the asymmetric partial breakage that
would make the matrix plausible-looking but wrong. Worth holding 11919 (it is
PENDING anyway) until the fix lands, rather than spending its 12h walltime on cells
we would discard. `--terrain <class>` with no `--difficulty` is unaffected and safe.


## 2026-09-12 (2) -- `apply_eval_conditions`'s difficulty pin looks broken for at least `flat`; check before job 11919 runs

Running the laptop slice of the eval matrix (PAS stage2, `--anneal-prob 1.0`, 128 envs,
1200 steps) via `a100/eval_matrix.py --only pas_oracle --difficulties 0.5`: `flat` scored
98.9% fall / mean episode length 30.2 steps, `rough` 99.3%, `stairs` 0.8%, `gaps`/`mixed`
in between -- backwards from what capability should predict, since flat ground should be
the easiest terrain in the set, not the hardest.

Isolated with one more run: same checkpoint, same terrain class (`--terrain flat`), but
**no `--difficulty`** (generator's default multi-row layout, difficulty spread uniformly
instead of pinned to a single row) -- **0% fall, 100% full-length episodes**, same 128
envs / 1200 steps. The only variable that changed between the two runs was the
difficulty-pinning path (`apply_eval_conditions` in `env_cfgs.py`: `num_rows=1,
difficulty_range=(d, d)`), so this looks like a terrain-generator artifact of collapsing
to a single row with one sub-terrain type at 100% proportion (spawn position/origin
indexing is my best guess, not confirmed), not a real PAS capability gap.

Full writeup + numbers: `findings.md` bug **#12**. Since `stairs` happened to come out
looking healthy in the same run, this isn't a uniform "everything breaks" failure --
which is exactly what makes it dangerous to miss: job 11919 would produce a matrix that
looks complete and plausible with some cells silently corrupted. Worth checking
`apply_eval_conditions` (or mjlab's terrain generator underneath it) for what changes
about spawn/origin computation when `num_rows=1` and only one sub-terrain type has
nonzero proportion, **before 11919 gets GPU time** -- `--terrain <class>` with no
`--difficulty` is not affected and is safe to use meanwhile. Not something I can fix from
here per `docs/CLAUDE.laptop.md` (task-config code is cluster-owned); flagging rather than
touching `env_cfgs.py` myself.

## 2026-09-12 -- user decisions on both blockers, plus the four changes they imply

User has now decided both of the things that were pending across the three sessions.
Recording them here so the decision is durable and not just in chat scrollback.

**Decision 1 -- cluster git auth: repo-scoped SSH deploy key.** Not `gh` on the
cluster (plaintext token in `~/.config/gh/hosts.yml` on a shared filesystem), not
a PAT in the remote URL, not "user pushes manually each time". Concretely, on the
cluster:
```
ssh-keygen -t ed25519 -f ~/.ssh/policyswitching_deploy -N ""
```
then hand the user the **public** half to add at BRUH-MAIN/policyswitching ->
Settings -> Deploy keys, with **write access** ticked (it is off by default -- a
read-only deploy key fetches fine and fails on push, which would look exactly like
the bug we just spent a round diagnosing). Then
`git remote set-url origin git@github.com:BRUH-MAIN/policyswitching.git`, plus a
`Host github.com / Hostname ssh.github.com / Port 443` block in `~/.ssh/config` if
outbound 22 is blocked from the node.

**Decision 2 -- specialists back up to a NEW PRIVATE HF repo, not `go2-pas-saro`.**
Verified anonymously (no token): `RohanRamesh/go2-pas-saro` returns HTTP 200 with
`private: false`, `gated: false`, and its `stage1/model_*.pt` files are listed --
it is genuinely world-readable, so syncing specialists there would publish them.
The user does not want unpublished experimental checkpoints public. Create a
separate **private** model repo for the specialists (`create_repo(..., private=True)`)
and point `HF_CHECKPOINT_REPO` at it. **Leave `go2-pas-saro` exactly as it is** --
the user was offered flipping it private and declined, so do not change its
visibility.

The laptop's SSH-key route was *not* chosen, so `laptop_pull_and_eval.sh`'s rsync
branch stays broken by design. HF is now the only path specialists reach this
machine by. That makes the four items below load-bearing rather than cleanup.

### 1. One-off backfill upload for the three finished specialists (unblocks everything, no GPU)
`go2_spec_flat` / `go2_spec_rough` / `go2_spec_stairs` were trained without
`HF_TOKEN`, so they exist only on cluster local disk. `HfSyncVelocityOnPolicyRunner`
only fires on `runner.save()` inside a live training call, so as you already noted
this needs a small standalone script over `push_checkpoint_to_hf` in
`rl/hf_upload.py` -- not a retrain, and not a re-submission.

Worth doing first: it needs no GPU, so it runs on the login node **while every node
is still drained**. It is the one thing on either side's list that the drain doesn't
block. Upload each `model_9999.pt` to `<private-repo>/<experiment_name>/model_9999.pt`,
matching the layout `train_specialist_slurm.sh` already uses
(`HF_CHECKPOINT_STAGE="$EXPERIMENT_NAME"`), so a backfilled checkpoint and a
future synced one land at the same path.

### 2. Future runs: set `HF_TOKEN` at submission, with `HF_CHECKPOINT_REPO` -> the private repo
Otherwise 11914/11918/11919's outputs repeat this whole problem when they finally run.

### 3. `a100/eval_matrix.py`: specialists need the same HF fallback PAS just got
This is the same bug you found and fixed for PAS, still live for every entry in
`LOCAL_POLICIES`. They resolve only through `latest_local_checkpoint(mjlab_dir,
experiment)`; on the laptop that finds nothing and takes the `[SKIP]` branch.

The failure mode is worth being precise about, because it is not a crash: the skip
prints one line to stdout, then `summary.md` is rendered from `order`, which is
built from the result rows that actually exist. So a laptop-run matrix produces a
clean, plausible-looking summary table containing only the PAS rows, with no marker
in the file itself that five policies were never evaluated. Anyone reading
`summary.md` later -- including us -- would have to notice an absence rather than
an error. Give each `LOCAL_POLICIES` entry an `hf_stage` (= its experiment name) and
the same local-then-HF fallback as PAS.

### 4. `cluster_update_status.sh` / `cluster.json`: specialist entries move to `source: "hf"`
`laptop_pull_and_eval.sh` branches on `run.source`; with `source: "local"` it takes
the rsync path, which now cannot work. Once #1 lands, the three specialist entries
need `"source": "hf"` plus `"hf_repo"` and `"hf_stage"` (both read at lines 58-59),
and the script should write those fields for future runs rather than only
`path_on_cluster`. Fold in the previously-noted gap while you are in there: the
script never writes `task`, and `laptop_pull_and_eval.sh` hard-errors on
`task == MISSING` (line 67), so any run_id it creates fresh breaks the laptop side.

I own the objective.md 2x2 pass and the stale `docs/CLAUDE.cluster.md`; those are
still coming and are not blocked on any of the above.

## Done
