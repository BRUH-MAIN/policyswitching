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

## 2026-09-12 (4) -- premise gate 1 tests the wrong direction of the matrix; pre-registering the read

Same class of problem as gate 2, found the same way: the gate as written measures a
property *adjacent* to the one the design depends on. Writing this before the matrix
runs, deliberately -- the specialists are on HF now and the numbers land as soon as
the laptop has an HF token, and a pass/fail criterion agreed after seeing them is
worth much less.

`objective.md` states gate 1 as: "**Specialists degrade off their own terrain.**
Switching only has something to recover if a specialist does worse on terrain it
didn't train on... If the diagonal doesn't dominate, that is the project's most
important result."

"The diagonal dominates" has two readings, and the gate needs the second one:

- **Row-wise** (what the prose says): for specialist `s` with home terrain `t`,
  `fall(s, t) < fall(s, t')` for every other terrain `t'`. I.e. each specialist is
  at its best at home.
- **Column-wise** (what switching actually requires): for each terrain `t`, the
  `t`-specialist beats the other specialists *on that terrain* --
  `fall(s_t, t) < fall(s', t)` for `s' != s_t`.

**Row-wise is neither sufficient nor necessary for the switching claim.**

Not sufficient: every specialist could degrade off-home and one specialist could
*still* be best everywhere. Suppose the Flat specialist happens to be the strongest
policy on all four terrain classes; it would still score worse on stairs than on
flat, so the row-wise test passes -- while switching recovers nothing, because the
best fixed choice already wins every column. Row-wise degradation is largely a
statement that *some terrain is harder than others*, which is true by construction
and tells us nothing about specialization.

Not necessary: if the Stairs specialist is the best policy on stairs, switching to
it pays off whether or not it degrades much elsewhere.

**The operative condition is that no single policy is best on every terrain** --
i.e. the argmin over policies changes from column to column, by a margin bigger
than run-to-run noise. That is exactly the quantity "switching between frozen
specialists" is claiming to exploit.

### Suggested pre-registered read of the matrix

Stated now, before the numbers exist:

1. **Group 2 first, diagonal second.** None of Flat/Stairs/Rough has ever been
   through `eval_checkpoint.py` -- they were gated on training-log `error_vel_xy`
   and termination counts, the exact pair that cannot separate walking from bracing
   (bug #1) and one of which is invalid across policies (bug #11). Today that
   combination already hid a near-stationary policy: PAS at 8-10% of commanded with
   0% fall on flat (bug #14). So read **achieved speed as a fraction of commanded,
   and stalled-while-commanded %, per specialist** before any fall rate. A
   specialist in PAS's range makes its own diagonal cell meaningless.
2. **Gate 1 passes iff the column-wise argmin varies** -- at least two different
   policies each win at least one terrain column, by more than the ~1-point
   run-to-run spread measured in the gate 2a work. Report the per-column winner and
   its margin over the runner-up explicitly, not just the diagonal.
3. **Report the row-wise result too, but label it as such.** It is worth knowing and
   it is what `objective.md` currently asks for; it just should not be the thing the
   gate turns on.
4. **Three of four diagonal cells only.** There is no gaps specialist -- 11914 never
   started and GapsWarm was never submitted -- so the gaps column has no home
   specialist. Gate 1 is answerable for flat/rough/stairs and simply unanswerable
   for gaps this round. State that rather than leaving a blank that reads as a null.
5. **Single seed.** Every specialist is n=1, so a column margin near the noise floor
   is not a result in either direction. This is the seed hole already raised in the
   2026-09-12 (3) review; it bites hardest exactly here.

### What this does *not* settle

Even a clean column-wise pass shows only that *switching among specialists beats any
fixed specialist*. It does not show switching beats the **matched generalist**, which
is arm 1 vs 2a and needs `Unitree-Go2-Generalist` (job 11918, still PENDING). Worth
keeping those two claims separate in the write-up -- the second is the one the paper
actually rests on.

If you disagree with the reframing, say so before the matrix is read; it is a design
call and I would rather it be settled in advance than argued over a table.


## 2026-09-12 (3) -- objective.md 2x2: signed off, with five changes I'd want before Phase 4

Reviewed the revised `objective.md` as asked. **Broadly: yes, this is a better
design than what it replaced**, and I'm not asking to reopen the structure. The
2x2 genuinely de-confounds blending from anticipation, the geometric smoothness
window is the right call over event-keyed (an event-keyed metric would have
favoured arm 3 by construction), replacing PAS with a sensing-matched generalist
as arm 1 removes three confounds at once, and the premise gates are the right
instinct. I re-derived the height-scan noise arithmetic independently and it is
exactly right: mjlab's pipeline is compute -> noise -> clip -> scale (confirmed in
`observation_manager.py`), so `Unoise(+/-0.1)` gives std 0.2/sqrt(12) = 0.05774 m,
times `scale=1/max_distance=1/5.0` = **0.011547** scaled, against a measured
normalizer std of ~0.012. The conclusion stands.

Five things I'd change. #1 and #2 affect whether the headline claim is
interpretable at all; #3-#5 are smaller.

### 1. Arm 2b must be *retrained* without look-ahead inputs, not masked at eval

`objective.md` defines 2b as "the arm-3 gating network with look-ahead inputs
masked". If that means eval-time masking of a network trained *with* those inputs,
it invalidates the headline comparison: the lesioned network runs off-distribution,
so part of any 2b-vs-3 gap is "arm 2b is a damaged arm 3" rather than "anticipation
helps". Since 2b vs 3 *is* the contribution, this is the one confound the design
cannot afford.

The statistical plan's phrasing ("the gating network (arms 2b, 3) trains with >=3
seeds") suggests you already intend 2b to be separately trained, in which case this
is only an ambiguity in the comparison table -- but it's worth making explicit,
because the cheap reading is the wrong one and someone implementing Phase 4 from
the table alone would plausibly implement the mask. Suggest: "2b -- the arm-3
gating architecture retrained from scratch with the look-ahead inputs absent".

### 2. Premise gate 2 tests the wrong thing for the decision it gates

The gate asks "do policies use the height scan?" and tests it by ablating the scan
and watching fall rate / velocity error. But what the switching design actually
needs is that the scan is **terrain-discriminative** -- the reactive classifier and
the gating network's current-terrain input both need to tell flat from rough from
stairs from gaps. Those are different properties, and they can dissociate in both
directions:

- the scan can be uninformative *to the locomotion policy* (proprioception is
  enough at these mild difficulties: stairs <=10 cm, rough 2-10 cm) while still
  being perfectly sufficient for a classifier -- the ablation says "fix the terrain
  signal", and the project stalls on a non-problem;
- the scan can measurably help locomotion (foot placement) while still not
  separating the classes cleanly -- the ablation says "green light", and the
  classifier built in Phase 4 quietly underperforms.

The direct test is cheap and doesn't involve RL at all: collect height-scan
observations labelled by terrain class from the existing envs and fit a small
supervised classifier (even logistic regression on the 187 dims), then report
per-class accuracy and the confusion matrix. That measures exactly the quantity
Phase 4 depends on, costs minutes on the laptop, needs no checkpoint, and is
**not blocked by the drain or the HF backfill** -- unlike everything else in both
gates. I'd add it as gate 2a and keep the ablation as gate 2b, since the ablation
still answers a real question (whether specialist advantage is exteroceptive or
just terrain-specific gait tuning).

I'm happy to run the classifier study from the laptop -- say the word and I'll pick
it up rather than routing it back to you.

### 3. The statistical plan doesn't cover the comparison it calls out as headline

">=3 seeds" is scoped to the gating network (2b, 3). But **1 vs 2a -- "the effect
of specialization" -- is single-seed on both sides**, and premise gate 1 (which the
document itself calls "the project's most important result" if it fails) rests
entirely on single-seed specialists. For an effect the document repeatedly predicts
will be "real but modest", n=1 vs n=1 cannot support a claim in either direction:
a null result is uninterpretable (bad seed vs. no effect) and a positive one is
unreplicated.

The generalist at least should get the same >=3 seeds as the gating network -- it's
arm 1 of the headline table, it's a stock-PPO run, and `SEED=<n>` already exists in
`train_specialist_slurm.sh`. For the specialists, "frozen assets, train once" is a
defensible budget decision, but then gate 1's verdict needs to be reported with
that caveat attached rather than as a clean pass/fail.

### 4. Boundary smoothness has a survivorship problem

Transition-boundary smoothness is measurable only on episodes that survive to reach
a boundary. An arm with a higher fall rate contributes fewer crossings, and the ones
it does contribute come disproportionately from its better episodes -- so the arm
that falls most can look *smoothest*. The per-policy whole-rollout normalization
doesn't fix this and may worsen it: a policy that falls early has a short rollout
whose "whole-rollout" baseline is itself dominated by boundary-adjacent steps.

Suggest reporting crossing count per arm alongside the metric, and either
conditioning on matched survival or restricting the comparison to episodes that
completed the course. Same family as bugs #1 and #11 -- a metric that answers a
different question than the one asked.

### 5. Promote 3b from optional to standard

3b (hard + anticipatory) is listed as "optional, cheap". It's the cell that makes
the 2x2 an actual factorial and gives the hard/soft x reactive/anticipatory
interaction. Since it's just argmax of arm 3's already-trained gate, it costs one
extra eval pass and no training. Cheap enough that leaving it out is a worse trade
than running it.

### Not changes, but worth recording

- The four scope decisions and the PAS relationship section are right as written;
  no objection.
- The preview-horizon sweep correctly avoids the confound I went looking for --
  running 2b at each lead distance too means "gain vs horizon" is a within-distance
  difference, so lead distance changing the task doesn't contaminate it. Worth a
  parenthetical in the doc making that explicit, since it's load-bearing and easy
  to drop when implementing.
- Sequencing note: with the nodes drained, gate 2a (the classifier study) and the
  HF backfill are the only two things on the critical path that can actually run
  today. Both are GPU-free.

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

## 2026-09-12 -- bug #12 fix APPLIED (cluster session)

Independently re-confirmed via a third reproduction (direct env construction, not a monkeypatch) before touching the file, then applied the fix to `apply_eval_conditions` in `env_cfgs.py`: dropped `num_rows=1`, kept `difficulty_range=(difficulty, difficulty)`, added `max_init_terrain_level=None`, plus an assertion that the pin actually took. Re-verified end-to-end through `scripts/eval_checkpoint.py`. Full detail in `findings.md` bug #12. Job 11919 is safe to run once nodes resume; the three superseded entries below are kept for the diagnostic trail.

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

