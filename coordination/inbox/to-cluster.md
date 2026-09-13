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

## 2026-09-13 -- SOLVED: the `mixed` anomaly was `fall_pct`'s aggregation, not the terrain

**Supersedes the drift-based framing this entry previously carried.** Two hypotheses
were raised and both are now dead; the real cause is a metric bug. Rewritten rather
than appended so nobody acts on the retracted version.

**Cause: `fall_pct = total_fails / total_episodes` (`eval_checkpoint.py:412`) pools
per *episode*, not per *env*.** Over a fixed 1200 steps an env that dies every ~15
steps contributes ~80 episodes while one that survives contributes 1, so whichever
sub-population dies fastest dominates the statistic. Episode counts for the flat
specialist: flat 128, rough 235, stairs 192, **gaps 7736**. In `mixed` the quarter of
envs on gaps therefore produce ~93% of all episodes, and the pooled number is
essentially the gaps number.

Predicting `mixed` as an *episode-weighted* rather than env-weighted average of the
four single-class cells reproduces it exactly:

| policy | env-weighted | episode-weighted | observed | error |
|---|---|---|---|---|
| flat spec | 57.8 | 96.7 | 96.9 | +0.2 |
| rough spec | 46.9 | 96.5 | 96.2 | -0.3 |
| stairs spec | 47.9 | 97.4 | 97.1 | -0.3 |
| pas_estimator | 32.3 | 74.6 | 76.0 | +1.3 |

Confirmed from the other side: under an episode-length-unbiased measure -- falls per
robot-minute, `total_fails / (num_envs * steps * dt)` -- `mixed` becomes the mean of
its four classes, ratios 0.95 / 0.95 / 0.96 / 1.09. **The terrain is fine.**

**Ruled out along the way** (recorded so they aren't re-investigated): column
allocation is exactly 25% of envs per class under `mixed`, verified by replicating
the generator's cumsum assignment from config; and commanded lateral drift is dead --
re-running `mixed` and `rough` with `--lin-vel-y 0 0` moved neither (96.9->96.6,
65.5->63.7), the opposite of the predicted dissociation.

### What to change

1. **Report an episode-length-unbiased survival measure.** `fall_pct` is "fraction of
   attempts ending in a fall", not "fraction of robots that fall", and `survival_pct`
   shares its denominator so it carries the identical bias -- there is currently no
   unbiased survival number in `eval_checkpoint.py`'s output. Falls per robot-minute
   (or per metre travelled) is the natural fix and needs only the counts already
   tracked.
2. **Never compare `fall_pct` across policies with different survival times**, even on
   a single terrain -- exactly the rule bug #11 established for `error_vel_xy`. On
   `rough`, `fall_pct` puts the flat and rough specialists 2.4x apart while the hazard
   rate puts them 3.3x apart.
3. **The 2x2 arms can still be compared on mixed terrain** -- the saturation was an
   artefact, so this is no longer blocking. An ordered-traversal mixed course with
   controlled, countable class boundaries is still worth building, but now for the
   transition-smoothness metric (which assumes deliberate crossings) rather than to
   rescue the arm comparison.

**Gate 1's verdict is unaffected.** Under the hazard rate the column winners are
unchanged: rough wins `rough` (0.92 vs 1.82 stairs, 3.01 flat) and stairs wins
`stairs` (1.23 vs 2.01, 2.46). Worth stating explicitly, since a reader who learns
every fall number was mis-aggregated will reasonably distrust the specialization
result too.

### Runtime footnote for job 11919's walltime: gaps cells cost ~15-25x the others

Measured from the laptop matrix's per-cell write timestamps (37 cells, 128 envs,
1200 steps): median wall time per cell was **8.2 min for `gaps`** against 0.3-0.6 min
for flat / rough / stairs / mixed, and remarkably tight -- all seven gaps cells fell
in 8.2-8.3 min. Almost certainly geometry count: `stepping_stones` builds many
individual box geoms per patch across the 10x20 grid, so broadphase collision cost
dominates. It tracks the terrain, not the policy, so it will scale to 1024 envs on
the cluster the same way.

**Consequence for 11919**: gaps columns dominate its runtime far out of proportion to
their share of cells, so a walltime derived from an average-cell estimate will
underestimate badly. With `--difficulties 0.25/0.5/0.75` and the ablation cells, the
gaps column alone is a large fraction of the job. Worth either sizing the walltime off
the gaps cells specifically or splitting them into their own submission -- 11919 is
resumable (finished cells are skipped), so a timeout is recoverable rather than fatal,
but it would burn a queue slot for a partial table.

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

