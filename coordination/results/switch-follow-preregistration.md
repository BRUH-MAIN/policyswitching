# Switching while following: pre-registration

**Date**: 2026-10-03 · **Machine**: `romen` · **Branch**: `vlm-pipeline` · **Written before any
switch-timing data exists.** The only data seen so far on this course is one run of the
existing no-VLM baseline (`vlm_nav_baseline.py`, `multi` L1, seed 300, 128 trials/arm, goal
controller, not the follow task): footprint-rule switching 58.6% success, fixed stairs 36.7%,
fixed rough 18.8%, fixed flat 0%; the switching arm's failures sit at the terrain transitions
(top of the up-stairs, top edge of the down-stairs). That run motivates the design; it is not
part of the result.

## What this tests

`objective.md`'s comparison: does choosing the specialist by terrain beat any one policy, does
blending beat a hard switch, and does **knowing the terrain further ahead than the onboard scan
reaches** (the followed person as preview) improve on either. It is the 2x2 of
{hard, soft} x {reactive, anticipatory}, plus fixed-policy baselines and, when its checkpoint
exists, the sensing-matched generalist.

## Two deviations from `objective.md`, both deliberate

1. **The soft arms are a parametric cross-fade, not a trained gating network.** A switch is
   described by two numbers: the lead `d_start` (metres before a terrain boundary at which the
   incoming specialist starts to take over) and `d_end` (where it has fully taken over). Hard
   switch: `d_start == d_end`. Soft: action = linear blend of the two specialists' actions in
   between. A trained gate adds seed variance and a second thing that can fail; the parametric
   family tests the same factor directly. If no schedule that needs the long horizon beats the
   best one that does not, there is no reason to expect a learned gate with look-ahead inputs
   to. If one does, a learned gate is the follow-up.
2. **Every arm reads ground-truth terrain labels; only the horizon differs.** This removes the
   confound `PROGRESS_REPORT.md` section 3 flagged (anticipatory arm on ground truth vs
   reactive arm on a noisy classifier would measure sensor quality, not horizon). The onboard
   height scan reaches 0.8 m ahead of the base, so a schedule is **reactive-feasible** iff
   `d_start <= 0.8` and **needs the leader** iff `d_start > 0.8`. Negative leads (switching
   only after the robot is already on the new terrain) are included to show what detection
   latency costs.

## Setup

- Course `multi`, level L1 (0.05 m risers, rough noise 0.02-0.06 m): flat 3 / rough 3 / flat 2 /
  stairs up 1.5 / flat 2 / stairs down 1.5 / flat 3. Defined in `course.py` before this
  experiment.
- Leader: kinematic point walking the centreline at 0.5 m/s, 3.0 m ahead of the robot's start.
  Robot commanded by `controllers.follow_command` (gap 3.0 m, ground-truth leader position and
  velocity, existing gains). 3.0 m exceeds every lead tested, so follow dynamics are identical
  across arms and the horizon enters only through which schedules are feasible.
- Terminations: the specialists' training terminations (bad orientation, or non-foot contact
  > 10 N). SARO's orientation-only definition is a labelled robustness check on the
  confirmation arms, not a primary outcome.
- A trial ends at the first of: **success** (base 1.5 m past the end of the last non-flat
  segment), **fall** (termination), **lost** (range to leader > 6.0 m), time limit.
- Exits are the same in every arm: the specialist releases when the rear of the footprint
  clears the segment (0.35 m), later by the lag for negative leads. Only entry timing varies.

## Terrain-to-specialist mapping

Two, both fixed before the timing data:

- `label`: flat -> flat, rough -> rough, stairs (up and down) -> stairs.
- `comp`: as `label` but stairs **up** -> rough. From the exploratory single-obstacle data of
  2026-09-17 (seed 101, L1): up-stairs rough 97% / stairs 81%; down-stairs stairs 66% /
  rough 25%. Up vs down is observable from the leader's height profile.

## Arms

- Fixed: `flat`, `rough`, `stairs`; `generalist` when its checkpoint exists.
- Hard, lead d in {-0.6, -0.3, 0.0, 0.3, 0.8, 1.5, 2.5} m.
- Soft `(d_start, d_end)`: (0.3, 0.0), (0.8, 0.0), (0.8, 0.3), (1.5, 0.3), (1.5, 0.8),
  (2.5, 0.3), (2.5, 0.8), (2.5, 1.5).
- Each switching arm under both mappings.

## Analysis, fixed now

**Stage 1, calibration** (seeds 400, 401; 256 trials each -> 512/arm): choose the mapping with
the higher success for hard d=0.3 (the existing footprint rule), then within that mapping pick,
by success rate (ties within 1 point: lower entry-boundary action rate), the best arm in each
cell: reactive hard (d <= 0.8), reactive soft (d_start <= 0.8), anticipatory hard (d > 0.8),
anticipatory soft (d_start > 0.8).

**Stage 2, confirmation** (fresh seeds 500, 501, 502; 256 trials each -> 768/arm): the four
selected arms, hard d=0.3, the three fixed specialists, the generalist when available. Nothing
selected on these seeds.

Claims and what counts:

- **H1 switching value**: hard d=0.3 vs the best fixed specialist (and vs the generalist).
  Supported if the success difference's 95% interval excludes 0.
- **H2 blending**: reactive soft vs reactive hard. Primary: action rate and actuator-force
  rate (mean, p99) in +-0.5 m of base travel around each terrain-class boundary, normalised
  by the same arm's whole-rollout value. Success must not be lower by more than 5 points.
- **H3 anticipation** (the project's claim): anticipatory vs reactive, within hard and within
  soft. Supported if success is higher with a 95% interval excluding 0, or success is within
  2 points and boundary smoothness is better with an interval excluding 0. **A null here is a
  result and gets reported as one**: it would mean the best moment to switch lies inside the
  onboard horizon, so the leader's preview has nothing to add for policy switching between
  these specialists on this course.
- Also reported, not gated: outcome breakdown, where falls happen relative to boundaries,
  per-step velocity-tracking error, the full calibration curve of success vs lead (the
  "gain vs preview horizon" figure), per-seed spread.

Intervals: Newcombe for a difference of proportions pooled over seeds, with the per-seed values
shown alongside so a seed-driven effect is visible.

## Known limits, stated in advance

- Single-seed specialists; one course layout family; L1 only (L2 stairs are 0-28% even with
  perfect choice, see `vlm-nav-phase1-calibration.md`).
- Selecting the best of several arms on 512 trials is optimistic; that is what stage 2 is for.
- The anticipatory arms get the best case for preview: exact labels, exact boundary location.
  A null under these conditions is strong; a positive result would still need a real sensor.

## Addendum, 2026-10-03: a real reactive arm (written before any classifier data exists)

Written after the stage 1/2 results above were in (on-time switching 91.5%, switching 1.5 m
early 74.6%, switching 0.3 m late ~60%), and before collecting a single labelled scan.

Every switching arm so far reads ground-truth labels. `objective.md`'s arm 2a is a *terrain
classifier* on the robot's own height scan. Stage 1 showed the switch has to land within a few
tenths of a metre of the boundary, so the open question is whether onboard sensing can deliver
that. If it can, the leader adds neither horizon nor labels. If it cannot, the leader's value
is as a label source, and the lead sweep says how much that is worth.

- **Arm `clf`**: an MLP on the 187-dim `height_scan` slice of the actor observation, predicting
  the class the footprint rule (hard switch 0.3 m ahead, `label` mapping) has active. Its output
  drives a hard switch through a smoothing-plus-hold filter (`scan_classifier.SwitchFilter`).
- **Training data**: single-obstacle courses `rough`, `stairs_up`, `stairs_down` at L1, seeds
  700 and 701, robot driven by hard:0.3. The `multi` course is never trained on.
- **Two sensing conditions**, each with its own classifier: observation noise as in training
  (height scan +-0.1 m; `--obs-noise`), and noise off (the twin env's default, used by every
  result above).
- **Filter**: (alpha, hold) chosen from {(1.0, 1), (0.3, 3), (0.1, 5), (0.05, 10)} by success on
  seeds 400/401; evaluated once on seeds 500-502.
- **H4**: under the same sensing condition, `clf` vs label-timed hard:0.3. "Onboard sensing
  suffices" if the 95% interval of the success difference lies inside +-5 points. "The leader is
  worth its labels" if `clf` is lower with an interval excluding 0; the size of that difference
  is the result. Also reported: agreement with the footprint rule, and the lead at which the
  classifier first selects each non-flat segment's class.
- Known limit, stated now: the stairs in `multi` are the same straight 0.05 m stairs the
  classifier trains on, so this is the easy case for a classifier. A failure here is strong; a
  success does not show it would generalise to unseen stair geometry.

## Addendum 2, 2026-10-04: generalist arm, and a better stairs specialist (written before either is run)

Both checkpoints are downloaded; neither has been run on any course. (The generalist's
iteration-0 checkpoint was run once on 16 trials as a code check of the loading path.)

**A. Generalist (arm 1).** `go2_generalist/model_9999.pt` (job 12479, seed 42, 10k iterations,
same curriculum as the specialists) as a fixed policy on `multi` L1, seeds 500-502, 768 trials,
next to hard:0.3 and stairs-only, with observation noise off and on
(`scripts/switch_follow_generalist.sh`). Claim: **switching vs the matched generalist**,
hard:0.3 minus generalist, "supported" if the 95% interval excludes 0 in the same direction in
both noise conditions; if the two conditions disagree, that is the result. Before reading
success, check the generalist is walking: lost % and tracking error (bugs #1, #14).

**B. Pre-collapse stairs specialist.** `go2_spec_stairs_it4800/model_4800.pt` is the Stairs
run's checkpoint from before the command range widened; on pyramid stairs it falls 2.6-3.6x
less often than `model_9999` (cluster, `2026-10-03-stairs-precollapse-checkpoint-eval.md`).
The early-switching penalty in the main result was traced to the stairs specialist clipping
the lip of the down-stairs when it walks the approach. So:

- Same course, task and seeds as the main experiment, with only the stairs slot of the bank
  replaced: L1 on seeds 500-502 (noise off and on), L2 on seeds 600-601 (noise off).
- Arms: fixed flat / rough / stairs, hard switch at leads -0.3, 0.0, 0.3, 0.8, 1.5.
- **H5a, does the early-switch penalty survive a competent stairs specialist?** hard:1.5 minus
  hard:0.3. Predicted: smaller than with `model_9999`. The anticipation conclusion changes only
  if early is *better* than on-time with an interval excluding 0; a difference whose interval
  includes 0 means "no cost, no gain", which still leaves nothing for a longer horizon to add.
- **H5b, does switching still beat the best fixed policy?** hard:0.3 minus the best fixed arm.
  A stairs specialist that is good on flat and rough ground too could make switching
  unnecessary; that would be reported as such.
- No selection: every arm above is reported, on the seeds named.
- Limit: `model_4800` is one earlier checkpoint of the same single-seed run, not a retrained
  specialist. Stairs v2 (job 12490) is the retrain; this is what can be tested today.

**Addendum 2, note added 2026-10-04 08:10 IST, after part A's first run and before the
replication.** Part A came out differently in the two noise conditions (seeds 500-502:
switching minus generalist -0.7 points with noise off, +9.2 with noise on). Because that
disagreement is the result and it rests on one run per condition, the same three arms are
re-run on fresh seeds 503, 504, 505 in both conditions (`TAG=generalist_rep`). Reported
alongside the first run and pooled with it; neither run is dropped whatever it shows.

## Addendum 3, 2026-10-04: the generalist's own pre-collapse checkpoint (written before it is run)

Addendum 2 compared an undamaged stairs policy (`model_4800`) with a generalist that trained
through the same curriculum collapse (`model_9999`). The fair version of "specialists plus a
switch against one generalist" has undamaged policies on both sides. `go2_generalist/model_4800.pt`
exists (job 12479 saved every 200 iterations) and has not been run on anything.

- Course `multi` L1, follow task, seeds 500-502 and 503-505, observation noise off and on,
  256 trials per seed. Bank: flat and rough as trained, stairs = `go2_spec_stairs_it4800/model_4800.pt`.
- Arms: `fixed:generalist48` (generalist iteration 4800), `fixed:generalist` (iteration 9999,
  same-run reference), `fixed:stairs` (stairs iteration 4800 alone), `hard:0.3:label`.
- Reported: all four arms, pooled over the six seeds per condition, with
  generalist48 minus generalist (did the collapse hurt the generalist too?),
  hard:0.3 minus generalist48, and stairs-alone minus generalist48.
- No selection; whatever comes out is reported. Whether the generalist's terrain curriculum
  collapsed at iteration 5000 has not been checked (asked of the cluster session); this run
  does not depend on the answer.

## Addendum 4, 2026-10-04: stairs v2 (written before its checkpoint exists)

Stairs v2 (`Unitree-Go2-Spec-StairsV2`, cluster job 12490) is the Stairs task retrained from
scratch with the command range held at stage 0. At the time of writing it is at about
iteration 8,300 of 10,000 and its terrain curriculum has not collapsed. When
`go2_spec_stairs_v2/model_9999.pt` exists it goes into the stairs slot and the Addendum 2B run
is repeated unchanged (`scripts/switch_follow_stairs_ckpt.sh`): L1 seeds 500-502 with noise off
and on, L2 seeds 600-601, the same eight arms, everything reported. Same two questions, H5a
(early minus on-time) and H5b (switch minus stairs alone), plus one comparison across
checkpoints: stairs v2 alone against `model_4800` alone, which says whether the retrain did as
well as, better than, or worse than stopping the original run early.

## Addendum 5, 2026-10-04: generalist v2 (written before it is trained)

Rohan approved retraining the generalist with the command range held (`Unitree-Go2-GeneralistV2`,
experiment `go2_generalist_v2`), the change that produced stairs v2. This is the comparison
`objective.md` calls the one the project rests on, with a properly trained policy on each side.

- Course `multi` L1, follow task, seeds 500-505, 256 trials each, observation noise off and on.
  Bank: flat and rough as trained, stairs = stairs v2.
- Arms: `fixed:generalist` (generalist v2 `model_9999`), `fixed:stairs` (stairs v2 alone),
  `hard:0.3:label`. Run with `scripts/switch_follow_generalist.sh`.
- Reported, pooled over the six seeds per condition: stairs alone minus generalist v2, switch
  minus generalist v2, and generalist v2 against the as-trained generalist's numbers from
  Addendum 2A (different runs, so read with the 2-3.5 point run-to-run spread in mind).
- Reading: if generalist v2 is within 2 points of stairs v2 alone in both conditions, "one
  well-trained policy is enough, specialist or not". If stairs v2 is ahead with an interval
  excluding 0, specialisation still pays on this course. Both arms are near ceiling at L1
  (stairs v2: 99.7-99.9%), so the same three arms are also run at L2 on seeds 600-601 (noise
  off), where there is room to differ. Whatever comes out is reported.
- Before reading success, check generalist v2 is walking (lost %, tracking error), and note
  its terrain level through iteration 5000 from the cluster log.

## Addendum 6, 2026-10-05: where does generalist v2 lose? (diagnostic, written before it is run)

Exploratory, to locate a cause, not to test a claim. Generalist v2 is better than the first
generalist on pinned pyramid stairs and worse on the mixed course, where it fails on the
straight down-staircase. Single-obstacle courses, follow task, each policy alone:

- Courses `stairs_down` and `stairs_up`; levels L1 (0.05 m) and L2 (0.07 m); leader speed 0.3,
  0.5 and 0.8 m/s; observation noise off and on; seeds 800 and 801, 256 trials each.
- Arms: first generalist, generalist v2, stairs v2 (reference), all `model_9999`.
- Reported: the full grid of success rates, and where failed trials end. Read for which
  factor (direction, riser height, speed, noise) the gap between the two generalists follows.
  No conclusion is drawn from a single cell; a cell is 512 trials.
