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
