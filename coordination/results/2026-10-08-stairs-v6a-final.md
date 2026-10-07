# Stairs v6a (laptop run), final: 15 cm both ways, 17 cm down only; ascent is unstable

**Date**: 2026-10-08, 03:00 IST (laptop) · **Run**: `go2_spec_stairs_v6a_lap`, 6,000 iterations at
1,536 envs from v5a `model_9999`, finished 02:30 IST · **Status**: result for the final
checkpoint; the checkpoint to put on the robot is not chosen yet (section 2).

## 1. Final checkpoint (`model_5999`) on real riser heights

`scripts/switch_follow_real_stairs.sh`: 5-step straight flights, tread 0.30 m, leader at
0.5 m/s that does not wait, training sensor noise on, 256 trials per cell (seeds 800, 801).
Success % / fall % / lost %.

Only tipping over counts as a fall:

| riser | up | down |
|---|---|---|
| 9 cm | 100.0 / 0.0 / 0.0 | 100.0 / 0.0 / 0.0 |
| 12 cm | 99.6 / 0.0 / 0.4 | 100.0 / 0.0 / 0.0 |
| 15 cm | 85.2 / 1.6 / 13.3 | 100.0 / 0.0 / 0.0 |
| 17 cm | 2.7 / 9.8 / 87.5 | 99.2 / 0.8 / 0.0 |

Any thigh or shin contact over 10 N counts as a fall:

| riser | up | down |
|---|---|---|
| 9 cm | 83.2 / 16.8 / 0.0 | 27.3 / 72.7 / 0.0 |
| 12 cm | 95.3 / 4.7 / 0.0 | 39.1 / 60.9 / 0.0 |
| 15 cm | 57.0 / 37.9 / 5.1 | 28.9 / 71.1 / 0.0 |
| 17 cm | 1.2 / 37.1 / 61.7 | 34.0 / 66.0 / 0.0 |

Against v5a final (up 100 / 96.5 / 0 / 0, down 100 / 76 / 0 / 0, tipping only): descents are
solved to 17 cm, ascent gained one riser height. **It brushes the steps with its shins on
about two descents in three**; it is trained with that penalised, not forbidden. Against the
acceptance bar (90% up and down at 17 cm): **fails going up**.

## 2. The 15 cm ascent swings from checkpoint to checkpoint

Same flight, up 15 cm, tipping only, 64 trials per checkpoint (one seed):

| iteration | 1200 | 1400 | 1600 | 1800 | 2000 | 2400 | 2800 | 3200 | 3600 | 4000 | 4400 | 4800 | 5200 | 5600 | 5999 (256 trials) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crossed % | 2 | 56 | 98 | 88 | **0** | 84 | 78 | 75 | 36 | 47 | 22 | 12 | 41 | 80 | 85 |

Nearly every failure is "lost": the robot stops at the foot of the flight and the leader
walks on. Down 15 and 17 cm were 98-100% at every checkpoint looked at. In the training env
the share of robots crossing a pinned 15 cm up-flight in one episode is about the same at
iteration 1,600 (56%) and at the end (62%), so the skill is there throughout and whether the
policy commits to the flight at the follow controller's command is what flips. **A single
checkpoint's ascent number is not a property of the run.** Choosing one for the robot needs
the full 256-trial run on each candidate; `model_1600` is being run now.

## 3. 17 cm going up: slow, and not paid

- With a leader that cannot be lost and a 60 s limit, up 17 cm: `model_1600` crosses 17% (66%
  tip over), `model_3200` 48% (34% tip over), median ~34 s. So it can climb 17 cm, slowly,
  and tips over when the follow controller pushes it to full speed.
- Sensor noise is not the limit: with it off, up 17 cm is 0% and 6% at iterations 1,600 and
  3,200 (15 cm: 94% and 89%).
- Robots crossing a pinned 17 cm up-flight in one training episode: 3% at iteration 600, 25%
  at 1,600, 24% at 3,200, 14-15% at the end. Not improving after 1,600.
- Reward per second at a pinned 17 cm riser, final checkpoint (`diag_reward_terms.py`): on the
  up-flight **0.86**, stalled at its foot **0.77**, walking on flat ground 2.4. On a 15 cm
  up-flight 1.62. The gating in v6a took away the wage for standing; it did not make a tall
  climb pay, and at a discount of 0.99 (a 2 s horizon) the flat ground beyond a 10-30 s climb
  does not count.

## 4. Next run: StairsV7a (training on the laptop since 02:50 IST)

`Unitree-Go2-Spec-StairsV7a` = v6a plus `climb_progress`: 5.0 x vertical base velocity
(signed, so hopping in place nets nothing) on up-flights only, when commanded to move. A
five-step 17 cm flight is worth 4.25 reward-seconds in total. Checked on final v6a at 17 cm:
the up-flight now pays 2.0/s against 0.8 stalled; the term is zero on down-flights. It also
logs the up and down row means, which v6a did not, and its overall mean row (flat at ~5.75
from iteration 500) showed none of the above.

Warm start from v6a `model_5999`, 1,536 envs, 6,000 iterations, due ~06:05 IST; unit
`go2-v7a-lap`, log `unitree_rl_mjlab/logs/train_StairsV7a_lap.log`, HF folder
`go2_spec_stairs_v7a_lap/`.

Decided before looking: it helps if a checkpoint crosses 17 cm up-flights at 50% or better on
the standard flights (256 trials) with down 17 cm still above 90% and up 15 cm above 90%; and
it is better than v6a only if the 15 cm ascent stops swinging (three consecutive checkpoints
above 80%). Speed staying above 85% of commanded is a condition, not a result.

## 5. v5c for comparison (cluster job 12608)

`model_800`, same quick flights: 0% at 12 / 15 / 17 cm in both directions, all stalls; 0-2%
of robots cross a pinned 15 cm flight per training episode. The cluster's log has its up and
down row means flat since iteration 100. Its reward is v5a's, with half the robots placed on
uniformly drawn rows.

Raw: `unitree_rl_mjlab/eval_results/switch_follow/real_stairs/v6alap_*`, `v5c_it800_*`,
`eval_results/reward_diag/v6alap_*`.

## 6. Added 03:50 IST: v6a `model_1600` in full, and the first v7a checkpoint

**v6a `model_1600`**, same acceptance run as section 1. Tipping only: up 100 / 100 / 95.3 /
0.5%, down 100 / 100 / 100 / 98.4% at 9 / 12 / 15 / 17 cm. Shin contact counted: up 94 / 71 /
26 / 0%, down 56 / 50 / 43 / 50%. (The tipping-only cells at 12, 15 and 17 cm are 192 trials,
not 256: the seed-800 files already existed from the 64-trial quick look and were reused.)
So of the two v6a checkpoints evaluated in full, `model_1600` is better going up 15 cm (95%
against 85%) and neither goes up 17 cm.

**v7a `model_1600`** (laptop, 1,600 iterations from v6a final), quick flights, 64 trials per
cell, one seed, tipping only: **up 15 cm 98%, up 17 cm 52%** (12% tip over, 36% left behind),
down 15 and 17 cm 100%. In its training env at a pinned 17 cm riser 57% of robots cross the
up-flight in one episode (v6a: 14-25%) and 83% the down-flight. Log at iteration ~1,640:
speed 98-99% of commanded, up row mean 6.6, down 5.9, rows 8-9 at 32%. One checkpoint; the
sweep over checkpoints and the 256-trial runs are what section 4's pass marks ask for.
