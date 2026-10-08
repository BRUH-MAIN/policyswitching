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

## 7. Added 05:00 IST: v7a through iteration 3,200. 15 cm is stable; 17 cm up tips over

Up-flights, 64 trials per checkpoint, one seed, noise on, tipping only, leader at 0.5 m/s
that does not wait (success / fall / lost %):

| v7a iteration | up 15 cm | up 17 cm |
|---|---|---|
| 800 | 45 / 14 / 41 | 3 / 13 / 84 |
| 1200 | 98 / 0 / 2 | 42 / 2 / 56 |
| 1600 | 98 / 0 / 2 | 52 / 13 / 36 |
| 2000 | 94 / 5 / 2 | 33 / 20 / 47 |
| 2400 | 94 / 6 / 0 | 64 / 20 / 16 |
| 2800 | 88 / 2 / 11 | 42 / 11 / 47 |
| 3200 | 83 / 9 / 8 | 30 / 16 / 55 |

Down 15 and 17 cm at iterations 1,600 and 3,200: 98-100%.

- **15 cm going up no longer swings**: 83-98% at every checkpoint from 1,200 on (v6a: 0-98%).
  That meets the stability mark set in section 4. It drifts down after 1,600, and falls
  appear from 2,000 on; the early checkpoints (1,200, 1,600) are the candidates.
- **17 cm going up is 30-64% and not rising after iteration 1,200.** The "lost" trials stop at
  the foot of the flight for six seconds or more while the leader walks on (they end at the
  first riser at ~10 s; successes take ~10 s in total).
- **With a leader that cannot be lost** (60 s limit): iteration 2,400 crosses 73% and tips
  over 27%; iteration 3,200 crosses 69% and tips over 31%; with a slow leader (0.25 m/s)
  iteration 3,200 crosses 45% and tips over 55%. So given time it always attempts the 17 cm
  flight, and **it tips over on about three attempts in ten** (more when commanded slowly).
  "Tips over" here is the base tilting past 70 degrees. Falls come mid-flight, in every
  direction.
- In the training env at a pinned 17 cm riser, iteration 3,200: 79% of robots cross the
  up-flight in one episode (v6a: 14-25%), 5% never pass the first step.

Reading: the reward for height gained removed the refusal at 17 cm and made the 15 cm ascent
dependable. What is left at 17 cm is not willingness but balance: a 0.6 s trot on a 30-degree
staircase with risers over half the leg length. **v7a is a 15 cm policy. At 17 cm it would
put the real robot on its back about one time in three.** The pass mark of section 4 for
17 cm (50% on the standard flights) is not met on average and would not be good enough if it
were. Full 256-trial runs on iterations 1,600, 2,400 and the last follow the end of the run.

## 8. Added 05:45 IST: the cluster's v7a run at 8,192 envs goes up 17 cm; v5c closed

**Cluster v7a** (job 12613, started 04:10, `go2_spec_stairs_v7a/` on HF): `SPEC=StairsV7a` at
8,192 envs, warm start from the laptop's v7a `model_1600`. Its `model_1000`, quick flights,
64 trials per cell, one seed, noise on, tipping only (success / fall / lost %):

| | 15 cm | 17 cm |
|---|---|---|
| up | 100 / 0 / 0 | **98.4 / 1.6 / 0** |
| down | 96.9 / 3.1 / 0 | **75.0 / 23.4 / 1.6** |

Up 17 cm with a leader that cannot be lost: 100% cross, none tip over, median 9.7 s (the
laptop run: 69-73% cross, 27-31% tip over). So with five times the batch the ascent at 17 cm
is clean after 1,000 iterations, where the laptop's 1,536 envs sat at 30-64% for 3,000. At
the same checkpoint going **down** 17 cm tips over on 23% of flights, which no v6a or laptop
v7a checkpoint did (98-100%). One checkpoint, 64 trials per cell: later checkpoints and the
256-trial runs decide whether the descent loss is real and whether it lasts.

**v5c final** (`model_3999`, job 12608 finished): 0% at 12 / 15 / 17 cm up and down, all
stalls, as at `model_800`. Closed.

## 9. Added 07:05 IST: cluster v7a `model_1600` passes on 5-step flights and fails on 10-step ones; StairsV8a

**Cluster v7a `model_1600`** (`go2_spec_stairs_v7a/`, job 12613), 256 trials per cell, noise
on, leader at 0.5 m/s that does not wait. Success % at 9 / 12 / 15 / 17 cm:

| flight | fall definition | up | down |
|---|---|---|---|
| 5 steps | tipping only | 100 / 100 / 100 / **98.8** | 100 / 100 / 100 / **98.8** |
| 5 steps | shin contact counted | 95 / 88 / 87 / 48 | 85 / 92 / 68 / 70 |
| 10 steps | tipping only | 100 / 100 / 93 / **45** | 100 / 100 / **72** / **41** |
| 10 steps | shin contact counted | 95 / 75 / 58 / 9 | 84 / 73 / 9 / 3 |

On 5-step flights this is the first checkpoint to meet the acceptance bar (90% up and down
at 17 cm, tipping only). On 10-step flights it does not: going down it tips over on 28% of
15 cm flights and 59% of 17 cm ones, and going up 17 cm it tips over on 32%. Every policy so
far trained on 5-step flights only; a building flight is 8-12 steps. (The descent dip at
`model_1000`, 75% at 17 cm, was gone by `model_1600`: 100% on the quick flights.)

**Laptop v7a, three checkpoints in full** (5-step, tipping only, up then down at 9 / 12 / 15 /
17 cm): iteration 1,600: 100 / 100 / 94.5 / 52.7 and 100 / 100 / 99.6 / 98.4; iteration 2,400:
100 / 100 / 94.1 / 69.1 and 100 / 100 / 100 / 100; final: 100 / 99.6 / 89.5 / 30.5 and 100 / 100
/ 100 / 100. So at 1,536 envs v7a is a 15 cm policy, as section 7 said; at 8,192 envs the
same task reaches 17 cm. The batch size was the difference.

**StairsV8a** = v7a on 10-step flights (platform 1.2 m, border 0.4 m in the same 8 m patch;
11 steps at the 0.26 m tread), row-rule distances and spawn jitter adjusted to match.
Training on the laptop from cluster v7a `model_1600` since 07:00 IST (unit `go2-v8a-lap`,
1,536 envs, 4,000 iterations, HF `go2_spec_stairs_v8a_lap/`). Pass mark, set now: on 10-step
flights, 256 trials, tipping only, at least 90% up and down at 15 cm with 5-step results not
worse than the table above; 17 cm on 10 steps is the stretch. Given what batch size did for
v7a, the laptop run is a first look and the 8,192-env version is the one to judge.

## 10. Added 11:30 IST: StairsV8a passes the acceptance on 5- and 10-step flights (tipping only)

The first laptop v8a launch (07:00) never trained: the 10-step terrain needs about a third
more GPU memory per env, 1,536 envs ran out of memory on all 30 restarts, and I reported it
as training after one iteration. Relaunched 09:17 at **1,024 envs** (5.5 GB peak), from
cluster v7a `model_1600`.

**v8a_lap `model_2400`**, `scripts/switch_follow_real_stairs.sh`, 256 trials per cell, noise
on, leader at 0.5 m/s that does not wait. Success % at 9 / 12 / 15 / 17 cm:

| flight | fall definition | up | down |
|---|---|---|---|
| 10 steps | tipping only | 100 / 100 / 99.6 / **95.3** | 100 / 100 / 100 / **99.6** |
| 5 steps | tipping only | 100 / 100 / 100 / **100** | 100 / 100 / 100 / **100** |
| 10 steps | shin contact counted | 94 / 88 / 49 / 14 | 65 / 52 / 5 / 5 |
| 5 steps | shin contact counted | 93 / 97 / 70 / 34 | 75 / 68 / 31 / 31 |

**This meets the acceptance bar of the plan (at least 90% up and down at 17 cm on 5- and
10-step flights) with tipping over as the failure.** The remaining tip-overs are going up
17 cm on 10 steps (3.9%). With any thigh or shin contact over 10 N counted as a failure it
is far from it: on long 15-17 cm descents a shin touches a step on 19 flights in 20. The
policy is trained with that contact penalised, not forbidden; whether it is acceptable on a
real nosing is a question for the robot, not for this table.

Quick looks along the run (10 steps, 64 trials, tipping only; up 15 / up 17 / down 15 / down
17): iteration 800: 98 / 75 / 100 / 100; 1,600: 100 / 91 / 100 / 100; 2,400: 100 / 95 / 100 /
100. **Cluster v8a** (job 12614, 8,192 envs, `go2_spec_stairs_v8a/`) `model_600`: 100 / 95 /
100 / 100.

Exported as `deploy_numpy/stairs_v8a_lap2400.npz`; through the robot's runner and the
height-map sampler it climbs a 17 cm flight in plain CPU MuJoCo. Scan-fault runs (delay,
missing cells, height bias) at 15 cm risers and on easy layouts are in progress.

## 11. Added 12:35 IST: v8a `model_2400` with a robot-like height scan

`scripts/switch_follow_scan_faults.sh`, three randomised mixed layouts (seeds 900-902; flat,
rough, a 5-step flight up and one down), 128 trials each, noise on, tipping only, the policy
alone. Success % (384 trials per cell). The riser heights below are read from the result
files' course geometry (a first "15 cm" run was really 7 cm: `findings.md` #35).

| scan fault | 7 cm risers | 15 cm risers | 17 cm risers |
|---|---|---|---|
| none | 100.0 | 100.0 | 100.0 |
| 40 ms late | 100.0 | 100.0 | 100.0 |
| 100 ms late | 100.0 | 100.0 | 99.5 |
| 200 ms late | 100.0 | 99.5 | 96.4 |
| 30% of cells stale each step | 100.0 | 100.0 | 100.0 |
| 60% of cells stale each step | 100.0 | 99.7 | 100.0 |
| whole scan off by up to ±3 cm | 100.0 | 100.0 | 99.7 |
| whole scan off by up to ±6 cm | 100.0 | 100.0 | 99.7 |
| 100 ms late + 30% stale + ±3 cm | 100.0 | 100.0 | 99.2 |

Against stairs v2 on 7 cm risers (section 2.2 of the plan: 94% clean, 75% with a ±6 cm
offset), v8a is far less sensitive to scan errors; in particular the ±6 cm height offset
that cost v2 a fifth of its crossings costs v8a nothing measurable. The one fault that
shows is a 200 ms delay at 17 cm (96.4%). Three layouts, 5-step flights, one checkpoint.
This also covers the "still fine on easy ground" check: the 7 cm column is 100% throughout.

With this the simulation acceptance of the plan (section 2.1) is complete for v8a_lap
`model_2400`, with tipping over as the failure criterion: 5- and 10-step flights at 9-17 cm
(section 10), scan faults at real riser heights, and easy ground.
