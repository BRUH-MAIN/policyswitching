# Stairs v5a, final checkpoint: still 12 cm; and why it refuses taller steps

**Date**: 2026-10-07, night (laptop) · **Checkpoint**: `go2_spec_stairs_v5a/model_9999.pt`
(job 12594, finished 16:06 IST) · **Status**: the eval is a result; the cause is a measured
candidate, under test by a training run (`StairsV6a`, section 3).

## 1. Final v5a on real riser heights

`scripts/switch_follow_real_stairs.sh`, 5-step straight flights, tread 0.30 m, leader at
0.5 m/s, training sensor noise on, 256 trials per cell (seeds 800, 801). Success % / fall % /
lost % (lost = the leader walked away from it).

Only tipping over counts as a fall (`--terminations saro`; v5 trains with shin contact
penalised, not terminal):

| riser | up | down |
|---|---|---|
| 9 cm | 100.0 / 0.0 / 0.0 | 100.0 / 0.0 / 0.0 |
| 12 cm | 96.5 / 1.6 / 2.0 | 76.2 / 0.0 / 23.8 |
| 15 cm | 0.0 / 0.4 / 99.6 | 0.0 / 0.0 / 100.0 |
| 17 cm | 0.0 / 0.0 / 100.0 | 0.0 / 0.0 / 100.0 |

Any knee or shin contact over 10 N counts as a fall (`training`):

| riser | up | down |
|---|---|---|
| 9 cm | 99.6 / 0.4 / 0.0 | 68.8 / 31.2 / 0.0 |
| 12 cm | 84.4 / 13.3 / 2.3 | 38.7 / 37.5 / 23.8 |
| 15 cm | 0.0 / 8.6 / 91.4 | 0.0 / 0.0 / 100.0 |
| 17 cm | 0.0 / 0.4 / 99.6 | 0.0 / 0.0 / 100.0 |

Against `model_4400` (128 trials per cell, 10-07 morning): going down 12 cm went from 1% to
76%; everything else is unchanged. The second half of the run bought one cell. **At 15 cm and
17 cm it crosses nothing in either direction, and every failure is a stall.** Not a
real-stairs policy. Raw: `unitree_rl_mjlab/eval_results/switch_follow/real_stairs/v5a_it9999_n5_*`.

## 2. What the reward pays for standing still

`scripts/diag_reward_terms.py` runs a checkpoint in its own training env (rewards, noise,
pushes and command range as trained) with the riser pinned and the row rule off, 256 robots
for 1,000 steps, and averages every reward term over steps on which the robot is commanded
above 0.3 m/s, split by what it was doing. Values are reward per second (weight × term).

Final v5a at a pinned 15 cm riser. It never goes onto the flight, so "moving" is walking on
the 3 m platform:

| term | moving on the platform | stalled |
|---|---|---|
| track_linear_velocity | 0.84 | 0.30 |
| track_angular_velocity | 0.98 | 0.97 |
| pose | 0.87 | 0.96 |
| foot_gait | 0.47 | 0.45 |
| everything else | −0.08 | −0.06 |
| **total** | **3.07** | **2.62** |

**A robot told to walk that stands and trots on the spot keeps 85% of the reward.** Posture,
gait and angular tracking pay in full for not going.

Reward per second while actually on the flight (1.6-3.0 m from the spawn point), against
that 2.6:

| riser | checkpoint | on the flight, down | on the flight, up |
|---|---|---|---|
| 9 cm | final | 2.72 | 2.68 |
| 12 cm | final | 2.52 | 2.47 |
| 15 cm | iteration 400 (final never goes there) | 1.78 | 1.87 |

At 15 cm the loss against standing is spread over several terms, none of them the one that
was suspected: linear tracking 0.26-0.42 (it is slow on the steps, and the term also charges
vertical body velocity), pose 0.61-0.74 (legs far from the default pose), angular tracking
0.84, action rate −0.14, orientation −0.06 to −0.08. **The shin-contact penalty is small:
−0.02 to −0.06.** A termination costs a further 4 (weight −200 × 0.02 s) plus the rest of
the episode.

So on a tall flight the reward function pays less for crossing than for refusing, before
any fall risk is counted, and the gap widens with the riser. That would produce what the
runs show: v3 (uniform rows) stopped walking; v4/v5 (adaptive rows) climb to the riser
where crossing and standing pay about the same (12 cm) and stall above it. The row rule
only decides where robots are put; it cannot make crossing pay.

Limits of this reading. It is observational: the two columns are different robots at
different moments, and the 15 cm flight numbers come from an early checkpoint with a rougher
gait. It says the incentive points the wrong way; it does not prove that reversing it is
enough to learn 15-17 cm. Raw: `unitree_rl_mjlab/eval_results/reward_diag/`.

## 3. The test: StairsV6a

`Unitree-Go2-Spec-StairsV6a` = v5a with `pose` and `foot_gait` multiplied by the share of
the commanded velocity the robot achieves along the command (clamped to 0-1; 1 when told to
stand). Checked on final v5a at 15 cm: a stalled robot now earns 1.19/s (was 2.62), a
walking one 2.72 at its present ~70% of commanded speed.

Running on the laptop GPU since 22:41 IST 10-07: warm start from v5a `model_9999`,
normaliser kept, 1,536 robots (what fits in 8 GB; the cluster uses 8,192), budget 6,000
iterations at 1.9 s each, checkpoints every 200 iterations to private HF under
`go2_spec_stairs_v6a_lap/`. Unit `go2-v6a-lap` (`systemctl --user status go2-v6a-lap`), log
`unitree_rl_mjlab/logs/train_StairsV6a_lap.log`, read with
`coordination/scripts/stairs_run_status.py`.

Decided before looking: it supports the diagnosis if the mean row passes 4.5 or rows 6-9
hold over 20% of robots while achieved speed stays above 65% of commanded, **and** a
checkpoint then crosses some 15 cm flights on the laptop. v5a for comparison: mean row
3.45-3.7, rows 6-7 at 4-6%, rows 8-9 empty, 0% at 15 cm. A higher mean row without 15 cm
crossings is not a pass.

This bears on v5c (job 12608, pending): it puts half the robots on uniformly drawn rows
under the unchanged reward, which is the pressure that made v3 stand still, at half
strength.
