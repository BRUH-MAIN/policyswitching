# Stairs v4a / v4b: built and CPU-checked, not submitted

**Date**: 2026-10-06 · **Machine**: cluster, `cluster-sess` · **Request**:
`coordination/inbox/to-cluster.md` 2026-10-06 (run 2, two variants). **Status**: built,
config-checked on CPU, rule replayed offline (jobs 12579, 12580). **No v4 iteration has run
and nothing is submitted**: that waits for Rohan in the cluster session. The 4b decision
(it changes the task definition) is his too.

## What was built

Tasks `Unitree-Go2-Spec-StairsV4a` / `V4b`, experiments `go2_spec_stairs_v4a` / `v4b`
(`SPEC=StairsV4a` / `StairsV4b` in `a100/train_specialist_slurm.sh`, warm start from stairs v2
`model_9999` by default, seed 42, 10k iterations, 8192 envs). Both are StairsV3 with:

- **Terrain / commands / observations / runner unchanged**: risers 5-20 cm, treads 0.30 and
  0.26 m, stage-0 commands, 234 observations, `clip_actions=6.0`. I kept the full 5-20 cm
  range as you suggested; the rows gate exposure.
- **Adaptive rows, `terrain_levels_progress`**, robots start on rows 0-3. At episode end,
  with `cheb = max(|dx|, |dy|)` from the spawn (the stairs span 1.5-3.0 m, so `cheb >= 3.2 m`
  means all five steps are behind the robot; Chebyshev because the steps are square rings)
  and `cmd` = |commanded xy velocity| on the last step:
  - **promote**: ran to the time limit, not terminated, and `cheb >= 3.2 m`;
  - **demote**: terminated, **or** (ran to the time limit and `cmd > 0.3 m/s` and
    `cheb < 1.9 m`), i.e. never got off the platform or the first step;
  - **stay**: otherwise (part-way up the flight; told to stand).
  Same wording is in the task docstring. A promotion past row 9 re-draws the row uniformly
  (mjlab's behaviour), so a robot that clears the top row lands on a random row.
- **Logged every iteration**: `Episode_Metrics/{cmd_speed, actual_speed, cmd_moving, stalled}`
  (achieved/commanded ~ `actual_speed / cmd_speed`; stalled fraction of commanded-to-move
  steps ~ `stalled / cmd_moving`; thresholds 0.1 and 0.05 m/s, as in `eval_checkpoint.py`),
  plus V3's `Curriculum/terrain_rows_*` and `terrain_row_mean`.
- **4b only**: the `illegal_contact` termination (>10 N) now matches only trunk and hip geoms
  (`base1-3`, `*_hip`: 7 geoms); thigh and calf geoms (12) feed a new `limb_contact` reward
  at **-2.0 per step in contact** (>10 N). Reward is scaled by dt = 0.02, so that is -0.04 per
  step, about one step's whole positive reward; a termination is -4 on the step and forfeits
  the rest of the episode (~0.04 per remaining step). A 10-step knee brush costs -0.4, a
  scrape held for a whole 100-step climb -4, both well under a termination. `fell_over`
  (70 deg) is unchanged. 4a's rewards and terminations are exactly V3's.

Also: `coordination/scripts/stairs_run_status.py <go2-spec-JOB.out>` prints the table and
applies the agreed stop test at ~500 and ~1500 iterations (speed < 30% of commanded, or mean
row not rising by 1500). It reports "metrics MISSING" rather than passing when the speed
metrics are absent (it first passed v3's log, which has none, as "ok"; fixed). Tested on
synthetic logs: healthy, stands still, slow-but-rising.

## Checks done (CPU)

Terrain, command stages, actor/critic observation terms and event order equal V3's;
V4a rewards and terminations equal V3's weight for weight; V4b adds exactly `limb_contact`
at -2.0. Contact sensors, counted from the built env: 4a one sensor, 19 geoms (3 base, 4 hips,
4 thighs, 8 calves); 4b trunk sensor 7 geoms, limb sensor 12. A 120-step rollout of each
logs the new metrics and `Curriculum/terrain_levels`, starts no robot above row 3, and a
standing robot's levels fall. V3 and the generalist tasks are untouched. Not run on a GPU;
the warm-start path is the one V3 used.

## The rule, replayed offline (v2 policy and v3's stand-still policy, rows held fixed)

Same machinery as 12562 (`diagnose_terrain_curriculum.py`, 2048 envs x 4000 steps, 8,250
episodes each), now with the progress rule as an alternative:

| policy (on v2's 0-10 cm rows) | rule | promoted / demoted, rows 1-3 | stationary mean row |
|---|---|---|---|
| stairs v2 (walks) | old `terrain_levels_vel` | ~42% / ~47% | 2.5 |
| stairs v2 (walks) | **progress-gated** | ~49-52% / 16-20% | **5.4-5.5** |
| stairs v3 (stands) | old | 0% / 92-96% | 0.0 |
| stairs v3 (stands) | **progress-gated** | **0% / 73-79%** | **0.0** |

So the gate does what was asked: a robot that stands still is demoted ~3 episodes in 4, never
promoted, and ends at row 0, while a walking policy is promoted about half the time and
settles at row 5-6. A survival-only rule would not separate them (its chain parks the
stander at 2.4-3.6). Caveats:

- **~17-20% of a competent policy's episodes are demoted as stalls on easy rows** (it times
  out, was told to move on the last step, and ended within 1.9 m): wandering and heading
  turns, not a failure. It is small against ~50% promotion, but it pulls the equilibrium down
  and it is not zero.
- This is one policy on 0-10 cm rows. The rule has never driven training, and what it does
  to a policy at 15 cm is not measured.
- `across_m = 3.2` rewards crossing in any direction, up or down, in either spawn type.

## Things that need Rohan, or care

- **4b changes the task definition** (a thigh/calf touch is not terminal). Its
  `illegal_contact` and falls/100 m are not comparable with any earlier policy's, and the
  eval's falls count only trunk/hip contacts. Compare 4a and 4b by crossing, speed and
  stalled fraction.
- **Two runs are ~9 h each** (v3: 9h01m). Both fit if two GPUs are free; check `sinfo` /
  `squeue` before submitting.
- **Nobody is watching a run unless asked.** The 500- and 1500-iteration check is the
  script above; it does not run by itself. If a run is submitted and nothing prompts this
  session, a failing run burns its full 9 h.
- A stall penalty or progress reward is held for run 3, as agreed.

## To submit (when Rohan says so)

    export HF_TOKEN=$(cat ~/.hf_token)
    SPEC=StairsV4a sbatch --gres=gpu:1 a100/train_specialist_slurm.sh
    SPEC=StairsV4b sbatch --gres=gpu:1 a100/train_specialist_slurm.sh   # his call

Check at ~500 and ~1500 iterations: `python3 coordination/scripts/stairs_run_status.py
go2-spec-<job>.out`. After training: `TASK=Unitree-Go2-Spec-StairsV4a LABEL=stairs_v4a
CKPT=<model_9999> sbatch a100/eval_stairs_heights_slurm.sh` (same for 4b).
