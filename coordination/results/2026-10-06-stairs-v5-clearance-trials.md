# Stairs v5a / v5b: the world-height clearance term was the cause of the slow-down

**Date**: 2026-10-06 (trials ran 17:43-18:53 IST) · **Machine**: cluster, `cluster-sess` ·
**Jobs**: 12589 (v5a), 12590 (v5b), 600 iterations each, both COMPLETED, exit 0 ·
**Status**: training-log result only. No pinned or flight eval of either yet. The full 10k run
of v5a is queued as 12594 and is waiting for a free GPU.

## What was tested

Each variant is v4b (progress-gated rows from 0-3, limb contact penalised, speed/stall metrics,
warm start from v2 with the normalizer kept) plus ONE change to `foot_clearance`, the term
`|foot_z_world - 0.10| * foot speed` (weight -1.0) whose world-frame height charges a robot for
the staircase it stands on:

- **v5a**: the term removed.
- **v5b**: the term measured above the robot's lowest foot (`mdp.feet_clearance_relative`), same
  weight, target and gate.

## Result (±15-iteration means)

| run | iteration | speed / commanded | stalled | mean row | tracking | clearance term | illegal_contact | reward |
|---|---|---|---|---|---|---|---|---|
| v4b_kn (stock term) | 150 / 300 / 500 | 0.26 / 0.28 / 0.30 | 0.11 / 0.10 / 0.09 | 0.33 / 0.04 / 0.04 | 0.53 / 0.55 / 0.56 | -0.28 / -0.25 / -0.26 | 0.00-0.06 | 46-47 |
| **v5a** (removed) | 150 / 300 / 500 | **0.72 / 0.71 / 0.67** | 0.02 / 0.02 / 0.03 | **1.98 / 2.78 / 3.34** | 0.75 / 0.74 / 0.72 | none | 0.08-0.12 | 50 |
| **v5b** (relative) | 150 / 300 / 500 | **0.67 / 0.66 / 0.66** | 0.03 | **1.82 / 2.34 / 2.85** | 0.73 / 0.72 / 0.73 | -0.10 / -0.10 / -0.09 | 0.06-0.09 | 49 |

Both interventions fix it: speed 65-72% of commanded where v4b sat at 26-30%, stalls 2-3%
instead of 10%, and the mean row rises through the whole trial (v5a 1.4 -> 3.4, v5b 1.4 -> 3.0;
rows 6-7 hold 9-12% of robots, rows 8-9 about 1%) instead of collapsing to row 0. The
clearance penalty in v5b is -0.09 against -0.26 for the stock term, so most of the stock
term's charge was the staircase height. Reward 49-50 against 46-47.

So the earlier readings were right in direction and the normalizer story was a side issue: with
the normalizer kept the policy still slowed to 28% because the stock term made moving the feet
costly on stairs; fixing the term, not the warm start, is what restored speed. This also fits v3,
the old runs sitting on 1-2 cm rows, and reward not telling slow from walking.

## What this does not show

- **Training logs only.** No pinned eval and no flight-harness eval of either checkpoint. Speed
  and row climb are training measurements; whether it crosses 12-17 cm flights is not known.
- **Nothing rewards swing height in v5a.** `foot_gait` and the geometry must do it; an eval will
  show whether it drags its feet or clips nosings. `illegal_contact` (0.09/iteration) is not
  zero and rose from 0.00 as the robots climb.
- **The rows are still low after 600 iterations**: mean row 3 means mostly risers under 12 cm.
  Whether it reaches rows 8-9 (17-20 cm) is the question the 10k run answers.
- **v5a vs v5b is not resolved**: 67% vs 65% speed and 3.4 vs 3.0 mean row are one seed each.
  v5a is simpler and went further; v5b keeps a swing-clearance incentive. I chose v5a for the
  full run because only one GPU slot is available, not because the difference is established.
- Row 0 spawn: the first 25 iterations show 54-55% speed and reward 26-27 before it recovers,
  as in every warm start so far.

## State and next steps

- `go2_spec_stairs_v5a/model_599.pt` and `go2_spec_stairs_v5b/model_599.pt` are on private HF.
- **12594** = v5a to 10,000 iterations (resumes from `model_599`), submitted ~22:15 IST, PENDING:
  all six GPUs are held by other users' jobs and 12578 (csasr-abl, a different project of this
  account). The scheduler's walltime-based estimate is 2026-10-08 17:03, a worst case; a freed
  GPU starts it sooner. Shorter walltimes do not change the estimate.
- After it runs: `coordination/scripts/stairs_run_status.py`, then
  `TASK=Unitree-Go2-Spec-StairsV5a LABEL=stairs_v5a CKPT=<model> sbatch a100/eval_stairs_heights_slurm.sh`.
