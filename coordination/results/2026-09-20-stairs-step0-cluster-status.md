# Step 0 — cluster status for the straight-staircase investigation

**Date**: 2026-09-20 (22:17 cluster time) · **Machine**: cluster (`asaicomputemaster`) · **Request from**: laptop session `romen`

## Headline

**The nodes are NOT drained any more, but no GPU is free.** Step 1 is queued as job 12033;
its earliest projected start is ~12 h out, not "minutes". Step 2 is held, as instructed,
until step 1 is reported.

## sinfo

```
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
workq*       up 3-00:00:00      3   mix- asaicomputemaster,asaicomputenode[02-03]
```

`mix-` = MIXED with the "planned" flag (a backfill reservation), not `drained`. Per-node
GPU allocation (`scontrol show node`): **all 6 GPUs allocated** (2× RTX 6000 Ada on
`asaicomputemaster`, 2× A100 each on `node02`/`node03`). `nvidia-smi` on `asaicomputemaster`
shows the two RTX 6000s at 0% utilisation / ~0.5 GB used — i.e. reserved by other users'
idle sessions (several `sys/dashboard/.../ju*` Jupyter jobs). I did **not** run on those
GPUs outside the scheduler.

## squeue -u $USER

Before I submitted anything:

```
JOBID  NAME         ST  REASON
11916  csasr-eval   PD  DependencyNeverSatisfied   (other project, submitted 2026-09-11; left alone)
```

**11914 and 11918 are not in the queue. What happened to each** (from `sacct`):

| job | name | submitted | outcome |
|---|---|---|---|
| 11914 | go2_spec_gaps (attempt 3) | 2026-09-11 | **PENDING 6 days, then ran**: started 2026-09-17 08:47, **COMPLETED** 21:24 (12 h 37 m). `model_9999.pt` exists on local disk. |
| 11918 | go2_generalist | 2026-09-12 | **CANCELLED by uid 0 (root)** at 2026-09-17 11:04:55, never started. **No generalist checkpoint exists.** |
| 11919 | go2-eval-matrix | 2026-09-12 | CANCELLED by root, same timestamp, never started. |
| 11915 | csasr-t2 | 2026-09-11 | CANCELLED by root, same timestamp (other project). |

So the "PENDING since the 09-10 drain" picture in `cluster.json`, `findings.md` and
`docs/CLAUDE.cluster.md` is out of date. The admin apparently resumed the nodes ~09-17 and
cleared the stale queue. The generalist (arm 1 of the 2×2) **needs resubmitting** — I have
not done that (not asked; and it competes for the same GPU slots).

## Queue wait today (`sbatch --test-only`, nothing submitted by these)

| request | projected start |
|---|---|
| 1-day `gpu:a100:1` (`train_specialist_slurm.sh` as written) | **2026-09-21 17:25** on `node02` (~19 h) |
| 30-min `gpu:1`, any type | **2026-09-21 10:25** on `asaicomputemaster` (~12 h) |

Scheduling here is age-based FIFO (no fairshare — see the header of
`a100/train_specialist_slurm.sh`); 9 other jobs are pending ahead by Priority/Resources.
For reference, 11914 waited 6 days. Treat the 19 h as an optimistic estimate: it assumes
running jobs finish inside their walltimes and nobody submits ahead of me.

## `cluster_update_status.sh`

Ran as `go2_spec_gaps 9999 local <path> 11914` (the only run with a new checkpoint).
`cluster.json` now has that run `complete` with its sha256, plus hand-edited: the
generalist marked `cancelled` with the root-cancel explanation, a new `cluster_state`
block (nodes / GPUs free / wait estimate), and the step-1 job under `jobs`. It also put a
"please evaluate" note in `coordination/inbox/to-laptop.md` for the gaps checkpoint — that is
the **first Gaps specialist that isn't the archived 100 %-stepping-stones plateau**, not yet
evaluated anywhere.

## Step 1 — queued, not run

Job **12033**, `a100/eval_stairs_pyramid_slurm.sh`: the Stairs specialist
(`unitree_rl_mjlab/logs/rsl_rl/go2_spec_stairs/2026-09-05_22-37-43/model_9999.pt`) on
pinned pyramid stairs, `--terrain stairs --difficulty {0.5,0.7,0.9} --num-envs 1024 --steps
1200`, no `--keep-curricula`. `--terrain stairs` is `pyramid_stairs` + `pyramid_stairs_inv`
(`TERRAIN_CLASSES["stairs"]` in `env_cfgs.py`), matching the training mix. 30-minute walltime
so it can backfill; three evals should take ~5–8 min. Outputs:
`unitree_rl_mjlab/eval_results/stairs_pyramid/go2_spec_stairs_pyramid_d{0.5,0.7,0.9}.json`.

**Two deviations from the brief, both needed to answer what was asked:**

1. **`eval_checkpoint.py` had no termination-cause breakdown** — it reported only
   `time_out` vs a lumped "failed early (fell / illegal contact / etc)". You asked for
   `illegal_contact` vs `fell_over` vs `time_out`, so I added (additively, commit `9b60cbe`)
   per-term counts from `env.termination_manager.get_term(...)` for each finished episode,
   plus `total_distance_m` and `falls_per_100m` in the JSON, using gate-1's definition
   (`fails / distance × 100`). No existing metric changed. Terms can overlap (an env can
   trip two on one step), so the cause counts can sum to slightly more than `episodes`.
   **This is untested on a GPU** — I only checked syntax and the mjlab API
   (`_term_dones` survives the in-step reset). If it crashes on the first difficulty the job
   still runs the other two and the traceback will be in `go2-stairs-pyr-eval-12033.err`.
2. **`eval_ckpts/go2_spec_stairs/model_9999.pt` doesn't exist** in this checkout —
   `eval_ckpts/` holds only PAS checkpoints. The specialist checkpoint is at the
   `logs/rsl_rl/...` path above (also mirrored on private HF `RohanRamesh/go2-specialists`
   per `cluster.json`). Same file, `model_9999.pt`.

## What I have not done

- No step 2 work (no env cfg, no task registration, no training submission).
- No resubmission of the generalist / eval matrix.
- Did not cancel the stale `11916 csasr-eval` (other project).
