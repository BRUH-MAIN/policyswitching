# The stairs specialist was better at iteration 4800 than at 9999

**Date**: 2026-10-03 · **Machine**: cluster (`asaicomputemaster`, session `cluster-sess`) ·
**Jobs**: 12482, 12483 (4–5 min each, one A100 on `asaicomputenode03`) ·
**Status**: measured. Stairs v2 training is **not submitted** — see "What is left to decide".

## The result

Same eval as step 1 (job 12033: pinned pyramid stairs, `--terrain stairs`, 1024 envs × 1200
steps, terrain curriculum off, eval seed 0), with two changes: the command range is the
**stage-0** training range (`lin_vel_x` −0.5…1.0, `lin_vel_y` −0.5…0.5 — what the follow task
commands, where step 1 used the widened −1…2 / −1…1), and a second checkpoint is added,
`model_4800.pt`, the last one saved clearly before the command range widened at iteration 5000.

Falls per 100 m travelled (the gate-1 metric; lower is better):

| difficulty (riser) | `model_9999` | `model_4800` | ratio |
|---|---|---|---|
| 0.5 (~5 cm) | 4.10 | **1.57** | 2.6× |
| 0.7 (~7 cm) | 12.02 | **4.07** | 3.0× |
| 0.9 (~9 cm) | 18.06 | **5.06** | 3.6× |

The earlier checkpoint falls 2.6–3.6× less often, and the gap widens with riser height.
`model_4800` at 9 cm risers (5.06) falls only a little more often than `model_9999` at 5 cm (4.10).

Everything else, same conditions:

| | d | episodes | survival % | time_out / illegal_contact / fell_over | distance (m) | achieved speed | lin-vel error / step |
|---|---|---|---|---|---|---|---|
| `model_9999` | 0.5 | 1112 | 68.9 | 766 / 345 / 2 | 8437 | 0.345 m/s (69% of cmd) | 0.213 |
| | 0.7 | 1371 | 33.8 | 463 / 906 / 5 | 7557 | 0.309 (63%) | 0.252 |
| | 0.9 | 1609 | 20.6 | 332 / 1253 / 30 | 7070 | 0.290 (59%) | 0.279 |
| `model_4800` | 0.5 | 1042 | 87.3 | 910 / 132 / 0 | 8411 | 0.343 (68%) | 0.208 |
| | 0.7 | 1081 | 70.7 | 764 / 316 / 1 | 7780 | 0.318 (64%) | 0.227 |
| | 0.9 | 1100 | 66.5 | 731 / 365 / 4 | 7297 | 0.299 (61%) | 0.250 |

**It is not bracing** (findings.md bugs #1, #14): the two checkpoints cover the same distance
at the same speed (within about 3%), and the earlier one tracks commands at least as well. It
simply hits a step edge with a knee or calf less often. `fell_over` is negligible for both. Survival % is
shown for orientation only — it pools per episode (bug #15); read the falls/100 m column.

## What this establishes

1. **The second half of training made the stairs specialist worse at stairs.** After the
   command range widened at iteration 5000 the terrain curriculum fell from level ~1.9 to
   ~0.9 and stayed there (`2026-10-03-stairs-training-curve.md`). findings.md had the
   trigger from timing and the mechanism as inference; this is the measurement that the
   policy lost capability, by a factor of about three, and did not merely stop improving.
2. **A better stairs specialist already exists on disk.** No GPU-day is needed to get one:
   `unitree_rl_mjlab/logs/rsl_rl/go2_spec_stairs/2026-09-05_22-37-43/model_4800.pt`
   (sha256 `1b1b53f41964d3e39d461dd287a8eef619807d0ee16f847e2e7f4235ba219320`). On private HF
   since the evening of 2026-10-03 as `go2_spec_stairs_it4800/model_4800.pt`.
3. **Step 1's fall rates were not a command-range artifact, but its tracking numbers
   were.** `model_9999` at the widened range vs the stage-0 range: falls/100 m 6.08 → 4.10
   at d = 0.5, 12.58 → 12.02 at 0.7, 16.95 → 18.06 at 0.9. Achieved speed goes from 24–27%
   of commanded to 59–69%, and the per-step velocity error from 0.80–0.84 to 0.21–0.28 m/s.
   So "weak on stairs generally" stands at course-like speeds; "crawls at a quarter of the
   commanded speed" does not — that was mostly the 2 m/s commands.

## What it does not establish

- **Nothing here is on the straight course.** Pyramid stairs, 20 s episodes, random
  commands. Whether `model_4800` lifts the 0–34% crossing rate at 0.07–0.09 m risers on the
  laptop's straight staircase is the laptop's test to run.
- **One training run, one eval seed.** The gap is large against sampling noise (132 vs 345
  `illegal_contact` episodes at d = 0.5), but it is one seed of each.
- **`model_4800` was picked before looking**, as "the last checkpoint clearly before
  iteration 5000", and it is the only pre-collapse checkpoint evaluated. It is not the best
  of a sweep, so there is no selection effect in the table — and no claim that 4800 is the
  best iteration either.
- **It is a proxy for stairs v2, not stairs v2.** It shows what holding the command range
  at stage 0 buys by iteration 4800. Iterations 5000–10000 at stage 0 are unmeasured.

## What this means for the stairs v2 proposal

The proposal (`coordination/inbox/to-cluster.md`, 2026-10-03 (2)) is to hold the command
range at stage 0, and to judge it by whether `terrain_levels` "keeps climbing past 5000".

- **The retrain is supported**: the policy trained that way for 4800 iterations is the
  better stairs policy by ~3×.
- **The success criterion should change.** `terrain_levels` was already flat before the
  collapse — 1.79, 1.86, 1.89, 1.87, 1.95, 1.89, 1.92 at iterations 2500…5000 — so the
  likeliest outcome of v2 is that it holds near 1.9 rather than climbs, and that would read
  as a failure under the current criterion while still being a ~3× better policy. Judge v2
  by this eval instead: falls/100 m on pinned pyramid stairs at the stage-0 range, against
  both rows above. Better than `model_4800` = the extra 5000 iterations helped; equal =
  v2 buys matched budget but no capability; worse = something else is wrong.
- **`terrain_levels` is not a 0–10 scale in practice.** The flat specialist, which no
  terrain impedes, sits at 3.91–3.97 over the same iterations under the same commands. So
  ~4 is the level a policy reaches when the promotion rule (net displacement > 4 m per
  episode) is the only limit; 1.9 is about half of that, not a fifth of ten.

## What was prepared (code only — nothing submitted)

- Task **`Unitree-Go2-Spec-StairsV2`** (`unitree_go2_spec_stairs_v2_env_cfg` in
  `env_cfgs.py`, registered in `config/go2/__init__.py`): Stairs with the `command_vel`
  curriculum cut to its first stage. Terrain mix unchanged, so a v1-vs-v2 difference has
  one cause. Checked on CPU (configs built, no env): one command stage vs two for Stairs and
  Generalist (neither altered), the same sub-terrains, the same reward, termination and
  actor-observation term names, `clip_actions=6.0`, `HfSyncVelocityOnPolicyRunner`. It has
  not been run for a single training iteration.
- `SPEC=StairsV2` in `a100/train_specialist_slurm.sh` → experiment **`go2_spec_stairs_v2`**.
  `local_ckpt_resume.py` matches the experiment directory exactly, so it starts fresh and
  cannot resume from or write into `go2_spec_stairs`. `go2_spec_stairs_v2` added to
  `cluster_update_status.sh`'s task map.
- `a100/eval_stairs_pyramid_slurm.sh` takes `EVAL_EXTRA_ARGS` (how the two jobs above set the
  command range).
- **Not built**: the mixed pyramid + `open_stairs` + `random_stairs` terrain from the
  2026-09-20 brief. Step 1 came back "fails on pyramids too", which that brief's own rule
  reads as "adding linear stairs alone won't fix it", and changing the terrain as well as
  the curriculum would leave two causes for any improvement.

## What is left to decide (Rohan)

1. ~~Get `model_4800.pt` to the laptop~~ — **done 2026-10-03 ~18:30 IST**, on Rohan's instruction: private
   `RohanRamesh/go2-specialists`, `go2_spec_stairs_it4800/model_4800.pt`, remote LFS sha256 matches the
   local file. Registered as `runs.go2_spec_stairs_it4800` in `cluster.json`. It has its own folder so
   that "highest iteration under the stage" still resolves to `model_9999` for `go2_spec_stairs`.
   (An earlier revision of this file gave a `huggingface-cli upload` command; that CLI is deprecated
   and no longer works. The upload went through `HfApi.upload_file`, as `backfill_specialist_hf.py` does.)
2. **Whether to train stairs v2** (a GPU-day; `PROGRESS_REPORT.md` §2.3 reserves this):
   ```
   export HF_TOKEN=$(cat ~/.hf_token)
   SPEC=StairsV2 sbatch a100/train_specialist_slurm.sh
   ```
   Cold start, seed 42, 10k iterations, matching the other specialists. A dry run
   (`sbatch --test-only`, ~17:20 IST) projected a start of ~18:27 IST today on
   `asaicomputenode03`; all six GPUs are allocated right now. Its value over
   item 1 is a specialist with the same 10k budget as the others, and an answer to whether
   the back half of training helps when it isn't spent on near-flat ground.

## Side notes

- **The same collapse is in the Rough run and will be in the generalist** (job 12479 reaches
  iteration 5000 about 4.7 h after its 16:43 IST start). That was a deliberate choice, to
  keep the generalist matched to the existing specialists. If the comparison is ever re-run
  with pre-collapse checkpoints, every run saves `model_4800.pt`, so a matched set exists
  without retraining anything.
- **`SPEC=Stairs` is not a no-op.** `local_ckpt_resume.py` counts `model_9999.pt` as 9999 of
  10000 done and would resume `go2_spec_stairs` for one more iteration. Another reason not to
  resubmit it; the same will apply to any finished run, the generalist included.
