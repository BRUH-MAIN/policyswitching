# Why `terrain_levels` sits at 1–2 of 10, and the stairs v3 task built from it

**Date**: 2026-10-05 · **Machine**: cluster, session `cluster-sess` · **Request**:
`coordination/inbox/to-cluster.md` 2026-10-05 (3) · **Jobs**: 12560 (discarded, see below),
12562 (diagnostic), 12561 (stairs v2 baseline at real riser heights).
**Status**: v3 built and CPU-checked. **Not submitted**: that waits for Rohan's word in the
cluster session. No v3 iteration has ever run.

## 1. The promotion rule is the cause (confirmed by replay)

Your hypothesis held, with one correction to its detail. `terrain_levels_vel` is:

    move_up   = distance_from_spawn > 4 m                      (patch is 8 m, spawn at its centre)
    move_down = distance_from_spawn < |command_xy| * 20 s * 0.5, and not move_up

For any command above 0.4 m/s the demotion threshold (`|cmd| × 10 m`) is already above the
4 m promotion threshold, so **every episode that is not promoted is demoted**. Promotion needs
the robot to walk to the patch edge: the staircase is only the inner 3 m (1.5 m platform
half-width + 5 steps × 0.3 m, then a 1 m flat border out to 4 m).

`scripts/diagnose_terrain_curriculum.py` replays the stairs-v2 final policy with every env
held on a fixed row (rows uniform 0–9, no promotion or demotion applied, stage-0 commands,
2048 envs × 4000 steps, 8,240 episodes) and records what the rule *would* have done. Its
reimplementation of the rule matches the curriculum's own decisions on 100.00% of episodes.

Rows 1–3 (risers 1–4 cm on v2's own 0–10 cm scale; row 0 has under 30 episodes per type), where the
policy times out in 98–100% of episodes, i.e. essentially never fails:

| | promoted | demoted | neither |
|---|---|---|---|
| pyramid (walks down), rows 1–3 | 41–43% | 48–50% | ~0% |
| inverted pyramid (walks up), rows 1–3 | 40–46% | 45–49% | ~0% |

So on terrain it has solved, the rule is a coin flip with a small downward bias; from row 5
up the policy starts to fall and the demotion share rises (row 9: 63% down / 24% up for
descending, 88% / 3% for ascending). **Stationary mean row under this rule: 2.47 (pyramid),
2.40 (inverted)**, from a birth–death chain over the measured per-row rates. The stairs v2
training log's mean `terrain_levels` was 1.9–2.1 over its second half. That is the same
place, within ~0.3, which is the check the chain needed; it is not an exact prediction.

This also explains why mean reward and episode length never showed it, and why the pinned
evals look fine on rows the training barely visited.

## 2. What I chose for v3, and why

**Uniform rows, re-drawn at every reset, no terrain curriculum.** Every env's row and column
are drawn uniformly at each reset (mjlab's `randomize_terrain` event, placed *before*
`reset_base`, which positions the robot at `env_origins`). Rows 4–9 (11–20 cm) are 60% of
training from iteration 1 and the share is fixed. The four sub-terrains are equal weight.

Why this over the other two options you listed:

- **A fixed promote/demote rule** needs a threshold on something the policy controls that
  does not depend on commands or patch size. I replayed one (promote if it timed out *and*
  walked ≥ 2.5 m; demote if it ended early; stalling neither promotes nor demotes). Offline it
  gives a stationary mean row of 6.3–7.2 on v2's rows, so it works as a rule. But it has never
  driven a training run, it adds a moving part whose behaviour on a warm-started policy at
  17 cm I cannot predict, and it can park robots on rows 4–5 if learning there is slow,
  which is exactly the failure you want to rule out. It is implemented
  (`terrain_levels_survival`) and **not registered**; it is the prepared next change.
- **High initial level, no demotion** puts robots on tall rows at the start and then lets the
  distribution drift upward, with no replay of easy rows (risk of forgetting 5–9 cm) and no
  way to see the spread stop being uniform.

Uniform costs efficiency: rows 0–3 (40% of envs) are mostly solved, and rows 8–9 may be
infeasible for this robot (see risks). It buys a guarantee that is visible in the logs, and a
single run answers "can it learn tall steps at all". `terrain_rows_0_1 … terrain_rows_8_9` and
`terrain_row_mean` are logged as `Curriculum/...` scalars (they print in the `.out`; checked
on CPU). Uniform draws give ~0.2 per pair of rows and a mean row of ~4.5; any other reading
means the event is not doing what this says.

## 3. What was built (all committed; nothing trained)

- **`Unitree-Go2-Spec-StairsV3`**, experiment `go2_spec_stairs_v3`: StairsV2 with risers
  5–20 cm, `pyramid_stairs` and `pyramid_stairs_inv` at treads 0.30 m and 0.26 m (four
  sub-terrains, equal weight), the row-spread above. Same stage-0 command range,
  observations (234 with scan), rewards, terminations, runner, `clip_actions=6.0`. The 0.26 m
  tread went in without trouble.
- **Built geometry checked** (CPU, the real generator): risers by row 0…9, all four
  sub-terrains: 5.8–6.1, 7.0–7.2, 8.6–9.3, 10.1–10.6, 11.7–11.9, 12.9–13.5, 14.5–15.0,
  16.1–16.4, 17.6–18.0, 19.3–19.4 cm. Each staircase is **5 steps** at both treads (the
  generator's `border_width=1.0` takes the room), with a 1 m border outside; the row-9
  inverted pyramid spawns in a pit ~1.15 m deep.
- `SPEC=StairsV3` in `a100/train_specialist_slurm.sh` warm-starts from stairs v2's
  `model_9999.pt` by default (`warm_start_ckpt.py`, normalizer reset), failing loudly if
  that file is missing. `INIT_FROM=` overrides it. Not exercised end to end: the warm-start
  path is the one `GapsWarm` uses, but I have not run it for v3.
- **Eval**: `eval_checkpoint.py --step-height H --step-width W` pins riser and tread
  (needs a stairs terrain; raises otherwise), and a `by_terrain` block splits episodes,
  terminations and falls/100 m by sub-terrain, because pyramid walks *down* and inverted
  walks *up* and the pooled number hides which one fails. `a100/eval_stairs_heights_slurm.sh`
  runs a checkpoint at 9/12/15/17 cm × both treads at stage-0 commands (9 min).
  `apply_eval_conditions` now drops `randomize_terrain` so pinned evals stay pinned.

## 4. Stairs v2 on the same eval (the number v3 has to beat)

Stage-0 commands, 1024 envs × 1200 steps, eval seed 0, falls per 100 m / survival % (survival
counts any episode that ran to the time limit, including one that stalled on a step):

| tread | riser | DOWN | UP | pooled falls/100 m |
|---|---|---|---|---|
| 0.30 | 9 cm | 1.9 / 86.8% | 2.4 / 85.7% | 2.14 |
| 0.30 | 12 cm | 6.7 / 59.0% | 5.1 / 72.4% | 5.95 |
| 0.30 | 15 cm | 17.2 / 22.5% | 6.5 / 66.3% | 12.30 |
| 0.30 | 17 cm | 21.5 / 16.3% | 31.9 / 7.7% | 26.64 |
| 0.26 | 9 cm | 2.8 / 81.0% | 2.5 / 84.5% | 2.67 |
| 0.26 | 12 cm | 10.5 / 39.7% | 5.6 / 68.1% | 8.24 |
| 0.26 | 15 cm | 18.2 / 20.4% | 7.2 / 60.7% | 13.04 |
| 0.26 | 17 cm | 20.1 / 17.4% | 28.0 / 9.9% | 23.99 |

**This does not reproduce the laptop's "0% up at 12 cm"**, and it should not be read as
contradicting it: this eval counts falls and survival, not "crossed a 5-step flight", and
random commands wander. Going up at 12–15 cm the policy mostly *survives* (66–72%) with few
falls, consistent with stalling; at 17 cm it falls on 92% of episodes. Going down it falls
from 15 cm. Use the laptop harness for the pass/fail bar; use this for falls/100 m and
direction. All eight cells ran, `.err` empty.

## 5. Risks, and what I do not know

- **Never run.** The v3 config was built and stepped on CPU (60 steps, 400 envs); no GPU
  iteration, no warm-start. Check the first lines of `go2-spec-<job>.out` after submitting.
- **Rows 8–9 (17–20 cm) may be infeasible** with 0.26–0.30 m treads and the 10 N
  non-foot-contact termination (a knee brushing an edge ends the episode); I left both as
  specified. If the tall rows only ever fail, that is 20% of envs of pure noise.
- **Uniform is not the efficient choice**; if run 1 stalls on rows 6–9 the next change is
  `terrain_levels_survival` or a narrower range.
- **A first diagnostic run (12560) was discarded.** It counted each env's initial-reset
  record (zero-length episode) as an episode, which made 20% of its data read as demotions.
  The file is kept as `…_WITH_INITIAL_RESET_RECORDS_DO_NOT_USE.json`; every number above is
  from 12562 with those dropped (2,048 of 10,288 records). The stationary means were 1.3 in the
  bad run and 2.4 in the good one, so the "matches the log" agreement depends on that fix.
- The replay is for one policy (v2 final) on its own 0–10 cm rows. It shows the rule's
  structure, not what it would do to a v3 policy at 15 cm.
- Per-tread, per-direction cells are 1024 envs × 1200 steps, one eval seed.

## 6. To submit (when Rohan says so)

    export HF_TOKEN=$(cat ~/.hf_token)
    SPEC=StairsV3 sbatch --gres=gpu:1 a100/train_specialist_slurm.sh

Warm start from stairs v2, seed 42, 10,000 iterations, 8192 envs, HF backup to private
`go2_spec_stairs_v3/`; roughly 10 hours on one RTX 6000. Then, after it finishes:
`CKPT=<model_9999> TASK=Unitree-Go2-Spec-StairsV3 LABEL=stairs_v3 sbatch a100/eval_stairs_heights_slurm.sh`.
