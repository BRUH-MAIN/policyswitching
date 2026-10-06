# Stairs v3 did not learn to climb: it learned to stand still

**Date**: 2026-10-06 · **Machine**: cluster, session `cluster-sess` · **Jobs**: 12563 (training,
9h01m, exit 0:0, 10,000 iterations, `model_9999.pt` on private HF `go2_spec_stairs_v3/`),
12576 (heights eval, 9 min). **Verdict: failed run. Do not use this checkpoint.** The eval's
falls/100 m look excellent and mean nothing here.

## What the eval says (stage-0 commands, pinned risers, 1024 envs × 1200 steps, seed 0)

Falls per 100 m [survival %], DOWN = `pyramid_stairs`, UP = `pyramid_stairs_inv`:

| tread | riser | v2 DOWN | v2 UP | v3 DOWN | v3 UP |
|---|---|---|---|---|---|
| 0.30 | 9 cm | 1.9 [87] | 2.4 [86] | 0.1 [100] | 0.3 [100] |
| 0.30 | 12 cm | 6.7 [59] | 5.1 [72] | 0.4 [99] | 0.0 [100] |
| 0.30 | 15 cm | 17.2 [23] | 6.5 [66] | 0.5 [99] | 0.1 [100] |
| 0.30 | 17 cm | 21.5 [16] | 31.9 [8] | 0.4 [99] | 0.1 [100] |
| 0.26 | 17 cm | 20.1 [17] | 28.0 [10] | 0.0 [100] | 0.1 [100] |

(All eight cells are in `unitree_rl_mjlab/eval_results/stairs_heights/stairs_v3_h*_w*.json`;
the 0.26 m rows for 9/12/15 cm look the same.)

The v3 numbers are a bracing artifact (findings.md bugs #1, #14), not a result: **v3 is
identical at 9 cm and 17 cm, because it barely moves.** Achieved speed is **12% of commanded
at every cell** (v2: 54–61%), mean speed 0.061 m/s against v2's 0.27, **53% of
commanded-to-move steps stalled** (v2: 4%), and the 1024 envs cover about 750 m per direction
against v2's 3,100–3,600 m. At 9 cm, a height v2 handles with 1.9–2.4 falls/100 m while
walking at 56%, v3 has stopped walking.

## What the training log says (`go2-spec-12563.out`)

- **Nothing learned after iteration ~200.** Mean reward 42.2–42.6 and mean episode length
  997.7–999 at iterations 500, 1000, … 9974 (±25-iteration means); illegal_contact per
  iteration 0.02–0.04 from iteration 500 (v2 ended at 0.32, v1 at 0.20).
- **Per-step tracking reward** (`Episode_Reward/track_linear_velocity`, last 500 iterations):
  v3 **0.453**, v2 0.777, original stairs 0.427; at iterations 200–300 v3 was already 0.457.
  `pose` 0.935 (v2 0.839), `foot_gait` 0.326 (v2 0.418).
- **The row spread was exactly as designed all run**: each pair of rows held 0.19–0.21 of envs,
  mean row 4.4–4.5 at every sampled point. So the failure is not the curriculum doing the
  wrong thing; robots really were on 11–20 cm rows 60% of the time.
- Warm start worked (v2 weights, counters reset), 3.1 s/iteration, no errors.

## My reading (inferred, not tested)

Warm-started from a policy that walks, with 60% of envs on rows (11–20 cm) it cannot climb,
the policy found within a few hundred iterations that standing still avoids the
`illegal_contact` termination and the `is_terminated` penalty and still collects the pose and
orientation rewards. A time-out costs nothing, so stalling is a stable optimum, and nothing in
the reward pays enough for the progress that would get it off the first step. v2's earlier
curriculum collapse (rows ~2) was a milder version of the same trade-off.

I have not isolated why it stopped moving even on 5–9 cm rows (20–40% of envs), which v2
walks well. Candidates: shared policy dominated by the tall-row signal; the `stand_still`
behaviour generalising; value-function disruption from the normalizer reset. None tested.

## What I would change for run 2 (proposals; nothing submitted)

1. **Put robots where there is a gradient.** Adaptive rule (`terrain_levels_survival`,
   implemented, not registered, replayed offline only) starting at rows 0–3 so the warm-started
   policy still walks, rising as it passes, with stalling demoted so a stalled robot is sent
   to easier rows rather than left to learn that standing is fine (`survive_or_stall_demote`
   in the diagnostic).
2. **Make stalling cost something**: a progress/forward-velocity term or a stall penalty on
   stairs, so a timed-out robot that did not move is not scored like a walking one. This
   changes the reward, which `objective.md`'s comparison has kept fixed; a decision for Rohan.
3. **Narrower range first** (e.g. 5–14 cm) so tall rows are not 60% of data from step 1.

I would not run uniform rows from a walking policy again. 1 + 3 is the smallest change and
reuses measured code; 2 is probably needed regardless if a stall is a stable optimum.
