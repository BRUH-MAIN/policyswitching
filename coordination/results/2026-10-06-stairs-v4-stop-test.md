# Stairs v4a/v4b stop test (iteration ~530): the normalizer was not the whole story

**Date**: 2026-10-06 · **Machine**: cluster, `cluster-sess` · **Jobs**: 12581 (v4a, normalizer
reset, cancelled at iteration 2127), 12586 (v4a_kn) and 12587 (v4b_kn) (normalizer kept,
running, iteration ~540). **Status**: both `_kn` runs fail the agreed stop test; I have not
cancelled them; that is Rohan's decision.

## Result

| run | speed / commanded | stalled | mean row | tracking reward | illegal_contact / iter | reward |
|---|---|---|---|---|---|---|
| 12581 v4a, reset, it. 500 | 29% | 10% | 0.00 | 0.55 | 0.09 | 46.3 |
| 12586 v4a_kn, kept, it. 520 | **28%** | 10% | **0.01** | 0.55 | 0.05 | 46.5 |
| 12587 v4b_kn, kept, it. 520 | **30%** | 9% | **0.04** | 0.56 | 0.03 (trunk only) | 46.8 |
| stairs v2 final (own log) | n/a (not logged) | | | 0.78 | 0.32 | 47.4 |

(±10-iteration means; speed = `actual_speed / cmd_speed`; stop test: speed < 30% of commanded,
or mean row not rising by 1,500: it is already failing the first at 500 and the second would
fail.)

## What the trajectory shows (12586; 12587 is the same shape)

| iteration | 10 | 25 | 50 | 100 | 150 | 200 | 300 | 520 |
|---|---|---|---|---|---|---|---|---|
| speed / commanded | 0.45 | 0.43 | 0.40 | 0.30 | 0.26 | 0.27 | 0.28 | 0.28 |
| mean row | 1.34 | 1.20 | 1.07 | 0.65 | 0.30 | 0.12 | 0.02 | 0.01 |
| illegal_contact / iter | 1.60 | 0.62 | 0.34 | 0.10 | 0.04 | 0.05 | 0.11 | 0.05 |

Keeping the normalizer helped at the start (speed 43-45% at iteration 25-36 against 28% for
the reset run) but the run then slows to the same 28% as terminations fall. Three facts:

- **Speed falls as terminations fall.** The policy gets safer by getting slower. Mean reward
  (46.5) already equals v2's final 47.4, so the reward cannot tell a slow, safe policy from a
  walking one, and tracking reward settles at 0.55 against v2's 0.78.
- **It is not the knee-contact termination alone.** 4b only terminates on trunk/hip contact
  (illegal_contact ~0.03) and is no faster (30%).
- **The progress gate then locks it in**: at 0.14 m/s mean speed a robot rarely gets 3.2 m
  from the spawn in 20 s, so nothing is promoted and everything stays on row 0 (5-6.5 cm
  risers). That is a consequence, not necessarily the cause: the speed was already at 30%
  at iteration 100 with the mean row at 0.65.

## What this does and does not say

- It does **not** say the progress rule is wrong: the problem is the policy's speed, and the
  rule only reflects it. v3 (uniform rows) went to 12% by the same route, from a harder start.
- The normalizer reset was a real cost (earlier speed) but **not the cause of the plateau**:
  I, and the laptop, had treated it as the cause. The iteration-0 line is identical either way.
- Not run: a pinned eval of a `_kn` checkpoint against v2. Both GPUs on `asaicomputemaster`
  are held by 12586/12587 and the A100s are full, so I cannot say yet whether the policy is slow
  in eval conditions too (same command range) or only under training perturbations
  (pushes, observation noise). `eval_stairs_heights_slurm.sh` takes 9 min once a GPU is free.

## Options (Rohan)

1. **Cancel both and run the reserved third change**: a progress reward or stall penalty, so
   standing/slow is no longer as cheap as walking (changes the reward; the objective.md
   comparison is finished). Also consider a lower termination cost relative to tracking.
2. **Cancel one, keep one** (e.g. keep 4b_kn, the variant that matters for the robot) and use
   the freed GPU for the eval above and a replay of the progress rule on the slow policy.
3. **Let both run** to 1,500: very likely the same verdict, ~1.5 h of two GPUs.

My recommendation: 2 or 1, with the eval first, since it separates "the policy is slow" from
"the training metric is".
