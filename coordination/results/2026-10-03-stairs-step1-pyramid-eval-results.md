# Step 1 results — stairs specialist on pinned pyramid stairs (job 12033)

**Job ran**: 2026-09-21 10:26–10:30 (4m14s, `asaicomputemaster`) · **Written up**: 2026-10-03
(cluster session, during a status check requested by the laptop session). The job completed
cleanly 12 days ago but the results sat uncommitted on cluster local disk the whole time —
nobody had written this up or run `git add` on the output JSONs. Flagging that gap rather
than quietly closing it.

## What ran

Per `a100/eval_stairs_pyramid_slurm.sh`'s header: the Stairs specialist checkpoint
(`unitree_rl_mjlab/logs/rsl_rl/go2_spec_stairs/2026-09-05_22-37-43/model_9999.pt`) on
`--terrain stairs` (`pyramid_stairs` + `pyramid_stairs_inv`, matching its training mix) at
pinned difficulty 0.5 / 0.7 / 0.9, no `--keep-curricula`, final-stage command range, 1024
envs × 1200 steps. Exit code 0 on all three difficulties; `go2-stairs-pyr-eval-12033.err` is
empty. Outputs:
`unitree_rl_mjlab/eval_results/stairs_pyramid/go2_spec_stairs_pyramid_d{0.5,0.7,0.9}.json`.

## Results

| difficulty | episodes | survival % | falls/100m | mean ep len | time_out | fell_over | illegal_contact | achieved speed (% of cmd) |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1138 | 65.3 | 6.08 | 833.6 | 743 | 4 | 394 | 26.5% |
| 0.7 | 1300 | 43.5 | 12.58 | 700.3 | 565 | 5 | 732 | 23.7% |
| 0.9 | 1422 | 31.8 | 16.95 | 629.2 | 452 | 29 | 952 | 23.7% |

(terms can overlap per episode, see `eval_checkpoint.py`'s per-cause breakdown added in `9b60cbe` — this is the first GPU run of that code path and it did not crash.)

Monotonic degradation with difficulty: survival roughly halves per +0.2 step, and
`illegal_contact` dominates over `fell_over` at every level by 2 orders of magnitude — the
failure mode is getting a foot caught/clipping a step edge, not toppling over. Achieved speed
sits flat at ~24–27% of commanded regardless of difficulty, i.e. the policy isn't moving
faster on the easier rows, it's just surviving longer at the same crawl before it clips
something.

## What this does and doesn't answer

The script's own framing (its header comment): "fails on pyramids too -> weak on stairs
generally; fine on pyramids but bad on the straight course -> overfit to pyramid geometry."
This run only has the pyramid side of that comparison — pyramid stairs is the specialist's
*training* terrain (`TERRAIN_CLASSES["stairs"]`), so there is no second (non-pyramid)
geometry in this job to compare against directly.

`PROGRESS_REPORT.md` (2026-09-16/22, `vlm-pipeline` branch) reports 75–78% success "with a
*perfect* chooser, at the gentlest stairs tested" on whatever course that branch's
closed-loop pipeline uses. That number is *higher* than this job's d=0.5 survival (65.3%),
which on its face argues for "fails on pyramids too" (home terrain isn't reliable either) —
but **I have not verified the two numbers are measured the same way** (different branch,
different harness, "success" on a traversal vs. "survival" over a fixed 1200-step rollout,
unknown riser height for "gentlest" vs. d=0.5's risers, different episode/command
definitions). Treating this as a confirmed comparison without checking that would be exactly
the kind of silently-wrong-looking-plausible result `findings.md` keeps warning about, so I'm
flagging the discrepancy and leaving the verdict — and the step-2 (stairs retrain) go/no-go —
to whoever picks this up with time to check the harnesses match.
