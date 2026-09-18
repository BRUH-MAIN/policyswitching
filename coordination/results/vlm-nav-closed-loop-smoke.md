# VLM navigation: first closed-loop runs (smoke scale, 2026-09-17)

**Not an evaluation**: 2–4 trials per cell, one seed. These runs check that the whole pipeline
executes and show *how* it succeeds or fails. **Raw**: `unitree_rl_mjlab/logs/vlm_nav/closed_loop_*/`
(result.json, per-agent event logs, VLM transcripts, ego-camera videos; gitignored).

Pipeline under test (`scripts/vlm_nav_run.py`, `src/vlm_nav/executor.py`):
- SARO sub-task state machine: planning → facing → across → to the goal, with a discriminator
  double-check.
- Planning subtasks carry a policy (flat / rough / stairs). A policy-selector question can override
  it when two answers in a row agree.
- The intermediation's edges come from depth geometry (Phase 2). The specialist engages at the near
  edge and releases once the footprint is past the far edge.
- Lockstep: simulation pauses while the VLM answers.

| run | arm | trials | success | policy match* | VLM calls / trial | s / call |
|---|---|---|---|---|---|---|
| rough L2, seed 7 | executor + ground-truth "VLM" | 4 | 4/4 | 98% | 39–87 | — |
| rough L2, seed 11 | **Gemma-4-E4B** | 2 | **2/2** | 63% | 43 | 2.6 |
| rough L2, seed 11 | ground truth nav + oracle policy | 2 | 2/2 | 100% | 0 | — |
| stairs_up L1, seed 11 | **Gemma-4-E4B** | 2 | **0/2** (both stuck) | 9% | 51 | 3.3 |

\* Fraction of steps where the active specialist equals the footprint ground truth. It is a
label-matching score, not an outcome score (the rough specialist walks flat ground safely).

## What happened

- **Executor with perfect perception** (oracle VLM): switches land within ~5 cm of the footprint rule
  (engage at x ≈ 2.65–2.69 vs 2.70; release 5.9–6.4 vs 6.35). An earlier version failed here: the
  camera's 0.86 m ground blind zone froze the near-edge estimate and the robot walked onto rough
  ground on the flat policy (policy match 36%). Fixed with `perception.EdgeTracker`.
- **Gemma on rough ground**: planned `rough ground` with the rough specialist for every sub-task.
  The specialist engaged at x = 2.68 / 2.72 (truth 2.70). Gemma never switched back to flat because
  its plan kept `rough` for "to the goal", hence the 63% match, harmlessly. Selector answers:
  rough 17, flat 38.
- **Gemma on 0.05 m up-stairs**: planned `none` on both trials. Selector answered flat 99/100. The
  flat specialist can't climb (0% in Phase 1), so both trials timed out at the first riser. This is
  the Phase-2 finding (Gemma-4-E4B describes the staircase as "a flat, gridded floor") showing up
  in closed loop.


## Four-arm comparison on the rough course (2026-09-18)

Rough L2, seeds 200/201, goal offset 0, 8 trials per cell (16 per arm), lockstep, depth
where-source, Gemma-4-E4B at 560 image tokens. Pooled with `scripts/vlm_nav_summarize.py`.

| arm | success (95% Wilson) | policy match | VLM calls / trial |
|---|---|---|---|
| `vlm` (VLM navigation + **VLM policy choice**) | 100% (80.6–100) | 79% | 27–37 |
| `gt+oracle` (ground truth both) | 100% (80.6–100) | 100% | 0 |
| `vlm+oracle` (VLM navigation + GT policy) | 94% (71.7–98.9) | 100% | 5–13 |
| `vlm+fixed:rough` (VLM navigation + one specialist) | 94% (71.7–98.9) | 51% | 4–10 |

**The arms are indistinguishable**, as the Phase-1 confirmation predicted for a single-obstacle
course: every arm is at the ceiling, and the confidence intervals overlap completely. 16 trials per
arm cannot separate 94% from 100% in any case.

**The decisions behind the 100% are mixed.** Across these runs the VLM planned `rough ground` 18
times, `none` 10 times, and `stairs` 3 times — on a course with no stairs — and the policy selector
answered rough 190 times vs flat 188. The `vlm` arm still reached the goal every time, because on
rough L2 both the rough (98%) and flat (73%) specialists usually get across. So this result shows
the pipeline executes end to end with a real VLM in the loop; it is **not** evidence that the VLM
chooses well. A course that separates good from mediocre choosing needs the best specialist to
change along the route — blocked on the stairs specialist (`vlm-nav-phase1-confirmation-rough.md`).
