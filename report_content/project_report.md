# Project Report — Terrain-Specialist Policies

**As of**: 2026-09-16 · **Scope of this report**: the training pipeline, the three trained
specialist policies, and their complete cross-terrain evaluation. Terrain-classification
work (the sensor-based auto-switching piece) is intentionally out of scope here — see §4.

---

## 1. Project pipeline

**Goal**: instead of one generalist walking policy for a quadruped (Unitree Go2), train a
small set of **terrain-specialist** policies, each expert at one terrain type, and pick the
right one at runtime. The eventual system follows a person and uses their path as an early
warning of upcoming terrain.

**How a specialist gets made, end to end:**

1. **Simulation** — `mjlab` (MuJoCo-based physics) simulates the Go2 robot on procedurally
   generated terrain. Four terrain classes exist: `flat`, `rough` (bumpy/wavy ground),
   `stairs`, and `gaps` (stepping stones). Terrain is geometry-only in this simulator — no
   material/friction textures (that's handled separately as randomized foot friction, not a
   terrain class).
2. **Training** — each specialist is trained with PPO (via RSL-RL) on its own terrain class
   only, for 10,000 iterations, on the SLURM GPU cluster. The robot's observations include a
   **height scan**: a small forward-facing grid of distance sensors (187 rays, covering
   roughly 1.6m × 1.0m in front of the robot, reaching about 0.8m ahead) — this is its only
   way to sense the ground ahead of it. All specialists share the exact same observation
   space and reward structure, differing only in which terrain they trained on — this is
   deliberate, so they can later be compared and combined fairly.
3. **Checkpointing** — training saves periodic checkpoints; the final one (iteration 9999)
   is what gets evaluated and what these videos show.
4. **Evaluation** — a separate script drives each trained policy through controlled test
   conditions (fixed terrain type, fixed difficulty, fixed commanded walking speed) and
   measures how often it falls, how fast it actually moves versus what it was told to do,
   and how it behaves on terrain it never trained on. This is what produces the numbers in
   §3.
5. **(Not yet built)** — a switching/decision layer that picks which specialist to run based
   on sensed or anticipated terrain. This is the next phase, currently on hold — see §4.

## 2. The three specialists

| Specialist | Trained on | SLURM job | Iterations | Checkpoint |
|---|---|---|---|---|
| **Flat** | flat ground | 11852 | 9999 | `eval_ckpts/go2_spec_flat/model_9999.pt` |
| **Rough** | uneven/wavy ground | 11851 | 9999 | `eval_ckpts/go2_spec_rough/model_9999.pt` |
| **Stairs** | staircases | 11849 | 9999 | `eval_ckpts/go2_spec_stairs/model_9999.pt` |

A fourth specialist (gaps/stepping-stones) does not exist — two training attempts plateaued
and were never completed. It's excluded from everything below.

## 3. Complete cross-terrain evaluation

Each specialist was run on every terrain type (not just its own), at a fixed, controlled
difficulty, with a fixed commanded walking speed — so the numbers are directly comparable to
each other. Full methodology and raw data: `coordination/results/gate1-cross-terrain-matrix-analysis.md`.

**The metric is falls per 100 meters actually travelled** (not raw fall-percentage — an
earlier version of this evaluation used a percentage that turned out to silently favor
policies that die fast over policies that die slow, an infra bug that's now fixed; falls-per-distance
doesn't have that problem, and also correctly penalizes a policy that "avoids falling" by
barely moving at all).

| Specialist walking on → | Flat | Rough | Stairs |
|---|---|---|---|
| **Flat specialist** | 0.00 | 14.41 | 10.42 |
| **Rough specialist** | 0.00 | **5.36** | 11.14 |
| **Stairs specialist** | 0.00 | 13.02 | **7.60** |

*(lower is better; bold = best policy for that terrain)*

**What this shows, plainly:**

- **Every specialist walks perfectly on flat ground** — 0 falls each. Not surprising, and it
  confirms none of the three is simply broken.
- **On rough terrain, the rough specialist is clearly the best choice** — 2.4× fewer falls
  than the next-best option (the stairs specialist; the flat specialist is worse still at
  14.41). This is a solid, well-supported result.
- **On stairs, the stairs specialist is the best choice** — 1.4× fewer falls than the
  runner-up. This result is real but weaker evidence: it comes from a single training run
  with no repeat, and the margin is closer to the noise floor than the rough result. Treat it
  as "very likely true, not yet fully confirmed."
- **No specialist is best at everything.** This is actually the important finding, not a
  weakness: it's the concrete evidence that a system able to pick the right specialist for
  the terrain in front of it would genuinely outperform using any single policy everywhere.
  That's the entire premise of the project, and it's now measured, not assumed.
- **Locomotion was independently sanity-checked**, not just the fall counts: each specialist
  actually walks at 24–46% of its commanded speed while doing this (versus a separate,
  clearly broken policy tested earlier that only moved at 8–10% and looked artificially safe
  purely because it was barely moving). These three specialists are genuinely walking, not
  gaming the metric.

Both `gaps` and `mixed`-terrain columns were also measured but are excluded from this table:
every specialist fails almost completely on gap terrain, which is expected and uninformative
(none of them ever trained on it), and `mixed` terrain needs a separate normalization to read
correctly (see the full write-up) — neither changes the ranking above.

## 4. What's intentionally not in this report

**Terrain classification** (can the robot automatically detect which terrain it's standing
on, using only its onboard height-scan sensor) was investigated in depth and the results
were not good enough to build on: the sensor reliably detects only gap terrain; flat, rough,
and stairs are frequently confused with each other. Full details are in
`coordination/results/gate2a-height-scan-discriminability-analysis.md` and
`coordination/results/rough-stairs-switching-decision.md` for anyone who wants them, but per
direction, this line of work is being **deprioritized rather than fixed**.

**The planned next direction is a camera/vision-based decision system** (using a real image
of the terrain, interpreted by a vision-language model, instead of the narrow height-scan)
rather than continuing to patch the current sensor-based classifier. This has not been
started — it is queued and will begin on explicit instruction.

## 5. Videos

One video per specialist, on each of the three terrain classes (9 total) — showing directly
what the numbers in §3 mean: each specialist walking normally on its own terrain, and how it
behaves when placed on ground it never trained for. Files in `videos/`, 6 seconds each, fixed
forward-walking command (not the robot's usual randomized command, so every clip actually
shows it moving rather than risking recording it standing still — a known gotcha with this
simulator's default video tooling, documented in `findings.md` bug #3).

| | On flat | On rough | On stairs |
|---|---|---|---|
| **Flat specialist** | `videos/flat_specialist_on_flat.mp4` | `videos/flat_specialist_on_rough.mp4` | `videos/flat_specialist_on_stairs.mp4` |
| **Rough specialist** | `videos/rough_specialist_on_flat.mp4` | `videos/rough_specialist_on_rough.mp4` | `videos/rough_specialist_on_stairs.mp4` |
| **Stairs specialist** | `videos/stairs_specialist_on_flat.mp4` | `videos/stairs_specialist_on_rough.mp4` | `videos/stairs_specialist_on_stairs.mp4` |

*(Diagonal = each specialist on its home terrain; off-diagonal = the degradation numbers in §3, visualized.)*
