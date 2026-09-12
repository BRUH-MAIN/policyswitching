# Project Objective

*Revised 2026-09-12 after a design review (see `findings.md`, "Review 2026-09-12"): the comparison is now a 2×2 over {hard, soft} × {reactive, anticipatory} plus a sensing-matched generalist, the novelty claim is stated as preview-horizon extension, the smoothness metric is defined geometrically, and two premise gates must pass before the switching module is built. Gate 1 revised again same day: it's a column-wise condition (best policy varies by terrain), not row-wise (each specialist worse off-terrain) — the two are neither equivalent nor is the row-wise one sufficient for the switching claim. Gate 2 confirmed and sharpened empirically (rough/stairs collide by pattern, not noise level).*

## In one sentence

Instead of one generalist locomotion policy that tries to handle every surface, train a small set of terrain-specialist RL policies for a quadruped and learn when and how to switch between them while the robot is actively following a person — and make that switching smarter than a reactive terrain classifier by using the person being followed as a preview signal for terrain that's coming up.

## The gap this targets

Most terrain-adaptive quadruped locomotion work — RMA (Rapid Motor Adaptation), ETH's perceptive-locomotion line, "Learning to Walk in Minutes" — trains a single policy with a terrain encoder so it implicitly generalizes across surfaces. Perceptive policies in that line are not blind to what's ahead: they read an onboard height scan, which *is* a short preview. In this repo's config (`terrain_scan`, a 1.6 m × 1.0 m grid centred on the base) that preview reaches about **0.8 m ahead**. What they can't see is terrain beyond their own sensor horizon.

Explicit multi-specialist switching between discrete terrain policies is less common, and — as far as this project's related-work read goes — essentially nobody exploits a fact specific to the *person-following* setting: the human being tracked is already walking a few steps ahead of the robot, well past the onboard scan, and is a free, causal preview of terrain the robot will reach after the scan horizon.

## Where the novelty sits

Use the leader's trajectory (from LiDAR/vision tracking) to anticipate terrain transitions and pre-switch or pre-blend the active policy *before* the new surface enters the robot's own sensing range, rather than reacting once it does. The claim being tested: **the person-following task supplies terrain preview beyond the onboard exteroception horizon, and using it measurably improves switching quality** (fewer falls, smoother transitions) over reactive switching that sees only what the robot's own scan sees.

The contribution is *horizon extension*, not preview vs. no preview — so the experiment has to show the gain as a function of how far ahead the preview reaches (see "Preview-horizon sweep" below). A single operating point can't distinguish "the idea doesn't help" from "the leader was too close to add anything beyond the scan."

This is a minor-but-real contribution, not a from-scratch architecture: the specialists, the switching mechanism, and the baselines are all things one would build anyway for a clean ablation. The anticipatory-preview idea is the one piece that isn't standard, and the project is structured so the ablation is informative — and the result worth reporting — even if the anticipatory gain turns out to be modest.

## Premise gates — must pass before the switching module is built

1. **No single policy is best on every terrain.** Switching between frozen specialists only has something to recover if the best choice *changes* by terrain — otherwise the switch always picks the same specialist and is a strictly worse way to deploy that one policy. Tested by the cross-terrain eval matrix (`a100/eval_matrix.py`): every policy × every terrain class × fixed difficulty.
   - **This is a column-wise condition, not a row-wise one, and the distinction matters.** Row-wise ("each specialist does worse off its own terrain than on it") is neither sufficient nor necessary: a specialist can degrade off-terrain in every row while still being the single best policy in every column (switching would then recover nothing — some terrain being harder than others is true by construction and says nothing about specialization), and a specialist can win its own column while barely degrading elsewhere (switching to it still pays off). The gate passes iff the per-column argmin varies across at least two policies, by more than the run-to-run spread from simulator nondeterminism (~1 point, per the gate-2a calibration below) — not iff the diagonal looks locally best in its own row.
   - **Read locomotion quality before fall rate.** None of the specialists have been through `eval_checkpoint.py`'s stalled%/achieved-speed metrics yet — they were gated on training-log survival and `error_vel_xy` alone, the exact combination that hid PAS's near-stationary behaviour (findings.md bug #14). A cell in that range makes its own diagonal entry meaningless regardless of fall rate.
   - **Report row-wise too, labelled as such** — it's informative context, just not what the gate turns on.
   - **A missing specialist leaves its column unanswerable, not zero.** State that explicitly rather than reading a gap as a null result.
   - **Every specialist is single-seed.** A column margin near the noise floor is not a result in either direction until re-run with seeds.
   - **A clean pass here does not establish the paper's actual claim.** It shows switching beats *any single fixed specialist* — not that it beats the sensing-matched generalist (arm 1 vs. 2a, which needs `Unitree-Go2-Generalist`). Keep the two claims separate in any write-up; the second is the one this project rests on.
2. **The height scan actually distinguishes the terrain classes.** The reactive classifier and the gating network's current-terrain input both assume `height_scan` carries usable terrain information. The specialists' observation-normalizer statistics suggest per-ray terrain variation is *smaller than the injected observation noise* (±0.1 m uniform). Tested two ways: directly (see the confirmed result below) and via the same matrix's height-scan ablation (scan replaced by an uninformative constant, `eval_matrix.py --ablate-height-scan`) on a *policy that saw scan variation in training*.
   - **The ablation is only meaningful on `Unitree-Go2-Generalist` (job 11918), not on a specialist.** `--ablate-height-scan` substitutes a policy's own normalizer mean for the scan — a specialist trained on one terrain class saw a near-constant scan throughout training (normalizer std ~0.012, of which ~0.0115 is injected noise, leaving ~0.003 of real signal — consistent with gate 2a's own within-sample ray-std of 0.0059 against the same 0.0115 noise floor). Ablating an already-near-constant input with a constant is a null result *by construction*: it says "this specialist's scan carried little it could use," which the normalizer stats already say without running anything, not "a policy cannot use a discriminative scan." **Specialist ablation cells should be reported as run-but-not-evidence, and this half of the gate cannot be closed until job 11918 exists** — independent of whether the drained nodes are fixed, since the generalist checkpoint is the thing that's actually informative here.
   - **Confirmed and sharpened, 2026-09-12** (`unitree_rl_mjlab/scripts/height_scan_classifier.py`, `coordination/results/gate2a-height-scan-discriminability-analysis.md`) — this half is unaffected by the ablation caveat above, since it measures discriminability directly from labelled observations with no policy in the loop: the scan separates gaps cleanly (~0.97 held-out accuracy) and flat workably (~0.60-1.00 depending on noise), but rough and stairs collide — their roughness magnitudes are nearly identical, so they differ by spatial *pattern*, not level, and no model tried resolves the pair even at the zero-noise limit (measured ceiling ~0.51 each). Reducing the injected noise recovers flat/stairs separation but is *provably insufficient* for rough-vs-stairs specifically, since that ceiling is already measured, not extrapolated. Three levers, none a config tweak since two touch settings every already-trained checkpoint depends on: reduce the scan noise, raise the terrain difficulty range, or narrow the taxonomy (merging rough+stairs into one "uneven" class keeps a 3-way switch the sensing already supports at ~0.6-0.97/class). Undecided as of this writing.

## What's being compared (the core empirical result)

A comparison on held-out mixed-terrain courses, all following the same scripted leader. The switching arms form a 2×2, so blending and anticipation are measured separately rather than confounded:

| | reactive (current terrain only) | anticipatory (+ look-ahead along leader's path) |
|---|---|---|
| **hard switch** (argmax + hysteresis) | **2a** — terrain classifier → hard switch between frozen specialists | 3b *(optional, cheap)* — argmax of arm 3's gate |
| **soft blend** (learned gating) | **2b** — the arm-3 gating network with look-ahead inputs masked | **3 — the contribution** |

plus

1. **Generalist baseline** — `Unitree-Go2-Generalist`: one stock-PPO policy over the union of the four specialist terrain classes, with the specialists' exact observation space (raw `height_scan` included), rewards, runner config and iteration budget. It differs from a specialist only in terrain breadth.

How to read it: **1 vs 2a** is the effect of specialization; **2a vs 2b** is the effect of blending; **2b vs 3** is the effect of anticipation — the actual claim. Without 2b, any 2a-vs-3 difference is attributable to either blending or anticipation, and the smoothness metric in particular is expected to separate hard from soft regardless of anticipation.

### Preview-horizon sweep

Run arms 2b and 3 across several leader lead distances — from inside the onboard scan horizon (≈0.5 m, a built-in control where anticipation should add roughly nothing) out to a few metres. The deliverable figure is anticipatory gain vs. preview horizon. Scan geometry is held fixed and identical across all arms.

## Metrics

All arms are evaluated under identical, pinned conditions: terrain curriculum off, terrain difficulty fixed (swept as a factor), commanded/leader-derived velocity range stated explicitly. A curriculum left on during eval adapts terrain difficulty to the policy under test, which invalidates cross-policy comparison (see `findings.md`).

- **Fall rate** — does the robot stay upright across terrain transitions?
- **Velocity-tracking error, per step** — does it still follow the commanded/leader-derived velocity? Reported per step. The training log's `error_vel_xy` accumulates over each episode, so it isn't comparable across policies that survive for different lengths.
- **Transition-boundary smoothness** — action rate and actuator-force rate (mean and 99th percentile) inside a **geometric** window: ±0.5 m of base travel around each crossing between terrain patches of different class, located from the known terrain layout. Reported normalized by the same policy's whole-rollout values, so a generally twitchy policy doesn't look worse only at boundaries. It is **never keyed to the policy's own switch decisions**: a soft blend has no discrete switch events, so an event-keyed metric would favour arm 3 by construction.

## Statistical plan

Specialists are frozen assets and train once (single seed). The variance that matters is on the comparison: the gating network (arms 2b, 3) trains with **≥3 seeds**, every arm is evaluated over **≥3 eval seeds / course layouts**, and results are reported as mean ± spread. A "real but modest" anticipatory gain — the expected outcome — is unreportable without this. `SEED=<n>` in `a100/train_specialist_slurm.sh` exists if a specialist or the generalist needs extra seeds.

## Scope decisions (locked in, see `findings.md` for how/why)

- **Terrain taxonomy**: four geometry-based classes — flat, rough, stairs, gaps — rather than material-based classes (grass/gravel/tile) from the original framing, since the simulator's terrain generator is geometry-only. Defined once in `TERRAIN_CLASSES` (`env_cfgs.py`), shared by training and eval.
- **The followed person, in simulation**: a scripted kinematic leader body with a known trajectory, not a simulated perception pipeline. This makes the "terrain along the leader's path" signal ground truth during the sim ablations that carry the empirical weight of the result, and defers real LiDAR/vision person-tracking to an optional hardware demo.
- **Specialists share one observation space** so the switching module can blend them, which constrains how per-terrain reward shaping can later be introduced (reward *terms* can differ; observation *shape* can't).
- **Specialists are specialists**: each trains on its own class. Where a class can't be learned from a cold start (gaps), the fix is initialization (warm start from another specialist), not diluting the training terrain.

## Relationship to the SARO/PAS work in this repo

The repo's "PAS" policy (Probability Annealing Selection, replicating arXiv:2407.16412) is a **single, privileged-then-distilled generalist policy** — explicitly *not* a runtime switch between frozen policies (see `docs/07-pas-implementation.md`). It is reported as a separate replication result and a reference point, **not** as arm 1:

- its deployable estimator-only mode reads **no** height scan, so comparing it to specialists that do measures exteroception vs. none, not generalist vs. switching;
- its oracle mode is roughly sensing-matched (encoded height scan) but adds privileged base velocity and foot friction;
- it adds reward terms (`energy`, `joint_vel_l2`) no specialist has, and got ~8× a specialist's training compute.

## What a good result looks like

The comparison table plus the preview-horizon figure are the deliverable. Hard-switching between specialists should reduce fall rate relative to the matched generalist on mixed terrain (**1 vs 2a**). Blending should reduce boundary jerk relative to hard switching (**2a vs 2b**) — expected, and not the claim. The open empirical question is whether anticipation helps *beyond* blending (**2b vs 3**), and whether that gain grows as the preview reaches further past the onboard scan. A real but modest, horizon-dependent gain is still a defensible, publishable result, because every comparison isolates one factor. Sim results are expected to carry the empirical weight; a qualitative hardware demo (a person walking the robot across a few real terrain patches) is a bonus, not a requirement.
