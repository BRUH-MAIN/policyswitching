# Project Objective

## In one sentence

Instead of one generalist locomotion policy that tries to handle every surface, train a small set of terrain-specialist RL policies for a quadruped and learn when and how to switch between them while the robot is actively following a person — and make that switching smarter than a reactive terrain classifier by using the person being followed as a preview signal for terrain that's coming up.

## The gap this targets

Most terrain-adaptive quadruped locomotion work — RMA (Rapid Motor Adaptation), ETH's perceptive-locomotion line, "Learning to Walk in Minutes" — trains a single policy with a terrain encoder so it implicitly generalizes across surfaces. That policy is **reactive**: it conditions only on current proprioception/exteroception, so it adapts to a surface once the robot is already on (or about to step onto) it. Explicit multi-specialist switching between discrete terrain policies is less common, and — as far as this project's related-work read goes — essentially nobody exploits a fact specific to the *person-following* setting: the human being tracked is already walking a few steps ahead of the robot and is a free, causal preview of the terrain the robot is about to enter.

## Where the novelty sits

Use the leader's trajectory (from LiDAR/vision tracking) to anticipate terrain transitions and pre-switch or pre-blend the active policy *before* the robot's feet reach the new surface, rather than reacting after the fact. The claim being tested: **the person-following task itself supplies a predictive terrain signal that reactive single-policy approaches don't use** — and that using it measurably improves switching quality (fewer falls, smoother transitions) over both a single generalist policy and a reactive hard-switch baseline.

This is a minor-but-real contribution, not a from-scratch architecture: the specialists, the switching mechanism, and the two baselines are all things one would build anyway for a clean ablation. The anticipatory-preview idea is the one piece that isn't standard, and the project is structured so the ablation is informative — and the result worth reporting — even if the anticipatory gain over reactive switching turns out to be modest.

## What's being compared (the core empirical result)

A three-way comparison on held-out mixed-terrain courses, all following the same scripted/tracked leader:

1. **Generalist baseline** — one policy trained across the full mixed-terrain curriculum (single-policy, implicitly adaptive, no explicit switching).
2. **Reactive hard-switch** — a terrain classifier over current sensing → hard switch between frozen terrain-specialist policies, with hysteresis/debounce to prevent boundary chattering.
3. **Anticipatory soft-switch (the contribution)** — a learned gating network over the same frozen specialists, conditioned on current terrain *plus* a short look-ahead terrain estimate taken along the leader's predicted path, producing a soft blend rather than a hard argmax.

## Metrics

- **Fall rate** — does the robot stay upright across terrain transitions?
- **Velocity-tracking error** — does it still follow the commanded/leader-derived velocity accurately?
- **Transition-boundary smoothness** (joint torque/jerk spikes at policy-switch events) — this is the metric that makes "switching" itself, as opposed to "which policy is active," measurable. It's expected to separate hard-switch from soft-switch even independent of any anticipatory gain.

## Scope decisions (locked in, see `findings.md` for how/why)

- **Terrain taxonomy**: four geometry-based classes — flat, rough, stairs, gaps — rather than material-based classes (grass/gravel/tile) from the original framing, since the simulator's terrain generator is geometry-only.
- **The followed person, in simulation**: a scripted kinematic leader body with a known trajectory, not a simulated perception pipeline. This makes the "terrain along the leader's path" signal ground truth during the sim ablations that carry the empirical weight of the result, and defers real LiDAR/vision person-tracking to an optional hardware demo.
- **Specialists share one observation space** so the switching module can blend them, which constrains how per-terrain reward shaping can later be introduced (reward *terms* can differ; observation *shape* can't).

## Relationship to the SARO/PAS work in this repo

The repo's actively-trained "PAS" policy (Probability Annealing Selection, replicating arXiv:2407.16412) is a **single, privileged-then-distilled generalist policy** — architecturally the closest thing to comparison arm (1) above, not an instance of arm (2) or (3). It is explicitly *not* a runtime switch between frozen policies (see `docs/07-pas-implementation.md`). It's being used as groundwork and a candidate generalist baseline, not as the project's novel contribution.

## What a good result looks like

The three-way table itself is the deliverable: hard-switch should reduce fall rate relative to the generalist on mixed terrain (specialists outperforming a generalist on their own terrain is close to given); the open empirical question is whether anticipatory soft-switching further reduces transition jerk — and ideally fall rate, via pre-emptive blending before a boundary — over reactive hard-switching. A real but modest gain there is still a defensible, publishable result, because the ablation against both baselines is clean regardless of effect size. Sim results are expected to carry the empirical weight; a qualitative hardware demo (a person walking the robot across a few real terrain patches) is a bonus, not a requirement.
