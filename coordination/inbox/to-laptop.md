# Inbox: cluster -> laptop

"Please evaluate" notices go here whenever a checkpoint worth testing is
ready -- `coordination/scripts/cluster_update_status.sh` appends these
automatically (right below "## Open"); you can also add one by hand. Laptop
session: move an entry under "## Done" once evaluated, linking the resulting
`coordination/results/<run_id>-analysis.md` -- don't delete it.

<!-- Example entry (this is the shape cluster_update_status.sh generates):

## 2026-09-11 -- go2_spec_gaps step 9999 ready
Source: local (cluster disk only, not on HF -- see findings.md bug #4).
Path: unitree_rl_mjlab/logs/rsl_rl/go2_spec_gaps/2026-09-11_.../model_9999.pt
(sha256 in coordination/status/cluster.json). SLURM job 11914.

-->

## Open

## 2026-09-12 -- eval_checkpoint.py defaults changed; objective.md revised (please review)

Not a "please evaluate" note -- a heads-up that affects every future laptop eval, plus a
design change that is arguably yours to sign off on. All of it is on `main` now
(commits `11c4c4d`, `255ea51`), made on the cluster at the user's direction *before* this
coordination kit landed, so it hasn't been through the role split.

**1. `eval_checkpoint.py` now runs under pinned conditions by default.** `laptop_pull_and_eval.sh`
passes no terrain flags, so it picks these up automatically:
- terrain curriculum **off**; `--terrain` (native/flat/rough/stairs/gaps/mixed) and
  `--difficulty` (omit = spread envs uniformly over difficulty rows) set it explicitly;
- command range pinned to the final training stage and printed;
- velocity error reported **per step**;
- new: whole-rollout smoothness (action rate, actuator-force rate), `--ablate-height-scan`,
  `--seed`, `--json-out`, `--label`.
`--keep-curricula` reproduces the old behaviour, and is only valid for checking a checkpoint
against its own training log.

**Consequence: new laptop numbers will NOT match the ones in `laptop.json`.** The existing
`go2_pas_stage2` verdict (oracle 75.5% / estimator-only 42.0%) was measured with the terrain
curriculum live, which is findings.md bug **#10**: `terrain_levels_vel` promotes an env to
harder terrain when it walks far enough and demotes it otherwise, *during the rollout*, so a
better policy is pushed onto harder ground until it also fails. Those numbers are fine as
"this checkpoint under its training conditions" and invalid as cross-policy comparisons. The
same verdict quotes `err_vel_xy 2.09`; bug **#11** is that `error_vel_xy` accumulates over an
episode, so it isn't comparable across policies with different survival lengths. Use the
per-step error instead.

**2. `objective.md` is revised** -- 2x2 of {hard, soft} x {reactive, anticipatory} (adds a
reactive *soft*-switch arm, so blending and anticipation stop being confounded); arm 1 is now
a sensing-matched stock-PPO generalist (`Unitree-Go2-Generalist`, job 11918) rather than PAS;
transition jerk defined in geometric windows around terrain-class boundaries instead of keyed
to switch events; anticipatory gain swept over leader distance, since the onboard scan already
previews ~0.8 m; seed plan. Rationale and the two review claims that turned out wrong are in
findings.md "Review 2026-09-12" and "Checked, not bugs". **If you disagree with any of it,
say so in `to-cluster.md`** -- it's a design call, and the cluster side shouldn't be making
those alone.

**3. Premise gates before the switching module gets built** (findings.md, `objective.md`):
job 11919 runs `a100/eval_matrix.py` -- every trained policy x terrain class x difficulty
under pinned conditions, plus a height-scan ablation. It answers (a) do specialists actually
degrade off their home terrain, and (b) do these policies use `height_scan` at all. On (b):
all three specialists' height-scan normalizer std is ~0.012, and the injected observation
noise alone accounts for ~0.0115 -- the per-ray terrain signal is near the noise floor. If the
ablation barely moves the numbers, the reactive classifier and the look-ahead feature are both
built on sand. Output lands in `unitree_rl_mjlab/eval_results/matrix/summary.md` (not
`coordination/results/`), and would be worth your read when it exists.

**4. Nothing is running.** All three GPU nodes have been `drained` since 2026-09-10 ("Kill
task failed"); 11914 (`go2_spec_gaps` attempt 3) never started, and 11918/11919 are queued
behind the same thing. `cluster.json`'s `go2_spec_gaps` placeholder is therefore unchanged.

**5. Two small kit issues, your call since you own these files.**
- `cluster_update_status.sh` never writes a `task` field, but `laptop_pull_and_eval.sh` reads
  `run.task` and hard-errors on `MISSING`. For a run_id the script creates fresh, the laptop
  side would fail. I've pre-seeded `go2_generalist` and `go2_spec_gapswarm` entries in
  `cluster.json` with their task names so this doesn't bite, but the script itself still has
  the gap.
- `docs/CLAUDE.cluster.md` is stale in four places: `clip_actions=6.0` is committed (`6faa66b`),
  the `SPEC` list is missing `GapsWarm` and `Generalist`, specialists now *can* back up to HF
  (opt-in, only when `HF_TOKEN` is set at submission), and it still refers to the three-way
  comparison. Also, the login node it says is unrecorded is `asaicomputemaster` -- which is
  simultaneously the RTX 6000 Ada compute node.

## Done
