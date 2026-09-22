# Findings

Running log of experiments, results, and infrastructure gotchas for this project. Written so a future session (human or Claude) can pick up work without re-deriving anything here from raw SLURM logs. Update this file whenever a training run finishes, a bug is found/fixed, or a result changes a prior conclusion — append rather than rewrite, and correct entries in place with a dated note rather than deleting them if a later result contradicts an earlier one.

For the *forward-looking* plan (what to build next), see the roadmap at `~/.claude/plans/the-idea-in-one-iridescent-wilkes.md` (on the machine this was written on — recover it via the Artifact/plan tooling `list` action if unavailable, or reconstruct from this file). This file is the *backward-looking* record: what was tried, what happened, why.

## Project state, one paragraph

Long-term goal: terrain-specialist policies + an anticipatory (person-trajectory-preview) switching module, compared against a sensing-matched generalist and reactive switching baselines (see `objective.md` for the 2×2 design). As of 2026-09-12: three of four terrain specialists (Flat/Stairs/Rough) are fully trained; Gaps failed twice under its original design, a blended-terrain retrain (job 11914) is queued, and a warm-started 100%-gaps redesign (`Unitree-Go2-Spec-GapsWarm`) is built but not submitted. PAS (SARO replication) is fully trained and is now a reference result, not arm 1. The matched generalist (job 11918) and the cross-terrain eval matrix that gates Phase 4 (job 11919) are queued. **All three GPU nodes have been drained since 2026-09-10**, so nothing queued is running. No switching module, leader-in-sim, or person-following code exists yet.

## Review 2026-09-12: design corrections

A review of the objective, roadmap and eval code found problems in the experimental design and the measurement layer. What changed, and where it's recorded:

- **Confounded ablation → 2×2.** Arm 2 (reactive + hard) vs. arm 3 (anticipatory + soft) changed two variables at once. Added arm 2b, a reactive soft-switch (arm 3's gating net with look-ahead masked). `objective.md`.
- **Premise untested → premise gates.** No specialist had ever been evaluated off its own terrain; if specialists transfer fine, switching has nothing to recover. Built `a100/eval_matrix.py` (policy × terrain × difficulty, plus a height-scan ablation) — job 11919. `objective.md`, "Premise gates".
- **Novelty claim → horizon extension.** `terrain_scan` already previews ~0.8 m ahead, so the claim is preview *beyond the onboard horizon*, tested by sweeping leader lead distance. `objective.md`.
- **Generalist not sensing-matched → `Unitree-Go2-Generalist`.** PAS estimator-only reads no height scan, adds reward terms, and got ~8× the compute. New stock-PPO generalist on the union of specialist terrains, otherwise identical to a specialist — job 11918. See also the corrected note in the PAS section.
- **Event-keyed jerk metric → geometric window.** A soft blend has no switch events, so the old definition favoured arm 3 by construction. Now: ±0.5 m around terrain-class boundary crossings, normalized by whole-rollout values. `eval_checkpoint.py` reports the whole-rollout smoothness baseline already. `objective.md`, "Metrics".
- **Two eval bugs** — #10 (terrain curriculum live during eval) and #11 (`error_vel_xy` compared across policies). See "Bugs found".
- **Gaps fix diluted the specialist** — the blended retrain trains 80% on flat/rough. Redesign: 100% stepping_stones, warm-started from the Rough specialist. See "Terrain specialists".
- **No seeds anywhere** — plan: ≥3 seeds on the gating network and ≥3 eval seeds per arm; specialists single-seed. `SEED=` added to `train_specialist_slurm.sh`. `objective.md`, "Statistical plan".
- **Checkpoints with no off-cluster copy** — added `HfSyncVelocityOnPolicyRunner` (pushes each checkpoint when `HF_TOKEN` is set). The three finished specialists are still local-only.

Two review claims turned out wrong on checking the code, and are recorded under "Checked, not bugs" so they aren't re-investigated.

## Infra facts

- **Canonical repo**: `/dist_home/d_palmani/c-08/policyswitching` (single clone as of 2026-09-05; a second, divergent clone at `/dist_home/d_palmani/policyswitching` was merged in and deleted — see "Bugs found" below).
- **venv**: `/dist_home/d_palmani/.venvs/policyswitching-pas` (isolated from the shared conda base env deliberately — see `docs/01-installation.md`).
- **Cluster**: SLURM, partition `workq`, nodes `asaicomputemaster` (2× RTX 6000 Ada) + `asaicomputenode02`/`03` (2× A100 each). Scheduling is **pure age-based FIFO** — `sprio`'s FAIRSHARE component is 0 for every user, so priority is just accumulated wait time. Practical consequence: **submit early and request short walltimes** (a 3-day request only starts when a 3-day window opens and can sit pending far longer than the extra hours are worth; a 1-day request with cheap resubmission schedules much faster). The cluster is shared with other users/projects (e.g. a `codeswitching` project noted in `hf_sync.py`'s comments) — expect contention, typically ~5-10 other jobs queued.
- **2026-09-12: all three nodes `drained`** (`sinfo -R`: "Kill task failed", set 2026-09-10 by root/slurmuser). Every queued job shows `ReqNodeNotAvail` until an admin resumes the nodes — not something a resubmit fixes. mujoco_warp does run on CPU (verified with a tiny eval rollout), which is enough for smoke tests but not real evals.
- **HF model repo**: `RohanRamesh/go2-pas-saro` (anonymously readable, i.e. public) — holds PAS's `stage1/` and `stage2/` checkpoints. Specialist checkpoints trained so far are **not** on HF (bug #4) and live only on local disk under `unitree_rl_mjlab/logs/rsl_rl/go2_spec_<name>/`. From 2026-09-12, specialist/generalist jobs submitted with `HF_TOKEN` set push every checkpoint to `<experiment>/` in that repo; jobs 11918/11919 were submitted without it.
- **Reference paper**: SARO, arXiv:2407.16412 (PDF at repo root). PAS = "Probability Annealing Selection," the paper's low-level locomotion technique this project replicates (not the paper's VLM high-level planner).
- ~~**Uncommitted, load-bearing change**: `clip_actions=6.0` ... still unstaged.~~ *Resolved 2026-09-11: committed and pushed in `6faa66b`.*

## Experiment log

| date | job(s) | what | outcome |
|---|---|---|---|
| 2026-08-28 | 11765-11767 | PAS Stage 1, early attempts | Failed on missing `HF_TOKEN`, then missing `scipy` (bug #6), then CUDA OOM at iter 9477 (bug #7) |
| 2026-08-29/30 | 11769 | PAS Stage 1 resumed | Reward diverged to ≈ −10M by iter 36199 (root cause: unclipped actions) — bug #8 |
| 2026-09-01/02 | 11802 | PAS Stage 1, post-`clip_actions` fix | Stable; timed out at iter 77508/79999 (3-day walltime) |
| 2026-09-03 | 11813 | Video eval, `stage1_model_31800.pt` | Video showed zero motion — later found non-representative, see bug #3 |
| 2026-09-04/05 | 11844→11846 | PAS Stage 1 finish + Stage 2 (anneal), full run | **Complete.** Stage 1 finished iter 39999; Stage 2 auto-continued and finished iter 79998/79999 |
| 2026-09-05 | 11847, 11848 | Sanity re-eval of known-bad ckpt + video, post repo-consolidation | Confirms repo move didn't break anything |
| 2026-09-05/06/07 | 11849, 11851, 11852 | Specialist training: Stairs, Rough, Flat | **All three complete** (see Specialists table — not yet gated on locomotion metrics, see note there) |
| 2026-09-06/07 | 11850 | Specialist training: Gaps (100% stepping_stones) | TIMEOUT at iter 7528/10000, plateaued — see Specialists table |
| 2026-09-08/09 | 11873 | Gaps resubmit | Restarted from iter 0 (bug #4), plateaued again by iter 6632, TIMEOUT |
| 2026-09-08 | 11874, 11875 | PAS final eval, `model_79998`, oracle + estimator-only | See PAS results table (measured under bug #10) |
| 2026-09-11 | 11914 | Gaps retrain, blended terrain (fix for the plateau) | *2026-09-12:* still PENDING, never started — nodes drained. See the Gaps redesign note before letting it run. |
| 2026-09-12 | 11918 | **Generalist** (`SPEC=Generalist`, 10k iters, seed 42) | PENDING (nodes drained) — the sensing-matched arm 1 |
| 2026-09-12 | 11919 | **Cross-terrain eval matrix** (`a100/eval_matrix.py`): flat/rough/stairs + PAS oracle/estimator × 5 terrains × d∈{0.25,0.5,0.75}, + height-scan ablation at d=0.5 — 87 cells | PENDING (nodes drained). Resumable; resubmit once 11918 (and a Gaps specialist) finish to add those rows |

## PAS (reference result) results

Two-stage training: Stage 1 ("oracle," `Unitree-Go2-PAS-Oracle`) trains an actor that reads a true privileged terrain latent; Stage 2 ("anneal," `Unitree-Go2-PAS-Anneal`, resumed from Stage 1) anneals it toward an estimator that predicts that latent from proprioception alone (`anneal_base=0.9998` per iteration). See `docs/07-pas-implementation.md` for the architecture.

**Checkpoints**: Stage 1 final = `logs/rsl_rl/go2_pas/slurm_stage1/model_39999.pt`. Stage 2 final = `logs/rsl_rl/go2_pas/2026-09-05_17-59-43_stage2/model_79998.pt`. Both also on HF at `stage1/model_39999.pt` / `stage2/model_79998.pt`.

**Critical eval gotcha**: `anneal_prob` (how much the actor trusts the true latent vs. the estimator's prediction) is **runtime state on `PasActorModel`, not stored in the checkpoint** — it's reset to `initial_anneal_prob=1.0` on load. Every eval silently runs in oracle mode unless `eval_checkpoint.py --anneal-prob 0.0` is passed explicitly. By the end of Stage 2 training, `anneal_prob ≈ 0.9998^40000 ≈ 0.0003` — i.e. the policy was almost always trained/tested using the *estimator's* prediction, not the oracle latent — so oracle-mode eval is actually evaluating an operating regime the final policy barely used during training.

*2026-09-12:* every number in this table was measured with the terrain curriculum live (bug #10). Each is valid as "this checkpoint under its training conditions", but they are **not comparable to other policies' numbers**. Superseded for `model_79998` by the per-terrain table below; job 11919 (once it runs) provides the equivalent for the specialists/generalist.

| checkpoint | mode | survival | mean ep. length | achieved/commanded speed | stalled-while-commanded |
|---|---|---|---|---|---|
| `model_77400` (intermediate) | oracle (`anneal_prob=1.0`) | 73.7% | 787.7/1200 | 15% | 31.6% |
| `model_77400` (intermediate) | estimator-only (`anneal_prob=0.0`) | 32.9% | 356.6/1200 | 15% | 29.9% |
| `model_79998` (**final**) | oracle | 75.5% | 784.4/1200 | 9% | 30.4% |
| `model_79998` (**final**) | estimator-only | **42.0%** | 450.8/1200 | 9% | 28.9% |
| `stage1_model_31800` (pre-`clip_actions` fix, for reference) | oracle | 77.5% | 821.5/1200 | 18% | 25.2% |

**Reading these**: estimator-only (the actually-deployable mode — real hardware has no privileged terrain sensor) improved 32.9%→42.0% survival over the last ~2600 training iterations, but achieved-speed-as-%-of-commanded dropped for *both* modes between the two checkpoints (15%→9%) — survival is improving without locomotion clearly improving, possibly a shift toward more conservative/cautious behavior. One data point each way, not conclusive.

**Open call, not yet resolved**: PAS oracle-mode is a legitimate *sim-only* baseline (with a privileged-information advantage no specialist has), but estimator-only — the only mode a real robot could run — falls roughly twice as often as oracle and hasn't clearly gotten better at walking as it's gotten better at surviving. Whether that's worth further Stage-2 investment before anchoring the generalist-baseline comparison on it, or whether it's accepted as-is (sim ablations were always meant to carry the empirical weight, per the original project framing), is an open decision.

*Correction 2026-09-12:* the privileged-information caveat above points the wrong way. Every specialist reads raw `height_scan`, so **oracle** PAS (encoded height scan + base velocity + friction) is the roughly sensing-matched comparison, and **estimator-only** (no height scan at all) is the unfairly handicapped one. The open call is also moot: PAS is no longer arm 1. It adds reward terms no specialist has (`energy`, `joint_vel_l2`) and got ~8× a specialist's env steps (80k vs. 10k iterations at the same 8192 envs × 24 steps). The sensing-matched arm 1 is `Unitree-Go2-Generalist` (job 11918), and PAS is reported separately as a replication result.

### Pinned-condition PAS oracle vs. estimator, per terrain class (2026-09-12, laptop)

`model_79998`, 128 envs, 1200 steps, `--terrain <class>` with **no `--difficulty`** (uniform spread over rows) — run directly via `eval_checkpoint.py` rather than `a100/eval_matrix.py`, which has no flag to omit `--difficulty`. Chosen specifically because this is the one path proven unaffected by bug #12 (the fix was still stuck on the cluster checkout, unpushed, at the time these ran). Supersedes the bug-#10/#11 table above for `model_79998`.

| terrain | mode | fall % | mean ep. length | achieved/commanded | stalled % |
|---|---|---|---|---|---|
| flat | oracle | 0.0 | 1000.0 | 8% | 31.4% |
| flat | estimator | 0.8 | 994.6 | 8% | 30.4% |
| rough | oracle | 49.1 | 545.9 | 11% | 22.2% |
| rough | estimator | 46.7 | 586.9 | 11% | 21.5% |
| stairs | oracle | 0.0 | 1000.0 | 8% | 31.4% |
| stairs | estimator | 0.8 | 994.5 | 8% | 30.6% |
| gaps | oracle | 74.0 | 317.7 | 8% | 32.1% |
| gaps | estimator | 93.1 | 97.4 | 10% | 31.0% |
| mixed | oracle | 44.6 | 608.6 | 9% | 28.8% |
| mixed | estimator | 59.4 | 455.0 | 10% | 27.6% |

**Reading these**: oracle and estimator track closely on flat/stairs/rough/mixed (within a few points), but diverge sharply on `gaps` (74.0% vs 93.1% fall, and mean episode length drops from 317.7 to 97.4 steps). The distilled proprioception-only latent holds up on the terrain PAS's native mix weighted heavily, but degrades specifically on gap-crossing — the one class PAS's own training gave only 15% weight to (see "Terrain specialists" below and objective.md's Gaps redesign). This is a real signal, not a bug-#12 artifact: none of these cells go through the pinned-difficulty path.

**Second observation: root-caused, see bug #14.** Achieved speed sits at a near-constant 8–11% of commanded across every terrain and both modes, including `flat`/`stairs` where fall rate is ~0%. **Read the fall-rate table above through bug #14 before concluding "PAS has flat/stairs solved" — it doesn't.**

## Terrain specialists

Stock single-stage PPO (`VelocityOnPolicyRunner` + `unitree_go2_ppo_runner_cfg()`, no terrain encoder / privileged state / annealing) — a specialist only ever sees one terrain class, so it has nothing to disambiguate. All four share one observation space (234-dim actor obs including `height_scan`) so a later gating network can blend them; see `_unitree_go2_specialist_env_cfg` in `env_cfgs.py` for why the Flat specialist uses a flat-only *terrain generator* rather than `unitree_go2_flat_env_cfg()`'s plane (which deletes `height_scan` from the obs and would break blendability). *2026-09-12:* terrain classes are now defined once in `TERRAIN_CLASSES` (`env_cfgs.py`), and the tasks register with `HfSyncVelocityOnPolicyRunner` (identical unless `HF_CHECKPOINT_REPO` is set). The existing task configs were verified unchanged against HEAD.

| specialist | terrain mix (training) | budget reached | error_vel_xy (final) | fell_over/ep | illegal_contact/ep | ep. length | checkpoint |
|---|---|---|---|---|---|---|---|
| Flat | 100% flat | 9999/10000 ✅ | 1.45 | 0 | 0 | 1000/1000 | `logs/rsl_rl/go2_spec_flat/2026-09-06_17-17-15/model_9999.pt` |
| Stairs | pyramid_stairs + inv, equal weight | 9999/10000 ✅ | 1.81 | 0 | 0.21 | 980/1000 | `logs/rsl_rl/go2_spec_stairs/2026-09-05_22-37-43/model_9999.pt` |
| Rough | random_rough + wave, equal weight | 9999/10000 ✅ | 1.88 | 0 | 0.50 | 983/1000 | `logs/rsl_rl/go2_spec_rough/2026-09-06_12-07-49/model_9999.pt` |
| Gaps (attempt 1) | 100% stepping_stones | 7528/10000 (TIMEOUT) | 0.03 (artifact — see below) | 807.7 | 47.9 | **9.8/1000** | archived, `logs/rsl_rl/_archived_go2_spec_gaps_100pct_plateaued/2026-09-06_09-11-12/` |
| Gaps (attempt 2) | 100% stepping_stones | 6632/10000 (TIMEOUT, restarted from 0 — bug #4) | similar | similar | similar | ~13.6/1000 | archived, `.../_archived_go2_spec_gaps_100pct_plateaued/2026-09-08_00-10-45/` |
| Gaps (attempt 3) | **20% stepping_stones / 40% flat / 40% random_rough** (fix, 2026-09-11) | job 11914, pending (nodes drained) | — | — | — | — | — |
| GapsWarm (redesign, 2026-09-12) | **100% stepping_stones, warm-started from Rough** | not submitted | — | — | — | — | would be `logs/rsl_rl/go2_spec_gapswarm/` |

**Flat/Stairs/Rough beat PAS's own tracking error** (2.09 oracle-mode `error_vel_xy` vs. 1.45-1.88) and have far better survival — the expected, positive specialist-vs-generalist result.

*Correction 2026-09-12:* that comparison isn't valid as stated (bug #11). `error_vel_xy` accumulates over each episode, and PAS's episodes (784 steps) were shorter than the specialists' (~980-1000), so the numbers aren't on a common scale. PAS actually accumulated its larger total in *fewer* steps, so the direction probably survives, but it's unproven. Separately, **none of Flat/Stairs/Rough has been through `eval_checkpoint.py`**. They were gated only on training-log `error_vel_xy` and termination counts, which can't tell walking from bracing: a policy standing still scores roughly its accumulated commanded speed, the same range as the 1.45–1.88 above. Flat (0 falls, full-length episodes, `error_vel_xy` 1.45 on the easiest terrain there is) is the one to check first. Job 11919 provides the pinned, per-step numbers, including stalled-while-commanded %. Don't treat these as finished assets until it reports.

**Gaps plateaued reproducibly, twice, under the original 100%-stepping_stones design.** Reward flat at ≈ −6 to −8 within a few hundred iterations both times, no improvement over 6600-7500 iterations each attempt, episodes ending in ~10-14 steps (immediate falls). The low `error_vel_xy` in that state is **not** good tracking — it's an artifact of episodes ending before velocity error has time to accumulate. Diagnosed as a task-design problem: isolating gap terrain at 100% gives a cold-start policy no easy terrain to learn basic locomotion on before also having to solve gap-crossing. Fixed 2026-09-11 by blending in flat/rough terrain, mirroring PAS's own gap terrain (which mixes `stepping_stones` at only 15% into 85% easier ground). **When job 11914 finishes, update this table and re-run the same plateau check (sample `Mean reward`/`Mean episode length` every few hundred iterations) before trusting the result — the fix is a reasonable bet, not a proven one yet.**

*2026-09-12, redesign:* the blended fix trains 80% on flat/rough — nearly the union of the Flat and Rough specialists' terrain. That makes it barely a gap specialist, hands the gating network a near-redundant expert, and quietly breaks the specialist framing for this arm. The diagnosis (cold start) points at initialization rather than terrain dilution. `Unitree-Go2-Spec-GapsWarm` keeps 100% `stepping_stones` and starts from the Rough specialist's weights via `a100/warm_start_ckpt.py`: iteration reset to 0, fresh Adam moments, `common_step_counter` 0, observation-normalizer stats reset (reason below). The warm-start round trip was verified in a scratch dir: weights identical to the source, idempotent on resubmit, and `local_ckpt_resume.py` picks it up as iteration 0. Rough was chosen over Flat because its action std stayed higher (0.43–0.57 vs. 0.34–0.46), leaving more exploration. **Open decision:** job 11914 hasn't started. Either cancel it and submit `SPEC=GapsWarm sbatch a100/train_specialist_slurm.sh`, or let both run and compare. If 11914 does run, evaluate it on `--terrain gaps` so the table reports gap competence rather than blended-terrain competence.

### Observation: the height scan's terrain signal is near the injected noise floor (2026-09-12)

Actor `height_scan` has `Unoise(±0.1 m)` applied before `scale=1/5` (mjlab's pipeline is compute → noise → clip → scale). Noise alone therefore gives a per-ray std of 0.02/√3 ≈ **0.0115** in scaled units. The trained specialists' actor observation-normalizer std over the 187 height-scan dims:

| specialist | min | median | max |
|---|---|---|---|
| Flat | 0.0118 | 0.0118 | 0.0164 |
| Rough | 0.0123 | 0.0128 | 0.0167 |
| Stairs | 0.0119 | 0.0124 | 0.0162 |

Rough and Stairs sit barely above Flat. Assuming the noise is independent, the terrain-plus-body-motion component is only ~0.005 scaled (~2–3 cm), well below the ~5.8 cm noise std. That is plausible given `ROUGH_TERRAINS_CFG`'s mild geometry (stairs ≤10 cm, rough noise 2–10 cm). This doesn't prove the policies ignore the scan: 187 spatially correlated rays can be pooled. But it's a real risk for the switching design, whose reactive classifier and gating input both assume the scan is informative. **Test**: the height-scan ablation cells in job 11919. If fall rate and error barely move with the scan replaced by a constant, the terrain signal (noise level, difficulty range) needs fixing before Phase 4. It also means any warm start onto stepping stones (gaps drop 2 m → ~17σ on these stats) must reset normalizer statistics, which `warm_start_ckpt.py` does by default.

## VLM navigation pipeline: SARO + VLM specialist selection (2026-09-17, laptop, branch `vlm-pipeline`)

User decision, 2026-09-16: build SARO's high-level pipeline (arXiv:2407.16412: VLM planning, perception
and discriminator in a closed sub-task loop), modified so the VLM also chooses which of the three
specialists runs, and follow SARO's own objective (goal tracking across one terrain intermediation). This
is a separate line from `objective.md`'s person-following 2×2. VLM: local Gemma-4-E4B (Q4_K_M + vision
projector) via llama.cpp (`scripts/vlm_server.sh`). Code: `unitree_rl_mjlab/src/vlm_nav/`; tests:
`unitree_rl_mjlab/tests/test_vlm_nav.py`.

- **Phase 0 (done)**: 848×480 RGB-D camera ahead of the Go2 head renders at 7–15 ms/step alongside the
  specialists; specialists hot-swap mid-episode; velocity command driven externally. Fits in 8 GB with the
  VLM server loaded.
- **Simulated courses (done)**: straight strips (`flat`, `stairs_up`, `stairs_down`, `rough`, `multi`) with
  ground-truth terrain regions, training-distribution geometry and a goal flag. Default visual is
  "tiled" (see bug #19).
- **Phase 1 calibration (done, fails the pre-registered bar)**: with perfect, ground-truth specialist choice,
  stairs crossings succeed only 75–78% at 0.05 m risers and 0–28% at 0.07–0.09 m; rough ground is 100% to
  0.08 m noise. Every fall is `illegal_contact` (knee/calf on a step edge), none `fell_over`. The switch is not the
  cause: the stairs specialist run throughout does as badly or worse. Full numbers and the decision options:
  `coordination/results/vlm-nav-phase1-calibration.md`; pre-registration:
  `coordination/results/vlm-nav-phase1-preregistration.md`. **Awaiting a user decision before the
  confirmation run.** Exploratory (seed 101): the best specialist is not the one named after the
  terrain and flips with step height (rough beats stairs on 0.05 m up-stairs, stairs beats rough at
  0.07 m). Under SARO's orientation-only fall definition, every specialist descends 0.05 m stairs 32/32.
- **Phase 2 perception (done)**: Gemma-4-E4B does not perceive these stairs. With neutral task
  instructions it plans "none" on 4/4 stairs start frames, the selector says "stairs" 0/32, and it
  describes a staircase as "a flat, gridded floor". Rough ground is recognized sometimes. SARO's box
  prompt returns `[0,0,0,0]` or the full frame; depth geometry locates edges to 0–8 cm and is the
  executor's default where-source. Gotcha: an instruction naming the intermediation let the planner
  score 8/8 without looking. `coordination/results/vlm-nav-phase2-perception.md`.
- **Closed loop (smoke)**: executor + ground-truth VLM 4/4 on rough; real Gemma 2/2 on rough, 0/2 on
  0.05 m up-stairs (planned "none", flat specialist stuck).
  `coordination/results/vlm-nav-closed-loop-smoke.md`.
- **Phase 1 confirmation, restricted to `rough` (done, 2026-09-18)**: stairs stayed blocked, so the
  pre-registered confirmation ran on the one course the specialists handle reliably. 192 trials/arm,
  fresh seeds 200/201, 3 goal offsets. **Gate A passes** (oracle 97.4%); **Gate B fails** — always
  running the rough specialist (98.4%) matches perfect oracle switching (97.4%), CIs overlapping.
  A single-obstacle course can't demonstrate switching value even with perfect choice: the wrong
  fixed specialist is costly (flat 73%, stairs 53%), but *one* good fixed choice is enough. The
  gate that would test the actual claim needs a route where the best specialist changes mid-course
  (`multi`), which contains stairs and is blocked on the same specialist-reliability problem.
  `coordination/results/vlm-nav-phase1-confirmation-rough.md`.
- **Closed-loop 4-arm comparison on `rough`, real Gemma (done, 2026-09-18)**: 16 trials/arm — VLM
  nav+VLM policy 100%, VLM nav+oracle policy 94%, VLM nav+fixed specialist 94%, ground-truth-both
  100%. **Statistically indistinguishable**, as predicted by the confirmation above: every arm sits
  at the course's ceiling. The VLM's own decisions were mixed underneath the good outcome (planned
  "stairs" 3× on a stairless course; selector split ~50/50 rough vs. flat) — the 100% is not evidence
  the VLM chooses well, only that this course can't tell a good chooser from a mediocre one.
  `coordination/results/vlm-nav-closed-loop-smoke.md` (§ "Four-arm comparison").
- **Net**: the stairs specialist's unreliability (not the VLM, not perception) is now the critical
  path — it blocks the only course design (`multi`) that could show anticipatory/reactive switching
  beating a fixed policy. See `PROGRESS_REPORT.md` for the decision this needs.

## Person-following (2026-09-21, laptop, branch `vlm-pipeline`)

`objective.md`'s setting reduced to its locomotion core: a scripted kinematic leader
(`src/vlm_nav/leader.py`, a fixed-base entity mjlab auto-wraps as a mocap body,
non-colliding and in the camera-only geom group so it cannot perturb the 187-dim
`height_scan`) walks a flat course at a speed that changes occasionally, and the robot
holds a standoff with `controllers.follow_command`. Runner: `scripts/vlm_nav_follow.py`.

- **Ground-truth leader, 2.5 m gap: gap error 0.159 m RMS / 0.256 m max** over 42 s and
  7 speed changes including a 4 s full stop, no falls. Flat specialist throughout.
- **Two control defects the varying speed exposed, both invisible to a constant-speed
  leader.** (1) Pure P-control against a *moving* set-point droops: it settles where
  `k_range * err` equals the leader's speed, so every speed change parked the robot at a
  different wrong distance (0.82 m RMS). Fixed with line-of-sight velocity feed-forward,
  estimated by differencing observed leader positions. (2) `FollowGains.v_max` was 0.9
  against a 0.95 m/s leader, so the robot *could not* close a gap once it opened -- error
  grew monotonically through the burst and peaked at exactly the moment the leader slowed
  (+1.19 m). v_max is now 1.0, the specialists' final-curriculum command limit, which
  caps how fast any leader may walk.
- **The camera cannot see a person at the follow distance.** Pitched 15 deg down with a
  42.5 deg vertical FOV, it sees only ~6.25 deg above horizontal: at 1.5-2.5 m only the
  leader's legs are in frame. Gemma-4-E4B describes those as "two blue cylindrical
  objects" and detects `person` 0/3 at 1.5/2.5/4.0 m. At **6 m the whole figure is in
  frame and detection succeeds**. Any camera-driven follower on this rig must stand off
  ~6 m, or the camera must be re-aimed.
- **The VLM's horizontal localization is far better than its boxes.** At 6 m the detected
  box centre was 423.2 px against a true 424.0 px -- a **0.08 deg bearing error** -- while
  its vertical extent was flatly wrong (it boxed ground *below* the figure). This is the
  usable signal: `perception.person_from_bearing_column` takes only the column from the
  VLM and recovers range from depth returns standing above the floor, discarding the box's
  y extent entirely. Same division of labour as the terrain edge estimator (bug #16 era):
  VLM for *what/which way*, geometry for *how far*.
- **Closed loop on VLM perception, 6 m gap**: runs end to end, but detection is
  unreliable (17/84 queries over a full run) and gap error is many times the
  ground-truth arm.
- **The follower locked onto the goal flag -- and its own metric said it was doing
  fine.** On a flat course the person is not the only object standing above the floor;
  the goal flag is too, in the same camera-visible geom group. Once the person outran
  reliable detection, `person_from_bearing_column` latched onto the flag and the
  controller held station 6.15 m from it, *frozen*, while the leader receded to 16.5 m.
  The estimate-derived range read 6.15 against a 6.0 m set-point the whole time -- a
  0.15 m "error" -- because it was measuring the distance to the wrong object. Only the
  independently logged ground-truth range exposed it. **This is the repo's dominant
  failure mode again** (see the review note on metrics that answer an adjacent
  question): a follower's self-reported range error cannot validate a follower, because
  it is computed from the very estimate under test. Fixed by removing the flag from the
  follow course (`CourseSpec.goal_marker=False`), but the lesson generalises -- any
  real scene has other vertical objects, so a production follower needs the person
  *identified*, not merely "something above the floor".

## Two-rate perception: YOLO tracker + VLM planner (2026-09-21, laptop, branch `vlm-pipeline`)

The closed-loop follow run put the VLM *inside* the control loop, which is why it lost
the person: ~2 s per call, and only 11 of 84 calls yielded a usable position, so the
robot navigated on estimates up to half a second stale and mostly missing. The fix is
structural, not a better prompt -- split *what* from *where*:

| layer | job | rate | measured |
|---|---|---|---|
| VLM | name the target, choose the specialist, judge sub-task completion | ~0.2 Hz, off the control path | ~2 s/call |
| detector | where is that target in this frame | every control step | **3.5-3.9 ms** (YOLO11s, RTX 5060) |
| depth geometry | how far | every step | 0-8 cm |

`src/vlm_nav/detector.py` is the seam (`Detector` protocol + `YoloDetector`); swapping
the detector never touches the planner. This is what makes the pipeline retargetable:
the VLM names a class in language, the detector localizes it at control rate.

- **Stock COCO YOLO cannot see this project's leader.** It reads the capsule legs as
  "baseball bat" (0.79 conf) and the head sphere as "sports ball", and at the 6 m follow
  distance detects *nothing* even at conf 0.05. Camera pitch is not the cause -- re-rendered
  at 0 deg and 8 deg pitch, same result. This is an **appearance gap specific to
  flat-shaded mjlab geoms**; on real hardware a real person is COCO's home ground, so
  this does not carry to the robot.
- **Fine-tuning fixes it, and labels are free.** The leader's pose is set by us and its
  geom extents are fixed, so the exact 2D box is the projection of its 3D box
  (`CameraSpec.project_body`, inverse of the existing deprojection, pinned by a
  round-trip test to 6e-14 px). 1000 auto-labelled frames (140 negatives) ->
  YOLO11n, 40 epochs: **P 1.000, R 0.942, mAP50 0.951, mAP50-95 0.892**.
  Tools: `scripts/vlm_nav_make_detector_dataset.py`, `scripts/vlm_nav_train_detector.py`.
  Sim-only by construction; do not ship these weights to hardware.

## SARO task replication, paper protocol (2026-09-22, laptop, branch `vlm-pipeline`)

SARO's own task (arXiv:2407.16412 SS III.A): goal-tracking across `{P1 -> I -> P2}` with a
goal G given as a pose plus a language description L, **20 trials per intermediation**,
goals off-axis, full closed loop (planning -> perception -> discriminator double-check).
Run via `scripts/vlm_nav_run.py --arms vlm gt+oracle --num-envs 20 --seed 300
--goal-y-offset 0.8 --level L1`; artefacts under `logs/vlm_nav/saro_protocol/`.

| intermediation | VLM Overall | VLM Across | falls | ground-truth ceiling | planner's answer |
|---|---|---|---|---|---|
| stairs_up | **0%** | 0% | 20% | 75% | `none` 20/20 |
| stairs_down | **45%** | 45% | 55% | 95% | `none` 20/20 |
| rough | **100%** | 100% | 0% | 100% | `none` 17, `rough ground` 17, `stairs` 1 |

SARO's Table I for comparison (real robot, LLaVA-34B): stair 60% overall / 70% across /
88% stable-loc; ramp 25/50/67; gap 45/80/94; door 30/50/63.

- **The failure is perception, and it is upstream of everything else.** On both stairs
  courses the planner answered `intermediation: none` on **40/40** trials -- it never
  emits a `climb` sub-task because it never concedes there is anything to climb. The
  ground-truth arm crosses the same courses at 75%/95%, so the course is crossable and
  the locomotion works; the whole gap is the VLM. This reproduces the Phase-2 offline
  result (`stairs` 0/32) at the paper's own 20-trial scale and in closed loop.
- **The rough 100% is not evidence of perception.** On rough the planner still said
  `none` half the time (17/35) and hallucinated `stairs` once, and the selector chose
  the *flat* policy on 267 of 391 calls -- on a rough course -- yet every trial
  succeeded, because every specialist crosses rough. Same ceiling effect as gate B: a
  course where the wrong choice is survivable cannot measure choice quality. **Do not
  read the rough column as validation.**
- **Deviations from the paper, all forced**, and they bound what this comparison means:
  intermediations are stairs_up/stairs_down/rough (no ramp or door segment exists in
  this sim, gaps has terrain but no course segment); the low-level policies are this
  project's three specialists, not PAS (the PAS replication is degenerate, bug #14, and
  its estimator-only actor has a different observation width so it cannot enter the
  policy bank); the VLM is Gemma-4-E4B on one 8 GB laptop GPU, not LLaVA-34B on an
  8x3090 server; and pose is ground truth. **Because localization is perfect here, the
  honest comparison is against SARO's `Stable Loc` column, not its `Overall`.**

## Bugs found and fixed

Ordered roughly by how much they'd silently corrupt a result if unnoticed — read top-to-bottom before trusting any new eval number. *#10 and #11 (2026-09-12) belong near the top by that ordering; they're appended rather than renumbered because other entries reference these numbers.*

1. **Survival metrics can't distinguish walking from bracing.** `stage1_model_31800.pt` scored 80.5% survival / +39.5 return in the original `eval_checkpoint.py` (pre-fix) while translating a measured ~0 m over an 8-second rollout — bracing in place actively maximizes the energy/joint_vel/action_rate penalty terms, so a degenerate "don't move" policy looks healthy on survival alone. **Fix**: `eval_checkpoint.py` now also reports commanded-vs-achieved speed, velocity tracking error, distance travelled, and a "stalled while commanded to move" percentage with an explicit warning past 50%.

2. **`anneal_prob` isn't stored in the checkpoint** (see PAS section above) — every eval silently ran in oracle mode. **Fix**: `eval_checkpoint.py --anneal-prob <val>` override, with the active mode printed explicitly and a warning when evaluating a Stage-2 checkpoint in oracle mode without an explicit override.

3. **Single-environment qualitative videos are not representative.** `play.py` renders only env 0, and across three independent runs (11813, 11848, and a manual repro) env 0 happened to draw `UniformVelocityCommandCfg`'s "stand still" branch (`rel_standing_envs=0.05`) every time, showing zero net motion regardless of the checkpoint's actual walking ability. Confirmed the resample logic itself is correct (an explicitly reseeded RNG produces properly varied nonzero commands) — this is a sampling-representativeness gap, not a training or measurement bug. **Trust the 1024-env numeric eval over a single video for judging locomotion quality.** Not yet fixed at the tooling level — future work: force a nonzero command for video eval, or render/sample multiple envs. *(2026-09-12 precision: the offscreen renderer tracks `viewer.env_idx` (default 0) and also draws its 2 nearest neighbours (`max_extra_envs=2`) for context; the camera follows env 0.)*

4. **Specialist checkpoints were never uploaded to HF, and the resume logic didn't know that.** `PasOnPolicyRunner.save()` has HF-upload logic; the stock `VelocityOnPolicyRunner` used by specialists does not. `hf_sync_specialist.py` (the original design) only checked HF, found nothing, and concluded "start fresh" — causing Gaps' resubmission (job 11873) to silently restart from iteration 0 instead of resuming from 7528, wasting a full GPU-day re-discovering the same plateau. Only Flat/Stairs/Rough escaped this because each finished within one job's walltime, so no resume was ever needed. **Fix**: `a100/local_ckpt_resume.py` scans local disk directly (`logs/rsl_rl/<experiment>/*/model_*.pt` across all run dirs, confirmed to persist across job boundaries on this cluster) instead of querying HF. `train_specialist_slurm.sh` updated to use it; specialist jobs no longer need `HF_TOKEN`. *2026-09-12:* `hf_sync_specialist.py` deleted (in git history). The upload helper moved to `src/tasks/velocity/rl/hf_upload.py`, and specialist/generalist tasks now use `HfSyncVelocityOnPolicyRunner`, which pushes each checkpoint when `HF_TOKEN` is set at submission. Resume still reads local disk only.

5. **tyro CLI boolean flags require an explicit value here, counter to the usual `--flag`/`--no-flag` convention** — `play.py`'s `--video` and `train.py`'s `--agent.resume` both render as `--video {True,False}` in their own `--help` output and fail with "Missing value for argument" if passed bare. This was flip-flopped once mid-project: job 11800 failed on a bare `--video` → corrected to `--video True` → job 11813 succeeded → later mistakenly "fixed" back to bare `--video` based on a misreading of the already-corrected script → caught via a direct `--help` check and a reproduction test before it could break another job. **If touching either flag again, verify empirically (`scripts/play.py <task> --help`) rather than trusting intuition about tyro's convention.**

6. **Missing `scipy` dependency.** `mjlab==1.2.0` imports `scipy.interpolate` in `heightfield_terrains.py` but doesn't declare it — crashes ~17 minutes into a run, right after the HF checkpoint download, with `ModuleNotFoundError`. Fixed by an explicit `pip install scipy` in `train_pas_slurm.sh`.

7. **CUDA OOM from allocator fragmentation, not true memory exhaustion.** Warp's CUDA-graph replay draws scratch memory from CUDA's stream-ordered mempool at replay time; PyTorch's caching allocator doesn't cooperate with that pool and fragments it over thousands of PPO iterations, eventually failing a graph launch even with `nvidia-smi` showing headroom (killed job 11767 at iteration 9477). Fixed via `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

8. **Training reward divergence without action clipping.** Unclipped actions let PPO's policy drift to reward ≈ −10M / value loss ≈ 5×10¹² over one run (job 11769). Fixed via `clip_actions=6.0` in both Go2 PPO runner configs — committed in `6faa66b` (2026-09-11).

9. **Two divergent git clones** existed (`/dist_home/d_palmani/c-08/policyswitching` and `/dist_home/d_palmani/policyswitching`), with SLURM scripts hardcoding paths into the latter. Consolidated into the `c-08` checkout on 2026-09-05 (`logs/`, `eval_ckpts/`, `stage1/` merged in; the second clone deleted after verifying no unique content remained). All `a100/*.sh` scripts now point at `c-08`.

10. **The terrain curriculum stayed live during eval, so eval difficulty adapted to the policy under test.** `eval_checkpoint.py` reused the training env config (`play=False`), which keeps `terrain_levels_vel`. That term promotes an env to a harder terrain row whenever it walked more than half a patch and demotes it otherwise, *during the eval rollout* (envs also start at `max_init_terrain_level=5` of 10 rows). A better policy is pushed onto harder ground until it too starts failing, so survival % measures distance to the curriculum's equilibrium rather than capability — and **every cross-policy comparison of eval numbers so far is invalid**, including the PAS table. Same family as #1/#2: a metric answering a different question than the one asked. **Fix (2026-09-12)**: `apply_eval_conditions` (`env_cfgs.py`) removes `terrain_levels` and pins terrain explicitly. `--terrain` picks the task's own mix, one class, or `mixed`; `--difficulty d` collapses the grid to one row generated at exactly `d`; omitting it spreads envs uniformly over rows. The step-keyed `command_vel` curriculum is also removed, with the command range pinned to the final training stage and printed. All of this is on by default; `--keep-curricula` restores the old behaviour for checking a checkpoint against its own training log only.

11. **`Metrics/twist/error_vel_xy` accumulates over each episode, so it isn't comparable across policies.** `velocity_command._update_metrics` does `error_vel_xy += ‖cmd − vel‖ / max_command_step` every step until reset, so the logged value grows with episode length. The Gaps entry already caught the extreme case (0.03 with 10-step episodes), but the specialists-vs-PAS comparison still used it across different episode lengths — see the correction in the Specialists section. **Fix**: cross-policy tables use `eval_checkpoint.py`'s per-step linear velocity error, labelled as such in its output; `CLAUDE.md` says never to compare `error_vel_xy` across policies.

12. **`apply_eval_conditions`'s difficulty-pinning path (`num_rows=1, difficulty_range=(d, d)`) produces near-total immediate failure on at least the `flat` terrain class, independent of policy capability — fixed 2026-09-12.** Running the laptop slice of job 11919's design (PAS stage2, `--anneal-prob 1.0`, 128 envs, 1200 steps) via `a100/eval_matrix.py --only pas_oracle --difficulties 0.5`: `flat` scored 98.9% fall / mean episode length 30.2 steps, `rough` 99.3%, while `stairs` scored 0.8% and `mixed`/`gaps` were in between — backwards from what capability should predict (flat ground should be the easiest terrain, not the hardest). Isolated by re-running the *exact same checkpoint and terrain class* with `--terrain flat` and no `--difficulty` (i.e. the generator's default multi-row layout, difficulty spread uniformly instead of pinned to one row): **0% fall, 100% full-length episodes**, same 128 envs / 1200 steps. The only variable that changed was the difficulty-pinning path itself, so this is very likely a terrain-generator artifact of collapsing to `num_rows=1` with a single sub-terrain type at 100% proportion (spawn position/origin indexing is a likely suspect), not a real capability gap. **This puts every difficulty-pinned cell in job 11919 under suspicion** (all of `--difficulties 0.25/0.5/0.75`), not just `flat` — `stairs` happening to look "healthy" in the same run is exactly the kind of asymmetric partial breakage that makes the matrix's numbers plausible-looking but wrong if trusted as-is. **Needs investigation in `apply_eval_conditions` (`env_cfgs.py`) or mjlab's terrain generator before job 11919's difficulty-pinned columns, or any `eval_matrix.py` cell with an explicit `--difficulties` value, are trusted** — `--terrain <class>` with no `--difficulty` (uniform spread) is not affected and remains safe to use. Flagged to the cluster session in `coordination/inbox/to-cluster.md`, since `env_cfgs.py` is cluster-owned. **Consequence for the height-scan ablation specifically**: the ablation cells `eval_matrix.py` would generate for an ablatable policy also run through this same pinned-difficulty path (the ablation-difficulties default is 0.5), so gate (b) evidence from any ablation cell inherits this same suspicion until the pinning bug is understood — `--terrain <class>` with no `--difficulty` is the only currently-trustworthy way to compare a policy across terrain classes.

**Diagnosed and confirmed (2026-09-12).** `orchestrator-session` traced it to mjlab's `terrain_generator.py`: `_generate_curriculum_terrains` computes `difficulty = lower + (upper - lower) * ((sub_row + rng.uniform()) / num_rows)`, which already collapses to exactly `d` for every row when `lower == upper == d`, regardless of `num_rows`. So `apply_eval_conditions`'s `num_rows=1` (in `gen = replace(gen, num_rows=1, difficulty_range=(difficulty, difficulty))`) contributes nothing to the pin itself — it only collapses the terrain's x-extent to one patch and forces every env onto row 0 (`max_init_terrain_level` is never reset on this branch, so it stays at its curriculum default and gets clamped to row 0 on a 1-row grid). Tested the fix candidate directly (throwaway script, `env_cfgs.py` untouched): same PAS oracle checkpoint, `--terrain flat --difficulty 0.5`, but with the generator's `num_rows` left at its configured value (10) and only `difficulty_range=(0.5, 0.5)` set, plus `max_init_terrain_level=None` (spread over the now-uniform-difficulty rows) — **0% fall, 100% full-length episodes**, fully reproducing the difficulty-unpinned result. Confirms `num_rows=1` (not the difficulty arithmetic) is the cause. Fix (for the cluster session, `env_cfgs.py` is theirs): in the pinned branch of `apply_eval_conditions`, replace `gen = replace(gen, num_rows=1, difficulty_range=(difficulty, difficulty))` with `gen = replace(gen, difficulty_range=(difficulty, difficulty))` plus `cfg.scene.terrain.max_init_terrain_level = None`. Not yet applied to the repo (cluster-owned file). The 30.2-step mean episode length on the broken `flat` cells (well under a second) is still unexplained by "walked off a one-patch terrain" and was flagged as a loose end before this test — resolved now that the fix candidate reproduces full-length episodes, so it was a downstream symptom of the row-0/no-spread collapse, not a separate bug.

**Fix applied and merged (`0e22346` → `origin/main`), then end-to-end validated (2026-09-12).** The cluster session applied exactly the candidate above, plus an assertion that `gen.difficulty_range` actually took after the `replace` (belt-and-suspenders against the pin silently no-op'ing again). That assertion only checks the config field was set, not that the *generated terrain* actually differs across difficulty values — a real gap, since job 11919's whole `--difficulties 0.25/0.5/0.75` design depends on that link holding. Verified it directly: PAS oracle, `--terrain rough` and `--terrain gaps`, `--difficulty {0.25, 0.5, 0.75}`, 128 envs, 1200 steps (6 cells; `rough`/`gaps` chosen over `flat`/`stairs` because PAS has dynamic range there — 49.1%/74.0% fall under uniform spread vs. `flat`/`stairs`' 0.0–0.8%, floored).

`rough` is cleanly monotonic — fall % 8.4 (d=0.25) → 33.3 (d=0.5) → 60.3 (d=0.75), mean episode length 947.4 → 701.9 → 440.8 in lockstep. The uniform-spread baseline (49.1% fall) lands between the d=0.5 and d=0.75 points, exactly where averaging over all rows should put it. **The pin does what it claims for `rough`.**

`gaps` is flat — 73.5 / 71.3 / 72.6% across the same three difficulties, statistically indistinguishable, matching the uniform-spread baseline (74.0%) at every point. Before treating this as a lingering bug, checked mjlab's `BoxSteppingStonesTerrainCfg.function` (`mjlab/terrains/primitive_terrains.py`) directly: stone *size* is fixed at the range midpoint regardless of difficulty (`base_size = (avg_s_min + avg_s_max) / 2`, never scaled); average stone *distance* only scales `difficulty` over the **upper half** of its configured range (`avg_distance = mid + difficulty * (d_high - d_low) / 2`, spanning `mid` at d=0 to `d_high` at d=1), moving only 0.3875 m → 0.4625 m over 0.25–0.75 — 7.5 cm on a mostly-fixed-size course. *Correction, same day:* that's only the spacing dimension. `stone_size_variation` (0.2), `displacement_range` (0.1) and `stone_height_variation` (0.1) all scale **linearly from zero**, so `displacement_range` and `stone_height_variation` each **triple** over the tested range (0.025 m → 0.075 m) — the stones get materially less uniform and more scattered, even though average spacing barely moves. So this is not a difficulty-invariant terrain overall, only a difficulty-invariant *spacing*. The flat fall-rate curve is at least as well explained by **PAS being near its own failure ceiling** (73% fall already at the easiest difficulty tested, leaving little headroom to get worse) as by the terrain lacking a difficulty response — the two are confounded here and this run can't distinguish them, since PAS was the only policy available on the laptop at the time. **Don't read 11919's `gaps` column as "expect no trend regardless of policy"** — a policy with real headroom (a trained specialist, once reachable) may well show one; the honest summary is "weak spacing response, real geometry-variation response, and an unresolved confound between terrain and PAS-ceiling for the flat fall-rate result specifically."

**Net: the fix is validated end-to-end for at least one terrain class with real dynamic range (`rough`), and `gaps`' flat response has an independent, source-level explanation rather than being a second instance of the same bug.** `--difficulties 0.25/0.5/0.75` is safe to trust for `flat`/`rough`/`stairs`/`mixed`; `gaps` difficulty sweeps specifically should not be expected to show much of a trend regardless of which policy is under test.

**Applied (2026-09-12, cluster session).** Independently re-derived and confirmed via a third, separate reproduction before touching the file: ran the actual `Unitree-Go2-PAS-Anneal` task + `stage2/model_79998.pt` (oracle mode) directly (not a monkeypatch) with three configurations — (1) `apply_eval_conditions` as originally written (`num_rows=1`): 19/32 envs fallen by step 60; (2) `difficulty=None` (uniform): 0/32; (3) the fix candidate, restoring `num_rows` to its configured 10 while keeping `difficulty_range=(0.5, 0.5)` and setting `max_init_terrain_level=None`: 0/32, matching (2) exactly. Same conclusion as the laptop's independent test via a different code path — strong convergent evidence. Applied the exact fix to `apply_eval_conditions` in `env_cfgs.py`: dropped `num_rows=1`, kept only `difficulty_range=(difficulty, difficulty)`, added `cfg.scene.terrain.max_init_terrain_level = None`, plus a defensive `assert gen.difficulty_range == (difficulty, difficulty)` so a future regression in the pin raises loudly rather than producing a plausible-looking wrong number the way this one did. Re-verified end-to-end through the real `scripts/eval_checkpoint.py` CLI (not just the raw function) post-fix. **Every difficulty-pinned number produced before this commit (all of it, since job 11919 never started) is superseded, not salvageable** — nothing to correct in the results tables since none were ever recorded from the broken path.

13. **`a100/eval_matrix.py` silently omits every specialist on the laptop, and `summary.md` doesn't record the omission.** All five `LOCAL_POLICIES` entries (flat/rough/stairs/gaps_blend/gaps_warm) resolve only through `latest_local_checkpoint(mjlab_dir, experiment)`; specialist checkpoints exist on cluster local disk only (bug #4), so on the laptop every one takes the `[SKIP]` branch. That prints a single stdout line, then `summary.md` is rendered from `order`, built only from the result rows that actually exist — producing a clean, plausible-looking matrix table containing only PAS rows, with nothing *in the file* indicating five policies were never evaluated. Same class of bug as #12's sibling problem the cluster already fixed for PAS's own local-path fallback (2026-09-12) — still live for every specialist. **Fix specced in `coordination/inbox/to-cluster.md` (commit b396892)**: give each `LOCAL_POLICIES` entry an `hf_stage` (= its experiment name) and the same local-then-HF fallback PAS got, once the specialists' private-HF backfill lands. Until then, any laptop-generated `summary.md` must be labelled PAS-only in write-ups quoting it — see `coordination/results/go2_pas_stage2-step79998-matrix-analysis.md` for an example.

14. **PAS's near-perfect survival on `flat`/`stairs` (the pinned-condition table above) is bug #1's mechanism recurring, not "PAS has these terrains solved."** Achieved speed is a near-constant 8–11% of commanded across every terrain and both modes in that table, including the two cells with 0% fall. Checked for a measurement artifact and found none: (1) two independent measures agree — `mean_actual_speed` (base-frame velocity magnitude) and `distance_rate` (actual XY displacement per step) match to within 1–5% in all ten cells (e.g. flat oracle 0.082 vs 0.081 m/s; rough estimator 0.110 vs 0.108 m/s) — not a units or frame bug in either, though `distance_rate` is per-step path length, not net displacement, so this rules out a measurement error without ruling out e.g. circling; (2) the pinned command range is not an eval-time mismatch — `unitree_go2_pas_env_cfg` never overrides `commands`, inheriting the same `command_vel` curriculum (and the same final-stage range `eval_checkpoint.py` pins) as every other policy in this repo, so PAS trained on exactly the range it's being evaluated against. What *is* unique to PAS: `unitree_go2_pas_env_cfg` adds `energy` (`joint_torques_l2`, weight -1e-6) and `joint_vel_l2` (weight -0.002) reward terms, its own comment noting they're "present in mjlab's reward library but not wired into the stock Go2 tasks" — i.e. absent from every specialist. Both terms are maximized by minimizing motion: this is bug #1's exact mechanism (`stage1_model_31800.pt`, 80.5% survival while translating ~0 m), recurring in a checkpoint nobody re-checked for it, except PAS still moves (~8-11% of commanded, not 0%), so it wasn't caught by survival or by `error_vel_xy` — which, per bug #11, looks *better*, not worse, for a slower policy. **Consequence: don't read the fall-rate table above (or the original bug-#10-tainted PAS table) as "PAS solved flat/stairs."** It survives in those cells because it is barely locomoting, uniformly, everywhere — not because it handles those terrains well. Strengthens the case (independent of the confounds already listed under "Review 2026-09-12") for dropping PAS as arm 1 in favor of the sensing-matched generalist. Not yet investigated: whether this is specific to the `energy`/`joint_vel_l2` weights chosen, or a broader Stage-2 optimization artifact.

15. **`fall_pct`/`survival_pct` are pooled per episode, not per env, so both silently over-weight whichever sub-population cycles through episodes fastest — same family as #1/#11, a metric answering a different question than the one asked.** `eval_checkpoint.py`: `fall_pct = 100 * total_fails / total_episodes`, where `total_episodes` accumulates over the *whole rollout*, not once per env — an env that dies every 15 steps over a 1200-step rollout contributes ~80 episodes to the denominator; an env that survives the full window contributes 1. Surfaced by the flat specialist's `mixed`-terrain cell (job-11919-shaped laptop matrix, 2026-09-12): naive per-class-column arithmetic predicted ~58% fall on `mixed` (env-weighted average of its flat/rough/stairs/gaps columns), observed was 96.9% — a ~39-point gap initially suspected to be a terrain-generation bug (allocation checked exactly correct: `mixed`'s 20 columns split 5/5/5/5 across classes, no rounding skew) and then a lateral-drift artifact (rejected by a controlled test: zeroing commanded `lin_vel_y` left both `mixed` and a single-class control terrain's fall rate unchanged, the opposite of the predicted dissociation). Root cause: recorded episode counts for the flat specialist were flat 128, rough 235, stairs 192, **gaps 7736** — the 25% of envs spawned on `stepping_stones` (bug #12's low-difficulty-response terrain, findings.md above) die so fast they generate ~93% of all episodes counted in the `mixed` cell, so the pooled `fall_pct` is essentially the gaps number wearing a "mixed" label. **Confirmed quantitatively, not just qualitatively**: predicting `mixed` as an *episode-weighted* (not env-weighted) average of the four single-class cells matches observed values to within 1.3 points for all four policies checked (flat 96.7 predicted / 96.9 observed, rough 96.5/96.2, stairs 97.4/97.1, pas_estimator 74.6/76.0). Confirmed from the other direction too: recomputing an unbiased hazard rate (`total_fails / (num_envs · steps · dt)`, i.e. falls per robot-minute rather than falls per episode) makes `mixed` match the env-weighted prediction almost exactly (predicted/observed ratios 0.95–1.09 across the four policies) — the anomaly is entirely a property of the metric, not the terrain. **This is not just a `mixed`-terrain problem**: `fall_pct` (and `survival_pct`, which shares the same denominator) shouldn't be compared across *any* two cells whose survival-time distributions differ, including two policies on the *same* single terrain class — on `rough`, `fall_pct` puts the flat specialist at 2.4× the rough specialist's fall rate (65.5% vs 27.2%), but the hazard rate puts them at 3.3× (3.01 vs 0.92 falls/robot-min) — so raw `fall_pct` differences are compressed relative to the underlying rate whenever the worse policy also dies faster (as it usually does). **Good news: gate 1's column-wise verdict survives the fix.** Recomputed as hazard rate, the specialist matrix's rough and stairs columns still pick the matching specialist as the least-often-falling policy (rough column: rough 0.92 < stairs 1.82 < flat 3.01 falls/robot-min; stairs column: stairs 1.23 < rough 2.01 < flat 2.46) — specialization is real and robust to the metric fix, only its exact magnitude was distorted. No unbiased survival measure exists in `eval_checkpoint.py`'s output yet; a fix is being proposed to the cluster session (`coordination/inbox/to-cluster.md`) rather than made here, since that script is cluster-owned. Until it lands, treat `fall_pct` cross-cell comparisons as directionally suggestive, not precise, and recompute the hazard rate from `episodes`/`fall_pct` in the raw JSON (`total_fails = round(fall_pct * episodes / 100)`) before trusting a magnitude claim.

16. **Gate 2a's rough/stairs "measured zero-noise ceiling" (~0.51 each) is substantially an artifact of collecting at `difficulty=None`, which labels flat ground as `stairs` and `rough` — diagnosed 2026-09-16 from the committed data, confirmation run pending.** `height_scan_classifier.py`'s `collect()` pinned `difficulty=None` (envs spread uniformly over all 10 difficulty rows) — deliberately, because bug #12's fix wasn't known to be in the checkout at the time. That is not neutral for a *discriminability* measurement. In `ROUGH_TERRAINS_CFG`, `pyramid_stairs`/`pyramid_stairs_inv` have `step_height_range=(0.0, 0.1)` and `wave_terrain` has `amplitude_range=(0.0, 0.2)` — **both start at zero relief**, so the bottom ~3 of 10 rows generate patches that are flat ground carrying a `stairs`/`rough` label. `random_rough` (`noise_range=(0.02, 0.10)`) never degenerates, `flat` is flat by definition, and `stepping_stones`' `stone_height`/`floor_depth` are difficulty-independent. **The only two classes that can degenerate to flat are the two that scored ~0.51.** Confirmed quantitatively against the committed confusion matrices (`eval_results/gate2a/height_scan_discriminability.json`): in the **clean** condition — i.e. a *perfect* sensor — the CNN predicts `flat` for **158/610 = 25.9%** of `stairs` samples and **83/530 = 15.7%** of `rough` samples, against ~30% and ~15% (half of `rough` is `wave_terrain`) predicted from the difficulty ranges alone; `flat` itself is 810/810 and `gaps` 2/610. Those samples aren't hard to classify, they have **no recoverable label**, and their entire cost lands on exactly the two classes whose ceiling was then read as a sensor property. Two secondary contributors, both also protocol rather than sensor: (a) `TERRAIN_CLASSES["rough"]` is `random_rough` ∪ `wave_terrain`, and a 4-wave/8 m wave is a 2 m wavelength seen through a 1.6 m scan footprint — a smooth ramp, geometrically nearer a staircase's mean slope than isotropic noise, so a class-level confusion matrix can't say whether the pair collides wholesale or only through the wave half; (b) `fit_cnn` applies `AdaptiveAvgPool2d(2)` to the 17×11 ray grid before its only linear layer, averaging away the periodic step edges that the same write-up correctly identifies as a staircase's actual signature. **(b) is the same inference the document already caught once** — the linear probe's "stairs is undetectable" was a model limitation that the CNN disproved; reading the CNN's number as a sensor ceiling makes that identical inference a second time. **Consequence**: `objective.md`'s gate 2 text and `PROGRESS_REPORT.md` §3.2 both weigh the three sensing levers against a ceiling that has not actually been measured — "provably insufficient" / "measured, not extrapolated" overstate what the run supports. Not yet re-run (laptop on battery, 2026-09-16); `height_scan_classifier.py` has been extended with `--difficulty` and sub-terrain-level `--classes` for exactly this, and the two commands to run are in `coordination/results/rough-stairs-switching-decision.md` §6, which also argues the design recommendation shouldn't wait on the outcome. Keep the `difficulty=None` run as the control when re-running — it's the only thing the committed numbers are comparable to.

17. **A specialist's height-scan ray debug-visualization renders as a visibly broken, contorted robot pose once it has walked several metres from its spawn point — a rendering artifact, not a policy or physics failure, confirmed by disabling it.** Recording qualitative videos of the three specialists (`report_content/`, 2026-09-16) with a fixed 1.0 m/s forward command over a 6 s / 300-step clip: the flat-terrain clips (the only ones where a specialist walks far without falling) show the robot's legs going into an increasingly unnatural pose from roughly step 200 onward, then the camera framing jumps abruptly by the final frames — visually indistinguishable from a physics explosion or the robot being launched into the air. This directly contradicted the gate-1 result (0% fall on flat for all three specialists), so it needed resolving rather than accepting at face value. Isolated in two steps: (1) an instrumented headless rollout (`render_mode=None`, no `VideoRecorder`) with the identical seed/checkpoint/command printed robot position, orientation, joint angles, velocity and contact force every step — all identical to a second run with rendering enabled, both showing a clean, stable, periodic gait (~60-step gait cycle) with `done=False` throughout; the physics is provably fine regardless of whether the artifact is visible. (2) Re-rendering the same clip with `terrain_scan`'s `debug_vis` (the `RayCastSensorCfg` field drawing the yellow/cyan sensor-ray markers seen in the videos) set to `False` eliminated the artifact completely — the final frame then looks identical to an early clean walking frame. **Fix**: `scripts/record_specialist_videos.py` disables `debug_vis` on `terrain_scan` by default (`--show-scan-rays` to restore it, safe only for short clips where the robot stays near its spawn point). All 9 committed videos were re-rendered with the fix. Root mechanism inside the debug visualizer (`mjlab/viewer/native/visualizer.py`'s `MujocoNativeDebugVisualizer`) not further investigated — likely a stale reference to the entity's spawn-relative transform rather than its current world pose, given the artifact scales with distance travelled — but isolating it to "the ray-overlay code path, not the sim" was sufficient to unblock the video deliverable and rule out a policy problem. Same family as bug #3 (video-only artifacts that look like capability problems): trust the numeric eval over a video's visual impression, and specifically here, treat any rendered debug overlay as suspect once the tracked body has moved a nontrivial distance from where the overlay was likely computed.

18. **mujoco_warp camera depth is distance along each pixel's ray, not optical-axis depth**, so the RealSense-style deprojection in SARO's Fig. 8 (`X = (i − x0)·d/fx`, with d as z-depth) is wrong for it. Confirmed on flat ground: deprojecting as ray distance puts every ground pixel at z = 0 ± 1e-7 m, while treating it as optical-axis depth spreads the floor over 6 cm. **Fix**: `CameraSpec.deproject_body` scales unit rays (`src/vlm_nav/camera.py`); a round-trip unit test pins it.

19. **mujoco_warp rendering has two properties that silently distort what a camera-based classifier sees.** (a) Shading has no light-intensity term: each light adds `base_colour · cos(incidence)`. mjlab's default overhead sun plus any extra light saturates horizontal surfaces to pure white, erasing rough-ground relief. (b) Only planes and meshes are texture-mapped; boxes and heightfields sample a single texel, so a checker material on stair boxes renders flat. Consequence: untextured down-stairs seen from the top render as uniform floor. The strip a rear light would shadow below each step edge is exactly the strip the step hides from the camera. **Fix**: courses bring two low oblique suns on a darker base colour, plus class-agnostic grout-line geoms on a 0.5 m grid in a camera-only, non-colliding geom group (verified in the compiled model: 92 geoms, group 4, `contype=conaffinity=0`). Separately, the terrain generator's default height colouring paints stairs and heightfields differently, a class shortcut for any VLM. Courses use uniform colours, and a `class_colors` visual exists only as a leak control.

20. **Two harness bugs that made a correct specialist look incompetent on stairs** (both caught in code-check traces before calibration). (a) A goal controller with `vy = k · distance · sin(heading error)` saturates far from the goal and walks the robot off the goal line and sideways across the steps. (b) An oracle choosing the specialist from the terrain one point ahead of the base switched back to the flat policy with the hind legs still on the last step, and the robot stalled there. **Fix**: velocity vector pointed at the goal (`controllers.goto_command`); footprint-overlap rule `CourseSpec.required_terrain` (−0.35 m … +0.30 m around the base). The VLM executor uses the same release geometry.

21. **`nconmax` overflow at environment setup on a course**: the robot's pre-reset keyframe sits at the world origin, which the generator puts mid-course, inside the L3 staircase (44 initial contacts > the task's `nconmax=35`). Crash only, not a silent error, and no runtime overflow was logged in any completed run. **Fix**: twin env `nconmax=96`. Also: with thinking enabled, Gemma-4 (llama.cpp) spends the whole `max_tokens` budget in `reasoning_content` and returns empty `content`. The client sends `chat_template_kwargs.enable_thinking=false`.

### Checked, not bugs (2026-09-12)

Recorded so they aren't re-investigated:

- **Evals did use the final command range.** `commands_vel` keys off `env.common_step_counter`, which is 0 in a fresh process, so it looked like 1200-step evals would stay on the stage-0 range. They didn't: `MjlabOnPolicyRunner.load()` restores `common_step_counter` from the checkpoint's `infos.env_state` unconditionally, even with `load_cfg={"actor": True}`. The specialists' checkpoints hold 240000 (= 10k iterations × 24). `eval_checkpoint.py` now pins the range explicitly anyway, rather than relying on that restore.
- **Env 0's terrain isn't fixed in videos.** In the training config, env *i*'s terrain column is `floor(i / (num_envs / num_cols))`, so env 0 is always column 0. But the play config adds a `randomize_terrain` reset event, so rendered videos don't inherit that. Bug #3 stands as recorded (command sampling), without a terrain component.

## Design decisions on record

- **Terrain taxonomy is geometry-based** (flat / rough / stairs / gaps), not material-based (grass / gravel / tile) as the original project idea sketched — `mjlab`'s terrain generator only does geometry (pyramid stairs, stepping stones, random grids, Perlin-noise/wave heightfields), no friction/material terrain classes. Friction variation is folded into domain randomization (`foot_friction` event term) instead of being a discrete class.
- **The followed "person" for the anticipatory-switching sim experiments will be a scripted kinematic leader body**, not a simulated perception pipeline (LiDAR + detector). Verified feasible: `mjlab` has first-class mocap support (`auto_wrap_fixed_base_mocap`, `Entity.write_mocap_pose_to_sim(pose, env_ids)` for a per-env-positionable non-physical body) — not yet built. Real perception is deferred to the hardware phase.
- **All specialists must share one observation space** so the switching module's gating network can blend them — drove the flat-specialist design (generator, not plane) and will constrain any future per-terrain reward shaping (reward *terms* can differ later; observation *shape* cannot).
- **Arm 1 is a sensing-matched stock-PPO generalist** (`Unitree-Go2-Generalist`), not PAS (2026-09-12). Union of the four classes at equal class weight, split evenly within a class; identical obs, rewards, runner and 10k-iteration budget to a specialist. Stepping stones make up 25% of its terrain — the Gaps plateau was at 100%, and PAS trained fine at 15%.
- **Switching arms form a 2×2** of {hard, soft} × {reactive, anticipatory}; **smoothness is measured in geometric windows** around terrain-class boundaries, never keyed to policy switch events; **anticipatory gain is reported against preview horizon** (leader lead distance), not at one operating point. Details in `objective.md` (2026-09-12).
- **Specialists stay specialists**: a class that can't be learned from a cold start gets a warm start, not a diluted terrain mix (2026-09-12).
- **Seeds**: specialists single-seed (frozen assets); gating network ≥3 seeds; every arm ≥3 eval seeds. `SEED=` in `train_specialist_slurm.sh` writes non-default seeds to `<experiment>_s<SEED>` so resume never mixes seeds.
