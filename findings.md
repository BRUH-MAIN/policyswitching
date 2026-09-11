# Findings

Running log of experiments, results, and infrastructure gotchas for this project. Written so a future session (human or Claude) can pick up work without re-deriving anything here from raw SLURM logs. Update this file whenever a training run finishes, a bug is found/fixed, or a result changes a prior conclusion — append rather than rewrite, and correct entries in place with a dated note rather than deleting them if a later result contradicts an earlier one.

For the *forward-looking* plan (what to build next), see the roadmap at `~/.claude/plans/the-idea-in-one-iridescent-wilkes.md` (on the machine this was written on — recover it via the Artifact/plan tooling `list` action if unavailable, or reconstruct from this file). This file is the *backward-looking* record: what was tried, what happened, why.

## Project state, one paragraph

Long-term goal: terrain-specialist policies + an anticipatory (person-trajectory-preview) switching module, vs. a generalist baseline, vs. a reactive hard-switch baseline. As of 2026-09-11: the generalist baseline (PAS, a SARO-paper replication) is fully trained; three of four terrain specialists (Flat/Stairs/Rough) are fully trained and healthy; the fourth (Gaps) failed twice under its original task design and is retraining under a fixed one. No switching module, leader-in-sim, or person-following code exists yet — that's all still ahead (see roadmap Phases 3-5).

## Infra facts

- **Canonical repo**: `/dist_home/d_palmani/c-08/policyswitching` (single clone as of 2026-09-05; a second, divergent clone at `/dist_home/d_palmani/policyswitching` was merged in and deleted — see "Bugs found" below).
- **venv**: `/dist_home/d_palmani/.venvs/policyswitching-pas` (isolated from the shared conda base env deliberately — see `docs/01-installation.md`).
- **Cluster**: SLURM, partition `workq`, nodes `asaicomputemaster` (2× RTX 6000 Ada) + `asaicomputenode02`/`03` (2× A100 each). Scheduling is **pure age-based FIFO** — `sprio`'s FAIRSHARE component is 0 for every user, so priority is just accumulated wait time. Practical consequence: **submit early and request short walltimes** (a 3-day request only starts when a 3-day window opens and can sit pending far longer than the extra hours are worth; a 1-day request with cheap resubmission schedules much faster). The cluster is shared with other users/projects (e.g. a `codeswitching` project noted in `hf_sync.py`'s comments) — expect contention, typically ~5-10 other jobs queued.
- **HF model repo**: `RohanRamesh/go2-pas-saro` — holds PAS's `stage1/` and `stage2/` checkpoints only. Specialist checkpoints are **not** on HF (see bug #4 below) — they live purely on local disk under `unitree_rl_mjlab/logs/rsl_rl/go2_spec_<name>/`.
- **Reference paper**: SARO, arXiv:2407.16412 (PDF at repo root). PAS = "Probability Annealing Selection," the paper's low-level locomotion technique this project replicates (not the paper's VLM high-level planner).
- **Uncommitted, load-bearing change**: `clip_actions=6.0` in `unitree_rl_mjlab/src/tasks/velocity/config/go2/rl_cfg.py` is still unstaged in git as of this writing. It is the fix that stopped PAS's training divergence (see below) — do not lose it, and consider committing it.

## Experiment log

| date | job(s) | what | outcome |
|---|---|---|---|
| 2026-08-28 | 11765-11767 | PAS Stage 1, early attempts | Failed on missing `HF_TOKEN`, then missing `scipy`, then CUDA OOM at iter 9477 — see bugs #1-3 |
| 2026-08-29/30 | 11769 | PAS Stage 1 resumed | Reward diverged to ≈ −10M by iter 36199 (root cause: unclipped actions) — bug #1 |
| 2026-09-01/02 | 11802 | PAS Stage 1, post-`clip_actions` fix | Stable; timed out at iter 77508/79999 (3-day walltime) |
| 2026-09-03 | 11813 | Video eval, `stage1_model_31800.pt` | Video showed zero motion — later found non-representative, see bug #8 |
| 2026-09-04/05 | 11844→11846 | PAS Stage 1 finish + Stage 2 (anneal), full run | **Complete.** Stage 1 finished iter 39999; Stage 2 auto-continued and finished iter 79998/79999 |
| 2026-09-05 | 11847, 11848 | Sanity re-eval of known-bad ckpt + video, post repo-consolidation | Confirms repo move didn't break anything |
| 2026-09-05/06/07 | 11849, 11851, 11852 | Specialist training: Stairs, Rough, Flat | **All three complete**, clean (see Specialists table) |
| 2026-09-06/07 | 11850 | Specialist training: Gaps (100% stepping_stones) | TIMEOUT at iter 7528/10000, plateaued — see Specialists table |
| 2026-09-08/09 | 11873 | Gaps resubmit | Restarted from iter 0 (bug #4), plateaued again by iter 6632, TIMEOUT |
| 2026-09-08 | 11874, 11875 | PAS final eval, `model_79998`, oracle + estimator-only | See PAS results table |
| 2026-09-11 | 11914 | Gaps retrain, blended terrain (fix for the plateau) | Queued/running — update this row when it finishes |

## PAS (generalist baseline) results

Two-stage training: Stage 1 ("oracle," `Unitree-Go2-PAS-Oracle`) trains an actor that reads a true privileged terrain latent; Stage 2 ("anneal," `Unitree-Go2-PAS-Anneal`, resumed from Stage 1) anneals it toward an estimator that predicts that latent from proprioception alone (`anneal_base=0.9998` per iteration). See `docs/07-pas-implementation.md` for the architecture.

**Checkpoints**: Stage 1 final = `logs/rsl_rl/go2_pas/slurm_stage1/model_39999.pt`. Stage 2 final = `logs/rsl_rl/go2_pas/2026-09-05_17-59-43_stage2/model_79998.pt`. Both also on HF at `stage1/model_39999.pt` / `stage2/model_79998.pt`.

**Critical eval gotcha**: `anneal_prob` (how much the actor trusts the true latent vs. the estimator's prediction) is **runtime state on `PasActorModel`, not stored in the checkpoint** — it's reset to `initial_anneal_prob=1.0` on load. Every eval silently runs in oracle mode unless `eval_checkpoint.py --anneal-prob 0.0` is passed explicitly. By the end of Stage 2 training, `anneal_prob ≈ 0.9998^40000 ≈ 0.0003` — i.e. the policy was almost always trained/tested using the *estimator's* prediction, not the oracle latent — so oracle-mode eval is actually evaluating an operating regime the final policy barely used during training.

| checkpoint | mode | survival | mean ep. length | achieved/commanded speed | stalled-while-commanded |
|---|---|---|---|---|---|
| `model_77400` (intermediate) | oracle (`anneal_prob=1.0`) | 73.7% | 787.7/1200 | 15% | 31.6% |
| `model_77400` (intermediate) | estimator-only (`anneal_prob=0.0`) | 32.9% | 356.6/1200 | 15% | 29.9% |
| `model_79998` (**final**) | oracle | 75.5% | 784.4/1200 | 9% | 30.4% |
| `model_79998` (**final**) | estimator-only | **42.0%** | 450.8/1200 | 9% | 28.9% |
| `stage1_model_31800` (pre-`clip_actions` fix, for reference) | oracle | 77.5% | 821.5/1200 | 18% | 25.2% |

**Reading these**: estimator-only (the actually-deployable mode — real hardware has no privileged terrain sensor) improved 32.9%→42.0% survival over the last ~2600 training iterations, but achieved-speed-as-%-of-commanded dropped for *both* modes between the two checkpoints (15%→9%) — survival is improving without locomotion clearly improving, possibly a shift toward more conservative/cautious behavior. One data point each way, not conclusive.

**Open call, not yet resolved**: PAS oracle-mode is a legitimate *sim-only* baseline (with a privileged-information advantage no specialist has), but estimator-only — the only mode a real robot could run — falls roughly twice as often as oracle and hasn't clearly gotten better at walking as it's gotten better at surviving. Whether that's worth further Stage-2 investment before anchoring the generalist-baseline comparison on it, or whether it's accepted as-is (sim ablations were always meant to carry the empirical weight, per the original project framing), is an open decision.

## Terrain specialists

Stock single-stage PPO (`VelocityOnPolicyRunner` + `unitree_go2_ppo_runner_cfg()`, no terrain encoder / privileged state / annealing) — a specialist only ever sees one terrain class, so it has nothing to disambiguate. All four share one observation space (234-dim actor obs including `height_scan`) so a later gating network can blend them; see `_unitree_go2_specialist_env_cfg` in `env_cfgs.py` for why the Flat specialist uses a flat-only *terrain generator* rather than `unitree_go2_flat_env_cfg()`'s plane (which deletes `height_scan` from the obs and would break blendability).

| specialist | terrain mix (training) | budget reached | error_vel_xy (final) | fell_over/ep | illegal_contact/ep | ep. length | checkpoint |
|---|---|---|---|---|---|---|---|
| Flat | 100% flat | 9999/10000 ✅ | 1.45 | 0 | 0 | 1000/1000 | `logs/rsl_rl/go2_spec_flat/2026-09-06_17-17-15/model_9999.pt` |
| Stairs | pyramid_stairs + inv, equal weight | 9999/10000 ✅ | 1.81 | 0 | 0.21 | 980/1000 | `logs/rsl_rl/go2_spec_stairs/2026-09-05_22-37-43/model_9999.pt` |
| Rough | random_rough + wave, equal weight | 9999/10000 ✅ | 1.88 | 0 | 0.50 | 983/1000 | `logs/rsl_rl/go2_spec_rough/2026-09-06_12-07-49/model_9999.pt` |
| Gaps (attempt 1) | 100% stepping_stones | 7528/10000 (TIMEOUT) | 0.03 (artifact — see below) | 807.7 | 47.9 | **9.8/1000** | archived, `logs/rsl_rl/_archived_go2_spec_gaps_100pct_plateaued/2026-09-06_09-11-12/` |
| Gaps (attempt 2) | 100% stepping_stones | 6632/10000 (TIMEOUT, restarted from 0 — bug #4) | similar | similar | similar | ~13.6/1000 | archived, `.../_archived_go2_spec_gaps_100pct_plateaued/2026-09-08_00-10-45/` |
| Gaps (attempt 3) | **20% stepping_stones / 40% flat / 40% random_rough** (fix, 2026-09-11) | job 11914, in progress | — | — | — | — | — |

**Flat/Stairs/Rough beat PAS's own tracking error** (2.09 oracle-mode `error_vel_xy` vs. 1.45-1.88) and have far better survival — the expected, positive specialist-vs-generalist result.

**Gaps plateaued reproducibly, twice, under the original 100%-stepping_stones design.** Reward flat at ≈ −6 to −8 within a few hundred iterations both times, no improvement over 6600-7500 iterations each attempt, episodes ending in ~10-14 steps (immediate falls). The low `error_vel_xy` in that state is **not** good tracking — it's an artifact of episodes ending before velocity error has time to accumulate. Diagnosed as a task-design problem: isolating gap terrain at 100% gives a cold-start policy no easy terrain to learn basic locomotion on before also having to solve gap-crossing. Fixed 2026-09-11 by blending in flat/rough terrain, mirroring PAS's own gap terrain (which mixes `stepping_stones` at only 15% into 85% easier ground). **When job 11914 finishes, update this table and re-run the same plateau check (sample `Mean reward`/`Mean episode length` every few hundred iterations) before trusting the result — the fix is a reasonable bet, not a proven one yet.**

## Bugs found and fixed

Ordered roughly by how much they'd silently corrupt a result if unnoticed — read top-to-bottom before trusting any new eval number.

1. **Survival metrics can't distinguish walking from bracing.** `stage1_model_31800.pt` scored 80.5% survival / +39.5 return in the original `eval_checkpoint.py` (pre-fix) while translating a measured ~0 m over an 8-second rollout — bracing in place actively maximizes the energy/joint_vel/action_rate penalty terms, so a degenerate "don't move" policy looks healthy on survival alone. **Fix**: `eval_checkpoint.py` now also reports commanded-vs-achieved speed, velocity tracking error, distance travelled, and a "stalled while commanded to move" percentage with an explicit warning past 50%.

2. **`anneal_prob` isn't stored in the checkpoint** (see PAS section above) — every eval silently ran in oracle mode. **Fix**: `eval_checkpoint.py --anneal-prob <val>` override, with the active mode printed explicitly and a warning when evaluating a Stage-2 checkpoint in oracle mode without an explicit override.

3. **Single-environment qualitative videos are not representative.** `play.py` renders only env 0, and across three independent runs (11813, 11848, and a manual repro) env 0 happened to draw `UniformVelocityCommandCfg`'s "stand still" branch (`rel_standing_envs=0.05`) every time, showing zero net motion regardless of the checkpoint's actual walking ability. Confirmed the resample logic itself is correct (an explicitly reseeded RNG produces properly varied nonzero commands) — this is a sampling-representativeness gap, not a training or measurement bug. **Trust the 1024-env numeric eval over a single video for judging locomotion quality.** Not yet fixed at the tooling level — future work: force a nonzero command for video eval, or render/sample multiple envs.

4. **Specialist checkpoints were never uploaded to HF, and the resume logic didn't know that.** `PasOnPolicyRunner.save()` has HF-upload logic; the stock `VelocityOnPolicyRunner` used by specialists does not. `hf_sync_specialist.py` (the original design) only checked HF, found nothing, and concluded "start fresh" — causing Gaps' resubmission (job 11873) to silently restart from iteration 0 instead of resuming from 7528, wasting a full GPU-day re-discovering the same plateau. Only Flat/Stairs/Rough escaped this because each finished within one job's walltime, so no resume was ever needed. **Fix**: `a100/local_ckpt_resume.py` scans local disk directly (`logs/rsl_rl/<experiment>/*/model_*.pt` across all run dirs, confirmed to persist across job boundaries on this cluster) instead of querying HF. `train_specialist_slurm.sh` updated to use it; specialist jobs no longer need `HF_TOKEN`.

5. **tyro CLI boolean flags require an explicit value here, counter to the usual `--flag`/`--no-flag` convention** — `play.py`'s `--video` and `train.py`'s `--agent.resume` both render as `--video {True,False}` in their own `--help` output and fail with "Missing value for argument" if passed bare. This was flip-flopped once mid-project: job 11800 failed on a bare `--video` → corrected to `--video True` → job 11813 succeeded → later mistakenly "fixed" back to bare `--video` based on a misreading of the already-corrected script → caught via a direct `--help` check and a reproduction test before it could break another job. **If touching either flag again, verify empirically (`scripts/play.py <task> --help`) rather than trusting intuition about tyro's convention.**

6. **Missing `scipy` dependency.** `mjlab==1.2.0` imports `scipy.interpolate` in `heightfield_terrains.py` but doesn't declare it — crashes ~17 minutes into a run, right after the HF checkpoint download, with `ModuleNotFoundError`. Fixed by an explicit `pip install scipy` in `train_pas_slurm.sh`.

7. **CUDA OOM from allocator fragmentation, not true memory exhaustion.** Warp's CUDA-graph replay draws scratch memory from CUDA's stream-ordered mempool at replay time; PyTorch's caching allocator doesn't cooperate with that pool and fragments it over thousands of PPO iterations, eventually failing a graph launch even with `nvidia-smi` showing headroom (killed job 11767 at iteration 9477). Fixed via `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

8. **Training reward divergence without action clipping.** Unclipped actions let PPO's policy drift to reward ≈ −10M / value loss ≈ 5×10¹² over one run (job 11769). Fixed via `clip_actions=6.0` in both Go2 PPO runner configs — **still uncommitted in git**, see "Infra facts" above.

9. **Two divergent git clones** existed (`/dist_home/d_palmani/c-08/policyswitching` and `/dist_home/d_palmani/policyswitching`), with SLURM scripts hardcoding paths into the latter. Consolidated into the `c-08` checkout on 2026-09-05 (`logs/`, `eval_ckpts/`, `stage1/` merged in; the second clone deleted after verifying no unique content remained). All `a100/*.sh` scripts now point at `c-08`.

## Design decisions on record

- **Terrain taxonomy is geometry-based** (flat / rough / stairs / gaps), not material-based (grass / gravel / tile) as the original project idea sketched — `mjlab`'s terrain generator only does geometry (pyramid stairs, stepping stones, random grids, Perlin-noise/wave heightfields), no friction/material terrain classes. Friction variation is folded into domain randomization (`foot_friction` event term) instead of being a discrete class.
- **The followed "person" for the anticipatory-switching sim experiments will be a scripted kinematic leader body**, not a simulated perception pipeline (LiDAR + detector). Verified feasible: `mjlab` has first-class mocap support (`auto_wrap_fixed_base_mocap`, `Entity.write_mocap_pose_to_sim(pose, env_ids)` for a per-env-positionable non-physical body) — not yet built. Real perception is deferred to the hardware phase.
- **All specialists must share one observation space** so the switching module's gating network can blend them — drove the flat-specialist design (generator, not plane) and will constrain any future per-terrain reward shaping (reward *terms* can differ later; observation *shape* cannot).
