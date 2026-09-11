# policyswitching

Terrain-adaptive quadruped locomotion for the Unitree Go2, built on a vendored copy of [`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab) (mjlab + RSL-RL PPO).

**Long-term goal**: instead of one generalist locomotion policy, train a small set of terrain-specialist RL policies and learn to switch between them while the robot follows a person — using the followed person's trajectory as an anticipatory preview of upcoming terrain, rather than reacting to terrain changes after the fact.

**Current status (2026-09-01)**: that specialist/switching architecture is not built yet. The active work is replicating **PAS** (Probability Annealing Selection) from the SARO paper (arXiv:2407.16412, PDF at [`2407.16412v3.pdf`](2407.16412v3.pdf)) — a two-stage single-policy distillation technique, *not* a runtime switch between policies — as groundwork and a candidate generalist baseline. See [`docs/07-pas-implementation.md`](docs/07-pas-implementation.md) for how it's implemented, and the roadmap plan for how it connects to the specialist-switching goal above.

## Orientation

- [`objective.md`](objective.md) — what this project is actually trying to show: the research idea, its novelty, the three-way comparison (generalist vs. reactive hard-switch vs. anticipatory soft-switch) and metrics that define a result. Read this to understand *why*, before `findings.md`'s *what's happened so far*.
- [`findings.md`](findings.md) — running log of experiments, results, and infra bugs found/fixed. Read this before starting new work or trusting an eval number — several past results looked fine until a specific gotcha was found (see its "Bugs found" section).
- [`docs/`](docs/) — hand-written reference docs for mjlab/`unitree_rl_mjlab` and this project's additions. Start at [`docs/README.md`](docs/README.md).
- [`docs/07-pas-implementation.md`](docs/07-pas-implementation.md) — the PAS implementation: two-stage training (`Unitree-Go2-PAS-Oracle` → `Unitree-Go2-PAS-Anneal`), architecture, known gaps.
- [`a100/`](a100/), [`kaggle/`](kaggle/) — training/eval infra for two compute backends (SLURM A100 cluster, Kaggle free-tier notebooks). Both sync checkpoints through a Hugging Face model repo (`a100/hf_sync.py`) so training survives job resubmission / ephemeral sessions.
- `unitree_rl_mjlab/` — the vendored simulator/training stack, flattened into this repo (not a submodule) with local modifications layered on top — see `docs/07` for what's project-specific vs. upstream.
- [`coordination/`](coordination/) — the mailbox for running this repo from two machines at once (SLURM cluster for training, laptop for local eval): heartbeats, inbox notes, eval results. See [`docs/CLAUDE.cluster.md`](docs/CLAUDE.cluster.md) / [`docs/CLAUDE.laptop.md`](docs/CLAUDE.laptop.md) for the role split.

**Layout**: this is the canonical checkout and all job outputs land here — SLURM logs at the top level, training runs and checkpoints under `unitree_rl_mjlab/logs/`, eval artifacts under `unitree_rl_mjlab/eval_ckpts/`. (A second clone at `/dist_home/d_palmani/policyswitching` was removed on 2026-09-05 and its contents merged in.)

## Roadmap

A phased plan from the current PAS-only state to the full specialist-policy-switching architecture (terrain taxonomy, specialist training, person-following-as-preview switching module, evaluation harness, hardware) lives at `~/.claude/plans/the-idea-in-one-iridescent-wilkes.md`.
